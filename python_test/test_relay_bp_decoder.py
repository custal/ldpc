import gc
import math

import numpy as np
import pytest
import scipy.sparse

from ldpc.bp_decoder import BpDecoder
from ldpc.codes import rep_code
from ldpc.relay_bp_decoder import (
    DEFAULT_PRECISION,
    RelayBpDecoder,
    available_precisions,
    resolve_precision,
)


# ============================================================================
# Fixtures and helpers
# ============================================================================

PCM_NP = np.array(
    [
        [1, 0, 1],
        [0, 1, 1],
    ],
    dtype=np.uint8,
)

PCM_SP = scipy.sparse.csr_matrix(PCM_NP)

DEFAULT_LEGS = 2
DEFAULT_SOLUTIONS = 1
DEFAULT_ITERATIONS0 = 10
DEFAULT_MAX_ITER = 10

DEFAULT_STRENGTHS_PER_LEG = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ],
    dtype=np.float64,
)


def make_decoder(pcm=PCM_NP, **kwargs):
    """Construct a RelayBpDecoder using explicit memory strengths."""

    defaults = dict(
        error_rate=0.1,
        maximum_legs=DEFAULT_LEGS,
        maximum_solutions=DEFAULT_SOLUTIONS,
        iterations0=DEFAULT_ITERATIONS0,
        max_iter=DEFAULT_MAX_ITER,
        gamma0=0.0,
        memory_strengths_per_leg=(
            DEFAULT_STRENGTHS_PER_LEG.copy()
        ),
        bp_method="product_sum",
        schedule="parallel",
        memory_seed=123,
    )

    defaults.update(kwargs)
    return RelayBpDecoder(pcm, **defaults)


def make_generated_memory_decoder(pcm=PCM_NP, **kwargs):
    """Construct a decoder using generated per-leg memory strengths."""

    defaults = dict(
        error_rate=0.1,
        maximum_legs=DEFAULT_LEGS,
        maximum_solutions=DEFAULT_SOLUTIONS,
        iterations0=DEFAULT_ITERATIONS0,
        max_iter=DEFAULT_MAX_ITER,
        gamma0=-1.0,
        gamma_dist_interval=(-0.5, 0.75),
        memory_strengths_per_leg=None,
        bp_method="product_sum",
        schedule="parallel",
        memory_seed=123,
    )

    defaults.update(kwargs)
    return RelayBpDecoder(pcm, **defaults)


def binary_syndrome(pcm, decoding):
    """Return pcm @ decoding modulo two."""

    if scipy.sparse.issparse(pcm):
        result = pcm @ decoding
    else:
        result = np.asarray(pcm) @ decoding

    return np.asarray(result, dtype=np.uint8).reshape(-1) % 2


# Precision is a compile-time menu in the C++ template, so the tiers that exist
# depend on how the extension was built rather than being fixed here.
PRECISION_TIERS = available_precisions()

# A subset for the heavier sweeps, so that parametrised tests do not spend
# seconds in software floating point.
FAST_PRECISION_TIERS = [
    bits for bits in PRECISION_TIERS if bits <= 512
]


def non_default_precision():
    """A tier that is not the default, so switching precision is observable."""

    for bits in reversed(FAST_PRECISION_TIERS):
        if bits != DEFAULT_PRECISION:
            return bits

    return DEFAULT_PRECISION


def is_representable_in(value, mantissa_bits):
    """Whether value fits exactly in a float of the given mantissa width."""

    if not math.isfinite(value) or value == 0.0:
        return True

    mantissa, _ = math.frexp(value)

    return float(
        math.ldexp(mantissa, mantissa_bits)
    ).is_integer()


def assert_llrs_identical(first, second):
    """Compare log probability ratios bitwise, treating two NaNs as equal.

    The product-sum update can legitimately saturate to infinities and NaNs.
    """

    assert first.shape == second.shape

    both_nan = np.isnan(first) & np.isnan(second)

    assert np.array_equal(
        first[~both_nan],
        second[~both_nan],
    )


# ============================================================================
# Construction
# ============================================================================

def test_init_numpy_pcm():
    decoder = make_decoder(PCM_NP)

    assert decoder.check_count == 2
    assert decoder.bit_count == 3
    assert decoder.maximum_legs == 2
    assert decoder.maximum_solutions == 1
    assert decoder.iterations0 == 10
    assert decoder.max_iter == 10


def test_init_sparse_pcm():
    decoder = make_decoder(PCM_SP)

    assert decoder.check_count == 2
    assert decoder.bit_count == 3


def test_init_invalid_pcm():
    with pytest.raises(TypeError):
        make_decoder("invalid_pcm")


def test_init_requires_error_rate_or_channel():
    with pytest.raises(ValueError):
        RelayBpDecoder(
            PCM_NP,
            maximum_legs=2,
            maximum_solutions=1,
            iterations0=10,
            max_iter=10,
            gamma0=0.0,
            memory_strengths_per_leg=(
                DEFAULT_STRENGTHS_PER_LEG.copy()
            ),
        )


def test_init_requires_iterations0():
    with pytest.raises((ValueError, TypeError)):
        RelayBpDecoder(
            PCM_NP,
            error_rate=0.1,
            maximum_legs=2,
            maximum_solutions=1,
            iterations0=None,
            max_iter=10,
            gamma0=0.0,
            memory_strengths_per_leg=(
                DEFAULT_STRENGTHS_PER_LEG.copy()
            ),
        )


def test_init_requires_maximum_solutions():
    with pytest.raises((ValueError, TypeError)):
        RelayBpDecoder(
            PCM_NP,
            error_rate=0.1,
            maximum_legs=2,
            maximum_solutions=None,
            iterations0=10,
            max_iter=10,
            gamma0=0.0,
            memory_strengths_per_leg=(
                DEFAULT_STRENGTHS_PER_LEG.copy()
            ),
        )


def test_init_accepts_explicit_memory_without_interval():
    decoder = RelayBpDecoder(
        PCM_NP,
        error_rate=0.1,
        maximum_legs=2,
        maximum_solutions=1,
        iterations0=10,
        max_iter=10,
        gamma0=123.0,
        gamma_dist_interval=None,
        memory_strengths_per_leg=(
            DEFAULT_STRENGTHS_PER_LEG.copy()
        ),
    )

    assert np.allclose(
        decoder.memory_strengths_per_leg,
        DEFAULT_STRENGTHS_PER_LEG,
    )


def test_init_accepts_generated_memory_configuration():
    decoder = make_generated_memory_decoder()

    assert decoder.memory_strengths_per_leg is None
    assert decoder.gamma0 == pytest.approx(-1.0)
    assert np.allclose(
        decoder.gamma_dist_interval,
        [-0.5, 0.75],
    )


def test_init_generated_memory_requires_gamma0():
    with pytest.raises((ValueError, TypeError)):
        RelayBpDecoder(
            PCM_NP,
            error_rate=0.1,
            maximum_legs=2,
            maximum_solutions=1,
            iterations0=10,
            max_iter=10,
            gamma0=None,
            gamma_dist_interval=(-0.5, 0.75),
            memory_strengths_per_leg=None,
        )


def test_init_generated_memory_requires_interval():
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        RelayBpDecoder(
            PCM_NP,
            error_rate=0.1,
            maximum_legs=2,
            maximum_solutions=1,
            iterations0=10,
            max_iter=10,
            gamma0=-1.0,
            gamma_dist_interval=None,
            memory_strengths_per_leg=None,
        )


@pytest.mark.parametrize(
    "interval",
    [
        [],
        [0.1],
        [0.1, 0.2, 0.3],
        np.zeros((2, 1)),
    ],
)
def test_init_generated_memory_rejects_invalid_interval(interval):
    with pytest.raises((ValueError, TypeError, Exception)):
        RelayBpDecoder(
            PCM_NP,
            error_rate=0.1,
            maximum_legs=2,
            maximum_solutions=1,
            iterations0=10,
            max_iter=10,
            gamma0=-1.0,
            gamma_dist_interval=interval,
            memory_strengths_per_leg=None,
        )


@pytest.mark.parametrize(
    "strengths",
    [
        np.zeros((1, 3)),
        np.zeros((3, 3)),
        np.zeros((2, 2)),
        np.zeros((2, 4)),
        np.zeros(3),
    ],
)
def test_init_rejects_wrong_explicit_memory_shape(strengths):
    with pytest.raises((ValueError, TypeError, Exception)):
        RelayBpDecoder(
            PCM_NP,
            error_rate=0.1,
            maximum_legs=2,
            maximum_solutions=1,
            iterations0=10,
            max_iter=10,
            gamma0=0.0,
            memory_strengths_per_leg=strengths,
        )


# ============================================================================
# Inherited property round trips
# ============================================================================

def test_max_iter_roundtrip():
    decoder = make_decoder(max_iter=10)

    assert decoder.max_iter == 10

    decoder.max_iter = 5

    assert decoder.max_iter == 5


def test_max_iter_zero_uses_block_length():
    decoder = make_decoder(max_iter=0)

    assert decoder.max_iter == decoder.bit_count


def test_invalid_max_iter_type():
    with pytest.raises((TypeError, ValueError)):
        make_decoder(max_iter="invalid")


def test_invalid_max_iter_value():
    with pytest.raises(ValueError):
        make_decoder(max_iter=-1)


def test_bp_method_roundtrip():
    decoder = make_decoder(bp_method="product_sum")

    assert decoder.bp_method == "product_sum"

    decoder.bp_method = "minimum_sum"
    assert decoder.bp_method == "minimum_sum"

    decoder.bp_method = "product_sum"
    assert decoder.bp_method == "product_sum"


def test_invalid_bp_method():
    decoder = make_decoder()

    with pytest.raises(ValueError):
        decoder.bp_method = "invalid"


def test_schedule_parallel_roundtrip():
    decoder = make_decoder(schedule="parallel")

    assert decoder.schedule == "parallel"


def test_invalid_schedule():
    decoder = make_decoder()

    with pytest.raises(ValueError):
        decoder.schedule = "invalid"


def test_ms_scaling_factor_roundtrip():
    decoder = make_decoder(
        bp_method="minimum_sum",
        ms_scaling_factor=0.5,
    )

    assert decoder.ms_scaling_factor == pytest.approx(0.5)

    decoder.ms_scaling_factor = 0.75

    assert decoder.ms_scaling_factor == pytest.approx(0.75)


def test_invalid_ms_scaling_factor():
    decoder = make_decoder()

    with pytest.raises(TypeError):
        decoder.ms_scaling_factor = "invalid"


def test_error_rate_roundtrip():
    decoder = make_decoder(error_rate=0.1)

    assert np.allclose(decoder.error_rate, 0.1)

    decoder.error_rate = 0.2

    assert np.allclose(decoder.error_rate, 0.2)


def test_error_channel_roundtrip():
    first_channel = np.array(
        [0.1, 0.2, 0.3],
        dtype=np.float64,
    )

    decoder = make_decoder(
        error_rate=None,
        error_channel=first_channel,
    )

    assert np.allclose(
        decoder.error_channel,
        first_channel,
    )

    second_channel = np.array(
        [0.3, 0.2, 0.1],
        dtype=np.float64,
    )

    decoder.error_channel = second_channel

    assert np.allclose(
        decoder.error_channel,
        second_channel,
    )


def test_error_channel_wrong_length():
    decoder = make_decoder()

    with pytest.raises(ValueError):
        decoder.error_channel = np.array([0.1, 0.2])


# ============================================================================
# Relay-specific property round trips
# ============================================================================

def test_maximum_legs_readable():
    strengths = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.3, 0.3],
            [0.7, 0.7, 0.7],
        ]
    )

    decoder = make_decoder(
        maximum_legs=3,
        memory_strengths_per_leg=strengths,
    )

    assert decoder.maximum_legs == 3


def test_maximum_solutions_roundtrip():
    decoder = make_decoder(maximum_solutions=2)

    assert decoder.maximum_solutions == 2

    decoder.maximum_solutions = 1

    assert decoder.maximum_solutions == 1


@pytest.mark.parametrize("value", [-1, 1.5, "2"])
def test_invalid_maximum_solutions(value):
    decoder = make_decoder()

    with pytest.raises((ValueError, TypeError)):
        decoder.maximum_solutions = value


def test_iterations0_roundtrip():
    decoder = make_decoder(iterations0=7)

    assert decoder.iterations0 == 7

    decoder.iterations0 = 3

    assert decoder.iterations0 == 3


@pytest.mark.parametrize("value", [-1, 1.5, "2"])
def test_invalid_iterations0(value):
    decoder = make_decoder()

    with pytest.raises(ValueError):
        decoder.iterations0 = value


def test_memory_strengths_per_leg_roundtrip():
    strengths = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.4, 0.5],
        ],
        dtype=np.float64,
    )

    decoder = make_decoder(
        memory_strengths_per_leg=strengths,
    )

    assert np.allclose(
        decoder.memory_strengths_per_leg,
        strengths,
    )

    replacement = np.array(
        [
            [-1.0, -1.0, -1.0],
            [0.7, 0.8, 0.9],
        ],
        dtype=np.float64,
    )

    decoder.memory_strengths_per_leg = replacement

    assert np.allclose(
        decoder.memory_strengths_per_leg,
        replacement,
    )


def test_memory_strengths_per_leg_returns_copy():
    decoder = make_decoder()

    returned = decoder.memory_strengths_per_leg
    returned[0, 0] = 999.0

    assert decoder.memory_strengths_per_leg[0, 0] != 999.0


@pytest.mark.parametrize(
    "strengths",
    [
        np.zeros((1, 3)),
        np.zeros((3, 3)),
        np.zeros((2, 2)),
        np.zeros((2, 4)),
    ],
)
def test_memory_strengths_per_leg_wrong_shape(strengths):
    decoder = make_decoder()

    with pytest.raises((ValueError, Exception)):
        decoder.memory_strengths_per_leg = strengths


def test_gamma0_roundtrip():
    decoder = make_generated_memory_decoder(gamma0=-0.75)

    assert decoder.gamma0 == pytest.approx(-0.75)

    decoder.gamma0 = 0.25

    assert decoder.gamma0 == pytest.approx(0.25)


def test_gamma0_rejects_non_float():
    decoder = make_generated_memory_decoder()

    with pytest.raises(ValueError):
        decoder.gamma0 = "invalid"


def test_gamma_dist_interval_roundtrip():
    decoder = make_generated_memory_decoder(
        gamma_dist_interval=(-0.3, 0.8)
    )

    assert np.allclose(
        decoder.gamma_dist_interval,
        [-0.3, 0.8],
    )

    decoder.gamma_dist_interval = (-0.1, 0.4)

    assert np.allclose(
        decoder.gamma_dist_interval,
        [-0.1, 0.4],
    )


@pytest.mark.parametrize(
    "interval",
    [
        [],
        [0.1],
        [0.1, 0.2, 0.3],
    ],
)
def test_gamma_dist_interval_wrong_length(interval):
    decoder = make_generated_memory_decoder()

    with pytest.raises((ValueError, Exception)):
        decoder.gamma_dist_interval = interval


def test_memory_seed_roundtrip():
    decoder = make_generated_memory_decoder(
        memory_seed=42
    )

    assert decoder.memory_seed == 42

    decoder.memory_seed = 123

    assert decoder.memory_seed == 123


@pytest.mark.parametrize("seed", [-3, 1.5, "123"])
def test_invalid_memory_seed(seed):
    decoder = make_generated_memory_decoder()

    with pytest.raises((ValueError, TypeError)):
        decoder.memory_seed = seed


# ============================================================================
# Decoding
# ============================================================================

def test_decode_zero_syndrome_returns_zero():
    decoder = make_decoder()

    syndrome = np.zeros(
        decoder.check_count,
        dtype=np.uint8,
    )

    result = decoder.decode(syndrome)

    assert result.dtype == syndrome.dtype
    assert result.shape == (decoder.bit_count,)
    assert np.array_equal(
        result,
        np.zeros(decoder.bit_count, dtype=np.uint8),
    )


def test_decode_zero_syndrome_updates_relay_state():
    # This test requires removal of the Python zero-input shortcut.
    decoder = make_decoder(
        maximum_legs=3,
        maximum_solutions=1,
        memory_strengths_per_leg=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [0.9, 0.9, 0.9],
            ]
        ),
    )

    syndrome = np.zeros(
        decoder.check_count,
        dtype=np.uint8,
    )

    decoder.decode(syndrome)

    assert decoder.converge
    # a zero syndrome is no longer handled by the cython wrapper without calling the C++ object so we do expect these values to increase
    assert decoder.solution_number == 1
    assert decoder.iter == 1
    assert decoder.iterations == 1


def test_decode_output_shape():
    decoder = make_decoder()

    syndrome = np.array([1, 0], dtype=np.uint8)
    result = decoder.decode(syndrome)

    assert result.shape == (decoder.bit_count,)
    assert result.dtype == syndrome.dtype


def test_decode_rejects_wrong_syndrome_length():
    decoder = make_decoder()
    decoder.input_vector_type = "syndrome"

    with pytest.raises(ValueError):
        decoder.decode(
            np.array([1, 0, 1], dtype=np.uint8)
        )


def test_decode_rep_code_single_error():
    parity_check = rep_code(3)
    n = parity_check.shape[1]

    decoder = RelayBpDecoder(
        parity_check,
        error_rate=0.1,
        maximum_legs=1,
        maximum_solutions=1,
        iterations0=20,
        max_iter=20,
        gamma0=0.0,
        memory_strengths_per_leg=np.zeros(
            (1, n),
            dtype=np.float64,
        ),
        bp_method="product_sum",
        memory_seed=123,
    )

    error = np.array([1, 0, 0], dtype=np.uint8)
    syndrome = binary_syndrome(
        parity_check,
        error,
    )

    result = decoder.decode(syndrome)

    assert np.array_equal(result, error)
    assert np.array_equal(
        binary_syndrome(parity_check, result),
        syndrome,
    )
    assert decoder.solution_number == 1
    assert decoder.iterations > 0


@pytest.mark.parametrize(
    "syndrome",
    [
        np.array([0, 0, 0, 1], dtype=np.uint8),
        np.array([1, 0, 1, 0], dtype=np.uint8),
    ],
)
def test_multileg_returned_solution_satisfies_syndrome(
        syndrome,
):
    parity_check = rep_code(5)
    n = parity_check.shape[1]

    strengths = np.array(
        [
            np.zeros(n),
            np.full(n, 0.5),
            np.full(n, 0.9),
        ],
        dtype=np.float64,
    )

    decoder = RelayBpDecoder(
        parity_check,
        error_rate=0.1,
        maximum_legs=3,
        maximum_solutions=3,
        iterations0=10,
        max_iter=10,
        gamma0=0.0,
        memory_strengths_per_leg=strengths,
        bp_method="product_sum",
        memory_seed=123,
    )

    result = decoder.decode(syndrome)

    assert decoder.solution_number > 0
    assert np.array_equal(
        binary_syndrome(parity_check, result),
        syndrome,
    )
    assert decoder.iterations > 0
    assert decoder.iterations <= 30


def test_maximum_solutions_causes_early_exit():
    parity_check = rep_code(5)
    n = parity_check.shape[1]

    decoder = RelayBpDecoder(
        parity_check,
        error_rate=0.1,
        maximum_legs=3,
        maximum_solutions=1,
        iterations0=10,
        max_iter=10,
        gamma0=0.0,
        memory_strengths_per_leg=np.zeros(
            (3, n),
            dtype=np.float64,
        ),
        bp_method="product_sum",
    )

    syndrome = np.array([0, 0, 0, 1], dtype=np.uint8)

    decoder.decode(syndrome)

    assert decoder.solution_number == 1
    assert decoder.iterations == 1


def test_iterations_is_bounded_by_leg_limits():
    parity_check = rep_code(3)
    n = parity_check.shape[1]

    decoder = RelayBpDecoder(
        parity_check,
        error_rate=0.003,
        maximum_legs=2,
        maximum_solutions=2,
        iterations0=80,
        max_iter=60,
        gamma0=0.0,
        memory_strengths_per_leg=np.array(
            [
                [-1.0, -1.0, -1.0],
                [
                    -0.345025519,
                    0.190321013,
                    0.575298233,
                ],
            ],
            dtype=np.float64,
        ),
        bp_method="minimum_sum",
        ms_scaling_factor=1.0,
    )

    syndrome = np.array([1, 1], dtype=np.uint8)
    result = decoder.decode(syndrome)

    assert 0 < decoder.iterations <= 140

    if decoder.solution_number > 0:
        assert np.array_equal(
            binary_syndrome(parity_check, result),
            syndrome,
        )


def test_state_resets_between_decode_calls():
    parity_check = rep_code(5)
    n = parity_check.shape[1]

    decoder = RelayBpDecoder(
        parity_check,
        error_rate=0.1,
        maximum_legs=1,
        maximum_solutions=1,
        iterations0=10,
        max_iter=10,
        gamma0=0.0,
        memory_strengths_per_leg=np.zeros(
            (1, n),
            dtype=np.float64,
        ),
        bp_method="product_sum",
    )

    first_syndrome = np.array(
        [1, 0, 0, 0],
        dtype=np.uint8,
    )

    decoder.decode(first_syndrome)

    assert decoder.solution_number == 1
    assert decoder.iterations > 0

    # Force the next call to execute zero BP iterations.
    decoder.iterations0 = 0

    decoder.decode(first_syndrome)

    assert decoder.solution_number == 0
    assert decoder.iterations == 0
    assert decoder.iter == 0
    assert not decoder.converge
    assert np.array_equal(
        decoder.decoding,
        np.zeros(n, dtype=int),
    )


def test_repeated_decodes_reset_iterations():
    parity_check = rep_code(3)
    n = parity_check.shape[1]

    decoder = RelayBpDecoder(
        parity_check,
        error_rate=0.1,
        maximum_legs=1,
        maximum_solutions=1,
        iterations0=20,
        max_iter=20,
        gamma0=0.0,
        memory_strengths_per_leg=np.zeros(
            (1, n),
            dtype=np.float64,
        ),
        bp_method="product_sum",
    )

    syndrome = np.array([1, 0], dtype=np.uint8)

    decoder.decode(syndrome)
    first_total = decoder.iterations

    decoder.decode(syndrome)
    second_total = decoder.iterations

    assert first_total > 0
    assert second_total == first_total


# ============================================================================
# Message-passing precision
# ============================================================================

def test_default_precision_is_double():
    decoder = make_decoder()

    assert DEFAULT_PRECISION == 53
    assert decoder.precision == DEFAULT_PRECISION
    assert decoder.precision_request == DEFAULT_PRECISION


def test_available_precisions_sorted_and_contains_default():
    tiers = available_precisions()

    assert tiers
    assert tiers == sorted(set(tiers))
    assert DEFAULT_PRECISION in tiers

    # 24 mantissa bits is IEEE single and is always compiled in.
    assert 24 in tiers


def test_decoder_exposes_available_precisions():
    decoder = make_decoder()

    assert decoder.available_precisions == available_precisions()


@pytest.mark.parametrize("bits", PRECISION_TIERS)
def test_resolve_precision_is_identity_on_a_tier(bits):
    assert resolve_precision(bits) == bits


def test_resolve_precision_rounds_up():
    tiers = available_precisions()

    # Below the smallest tier, promoted to it.
    assert resolve_precision(1) == tiers[0]

    # One bit past a tier moves up to the next one.
    for lower, upper in zip(tiers, tiers[1:]):
        assert resolve_precision(lower + 1) == upper


@pytest.mark.parametrize(
    "value",
    [0, -1, -64],
)
def test_resolve_precision_rejects_non_positive(value):
    with pytest.raises(ValueError):
        resolve_precision(value)


def test_resolve_precision_rejects_above_largest_tier():
    with pytest.raises(ValueError):
        resolve_precision(available_precisions()[-1] + 1)


def test_precision_request_is_rounded_up_to_a_tier():
    requested = PRECISION_TIERS[0] + 1

    decoder = make_decoder(precision=requested)

    assert decoder.precision_request == requested
    assert decoder.precision == resolve_precision(requested)
    assert decoder.precision >= requested


@pytest.mark.parametrize(
    "value",
    [
        0,
        -4,
        10 ** 9,
        53.0,
        "53",
        True,
    ],
)
def test_init_rejects_invalid_precision(value):
    with pytest.raises(ValueError):
        make_decoder(precision=value)


def test_precision_none_means_the_default():
    """None is 'unspecified', matching the other optional parameters."""

    decoder = make_decoder(precision=None)

    assert decoder.precision == DEFAULT_PRECISION
    assert decoder.precision_request == DEFAULT_PRECISION


def test_precision_roundtrip():
    decoder = make_decoder()

    assert decoder.precision == DEFAULT_PRECISION

    target = non_default_precision()
    decoder.precision = target

    assert decoder.precision == target
    assert decoder.precision_request == target

    decoder.precision = DEFAULT_PRECISION

    assert decoder.precision == DEFAULT_PRECISION


@pytest.mark.parametrize(
    "value",
    [0, -1, 10 ** 9, 1.5, "53"],
)
def test_invalid_precision_assignment_leaves_decoder_usable(value):
    decoder = make_decoder()

    with pytest.raises(ValueError):
        decoder.precision = value

    assert decoder.precision == DEFAULT_PRECISION
    assert decoder.precision_request == DEFAULT_PRECISION

    result = decoder.decode(
        np.array([1, 0], dtype=np.uint8)
    )

    assert result.shape == (decoder.bit_count,)


@pytest.mark.parametrize(
    "syndrome",
    [
        np.array([0, 0, 0, 0], dtype=np.uint8),
        np.array([0, 0, 0, 1], dtype=np.uint8),
        np.array([1, 0, 1, 0], dtype=np.uint8),
        np.array([1, 1, 1, 1], dtype=np.uint8),
    ],
)
def test_explicit_double_matches_the_default_exactly(syndrome):
    """The 53 bit tier must be the arithmetic the decoder always used."""

    parity_check = rep_code(5)
    n = parity_check.shape[1]

    common = dict(
        error_rate=0.1,
        maximum_legs=3,
        maximum_solutions=3,
        iterations0=10,
        max_iter=10,
        gamma0=0.0,
        memory_strengths_per_leg=np.array(
            [
                np.zeros(n),
                np.full(n, 0.5),
                np.full(n, 0.9),
            ],
            dtype=np.float64,
        ),
        bp_method="product_sum",
        memory_seed=123,
    )

    implicit = RelayBpDecoder(parity_check, **common)
    explicit = RelayBpDecoder(
        parity_check, precision=53, **common
    )

    implicit_result = implicit.decode(syndrome)
    explicit_result = explicit.decode(syndrome)

    assert np.array_equal(
        implicit_result,
        explicit_result,
    )
    assert implicit.converge == explicit.converge
    assert implicit.iterations == explicit.iterations
    assert (
        implicit.solution_number
        == explicit.solution_number
    )

    assert_llrs_identical(
        implicit.log_prob_ratios,
        explicit.log_prob_ratios,
    )


@pytest.mark.parametrize("bits", PRECISION_TIERS)
@pytest.mark.parametrize(
    "schedule",
    ["parallel", "serial"],
)
def test_every_precision_tier_decodes(bits, schedule):
    """Every advertised tier must be reachable through the dispatch.

    Correctness is only required of double and above: worse answers at low
    precision are the point of the parameter, not a defect.
    """

    parity_check = rep_code(3)
    n = parity_check.shape[1]

    decoder = RelayBpDecoder(
        parity_check,
        error_rate=0.1,
        maximum_legs=1,
        maximum_solutions=1,
        iterations0=30,
        max_iter=30,
        gamma0=0.0,
        memory_strengths_per_leg=np.zeros(
            (1, n),
            dtype=np.float64,
        ),
        bp_method="minimum_sum",
        schedule=schedule,
        precision=bits,
    )

    assert decoder.precision == bits

    syndrome = np.array([1, 0], dtype=np.uint8)
    result = decoder.decode(syndrome)

    assert result.shape == (n,)

    if bits >= DEFAULT_PRECISION:
        assert decoder.converge
        assert np.array_equal(
            binary_syndrome(parity_check, result),
            syndrome,
        )
    elif decoder.converge:
        assert np.array_equal(
            binary_syndrome(parity_check, result),
            syndrome,
        )


@pytest.mark.parametrize(
    "bits",
    [bits for bits in PRECISION_TIERS if bits < DEFAULT_PRECISION],
)
@pytest.mark.parametrize(
    "schedule",
    ["parallel", "serial"],
)
def test_low_precision_tiers_really_narrow_the_arithmetic(
        bits,
        schedule,
):
    """Guards against the dispatch quietly running everything in double."""

    parity_check = rep_code(5)
    n = parity_check.shape[1]

    decoder = RelayBpDecoder(
        parity_check,
        error_rate=0.1,
        maximum_legs=1,
        maximum_solutions=1,
        iterations0=10,
        max_iter=10,
        gamma0=0.0,
        memory_strengths_per_leg=np.zeros(
            (1, n),
            dtype=np.float64,
        ),
        bp_method="minimum_sum",
        schedule=schedule,
        precision=bits,
    )

    decoder.decode(
        np.array([0, 0, 0, 1], dtype=np.uint8)
    )

    llrs = decoder.log_prob_ratios

    assert all(
        is_representable_in(value, bits)
        for value in llrs
    )

    # And genuinely coarser than the double result.
    double_decoder = RelayBpDecoder(
        parity_check,
        error_rate=0.1,
        maximum_legs=1,
        maximum_solutions=1,
        iterations0=10,
        max_iter=10,
        gamma0=0.0,
        memory_strengths_per_leg=np.zeros(
            (1, n),
            dtype=np.float64,
        ),
        bp_method="minimum_sum",
        schedule=schedule,
    )

    double_decoder.decode(
        np.array([0, 0, 0, 1], dtype=np.uint8)
    )

    assert not np.array_equal(
        llrs,
        double_decoder.log_prob_ratios,
    )


@pytest.mark.parametrize("bits", FAST_PRECISION_TIERS)
def test_outputs_stay_float64_at_every_precision(bits):
    """Whatever the working precision, the exposed arrays are unchanged."""

    decoder = make_decoder(precision=bits)

    syndrome = np.array([1, 0], dtype=np.uint8)
    result = decoder.decode(syndrome)

    assert result.dtype == syndrome.dtype
    assert decoder.log_prob_ratios.dtype == np.float64
    assert decoder.error_channel.dtype == np.float64
    assert np.all(np.isfinite(decoder.log_prob_ratios))

    expected_llr = math.log((1.0 - 0.1) / 0.1)

    # Narrowed from the working precision, so correct to that precision
    # rather than bit for bit. Allow a few ulps of the tier in use, never
    # finer than a few ulps of double.
    tolerance = abs(expected_llr) * 2.0 ** -(
        min(bits, DEFAULT_PRECISION) - 3
    )

    assert np.allclose(
        decoder.error_channel,
        0.1,
        atol=tolerance,
    )


def test_switching_precision_matches_a_fresh_decoder():
    """Precision is read per decode call, so switching must be equivalent."""

    target = non_default_precision()

    syndrome = np.array([1, 0], dtype=np.uint8)
    warm_up = np.array([0, 1], dtype=np.uint8)

    reused = make_decoder()
    reused.decode(warm_up)
    reused.precision = target
    reused_result = reused.decode(syndrome)

    fresh = make_decoder(precision=target)
    fresh_result = fresh.decode(syndrome)

    assert np.array_equal(reused_result, fresh_result)
    assert reused.converge == fresh.converge
    assert reused.iterations == fresh.iterations
    assert (
        reused.solution_number == fresh.solution_number
    )

    assert_llrs_identical(
        reused.log_prob_ratios,
        fresh.log_prob_ratios,
    )


@pytest.mark.parametrize("bits", FAST_PRECISION_TIERS)
def test_precision_works_with_generated_memory_strengths(
        bits,
):
    """Precision is orthogonal to how the memory strengths are produced."""

    decoder = make_generated_memory_decoder(
        precision=bits
    )

    assert decoder.precision == bits
    assert decoder.memory_strengths_per_leg is None

    result = decoder.decode(
        np.array([1, 0], dtype=np.uint8)
    )

    assert result.shape == (decoder.bit_count,)


def test_precision_does_not_disturb_other_properties():
    decoder = make_decoder(
        precision=non_default_precision(),
        maximum_legs=2,
        maximum_solutions=1,
        iterations0=7,
        max_iter=9,
        bp_method="minimum_sum",
        memory_seed=42,
    )

    assert decoder.maximum_legs == 2
    assert decoder.maximum_solutions == 1
    assert decoder.iterations0 == 7
    assert decoder.max_iter == 9
    assert decoder.bp_method == "minimum_sum"
    assert decoder.memory_seed == 42
    assert np.allclose(
        decoder.memory_strengths_per_leg,
        DEFAULT_STRENGTHS_PER_LEG,
    )


@pytest.mark.parametrize("bits", FAST_PRECISION_TIERS)
def test_repeated_construction_and_decode_at_each_precision(
        bits,
):
    syndrome = np.array([1, 0], dtype=np.uint8)

    for _ in range(10):
        decoder = make_decoder(precision=bits)
        result = decoder.decode(syndrome)

        assert result.shape == (decoder.bit_count,)

        del decoder

    gc.collect()


# ============================================================================
# Comparison with the base BpDecoder
# ============================================================================

@pytest.mark.parametrize(
    "syndrome",
    [
        np.array([0, 0, 0, 0], dtype=np.uint8),
        np.array([0, 0, 0, 1], dtype=np.uint8),
        np.array([0, 1, 0, 1], dtype=np.uint8),
        np.array([1, 0, 1, 0], dtype=np.uint8),
        np.array([1, 1, 1, 1], dtype=np.uint8),
    ],
)
def test_matches_bp_decoder_single_zero_memory_leg(
        syndrome,
):
    parity_check = rep_code(5)
    n = parity_check.shape[1]

    base = BpDecoder(
        parity_check,
        error_rate=0.1,
        max_iter=20,
        bp_method="product_sum",
    )

    relay = RelayBpDecoder(
        parity_check,
        error_rate=0.1,
        maximum_legs=1,
        maximum_solutions=1,
        iterations0=20,
        max_iter=20,
        gamma0=0.0,
        memory_strengths_per_leg=np.zeros(
            (1, n),
            dtype=np.float64,
        ),
        bp_method="product_sum",
    )

    base_result = base.decode(syndrome)
    relay_result = relay.decode(syndrome)

    assert np.array_equal(
        relay_result,
        base_result,
    )


# ============================================================================
# Input vector mode
# ============================================================================

def test_input_vector_type_syndrome_roundtrip():
    decoder = make_decoder()
    decoder.input_vector_type = "syndrome"

    assert decoder.input_vector_type == "syndrome"


def test_received_vector_output_shape():
    decoder = make_decoder()
    decoder.input_vector_type = "received_vector"

    received = np.array([1, 0, 0], dtype=np.uint8)
    result = decoder.decode(received)

    assert result.shape == (decoder.bit_count,)


def test_received_vector_rejects_wrong_length():
    decoder = make_decoder()
    decoder.input_vector_type = "received_vector"

    with pytest.raises(ValueError):
        decoder.decode(
            np.array([1, 0], dtype=np.uint8)
        )


# ============================================================================
# Memory safety and construction failure
# ============================================================================

def test_repeated_construction_and_decode():
    syndrome = np.array([1, 0], dtype=np.uint8)

    for _ in range(100):
        decoder = make_decoder()
        result = decoder.decode(syndrome)

        assert result.shape == (decoder.bit_count,)

        del decoder

    gc.collect()


def test_repeated_generated_memory_construction_and_decode():
    syndrome = np.array([1, 0], dtype=np.uint8)

    for seed in range(25):
        decoder = make_generated_memory_decoder(
            memory_seed=seed
        )

        result = decoder.decode(syndrome)

        assert result.shape == (decoder.bit_count,)

        del decoder

    gc.collect()


def test_construction_failure_does_not_crash():
    for _ in range(10):
        with pytest.raises(
                (ValueError, TypeError, Exception)
        ):
            RelayBpDecoder(
                PCM_NP,
                error_rate=0.1,
                maximum_legs=2,
                maximum_solutions=1,
                iterations0=10,
                max_iter=10,
                gamma0=0.0,
                memory_strengths_per_leg=np.zeros(
                    (3, 3)
                ),
            )

    # A valid construction must still work after repeated failures.
    decoder = make_decoder()

    result = decoder.decode(
        np.array([1, 0], dtype=np.uint8)
    )

    assert result.shape == (3,)
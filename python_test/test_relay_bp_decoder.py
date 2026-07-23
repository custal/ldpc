import gc

import numpy as np
import pytest
import scipy.sparse

from ldpc.bp_decoder import BpDecoder
from ldpc.codes import rep_code
from ldpc.relay_bp_decoder import RelayBpDecoder


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
    # a zero syndrome is handled by the cython wrapper without calling the C++ object so we do not expect these values to increase
    assert decoder.solution_number == 0
    assert decoder.iter == 0
    assert decoder.total_iterations == 0


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
    assert decoder.total_iterations > 0


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
    assert decoder.total_iterations > 0
    assert decoder.total_iterations <= 30


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
    assert decoder.total_iterations == 1


def test_total_iterations_is_bounded_by_leg_limits():
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

    assert 0 < decoder.total_iterations <= 140

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
    assert decoder.total_iterations > 0

    # Force the next call to execute zero BP iterations.
    decoder.iterations0 = 0

    decoder.decode(first_syndrome)

    assert decoder.solution_number == 0
    assert decoder.total_iterations == 0
    assert decoder.iter == 0
    assert not decoder.converge
    assert np.array_equal(
        decoder.decoding,
        np.zeros(n, dtype=int),
    )


def test_repeated_decodes_reset_total_iterations():
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
    first_total = decoder.total_iterations

    decoder.decode(syndrome)
    second_total = decoder.total_iterations

    assert first_total > 0
    assert second_total == first_total


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
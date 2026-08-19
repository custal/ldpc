"""Tests for the Python/Cython AutBpDecoder wrapper.

These tests intentionally focus on the wrapper contract: input conversion,
validation, ownership, immutable snapshots, and last-decode state.  Numerical
BP behaviour is checked through invariants rather than a particular tie-break.
"""

import gc

import numpy as np
import pytest
import scipy.sparse

from ldpc.aut_bp_decoder import AutBpDecoder


PCM_NP = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
PCM_SP = scipy.sparse.csr_matrix(PCM_NP)
PRIORS = (0.1, 0.1, 0.1)
IDENTITY = ((0, 1, 2), (0, 1))
SWAP_01 = ((1, 0, 2), (1, 0))


def binary_syndrome(pcm, decoding):
    return np.asarray(pcm @ np.asarray(decoding, dtype=np.uint8)).ravel() % 2


def make_decoder(pcm=PCM_NP, **kwargs):
    defaults = dict(
        priors=PRIORS,
        permutations=[IDENTITY],
        maximum_solutions=1,
        decoder_type="bp",
        maximum_iterations=10,
    )
    defaults.update(kwargs)
    return AutBpDecoder(pcm, **defaults)


def make_relay_decoder(pcm=PCM_NP, **kwargs):
    defaults = dict(
        priors=PRIORS,
        permutations=[IDENTITY],
        maximum_solutions=1,
        decoder_type="relay",
        maximum_legs=2,
        relay_maximum_solutions=1,
        iterations0=3,
        maximum_iterations=5,
        gamma0=0.0,
        memory_strengths_per_leg=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
    )
    defaults.update(kwargs)
    return AutBpDecoder(pcm, **defaults)


# ---------------------------------------------------------------------------
# Construction and PCM conversion
# ---------------------------------------------------------------------------


def test_init_numpy_pcm():
    decoder = make_decoder(PCM_NP)
    assert (decoder.check_count, decoder.bit_count) == PCM_NP.shape


def test_init_sparse_pcm():
    decoder = make_decoder(PCM_SP)
    assert (decoder.check_count, decoder.bit_count) == PCM_NP.shape


def test_dense_and_sparse_have_same_zero_syndrome_result():
    syndrome = np.zeros(PCM_NP.shape[0], dtype=np.uint8)
    dense = make_decoder(PCM_NP)
    sparse = make_decoder(PCM_SP)
    assert dense.decode(syndrome) == sparse.decode(syndrome)


@pytest.mark.parametrize("pcm", ["not-a-matrix", np.zeros(3), np.zeros((1, 1, 1))])
def test_init_rejects_invalid_pcm(pcm):
    with pytest.raises((TypeError, ValueError)):
        make_decoder(pcm)


def test_priors_length_must_match_pcm_columns():
    with pytest.raises(ValueError, match="priors length"):
        make_decoder(priors=(0.1, 0.1))


def test_pcm_is_copied_and_need_not_remain_alive():
    pcm = PCM_NP.copy()
    decoder = make_decoder(pcm)
    pcm[:, :] = 0
    del pcm
    gc.collect()
    assert decoder.decode(np.array([1, 0], dtype=np.uint8)) is not None


def test_sparse_pcm_is_copied_and_need_not_remain_alive():
    pcm = PCM_SP.copy()
    decoder = make_decoder(pcm)
    del pcm
    gc.collect()
    assert decoder.decode(np.array([1, 0], dtype=np.uint8)) is not None


# ---------------------------------------------------------------------------
# Automorphism source and permutation conversion
# ---------------------------------------------------------------------------


def test_requires_exactly_one_automorphism_source():
    with pytest.raises(ValueError, match="max_automorphisms"):
        AutBpDecoder(PCM_NP, PRIORS, decoder_type="bp")

    with pytest.raises(ValueError, match="not both"):
        AutBpDecoder(
            PCM_NP,
            PRIORS,
            max_automorphisms=2,
            permutations=[IDENTITY],
            decoder_type="bp",
        )


@pytest.mark.parametrize("value", [None, 0, -1])
def test_automatic_mode_requires_positive_max_automorphisms(value):
    with pytest.raises(ValueError, match="positive"):
        AutBpDecoder(
            PCM_NP,
            PRIORS,
            max_automorphisms=value,
            decoder_type="bp",
        )


def test_explicit_permutations_must_be_non_empty():
    with pytest.raises(ValueError, match="non-empty"):
        make_decoder(permutations=[])


def test_permutation_accepts_pair_and_dict_forms():
    decoder = make_decoder(
        permutations=[
            IDENTITY,
            {"old_col_for_new": SWAP_01[0], "old_row_for_new": SWAP_01[1]},
        ]
    )
    assert decoder.get_permutations() == (IDENTITY, SWAP_01)


@pytest.mark.parametrize(
    "permutations",
    [
        [((-1, 1, 2), (0, 1))],
        [((0, 1, 2), (-1, 1))],
    ],
)
def test_permutation_indices_must_be_non_negative(permutations):
    with pytest.raises(ValueError, match="non-negative"):
        make_decoder(permutations=permutations)


def test_permutation_input_is_snapshotted():
    columns = [0, 1, 2]
    rows = [0, 1]
    specifications = [(columns, rows)]
    decoder = make_decoder(permutations=specifications)
    columns[0] = 2
    rows[0] = 1
    specifications.clear()
    assert decoder.get_permutations() == (IDENTITY,)


def test_permutations_property_returns_fresh_immutable_snapshot():
    decoder = make_decoder(permutations=[IDENTITY, SWAP_01])
    first = decoder.get_permutations()
    second = decoder.get_permutations()
    assert first == second == (IDENTITY, SWAP_01)
    assert first is not second
    with pytest.raises(TypeError):
        first[0][0][0] = 2


def test_permutations_can_be_a_generator():
    decoder = make_decoder(permutations=(item for item in [IDENTITY, SWAP_01]))
    assert decoder.get_permutations() == (IDENTITY, SWAP_01)


# ---------------------------------------------------------------------------
# Constructor snapshots and configuration properties
# ---------------------------------------------------------------------------


def test_priors_accept_generator_and_are_snapshotted():
    decoder = make_decoder(priors=(x for x in PRIORS))
    assert decoder.priors == PRIORS
    assert isinstance(decoder.priors, tuple)


def test_bp_configuration_roundtrip():
    decoder = make_decoder(
        maximum_iterations=7,
        bp_method=0,
        schedule=0,
        min_sum_scaling_factor=0.625,
        omp_threads=2,
        serial_schedule=(2, 0, 1),
        random_schedule_seed=123,
        random_serial_schedule=False,
        bp_input_type=0,
    )
    assert decoder.decoder_type == "bp"
    assert decoder.maximum_iterations == 7
    assert decoder.bp_method == "product_sum"
    assert decoder.schedule == "serial"
    assert decoder.min_sum_scaling_factor == pytest.approx(0.625)
    assert decoder.omp_threads == 2
    assert decoder.serial_schedule == (2, 0, 1)
    assert decoder.random_schedule_seed == 123
    assert decoder.random_serial_schedule is False
    assert decoder.bp_input_type == "syndrome"


def test_relay_configuration_roundtrip():
    strengths = ((0.0, 0.0, 0.0), (0.25, 0.5, 0.75))
    decoder = make_relay_decoder(
        maximum_legs=2,
        relay_maximum_solutions=2,
        iterations0=4,
        maximum_iterations=6,
        gamma0=-0.5,
        gamma_dist_interval=(-0.25, 0.75),
        memory_strengths_per_leg=strengths,
        memory_seed=42,
    )
    assert decoder.decoder_type == "relay"
    assert decoder.maximum_legs == 2
    assert decoder.relay_maximum_solutions == 2
    assert decoder.iterations0 == 4
    assert decoder.maximum_iterations == 6
    assert decoder.gamma0 == pytest.approx(-0.5)
    assert decoder.gamma_dist_interval == (-0.25, 0.75)
    assert decoder.memory_strengths_per_leg == strengths
    assert decoder.memory_seed == 42


def test_iterable_configuration_inputs_are_materialised_once():
    decoder = make_relay_decoder(
        gamma_dist_interval=(x for x in (-0.5, 0.5)),
        memory_strengths_per_leg=(
            (x for x in row)
            for row in ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
        ),
        serial_schedule=(x for x in (2, 1, 0)),
    )
    assert decoder.gamma_dist_interval == (-0.5, 0.5)
    assert decoder.memory_strengths_per_leg == (
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
    )
    assert decoder.serial_schedule == (2, 1, 0)


@pytest.mark.parametrize("decoder_type", ["", "BP", "relay_bp", "invalid"])
def test_invalid_decoder_type(decoder_type):
    with pytest.raises(ValueError, match="decoder_type"):
        make_decoder(decoder_type=decoder_type)


@pytest.mark.parametrize("value", [0, -1])
def test_maximum_solutions_must_be_positive_or_none(value):
    with pytest.raises(ValueError, match="maximum_solutions"):
        make_decoder(maximum_solutions=value)


def test_maximum_solutions_none_roundtrip():
    decoder = make_decoder(maximum_solutions=None)
    assert decoder.maximum_solutions is None


def test_include_identity_roundtrip_in_automatic_mode():
    decoder = AutBpDecoder(
        PCM_NP,
        PRIORS,
        max_automorphisms=2,
        include_identity=False,
        decoder_type="bp",
        maximum_iterations=5,
    )
    assert decoder.max_automorphisms == 2
    assert decoder.include_identity is False


# ---------------------------------------------------------------------------
# Decode and state
# ---------------------------------------------------------------------------


def test_decode_returns_python_list_and_updates_public_state():
    decoder = make_decoder()
    returned = decoder.decode(np.zeros(PCM_NP.shape[0], dtype=np.uint8))
    assert isinstance(returned, list)
    assert returned == decoder.decoding
    assert len(returned) == decoder.bit_count
    assert all(bit in (0, 1) for bit in returned)
    assert isinstance(decoder.solution_number, int)
    assert isinstance(decoder.iterations, int)
    assert isinstance(decoder.converge, bool)
    assert len(decoder.log_prob_ratios) == decoder.bit_count


def test_decoding_satisfies_syndrome_when_converged():
    decoder = make_decoder(maximum_iterations=20)
    syndrome = np.array([1, 0], dtype=np.uint8)
    result = decoder.decode(syndrome)
    if not decoder.converge:
        pytest.skip("This valid test instance did not converge with this backend")
    np.testing.assert_array_equal(binary_syndrome(PCM_NP, result), syndrome)


@pytest.mark.parametrize(
    "syndrome",
    [
        np.array([0], dtype=np.uint8),
        np.array([0, 0, 0], dtype=np.uint8),
    ],
)
def test_decode_rejects_wrong_syndrome_length(syndrome):
    decoder = make_decoder()
    with pytest.raises((ValueError, TypeError)):
        decoder.decode(syndrome)


def test_decode_accepts_array_like_syndrome():
    decoder = make_decoder()
    result = decoder.decode([0, 0])
    assert len(result) == decoder.bit_count


def test_repeated_decode_resets_member_stats():
    decoder = make_decoder(permutations=[IDENTITY, SWAP_01])
    decoder.decode([1, 0])
    first = decoder.last_member_stats
    decoder.decode([0, 0])
    second = decoder.last_member_stats
    assert len(first) == len(decoder.get_permutations())
    assert len(second) == len(decoder.get_permutations())
    assert all(isinstance(i, int) and isinstance(c, bool) for i, c in second)


def test_last_member_stats_returns_fresh_immutable_snapshot():
    decoder = make_decoder(permutations=[IDENTITY, SWAP_01])
    decoder.decode([0, 0])
    first = decoder.last_member_stats
    second = decoder.last_member_stats
    assert first == second
    assert first is not second
    assert isinstance(first, tuple)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in first)


def test_decoding_and_log_prob_ratios_are_defensive_copies():
    decoder = make_decoder()
    decoder.decode([0, 0])
    decoding = decoder.decoding
    ratios = decoder.log_prob_ratios
    decoding[:] = [1] * len(decoding)
    ratios[:] = [999.0] * len(ratios)
    assert decoder.decoding != decoding
    assert decoder.log_prob_ratios != ratios


# ---------------------------------------------------------------------------
# Repeated use and failure cleanup
# ---------------------------------------------------------------------------


def test_repeated_construction_and_decode():
    for _ in range(25):
        decoder = make_decoder()
        assert len(decoder.decode([1, 0])) == PCM_NP.shape[1]
        del decoder
    gc.collect()


def test_repeated_relay_construction_and_decode():
    for _ in range(25):
        decoder = make_relay_decoder()
        assert len(decoder.decode([1, 0])) == PCM_NP.shape[1]
        del decoder
    gc.collect()


def test_repeated_construction_failure_does_not_crash():
    for _ in range(25):
        with pytest.raises(ValueError):
            make_decoder(priors=(0.1, 0.1))
    gc.collect()

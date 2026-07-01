import pytest
import numpy as np
import scipy.sparse
from ldpc.codes import rep_code, ring_code
from ldpc.relay_bp_decoder import RelayBpDecoder
from ldpc.bp_decoder import BpDecoder


# ── fixtures ────────────────────────────────────────────────────────────────

PCM_NP = np.array([[1, 0, 1], [0, 1, 1]])
PCM_SP = scipy.sparse.csr_matrix([[1, 0, 1], [0, 1, 1]])

# A simple 2-leg config for the 3-bit code
DEFAULT_LEGS = 2
DEFAULT_SOLUTIONS = 1
DEFAULT_ITERS_PER_LEG = np.array([10, 10])
DEFAULT_STRENGTHS_PER_LEG = np.ones((2, 3)) * 0.5
IBM_IMPLEMENTATION = False


def make_decoder(pcm=PCM_NP, **kwargs):
    """Helper to construct a RelayBpDecoder with sensible defaults."""
    defaults = dict(
        error_rate=0.1,
        maximum_legs=DEFAULT_LEGS,
        maximum_solutions=DEFAULT_SOLUTIONS,
        maximum_iterations_per_leg=DEFAULT_ITERS_PER_LEG.copy(),
        memory_strengths_per_leg=DEFAULT_STRENGTHS_PER_LEG.copy(),
        ibm_implementation=IBM_IMPLEMENTATION
    )
    defaults.update(kwargs)
    return RelayBpDecoder(pcm, **defaults)


# ── construction ────────────────────────────────────────────────────────────

def test_init_numpy_pcm():
    decoder = make_decoder(PCM_NP)
    assert decoder is not None
    assert decoder.check_count == 2
    assert decoder.bit_count == 3


def test_init_sparse_pcm():
    decoder = make_decoder(PCM_SP)
    assert decoder is not None
    assert decoder.check_count == 2
    assert decoder.bit_count == 3


def test_init_invalid_pcm():
    with pytest.raises(TypeError):
        make_decoder("invalid_pcm")


def test_init_requires_maximum_legs():
    with pytest.raises((ValueError, TypeError)):
        RelayBpDecoder(
            PCM_NP,
            error_rate=0.1,
            maximum_solutions=DEFAULT_SOLUTIONS,
            maximum_iterations_per_leg=DEFAULT_ITERS_PER_LEG.copy(),
            memory_strengths_per_leg=DEFAULT_STRENGTHS_PER_LEG.copy(),
        )


def test_init_requires_maximum_solutions():
    with pytest.raises((ValueError, TypeError)):
        RelayBpDecoder(
            PCM_NP,
            error_rate=0.1,
            maximum_legs=DEFAULT_LEGS,
            maximum_iterations_per_leg=DEFAULT_ITERS_PER_LEG.copy(),
            memory_strengths_per_leg=DEFAULT_STRENGTHS_PER_LEG.copy(),
        )


def test_init_requires_maximum_iterations_per_leg():
    with pytest.raises((ValueError, TypeError)):
        RelayBpDecoder(
            PCM_NP,
            error_rate=0.1,
            maximum_legs=DEFAULT_LEGS,
            maximum_solutions=DEFAULT_SOLUTIONS,
            memory_strengths_per_leg=DEFAULT_STRENGTHS_PER_LEG.copy(),
        )


def test_init_requires_memory_strengths_per_leg():
    with pytest.raises((ValueError, TypeError)):
        RelayBpDecoder(
            PCM_NP,
            error_rate=0.1,
            maximum_legs=DEFAULT_LEGS,
            maximum_solutions=DEFAULT_SOLUTIONS,
            maximum_iterations_per_leg=DEFAULT_ITERS_PER_LEG.copy(),
        )


def test_init_requires_error_rate_or_channel():
    with pytest.raises(ValueError):
        RelayBpDecoder(
            PCM_NP,
            maximum_legs=DEFAULT_LEGS,
            maximum_solutions=DEFAULT_SOLUTIONS,
            maximum_iterations_per_leg=DEFAULT_ITERS_PER_LEG.copy(),
            memory_strengths_per_leg=DEFAULT_STRENGTHS_PER_LEG.copy(),
        )


# ── inherited property round-trips ──────────────────────────────────────────

def test_max_iter_roundtrip():
    decoder = make_decoder(max_iter=10)
    assert decoder.max_iter == 10
    decoder.max_iter = 5
    assert decoder.max_iter == 5


def test_invalid_max_iter_type():
    with pytest.raises(TypeError):
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
    with pytest.raises(ValueError):
        d = make_decoder()
        d.bp_method = "invalid"


def test_schedule_roundtrip():
    decoder = make_decoder(schedule="parallel")
    assert decoder.schedule == "parallel"
    decoder.schedule = "serial"
    assert decoder.schedule == "serial"


def test_invalid_schedule():
    with pytest.raises(ValueError):
        d = make_decoder()
        d.schedule = "invalid"


def test_ms_scaling_factor_roundtrip():
    decoder = make_decoder(ms_scaling_factor=0.5)
    assert decoder.ms_scaling_factor == 0.5
    decoder.ms_scaling_factor = 0.75
    assert decoder.ms_scaling_factor == 0.75


def test_invalid_ms_scaling_factor():
    with pytest.raises(TypeError):
        d = make_decoder()
        d.ms_scaling_factor = "invalid"


def test_error_rate_roundtrip():
    decoder = make_decoder(error_rate=0.1)
    assert np.allclose(decoder.error_rate, 0.1)
    decoder.error_rate = 0.2
    assert np.allclose(decoder.error_rate, 0.2)


def test_error_channel_roundtrip():
    channel = np.array([0.1, 0.2, 0.3])
    decoder = make_decoder(error_channel=channel)
    assert np.allclose(decoder.error_channel, channel)
    new_channel = np.array([0.3, 0.2, 0.1])
    decoder.error_channel = new_channel
    assert np.allclose(decoder.error_channel, new_channel)


def test_error_channel_wrong_length():
    with pytest.raises(ValueError):
        d = make_decoder()
        d.error_channel = np.array([0.1, 0.2])  # should be length 3


# ── relay-specific property round-trips ─────────────────────────────────────

def test_maximum_legs_readable():
    decoder = make_decoder(maximum_legs=3,
                           maximum_iterations_per_leg=np.array([5, 5, 5]),
                           memory_strengths_per_leg=np.ones((3, 3)) * 0.5)
    assert decoder.maximum_legs == 3


def test_maximum_solutions_readable():
    decoder = make_decoder(maximum_solutions=2)
    assert decoder.maximum_solutions == 2


def test_maximum_iterations_per_leg_roundtrip():
    iters = np.array([5, 10])
    decoder = make_decoder(maximum_iterations_per_leg=iters)
    assert np.array_equal(decoder.maximum_iterations_per_leg, iters)
    new_iters = np.array([3, 7])
    decoder.maximum_iterations_per_leg = new_iters
    assert np.array_equal(decoder.maximum_iterations_per_leg, new_iters)


def test_maximum_iterations_per_leg_wrong_length():
    with pytest.raises(Exception):
        d = make_decoder()
        d.maximum_iterations_per_leg = np.array([5])  # should be length 2


def test_memory_strengths_per_leg_roundtrip():
    strengths = np.ones((2, 3)) * 0.3
    decoder = make_decoder(memory_strengths_per_leg=strengths)
    assert np.allclose(decoder.memory_strengths_per_leg, strengths)
    new_strengths = np.ones((2, 3)) * 0.7
    decoder.memory_strengths_per_leg = new_strengths
    assert np.allclose(decoder.memory_strengths_per_leg, new_strengths)


def test_memory_strengths_per_leg_wrong_shape():
    with pytest.raises(Exception):
        d = make_decoder()
        d.memory_strengths_per_leg = np.ones((3, 3)) * 0.5  # wrong leg count


def test_ibm_implementation_roundtrip():
    ibm_implementation = True
    decoder = make_decoder(ibm_implementation=ibm_implementation)
    assert decoder.ibm_implementation == ibm_implementation
    new_ibm_implementation = False
    decoder.ibm_implementation = new_ibm_implementation
    assert decoder.ibm_implementation == new_ibm_implementation

# ── decoding ────────────────────────────────────────────────────────────────

def test_decode_zero_syndrome_returns_zero():
    decoder = make_decoder()
    syndrome = np.zeros(decoder.check_count, dtype=np.uint8)
    result = decoder.decode(syndrome)
    assert np.all(result == 0)
    assert result.shape == (decoder.bit_count,)


def test_decode_output_shape():
    decoder = make_decoder()
    syndrome = np.array([1, 0], dtype=np.uint8)
    result = decoder.decode(syndrome)
    assert result.shape == (decoder.bit_count,)


def test_decode_rep_code_single_error():
    """Single bit flip should be correctable."""
    H = rep_code(3)
    iters = np.array([20, 20])
    strengths = np.zeros((2, 3))  # zero memory strength = standard BP
    decoder = RelayBpDecoder(
        H,
        error_rate=0.1,
        maximum_legs=2,
        maximum_solutions=1,
        maximum_iterations_per_leg=iters,
        memory_strengths_per_leg=strengths,
    )
    error = np.array([1, 0, 0], dtype=np.uint8)
    syndrome = (H.toarray() @ error) % 2
    result = decoder.decode(syndrome.astype(np.uint8))
    assert np.array_equal(result, error)


# ── per-leg outputs ──────────────────────────────────────────────────────────

def test_per_leg_output_shapes():
    legs = 3
    decoder = RelayBpDecoder(
        PCM_NP,
        error_rate=0.1,
        maximum_legs=legs,
        maximum_solutions=1,
        maximum_iterations_per_leg=np.array([5, 5, 5]),
        memory_strengths_per_leg=np.ones((legs, 3)) * 0.5,
    )
    syndrome = np.zeros(decoder.check_count, dtype=np.uint8)
    decoder.decode(syndrome)
    assert decoder.decoding_per_leg.shape == (legs, decoder.bit_count)
    assert decoder.iterations_per_leg.shape == (legs,)
    assert decoder.convergence_per_leg.shape == (legs,)
    assert decoder.log_prob_ratios_per_leg.shape == (legs, decoder.bit_count)


def test_solution_number_readable():
    decoder = make_decoder()
    syndrome = np.zeros(decoder.check_count, dtype=np.uint8)
    decoder.decode(syndrome)
    assert isinstance(decoder.solution_number, int)


# ── memory safety ────────────────────────────────────────────────────────────

def test_repeated_construction_and_decode():
    """Exercises __cinit__ and __dealloc__ repeatedly to catch leaks or crashes."""
    for _ in range(100):
        decoder = make_decoder()
        syndrome = np.zeros(decoder.check_count, dtype=np.uint8)
        decoder.decode(syndrome)


def test_construction_failure_does_not_crash():
    """Invalid args should raise cleanly without corrupting state."""
    for _ in range(10):
        with pytest.raises((ValueError, TypeError)):
            RelayBpDecoder(PCM_NP, error_rate=0.1)


# ── comparison with base BpDecoder ───────────────────────────────────────────

def test_matches_bp_decoder_zero_memory():
    """
    With zero memory strength and 1 leg, relay BP reduces to standard BP.
    Results should match for the same syndrome.
    """
    H = rep_code(5)
    n = H.shape[1]
    syndrome = np.array([1, 1, 0, 0], dtype=np.uint8)

    base = BpDecoder(H, error_rate=0.1, max_iter=20, bp_method="product_sum")

    relay = RelayBpDecoder(
        H,
        error_rate=0.1,
        maximum_legs=1,
        maximum_solutions=1,
        maximum_iterations_per_leg=np.array([20]),
        memory_strengths_per_leg=np.zeros((1, n)),
        bp_method="product_sum",
    )

    base_result = base.decode(syndrome)
    relay_result = relay.decode(syndrome)
    assert np.array_equal(base_result, relay_result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
from libc.stdint import uint8_t
from libc.stddef import size_t
from libcpp import bool as cpp_bool
from libcpp.optional import optional
from libcpp.vector import vector
from libcpp.memory import unique_ptr
from libcpp.utility import move
import cython


def _normalise_permutations(object specifications):
    """Return a stable, immutable Python snapshot of permutation inputs."""


class AutBpDecoder:
    """Python owner for ``ldpc::autbp::AutBpDecoder``.

    Supply either ``max_automorphisms`` for BLISS discovery or ``permutations``
    for a predefined sequence. Each predefined item is a dict with
    ``old_col_for_new`` and optional ``old_row_for_new``, or a ``(cols, rows)``
    pair. ``decoder_type`` is either ``"relay"`` (default) or ``"bp"``. Relay-only
    arguments are retained as properties even when the plain BP factory is
    selected, making the complete construction configuration inspectable.
    """

    def __cinit__(
        self,
        pcm,
        priors,
        max_automorphisms=None,
        permutations=None,
        maximum_solutions=None,
        bint include_identity=True,
        str decoder_type="relay",
        int maximum_legs=1,
        int relay_maximum_solutions=1,
        int iterations0=0,
        int maximum_iterations=0,
        double gamma0=0.0,
        gamma_dist_interval=(),
        memory_strengths_per_leg=(),
        int bp_method=<int>MINIMUM_SUM,
        int schedule=<int>PARALLEL,
        double min_sum_scaling_factor=1.0,
        int omp_threads=1,
        serial_schedule=(),
        int random_schedule_seed=0,
        bint random_serial_schedule=False,
        int bp_input_type=<int>AUTO,
        int memory_seed=-1,
    ): ...

    def decode(self, syndrome): ...
    @property
    def check_count(self): return self._pcm.get().m
    @property
    def bit_count(self): return self._pcm.get().n
    @property
    def priors(self): return self._priors

    def get_permutations(self):
        """All permutations actually used by the C++ decoder.

        This includes BLISS-discovered permutations in automatic mode and the
        validated predefined permutations in explicit mode. A fresh immutable
        Python snapshot is produced on every access.
        """

    @property
    def last_member_stats(self):
        """Per-member ``(iterations, converged)`` results from the last decode.

        The entries follow the same order as :attr:`permutations`. A fresh
        immutable Python snapshot is produced on every access.
        """

    @property
    def max_automorphisms(self): return self._max_automorphisms
    @property
    def maximum_solutions(self): return self._maximum_solutions
    @property
    def include_identity(self): return bool(self._include_identity)
    @property
    def decoder_type(self): return self._decoder_type
    @property
    def maximum_legs(self): return self._maximum_legs
    @property
    def relay_maximum_solutions(self): return self._relay_maximum_solutions
    @property
    def iterations0(self): return self._iterations0
    @property
    def maximum_iterations(self): return self._maximum_iterations
    @property
    def gamma0(self): return self._gamma0
    @property
    def gamma_dist_interval(self): return self._gamma_dist_interval
    @property
    def memory_strengths_per_leg(self): return self._memory_strengths_per_leg
    @property
    def bp_method(self): return self._bp_method
    @property
    def schedule(self): return self._schedule
    @property
    def min_sum_scaling_factor(self): return self._min_sum_scaling_factor
    @property
    def omp_threads(self): return self._omp_threads
    @property
    def serial_schedule(self): return self._serial_schedule
    @property
    def random_schedule_seed(self): return self._random_schedule_seed
    @property
    def random_serial_schedule(self): return bool(self._random_serial_schedule)
    @property
    def bp_input_type(self): return self._bp_input_type
    @property
    def memory_seed(self): return self._memory_seed
    @property
    def solution_number(self): return self._decoder.get().solution_number
    @property
    def iterations(self): return self._decoder.get().iterations
    @property
    def decoding(self): ...
    @property
    def converge(self): return bool(self._decoder.get().converge)
    @property
    def log_prob_ratios(self): ...

# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False

"""Cython wrapper for ldpc::autbp::AutBpDecoder.

`pcm` may be a dense two-dimensional array-like object or any SciPy-style
sparse matrix exposing ``tocoo()``.  The wrapper copies the nonzero pattern,
so the Python matrix need not remain alive.
"""

from libc.stdint cimport uint8_t
from libc.stddef cimport size_t
from libcpp cimport bool as cpp_bool
from libcpp.optional cimport optional
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

cimport cython

cdef str _normalise_enum_string(object value, str parameter_name):
    """Convert an enum-like Python value to a normalised string."""
    if not isinstance(value, str):
        raise TypeError(
            f"{parameter_name} must be a string, "
            f"not {type(value).__name__}"
        )

    return value.strip().lower().replace("-", "_").replace(" ", "_")


cdef BpMethod _bp_method_from_python(object value) except *:
    cdef str normalised

    # Also permit an already-converted enum value.
    if isinstance(value, int):
        if value == <int>BpMethod.MINIMUM_SUM:
            return BpMethod.MINIMUM_SUM
        if value == <int>BpMethod.PRODUCT_SUM:
            return BpMethod.PRODUCT_SUM

    normalised = _normalise_enum_string(value, "bp_method")

    if normalised in ("minimum_sum", "min_sum", "minsum", "ms"):
        return BpMethod.MINIMUM_SUM

    if normalised in ("product_sum", "sum_product", "productsum", "ps"):
        return BpMethod.PRODUCT_SUM

    raise ValueError(
        "bp_method must be one of "
        "'minimum_sum' or 'product_sum'; "
        f"received {value!r}"
    )


cdef BpSchedule _bp_schedule_from_python(object value) except *:
    cdef str normalised

    if isinstance(value, int):
        if value == <int>BpSchedule.PARALLEL:
            return BpSchedule.PARALLEL
        if value == <int>BpSchedule.SERIAL:
            return BpSchedule.SERIAL
        if value == <int>BpSchedule.SERIAL_RELATIVE:
            return BpSchedule.SERIAL_RELATIVE

    normalised = _normalise_enum_string(value, "bp_schedule")

    if normalised == "parallel":
        return BpSchedule.PARALLEL

    if normalised == "serial":
        return BpSchedule.SERIAL

    if normalised in ("serial_relative", "relative_serial"):
        return BpSchedule.SERIAL_RELATIVE

    raise ValueError(
        "bp_schedule must be one of "
        "'parallel', 'serial', or 'serial_relative'; "
        f"received {value!r}"
    )


cdef BpInputType _bp_input_type_from_python(object value) except *:
    cdef str normalised

    if isinstance(value, int):
        if value == <int>BpInputType.AUTO:
            return BpInputType.AUTO
        if value == <int>BpInputType.SYNDROME:
            return BpInputType.SYNDROME
        if value == <int>BpInputType.RECEIVED_VECTOR:
            return BpInputType.RECEIVED_VECTOR

    normalised = _normalise_enum_string(value, "bp_input_type")

    if normalised == "auto":
        return BpInputType.AUTO

    if normalised == "syndrome":
        return BpInputType.SYNDROME

    if normalised in ("received_vector", "received"):
        return BpInputType.RECEIVED_VECTOR

    raise ValueError(
        "bp_input_type must be one of "
        "'auto', 'syndrome', or 'received_vector'; "
        f"received {value!r}"
    )


cdef vector[double] _double_vector(object values):
    cdef vector[double] out
    cdef object value
    for value in values:
        out.push_back(float(value))
    return out


cdef vector[int] _int_vector(object values):
    cdef vector[int] out
    cdef object value
    for value in values:
        out.push_back(int(value))
    return out


cdef vector[vector[double]] _double_matrix(object rows):
    cdef vector[vector[double]] out
    cdef object row
    for row in rows:
        out.push_back(_double_vector(row))
    return out



cdef vector[size_t] _size_t_vector(object values) except *:
    cdef vector[size_t] out
    cdef object value
    cdef long long converted
    for value in values:
        converted = int(value)
        if converted < 0:
            raise ValueError("permutation indices must be non-negative")
        out.push_back(<size_t>converted)
    return out



cdef vector[Permutation] _permutation_vector(object specifications) except *:
    """Convert dict or (columns, rows) permutation specifications to C++."""
    cdef vector[Permutation] out
    cdef Permutation permutation
    cdef object specification, columns, rows

    for specification in specifications:
        if isinstance(specification, dict):
            if "old_col_for_new" not in specification:
                raise ValueError("each permutation dict requires old_col_for_new")
            if "old_row_for_new" not in specification:
                raise ValueError("each permutation dict requires old_row_for_new")
            columns = specification["old_col_for_new"]
            rows = specification["old_row_for_new"]
        else:
            try:
                columns, rows = specification
            except (TypeError, ValueError):
                raise TypeError(
                    "each permutation must be a dict or an "
                    "(old_col_for_new, old_row_for_new) pair"
                )

        if rows is None:
            raise ValueError("old_row_for_new must not be None")

        permutation = Permutation()
        permutation.old_col_for_new = _size_t_vector(columns)
        permutation.old_row_for_new = _size_t_vector(rows)
        out.push_back(permutation)
    return out


def _normalise_permutations(object specifications):
    """Return a stable, immutable Python snapshot of permutation inputs."""
    normalised = []
    for specification in specifications:
        if isinstance(specification, dict):
            if "old_col_for_new" not in specification:
                raise ValueError("each permutation dict requires old_col_for_new")
            if "old_row_for_new" not in specification:
                raise ValueError("each permutation dict requires old_row_for_new")
            columns = specification["old_col_for_new"]
            rows = specification["old_row_for_new"]
        else:
            columns, rows = specification
        if rows is None:
            raise ValueError("old_row_for_new must not be None")
        normalised.append((
            tuple([int(x) for x in columns]),
            tuple([int(x) for x in rows]),
        ))
    return tuple(normalised)


cdef unique_ptr[BpSparse] _copy_pcm(object pcm) except *:
    cdef Py_ssize_t rows, cols, i, j, k
    cdef object coo
    cdef unique_ptr[BpSparse] out

    if hasattr(pcm, "tocoo"):
        coo = pcm.tocoo()
        if len(coo.shape) != 2:
            raise ValueError("pcm must be two-dimensional")
        rows, cols = coo.shape
        out = unique_ptr[BpSparse](new BpSparse(<int>rows, <int>cols))
        for k in range(coo.nnz):
            if int(coo.data[k]) & 1:
                out.get().insert_entry(<int>coo.row[k], <int>coo.col[k])
        return move(out)

    if not hasattr(pcm, "shape") or len(pcm.shape) != 2:
        raise TypeError("pcm must be a 2-D array or expose tocoo()")
    rows, cols = pcm.shape
    out = unique_ptr[BpSparse](new BpSparse(<int>rows, <int>cols))
    for i in range(rows):
        for j in range(cols):
            if int(pcm[i, j]) & 1:
                out.get().insert_entry(<int>i, <int>j)
    return move(out)


cdef class AutBpDecoder:
    """Python owner for ``ldpc::autbp::AutBpDecoder``.

    Supply either ``max_automorphisms`` for BLISS discovery or ``permutations``
    for a predefined sequence. Each predefined item is a dict with
    ``old_col_for_new`` and ``old_row_for_new``, or a ``(cols, rows)``
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
        object bp_method="minimum_sum",
        object schedule="parallel",
        double min_sum_scaling_factor=1.0,
        int omp_threads=1,
        serial_schedule=(),
        int random_schedule_seed=0,
        bint random_serial_schedule=False,
        object bp_input_type="auto",
        int memory_seed=-1,
    ):
        cdef vector[double] c_priors
        cdef vector[int] c_serial_schedule
        cdef vector[Permutation] c_permutations
        cdef size_t c_max_automorphisms
        cdef DecoderFactory factory
        cdef optional[size_t] c_maximum_solutions
        cdef object permutation_snapshot = None

        cdef BpMethod c_bp_method
        cdef BpSchedule c_schedule
        cdef BpInputType c_bp_input_type

        c_bp_method = _bp_method_from_python(bp_method)
        c_schedule = _bp_schedule_from_python(schedule)
        c_bp_input_type = _bp_input_type_from_python(bp_input_type)

        # Materialise iterable inputs once so generators are handled correctly.
        priors = tuple(priors)
        gamma_dist_interval = tuple(gamma_dist_interval)
        memory_strengths_per_leg = tuple([tuple(row) for row in memory_strengths_per_leg])
        serial_schedule = tuple(serial_schedule)
        if permutations is not None:
            permutation_snapshot = _normalise_permutations(permutations)
            permutations = permutation_snapshot

        if decoder_type not in ("bp", "relay"):
            raise ValueError("decoder_type must be 'bp' or 'relay'")
        if permutations is None:
            if max_automorphisms is None or int(max_automorphisms) <= 0:
                raise ValueError(
                    "max_automorphisms must be positive when permutations is None"
                )
            c_max_automorphisms = <size_t>int(max_automorphisms)
        else:
            if max_automorphisms is not None:
                raise ValueError(
                    "provide permutations or max_automorphisms, not both"
                )
            c_permutations = _permutation_vector(permutations)
            if c_permutations.empty():
                raise ValueError("permutations must be non-empty")
        if maximum_solutions is not None:
            if int(maximum_solutions) <= 0:
                raise ValueError("maximum_solutions must be positive or None")
            c_maximum_solutions = optional[size_t](<size_t>int(maximum_solutions))

        self._pcm = move(_copy_pcm(pcm))
        c_priors = _double_vector(priors)
        if c_priors.size() != <size_t>self._pcm.get().n:
            raise ValueError("priors length must equal the number of PCM columns")

        c_serial_schedule = _int_vector(serial_schedule)
        if decoder_type == "bp":
            factory = make_bp_factory(
                maximum_iterations,
                <BpMethod>c_bp_method,
                <BpSchedule>c_schedule,
                min_sum_scaling_factor,
                omp_threads,
                c_serial_schedule,
                random_schedule_seed,
                <cpp_bool>random_serial_schedule,
                <BpInputType>c_bp_input_type,
            )
        else:
            factory = make_relay_bp_factory(
                maximum_legs,
                relay_maximum_solutions,
                iterations0,
                maximum_iterations,
                gamma0,
                _double_vector(gamma_dist_interval),
                _double_matrix(memory_strengths_per_leg),
                <BpMethod>c_bp_method,
                <BpSchedule>c_schedule,
                min_sum_scaling_factor,
                omp_threads,
                c_serial_schedule,
                random_schedule_seed,
                <cpp_bool>random_serial_schedule,
                <BpInputType>c_bp_input_type,
                memory_seed,
            )

        if permutations is None:
            self._decoder = unique_ptr[AutBpDecoderCpp](new AutBpDecoderCpp(
                self._pcm.get()[0], c_priors, factory, c_max_automorphisms,
                c_maximum_solutions, <cpp_bool>include_identity
            ))
        else:
            self._decoder = unique_ptr[AutBpDecoderCpp](new AutBpDecoderCpp(
                self._pcm.get()[0], c_priors, c_permutations, factory,
                c_maximum_solutions
            ))

        # Preserve immutable Python snapshots of every input except pcm.
        self._priors = tuple([float(x) for x in priors])
        self._permutations = permutation_snapshot
        self._max_automorphisms = max_automorphisms
        self._maximum_solutions = maximum_solutions
        self._include_identity = include_identity
        self._decoder_type = decoder_type
        self._maximum_legs = maximum_legs
        self._relay_maximum_solutions = relay_maximum_solutions
        self._iterations0 = iterations0
        self._maximum_iterations = maximum_iterations
        self._gamma0 = gamma0
        self._gamma_dist_interval = tuple([float(x) for x in gamma_dist_interval])
        self._memory_strengths_per_leg = tuple([
            tuple([float(x) for x in row]) for row in memory_strengths_per_leg
        ])
        self._min_sum_scaling_factor = min_sum_scaling_factor
        self._omp_threads = omp_threads
        self._serial_schedule = tuple([int(x) for x in serial_schedule])
        self._random_schedule_seed = random_schedule_seed
        self._random_serial_schedule = random_serial_schedule
        self._memory_seed = memory_seed

        if c_bp_method == BpMethod.MINIMUM_SUM:
            self._bp_method = "minimum_sum"
        elif c_bp_method == BpMethod.PRODUCT_SUM:
            self._bp_method = "product_sum"
        else:
            raise RuntimeError("unknown BpMethod value")

        if c_schedule == BpSchedule.PARALLEL:
            self._schedule = "parallel"
        elif c_schedule == BpSchedule.SERIAL:
            self._schedule = "serial"
        elif c_schedule == BpSchedule.SERIAL_RELATIVE:
            self._schedule = "serial_relative"
        else:
            raise RuntimeError("unknown BpSchedule value")

        if c_bp_input_type == BpInputType.AUTO:
            self._bp_input_type = "auto"
        elif c_bp_input_type == BpInputType.SYNDROME:
            self._bp_input_type = "syndrome"
        elif c_bp_input_type == BpInputType.RECEIVED_VECTOR:
            self._bp_input_type = "received_vector"
        else:
            raise RuntimeError("unknown BpInputType value")

    def decode(self, syndrome):
        cdef vector[uint8_t] c_syndrome
        cdef vector[uint8_t] result
        cdef object value
        for value in syndrome:
            c_syndrome.push_back(<uint8_t>(int(value) & 1))
        if c_syndrome.size() != <size_t>self._pcm.get().m:
            raise ValueError("syndrome length must equal check_count")
        result = self._decoder.get().decode(c_syndrome)
        return [int(result[i]) for i in range(result.size())]

    # PCM-derived dimensions (the matrix itself is intentionally not exposed).
    @property
    def check_count(self): return self._pcm.get().m
    @property
    def bit_count(self): return self._pcm.get().n

    # Constructor-input getters.
    @property
    def priors(self): return self._priors

    def get_permutations(self):
        """All permutations actually used by the C++ decoder.

        This includes BLISS-discovered permutations in automatic mode and the
        validated predefined permutations in explicit mode. A fresh immutable
        Python snapshot is produced on every access.
        """
        cdef const vector[Permutation]* values = &self._decoder.get().permutations()
        cdef const Permutation* permutation
        cdef const vector[size_t]* col_vec
        cdef const vector[size_t]* row_vec
        cdef size_t i, j, n_cols, n_rows
        cdef object columns
        cdef object rows
        cdef list result = []

        for i in range(values[0].size()):
            permutation = &values[0][i]

            col_vec = &permutation.old_col_for_new
            n_cols = col_vec.size()
            columns = tuple([int(col_vec[0][j]) for j in range(n_cols)])

            row_vec = &permutation.old_row_for_new
            n_rows = row_vec.size()
            rows = tuple([int(row_vec[0][j]) for j in range(n_rows)])

            result.append((columns, rows))
        return tuple(result)

    @property
    def last_member_stats(self):
        """Per-member ``(iterations, converged)`` results from the last decode.

        The entries follow the same order as :attr:`permutations`. A fresh
        immutable Python snapshot is produced on every access.
        """
        cdef const vector[MemberStats]* values = &self._decoder.get().last_member_stats()
        cdef size_t i
        return tuple([
            (int(values[0][i].iterations), bool(values[0][i].converged))
            for i in range(values[0].size())
        ])

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


    # Last-decode result getters.
    @property
    def solution_number(self): return self._decoder.get().solution_number
    @property
    def iterations(self): return self._decoder.get().iterations
    @property
    def decoding(self):
        cdef vector[uint8_t]* values = &self._decoder.get().decoding
        return [int(values[0][i]) for i in range(values[0].size())]
    @property
    def converge(self): return bool(self._decoder.get().converge)
    @property
    def log_prob_ratios(self):
        cdef vector[double]* values = &self._decoder.get().log_prob_ratios
        return [values[0][i] for i in range(values[0].size())]

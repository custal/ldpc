# distutils: language = c++
# cython: language_level=3

from libc.stdint cimport uint8_t
from libcpp cimport bool as cpp_bool
from libcpp.optional cimport optional
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from libc.stddef cimport size_t

cdef extern from "bp.hpp" namespace "ldpc::bp":
    cdef enum BpMethod:
        MINIMUM_SUM
        PRODUCT_SUM

    cdef enum BpSchedule:
        PARALLEL
        SERIAL
        SERIAL_RELATIVE

    cdef enum BpInputType:
        AUTO
        SYNDROME
        RECEIVED_VECTOR

cdef extern from "bp.hpp":
    cdef cppclass BpSparse "ldpc::bp::BpSparse":
        int m
        int n
        BpSparse(int rows, int cols) except +
        void insert_entry(int row, int col) except +

    cdef cppclass BpDecoder "ldpc::bp::BpDecoder":
        pass


cdef extern from "autbp.cpp" namespace "ldpc::autbp":
    cdef cppclass Permutation "ldpc::autbp::Permutation":
        Permutation() except +
        vector[size_t] old_col_for_new
        optional[vector[size_t]] old_row_for_new

    cdef cppclass MemberStats "ldpc::autbp::MemberStats":
        int iterations
        cpp_bool converged

    cdef cppclass DecoderFactory "ldpc::autbp::DecoderFactory":
        pass

    DecoderFactory make_bp_factory(
        int maximum_iterations,
        BpMethod bp_method,
        BpSchedule schedule,
        double min_sum_scaling_factor,
        int omp_threads,
        const vector[int]& serial_schedule,
        int random_schedule_seed,
        cpp_bool random_serial_schedule,
        BpInputType bp_input_type
    ) except +

    DecoderFactory make_relay_bp_factory(
        int maximum_legs,
        int maximum_solutions,
        int iterations0,
        int maximum_iterations,
        double gamma0,
        vector[double] gamma_dist_interval,
        vector[vector[double]] memory_strengths_per_leg,
        BpMethod bp_method,
        BpSchedule schedule,
        double min_sum_scaling_factor,
        int omp_threads,
        const vector[int]& serial_schedule,
        int random_schedule_seed,
        cpp_bool random_serial_schedule,
        BpInputType bp_input_type,
        int memory_seed
    ) except +

    cdef cppclass AutBpDecoderCpp "ldpc::autbp::AutBpDecoder":
        int iterations
        size_t solution_number
        vector[uint8_t] decoding
        cpp_bool converge
        vector[double] log_prob_ratios

        AutBpDecoderCpp(
            const BpSparse& base_pcm,
            vector[double] priors,
            vector[Permutation] permutations,
            DecoderFactory factory,
            optional[size_t] maximum_solutions
        ) except +

        AutBpDecoderCpp(
            const BpSparse& base_pcm,
            vector[double] priors,
            DecoderFactory factory,
            size_t max_automorphisms,
            optional[size_t] maximum_solutions,
            cpp_bool include_identity
        ) except +

        vector[uint8_t] decode(const vector[uint8_t]& syndrome) except +
        const vector[MemberStats]& last_member_stats() noexcept
        const vector[Permutation]& permutations() noexcept


cdef class AutBpDecoder:
    cdef unique_ptr[BpSparse] _pcm
    cdef unique_ptr[AutBpDecoderCpp] _decoder

    # Stored constructor inputs, used by read-only Python properties.
    cdef object _priors
    cdef object _permutations
    cdef object _max_automorphisms
    cdef object _maximum_solutions
    cdef bint _include_identity
    cdef str _decoder_type
    cdef int _maximum_legs
    cdef int _relay_maximum_solutions
    cdef int _iterations0
    cdef int _maximum_iterations
    cdef double _gamma0
    cdef object _gamma_dist_interval
    cdef object _memory_strengths_per_leg
    cdef int _bp_method
    cdef int _schedule
    cdef double _min_sum_scaling_factor
    cdef int _omp_threads
    cdef object _serial_schedule
    cdef int _random_schedule_seed
    cdef bint _random_serial_schedule
    cdef int _bp_input_type
    cdef int _memory_seed

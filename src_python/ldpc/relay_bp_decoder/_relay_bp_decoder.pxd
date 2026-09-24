#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
from ldpc.bp_decoder._bp_decoder cimport BpSparse, NULL_INT_VECTOR, BpMethod, BpInputType, BpSchedule, BpEntry, BpSparse, BpDecoderCpp
ctypedef np.uint8_t uint8_t


# Free functions / constants describing the precision tiers that relay_bp.hpp was
# compiled with. Declared with explicit C names so they are unambiguous.
cdef extern from "relay_bp.hpp":
    vector[int] cpp_available_precisions "ldpc::relay::available_precisions" ()
    int cpp_resolve_precision "ldpc::relay::resolve_precision" (int requested) except +
    int CPP_DEFAULT_PRECISION "ldpc::relay::DEFAULT_PRECISION"


cdef extern from "relay_bp.hpp" namespace "ldpc::relay":
    cdef cppclass RelayBpDecoderCpp "ldpc::relay::RelayBpDecoder" (BpDecoderCpp):
            RelayBpDecoderCpp(
                BpSparse& parity_check_matrix,
                vector[double] channel_probabilities,
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
                vector[int] serial_schedule,
                int random_schedule_seed,
                bool random_serial_schedule,
                BpInputType bp_input_type,
                int memory_seed,
                int precision,
                bool debug) except +
            int maximum_legs
            int maximum_solutions
            vector[vector[double]] memory_strengths_per_leg
            int iterations0
            double gamma0
            vector[double] gamma_dist_interval
            int memory_seed
            void set_memory_seed(int seed)

            # Mantissa bits used to carry the messages. `precision` is the tier
            # actually in use, `precision_request` is what the caller asked for.
            int precision
            int precision_request
            void set_precision(int requested_precision) except +

            int solution_number
            int iterations

            # Debugging: per-iteration snapshots of the variable node llrs, filled
            # only while `debug` is true and cleared at the start of every decode.
            bool debug
            vector[vector[double]] llr_history
            vector[int] llr_history_leg
            vector[int] llr_history_iteration
            void clear_llr_history()

cdef class RelayBpDecoderBase:
    cdef BpSparse *pcm
    cdef int m, n
    cdef vector[uint8_t] _syndrome
    cdef vector[double] _error_channel
    cdef vector[int] _serial_schedule_order
    cdef vector[vector[double]] _memory_strengths_per_leg
    cdef vector[double] _gamma_dist_interval
    cdef bool MEMORY_ALLOCATED
    cdef RelayBpDecoderCpp *bpd
    cdef str user_dtype

cdef class RelayBpDecoder(RelayBpDecoderBase):
    cdef vector[uint8_t] _received_vector
    pass

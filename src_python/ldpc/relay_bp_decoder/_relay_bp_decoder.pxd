#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
from ldpc.bp_decoder._bp_decoder cimport BpSparse, NULL_INT_VECTOR, BpMethod, BpInputType, BpSchedule, BpEntry, BpSparse, BpDecoderCpp
ctypedef np.uint8_t uint8_t


cdef extern from "relay_bp.hpp" namespace "ldpc::relay":
    cdef cppclass RelayBpDecoderCpp "ldpc::relay::RelayBpDecoder" (BpDecoderCpp):
            RelayBpDecoderCpp(
                BpSparse& parity_check_matrix,
                vector[double] channel_probabilities,
                int maximum_legs,
                int maximum_solutions,
                int maximum_iterations,
                int iterations0,
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
                int memory_seed) except +
            int maximum_legs
            int maximum_solutions
            vector[vector[double]] memory_strengths_per_leg
            int iterations0
            double gamma0
            vector[double] gamma_dist_interval
            int memory_seed
            void set_memory_seed(int seed)

            int solution_number
            int total_iterations

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
    cdef int random_schedule_seed

cdef class RelayBpDecoder(RelayBpDecoderBase):
    cdef vector[uint8_t] _received_vector
    pass

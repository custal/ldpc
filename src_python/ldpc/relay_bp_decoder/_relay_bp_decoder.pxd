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
                vector[int] maximum_iterations_per_leg,
                vector[vector[double]] memory_strengths_per_leg,
                int maximum_iterations,
                BpMethod bp_method,
                BpSchedule schedule,
                double min_sum_scaling_factor,
                int omp_threads,
                vector[int] serial_schedule,
                int random_schedule_seed,
                bool random_serial_schedule,
                BpInputType bp_input_type) except +
            int maximum_legs
            int maximum_solutions
            vector[int] maximum_iterations_per_leg
            vector[vector[double]] memory_strengths_per_leg

            vector[vector[uint8_t]] decoding_per_leg
            vector[vector[double]] log_prob_ratios_per_leg
            vector[int] iterations_per_leg
            vector[bool] convergence_per_leg
            int solution_number

cdef class RelayBpDecoderBase:
    cdef BpSparse *pcm
    cdef int m, n
    cdef vector[uint8_t] _syndrome
    cdef vector[double] _error_channel
    cdef vector[int] _serial_schedule_order
    cdef vector[int] _maximum_iterations_per_leg
    cdef vector[vector[double]] _memory_strengths_per_leg
    cdef bool MEMORY_ALLOCATED
    cdef RelayBpDecoderCpp *bpd
    cdef str user_dtype
    # cdef int random_schedule_seed

cdef class RelayBpDecoder(RelayBpDecoderBase):
    cdef vector[uint8_t] _received_vector
    pass

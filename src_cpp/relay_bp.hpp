#ifndef RELAY_H
#define RELAY_H

#include <utility>
#include <vector>
#include <memory>
#include <iterator>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>
#include <stdexcept> // required for std::runtime_error
#include <set>
#include <algorithm>

#include "math.h"
#include "sparse_matrix_base.hpp"
#include "gf2sparse.hpp"
#include "rng.hpp"
#include "bp.hpp"

namespace ldpc::relay {

    using BpSparse = ldpc::bp::BpSparse;
    using BpMethod = ldpc::bp::BpMethod;
    using BpSchedule = ldpc::bp::BpSchedule;
    using BpInputType = ldpc::bp::BpInputType;
    using BpDecoder = ldpc::bp::BpDecoder;

    class RelayBpDecoder : public BpDecoder {
        // TODO This class is effectively a duplicate of BpDecoder but with an altered message passing algorithm which
        // accepts memory and legs. We should consider refactoring BpDecoder so it accepts memory as an input in order
        // to implement DMem-BP and then create an independent class which chains these together to make the relay legs.
        // I don't want to edit the original files however so this is a work around
    public:
        int maximum_legs;
        int maximum_solutions;
        std::vector<int> maximum_iterations_per_leg;
        std::vector<std::vector<double>> memory_strengths_per_leg;

        std::vector<std::vector<uint8_t>> decoding_per_leg;
        std::vector<std::vector<double>> log_prob_ratios_per_leg;
        std::vector<int> iterations_per_leg;
        std::vector<bool> convergence_per_leg;
        int solution_number;

        RelayBpDecoder(
                BpSparse &parity_check_matrix,
                std::vector<double> channel_probabilities,
                int maximum_legs,
                int maximum_solutions,
                std::vector<int> maximum_iterations_per_leg,
                std::vector<std::vector<double>> memory_strengths_per_leg,
                int maximum_iterations = 0, //Redundant for Relay-BP as relevant information is in maximum_iterations_per_leg
                BpMethod bp_method = ldpc::bp::PRODUCT_SUM,
                BpSchedule schedule = ldpc::bp::PARALLEL,
                double min_sum_scaling_factor = 1.0,
                int omp_threads = 1,
                const std::vector<int> &serial_schedule = ldpc::bp::NULL_INT_VECTOR,
                int random_schedule_seed = 0,
                bool random_serial_schedule = false,
                BpInputType bp_input_type = ldpc::bp::AUTO
                ) :
                BpDecoder(
                    parity_check_matrix,
                      std::move(channel_probabilities),
                      maximum_iterations,
                      bp_method,
                      schedule,
                      min_sum_scaling_factor,
                      omp_threads,
                      serial_schedule,
                      random_schedule_seed,
                      random_serial_schedule,
                      bp_input_type), maximum_legs(maximum_legs), maximum_solutions(maximum_solutions),
                        maximum_iterations_per_leg(std::move(maximum_iterations_per_leg)),
                        memory_strengths_per_leg(std::move(memory_strengths_per_leg))
        {
            this->iterations_per_leg.resize(maximum_legs);
            this->convergence_per_leg.resize(maximum_legs);
            this->solution_number = 0;
            this->decoding_per_leg.resize(maximum_legs);
            this->log_prob_ratios_per_leg.resize(maximum_legs);

            for (int leg = 0; leg < maximum_legs; leg++) {
                this->decoding_per_leg[leg] = std::vector<uint8_t>(this->bit_count);
                this->log_prob_ratios_per_leg[leg] = std::vector<double>(this->bit_count);
            }


            if (this->memory_strengths_per_leg.size() != this->maximum_legs
                || this->maximum_iterations_per_leg.size() != this->maximum_legs) {
                throw std::runtime_error("memory_strengths_per_leg and maximum_iterations_per_leg must be the same size "
                                         "as maximum_legs");
            }
            for (int leg = 0; leg < maximum_legs; leg++) {
                if (this->memory_strengths_per_leg[leg].size() != this->bit_count) {
                    throw std::runtime_error("Each memory_strengths_per_leg entry must have length equal to bit_count");
                }
            }

            //Initialise OMP thread pool
            // this->omp_thread_count = omp_threads;
            // this->set_omp_thread_count(this->omp_thread_count);
        }

        ~RelayBpDecoder() = default;

        void initialise_log_domain_bp_relay(int leg) {
            // initialise BP
            for (int i = 0; i < this->bit_count; i++) {
                if (leg == 0) {
                    this->initial_log_prob_ratios[i] = std::log(
                        (1 - this->channel_probabilities[i]) / this->channel_probabilities[i]);
                } else { //Carry over log_prob_ratios from previous leg
                    this->initial_log_prob_ratios[i] = this->log_prob_ratios[i];
                }

                for (auto &e: this->pcm.iterate_column(i)) {
                    e.bit_to_check_msg = this->initial_log_prob_ratios[i];
                }
            }
        }

        double decoding_weight(const std::vector<uint8_t> &decoding) {
            double weight = 0;
            for (int i = 0; i < this->bit_count; i++) {
                weight +=
                    decoding[i] * std::log((1 - this->channel_probabilities[i]) / this->channel_probabilities[i]);
            }
            return weight;
        }


        std::vector<uint8_t> &bp_decode_parallel(std::vector<uint8_t> &syndrome) override {
            //Reset outputs from previous run
            std::fill(this->iterations_per_leg.begin(), this->iterations_per_leg.end(), 0);
            std::fill(this->convergence_per_leg.begin(), this->convergence_per_leg.end(), false);
            this->solution_number = 0;
            for (int leg = 0; leg < maximum_legs; leg++) {
                std::fill(this->decoding_per_leg[leg].begin(), this->decoding_per_leg[leg].end(), 0);
                std::fill(this->log_prob_ratios_per_leg[leg].begin(), this->log_prob_ratios_per_leg[leg].end(), 0);
            }

            for (int leg = 0; leg < this->maximum_legs; leg++) {
                this->iterations = 0;
                this->converge = 0;

                if (this->solution_number == this->maximum_solutions) {
                    break;
                }

                this->initialise_log_domain_bp_relay(leg);
                int maximum_iterations = this->maximum_iterations_per_leg[leg];
                std::vector<double> memory_strengths = this->memory_strengths_per_leg[leg];

                //main interation loop
                for (int it = 1; it <= maximum_iterations; it++) {

                    if (this->bp_method == ldpc::bp::PRODUCT_SUM) {
                        for (int i = 0; i < this->check_count; i++) {
                            this->candidate_syndrome[i] = 0;

                            double temp = 1.0;
                            for (auto &e: this->pcm.iterate_row(i)) {
                                e.check_to_bit_msg = temp;
                                temp *= std::tanh(e.bit_to_check_msg / 2);
                            }

                            temp = 1;
                            for (auto &e: this->pcm.reverse_iterate_row(i)) {
                                e.check_to_bit_msg *= temp;
                                int message_sign = syndrome[i] != 0u ? -1.0 : 1.0;
                                e.check_to_bit_msg =
                                        message_sign * std::log((1 + e.check_to_bit_msg) / (1 - e.check_to_bit_msg));
                                temp *= std::tanh(e.bit_to_check_msg / 2);
                            }
                        }
                    } else if (this->bp_method == ldpc::bp::MINIMUM_SUM) {

                        double alpha;
                        if(this->ms_scaling_factor == 0.0) {
                            alpha = 1.0 - std::pow(2.0, -1.0*it);
                        }
                        else {
                            alpha = this->ms_scaling_factor;
                        }

                        //check to bit updates
                        for (int i = 0; i < check_count; i++) {

                            this->candidate_syndrome[i] = 0;
                            int total_sgn = 0;
                            int sgn = 0;
                            total_sgn = syndrome[i];
                            double temp = std::numeric_limits<double>::max();

                            for (auto &e: this->pcm.iterate_row(i)) {
                                if (e.bit_to_check_msg <= 0) {
                                    total_sgn += 1;
                                }
                                e.check_to_bit_msg = temp;
                                double abs_bit_to_check_msg = std::abs(e.bit_to_check_msg);
                                if (abs_bit_to_check_msg < temp) {
                                    temp = abs_bit_to_check_msg;
                                }
                            }

                            temp = std::numeric_limits<double>::max();
                            for (auto &e: this->pcm.reverse_iterate_row(i)) {
                                sgn = total_sgn;
                                if (e.bit_to_check_msg <= 0) {
                                    sgn += 1;
                                }
                                if (temp < e.check_to_bit_msg) {
                                    e.check_to_bit_msg = temp;
                                }

                                int message_sign = (sgn % 2 == 0) ? 1.0 : -1.0;

                                e.check_to_bit_msg *= message_sign * alpha;


                                double abs_bit_to_check_msg = std::abs(e.bit_to_check_msg);
                                if (abs_bit_to_check_msg < temp) {
                                    temp = abs_bit_to_check_msg;
                                }

                            }

                        }
                    }


                    //compute log probability ratios
                    for (int i = 0; i < this->bit_count; i++) {

                        double temp; //Implement DMem-BP
                        if (it  == 1) {
                            temp = this->initial_log_prob_ratios[i];
                        } else {
                            temp = (1 - memory_strengths[i]) * this->initial_log_prob_ratios[i] +
                                memory_strengths[i] * this->log_prob_ratios[i];
                        }
                        for (auto &e: this->pcm.iterate_column(i)) {
                            e.bit_to_check_msg = temp;
                            temp += e.check_to_bit_msg;
                            // if(isnan(temp)) temp = e.bit_to_check_msg;


                        }

                        //make hard decision on basis of log probability ratio for bit i
                        this->log_prob_ratios[i] = temp;
                        // if(isnan(log_prob_ratios[i])) log_prob_ratios[i] = initial_log_prob_ratios[i];
                        if (temp <= 0) {
                            this->decoding[i] = 1;
                            for (auto &e: this->pcm.iterate_column(i)) {
                                this->candidate_syndrome[e.row_index] ^= 1;
                            }
                        } else {
                            this->decoding[i] = 0;
                        }
                    }

                    if (std::equal(candidate_syndrome.begin(), candidate_syndrome.end(), syndrome.begin())) {
                        this->converge = true;
                    }

                    this->iterations = it;

                    if (this->converge) {
                        break;
                    }

                    //compute bit to check update
                    for (int i = 0; i < bit_count; i++) {
                        double temp = 0;
                        for (auto &e: this->pcm.reverse_iterate_column(i)) {
                            e.bit_to_check_msg += temp;
                            temp += e.check_to_bit_msg;
                        }
                    }
                }

                this->decoding_per_leg[leg] = this->decoding;
                this->log_prob_ratios_per_leg[leg] = this->log_prob_ratios;
                this->iterations_per_leg[leg] = this->iterations;
                this->convergence_per_leg[leg] = this->converge;
                this->solution_number += 1;
            }
            //If no solutions found return the best effort (final) decoding
            if (this->solution_number == 0) {
                this->decoding = this->decoding_per_leg[this->maximum_legs-1];
            }
            //Find best decoding (lowest weight) result and return
            double temp = std::numeric_limits<double>::max();
            for (int leg = 0; leg < this->maximum_legs; leg++) {
                if (!this->convergence_per_leg[leg]) {
                    continue;
                }
                double weight = this->decoding_weight(this->decoding_per_leg[leg]);
                if (weight < temp) {
                    temp = weight;
                    this->decoding = this->decoding_per_leg[leg];
                }
            }
            return this->decoding;
        }

        std::vector<uint8_t> &bp_decode_single_scan(std::vector<uint8_t> &syndrome) override  {
            throw std::logic_error("Not implemented");
        }

        std::vector<uint8_t> &bp_decode_serial(std::vector<uint8_t> &syndrome) override {
            throw std::logic_error("Not implemented");
        }

        std::vector<uint8_t> &
        soft_info_decode_serial(std::vector<double> &soft_info_syndrome, double cutoff, double sigma) override {
            throw std::logic_error("Not implemented");
        }
    };
    }
// namespace ldpc::relay

#endif
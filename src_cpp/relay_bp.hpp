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

        // Leg 0 runs for `iterations0` iterations (paired with memory strength gamma0).
        // Every subsequent leg runs for the inherited `maximum_iterations` iterations
        // (paired with the randomly-drawn per-leg memory strengths).
        int iterations0;

        // Memory strengths can either be given explicitly via `memory_strengths_per_leg`
        // (a (maximum_legs x bit_count) array), or auto-generated on the fly each leg:
        //   - leg 0 uses a constant memory strength `gamma0` for every bit
        //   - legs > 0 draw a memory strength per bit uniformly at random from
        //     [gamma_dist_interval[0], gamma_dist_interval[1]]
        // If memory_strengths_per_leg is non-empty it always takes precedence over
        // gamma0/gamma_dist_interval, which are then ignored.
        double gamma0;
        std::vector<double> gamma_dist_interval;
        std::vector<std::vector<double>> memory_strengths_per_leg;
        std::mt19937 memory_rng;

        // Only the best decoding (lowest weight, converged) found across all legs is kept,
        // rather than storing every leg's decoding/log_prob_ratios/iterations/convergence.
        // The winning result is written into the inherited `decoding` / `log_prob_ratios`
        // members at the end of bp_decode_parallel.
        int solution_number;

        int memory_seed;

        RelayBpDecoder(
                BpSparse &parity_check_matrix,
                std::vector<double> channel_probabilities,
                int maximum_legs,
                int maximum_solutions,
                int iterations0, //Number of BP iterations run on leg 0 (paired with gamma0)
                int maximum_iterations, //Number of BP iterations run on every leg after the first (paired with gamma_dist_interval)
                double gamma0 = 0.0, //Ignored if memory_strengths_per_leg is provided explicitly
                std::vector<double> gamma_dist_interval = {}, //Ignored if memory_strengths_per_leg is provided explicitly
                std::vector<std::vector<double>> memory_strengths_per_leg = {}, //Optional explicit (maximum_legs x bit_count) override; if given, takes precedence over gamma0/gamma_dist_interval
                BpMethod bp_method = ldpc::bp::PRODUCT_SUM,
                BpSchedule schedule = ldpc::bp::PARALLEL,
                double min_sum_scaling_factor = 1.0,
                int omp_threads = 1,
                const std::vector<int> &serial_schedule = ldpc::bp::NULL_INT_VECTOR,
                int random_schedule_seed = 0,
                bool random_serial_schedule = false,
                BpInputType bp_input_type = ldpc::bp::AUTO,
                int memory_seed = -1 // seed for the per-leg memory strength RNG; -1 -> seed non-deterministically
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
                        iterations0(iterations0),
                        gamma0(gamma0),
                        gamma_dist_interval(std::move(gamma_dist_interval)),
                        memory_strengths_per_leg(std::move(memory_strengths_per_leg)), memory_seed(memory_seed)
        {
            this->solution_number = 0;
            this->iterations = 0;

            if (!this->memory_strengths_per_leg.empty()) {
                if (this->memory_strengths_per_leg.size() != static_cast<size_t>(this->maximum_legs)) {
                    throw std::runtime_error("memory_strengths_per_leg must have exactly maximum_legs entries");
                }
                for (int leg = 0; leg < this->maximum_legs; leg++) {
                    if (this->memory_strengths_per_leg[leg].size() != static_cast<size_t>(this->bit_count)) {
                        throw std::runtime_error("Each memory_strengths_per_leg entry must have length equal to bit_count");
                    }
                }
            } else if (this->gamma_dist_interval.size() != 2) {
                throw std::runtime_error("Either provide memory_strengths_per_leg explicitly, or provide gamma0 "
                                         "together with a 2-element gamma_dist_interval so memory strengths can "
                                         "be auto-generated.");
            }

            if (this->memory_seed >= 0) {
                this->memory_rng.seed(static_cast<unsigned int>(this->memory_seed));
            } else {
                std::random_device rd;
                this->memory_rng.seed(rd());
            }

            //Initialise OMP thread pool
            // this->omp_thread_count = omp_threads;
            // this->set_omp_thread_count(this->omp_thread_count);
        }
        
        void set_memory_seed(int seed) {
            this->memory_seed = seed;
            if (this->memory_seed >= 0) {
                this->memory_rng.seed(static_cast<unsigned int>(this->memory_seed));
            } else {
                std::random_device rd;
                this->memory_rng.seed(rd());
            }
        }

        // Returns the memory strength vector (length bit_count) for a given leg.
        // If an explicit memory_strengths_per_leg override was provided at construction,
        // that is returned directly. Otherwise strengths are auto-generated: leg 0 always
        // uses the constant `gamma0` for every bit, and legs > 0 draw a fresh random value
        // per bit, uniformly from [gamma_dist_interval[0], gamma_dist_interval[1]].
        std::vector<double> generate_memory_strengths_for_leg(int leg) {
            if (!this->memory_strengths_per_leg.empty()) {
                return this->memory_strengths_per_leg[leg];
            }

            std::vector<double> memory_strengths(this->bit_count);
            if (leg == 0) {
                std::fill(memory_strengths.begin(), memory_strengths.end(), this->gamma0);
            } else {
                std::uniform_real_distribution<double> dist(this->gamma_dist_interval[0],
                                                              this->gamma_dist_interval[1]);
                for (int i = 0; i < this->bit_count; i++) {
                    memory_strengths[i] = dist(this->memory_rng);
                }
            }
            return memory_strengths;
        }

        ~RelayBpDecoder() = default;

        void initialise_log_domain_bp_relay(int leg) {
            // initialise BP
            for (int i = 0; i < this->bit_count; i++) {
                if (leg == 0) {
                    this->initial_log_prob_ratios[i] = std::log(
                        (1 - this->channel_probabilities[i]) / this->channel_probabilities[i]);
                    this->log_prob_ratios[i] = this->initial_log_prob_ratios[i];
                }
                // For leg > 0, initial_log_prob_ratios[i] is intentionally left as it was set on
                // leg 0 (no carry-over of the previous leg's a-posteriori log_prob_ratios).

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
            std::fill(this->decoding.begin(), this->decoding.end(), 0);
            std::fill(this->log_prob_ratios.begin(), this->log_prob_ratios.end(), 0);
            this->solution_number = 0;
            this->iterations = 0;

            // Tracks the best (lowest-weight, converged) solution seen so far across legs,
            // without needing to keep every leg's decoding/log_prob_ratios around.
            bool any_converged = false;
            double best_weight = std::numeric_limits<double>::max();
            std::vector<uint8_t> best_decoding(this->bit_count, 0);
            std::vector<double> best_log_prob_ratios(this->bit_count, 0.0);

            for (int leg = 0; leg < this->maximum_legs; leg++) {
                int leg_iterations = 0;
                this->converge = 0;

                this->initialise_log_domain_bp_relay(leg);
                int leg_max_iterations = (leg == 0) ? this->iterations0 : this->maximum_iterations;
                std::vector<double> memory_strengths = this->generate_memory_strengths_for_leg(leg);

                //main interation loop
                for (int it = 1; it <= leg_max_iterations; it++) {

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
                                //Change from <= to < to be consistent with ibm implementation. This is of no
                                //consequence but I did this to ensure consistency on a shot by shot basis during debugging
                                if (e.bit_to_check_msg < 0) {
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
                                //Change from <= to < to be consistent with ibm implementation. This is of no
                                //consequence but I did this to ensure consistency on a shot by shot basis during debugging
                                if (e.bit_to_check_msg < 0) {
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
                        temp = (1 - memory_strengths[i]) * this->initial_log_prob_ratios[i] +
                            memory_strengths[i] * this->log_prob_ratios[i];
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

                    leg_iterations = it;

                    if (this->converge) {
                        this->solution_number += 1;
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

                this->iterations += leg_iterations;

                if (this->converge) {
                    double weight = this->decoding_weight(this->decoding);
                    if (!any_converged || weight < best_weight) {
                        best_weight = weight;
                        best_decoding = this->decoding;
                        best_log_prob_ratios = this->log_prob_ratios;
                        any_converged = true;
                    }
                }

                if (this->solution_number == this->maximum_solutions) {
                    break;
                }
            }

            if (any_converged) {
                //Return the lowest-weight converged solution found across all legs
                this->decoding = best_decoding;
                this->log_prob_ratios = best_log_prob_ratios;
            }
            //If no leg converged, this->decoding / this->log_prob_ratios already hold the
            //best-effort result from the final leg that was run.
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
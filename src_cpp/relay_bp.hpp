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
#include <string>

#include "math.h"
#include "sparse_matrix_base.hpp"
#include "gf2sparse.hpp"
#include "rng.hpp"
#include "bp.hpp"

// Arbitrary precision for the message passing. The working type is a template
// parameter, so the available precisions are a compile-time menu given in
// mantissa bits: 24 -> float, 53 -> double (the default, and the precision this
// decoder used before the option existed), anything larger -> Boost's
// header-only cpp_bin_float. A requested precision is rounded up to the smallest
// tier that is at least as precise. Redefine RELAY_BP_PRECISION_TIERS before
// including this header to add or remove tiers; each one instantiates the
// decoder again, so extra tiers cost compile time.
//
// Boost.Multiprecision is a hard requirement. There is deliberately no
// double-only fallback: a build that cannot offer the precisions it was asked
// for should fail here, loudly, rather than compile and then behave differently
// from an identical build elsewhere.
#if defined(__has_include)
#  if !__has_include(<boost/multiprecision/cpp_bin_float.hpp>)
#    error "relay_bp.hpp requires the Boost.Multiprecision headers, which were not found on the include path. Boost.Multiprecision is header-only, so no library needs linking; add the Boost include directory to this target (for example find_package(Boost) plus target_include_directories(<target> PRIVATE ${Boost_INCLUDE_DIRS}), and the same include directory in setup.py for the Python extension)."
#  endif
#endif

#include <boost/multiprecision/cpp_bin_float.hpp>

#ifndef RELAY_BP_PRECISION_TIERS
#define RELAY_BP_PRECISION_TIERS \
    RELAY_BP_TIER(11) RELAY_BP_TIER(24) RELAY_BP_TIER(53) RELAY_BP_TIER(64) \
    RELAY_BP_TIER(113) RELAY_BP_TIER(128) RELAY_BP_TIER(192) RELAY_BP_TIER(256) \
    RELAY_BP_TIER(384) RELAY_BP_TIER(512) RELAY_BP_TIER(1024) RELAY_BP_TIER(2048) \
    RELAY_BP_TIER(4096)
#endif

namespace ldpc::relay {

    using BpSparse = ldpc::bp::BpSparse;
    using BpEntry = ldpc::bp::BpEntry;
    using BpMethod = ldpc::bp::BpMethod;
    using BpSchedule = ldpc::bp::BpSchedule;
    using BpInputType = ldpc::bp::BpInputType;
    using BpDecoder = ldpc::bp::BpDecoder;

    // Mantissa bits of IEEE double: the precision used when none is requested.
    constexpr int DEFAULT_PRECISION = 53;

    // A tier list that omits the default would quietly move every decoder that
    // did not ask for a precision onto a different one, so refuse to build.
    constexpr bool precision_tiers_include_default() {
        bool found = false;
#define RELAY_BP_TIER(Bits) found = found || (Bits == DEFAULT_PRECISION);
        RELAY_BP_PRECISION_TIERS
#undef RELAY_BP_TIER
        return found;
    }

    static_assert(precision_tiers_include_default(),
                  "RELAY_BP_PRECISION_TIERS must include the default precision "
                  "of 53 mantissa bits (IEEE double).");

    // Maps a tier, in mantissa bits, onto the type the messages are carried in.
    template<unsigned Bits>
    struct PrecisionTraits {
        using type = boost::multiprecision::number<
                boost::multiprecision::cpp_bin_float<Bits, boost::multiprecision::digit_base_2>,
                boost::multiprecision::et_off>;
    };

    template<> struct PrecisionTraits<24> { using type = float; };
    template<> struct PrecisionTraits<53> { using type = double; };

    // The precisions this translation unit was compiled with, ascending.
    inline const std::vector<int> &available_precisions() {
        static const std::vector<int> tiers = {
#define RELAY_BP_TIER(Bits) Bits,
                RELAY_BP_PRECISION_TIERS
#undef RELAY_BP_TIER
        };
        return tiers;
    }

    // Rounds a requested precision up to the smallest tier at least as precise.
    inline int resolve_precision(int requested) {
        if (requested <= 0) {
            throw std::runtime_error("'precision' must be a positive number of mantissa bits "
                                     "(53 for the default double precision).");
        }
        for (int tier: available_precisions()) {
            if (tier >= requested) return tier;
        }
        throw std::runtime_error(
                "The requested precision of " + std::to_string(requested) + " mantissa bits exceeds the largest "
                "compiled-in tier (" + std::to_string(available_precisions().back()) + " bits). Define "
                "RELAY_BP_PRECISION_TIERS with the tier you need before including relay_bp.hpp and recompile.");
    }

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

        // Mantissa bits used for the messages. `precision` is the tier in use,
        // i.e. `precision_request` rounded up; see available_precisions().
        int precision;
        int precision_request;

        // The BpSparse entries only hold double messages, so above double the
        // messages live in flat arrays instead. Every non-zero is given an index
        // and these tables record the orders iterate_row/iterate_column visit them
        // in, so the message passing itself is unchanged.
        int edge_count;
        std::vector<std::vector<int>> row_edges; // check -> edge ids, in iterate_row order
        std::vector<std::vector<int>> col_edges; // bit   -> edge ids, in iterate_column order
        std::vector<int> edge_row;               // edge id -> check index
        std::vector<BpEntry *> edge_entry;       // edge id -> the entry it mirrors

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
                int memory_seed = -1, // seed for the per-leg memory strength RNG; -1 -> seed non-deterministically
                int precision = DEFAULT_PRECISION // mantissa bits used for the messages; 53 == double == previous behaviour
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

            this->precision_request = precision;
            this->precision = resolve_precision(precision);

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

            this->build_edge_index();

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

        // Rounded up to a tier exactly as at construction; applies from the next
        // decode. Resolved first so a rejected request leaves the decoder unchanged.
        void set_precision(int requested_precision) {
            int resolved = resolve_precision(requested_precision);
            this->precision_request = requested_precision;
            this->precision = resolved;
        }

        // Indexes the non-zeros of the parity check matrix. Called once from the
        // constructor; call again if the sparsity pattern is ever changed. The edge
        // id is stashed in the entry's own message field during the row pass so the
        // column pass can recover it without a side table; it is cleared afterwards.
        void build_edge_index() {
            this->row_edges.assign(this->check_count, std::vector<int>());
            this->col_edges.assign(this->bit_count, std::vector<int>());
            this->edge_row.clear();
            this->edge_entry.clear();

            this->edge_count = 0;
            for (int i = 0; i < this->check_count; i++) {
                for (auto &e: this->pcm.iterate_row(i)) {
                    e.bit_to_check_msg = static_cast<double>(this->edge_count);
                    this->row_edges[i].push_back(this->edge_count);
                    this->edge_row.push_back(i);
                    this->edge_entry.push_back(&e);
                    this->edge_count++;
                }
            }

            for (int i = 0; i < this->bit_count; i++) {
                for (auto &e: this->pcm.iterate_column(i)) {
                    this->col_edges[i].push_back(static_cast<int>(e.bit_to_check_msg));
                    e.bit_to_check_msg = 0.0;
                }
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

        template<typename T>
        void initialise_log_domain_bp_relay(int leg, std::vector<T> &initial_llr, std::vector<T> &llr,
                                            std::vector<T> &bit_to_check) {
            using std::log;
            // initialise BP
            for (int i = 0; i < this->bit_count; i++) {
                if (leg == 0) {
                    T channel_probability = static_cast<T>(this->channel_probabilities[i]);
                    initial_llr[i] = log((T(1) - channel_probability) / channel_probability);
                    llr[i] = initial_llr[i];
                }
                // For leg > 0, initial_llr[i] is intentionally left as it was set on
                // leg 0 (no carry-over of the previous leg's a-posteriori log_prob_ratios).

                for (int e: this->col_edges[i]) {
                    bit_to_check[e] = initial_llr[i];
                }
            }
        }

        template<typename T = double>
        T decoding_weight(const std::vector<uint8_t> &decoding) {
            using std::log;
            T weight = T(0);
            for (int i = 0; i < this->bit_count; i++) {
                T channel_probability = static_cast<T>(this->channel_probabilities[i]);
                weight +=
                    static_cast<T>(decoding[i]) * log((T(1) - channel_probability) / channel_probability);
            }
            return weight;
        }

        // Narrows the working-precision state back into the double members inherited
        // from BpDecoder, so everything downstream sees exactly what it always did.
        template<typename T>
        void write_back_to_base(const std::vector<T> &initial_llr, const std::vector<T> &llr,
                                const std::vector<T> &bit_to_check, const std::vector<T> &check_to_bit) {
            for (int i = 0; i < this->bit_count; i++) {
                this->initial_log_prob_ratios[i] = static_cast<double>(initial_llr[i]);
                this->log_prob_ratios[i] = static_cast<double>(llr[i]);
            }
            for (int e = 0; e < this->edge_count; e++) {
                this->edge_entry[e]->bit_to_check_msg = static_cast<double>(bit_to_check[e]);
                this->edge_entry[e]->check_to_bit_msg = static_cast<double>(check_to_bit[e]);
            }
        }

        template<typename T>
        std::vector<uint8_t> &bp_decode_parallel_impl(std::vector<uint8_t> &syndrome) {
            using std::log; using std::tanh; using std::abs; using std::pow;

            //Reset outputs from previous run
            std::fill(this->decoding.begin(), this->decoding.end(), 0);
            std::fill(this->log_prob_ratios.begin(), this->log_prob_ratios.end(), 0);
            this->solution_number = 0;
            this->iterations = 0;

            // Working-precision state, persisting across legs: llr in particular is
            // deliberately carried from one leg into the next.
            std::vector<T> initial_llr(this->bit_count, T(0));
            std::vector<T> llr(this->bit_count, T(0));
            std::vector<T> bit_to_check(this->edge_count, T(0));
            std::vector<T> check_to_bit(this->edge_count, T(0));

            // Tracks the best (lowest-weight, converged) solution seen so far across legs,
            // without needing to keep every leg's decoding/log_prob_ratios around.
            bool any_converged = false;
            T best_weight = std::numeric_limits<T>::max();
            std::vector<uint8_t> best_decoding(this->bit_count, 0);
            std::vector<T> best_log_prob_ratios(this->bit_count, T(0));

            for (int leg = 0; leg < this->maximum_legs; leg++) {
                int leg_iterations = 0;
                this->converge = 0;

                this->initialise_log_domain_bp_relay<T>(leg, initial_llr, llr, bit_to_check);
                int leg_max_iterations = (leg == 0) ? this->iterations0 : this->maximum_iterations;
                std::vector<double> memory_strengths = this->generate_memory_strengths_for_leg(leg);

                //main interation loop
                for (int it = 1; it <= leg_max_iterations; it++) {

                    if (this->bp_method == ldpc::bp::PRODUCT_SUM) {
                        for (int i = 0; i < this->check_count; i++) {
                            this->candidate_syndrome[i] = 0;

                            T temp = T(1.0);
                            for (int e: this->row_edges[i]) {
                                check_to_bit[e] = temp;
                                temp *= tanh(bit_to_check[e] / T(2));
                            }

                            temp = T(1);
                            for (auto rit = this->row_edges[i].rbegin(); rit != this->row_edges[i].rend(); ++rit) {
                                const int e = *rit;
                                check_to_bit[e] *= temp;
                                int message_sign = syndrome[i] != 0u ? -1.0 : 1.0;
                                check_to_bit[e] =
                                        T(message_sign) * log((T(1) + check_to_bit[e]) / (T(1) - check_to_bit[e]));
                                temp *= tanh(bit_to_check[e] / T(2));
                            }
                        }
                    } else if (this->bp_method == ldpc::bp::MINIMUM_SUM) {

                        T alpha;
                        if(this->ms_scaling_factor == 0.0) {
                            alpha = T(1.0) - pow(T(2.0), T(-1.0*it));
                        }
                        else {
                            alpha = static_cast<T>(this->ms_scaling_factor);
                        }

                        //check to bit updates
                        for (int i = 0; i < check_count; i++) {

                            this->candidate_syndrome[i] = 0;
                            int total_sgn = 0;
                            int sgn = 0;
                            total_sgn = syndrome[i];
                            T temp = std::numeric_limits<T>::max();

                            for (int e: this->row_edges[i]) {
                                //Change from <= to < to be consistent with ibm implementation. This is of no
                                //consequence but I did this to ensure consistency on a shot by shot basis during debugging
                                if (bit_to_check[e] < 0) {
                                    total_sgn += 1;
                                }
                                check_to_bit[e] = temp;
                                T abs_bit_to_check_msg = abs(bit_to_check[e]);
                                if (abs_bit_to_check_msg < temp) {
                                    temp = abs_bit_to_check_msg;
                                }
                            }

                            temp = std::numeric_limits<T>::max();
                            for (auto rit = this->row_edges[i].rbegin(); rit != this->row_edges[i].rend(); ++rit) {
                                const int e = *rit;
                                sgn = total_sgn;
                                //Change from <= to < to be consistent with ibm implementation. This is of no
                                //consequence but I did this to ensure consistency on a shot by shot basis during debugging
                                if (bit_to_check[e] < 0) {
                                    sgn += 1;
                                }
                                if (temp < check_to_bit[e]) {
                                    check_to_bit[e] = temp;
                                }

                                int message_sign = (sgn % 2 == 0) ? 1.0 : -1.0;

                                check_to_bit[e] *= T(message_sign) * alpha;


                                T abs_bit_to_check_msg = abs(bit_to_check[e]);
                                if (abs_bit_to_check_msg < temp) {
                                    temp = abs_bit_to_check_msg;
                                }

                            }

                        }
                    }


                    //compute log probability ratios
                    for (int i = 0; i < this->bit_count; i++) {

                        T temp; //Implement DMem-BP
                        // elif statements to catch edge case of infinite log prob ratios and zero memory returning nan when multiplied
                        if (memory_strengths[i] == 0.0)
                            temp = initial_llr[i];
                        else if (memory_strengths[i] == 1.0)
                            temp = llr[i];
                        else
                            temp = (T(1) - T(memory_strengths[i])) * initial_llr[i] +
                                T(memory_strengths[i]) * llr[i];
                        for (int e: this->col_edges[i]) {
                            bit_to_check[e] = temp;
                            temp += check_to_bit[e];
                            // if(isnan(temp)) temp = bit_to_check[e];


                        }

                        //make hard decision on basis of log probability ratio for bit i
                        llr[i] = temp;
                        // if(isnan(llr[i])) llr[i] = initial_llr[i];
                        if (temp <= 0) {
                            this->decoding[i] = 1;
                            for (int e: this->col_edges[i]) {
                                this->candidate_syndrome[this->edge_row[e]] ^= 1;
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
                        T temp = T(0);
                        for (auto rit = this->col_edges[i].rbegin(); rit != this->col_edges[i].rend(); ++rit) {
                            const int e = *rit;
                            bit_to_check[e] += temp;
                            temp += check_to_bit[e];
                        }
                    }
                }

                this->iterations += leg_iterations;

                if (this->converge) {
                    T weight = this->decoding_weight<T>(this->decoding);
                    if (!any_converged || weight < best_weight) {
                        best_weight = weight;
                        best_decoding = this->decoding;
                        best_log_prob_ratios = llr;
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
                llr = best_log_prob_ratios;
            }
            //If no leg converged, this->decoding / llr already hold the
            //best-effort result from the final leg that was run.
            this->write_back_to_base<T>(initial_llr, llr, bit_to_check, check_to_bit);
            return this->decoding;
        }

        template<typename T>
        std::vector<uint8_t> &bp_decode_serial_impl(std::vector<uint8_t> &syndrome) {
            using std::log; using std::tanh; using std::abs; using std::pow;

                std::fill(this->decoding.begin(), this->decoding.end(), 0);
                std::fill(this->log_prob_ratios.begin(), this->log_prob_ratios.end(), 0);
                this->solution_number = 0;
                this->iterations = 0;

                std::vector<T> initial_llr(this->bit_count, T(0));
                std::vector<T> llr(this->bit_count, T(0));
                std::vector<T> bit_to_check(this->edge_count, T(0));
                std::vector<T> check_to_bit(this->edge_count, T(0));

                bool any_converged = false;
                T best_weight = std::numeric_limits<T>::max();
                std::vector<uint8_t> best_decoding(this->bit_count, 0);
                std::vector<T> best_log_prob_ratios(this->bit_count, T(0));

                int check_index = 0;

                for (int leg = 0; leg < this->maximum_legs; leg++) {
                    int leg_iterations = 0;
                    this->converge = false;
                    this->initialise_log_domain_bp_relay<T>(leg, initial_llr, llr, bit_to_check);
                    int leg_max_iterations = (leg == 0) ? this->iterations0 : this->maximum_iterations;
                    std::vector<double> memory_strengths = this->generate_memory_strengths_for_leg(leg);

                    for (int it = 1; it <= leg_max_iterations; it++) {

                        T alpha;
                        if(this->ms_scaling_factor == 0.0) {
                            alpha = T(1.0) - pow(T(2.0), T(-1.0*it));
                        }
                        else {
                            alpha = static_cast<T>(this->ms_scaling_factor);
                        }

                        if (this->random_serial_schedule) {
                            this->rng_list_shuffle.shuffle(this->serial_schedule_order);
                        } else if (this->schedule == BpSchedule::SERIAL_RELATIVE) {
                            throw std::logic_error("Not implemented");
                        }

                        for (int bit_index: this->serial_schedule_order) {
                            T temp = T(0);
                            // Handle edge cases of 0 and 1 memory strengths to avoid infty * 0
                            if (memory_strengths[bit_index] == 0.0)
                                llr[bit_index] =
                                        initial_llr[bit_index];
                            else if (memory_strengths[bit_index] == 1.0)
                                llr[bit_index] =
                                        llr[bit_index];
                            else
                                llr[bit_index] =
                                        (T(1) - T(memory_strengths[bit_index])) *
                                        initial_llr[bit_index] +
                                        T(memory_strengths[bit_index]) *
                                        llr[bit_index];
                            if (this->bp_method == 0) {
                                for (int e: this->col_edges[bit_index]) {
                                    check_index = this->edge_row[e];
                                    check_to_bit[e] = T(1.0);
                                    for (int g: this->row_edges[check_index]) {
                                        if (g != e) {
                                            check_to_bit[e] *= tanh(bit_to_check[g] / T(2));
                                        }
                                    }
                                    // pow(-1, syndrome[check_index]) for a 0/1 syndrome bit
                                    check_to_bit[e] = T(syndrome[check_index] % 2 == 0 ? 1 : -1) *
                                                         log((T(1) + check_to_bit[e]) / (T(1) - check_to_bit[e]));
                                    bit_to_check[e] = llr[bit_index];
                                    llr[bit_index] += check_to_bit[e];
                                }
                            } else if (this->bp_method == 1) {
                                for (int e: this->col_edges[bit_index]) {
                                    check_index = this->edge_row[e];
                                    int sgn = syndrome[check_index];
                                    temp = std::numeric_limits<T>::max();
                                    for (int g: this->row_edges[check_index]) {
                                        // TODO: Can this be made more efficient by finding the min and second min value
                                        // just the once for a given syndrome then recycling them for each check to bit,
                                        // only using second min if the min is the edge to be passed to?
                                        // Perhaps this only works for parallel schedule
                                        if (g != e) {
                                            T abs_bit_to_check_msg = abs(bit_to_check[g]);
                                            if (abs_bit_to_check_msg < temp) {
                                                temp = abs_bit_to_check_msg;
                                            }
                                            if (bit_to_check[g] <= 0) {
                                                sgn += 1;
                                            }
                                        }
                                    }
                                    T message_sign = T((sgn % 2 == 0) ? 1.0 : -1.0);
                                    check_to_bit[e] = alpha * message_sign * temp;
                                    bit_to_check[e] = llr[bit_index];
                                    llr[bit_index] += check_to_bit[e];
                                }
                            }
                            if (llr[bit_index] <= 0) {
                                this->decoding[bit_index] = 1;
                            } else {
                                this->decoding[bit_index] = 0;
                            }
                            temp = T(0);
                            for (auto rit = this->col_edges[bit_index].rbegin();
                                 rit != this->col_edges[bit_index].rend(); ++rit) {
                                const int e = *rit;
                                bit_to_check[e] += temp;
                                temp += check_to_bit[e];
                            }
                        }

                        // compute the syndrome for the current candidate decoding solution
                        this->candidate_syndrome = pcm.mulvec(decoding, candidate_syndrome);
                        leg_iterations = it;
                        if (std::equal(candidate_syndrome.begin(), candidate_syndrome.end(), syndrome.begin())) {
                            this->converge = true;
                            this->solution_number += 1;
                            break;
                        }
                    }

                    this->iterations += leg_iterations;

                    if (this->converge) {
                        T weight = this->decoding_weight<T>(this->decoding);
                        if (!any_converged || weight < best_weight) {
                            best_weight = weight;
                            best_decoding = this->decoding;
                            best_log_prob_ratios = llr;
                            any_converged = true;
                        }
                    }

                    if (this->solution_number == this->maximum_solutions) {
                        break;
                    }
                }

                if (any_converged) {
                    this->decoding = best_decoding;
                    llr = best_log_prob_ratios;
                    this->converge = true;
                } else {
                    this->converge = false;
                }

                this->write_back_to_base<T>(initial_llr, llr, bit_to_check, check_to_bit);
                return this->decoding;
            }

        // Runtime dispatch onto the compiled-in precision tiers.
        std::vector<uint8_t> &bp_decode_parallel(std::vector<uint8_t> &syndrome) override {
            switch (this->precision) {
#define RELAY_BP_TIER(Bits) case Bits: return this->bp_decode_parallel_impl<typename PrecisionTraits<Bits>::type>(syndrome);
                RELAY_BP_PRECISION_TIERS
#undef RELAY_BP_TIER
                default: throw std::runtime_error("Unsupported precision: " + std::to_string(this->precision));
            }
        }

        std::vector<uint8_t> &bp_decode_serial(std::vector<uint8_t> &syndrome) override {
            switch (this->precision) {
#define RELAY_BP_TIER(Bits) case Bits: return this->bp_decode_serial_impl<typename PrecisionTraits<Bits>::type>(syndrome);
                RELAY_BP_PRECISION_TIERS
#undef RELAY_BP_TIER
                default: throw std::runtime_error("Unsupported precision: " + std::to_string(this->precision));
            }
        }

        std::vector<uint8_t> &bp_decode_single_scan(std::vector<uint8_t> &syndrome) override  {
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

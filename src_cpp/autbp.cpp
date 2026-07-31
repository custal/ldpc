#ifndef LDPC_AUTBP_HPP
#define LDPC_AUTBP_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
extern "C" { // Claude told me to do this to fix a linkage issue
#include <bliss/bliss_C.h>
}
#include "bp.hpp"
#include "relay_bp.hpp"

namespace ldpc::autbp {
using BpDecoder = ldpc::bp::BpDecoder;
using BpSparse = ldpc::bp::BpSparse;

/* old_col_for_new[j] = i means transformed column j is original column i.
   old_row_for_new has the equivalent convention for rows. */
struct Permutation {
    std::vector<std::size_t> old_col_for_new;
    std::optional<std::vector<std::size_t>> old_row_for_new;
};
struct MemberStats { int iterations = -1; bool converged = false; };
using DecoderFactory = std::function<std::unique_ptr<BpDecoder>(BpSparse&, std::vector<double>)>;

namespace graph_automorphisms {
    using VertexPermutation = std::vector<std::size_t>; // image[old_vertex] = new_vertex

    struct BlissGenerators {
        std::vector<VertexPermutation> generators;
        std::string group_size;
    };

    inline void bliss_hook(void* user, unsigned int n, const unsigned int* aut) {
        auto& out = *static_cast<std::vector<VertexPermutation>*>(user);
        VertexPermutation p(n);
        for (unsigned int i = 0; i < n; ++i) p[i] = aut[i];
        out.push_back(std::move(p));
    }

    /* Build the colored Tanner graph: variable vertices [0,n), check vertices
       [n,n+m). Different colors prevent BLISS from exchanging the two parts. */
    inline BlissGenerators generators_from_pcm(BpSparse& pcm) {
        const std::size_t nv = static_cast<std::size_t>(pcm.n);
        const std::size_t nc = static_cast<std::size_t>(pcm.m);
        BlissGraph* raw = bliss_new(0);
        if (!raw) throw std::bad_alloc();
        struct Guard { BlissGraph* p; ~Guard(){ if (p) bliss_release(p); } } guard{raw};
        for (std::size_t i = 0; i < nv; ++i) bliss_add_vertex(raw, 0);
        for (std::size_t i = 0; i < nc; ++i) bliss_add_vertex(raw, 1);
        for (int r = 0; r < pcm.m; ++r)
            for (auto& e : pcm.iterate_row(r))
                bliss_add_edge(raw, static_cast<unsigned int>(e.col_index),
                               static_cast<unsigned int>(nv + static_cast<std::size_t>(e.row_index)));
        BlissStats stats{};
        BlissGenerators result;
        bliss_find_automorphisms(raw, &bliss_hook, &result.generators, &stats);
        /* The C API's exact group-size representation varies by BLISS release, so
           enumeration below is the authoritative source for the returned count. */
        return result;
    }

    inline VertexPermutation identity(std::size_t n) {
        VertexPermutation p(n); for (std::size_t i=0;i<n;++i) p[i]=i; return p;
    }
    inline VertexPermutation compose(const VertexPermutation& a,
                                     const VertexPermutation& b) {
        // a after b: x -> a[b[x]]
        if (a.size()!=b.size()) throw std::invalid_argument("permutation size mismatch");
        VertexPermutation c(a.size());
        for (std::size_t i=0;i<c.size();++i) c[i]=a[b[i]];
        return c;
    }
    inline std::string key(const VertexPermutation& p) {
        std::string s; s.reserve(p.size()*4);
        for (auto x:p) { s.append(reinterpret_cast<const char*>(&x), sizeof(x)); }
        return s;
    }

    /* Enumerate the generated group by closure. max_elements is mandatory because
       Tanner-graph automorphism groups can be enormous. Identity is included. */
    inline std::vector<VertexPermutation> enumerate_group(
        const std::vector<VertexPermutation>& generators,
        std::size_t vertex_count,
        std::size_t max_elements) {
        if (max_elements == 0) throw std::invalid_argument("max_elements must be positive");
        std::vector<VertexPermutation> elements{identity(vertex_count)};
        std::unordered_set<std::string> seen{key(elements.front())};
        for (std::size_t head=0; head<elements.size(); ++head) {
            for (const auto& g: generators) {
                if (g.size()!=vertex_count) throw std::invalid_argument("generator size mismatch");
                auto next=compose(g,elements[head]);
                if (seen.insert(key(next)).second) {
                    if (elements.size()==max_elements)
                        throw std::overflow_error("automorphism group exceeds max_elements");
                    elements.push_back(std::move(next));
                }
            }
        }
        return elements;
    }

    inline Permutation split_tanner_permutation(const VertexPermutation& image,
                                                std::size_t bit_count,
                                                std::size_t check_count) {
        if (image.size()!=bit_count+check_count) throw std::invalid_argument("Tanner permutation size mismatch");
        Permutation out;
        out.old_col_for_new.resize(bit_count);
        std::vector<std::size_t> rows(check_count);
        for (std::size_t old=0; old<bit_count; ++old) {
            const auto nw=image[old];
            if (nw>=bit_count) throw std::runtime_error("automorphism mixes Tanner vertex colors");
            out.old_col_for_new[nw]=old; // inverse: old index for each new index
        }
        for (std::size_t old=0; old<check_count; ++old) {
            const auto nw=image[bit_count+old];
            if (nw<bit_count || nw>=bit_count+check_count)
                throw std::runtime_error("automorphism mixes Tanner vertex colors");
            rows[nw-bit_count]=old;
        }
        out.old_row_for_new=std::move(rows);
        return out;
    }

    inline std::vector<Permutation> find_from_pcm(BpSparse& pcm,
                                                   std::size_t max_elements,
                                                   bool include_identity=true) {
        auto gens=generators_from_pcm(pcm);
        auto group=enumerate_group(gens.generators,
                                   static_cast<std::size_t>(pcm.n+pcm.m), max_elements);
        std::vector<Permutation> out;
        out.reserve(group.size());
        for (std::size_t i=include_identity?0:1; i<group.size(); ++i)
            out.push_back(split_tanner_permutation(group[i], pcm.n, pcm.m));
        return out;
    }
}

DecoderFactory make_bp_factory(
        int maximum_iterations,
        ldpc::bp::BpMethod bp_method = ldpc::bp::MINIMUM_SUM,
        ldpc::bp::BpSchedule schedule = ldpc::bp::PARALLEL,
        double min_sum_scaling_factor = 1.0,
        int omp_threads = 1,
        const std::vector<int> &serial_schedule = ldpc::bp::NULL_INT_VECTOR,
        int random_schedule_seed = 0,
        bool random_serial_schedule = false,
        ldpc::bp::BpInputType bp_input_type = ldpc::bp::AUTO
    ) {
    return [
        maximum_iterations,
        bp_method,
        schedule,
        min_sum_scaling_factor,
        omp_threads,
        serial_schedule,
        random_schedule_seed,
        random_serial_schedule,
        bp_input_type
    ](
        ldpc::bp::BpSparse& pcm,
        std::vector<double> channel_probabilities
    ) -> std::unique_ptr<ldpc::bp::BpDecoder> {
        return std::make_unique<ldpc::relay::RelayBpDecoder>(
            pcm,
            std::move(channel_probabilities),
            1,
            1,
            maximum_iterations,
            maximum_iterations,
            0.0,
            std::vector<double>{0.0, 0.0},
            std::vector<std::vector<double>>{},
            bp_method,
            schedule,
            min_sum_scaling_factor,
            omp_threads,
            serial_schedule,
            random_schedule_seed,
            random_serial_schedule,
            bp_input_type,
            -1
        );
    };
}

DecoderFactory make_relay_bp_factory(
            int maximum_legs,
            int maximum_solutions,
            int iterations0, //Number of BP iterations run on leg 0 (paired with gamma0)
            int maximum_iterations, //Number of BP iterations run on every leg after the first (paired with gamma_dist_interval)
            double gamma0 = 0.0, //Ignored if memory_strengths_per_leg is provided explicitly
            std::vector<double> gamma_dist_interval = {}, //Ignored if memory_strengths_per_leg is provided explicitly
            std::vector<std::vector<double>> memory_strengths_per_leg = {}, //Optional explicit (maximum_legs x bit_count) override; if given, takes precedence over gamma0/gamma_dist_interval
            ldpc::bp::BpMethod bp_method = ldpc::bp::MINIMUM_SUM,
            ldpc::bp::BpSchedule schedule = ldpc::bp::PARALLEL,
            double min_sum_scaling_factor = 1.0,
            int omp_threads = 1,
            const std::vector<int> &serial_schedule = ldpc::bp::NULL_INT_VECTOR,
            int random_schedule_seed = 0,
            bool random_serial_schedule = false,
            ldpc::bp::BpInputType bp_input_type = ldpc::bp::AUTO,
            int memory_seed = -1 // seed for the per-leg memory strength RNG; -1 -> seed non-deterministically
    ) {
    return [
        maximum_legs,
        maximum_solutions,
        iterations0,
        maximum_iterations,
        gamma0,
        gamma_dist_interval,
        memory_strengths_per_leg,
        bp_method,
        schedule,
        min_sum_scaling_factor,
        omp_threads,
        serial_schedule,
        random_schedule_seed,
        random_serial_schedule,
        bp_input_type,
        memory_seed
    ](
        ldpc::bp::BpSparse& pcm,
        std::vector<double> channel_probabilities
    ) -> std::unique_ptr<ldpc::bp::BpDecoder> {
        return std::make_unique<ldpc::relay::RelayBpDecoder>(
            pcm,
            std::move(channel_probabilities),
            maximum_legs,
            maximum_solutions,
            iterations0,
            maximum_iterations,
            gamma0,
            gamma_dist_interval,
            memory_strengths_per_leg,
            bp_method,
            schedule,
            min_sum_scaling_factor,
            omp_threads,
            serial_schedule,
            random_schedule_seed,
            random_serial_schedule,
            bp_input_type,
            memory_seed
        );
    };
}


class AutBpDecoder {
public:
    int iterations;
    std::size_t solution_number;
    std::vector<std::uint8_t> decoding;
    bool converge;
    std::vector<double> log_prob_ratios;

    AutBpDecoder(const BpSparse& base_pcm, std::vector<double> priors,
                 std::vector<Permutation> permutations, DecoderFactory factory,
                 std::optional<std::size_t> maximum_solutions=std::nullopt)
      : base_pcm_(clone_matrix(base_pcm)), priors_(std::move(priors)),
        permutations_(std::move(permutations)),
        maximum_solutions_(maximum_solutions) {
        initialise(std::move(factory));
        this->solution_number = 0;
        this->iterations = 0;
        this->converge = 0;
        this->decoding.resize(priors_.size());
        this->log_prob_ratios.resize(priors_.size());
    }

    /* Convenience constructor: discover Tanner-graph automorphisms with BLISS,
       then construct one decoder for the original PCM. Each automorphism acts on
       the syndrome and returned decoder coordinates. */
    AutBpDecoder(const BpSparse& base_pcm, std::vector<double> priors,
                 DecoderFactory factory, std::size_t max_automorphisms,
                 std::optional<std::size_t> maximum_solutions=std::nullopt,
                 bool include_identity=true)
      : base_pcm_(clone_matrix(base_pcm)), priors_(std::move(priors)),
        maximum_solutions_(maximum_solutions) {
        permutations_=graph_automorphisms::find_from_pcm(
            *base_pcm_,max_automorphisms,include_identity);
        initialise(std::move(factory));
        this->solution_number = 0;
        this->iterations = 0;
        this->converge = 0;
        this->decoding.resize(priors_.size());
        this->log_prob_ratios.resize(priors_.size());
    }

    std::vector<std::uint8_t> decode(const std::vector<std::uint8_t>& syndrome) {
        if (syndrome.size()!=static_cast<std::size_t>(base_pcm_->m))
            throw std::invalid_argument("syndrome length must equal H.m");
        std::fill(stats_.begin(),stats_.end(),MemberStats{});
        std::vector<std::uint8_t> fallback,best;
        double best_score=-std::numeric_limits<double>::infinity();
        this->solution_number=0;
        this->iterations=0;
        this->converge=0;
        std::fill(this->decoding.begin(), this->decoding.end(), 0);
        std::fill(this->log_prob_ratios.begin(), this->log_prob_ratios.end(), 0);

        for (std::size_t k=0;k<permutations_.size();++k) {
            auto s = map_syndrome_to_decoder_coordinates(syndrome, permutations_[k]);
            decoder_->channel_probabilities = map_channel_probabilities_to_decoder_coordinates(priors_,
                permutations_[k].old_col_for_new);
            auto x=decoder_->decode(s);
            auto correction = map_decoder_output_to_original_coordinates(
                x, permutations_[k].old_col_for_new);
            auto candidate_log_prob_ratios = map_decoder_output_to_original_coordinates(
                decoder_->log_prob_ratios, permutations_[k].old_col_for_new);
            stats_[k]={decoder_->iterations,decoder_->converge};
            this->iterations += decoder_->iterations;
            if (k==0) {
                fallback=correction;
                this->log_prob_ratios=candidate_log_prob_ratios;
            }
            if (base_pcm_->mulvec(correction)!=syndrome) continue;
            ++this->solution_number;
            this->converge=true;
            double score=log_likelihood(correction);
            if (best.empty() || score>best_score) { best=std::move(correction); best_score=score;
                this->log_prob_ratios=std::move(candidate_log_prob_ratios);}
            if (maximum_solutions_ && this->solution_number>=*maximum_solutions_) break;
        }
        decoder_->channel_probabilities = priors_;
        if (!best.empty()) this->decoding = best;
        else if (!fallback.empty()) this->decoding = fallback;
        else this->decoding = std::vector<std::uint8_t>(priors_.size(),0);
        return this->decoding;
    }

    const std::vector<MemberStats>& last_member_stats() const noexcept { return stats_; }
    const std::vector<Permutation>& permutations() const noexcept { return permutations_; }

private:
    std::unique_ptr<BpSparse> base_pcm_;
    std::vector<double> priors_;
    std::vector<Permutation> permutations_;
    std::optional<std::size_t> maximum_solutions_;
    // One fixed decoder; permutations only transform its syndrome and outputs.
    std::unique_ptr<BpDecoder> decoder_;
    std::vector<MemberStats> stats_;

    void initialise(DecoderFactory factory) {
        if(!factory) throw std::invalid_argument("decoder factory is empty");
        if(permutations_.empty()) throw std::invalid_argument("permutations must be non-empty");
        if(priors_.size()!=static_cast<std::size_t>(base_pcm_->n)) throw std::invalid_argument("priors length must equal H.n");
        if(maximum_solutions_ && *maximum_solutions_==0) throw std::invalid_argument("maximum_solutions must be positive");

        for(const auto& p:permutations_) {
            validate_perm(p.old_col_for_new,base_pcm_->n);
            if(p.old_row_for_new) validate_perm(*p.old_row_for_new,base_pcm_->m);
        }

        // The ensemble reuses one decoder on the original PCM and prior order.
        // Each permutation acts only on the syndrome and decoder outputs.
        decoder_=factory(*base_pcm_,priors_);
        if(!decoder_) throw std::runtime_error("decoder factory returned null");
        stats_.resize(permutations_.size());
    }
    static void validate_perm(const std::vector<std::size_t>& p,std::size_t n) {
        if(p.size()!=n) throw std::invalid_argument("permutation has wrong size");
        std::vector<bool> seen(n,false); for(auto x:p) { if(x>=n||seen[x]) throw std::invalid_argument("invalid permutation"); seen[x]=true; }
    }

    static std::unique_ptr<BpSparse> clone_matrix(const BpSparse& src) {
        auto out=std::make_unique<BpSparse>(src.m,src.n); auto& s=const_cast<BpSparse&>(src);
        for(int r=0;r<src.m;++r) for(auto& e:s.iterate_row(r)) out->insert_entry(e.row_index,e.col_index);
        return out;
    }

    static std::vector<std::uint8_t> map_syndrome_to_decoder_coordinates(const std::vector<std::uint8_t>& s,const Permutation& p) {
        if(!p.old_row_for_new)return s;
        std::vector<std::uint8_t> o(s.size());
        for(std::size_t new_row=0;new_row<o.size();++new_row) {
            const std::size_t old_row = (*p.old_row_for_new)[new_row];
            o[old_row] = s[new_row] &1u;
        }
        return o;
    }

    template<typename T> // Template as same mapping used for correction <uint8_t> and log_prob_ratios <double>
    static std::vector<T> map_decoder_output_to_original_coordinates(
        const std::vector<T>& decoder_output,
        const std::vector<std::size_t>& old_col_for_new) {
        if(decoder_output.size()!=old_col_for_new.size())
            throw std::runtime_error("decoder output has wrong length");
        std::vector<T> original_output(decoder_output.size());
        for(std::size_t new_col=0;new_col<original_output.size();++new_col) {
            const std::size_t old_col = old_col_for_new[new_col];
            original_output[new_col] = decoder_output[old_col];
        }
        return original_output;
    }

    static std::vector<double> map_channel_probabilities_to_decoder_coordinates(
        const std::vector<double>& original_channel_probabilities,
        const std::vector<std::size_t>& old_col_for_new) {
        if(original_channel_probabilities.size()!=old_col_for_new.size())
            throw std::runtime_error("decoder input has wrong length");
        std::vector<double> decoder_channel_probabilities(original_channel_probabilities.size());
        for(std::size_t new_col=0;new_col<decoder_channel_probabilities.size();++new_col) {
            const std::size_t old_col = old_col_for_new[new_col];
            decoder_channel_probabilities[old_col] = original_channel_probabilities[new_col];
        }
        return decoder_channel_probabilities;
    }

    double log_likelihood(const std::vector<std::uint8_t>& e) const {
        double z=0; for(std::size_t i=0;i<e.size();++i){double p=priors_[i]; if(!(p>=0&&p<=1))throw std::domain_error("prior outside [0,1]"); z+=e[i]?std::log(p):std::log1p(-p);} return z;
    }
};
} // namespace ldpc::autbp
#endif

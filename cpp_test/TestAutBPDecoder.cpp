// TestAutBPDecoder.cpp
//
// Tests for AutBpDecoder (aut_bp.hpp).  These focus on the wrapper contract:
// permutation handling, validation, ensemble execution, stopping, statistics,
// and end-to-end decoding.  Tests of RelayBpDecoder internals remain in the
// relay-specific test file.

#include <gtest/gtest.h>

#include "autbp.cpp"
#include "gf2codes.hpp"
#include "ldpc.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

using std::size_t;
using std::uint8_t;
using std::vector;
using ldpc::autbp::AutBpDecoder;
using ldpc::autbp::DecoderFactory;
using ldpc::autbp::OptionalSerialSchedule;
using ldpc::autbp::Permutation;

namespace {

Permutation identity_permutation(size_t n, size_t m) {
    Permutation p;
    p.old_col_for_new.resize(n);
    p.old_row_for_new = vector<size_t>(m);
    for (size_t i = 0; i < n; ++i) p.old_col_for_new[i] = i;
    for (size_t i = 0; i < m; ++i) p.old_row_for_new[i] = i;
    return p;
}

// Reversal is an automorphism of the path-form repetition-code Tanner graph.
// Variable i -> n-1-i and check r -> m-1-r.  Since reversal is self-inverse,
// the old-for-new arrays have the same values as the forward maps.
Permutation reversal_permutation(size_t n, size_t m) {
    Permutation p;
    p.old_col_for_new.resize(n);
    p.old_row_for_new = vector<size_t>(m);
    for (size_t j = 0; j < n; ++j) p.old_col_for_new[j] = n - 1 - j;
    for (size_t j = 0; j < m; ++j) p.old_row_for_new[j] = m - 1 - j;
    return p;
}

DecoderFactory basic_bp_factory(int maximum_iterations = 10) {
    return ldpc::autbp::make_bp_factory(
        maximum_iterations
    );
}

AutBpDecoder make_rep_autbp(
    ldpc::bp::BpSparse& pcm,
    vector<double> priors,
    std::optional<size_t> maximum_solutions = std::nullopt,
    int maximum_iterations = 10
) {
    // rep decoder only has these two permutations
    vector<Permutation> permutations{
        identity_permutation(static_cast<size_t>(pcm.n),
                             static_cast<size_t>(pcm.m)),
        reversal_permutation(static_cast<size_t>(pcm.n),
                             static_cast<size_t>(pcm.m))
    };
    return AutBpDecoder(
        pcm,
        std::move(priors),
        std::move(permutations),
        basic_bp_factory(maximum_iterations),
        maximum_solutions
    );
}

} // namespace

// ============================================================
// Graph-permutation utilities
// ============================================================

TEST(AutBpGraphAutomorphisms, IdentityHasExpectedImages) {
    const auto p = ldpc::autbp::graph_automorphisms::identity(6);
    EXPECT_EQ((vector<size_t>{0, 1, 2, 3, 4, 5}), p);
}

TEST(AutBpGraphAutomorphisms, ComposeMeansAAfterB) {
    const vector<size_t> a{1, 2, 0};
    const vector<size_t> b{2, 0, 1};
    EXPECT_EQ((vector<size_t>{0, 1, 2}),
              ldpc::autbp::graph_automorphisms::compose(a, b));
}

TEST(AutBpGraphAutomorphisms, ComposeRejectsDifferentSizes) {
    EXPECT_THROW(
        ldpc::autbp::graph_automorphisms::compose(
            vector<size_t>{0}, vector<size_t>{0, 1}),
        std::invalid_argument
    );
}

TEST(AutBpGraphAutomorphisms, EnumerateGroupIncludesIdentityAndClosure) {
    // Generator (0 1 2), fixing vertex 3, generates three elements.
    const vector<size_t> cycle{1, 2, 0, 3};
    const auto group = ldpc::autbp::graph_automorphisms::enumerate_group(
        {cycle}, 4, 3);
    ASSERT_EQ(3u, group.size());
    EXPECT_EQ((vector<size_t>{0, 1, 2, 3}), group[0]);
    EXPECT_EQ(cycle, group[1]);
    EXPECT_EQ((vector<size_t>{2, 0, 1, 3}), group[2]);
}

TEST(AutBpGraphAutomorphisms, EnumerateGroupValidatesLimitAndGeneratorSize) {
    EXPECT_THROW(
        ldpc::autbp::graph_automorphisms::enumerate_group({}, 3, 0),
        std::invalid_argument
    );
    EXPECT_THROW(
        ldpc::autbp::graph_automorphisms::enumerate_group(
            {vector<size_t>{0, 1}}, 3, 10),
        std::invalid_argument
    );
}

TEST(AutBpGraphAutomorphisms, EnumerateGroupThrowsWhenLimitIsExceeded) {
    EXPECT_THROW(
        ldpc::autbp::graph_automorphisms::enumerate_group(
            {vector<size_t>{1, 2, 0}}, 3, 2),
        std::overflow_error
    );
}

TEST(AutBpGraphAutomorphisms, SplitTannerPermutationBuildsInverseMaps) {
    // Forward images: variables 0->1, 1->2, 2->0; checks 0<->1.
    const vector<size_t> image{1, 2, 0, 4, 3};
    const auto p = ldpc::autbp::graph_automorphisms::split_tanner_permutation(
        image, 3, 2);
    EXPECT_EQ((vector<size_t>{2, 0, 1}), p.old_col_for_new);
    EXPECT_EQ((vector<size_t>{1, 0}), p.old_row_for_new);
}

TEST(AutBpGraphAutomorphisms, SplitTannerPermutationRejectsWrongSizeOrMixedColors) {
    EXPECT_THROW(
        ldpc::autbp::graph_automorphisms::split_tanner_permutation(
            vector<size_t>{0, 1}, 2, 1),
        std::invalid_argument
    );
    // The two colours are bit and check. The first 2 vetices are bit and the last is check but the defined map
    // permutes 2<->0 which maps a check to a bit so should be invalid
    EXPECT_THROW(
        ldpc::autbp::graph_automorphisms::split_tanner_permutation(
            vector<size_t>{2, 1, 0}, 2, 1),
        std::runtime_error
    );
}

// ============================================================
// Construction and validation
// ============================================================

TEST(AutBpDecoder, StoresExplicitPermutations) {
    const int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    const auto identity = identity_permutation(pcm.n, pcm.m);
    const auto reversal = reversal_permutation(pcm.n, pcm.m);
    AutBpDecoder decoder(
        pcm, vector<double>(n, 0.1), {identity, reversal}, basic_bp_factory());

    ASSERT_EQ(2u, decoder.permutations().size());
    EXPECT_EQ(identity.old_col_for_new,
              decoder.permutations()[0].old_col_for_new);
    EXPECT_EQ(reversal.old_col_for_new,
              decoder.permutations()[1].old_col_for_new);
}

TEST(AutBpDecoder, RejectsEmptyFactory) {
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(3);
    EXPECT_THROW(
        AutBpDecoder(pcm, vector<double>(3, 0.1),
                     {identity_permutation(pcm.n, pcm.m)}, DecoderFactory{}),
        std::invalid_argument
    );
}

TEST(AutBpDecoder, RejectsEmptyPermutationEnsemble) {
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(3);
    EXPECT_THROW(
        AutBpDecoder(pcm, vector<double>(3, 0.1), {}, basic_bp_factory()),
        std::invalid_argument
    );
}

TEST(AutBpDecoder, RejectsWrongPriorCount) {
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(5);
    EXPECT_THROW(
        AutBpDecoder(pcm, vector<double>(4, 0.1),
                     {identity_permutation(pcm.n, pcm.m)}, basic_bp_factory()),
        std::invalid_argument
    );
}

TEST(AutBpDecoder, RejectsZeroMaximumSolutions) {
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(3);
    EXPECT_THROW(
        AutBpDecoder(pcm, vector<double>(3, 0.1),
                     {identity_permutation(pcm.n, pcm.m)}, basic_bp_factory(), size_t{0}),
        std::invalid_argument
    );
}

TEST(AutBpDecoder, RejectsWrongSizedDuplicateAndOutOfRangeColumnPermutations) {
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(3);
    const auto factory = basic_bp_factory();

    Permutation wrong_size{{0, 1}, {0, 1}};
    EXPECT_THROW(AutBpDecoder(pcm, vector<double>(3, 0.1), {wrong_size}, factory),
                 std::invalid_argument);

    Permutation duplicate{{0, 0, 2}, {0, 1}};
    EXPECT_THROW(AutBpDecoder(pcm, vector<double>(3, 0.1), {duplicate}, factory),
                 std::invalid_argument);

    Permutation out_of_range{{0, 1, 3}, {0, 1}};
    EXPECT_THROW(AutBpDecoder(pcm, vector<double>(3, 0.1), {out_of_range}, factory),
                 std::invalid_argument);
}

TEST(AutBpDecoder, RejectsInvalidRowPermutation) {
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(3);
    Permutation p{{0, 1, 2}, vector<size_t>{0, 0}};
    EXPECT_THROW(
        AutBpDecoder(pcm, vector<double>(3, 0.1), {p}, basic_bp_factory()),
        std::invalid_argument
    );
}

TEST(AutBpDecoder, RejectsNullDecoderFromFactory) {
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(3);

    DecoderFactory null_factory = [](
        ldpc::bp::BpSparse&,
        vector<double>,
        const OptionalSerialSchedule&)
     -> std::unique_ptr<ldpc::bp::BpDecoder> {
        return nullptr;
    };

    EXPECT_THROW(
        AutBpDecoder(
            pcm,
            vector<double>(3, 0.1),
            {identity_permutation(pcm.n, pcm.m)},
            std::move(null_factory)
        ),
        std::runtime_error
    );
}

TEST(AutBpDecoder, DecodeRejectsWrongSyndromeLength) {
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(5);
    auto decoder = make_rep_autbp(pcm, vector<double>(5, 0.1));
    EXPECT_THROW(decoder.decode(vector<uint8_t>(pcm.m - 1, 0)),
                 std::invalid_argument);
}

// ============================================================
// End-to-end decoding and ensemble behaviour
// ============================================================

TEST(AutBpDecoder, IdentityMemberMatchesExpectedRepCodeDecodings) {
    const int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    AutBpDecoder decoder(
        pcm,
        vector<double>(n, 0.1),
        {identity_permutation(pcm.n, pcm.m)},
        basic_bp_factory()
    );

    const vector<vector<uint8_t>> syndromes{
        {0, 0, 0, 0},
        {0, 0, 0, 1},
        {0, 1, 0, 1},
        {1, 0, 1, 0},
        {1, 1, 1, 1}
    };
    const vector<vector<uint8_t>> expected{
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1},
        {0, 0, 1, 1, 0},
        {0, 1, 1, 0, 0},
        {0, 1, 0, 1, 0}
    };

    for (size_t i = 0; i < syndromes.size(); ++i) {
        auto result = decoder.decode(syndromes[i]);
        EXPECT_EQ(expected[i], result);
        EXPECT_EQ(syndromes[i], pcm.mulvec(result));
    }
}

TEST(AutBpDecoder, ReversedMemberReturnsCorrectionInOriginalCoordinates) {
    const int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    AutBpDecoder decoder(
        pcm,
        vector<double>(n, 0.1),
        {reversal_permutation(pcm.n, pcm.m)},
        basic_bp_factory()
    );

    const vector<uint8_t> syndrome{0, 0, 0, 1};
    auto result = decoder.decode(syndrome);
    EXPECT_EQ((vector<uint8_t>{0, 0, 0, 0, 1}), result);
    EXPECT_EQ(syndrome, pcm.mulvec(result));
}

TEST(AutBpDecoder, EveryReturnedConvergedCorrectionHasOriginalSyndrome) {
    const int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_rep_autbp(pcm, vector<double>(n, 0.1));

    const vector<vector<uint8_t>> syndromes{
        {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 1, 0, 1},
        {1, 0, 1, 0}, {1, 1, 1, 1}
    };
    for (const auto& syndrome : syndromes) {
        auto result = decoder.decode(syndrome);
        EXPECT_EQ(syndrome, pcm.mulvec(result));
    }
}

TEST(AutBpDecoder, MaximumSolutionsStopsAfterFirstValidMember) {
    const int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_rep_autbp(pcm, vector<double>(n, 0.1), size_t{1});

    EXPECT_EQ((vector<uint8_t>(n, 0)),
              decoder.decode(vector<uint8_t>(pcm.m, 0)));
    const auto& stats = decoder.last_member_stats();
    ASSERT_EQ(2u, stats.size());
    EXPECT_TRUE(stats[0].converged);
    EXPECT_GT(stats[0].iterations, 0);
    EXPECT_FALSE(stats[1].converged);
    EXPECT_EQ(-1, stats[1].iterations);
}

TEST(AutBpDecoder, WithoutSolutionLimitAllMembersRun) {
    const int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_rep_autbp(pcm, vector<double>(n, 0.1));

    decoder.decode(vector<uint8_t>(pcm.m, 0));
    const auto& stats = decoder.last_member_stats();
    ASSERT_EQ(2u, stats.size());
    EXPECT_TRUE(stats[0].converged);
    EXPECT_TRUE(stats[1].converged);
    EXPECT_GT(stats[0].iterations, 0);
    EXPECT_GT(stats[1].iterations, 0);
}

TEST(AutBpDecoder, MemberStatsAreResetBetweenDecodeCalls) {
    const int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_rep_autbp(pcm, vector<double>(n, 0.1), size_t{1});

    decoder.decode(vector<uint8_t>(pcm.m, 0));
    ASSERT_EQ(-1, decoder.last_member_stats()[1].iterations);

    decoder.decode(vector<uint8_t>{0, 0, 0, 1});
    const auto& stats = decoder.last_member_stats();
    ASSERT_EQ(2u, stats.size());
    EXPECT_GT(stats[0].iterations, 0);
    EXPECT_EQ(-1, stats[1].iterations);
    EXPECT_FALSE(stats[1].converged);
}

TEST(AutBpDecoder, ZeroIterationMemberReturnsFallback) {
    const int n = 3;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    AutBpDecoder decoder(
        pcm,
        vector<double>(n, 0.1),
        {identity_permutation(pcm.n, pcm.m)},
        basic_bp_factory(0)
    );

    auto result = decoder.decode(vector<uint8_t>{1, 1});
    EXPECT_EQ((vector<uint8_t>(n, 0)), result);
    ASSERT_EQ(1u, decoder.last_member_stats().size());
    EXPECT_FALSE(decoder.last_member_stats()[0].converged);
    EXPECT_EQ(0, decoder.last_member_stats()[0].iterations);
}

TEST(AutBpDecoder, PublicDecodeAttributesMatchSingleMemberDecoder) {
    const int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    const vector<double> priors(n, 0.1);
    vector<uint8_t> syndrome{0, 0, 0, 1};

    // Run the underlying decoder directly to obtain deterministic reference
    // values for every field copied or accumulated by AutBpDecoder.
    auto reference = basic_bp_factory()(pcm, priors, std::nullopt);
    ASSERT_NE(nullptr, reference);
    auto expected_decoding = reference->decode(syndrome);
    ASSERT_EQ(syndrome, pcm.mulvec(expected_decoding));

    AutBpDecoder decoder(
        pcm,
        priors,
        {identity_permutation(pcm.n, pcm.m)},
        basic_bp_factory()
    );
    const auto returned_decoding = decoder.decode(syndrome);

    EXPECT_EQ(expected_decoding, returned_decoding);
    EXPECT_EQ(returned_decoding, decoder.decoding);
    EXPECT_EQ(reference->iterations, decoder.iterations);
    EXPECT_EQ(1u, decoder.solution_number);
    EXPECT_TRUE(decoder.converge);
    EXPECT_EQ(reference->log_prob_ratios, decoder.log_prob_ratios);
    EXPECT_EQ(static_cast<size_t>(n), decoder.log_prob_ratios.size());
}

TEST(AutBpDecoder, PublicDecodeAttributesAggregateAndResetBetweenCalls) {
    const int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_rep_autbp(pcm, vector<double>(n, 0.1));

    const vector<uint8_t> first_syndrome(pcm.m, 0);
    const auto first_result = decoder.decode(first_syndrome);
    const auto& first_stats = decoder.last_member_stats();
    ASSERT_EQ(2u, first_stats.size());
    ASSERT_TRUE(first_stats[0].converged);
    ASSERT_TRUE(first_stats[1].converged);

    EXPECT_EQ(first_result, decoder.decoding);
    EXPECT_TRUE(decoder.converge);
    EXPECT_EQ(2u, decoder.solution_number);
    EXPECT_EQ(std::max(first_stats[0].iterations, first_stats[1].iterations),
              decoder.iterations);
    EXPECT_EQ(static_cast<size_t>(n), decoder.log_prob_ratios.size());

    // A second call must describe that call only, rather than accumulate the
    // public values from the first call.
    const vector<uint8_t> second_syndrome{0, 0, 0, 1};
    const auto second_result = decoder.decode(second_syndrome);
    const auto& second_stats = decoder.last_member_stats();
    ASSERT_EQ(2u, second_stats.size());

    EXPECT_EQ(second_result, decoder.decoding);
    EXPECT_EQ(second_syndrome, pcm.mulvec(decoder.decoding));
    EXPECT_TRUE(decoder.converge);
    EXPECT_EQ(2u, decoder.solution_number);
    EXPECT_EQ(std::max(second_stats[0].iterations, second_stats[1].iterations),
              decoder.iterations);
    EXPECT_EQ(static_cast<size_t>(n), decoder.log_prob_ratios.size());
}

TEST(AutBpDecoder, PublicDecodeAttributesDescribeUnsuccessfulDecode) {
    const int n = 3;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    AutBpDecoder decoder(
        pcm,
        vector<double>(n, 0.1),
        {identity_permutation(pcm.n, pcm.m)},
        basic_bp_factory(0)
    );

    const auto returned_decoding = decoder.decode(vector<uint8_t>{1, 1});

    EXPECT_EQ(returned_decoding, decoder.decoding);
    EXPECT_EQ((vector<uint8_t>(n, 0)), decoder.decoding);
    EXPECT_EQ(0u, decoder.solution_number);
    EXPECT_EQ(0, decoder.iterations);
    EXPECT_FALSE(decoder.converge);
    EXPECT_EQ(static_cast<size_t>(n), decoder.log_prob_ratios.size());
}

TEST(AutBpDecoder, DoesNotMutateSourceMatrix) {
    const int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    vector<uint8_t> probe{1, 0, 1, 0, 1};
    auto before = pcm.mulvec(probe);

    auto decoder = make_rep_autbp(pcm, vector<double>(n, 0.1));
    decoder.decode(vector<uint8_t>(pcm.m, 0));

    EXPECT_EQ(before, pcm.mulvec(probe));
}

// BLISS integration smoke test. .
TEST(AutBpGraphAutomorphisms, DiscoveryIncludesIdentityAndProducesValidMaps) {
    const int n = 3;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    const auto automorphisms =
        ldpc::autbp::graph_automorphisms::find_from_pcm(pcm, 16, true);

    ASSERT_FALSE(automorphisms.empty());
    EXPECT_EQ((vector<size_t>{0, 1, 2}),
              automorphisms[0].old_col_for_new);
    EXPECT_EQ((vector<size_t>{0, 1}),
              automorphisms[0].old_row_for_new);

    // Expected permutation for 3 bit repetition code
    EXPECT_EQ((vector<size_t>{2, 1, 0}),
              automorphisms[1].old_col_for_new);
    EXPECT_EQ((vector<size_t>{1, 0}),
              automorphisms[1].old_row_for_new);
}

TEST(AutBpDecoder, PermutingPriorsBeforeDecodingIsCorrectOperation) {
    const int n = 3;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    // Choose extreme priors which will only converge if priors are appropriately permuted
    auto decoder = make_rep_autbp(pcm, vector<double>{0, 0, 1});

    // Expect this to be decoded to an error on the far right bit
    const vector<uint8_t> syndrome1 {0,1};
    vector<uint8_t> expected {0,0,1};

    auto result1 = decoder.decode(syndrome1);
    auto last_member_stats1 = decoder.last_member_stats();
    EXPECT_EQ(expected, result1);
    EXPECT_EQ(last_member_stats1[0].converged, true);
    EXPECT_EQ(last_member_stats1[1].converged, true);

    // Expect this not to converge.
    const vector<uint8_t> syndrome2 = {1,0};

    auto result2 = decoder.decode(syndrome2);
    auto last_member_stats = decoder.last_member_stats();
    EXPECT_EQ(last_member_stats[0].converged, false);
    EXPECT_EQ(last_member_stats[1].converged, false);
}

TEST(AutBpDecoder, RejectsWrongNumberOfSerialSchedules) {
    const int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);

    vector<Permutation> permutations{
        identity_permutation(
            static_cast<size_t>(pcm.n),
            static_cast<size_t>(pcm.m)
        ),
        reversal_permutation(
            static_cast<size_t>(pcm.n),
            static_cast<size_t>(pcm.m)
        )
    };

    // Two permutations, but only one serial-schedule entry.
    vector<OptionalSerialSchedule> serial_schedules{
        vector<int>{0, 1, 2, 3, 4}
    };

    EXPECT_THROW(
        AutBpDecoder(
            pcm,
            vector<double>(n, 0.1),
            std::move(permutations),
            basic_bp_factory(),
            std::nullopt,
            std::move(serial_schedules)
        ),
        std::invalid_argument
    );
}

TEST(AutBpDecoder, PassesSerialScheduleOverrideForEachPermutation) {
    const int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);

    vector<Permutation> permutations{
        identity_permutation(
            static_cast<size_t>(pcm.n),
            static_cast<size_t>(pcm.m)
        ),
        reversal_permutation(
            static_cast<size_t>(pcm.n),
            static_cast<size_t>(pcm.m)
        )
    };

    vector<OptionalSerialSchedule> serial_schedules{
        vector<int>{0, 1, 2, 3, 4},
        vector<int>{4, 3, 2, 1, 0}
    };

    auto received_schedules =
        std::make_shared<vector<OptionalSerialSchedule>>();

    DecoderFactory underlying_factory = basic_bp_factory();

    DecoderFactory recording_factory = [
        received_schedules,
        underlying_factory
    ](
        ldpc::bp::BpSparse& member_pcm,
        vector<double> member_priors,
        const OptionalSerialSchedule& member_serial_schedule
    ) mutable -> std::unique_ptr<ldpc::bp::BpDecoder> {
        received_schedules->push_back(member_serial_schedule);

        return underlying_factory(
            member_pcm,
            std::move(member_priors),
            member_serial_schedule
        );
    };

    AutBpDecoder decoder(
        pcm,
        vector<double>(n, 0.1),
        std::move(permutations),
        std::move(recording_factory),
        std::nullopt,
        serial_schedules
    );

    ASSERT_EQ(2u, received_schedules->size());

    ASSERT_TRUE((*received_schedules)[0].has_value());
    ASSERT_TRUE((*received_schedules)[1].has_value());

    EXPECT_EQ(
        vector<int>({0, 1, 2, 3, 4}),
        (*received_schedules)[0].value()
    );

    EXPECT_EQ(
        vector<int>({4, 3, 2, 1, 0}),
        (*received_schedules)[1].value()
    );
}

TEST(AutBpDecoder, NullSerialScheduleIsPassedToFactory) {
    const int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);

    auto received_schedules =
        std::make_shared<vector<OptionalSerialSchedule>>();

    DecoderFactory underlying_factory = basic_bp_factory();

    DecoderFactory recording_factory = [
        received_schedules,
        underlying_factory
    ](
        ldpc::bp::BpSparse& member_pcm,
        vector<double> member_priors,
        const OptionalSerialSchedule& member_serial_schedule
    ) mutable -> std::unique_ptr<ldpc::bp::BpDecoder> {
        received_schedules->push_back(member_serial_schedule);

        return underlying_factory(
            member_pcm,
            std::move(member_priors),
            member_serial_schedule
        );
    };

    vector<OptionalSerialSchedule> serial_schedules{
        std::nullopt
    };

    AutBpDecoder decoder(
        pcm,
        vector<double>(n, 0.1),
        {identity_permutation(pcm.n, pcm.m)},
        std::move(recording_factory),
        std::nullopt,
        std::move(serial_schedules)
    );

    ASSERT_EQ(1u, received_schedules->size());
    EXPECT_FALSE((*received_schedules)[0].has_value());

    const vector<uint8_t> syndrome{0, 0, 0, 1};
    auto result = decoder.decode(syndrome);

    EXPECT_EQ(syndrome, pcm.mulvec(result));
}

TEST(AutBpDecoder, IdentityPermutationSerialScheduleEnsembleDecodesCorrectly) {
    const int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);

    const Permutation identity = identity_permutation(
        static_cast<size_t>(pcm.n),
        static_cast<size_t>(pcm.m)
    );

    // Both members use the original PCM coordinates. Their only difference is
    // the order in which variable nodes are processed.
    vector<Permutation> permutations{
        identity,
        identity
    };

    vector<OptionalSerialSchedule> serial_schedules{
        vector<int>{0, 1, 2, 3, 4},
        vector<int>{4, 3, 2, 1, 0}
    };

    DecoderFactory factory = ldpc::autbp::make_bp_factory(
        10,
        ldpc::bp::MINIMUM_SUM,
        ldpc::bp::SERIAL
    );

    AutBpDecoder decoder(
        pcm,
        vector<double>(n, 0.1),
        std::move(permutations),
        std::move(factory),
        std::nullopt,
        std::move(serial_schedules)
    );

    const vector<uint8_t> syndrome{0, 0, 0, 1};
    const vector<uint8_t> expected{0, 0, 0, 0, 1};

    vector<uint8_t> result = decoder.decode(syndrome);

    EXPECT_EQ(expected, result);
    EXPECT_EQ(syndrome, pcm.mulvec(result));

    EXPECT_EQ(result, decoder.decoding);
    EXPECT_TRUE(decoder.converge);
    EXPECT_EQ(2u, decoder.solution_number);

    const auto& stats = decoder.last_member_stats();

    ASSERT_EQ(2u, stats.size());

    EXPECT_TRUE(stats[0].converged);
    EXPECT_TRUE(stats[1].converged);

    EXPECT_GT(stats[0].iterations, 0);
    EXPECT_GT(stats[1].iterations, 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

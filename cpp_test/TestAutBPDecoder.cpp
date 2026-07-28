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
using ldpc::autbp::Permutation;

namespace {

Permutation identity_permutation(size_t n, size_t m) {
    Permutation p;
    p.old_col_for_new.resize(n);
    p.old_row_for_new = vector<size_t>(m);
    for (size_t i = 0; i < n; ++i) p.old_col_for_new[i] = i;
    for (size_t i = 0; i < m; ++i) (*p.old_row_for_new)[i] = i;
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
    for (size_t j = 0; j < m; ++j) (*p.old_row_for_new)[j] = m - 1 - j;
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
    ASSERT_TRUE(p.old_row_for_new.has_value());
    EXPECT_EQ((vector<size_t>{1, 0}), *p.old_row_for_new);
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

    Permutation wrong_size{{0, 1}, std::nullopt};
    EXPECT_THROW(AutBpDecoder(pcm, vector<double>(3, 0.1), {wrong_size}, factory),
                 std::invalid_argument);

    Permutation duplicate{{0, 0, 2}, std::nullopt};
    EXPECT_THROW(AutBpDecoder(pcm, vector<double>(3, 0.1), {duplicate}, factory),
                 std::invalid_argument);

    Permutation out_of_range{{0, 1, 3}, std::nullopt};
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
    DecoderFactory null_factory =
        [](ldpc::bp::BpSparse&, vector<double>)
            -> std::unique_ptr<ldpc::bp::BpDecoder> { return nullptr; };
    EXPECT_THROW(
        AutBpDecoder(pcm, vector<double>(3, 0.1),
                     {identity_permutation(pcm.n, pcm.m)}, null_factory),
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
    ASSERT_TRUE(automorphisms[0].old_row_for_new.has_value());
    EXPECT_EQ((vector<size_t>{0, 1}),
              *automorphisms[0].old_row_for_new);

    // Expected permutation for 3 bit repetition code
    EXPECT_EQ((vector<size_t>{2, 1, 0}),
              automorphisms[1].old_col_for_new);
    ASSERT_TRUE(automorphisms[0].old_row_for_new.has_value());
    EXPECT_EQ((vector<size_t>{1, 0}),
              *automorphisms[1].old_row_for_new);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

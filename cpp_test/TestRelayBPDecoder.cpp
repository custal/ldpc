// TestRelayBPDecoder.cpp
//
// Tests for the refactored RelayBpDecoder (relay_bp.hpp).

#include <gtest/gtest.h>

#include "relay_bp.hpp"
#include "gf2codes.hpp"
#include "ldpc.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

using namespace std;
using namespace ldpc::relay;

// ============================================================
// Test helpers
// ============================================================

// Builds a relay decoder using explicit memory strengths.
//
// Leg 0 is assigned zero memory by default. Later legs use
// uniform_memory_strength.
static RelayBpDecoder make_relay_decoder(
    ldpc::bp::BpSparse &pcm,
    int n,
    int maximum_legs,
    int maximum_solutions,
    double uniform_memory_strength = 0.0,
    int iterations0 = 10,
    int later_leg_iterations = 10,
    ldpc::bp::BpMethod method = ldpc::bp::PRODUCT_SUM,
    int precision = DEFAULT_PRECISION,
    ldpc::bp::BpSchedule schedule = ldpc::bp::PARALLEL
) {
    auto channel_probabilities = vector<double>(n, 0.1);

    auto memory_strengths =
        vector<vector<double>>(
            maximum_legs,
            vector<double>(n, uniform_memory_strength)
        );

    if (maximum_legs > 0) {
        memory_strengths[0] = vector<double>(n, 0.0);
    }

    return RelayBpDecoder(
        pcm,
        channel_probabilities,
        maximum_legs,
        maximum_solutions,
        iterations0,
        later_leg_iterations,
        0.0,                       // gamma0 ignored: explicit memory supplied
        {},                        // distribution ignored
        memory_strengths,
        method,
        schedule,
        1.0,                       // min_sum_scaling_factor
        1,                         // omp_threads
        ldpc::bp::NULL_INT_VECTOR,
        0,                         // random_schedule_seed
        false,                     // random_serial_schedule
        ldpc::bp::AUTO,
        -1,                        // memory_seed
        precision
    );
}

// The precision tiers this build offers, excluding the widest ones so that loops
// over every tier stay quick. The widest tier is covered separately by
// MaximumPrecisionTierDecodes.
static vector<int> fast_precision_tiers(int max_bits = 512) {
    auto tiers = vector<int>();

    for (int bits : available_precisions()) {
        if (bits <= max_bits) {
            tiers.push_back(bits);
        }
    }

    return tiers;
}

// A tier that is not the default, so that switching precision stays observable
// even if RELAY_BP_PRECISION_TIERS has been trimmed down.
static int non_default_precision_tier(int max_bits = 512) {
    auto tiers = fast_precision_tiers(max_bits);

    for (auto it = tiers.rbegin(); it != tiers.rend(); ++it) {
        if (*it != DEFAULT_PRECISION) {
            return *it;
        }
    }

    return DEFAULT_PRECISION;
}

// ============================================================
// RelayBpDecoder: construction and validation
// ============================================================

TEST(RelayBpDecoder, InitializationWithExplicitMemoryStrengths) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);

    int maximum_legs = 3;
    int maximum_solutions = 2;
    int iterations0 = 7;
    int later_leg_iterations = 11;

    auto channel_probabilities = vector<double>(n, 0.1);
    auto memory_strengths =
        vector<vector<double>>(
            maximum_legs,
            vector<double>(n, 0.25)
        );

    memory_strengths[0] = vector<double>(n, 0.0);

    auto decoder = RelayBpDecoder(
        pcm,
        channel_probabilities,
        maximum_legs,
        maximum_solutions,
        iterations0,
        later_leg_iterations,
        123.0,                     // ignored because explicit memory is supplied
        {456.0, 789.0},            // ignored because explicit memory is supplied
        memory_strengths
    );

    // Inherited BpDecoder fields
    EXPECT_TRUE(pcm == decoder.pcm);
    EXPECT_EQ(pcm.m, decoder.check_count);
    EXPECT_EQ(pcm.n, decoder.bit_count);
    EXPECT_EQ(channel_probabilities, decoder.channel_probabilities);
    EXPECT_DOUBLE_EQ(1.0, decoder.ms_scaling_factor);
    EXPECT_EQ(ldpc::bp::PRODUCT_SUM, decoder.bp_method);
    EXPECT_EQ(ldpc::bp::PARALLEL, decoder.schedule);
    EXPECT_EQ(1, decoder.omp_thread_count);

    // RelayBpDecoder fields
    EXPECT_EQ(maximum_legs, decoder.maximum_legs);
    EXPECT_EQ(maximum_solutions, decoder.maximum_solutions);
    EXPECT_EQ(iterations0, decoder.iterations0);
    EXPECT_EQ(later_leg_iterations, decoder.maximum_iterations);
    EXPECT_EQ(memory_strengths, decoder.memory_strengths_per_leg);

    EXPECT_EQ(0, decoder.solution_number);
    EXPECT_EQ(0, decoder.iterations);
}

TEST(RelayBpDecoder, ExplicitMemoryMustHaveMaximumLegEntries) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto channel_probabilities = vector<double>(n, 0.1);

    // Only two entries supplied for a three-leg decoder.
    auto memory_strengths =
        vector<vector<double>>(2, vector<double>(n, 0.0));

    EXPECT_THROW(
        RelayBpDecoder(
            pcm,
            channel_probabilities,
            3,
            1,
            10,
            10,
            0.0,
            {},
            memory_strengths
        ),
        std::runtime_error
    );
}

TEST(RelayBpDecoder, ExplicitMemoryEntriesMustMatchBitCount) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto channel_probabilities = vector<double>(n, 0.1);

    auto memory_strengths = vector<vector<double>>{
        vector<double>(n, 0.0),
        vector<double>(n - 1, 0.5)
    };

    EXPECT_THROW(
        RelayBpDecoder(
            pcm,
            channel_probabilities,
            2,
            1,
            10,
            10,
            0.0,
            {},
            memory_strengths
        ),
        std::runtime_error
    );
}

TEST(RelayBpDecoder, AutoMemoryRequiresTwoElementInterval) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto channel_probabilities = vector<double>(n, 0.1);

    EXPECT_THROW(
        RelayBpDecoder(
            pcm,
            channel_probabilities,
            2,
            1,
            10,
            10,
            -1.0,
            {},                    // invalid when explicit memory is absent
            {}
        ),
        std::runtime_error
    );

    EXPECT_THROW(
        RelayBpDecoder(
            pcm,
            channel_probabilities,
            2,
            1,
            10,
            10,
            -1.0,
            {-0.5},                // must have exactly two elements
            {}
        ),
        std::runtime_error
    );
}

TEST(RelayBpDecoder, ConstructorRejectsInvalidPrecision) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto channel_probabilities = vector<double>(n, 0.1);
    auto memory = vector<vector<double>>(1, vector<double>(n, 0.0));

    auto construct_with_precision = [&](int precision) {
        return RelayBpDecoder(
            pcm,
            channel_probabilities,
            1,
            1,
            10,
            10,
            0.0,
            {},
            memory,
            ldpc::bp::PRODUCT_SUM,
            ldpc::bp::PARALLEL,
            1.0,
            1,
            ldpc::bp::NULL_INT_VECTOR,
            0,
            false,
            ldpc::bp::AUTO,
            -1,
            precision
        );
    };

    EXPECT_THROW(construct_with_precision(0), std::runtime_error);
    EXPECT_THROW(construct_with_precision(-8), std::runtime_error);
    EXPECT_THROW(
        construct_with_precision(available_precisions().back() + 1),
        std::runtime_error
    );
}

// ============================================================
// RelayBpDecoder: edge index over the parity check matrix
// ============================================================
//
// Messages are held in flat arrays indexed by edge rather than in the
// BpSparse entries, so that they can be carried at a precision the
// entries cannot store. These tests pin down the invariant the message
// passing relies on: that row_edges and col_edges reproduce exactly the
// orders iterate_row and iterate_column visit the non-zeros in.

// A 4 x 6 matrix with varying row and column degrees, so that the index is
// exercised on something less uniform than a repetition code.
static ldpc::bp::BpSparse make_irregular_pcm() {
    auto pcm = ldpc::bp::BpSparse(4, 6);

    const int coordinates[][2] = {
        {0, 0}, {0, 1}, {0, 3},
        {1, 1}, {1, 2}, {1, 4}, {1, 5},
        {2, 0}, {2, 2}, {2, 5},
        {3, 3}, {3, 4}
    };

    for (auto &coordinate : coordinates) {
        pcm.insert_entry(coordinate[0], coordinate[1]);
    }

    return pcm;
}

TEST(RelayBpDecoder, EdgeIndexMatchesRowTraversalOrder) {
    auto pcm = make_irregular_pcm();
    auto decoder = make_relay_decoder(pcm, pcm.n, 1, 1);

    int visited = 0;

    for (int i = 0; i < pcm.m; i++) {
        size_t position = 0;

        for (auto &e : decoder.pcm.iterate_row(i)) {
            ASSERT_LT(position, decoder.row_edges[i].size());

            int edge = decoder.row_edges[i][position];

            EXPECT_EQ(&e, decoder.edge_entry[edge]);
            EXPECT_EQ(i, decoder.edge_row[edge]);

            position++;
            visited++;
        }

        EXPECT_EQ(position, decoder.row_edges[i].size());
    }

    EXPECT_EQ(visited, decoder.edge_count);
}

TEST(RelayBpDecoder, EdgeIndexMatchesColumnTraversalOrder) {
    auto pcm = make_irregular_pcm();
    auto decoder = make_relay_decoder(pcm, pcm.n, 1, 1);

    auto times_indexed = vector<int>(decoder.edge_count, 0);

    for (int i = 0; i < pcm.n; i++) {
        size_t position = 0;

        for (auto &e : decoder.pcm.iterate_column(i)) {
            ASSERT_LT(position, decoder.col_edges[i].size());

            int edge = decoder.col_edges[i][position];

            EXPECT_EQ(&e, decoder.edge_entry[edge]);
            EXPECT_EQ(i, e.col_index);

            times_indexed[edge]++;
            position++;
        }

        EXPECT_EQ(position, decoder.col_edges[i].size());
    }

    // Every non-zero appears exactly once across all the columns, so the
    // column index is a permutation of the row index.
    for (int edge = 0; edge < decoder.edge_count; edge++) {
        EXPECT_EQ(1, times_indexed[edge]) << "edge " << edge;
    }
}

TEST(RelayBpDecoder, EdgeIndexLeavesMatrixMessagesZeroed) {
    // build_edge_index parks each edge id in the entry's own message field
    // while it runs; it must clear them again before returning.
    auto pcm = make_irregular_pcm();
    auto decoder = make_relay_decoder(pcm, pcm.n, 1, 1);

    for (int i = 0; i < pcm.m; i++) {
        for (auto &e : decoder.pcm.iterate_row(i)) {
            EXPECT_DOUBLE_EQ(0.0, e.bit_to_check_msg);
            EXPECT_DOUBLE_EQ(0.0, e.check_to_bit_msg);
        }
    }
}

TEST(RelayBpDecoder, RebuildingTheEdgeIndexReproducesIt) {
    auto pcm = make_irregular_pcm();
    auto decoder = make_relay_decoder(pcm, pcm.n, 1, 1);

    auto expected_row_edges = decoder.row_edges;
    auto expected_col_edges = decoder.col_edges;
    auto expected_edge_row = decoder.edge_row;
    int expected_edge_count = decoder.edge_count;

    // Decoding leaves messages in the entries; rebuilding must still work.
    auto syndrome = vector<uint8_t>(pcm.m, 0);
    decoder.decode(syndrome);

    decoder.build_edge_index();

    EXPECT_EQ(expected_edge_count, decoder.edge_count);
    EXPECT_EQ(expected_row_edges, decoder.row_edges);
    EXPECT_EQ(expected_col_edges, decoder.col_edges);
    EXPECT_EQ(expected_edge_row, decoder.edge_row);

    for (int i = 0; i < pcm.m; i++) {
        for (auto &e : decoder.pcm.iterate_row(i)) {
            EXPECT_DOUBLE_EQ(0.0, e.bit_to_check_msg);
        }
    }
}

// ============================================================
// RelayBpDecoder: memory strength generation
// ============================================================

TEST(RelayBpDecoder, ExplicitMemoryStrengthsTakePrecedence) {
    int n = 3;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);

    auto explicit_memory = vector<vector<double>>{
        {-1.0, -0.5, 0.0},
        {0.1, 0.2, 0.3}
    };

    auto decoder = RelayBpDecoder(
        pcm,
        vector<double>(n, 0.1),
        2,
        2,
        5,
        5,
        999.0,
        {1000.0, 2000.0},
        explicit_memory,
        ldpc::bp::PRODUCT_SUM,
        ldpc::bp::PARALLEL,
        1.0,
        1,
        ldpc::bp::NULL_INT_VECTOR,
        0,
        false,
        ldpc::bp::AUTO,
        1234
    );

    EXPECT_EQ(explicit_memory[0],
              decoder.generate_memory_strengths_for_leg(0));
    EXPECT_EQ(explicit_memory[1],
              decoder.generate_memory_strengths_for_leg(1));
}

TEST(RelayBpDecoder, AutoMemoryLeg0UsesGamma0) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);

    double gamma0 = -0.75;

    auto decoder = RelayBpDecoder(
        pcm,
        vector<double>(n, 0.1),
        3,
        3,
        10,
        10,
        gamma0,
        {-0.25, 0.5},
        {},
        ldpc::bp::PRODUCT_SUM,
        ldpc::bp::PARALLEL,
        1.0,
        1,
        ldpc::bp::NULL_INT_VECTOR,
        0,
        false,
        ldpc::bp::AUTO,
        123
    );

    EXPECT_EQ(
        vector<double>(n, gamma0),
        decoder.generate_memory_strengths_for_leg(0)
    );
}

TEST(RelayBpDecoder, AutoMemoryLaterLegsStayWithinInterval) {
    int n = 100;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);

    double lower = -0.4;
    double upper = 0.6;

    auto decoder = RelayBpDecoder(
        pcm,
        vector<double>(n, 0.1),
        2,
        2,
        10,
        10,
        -1.0,
        {lower, upper},
        {},
        ldpc::bp::PRODUCT_SUM,
        ldpc::bp::PARALLEL,
        1.0,
        1,
        ldpc::bp::NULL_INT_VECTOR,
        0,
        false,
        ldpc::bp::AUTO,
        123
    );

    auto strengths = decoder.generate_memory_strengths_for_leg(1);

    ASSERT_EQ(static_cast<size_t>(n), strengths.size());

    for (double strength : strengths) {
        EXPECT_GE(strength, lower);
        EXPECT_LE(strength, upper);
    }
}

TEST(RelayBpDecoder, AutoMemoryIsReproducibleWithFixedSeed) {
    int n = 20;
    auto pcm1 = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto pcm2 = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);

    auto decoder1 = RelayBpDecoder(
        pcm1,
        vector<double>(n, 0.1),
        2,
        2,
        10,
        10,
        -1.0,
        {-0.5, 0.8},
        {},
        ldpc::bp::PRODUCT_SUM,
        ldpc::bp::PARALLEL,
        1.0,
        1,
        ldpc::bp::NULL_INT_VECTOR,
        0,
        false,
        ldpc::bp::AUTO,
        9876
    );

    auto decoder2 = RelayBpDecoder(
        pcm2,
        vector<double>(n, 0.1),
        2,
        2,
        10,
        10,
        -1.0,
        {-0.5, 0.8},
        {},
        ldpc::bp::PRODUCT_SUM,
        ldpc::bp::PARALLEL,
        1.0,
        1,
        ldpc::bp::NULL_INT_VECTOR,
        0,
        false,
        ldpc::bp::AUTO,
        9876
    );

    EXPECT_EQ(
        decoder1.generate_memory_strengths_for_leg(1),
        decoder2.generate_memory_strengths_for_leg(1)
    );
}

TEST(RelayBpDecoder, SetMemorySeedRestartsRandomSequence) {
    int n = 20;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);

    auto decoder = RelayBpDecoder(
        pcm,
        vector<double>(n, 0.1),
        2,
        2,
        10,
        10,
        -1.0,
        {-0.5, 0.8},
        {},
        ldpc::bp::PRODUCT_SUM,
        ldpc::bp::PARALLEL,
        1.0,
        1,
        ldpc::bp::NULL_INT_VECTOR,
        0,
        false,
        ldpc::bp::AUTO,
        42
    );

    auto first = decoder.generate_memory_strengths_for_leg(1);

    decoder.set_memory_seed(42);

    auto repeated = decoder.generate_memory_strengths_for_leg(1);

    EXPECT_EQ(first, repeated);
}

// ============================================================
// RelayBpDecoder: initialise_log_domain_bp_relay
// ============================================================

TEST(RelayBpDecoder, InitialiseRelayLogDomainBpLeg0UsesChannelProbabilities) {
    int n = 3;
    auto pcm = ldpc::bp::BpSparse(n - 1, n);

    for (int i = 0; i < n - 1; i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }

    auto channel_probabilities = vector<double>{0.1, 0.2, 0.3};
    auto memory = vector<vector<double>>(1, vector<double>(n, 0.0));

    auto decoder = RelayBpDecoder(
        pcm,
        channel_probabilities,
        1,
        1,
        10,
        10,
        0.0,
        {},
        memory
    );

    // The working state is now passed in explicitly, at whatever precision the
    // messages are being carried in.
    auto initial_llrs = vector<double>(n, 0.0);
    auto llrs = vector<double>(n, 0.0);
    auto bit_to_check = vector<double>(decoder.edge_count, 0.0);

    decoder.initialise_log_domain_bp_relay<double>(
        0, initial_llrs, llrs, bit_to_check);

    for (int i = 0; i < n; i++) {
        double expected_llr =
            log((1.0 - channel_probabilities[i]) /
                channel_probabilities[i]);

        EXPECT_DOUBLE_EQ(expected_llr, initial_llrs[i]);
        EXPECT_DOUBLE_EQ(expected_llr, llrs[i]);

        for (int e : decoder.col_edges[i]) {
            EXPECT_DOUBLE_EQ(expected_llr, bit_to_check[e]);
        }
    }
}

TEST(RelayBpDecoder, InitialiseRelayLogDomainBpLaterLegKeepsChannelPrior) {
    // The refactored implementation does not carry the previous leg's
    // posterior LLR into initial_log_prob_ratios.
    int n = 3;
    auto pcm = ldpc::bp::BpSparse(n - 1, n);

    for (int i = 0; i < n - 1; i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }

    auto channel_probabilities = vector<double>{0.1, 0.2, 0.3};
    auto memory = vector<vector<double>>{
        vector<double>(n, 0.0),
        vector<double>(n, 0.5)
    };

    auto decoder = RelayBpDecoder(
        pcm,
        channel_probabilities,
        2,
        2,
        10,
        10,
        0.0,
        {},
        memory
    );

    auto initial_llrs = vector<double>(n, 0.0);
    auto llrs = vector<double>(n, 0.0);
    auto bit_to_check = vector<double>(decoder.edge_count, 0.0);

    decoder.initialise_log_domain_bp_relay<double>(
        0, initial_llrs, llrs, bit_to_check);

    auto expected_initial_llrs = initial_llrs;

    for (int i = 0; i < n; i++) {
        llrs[i] = static_cast<double>(i + 1) * 3.14;
    }

    decoder.initialise_log_domain_bp_relay<double>(
        1, initial_llrs, llrs, bit_to_check);

    EXPECT_EQ(expected_initial_llrs, initial_llrs);

    for (int i = 0; i < n; i++) {
        // initialise_log_domain_bp_relay(1) does not overwrite the
        // current posterior vector.
        EXPECT_DOUBLE_EQ(
            static_cast<double>(i + 1) * 3.14,
            llrs[i]
        );

        // However, the edge messages restart from the original prior.
        for (int e : decoder.col_edges[i]) {
            EXPECT_DOUBLE_EQ(
                expected_initial_llrs[i],
                bit_to_check[e]
            );
        }
    }
}

// ============================================================
// RelayBpDecoder: decoding_weight
// ============================================================

TEST(RelayBpDecoder, DecodingWeightZeroVector) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_relay_decoder(pcm, n, 1, 1);

    auto zero_decoding = vector<uint8_t>(n, 0);

    EXPECT_DOUBLE_EQ(
        0.0,
        decoder.decoding_weight(zero_decoding)
    );
}

TEST(RelayBpDecoder, DecodingWeightSingleBit) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_relay_decoder(pcm, n, 1, 1);

    auto decoding = vector<uint8_t>{1, 0, 0, 0, 0};
    double expected = log((1.0 - 0.1) / 0.1);

    EXPECT_DOUBLE_EQ(
        expected,
        decoder.decoding_weight(decoding)
    );
}

TEST(RelayBpDecoder, DecodingWeightAllOnes) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_relay_decoder(pcm, n, 1, 1);

    auto all_ones = vector<uint8_t>(n, 1);
    double expected =
        static_cast<double>(n) * log((1.0 - 0.1) / 0.1);

    EXPECT_DOUBLE_EQ(
        expected,
        decoder.decoding_weight(all_ones)
    );
}

TEST(RelayBpDecoder, LowerWeightMeansMoreLikelyForLowErrorRate) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_relay_decoder(pcm, n, 1, 1);

    auto zero = vector<uint8_t>(n, 0);
    auto single_error = vector<uint8_t>{1, 0, 0, 0, 0};
    auto all_ones = vector<uint8_t>(n, 1);

    EXPECT_LT(
        decoder.decoding_weight(zero),
        decoder.decoding_weight(single_error)
    );

    EXPECT_LT(
        decoder.decoding_weight(single_error),
        decoder.decoding_weight(all_ones)
    );
}

// ============================================================
// Single-leg decoding
// ============================================================

TEST(RelayBpDecoder, ProductSumSingleLegRepCode3) {
    int n = 3;
    auto pcm = ldpc::bp::BpSparse(n - 1, n);

    for (int i = 0; i < n - 1; i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }

    auto decoder = make_relay_decoder(
        pcm,
        n,
        1,
        1,
        0.0,
        10,
        10,
        ldpc::bp::PRODUCT_SUM
    );

    auto syndromes = vector<vector<uint8_t>>{
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    auto expected = vector<vector<uint8_t>>{
        {0, 0, 0},
        {0, 0, 1},
        {1, 0, 0},
        {0, 1, 0}
    };

    for (size_t i = 0; i < syndromes.size(); i++) {
        EXPECT_EQ(expected[i], decoder.decode(syndromes[i]));
        EXPECT_TRUE(decoder.converge);
        EXPECT_EQ(1, decoder.solution_number);
    }
}

TEST(RelayBpDecoder, ProductSumSingleLegRepCode5) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);

    auto decoder = make_relay_decoder(
        pcm,
        n,
        1,
        1,
        0.0,
        10,
        10,
        ldpc::bp::PRODUCT_SUM
    );

    auto syndromes = vector<vector<uint8_t>>{
        {0, 0, 0, 0},
        {0, 0, 0, 1},
        {0, 1, 0, 1},
        {1, 0, 1, 0},
        {1, 1, 1, 1}
    };

    auto expected = vector<vector<uint8_t>>{
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1},
        {0, 0, 1, 1, 0},
        {0, 1, 1, 0, 0},
        {0, 1, 0, 1, 0}
    };

    for (size_t i = 0; i < syndromes.size(); i++) {
        auto result = decoder.decode(syndromes[i]);

        EXPECT_EQ(expected[i], result);
        EXPECT_EQ(syndromes[i], pcm.mulvec(result));
    }
}

// ============================================================
// Relay-specific behavior
// ============================================================

TEST(RelayBpDecoder, MaximumSolutionsStopsAfterFirstConvergence) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);

    auto decoder = make_relay_decoder(
        pcm,
        n,
        3,
        1,
        0.5,
        10,
        10
    );

    auto syndrome = vector<uint8_t>(pcm.m, 0);
    auto decoding = decoder.decode(syndrome);

    EXPECT_TRUE(decoder.converge);
    EXPECT_EQ(1, decoder.solution_number);

    // An all-zero syndrome converges in the first iteration of leg 0.
    // Since maximum_solutions == 1, no later leg should execute.
    EXPECT_EQ(1, decoder.iterations);

    EXPECT_EQ(vector<uint8_t>(n, 0), decoding);
}

TEST(RelayBpDecoder, MultipleLegsAccumulateTotalIterations) {
    int n = 3;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);

    auto memory = vector<vector<double>>{
        {-1.0, -1.0, -1.0},
        {-0.345025519, 0.190321013, 0.575298233}
    };

    auto decoder = RelayBpDecoder(
        pcm,
        vector<double>(n, 0.003),
        2,
        2,
        80,
        60,
        0.0,
        {},
        memory,
        ldpc::bp::MINIMUM_SUM
    );

    auto syndrome = vector<uint8_t>{1, 1};
    auto decoding = decoder.decode(syndrome);

    EXPECT_GT(decoder.iterations, 0);
    EXPECT_LE(decoder.iterations, 80 + 60);

    if (decoder.solution_number > 0) {
        EXPECT_EQ(syndrome, pcm.mulvec(decoding));
    }
}

TEST(RelayBpDecoder, MultipleLegsWithMemoryReturnValidConvergedResult) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);

    auto memory = vector<vector<double>>{
        vector<double>(n, 0.0),
        vector<double>(n, 0.5),
        vector<double>(n, 0.9)
    };

    auto decoder = RelayBpDecoder(
        pcm,
        vector<double>(n, 0.1),
        3,
        3,
        10,
        10,
        0.0,
        {},
        memory
    );

    auto syndromes = vector<vector<uint8_t>>{
        {0, 0, 0, 0},
        {0, 0, 0, 1},
        {1, 0, 1, 0}
    };

    for (auto syndrome : syndromes) {
        auto decoding = decoder.decode(syndrome);

        ASSERT_GT(decoder.solution_number, 0);
        ASSERT_EQ(syndrome, pcm.mulvec(decoding));
        EXPECT_GT(decoder.iterations, 0);
        EXPECT_LE(decoder.iterations, 3 * 10);
    }
}

TEST(RelayBpDecoder, StateIsResetBetweenDecodeCalls) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);

    auto memory =
        vector<vector<double>>(1, vector<double>(n, 0.0));

    auto channel_probabilities = vector<double>(n, 0.1);

    auto decoder = RelayBpDecoder(
        pcm,
        channel_probabilities,
        1,
        1,
        5,
        5,
        0.0,
        {},
        memory
    );

    auto syndrome = vector<uint8_t>(pcm.m, 0);
    syndrome[0] = 1;

    decoder.decode(syndrome);

    ASSERT_TRUE(decoder.converge);
    ASSERT_EQ(1, decoder.solution_number);
    ASSERT_GT(decoder.iterations, 0);

    // Prevent leg 0 from running any iterations during the next call.
    decoder.iterations0 = 0;

    decoder.decode(syndrome);

    EXPECT_FALSE(decoder.converge);
    EXPECT_EQ(0, decoder.solution_number);
    EXPECT_EQ(0, decoder.iterations);
    EXPECT_EQ(vector<uint8_t>(n, 0), decoder.decoding);

    // initialise_log_domain_bp_relay(0) still initializes the LLR vectors
    // before the zero-iteration loop is entered.
    for (int i = 0; i < n; i++) {
        double expected =
            log((1.0 - channel_probabilities[i]) /
                channel_probabilities[i]);

        EXPECT_DOUBLE_EQ(
            expected,
            decoder.initial_log_prob_ratios[i]
        );

        EXPECT_DOUBLE_EQ(
            expected,
            decoder.log_prob_ratios[i]
        );
    }
}

TEST(RelayBpDecoder, ZeroIterationLegsDoNotConverge) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);

    auto memory = vector<vector<double>>{
        vector<double>(n, 0.0),
        vector<double>(n, 0.5)
    };

    auto decoder = RelayBpDecoder(
        pcm,
        vector<double>(n, 0.1),
        2,
        2,
        0,
        0,
        0.0,
        {},
        memory
    );

    auto syndrome = vector<uint8_t>(pcm.m, 0);
    auto result = decoder.decode(syndrome);

    EXPECT_FALSE(decoder.converge);
    EXPECT_EQ(0, decoder.solution_number);
    EXPECT_EQ(0, decoder.iterations);
    EXPECT_EQ(vector<uint8_t>(n, 0), result);
}

TEST(RelayBpDecoder, FinalLegBestEffortIsReturnedWhenNothingConverges) {
    int n = 3;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);

    auto memory = vector<vector<double>>{
        vector<double>(n, 0.0),
        vector<double>(n, 0.5)
    };

    auto decoder = RelayBpDecoder(
        pcm,
        vector<double>(n, 0.1),
        2,
        1,
        0,
        0,
        0.0,
        {},
        memory
    );

    auto syndrome = vector<uint8_t>{1, 1};
    auto result = decoder.decode(syndrome);

    EXPECT_EQ(0, decoder.solution_number);
    EXPECT_FALSE(decoder.converge);
    EXPECT_EQ(decoder.decoding, result);
    EXPECT_EQ(vector<uint8_t>(n, 0), result);
}

// ============================================================
// Message-passing precision
// ============================================================

// Bitwise comparison that treats two NaNs as equal, since the product-sum
// update can legitimately produce infinities and NaNs.
static void expect_identical_llrs(
    const vector<double> &expected,
    const vector<double> &actual
) {
    ASSERT_EQ(expected.size(), actual.size());

    for (size_t i = 0; i < expected.size(); i++) {
        if (std::isnan(expected[i]) && std::isnan(actual[i])) {
            continue;
        }

        EXPECT_EQ(expected[i], actual[i]) << "index " << i;
    }
}

TEST(RelayBpDecoder, DefaultPrecisionIsDouble) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto channel_probabilities = vector<double>(n, 0.1);
    auto memory = vector<vector<double>>(1, vector<double>(n, 0.0));

    // Constructed without naming a precision at all.
    auto decoder = RelayBpDecoder(
        pcm,
        channel_probabilities,
        1,
        1,
        10,
        10,
        0.0,
        {},
        memory
    );

    EXPECT_EQ(53, DEFAULT_PRECISION);
    EXPECT_EQ(DEFAULT_PRECISION, decoder.precision);
    EXPECT_EQ(DEFAULT_PRECISION, decoder.precision_request);
}

TEST(RelayBpDecoder, AvailablePrecisionsAreAscendingAndIncludeTheDefault) {
    auto tiers = available_precisions();

    ASSERT_FALSE(tiers.empty());

    for (size_t i = 1; i < tiers.size(); i++) {
        EXPECT_LT(tiers[i - 1], tiers[i]);
    }

    EXPECT_NE(
        std::find(tiers.begin(), tiers.end(), DEFAULT_PRECISION),
        tiers.end()
    );

    // 24 mantissa bits is IEEE single and is always available.
    EXPECT_NE(std::find(tiers.begin(), tiers.end(), 24), tiers.end());
}

TEST(RelayBpDecoder, ResolvePrecisionRoundsUpToATier) {
    auto tiers = available_precisions();

    // A request landing exactly on a tier is left alone.
    for (int bits : tiers) {
        EXPECT_EQ(bits, resolve_precision(bits));
    }

    // Anything below the smallest tier is promoted to it.
    EXPECT_EQ(tiers.front(), resolve_precision(1));

    // One bit past a tier moves up to the next one.
    for (size_t i = 1; i < tiers.size(); i++) {
        EXPECT_EQ(tiers[i], resolve_precision(tiers[i - 1] + 1));
    }
}

TEST(RelayBpDecoder, ResolvePrecisionRejectsOutOfRangeRequests) {
    EXPECT_THROW(resolve_precision(0), std::runtime_error);
    EXPECT_THROW(resolve_precision(-1), std::runtime_error);
    EXPECT_THROW(
        resolve_precision(available_precisions().back() + 1),
        std::runtime_error
    );
}

TEST(RelayBpDecoder, ConstructorRecordsRequestedAndResolvedPrecision) {
    int n = 5;
    auto tiers = available_precisions();

    // A request that falls between two tiers, if this build has one.
    int requested = tiers.front() + 1;

    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder =
        make_relay_decoder(pcm, n, 1, 1, 0.0, 10, 10,
                           ldpc::bp::PRODUCT_SUM, requested);

    EXPECT_EQ(requested, decoder.precision_request);
    EXPECT_EQ(resolve_precision(requested), decoder.precision);
    EXPECT_GE(decoder.precision, requested);
}

TEST(RelayBpDecoder, SetPrecisionChangesTheTierInUse) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_relay_decoder(pcm, n, 1, 1);

    ASSERT_EQ(DEFAULT_PRECISION, decoder.precision);

    int target = non_default_precision_tier();
    decoder.set_precision(target);

    EXPECT_EQ(target, decoder.precision);
    EXPECT_EQ(target, decoder.precision_request);

    decoder.set_precision(DEFAULT_PRECISION);

    EXPECT_EQ(DEFAULT_PRECISION, decoder.precision);
}

TEST(RelayBpDecoder, RejectedSetPrecisionLeavesTheDecoderUnchanged) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_relay_decoder(pcm, n, 1, 1);

    EXPECT_THROW(decoder.set_precision(0), std::runtime_error);
    EXPECT_THROW(
        decoder.set_precision(available_precisions().back() + 1),
        std::runtime_error
    );

    EXPECT_EQ(DEFAULT_PRECISION, decoder.precision);
    EXPECT_EQ(DEFAULT_PRECISION, decoder.precision_request);

    // Still usable afterwards.
    auto syndrome = vector<uint8_t>(pcm.m, 0);
    auto decoding = decoder.decode(syndrome);

    EXPECT_TRUE(decoder.converge);
    EXPECT_EQ(vector<uint8_t>(n, 0), decoding);
}

TEST(RelayBpDecoder, ExplicitDoublePrecisionMatchesTheDefaultExactly) {
    // The 53-bit tier must be the same arithmetic as the unparameterised
    // decoder, bit for bit, on every output.
    int n = 5;

    auto default_pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto explicit_pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);

    auto default_decoder =
        make_relay_decoder(default_pcm, n, 3, 3, 0.5, 10, 10);
    auto explicit_decoder =
        make_relay_decoder(explicit_pcm, n, 3, 3, 0.5, 10, 10,
                           ldpc::bp::PRODUCT_SUM, 53);

    auto syndromes = vector<vector<uint8_t>>{
        {0, 0, 0, 0},
        {0, 0, 0, 1},
        {0, 1, 0, 1},
        {1, 0, 1, 0},
        {1, 1, 1, 1}
    };

    for (auto syndrome : syndromes) {
        auto default_syndrome = syndrome;
        auto explicit_syndrome = syndrome;

        auto from_default = default_decoder.decode(default_syndrome);
        auto from_explicit = explicit_decoder.decode(explicit_syndrome);

        EXPECT_EQ(from_default, from_explicit);
        EXPECT_EQ(default_decoder.converge, explicit_decoder.converge);
        EXPECT_EQ(default_decoder.iterations, explicit_decoder.iterations);
        EXPECT_EQ(default_decoder.solution_number,
                  explicit_decoder.solution_number);

        expect_identical_llrs(default_decoder.log_prob_ratios,
                              explicit_decoder.log_prob_ratios);
    }
}

TEST(RelayBpDecoder, EveryPrecisionTierCanDecode) {
    // Reaching every tier proves the runtime dispatch covers each precision
    // that available_precisions() advertises. Correctness is only required of
    // double and above: producing worse answers at low precision is the point
    // of the parameter, not a defect.
    int n = 3;

    auto syndromes = vector<vector<uint8_t>>{
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    for (int bits : available_precisions()) {
        auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
        auto decoder =
            make_relay_decoder(pcm, n, 1, 1, 0.0, 30, 30,
                               ldpc::bp::PRODUCT_SUM, bits);

        ASSERT_EQ(bits, decoder.precision);

        for (auto syndrome : syndromes) {
            auto decoding = decoder.decode(syndrome);

            ASSERT_EQ(static_cast<size_t>(n), decoding.size());

            if (bits >= DEFAULT_PRECISION) {
                EXPECT_TRUE(decoder.converge) << "precision " << bits;
                EXPECT_EQ(syndrome, pcm.mulvec(decoding))
                    << "precision " << bits;
            } else if (decoder.converge) {
                EXPECT_EQ(syndrome, pcm.mulvec(decoding))
                    << "precision " << bits;
            }
        }
    }
}

TEST(RelayBpDecoder, EveryPrecisionTierCanDecodeWithMinimumSum) {
    int n = 3;

    auto syndromes = vector<vector<uint8_t>>{
        {0, 1},
        {1, 0},
        {1, 1}
    };

    for (int bits : available_precisions()) {
        auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
        auto decoder =
            make_relay_decoder(pcm, n, 1, 1, 0.0, 30, 30,
                               ldpc::bp::MINIMUM_SUM, bits);

        for (auto syndrome : syndromes) {
            auto decoding = decoder.decode(syndrome);

            ASSERT_EQ(static_cast<size_t>(n), decoding.size());

            if (bits >= DEFAULT_PRECISION) {
                EXPECT_TRUE(decoder.converge) << "precision " << bits;
                EXPECT_EQ(syndrome, pcm.mulvec(decoding))
                    << "precision " << bits;
            }
        }
    }
}

TEST(RelayBpDecoder, SerialScheduleDecodesAtEveryPrecision) {
    // The serial path is templated separately from the parallel one, so it
    // needs its own sweep over the tiers.
    int n = 5;

    auto syndromes = vector<vector<uint8_t>>{
        {0, 0, 0, 1},
        {1, 0, 1, 0}
    };

    for (int bits : fast_precision_tiers()) {
        auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
        auto decoder =
            make_relay_decoder(pcm, n, 2, 2, 0.5, 20, 20,
                               ldpc::bp::MINIMUM_SUM, bits,
                               ldpc::bp::SERIAL);

        for (auto syndrome : syndromes) {
            auto decoding = decoder.decode(syndrome);

            ASSERT_EQ(static_cast<size_t>(n), decoding.size());

            if (bits >= DEFAULT_PRECISION) {
                EXPECT_TRUE(decoder.converge) << "precision " << bits;
                EXPECT_EQ(syndrome, pcm.mulvec(decoding))
                    << "precision " << bits;
            }
        }
    }
}

TEST(RelayBpDecoder, MaximumPrecisionTierDecodes) {
    int n = 3;
    int bits = available_precisions().back();

    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder =
        make_relay_decoder(pcm, n, 1, 1, 0.0, 20, 20,
                           ldpc::bp::PRODUCT_SUM, bits);

    ASSERT_EQ(bits, decoder.precision);

    auto syndrome = vector<uint8_t>{1, 1};
    auto decoding = decoder.decode(syndrome);

    EXPECT_TRUE(decoder.converge);
    EXPECT_EQ(syndrome, pcm.mulvec(decoding));
}

TEST(RelayBpDecoder, InheritedDoubleStateIsPopulatedAtEveryPrecision) {
    // Whatever the working precision, the double members inherited from
    // BpDecoder must be filled in, since that is all the Python wrappers and
    // the post-processors ever see.
    int n = 5;
    double error_rate = 0.1;

    double expected_llr = log((1.0 - error_rate) / error_rate);

    for (int bits : fast_precision_tiers()) {
        auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
        auto decoder =
            make_relay_decoder(pcm, n, 1, 1, 0.0, 10, 10,
                               ldpc::bp::PRODUCT_SUM, bits);

        auto syndrome = vector<uint8_t>{0, 0, 0, 1};
        decoder.decode(syndrome);

        // The LLR is computed at the working precision and then narrowed, so
        // it is only correct to that precision: at 11 mantissa bits the right
        // answer really is 2.197265625. Allow a few ulps of the tier in use,
        // never finer than a few ulps of double, since above 53 bits it is
        // std::log rather than the decoder that carries the rounding error.
        double tolerance =
            std::abs(expected_llr) *
            std::ldexp(1.0, -(std::min(bits, DEFAULT_PRECISION) - 3));

        for (int i = 0; i < n; i++) {
            EXPECT_NEAR(expected_llr,
                        decoder.initial_log_prob_ratios[i],
                        tolerance)
                << "precision " << bits << ", bit " << i;

            EXPECT_FALSE(std::isnan(decoder.log_prob_ratios[i]))
                << "precision " << bits << ", bit " << i;
        }
    }
}

TEST(RelayBpDecoder, LowPrecisionTiersReallyNarrowTheArithmetic) {
    // Guards against the dispatch quietly running everything in double: below
    // 53 bits the results must be exactly representable in the requested
    // number of mantissa bits, and must differ from the double answer.
    int n = 5;
    double error_rate = 0.1;
    double double_llr = log((1.0 - error_rate) / error_rate);

    auto schedules = vector<ldpc::bp::BpSchedule>{
        ldpc::bp::PARALLEL,
        ldpc::bp::SERIAL
    };

    for (int bits : available_precisions()) {
        if (bits >= DEFAULT_PRECISION) {
            continue;
        }

        for (auto schedule : schedules) {
            auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
            auto decoder =
                make_relay_decoder(pcm, n, 1, 1, 0.0, 10, 10,
                                   ldpc::bp::MINIMUM_SUM, bits, schedule);

            auto syndrome = vector<uint8_t>{0, 0, 0, 1};
            decoder.decode(syndrome);

            double value = decoder.initial_log_prob_ratios[0];

            EXPECT_NE(double_llr, value)
                << "precision " << bits << " gave the double result";

            // Exactly representable in `bits` mantissa bits.
            int exponent = 0;
            double scaled = std::ldexp(std::frexp(value, &exponent), bits);

            EXPECT_DOUBLE_EQ(std::trunc(scaled), scaled)
                << "precision " << bits << " carried more than " << bits
                << " mantissa bits";
        }
    }
}

TEST(RelayBpDecoder, MatrixMessagesAreWrittenBackAtEveryPrecision) {
    int n = 5;

    for (int bits : fast_precision_tiers()) {
        auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
        auto decoder =
            make_relay_decoder(pcm, n, 1, 1, 0.0, 10, 10,
                               ldpc::bp::PRODUCT_SUM, bits);

        auto syndrome = vector<uint8_t>{0, 0, 0, 1};
        decoder.decode(syndrome);

        bool any_message_set = false;

        for (int i = 0; i < pcm.m; i++) {
            for (auto &e : decoder.pcm.iterate_row(i)) {
                if (e.bit_to_check_msg != 0.0) {
                    any_message_set = true;
                }

                EXPECT_FALSE(std::isnan(e.bit_to_check_msg))
                    << "precision " << bits;
            }
        }

        EXPECT_TRUE(any_message_set) << "precision " << bits;
    }
}

TEST(RelayBpDecoder, SwitchingPrecisionMatchesAFreshlyBuiltDecoder) {
    // Precision is read per decode call, so changing it must be equivalent to
    // having constructed the decoder at the new precision in the first place.
    int n = 5;
    int target = non_default_precision_tier();

    auto syndrome = vector<uint8_t>{0, 1, 0, 1};

    auto reused_pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto reused = make_relay_decoder(reused_pcm, n, 2, 2, 0.5, 10, 10);

    auto warm_up_syndrome = vector<uint8_t>{1, 1, 1, 1};
    reused.decode(warm_up_syndrome);
    reused.set_precision(target);

    auto switched_syndrome = syndrome;
    auto from_switched = reused.decode(switched_syndrome);

    auto fresh_pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto fresh = make_relay_decoder(fresh_pcm, n, 2, 2, 0.5, 10, 10,
                                    ldpc::bp::PRODUCT_SUM, target);

    auto fresh_syndrome = syndrome;
    auto from_fresh = fresh.decode(fresh_syndrome);

    EXPECT_EQ(from_fresh, from_switched);
    EXPECT_EQ(fresh.converge, reused.converge);
    EXPECT_EQ(fresh.iterations, reused.iterations);
    EXPECT_EQ(fresh.solution_number, reused.solution_number);

    expect_identical_llrs(fresh.log_prob_ratios, reused.log_prob_ratios);
}

TEST(RelayBpDecoder, DecodingWeightDefaultsToDoubleAndAcceptsOtherPrecisions) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_relay_decoder(pcm, n, 1, 1);

    auto decoding = vector<uint8_t>{1, 0, 1, 0, 0};
    double expected = 2.0 * log((1.0 - 0.1) / 0.1);

    // Called without a template argument, exactly as before.
    double as_double = decoder.decoding_weight(decoding);

    EXPECT_DOUBLE_EQ(expected, as_double);
    EXPECT_DOUBLE_EQ(as_double, decoder.decoding_weight<double>(decoding));

    // And at a wider precision, agreeing to within double rounding.
    using WideFloat = PrecisionTraits<128>::type;
    double as_wide = static_cast<double>(
        decoder.decoding_weight<WideFloat>(decoding));

    EXPECT_NEAR(expected, as_wide, 1e-12);
}

// ============================================================
// Unimplemented methods
// ============================================================


TEST(RelayBpDecoder, BpDecodeSingleScanThrowsNotImplemented) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_relay_decoder(pcm, n, 1, 1);
    auto syndrome = vector<uint8_t>(pcm.m, 0);

    EXPECT_THROW(
        decoder.bp_decode_single_scan(syndrome),
        std::logic_error
    );
}

TEST(RelayBpDecoder, SoftInfoDecodeSerialThrowsNotImplemented) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_relay_decoder(pcm, n, 1, 1);
    auto soft_syndrome = vector<double>(pcm.m, 0.5);

    EXPECT_THROW(
        decoder.soft_info_decode_serial(
            soft_syndrome,
            1.0,
            1.0
        ),
        std::logic_error
    );
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
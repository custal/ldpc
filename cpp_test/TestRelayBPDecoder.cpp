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
    ldpc::bp::BpMethod method = ldpc::bp::PRODUCT_SUM
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
        method
    );
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

    decoder.initialise_log_domain_bp_relay(0);

    for (int i = 0; i < n; i++) {
        double expected_llr =
            log((1.0 - channel_probabilities[i]) /
                channel_probabilities[i]);

        EXPECT_DOUBLE_EQ(expected_llr,
                         decoder.initial_log_prob_ratios[i]);
        EXPECT_DOUBLE_EQ(expected_llr,
                         decoder.log_prob_ratios[i]);

        for (auto &e : decoder.pcm.iterate_column(i)) {
            EXPECT_DOUBLE_EQ(expected_llr, e.bit_to_check_msg);
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

    decoder.initialise_log_domain_bp_relay(0);

    auto expected_initial_llrs = decoder.initial_log_prob_ratios;

    for (int i = 0; i < n; i++) {
        decoder.log_prob_ratios[i] =
            static_cast<double>(i + 1) * 3.14;
    }

    decoder.initialise_log_domain_bp_relay(1);

    EXPECT_EQ(expected_initial_llrs,
              decoder.initial_log_prob_ratios);

    for (int i = 0; i < n; i++) {
        // initialise_log_domain_bp_relay(1) does not overwrite the
        // current posterior vector.
        EXPECT_DOUBLE_EQ(
            static_cast<double>(i + 1) * 3.14,
            decoder.log_prob_ratios[i]
        );

        // However, the edge messages restart from the original prior.
        for (auto &e : decoder.pcm.iterate_column(i)) {
            EXPECT_DOUBLE_EQ(
                expected_initial_llrs[i],
                e.bit_to_check_msg
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
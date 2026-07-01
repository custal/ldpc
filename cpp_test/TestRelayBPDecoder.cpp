// TestRelayBPDecoder.cpp
//
// Tests for RelayBpDecoder (relay_bp.hpp).

#include <gtest/gtest.h>
#include "relay_bp.hpp"
#include "gf2codes.hpp"
#include "ldpc.hpp"
#include <cmath>
#include <numeric>

using namespace std;
using namespace ldpc::relay;

// ============================================================
// Test helpers
// ============================================================

// Builds a relay decoder for a given parity-check matrix and bit count.
// All legs use the same per-iteration limit. memory_strength is applied uniformly to
// all bits for all legs; leg 0 is always forced to 0.0 (pure channel LLR start).
static RelayBpDecoder make_relay_decoder(
    ldpc::bp::BpSparse &pcm,
    int n,
    int maximum_legs,
    int maximum_solutions,
    double uniform_memory_strength = 0.0,
    int iters_per_leg = 10,
    ldpc::bp::BpMethod method = ldpc::bp::PRODUCT_SUM
) {
    auto channel_probabilities = vector<double>(n, 0.1);
    auto max_iters = vector<int>(maximum_legs, iters_per_leg);
    // Build per-leg memory vectors: leg 0 always has zero memory
    auto mem_strengths = vector<vector<double>>(maximum_legs, vector<double>(n, uniform_memory_strength));
    mem_strengths[0] = vector<double>(n, 0.0);

    return RelayBpDecoder(pcm, channel_probabilities, maximum_legs, maximum_solutions,
                          max_iters, mem_strengths, 0, method);
}

// ============================================================
// RelayBpDecoder: initialization
// ============================================================

TEST(RelayBpDecoder, InitializationTest) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    int maximum_legs = 3;
    int maximum_solutions = 2;
    auto channel_probabilities = vector<double>(n, 0.1);
    auto max_iters = vector<int>(maximum_legs, 10);
    auto mem_strengths = vector<vector<double>>(maximum_legs, vector<double>(n, 0.0));

    auto decoder = RelayBpDecoder(pcm, channel_probabilities, maximum_legs, maximum_solutions,
                                  max_iters, mem_strengths);

    // Inherited BpDecoder fields
    EXPECT_TRUE(pcm == decoder.pcm);
    EXPECT_EQ(pcm.m, decoder.check_count);
    EXPECT_EQ(pcm.n, decoder.bit_count);
    EXPECT_EQ(channel_probabilities, decoder.channel_probabilities);
    EXPECT_EQ(1.0, decoder.ms_scaling_factor);
    EXPECT_EQ(ldpc::bp::PRODUCT_SUM, decoder.bp_method);
    EXPECT_EQ(ldpc::bp::PARALLEL, decoder.schedule);
    EXPECT_EQ(1, decoder.omp_thread_count);

    // RelayBpDecoder-specific fields
    EXPECT_EQ(maximum_legs, decoder.maximum_legs);
    EXPECT_EQ(maximum_solutions, decoder.maximum_solutions);
    EXPECT_EQ(max_iters, decoder.maximum_iterations_per_leg);
    EXPECT_EQ(mem_strengths, decoder.memory_strengths_per_leg);
    EXPECT_EQ(0, decoder.solution_number);

    // Per-leg tracking vectors are sized correctly and start unconverged
    ASSERT_EQ(maximum_legs, (int)decoder.convergence_per_leg.size());
    ASSERT_EQ(maximum_legs, (int)decoder.iterations_per_leg.size());
    ASSERT_EQ(maximum_legs, (int)decoder.decoding_per_leg.size());
    ASSERT_EQ(maximum_legs, (int)decoder.log_prob_ratios_per_leg.size());
    for (int leg = 0; leg < maximum_legs; leg++) {
        EXPECT_FALSE(decoder.convergence_per_leg[leg]);
        EXPECT_EQ(0, decoder.iterations_per_leg[leg]);
    }
}

// ============================================================
// RelayBpDecoder: initialise_log_domain_bp_relay
// Note: the renamed method from the code review; originally 'initialise_log_domain_bp(int leg)'
// ============================================================

TEST(RelayBpDecoder, InitialiseRelayLogDomainBp_Leg0_UsesChannelProbabilities) {
    int n = 3;
    auto pcm = ldpc::bp::BpSparse(n - 1, n);
    for (int i = 0; i < n - 1; i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    auto channel_probabilities = vector<double>{0.1, 0.2, 0.3};
    auto mem = vector<vector<double>>(1, vector<double>(n, 0.0));

    auto decoder = RelayBpDecoder(pcm, channel_probabilities, 1, 1,
                                  vector<int>{10}, mem);

    decoder.initialise_log_domain_bp_relay(0);

    for (int i = 0; i < n; i++) {
        double expected_llr = log((1 - channel_probabilities[i]) / channel_probabilities[i]);
        EXPECT_DOUBLE_EQ(expected_llr, decoder.initial_log_prob_ratios[i]);
        for (auto &e: decoder.pcm.iterate_column(i)) {
            EXPECT_DOUBLE_EQ(decoder.initial_log_prob_ratios[i], e.bit_to_check_msg);
        }
    }
}

TEST(RelayBpDecoder, InitialiseRelayLogDomainBp_LegN_CarriesOverLogProbRatios) {
    // For leg > 0, initial_log_prob_ratios should be set to the current log_prob_ratios
    // (the final state from the previous leg), not recomputed from channel probabilities.
    int n = 3;
    auto pcm = ldpc::bp::BpSparse(n - 1, n);
    for (int i = 0; i < n - 1; i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    auto mem = vector<vector<double>>(2, vector<double>(n, 0.0));

    auto decoder = RelayBpDecoder(pcm, vector<double>(n, 0.1), 2, 1,
                                  vector<int>{10, 10}, mem);

    // Run leg 0 to produce non-trivial log_prob_ratios
    auto syndrome = vector<uint8_t>{0, 0};
    decoder.decode(syndrome);

    // Manually set recognisable sentinel values so we can verify carry-over
    for (int i = 0; i < n; i++) {
        decoder.log_prob_ratios[i] = static_cast<double>(i + 1) * 3.14;
    }

    decoder.initialise_log_domain_bp_relay(1);

    for (int i = 0; i < n; i++) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1) * 3.14, decoder.initial_log_prob_ratios[i]);
    }
}

// ============================================================
// RelayBpDecoder: decoding_weight
// ============================================================

TEST(RelayBpDecoder, DecodingWeight_ZeroVector) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_relay_decoder(pcm, n, 1, 1);

    auto zero_decoding = vector<uint8_t>(n, 0);
    EXPECT_DOUBLE_EQ(0.0, decoder.decoding_weight(zero_decoding));
}

TEST(RelayBpDecoder, DecodingWeight_SingleBit) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    double p = 0.1;
    auto decoder = make_relay_decoder(pcm, n, 1, 1);

    // Only bit 0 set: weight = log((1-p)/p)
    auto decoding = vector<uint8_t>{1, 0, 0, 0, 0};
    double expected = log((1 - p) / p);
    EXPECT_DOUBLE_EQ(expected, decoder.decoding_weight(decoding));
}

TEST(RelayBpDecoder, DecodingWeight_AllOnes) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    double p = 0.1;
    auto decoder = make_relay_decoder(pcm, n, 1, 1);

    auto all_ones = vector<uint8_t>(n, 1);
    double expected = n * log((1 - p) / p);
    EXPECT_DOUBLE_EQ(expected, decoder.decoding_weight(all_ones));
}

TEST(RelayBpDecoder, DecodingWeight_ZeroIsLowestForLowErrorRate) {
    // With p=0.1 (low error rate), log((1-p)/p) > 0, so all-zeros weight (0) is lower
    // than any non-zero decoding. Lower weight = higher likelihood under this channel.
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_relay_decoder(pcm, n, 1, 1);

    auto zero_decoding = vector<uint8_t>(n, 0);
    auto single_error  = vector<uint8_t>{1, 0, 0, 0, 0};
    auto all_ones      = vector<uint8_t>(n, 1);

    EXPECT_LT(decoder.decoding_weight(zero_decoding), decoder.decoding_weight(single_error));
    EXPECT_LT(decoder.decoding_weight(single_error),  decoder.decoding_weight(all_ones));
}

// ============================================================
// RelayBpDecoder: decode correctness — single leg, zero memory
// (should match base BpDecoder behaviour exactly)
// ============================================================

TEST(RelayBpDecoder, ProdSumParallel_SingleLeg_RepCode3) {
    int n = 3;
    auto pcm = ldpc::bp::BpSparse(n - 1, n);
    for (int i = 0; i < n - 1; i++) {
        pcm.insert_entry(i, i);
        pcm.insert_entry(i, (i + 1) % n);
    }
    auto decoder = make_relay_decoder(pcm, n, 1, 1);

    auto syndromes         = vector<vector<uint8_t>>{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    auto expected_decoding = vector<vector<uint8_t>>{{0,0,0}, {0,0,1}, {1,0,0}, {0,1,0}};

    for (int i = 0; i < (int)syndromes.size(); i++) {
        ASSERT_EQ(expected_decoding[i], decoder.decode(syndromes[i]));
    }
}

TEST(RelayBpDecoder, ProdSumParallel_SingleLeg_RepCode5) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_relay_decoder(pcm, n, 1, 1);

    auto syndromes = vector<vector<uint8_t>>{{0,0,0,0}, {0,0,0,1}, {0,1,0,1},
                                             {1,0,1,0}, {1,1,1,1}};
    auto expected = vector<vector<uint8_t>>{{0,0,0,0,0}, {0,0,0,0,1}, {0,0,1,1,0},
                                            {0,1,1,0,0}, {0,1,0,1,0}};

    for (int i = 0; i < (int)syndromes.size(); i++) {
        ASSERT_EQ(expected[i], decoder.decode(syndromes[i]));
    }
}

TEST(RelayBpDecoder, ProductSumParallel_SingleLeg_RepCode5) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_relay_decoder(pcm, n, 1, 1, 0.0, 10, ldpc::bp::PRODUCT_SUM);

    auto syndromes = vector<vector<uint8_t>>{{0,0,0,0}, {0,0,0,1}, {0,1,0,1},
                                             {1,0,1,0}, {1,1,1,1}};
    auto expected = vector<vector<uint8_t>>{{0,0,0,0,0}, {0,0,0,0,1}, {0,0,1,1,0},
                                            {0,1,1,0,0}, {0,1,0,1,0}};

    for (int i = 0; i < (int)syndromes.size(); i++) {
        ASSERT_EQ(expected[i], decoder.decode(syndromes[i]));
    }
}

// ============================================================
// RelayBpDecoder: relay-specific behaviour
// ============================================================

TEST(RelayBpDecoder, ConvergenceAndIterationsTrackedPerLeg) {
    // A trivial all-zero syndrome should converge on leg 0 in 1 iteration.
    // With maximum_solutions=1, leg 1 should never run.
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_relay_decoder(pcm, n, 2, 1);

    auto syndrome = vector<uint8_t>(pcm.m, 0);
    decoder.decode(syndrome);

    EXPECT_TRUE(decoder.convergence_per_leg[0]);
    EXPECT_GT(decoder.iterations_per_leg[0], 0);

    // Leg 1 should not have been processed (maximum_solutions reached after leg 0)
    EXPECT_FALSE(decoder.convergence_per_leg[1]);
    EXPECT_EQ(0, decoder.iterations_per_leg[1]);

    EXPECT_EQ(1, decoder.solution_number);
}

TEST(RelayBpDecoder, BestDecodingMatchesConvergedLeg) {
    // When only one leg converges, returned decoding should equal that leg's decoding.
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_relay_decoder(pcm, n, 1, 1);

    auto syndrome = vector<uint8_t>{0, 0, 0, 1};  // single error at bit 4
    auto result = decoder.decode(syndrome);

    EXPECT_TRUE(decoder.convergence_per_leg[0]);
    ASSERT_EQ(decoder.decoding_per_leg[0], decoder.decoding);
    ASSERT_EQ(result, decoder.decoding);
}

TEST(RelayBpDecoder, MaximumSolutionsEarlyExit) {
    // With maximum_solutions=1 and leg 0 converging, legs 1 and 2 should never run.
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_relay_decoder(pcm, n, 3, 1);

    auto syndrome = vector<uint8_t>(pcm.m, 0);
    decoder.decode(syndrome);

    EXPECT_TRUE(decoder.convergence_per_leg[0]);
    EXPECT_FALSE(decoder.convergence_per_leg[1]);
    EXPECT_FALSE(decoder.convergence_per_leg[2]);
    EXPECT_EQ(0, decoder.iterations_per_leg[1]);
    EXPECT_EQ(0, decoder.iterations_per_leg[2]);
}

// [FIXED BUG] Verifies the state-reset fix. Without the fix, stale convergence flags from
// the first decode call would cause the second call to immediately exit (solution_number
// would already equal maximum_solutions), and convergence_per_leg[0] would not be updated.
TEST(RelayBpDecoder, StateResetBetweenDecodeCalls) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);

    // Single leg, 5 iterations: enough to converge on a trivial syndrome.
    auto channel_probabilities = vector<double>(n, 0.1);
    auto decoder = RelayBpDecoder(pcm, channel_probabilities, 1, 1,
                                  vector<int>{5},
                                  vector<vector<double>>(1, vector<double>(n, 0.0)));

    // First decode: trivial syndrome — should converge.
    auto syndrome = vector<uint8_t>(pcm.m, 0);
    syndrome[0] = 1;
    decoder.decode(syndrome);
    ASSERT_TRUE(decoder.convergence_per_leg[0]);
    EXPECT_NE(decoder.iterations_per_leg[0], 0);
    EXPECT_NE(decoder.solution_number, 0);
    EXPECT_NE(decoder.decoding, vector<uint8_t>(pcm.n, 0));
    EXPECT_NE(decoder.decoding_per_leg[0], vector<uint8_t>(pcm.n, 0));
    EXPECT_NE(decoder.log_prob_ratios_per_leg[0], vector<double>(pcm.n, 0));

    // Now reduce max iterations to 0 so the next call cannot converge even if it runs.
    decoder.maximum_iterations_per_leg[0] = 0;

    // Second decode: without the state-reset fix, solution_number would already equal
    // maximum_solutions (1) from the first call, and the loop would break immediately —
    // leaving convergence_per_leg[0] stale (true). With the fix, it is reset and the leg
    // runs (completing 0 iterations), leaving convergence_per_leg[0] correctly false.
    decoder.decode(syndrome);
    EXPECT_FALSE(decoder.convergence_per_leg[0]);
    EXPECT_EQ(decoder.iterations_per_leg[0], 0);
    EXPECT_EQ(decoder.solution_number, 0);
    EXPECT_EQ(decoder.decoding, vector<uint8_t>(pcm.n, 0));
    EXPECT_EQ(decoder.decoding_per_leg[0], vector<uint8_t>(pcm.n, 0));

    auto initial_log_prob_ratios = vector<double>(pcm.n, 0);
    for (int i = 0; i < pcm.n; i++) {
        initial_log_prob_ratios[i] = std::log(
            (1 - channel_probabilities[i]) / channel_probabilities[i]);
    }
    EXPECT_EQ(decoder.log_prob_ratios_per_leg[0], initial_log_prob_ratios);
}

TEST(RelayBpDecoder, MultipleLegsWithMemory_ConvergesCorrectly) {
    // Verify that multi-leg decoding with non-zero memory still produces a valid codeword
    // on a simple syndrome. We don't pin the exact decoding (it depends on LLR propagation
    // across legs), but we verify that the decoded result satisfies the syndrome.
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto channel_probabilities = vector<double>(n, 0.1);

    // Three legs; legs 1 and 2 use 0.5 memory strength.
    auto mem = vector<vector<double>>{
        vector<double>(n, 0.0),   // leg 0: pure channel
        vector<double>(n, 0.5),   // leg 1: 50% memory
        vector<double>(n, 0.9),   // leg 2: 90% memory
    };

    auto decoder = RelayBpDecoder(pcm, channel_probabilities, 3, 3,
                                  vector<int>{10, 10, 10}, mem);

    auto syndromes = vector<vector<uint8_t>>{{0,0,0,0}, {0,0,0,1}, {1,0,1,0}};

    for (auto syndrome : syndromes) {
        auto decoding = decoder.decode(syndrome);
        // Check that the decoding is a valid codeword: PCM * decoding == syndrome (mod 2)
        auto candidate = pcm.mulvec(decoding);
        ASSERT_EQ(syndrome, candidate)
            << "Decoded result does not satisfy the syndrome";
    }
}

// ============================================================
// RelayBpDecoder: unimplemented methods
// ============================================================

TEST(RelayBpDecoder, BpDecodeSerialThrowsNotImplemented) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_relay_decoder(pcm, n, 1, 1);
    auto syndrome = vector<uint8_t>(pcm.m, 0);

    EXPECT_THROW(decoder.bp_decode_serial(syndrome), std::logic_error);
}

TEST(RelayBpDecoder, BpDecodeSingleScanThrowsNotImplemented) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_relay_decoder(pcm, n, 1, 1);
    auto syndrome = vector<uint8_t>(pcm.m, 0);

    EXPECT_THROW(decoder.bp_decode_single_scan(syndrome), std::logic_error);
}

TEST(RelayBpDecoder, SoftInfoDecodeSerialThrowsNotImplemented) {
    int n = 5;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto decoder = make_relay_decoder(pcm, n, 1, 1);
    auto soft_syndrome = vector<double>(pcm.m, 0.5);

    EXPECT_THROW(decoder.soft_info_decode_serial(soft_syndrome, 1.0, 1.0), std::logic_error);
}


TEST(RelayBpDecoder, NonConvergingLegsSaveResults) {
    //There is an inconsistency between the log-likelihoods for negative initial memories compared to ibms rust implementation.
    //This was due to a bug where legs which didn't converge didn't have their results saved which I have now fixed.
    //This test makes sure the bug is no longer present
    double gamma0 = -1;
    int pre_iter = 80;
    int set_max_iter = 60;
    int maximum_solutions = 2;
    int n = 3;
    int maximum_legs = 2;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto channel_probabilities = vector<double>(n, 0.003);
    auto max_iters = vector<int>{pre_iter, set_max_iter};
    // Build per-leg memory vectors: leg 0 always has zero memory
    auto mem_strengths = vector<vector<double>>{{gamma0,gamma0,gamma0}, {-0.345025519,  0.190321013,  0.575298233}};

    auto decoder = RelayBpDecoder(pcm, channel_probabilities, maximum_legs, maximum_solutions,
                          max_iters, mem_strengths, false, 0,
                          ldpc::bp::MINIMUM_SUM, ldpc::bp::PARALLEL, 1);

    auto syndrome = vector<uint8_t>{1,1};

    decoder.decode(syndrome);

    //Expected outcomes
    EXPECT_NE(decoder.iterations_per_leg[0], 0);
    EXPECT_NE(decoder.log_prob_ratios_per_leg[0], std::vector<double>(n, 0.0));
}

TEST(RelayBpDecoder, MultiLegExampleExpectedResults) {
    //Check I find the expected results for a multileg example
    double gamma0 = -1;
    int pre_iter = 80;
    int set_max_iter = 60;
    int maximum_solutions = 2;
    int n = 3;
    int maximum_legs = 2;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto channel_probabilities = vector<double>(n, 0.003);
    auto max_iters = vector<int>{pre_iter, set_max_iter};
    // Build per-leg memory vectors: leg 0 always has zero memory
    auto mem_strengths = vector<vector<double>>{{gamma0,gamma0,gamma0}, {-0.345025519,  0.190321013,  0.575298233}};

    auto decoder = RelayBpDecoder(pcm, channel_probabilities, maximum_legs, maximum_solutions,
                          max_iters, mem_strengths, false, 0,
                          ldpc::bp::MINIMUM_SUM, ldpc::bp::PARALLEL, 1);

    auto syndrome = vector<uint8_t>{1,1};

    decoder.decode(syndrome);

    //Expected outcomes
    auto convergence_per_leg = vector<bool>{false, true};
    auto decoding_per_leg = vector<vector<uint8_t>>{{0,0,0}, {0,1,0}};
    auto iterations_per_leg = vector<int>{80, 2};
    auto log_prob_ratios_per_leg = vector<vector<double>>{{3.6418351862591384e+22, 4.2792000397402358e+22, 3.6418351862591384e+22},
                                                       {4.4809035473942366e+22,-4.3907058564338073e+22,5.4265411126195346e+21}};
    auto decoding = vector<uint8_t>{0,1,0};
    EXPECT_EQ(decoder.convergence_per_leg, convergence_per_leg);
    EXPECT_EQ(decoder.decoding_per_leg, decoding_per_leg);
    EXPECT_EQ(decoder.iterations_per_leg, iterations_per_leg);
    EXPECT_EQ(decoder.log_prob_ratios_per_leg, log_prob_ratios_per_leg);
    EXPECT_EQ(decoder.decoding, decoding);
}

TEST(RelayBpDecoder, MultiLegExampleExpectedResultsIBM) {
    //Check I find the expected results for a multileg example using the IBM implementation of the code
    double gamma0 = -1;
    int pre_iter = 80;
    int set_max_iter = 60;
    int maximum_solutions = 2;
    int n = 3;
    int maximum_legs = 2;
    auto pcm = ldpc::gf2codes::rep_code<ldpc::bp::BpEntry>(n);
    auto channel_probabilities = vector<double>(n, 0.003);
    auto max_iters = vector<int>{pre_iter, set_max_iter};
    // Build per-leg memory vectors: leg 0 always has zero memory
    auto mem_strengths = vector<vector<double>>{{gamma0,gamma0,gamma0}, {-0.345025519,  0.190321013,  0.575298233}};

    auto decoder = RelayBpDecoder(pcm, channel_probabilities, maximum_legs, maximum_solutions,
                          max_iters, mem_strengths, true, 0,
                          ldpc::bp::MINIMUM_SUM, ldpc::bp::PARALLEL, 1);

    auto syndrome = vector<uint8_t>{1,1};

    decoder.decode(syndrome);

    //Expected outcomes
    auto convergence_per_leg = vector<bool>{false, true};
    auto decoding_per_leg = vector<vector<uint8_t>>{{0,0,0}, {1,0,1}};
    auto iterations_per_leg = vector<int>{80, 6};
    auto log_prob_ratios_per_leg = vector<vector<double>>{{3.6418351862591384e+22, 4.2792000397402358e+22, 3.6418351862591384e+22},
                                                       {-6.7064626093939113e+21,5.2349039409693396e+21,-3.5295647500585984e+21}};
    auto decoding = vector<uint8_t>{1,0,1}; //Note the IBM version converges to the wrong solution
    EXPECT_EQ(decoder.convergence_per_leg, convergence_per_leg);
    EXPECT_EQ(decoder.decoding_per_leg, decoding_per_leg);
    EXPECT_EQ(decoder.iterations_per_leg, iterations_per_leg);
    EXPECT_EQ(decoder.log_prob_ratios_per_leg, log_prob_ratios_per_leg);
    EXPECT_EQ(decoder.decoding, decoding);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
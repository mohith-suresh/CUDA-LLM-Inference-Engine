// tests/test_engine.cu
// Integration test: GPT-2 engine end-to-end
#include <gtest/gtest.h>
#include "gpt2_engine.cuh"
#include "tokenizer.h"
#include <vector>
#include <string>
#include <fstream>

static const char* MODEL_DIR = "models/gpt2-int8";

static bool model_available() {
    std::ifstream f(std::string(MODEL_DIR) + "/config.json");
    return f.good();
}

class EngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!model_available()) {
            GTEST_SKIP() << "models/gpt2-int8 not present — run python/export_gpt2.py";
        }
        engine_ = new GPT2Engine(MODEL_DIR);
        tokenizer_ = new BPETokenizer(MODEL_DIR);
    }
    void TearDown() override {
        delete engine_;
        delete tokenizer_;
    }
    GPT2Engine* engine_ = nullptr;
    BPETokenizer* tokenizer_ = nullptr;
};

TEST_F(EngineTest, GreedyDeterministic) {
    auto prompt = tokenizer_->encode("The capital of France is");
    std::vector<int> run1, run2;
    InferenceMetrics m;

    engine_->generate(prompt, 10, 1.0f, 1, true,
        [&](int id) { run1.push_back(id); }, m);

    engine_->generate(prompt, 10, 1.0f, 1, true,
        [&](int id) { run2.push_back(id); }, m);

    ASSERT_EQ(run1.size(), run2.size());
    for (size_t i = 0; i < run1.size(); ++i)
        EXPECT_EQ(run1[i], run2[i]) << "Token mismatch at position " << i;
}

TEST_F(EngineTest, MetricsPopulated) {
    auto prompt = tokenizer_->encode("Hello");
    InferenceMetrics m;

    engine_->generate(prompt, 5, 1.0f, 1, true,
        [](int) {}, m);

    EXPECT_GT(m.ttft_ms, 0.0f);
    EXPECT_GT(m.tps, 0.0f);
    EXPECT_GT(m.kv_cache_pct, 0.0f);
    EXPECT_LE(m.tokens_generated, 5);
    EXPECT_GT(m.tokens_generated, 0);
}

TEST_F(EngineTest, EosStopsGeneration) {
    auto prompt = tokenizer_->encode("The end.");
    InferenceMetrics m;
    int count = 0;

    engine_->generate(prompt, 64, 1.0f, 1, true,
        [&](int) { count++; }, m);

    EXPECT_LE(count, 64);
    EXPECT_GT(count, 0);
}

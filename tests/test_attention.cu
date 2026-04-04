#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include "timer.cuh"
#include "flash_attention/10_flash_attn_v2.cuh"
#include "flash_attention/10b_flash_attn_v2_opt.cuh"

static void attention_cpu_reference(int B, int H, int N, int d,
                                    const float* Q, const float* K,
                                    const float* V, float* O, bool causal) {
    float scale = 1.0f / sqrtf((float)d);
    for (int bh = 0; bh < B * H; ++bh) {
        const float* q = Q + bh * N * d;
        const float* k = K + bh * N * d;
        const float* v = V + bh * N * d;
        float* o       = O + bh * N * d;

        float* S = new float[N * N];
        float* P = new float[N * N];

        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int kk = 0; kk < d; ++kk)
                    sum += q[i * d + kk] * k[j * d + kk];
                S[i * N + j] = sum * scale;
            }

        if (causal)
            for (int i = 0; i < N; ++i)
                for (int j = i + 1; j < N; ++j)
                    S[i * N + j] = -FLT_MAX;

        for (int i = 0; i < N; ++i) {
            float max_val = -FLT_MAX;
            for (int j = 0; j < N; ++j)
                max_val = fmaxf(max_val, S[i * N + j]);
            float sum = 0.0f;
            for (int j = 0; j < N; ++j) {
                P[i * N + j] = expf(S[i * N + j] - max_val);
                sum += P[i * N + j];
            }
            for (int j = 0; j < N; ++j)
                P[i * N + j] /= sum;
        }

        for (int i = 0; i < N; ++i)
            for (int j = 0; j < d; ++j) {
                float sum = 0.0f;
                for (int kk = 0; kk < N; ++kk)
                    sum += P[i * N + kk] * v[kk * d + j];
                o[i * d + j] = sum;
            }

        delete[] S;
        delete[] P;
    }
}

class AttentionTest : public ::testing::Test {
protected:
    float *d_Q, *d_K, *d_V, *d_O;
    float *h_Q, *h_K, *h_V, *h_O_ref;
    int B, H, N, d, total;

    void SetUpAttention(int b, int h, int n, int dim) {
        B = b; H = h; N = n; d = dim;
        total = B * H * N * d;

        CUDA_CHECK(cudaMalloc(&d_Q, total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V, total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O, total * sizeof(float)));

        h_Q = new float[total];
        h_K = new float[total];
        h_V = new float[total];
        h_O_ref = new float[total];

        srand(42);
        for (int i = 0; i < total; ++i) {
            h_Q[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            h_K[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            h_V[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        }
        CUDA_CHECK(cudaMemcpy(d_Q, h_Q, total * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_K, total * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_V, total * sizeof(float), cudaMemcpyHostToDevice));
    }

    void RunAndCheck(bool causal, float tol = 1e-5f) {
        attention_cpu_reference(B, H, N, d, h_Q, h_K, h_V, h_O_ref, causal);

        CUDA_CHECK(cudaMemset(d_O, 0, total * sizeof(float)));
        run_flash_attn_v2(B, H, N, d, d_Q, d_K, d_V, d_O, causal);
        CUDA_CHECK(cudaDeviceSynchronize());

        float* h_O = new float[total];
        CUDA_CHECK(cudaMemcpy(h_O, d_O, total * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float max_err = 0.0f;
        for (int i = 0; i < total; ++i) {
            float err = fabsf(h_O[i] - h_O_ref[i]);
            if (err > max_err) max_err = err;
        }
        delete[] h_O;
        EXPECT_LT(max_err, tol) << "Max error: " << max_err;
    }

    void RunAndCheckOpt(bool causal, float tol = 1e-5f) {
        attention_cpu_reference(B, H, N, d, h_Q, h_K, h_V, h_O_ref, causal);

        CUDA_CHECK(cudaMemset(d_O, 0, total * sizeof(float)));
        run_flash_attn_v2_opt(B, H, N, d, d_Q, d_K, d_V, d_O, causal);
        CUDA_CHECK(cudaDeviceSynchronize());

        float* h_O = new float[total];
        CUDA_CHECK(cudaMemcpy(h_O, d_O, total * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float max_err = 0.0f;
        for (int i = 0; i < total; ++i) {
            float err = fabsf(h_O[i] - h_O_ref[i]);
            if (err > max_err) max_err = err;
        }
        delete[] h_O;
        EXPECT_LT(max_err, tol) << "Max error: " << max_err;
    }

    void TearDown() override {
        delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_O_ref;
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    }
};

// --- Causal attention at different sequence lengths ---

struct AttnParams {
    int B, H, N, d;
};

class AttentionCausalTest : public AttentionTest,
                            public ::testing::WithParamInterface<AttnParams> {
protected:
    void SetUp() override {
        auto p = GetParam();
        SetUpAttention(p.B, p.H, p.N, p.d);
    }
};

TEST_P(AttentionCausalTest, Causal)    { RunAndCheck(true); }
TEST_P(AttentionCausalTest, NonCausal) { RunAndCheck(false); }

INSTANTIATE_TEST_SUITE_P(Configs, AttentionCausalTest, ::testing::Values(
    AttnParams{1, 1,  128, 64},
    AttnParams{1, 12, 128, 64},
    AttnParams{1, 12, 256, 64},
    AttnParams{1, 12, 512, 64}
));

// --- K10b optimized parameterized tests ---

class AttentionOptTest : public AttentionTest,
                         public ::testing::WithParamInterface<AttnParams> {
protected:
    void SetUp() override {
        auto p = GetParam();
        SetUpAttention(p.B, p.H, p.N, p.d);
    }
};

TEST_P(AttentionOptTest, CausalOpt)    { RunAndCheckOpt(true); }
TEST_P(AttentionOptTest, NonCausalOpt) { RunAndCheckOpt(false); }

INSTANTIATE_TEST_SUITE_P(ConfigsOpt, AttentionOptTest, ::testing::Values(
    AttnParams{1, 1,  128, 64},
    AttnParams{1, 12, 128, 64},
    AttnParams{1, 12, 256, 64},
    AttnParams{1, 12, 512, 64}
));

// --- Multi-batch ---
TEST_F(AttentionTest, MultiBatchCausalOpt) {
    SetUpAttention(2, 12, 256, 64);
    RunAndCheckOpt(true);
}

TEST_F(AttentionTest, FullGPT2CausalOpt) {
    SetUpAttention(1, 12, 1024, 64);
    RunAndCheckOpt(true);
}

TEST_F(AttentionTest, MultiBatchCausal) {
    SetUpAttention(2, 12, 256, 64);
    RunAndCheck(true);
}

// --- Full GPT-2 scale ---
TEST_F(AttentionTest, FullGPT2Causal) {
    SetUpAttention(1, 12, 1024, 64);
    RunAndCheck(true);
}

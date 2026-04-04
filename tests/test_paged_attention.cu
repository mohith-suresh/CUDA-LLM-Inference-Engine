#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include "timer.cuh"
#include "flash_attention/10_flash_attn_v2.cuh"
#include "paged_attention/11_paged_attn.cuh"
#include "paged_attention/12_gqa.cuh"

// Helper: convert contiguous K/V [B, H, N, d] into paged cache layout
// k_cache_out: [num_physical_blocks][block_size][H][d]
// block_table_out: [B][num_blocks_per_seq] with contiguous mapping
static void build_paged_kv_cache(
    const float* h_K, const float* h_V,
    int B, int H, int N, int d, int block_size,
    float* h_k_cache, float* h_v_cache,
    int* h_block_table, int* h_context_lens)
{
    int blocks_per_seq = (N + block_size - 1) / block_size;
    for (int b = 0; b < B; ++b) {
        h_context_lens[b] = N;
        for (int blk = 0; blk < blocks_per_seq; ++blk) {
            int phys = b * blocks_per_seq + blk;
            h_block_table[b * blocks_per_seq + blk] = phys;
            for (int t = 0; t < block_size; ++t) {
                int seq_pos = blk * block_size + t;
                for (int h_idx = 0; h_idx < H; ++h_idx) {
                    for (int dd = 0; dd < d; ++dd) {
                        int cache_idx = ((phys * block_size + t) * H + h_idx) * d + dd;
                        if (seq_pos < N) {
                            int src_idx = ((b * H + h_idx) * N + seq_pos) * d + dd;
                            h_k_cache[cache_idx] = h_K[src_idx];
                            h_v_cache[cache_idx] = h_V[src_idx];
                        } else {
                            h_k_cache[cache_idx] = 0.0f;
                            h_v_cache[cache_idx] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

class PagedAttnTest : public ::testing::Test {
protected:
    void RunK11vsK10(int B, int H, int N, int d, int block_size, bool causal, float tol = 1e-5f) {
        int total_qo = B * H * N * d;
        int blocks_per_seq = (N + block_size - 1) / block_size;
        int num_phys_blocks = B * blocks_per_seq;
        int cache_size = num_phys_blocks * block_size * H * d;

        std::vector<float> h_Q(total_qo), h_K(total_qo), h_V(total_qo);
        std::vector<float> h_k_cache(cache_size), h_v_cache(cache_size);
        std::vector<int> h_block_table(B * blocks_per_seq);
        std::vector<int> h_context_lens(B);

        srand(42);
        for (int i = 0; i < total_qo; ++i) {
            h_Q[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            h_K[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            h_V[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        }

        build_paged_kv_cache(h_K.data(), h_V.data(), B, H, N, d, block_size,
                             h_k_cache.data(), h_v_cache.data(),
                             h_block_table.data(), h_context_lens.data());

        float *d_Q, *d_K, *d_V, *d_O_k10, *d_O_k11;
        float *d_k_cache, *d_v_cache;
        int *d_block_table, *d_context_lens;

        CUDA_CHECK(cudaMalloc(&d_Q, total_qo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, total_qo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V, total_qo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O_k10, total_qo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O_k11, total_qo * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_k_cache, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v_cache, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_block_table, B * blocks_per_seq * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_context_lens, B * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), total_qo * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), total_qo * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), total_qo * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_k_cache, h_k_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v_cache, h_v_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_block_table, h_block_table.data(), B * blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_context_lens, h_context_lens.data(), B * sizeof(int), cudaMemcpyHostToDevice));

        // K10 reference
        CUDA_CHECK(cudaMemset(d_O_k10, 0, total_qo * sizeof(float)));
        run_flash_attn_v2(B, H, N, d, d_Q, d_K, d_V, d_O_k10, causal);
        CUDA_CHECK(cudaDeviceSynchronize());

        // K11
        CUDA_CHECK(cudaMemset(d_O_k11, 0, total_qo * sizeof(float)));
        run_paged_attn(B, H, N, d, d_Q, d_k_cache, d_v_cache,
                       d_block_table, d_context_lens,
                       N, block_size, blocks_per_seq, d_O_k11, causal);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compare
        std::vector<float> h_O_k10(total_qo), h_O_k11(total_qo);
        CUDA_CHECK(cudaMemcpy(h_O_k10.data(), d_O_k10, total_qo * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_O_k11.data(), d_O_k11, total_qo * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        for (int i = 0; i < total_qo; ++i) {
            float err = fabsf(h_O_k10[i] - h_O_k11[i]);
            if (err > max_err) max_err = err;
        }

        EXPECT_LT(max_err, tol) << "K11 vs K10 max error: " << max_err;

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
        cudaFree(d_O_k10); cudaFree(d_O_k11);
        cudaFree(d_k_cache); cudaFree(d_v_cache);
        cudaFree(d_block_table); cudaFree(d_context_lens);
    }
};

// --- K11 PagedAttention tests ---
TEST_F(PagedAttnTest, K11_SmallCausal)    { RunK11vsK10(1, 1, 64, 64, 16, true); }
TEST_F(PagedAttnTest, K11_MediumCausal)   { RunK11vsK10(1, 8, 128, 64, 16, true); }
TEST_F(PagedAttnTest, K11_LargeCausal)    { RunK11vsK10(1, 8, 256, 64, 16, true); }
TEST_F(PagedAttnTest, K11_NonCausal)      { RunK11vsK10(1, 8, 128, 64, 16, false); }
TEST_F(PagedAttnTest, K11_MultiBatch)     { RunK11vsK10(2, 8, 128, 64, 16, true); }
TEST_F(PagedAttnTest, K11_LargeMultiHead) { RunK11vsK10(1, 12, 512, 64, 16, true); }

// ============================================================
// K12 GQA Tests
// ============================================================

// CPU reference for GQA attention
static void gqa_cpu_reference(int B, int H_q, int H_kv, int N, int d,
                               const float* Q, const float* K, const float* V,
                               float* O, bool causal) {
    float scale = 1.0f / sqrtf((float)d);
    int group_size = H_q / H_kv;

    for (int b = 0; b < B; ++b) {
        for (int hq = 0; hq < H_q; ++hq) {
            int hkv = hq / group_size;
            const float* q = Q + ((long long)b * H_q + hq) * N * d;
            const float* k = K + ((long long)b * H_kv + hkv) * N * d;
            const float* v = V + ((long long)b * H_kv + hkv) * N * d;
            float* o       = O + ((long long)b * H_q + hq) * N * d;

            for (int i = 0; i < N; ++i) {
                float max_s = -FLT_MAX;
                std::vector<float> scores(N);
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < d; ++kk)
                        sum += q[i * d + kk] * k[j * d + kk];
                    scores[j] = sum * scale;
                    if (causal && j > i) scores[j] = -FLT_MAX;
                    if (scores[j] > max_s) max_s = scores[j];
                }
                float sum_exp = 0.0f;
                for (int j = 0; j < N; ++j) {
                    scores[j] = expf(scores[j] - max_s);
                    sum_exp += scores[j];
                }
                for (int j = 0; j < N; ++j)
                    scores[j] /= sum_exp;
                for (int dd = 0; dd < d; ++dd) {
                    float sum = 0.0f;
                    for (int j = 0; j < N; ++j)
                        sum += scores[j] * v[j * d + dd];
                    o[i * d + dd] = sum;
                }
            }
        }
    }
}

class GQATest : public ::testing::Test {
protected:
    void RunGQA(int B, int H_q, int H_kv, int N, int d, int block_size,
                bool causal, float tol = 1e-5f) {
        int total_q = B * H_q * N * d;
        int total_kv = B * H_kv * N * d;
        int blocks_per_seq = (N + block_size - 1) / block_size;
        int num_phys_blocks = B * blocks_per_seq;
        int cache_size = num_phys_blocks * block_size * H_kv * d;

        std::vector<float> h_Q(total_q), h_K(total_kv), h_V(total_kv);
        std::vector<float> h_k_cache(cache_size), h_v_cache(cache_size);
        std::vector<int> h_block_table(B * blocks_per_seq);
        std::vector<int> h_context_lens(B);
        std::vector<float> h_O_ref(total_q);

        srand(42);
        for (int i = 0; i < total_q; ++i)
            h_Q[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        for (int i = 0; i < total_kv; ++i) {
            h_K[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            h_V[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        }

        // Build paged cache with H_kv heads
        for (int b = 0; b < B; ++b) {
            h_context_lens[b] = N;
            for (int blk = 0; blk < blocks_per_seq; ++blk) {
                int phys = b * blocks_per_seq + blk;
                h_block_table[b * blocks_per_seq + blk] = phys;
                for (int t = 0; t < block_size; ++t) {
                    int seq_pos = blk * block_size + t;
                    for (int h = 0; h < H_kv; ++h) {
                        for (int dd = 0; dd < d; ++dd) {
                            int cache_idx = ((phys * block_size + t) * H_kv + h) * d + dd;
                            if (seq_pos < N) {
                                int src_idx = ((b * H_kv + h) * N + seq_pos) * d + dd;
                                h_k_cache[cache_idx] = h_K[src_idx];
                                h_v_cache[cache_idx] = h_V[src_idx];
                            } else {
                                h_k_cache[cache_idx] = 0.0f;
                                h_v_cache[cache_idx] = 0.0f;
                            }
                        }
                    }
                }
            }
        }

        gqa_cpu_reference(B, H_q, H_kv, N, d, h_Q.data(), h_K.data(), h_V.data(),
                          h_O_ref.data(), causal);

        float *d_Q, *d_O, *d_k_cache, *d_v_cache;
        int *d_block_table, *d_context_lens;

        CUDA_CHECK(cudaMalloc(&d_Q, total_q * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O, total_q * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_k_cache, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v_cache, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_block_table, B * blocks_per_seq * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_context_lens, B * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), total_q * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_k_cache, h_k_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v_cache, h_v_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_block_table, h_block_table.data(), B * blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_context_lens, h_context_lens.data(), B * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(d_O, 0, total_q * sizeof(float)));
        run_gqa_paged_attn(B, H_q, H_kv, N, d, d_Q, d_k_cache, d_v_cache,
                           d_block_table, d_context_lens,
                           N, block_size, blocks_per_seq, d_O, causal);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_O(total_q);
        CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, total_q * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        for (int i = 0; i < total_q; ++i) {
            float err = fabsf(h_O[i] - h_O_ref[i]);
            if (err > max_err) max_err = err;
        }

        EXPECT_LT(max_err, tol) << "GQA max error: " << max_err
                                 << " (H_q=" << H_q << ", H_kv=" << H_kv << ")";

        cudaFree(d_Q); cudaFree(d_O);
        cudaFree(d_k_cache); cudaFree(d_v_cache);
        cudaFree(d_block_table); cudaFree(d_context_lens);
    }
};

// K12 GQA: GROUP_SIZE=1 (MHA, should match K11)
TEST_F(GQATest, GQA_Group1_Causal)    { RunGQA(1, 8, 8, 128, 64, 16, true); }
TEST_F(GQATest, GQA_Group1_NonCausal) { RunGQA(1, 8, 8, 128, 64, 16, false); }
// K12 GQA: GROUP_SIZE=2
TEST_F(GQATest, GQA_Group2_Causal)    { RunGQA(1, 8, 4, 128, 64, 16, true); }
// K12 GQA: GROUP_SIZE=4
TEST_F(GQATest, GQA_Group4_Causal)    { RunGQA(1, 8, 2, 128, 64, 16, true); }
TEST_F(GQATest, GQA_Group4_Large)     { RunGQA(1, 16, 4, 256, 64, 16, true); }
// K12 GQA: GROUP_SIZE=8
TEST_F(GQATest, GQA_Group8_Causal)    { RunGQA(1, 8, 1, 128, 64, 16, true); }
// Multi-batch GQA
TEST_F(GQATest, GQA_MultiBatch)       { RunGQA(2, 8, 4, 128, 64, 16, true); }

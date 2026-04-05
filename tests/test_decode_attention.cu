// tests/test_decode_attention.cu
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include "timer.cuh"
#include "paged_attention/11_paged_attn.cuh"
#include "paged_attention/12_gqa.cuh"
#include "decode/13_decode_attn.cuh"

// Helper: build paged KV cache from contiguous K/V [B, H_kv, N, d]
static void build_paged_kv_cache(
    const float* h_K, const float* h_V,
    int B, int H_kv, int N, int d, int block_size,
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
                for (int h = 0; h < H_kv; ++h) {
                    for (int dd = 0; dd < d; ++dd) {
                        int idx = ((phys * block_size + t) * H_kv + h) * d + dd;
                        if (seq_pos < N) {
                            h_k_cache[idx] = h_K[((b * H_kv + h) * N + seq_pos) * d + dd];
                            h_v_cache[idx] = h_V[((b * H_kv + h) * N + seq_pos) * d + dd];
                        } else {
                            h_k_cache[idx] = 0.0f;
                            h_v_cache[idx] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

class DecodeAttnTest : public ::testing::Test {
protected:
    // Compare K13 decode attention (N=1 query) against K11 paged attention (N=1)
    void RunK13vsK11(int B, int H_q, int H_kv, int ctx_len, int d,
                     int block_size, float tol = 1e-5f) {
        // K13 query: [B, H_q, 1, d] — single token
        int q_size = B * H_q * 1 * d;
        // KV cache: ctx_len tokens
        int blocks_per_seq = (ctx_len + block_size - 1) / block_size;
        int num_phys_blocks = B * blocks_per_seq;
        int cache_size = num_phys_blocks * block_size * H_kv * d;

        // For K11 reference: Q is [B, H_q, 1, d], K/V are [B, H_kv, ctx_len, d]
        int kv_total = B * H_kv * ctx_len * d;

        std::vector<float> h_Q(q_size);
        std::vector<float> h_K(kv_total), h_V(kv_total);
        std::vector<float> h_k_cache(cache_size), h_v_cache(cache_size);
        std::vector<int> h_block_table(B * blocks_per_seq);
        std::vector<int> h_context_lens(B);

        srand(42);
        for (int i = 0; i < q_size; ++i)
            h_Q[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (int i = 0; i < kv_total; ++i) {
            h_K[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            h_V[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }

        build_paged_kv_cache(h_K.data(), h_V.data(), B, H_kv, ctx_len, d,
                             block_size, h_k_cache.data(), h_v_cache.data(),
                             h_block_table.data(), h_context_lens.data());

        // Device allocations
        float *d_Q, *d_O_k11, *d_O_k13;
        float *d_k_cache, *d_v_cache, *d_workspace;
        int *d_block_table, *d_context_lens;

        CUDA_CHECK(cudaMalloc(&d_Q, q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O_k11, q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O_k13, q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_k_cache, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v_cache, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_block_table, B * blocks_per_seq * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_context_lens, B * sizeof(int)));
        // Workspace for K13: [B, H_q, DA_MAX_SPLITS, d+2]
        int ws_size = B * H_q * DA_MAX_SPLITS * (d + 2);
        CUDA_CHECK(cudaMalloc(&d_workspace, ws_size * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), q_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_k_cache, h_k_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v_cache, h_v_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_block_table, h_block_table.data(), B * blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_context_lens, h_context_lens.data(), B * sizeof(int), cudaMemcpyHostToDevice));

        // K11 reference with N=1, non-causal (decode attends to all KV tokens)
        CUDA_CHECK(cudaMemset(d_O_k11, 0, q_size * sizeof(float)));
        if (H_q == H_kv) {
            run_paged_attn(B, H_q, 1, d, d_Q, d_k_cache, d_v_cache,
                           d_block_table, d_context_lens,
                           ctx_len, block_size, blocks_per_seq, d_O_k11, false);
        } else {
            run_gqa_paged_attn(B, H_q, H_kv, 1, d, d_Q, d_k_cache, d_v_cache,
                               d_block_table, d_context_lens,
                               ctx_len, block_size, blocks_per_seq, d_O_k11, false);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // K13
        CUDA_CHECK(cudaMemset(d_O_k13, 0, q_size * sizeof(float)));
        run_decode_attn(B, H_q, H_kv, d, d_Q, d_k_cache, d_v_cache,
                        d_block_table, d_context_lens,
                        ctx_len, block_size, blocks_per_seq,
                        d_O_k13, d_workspace);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compare
        std::vector<float> h_O_k11(q_size), h_O_k13(q_size);
        CUDA_CHECK(cudaMemcpy(h_O_k11.data(), d_O_k11, q_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_O_k13.data(), d_O_k13, q_size * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        for (int i = 0; i < q_size; ++i) {
            float err = fabsf(h_O_k11[i] - h_O_k13[i]);
            if (err > max_err) max_err = err;
        }

        EXPECT_LT(max_err, tol) << "K13 vs K11 max error: " << max_err
                                 << " (B=" << B << " H_q=" << H_q
                                 << " H_kv=" << H_kv << " ctx=" << ctx_len << ")";

        cudaFree(d_Q); cudaFree(d_O_k11); cudaFree(d_O_k13);
        cudaFree(d_k_cache); cudaFree(d_v_cache);
        cudaFree(d_block_table); cudaFree(d_context_lens);
        cudaFree(d_workspace);
    }
};

// --- K13 MHA tests (H_q == H_kv, GROUP_SIZE=1) ---
TEST_F(DecodeAttnTest, K13_MHA_Small)     { RunK13vsK11(1, 8, 8, 128, 64, 16); }
TEST_F(DecodeAttnTest, K13_MHA_Medium)    { RunK13vsK11(1, 16, 16, 512, 64, 16); }
TEST_F(DecodeAttnTest, K13_MHA_Batch)     { RunK13vsK11(4, 8, 8, 256, 64, 16); }
TEST_F(DecodeAttnTest, K13_MHA_Long)      { RunK13vsK11(1, 8, 8, 1024, 64, 16); }

// --- K13 GQA tests (H_q != H_kv) ---
TEST_F(DecodeAttnTest, K13_GQA_Group4)    { RunK13vsK11(1, 16, 4, 256, 64, 16); }
TEST_F(DecodeAttnTest, K13_GQA_Group4_Lg) { RunK13vsK11(8, 32, 8, 1024, 64, 16); }

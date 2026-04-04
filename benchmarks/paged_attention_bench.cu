// benchmarks/paged_attention_bench.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <cuda_runtime.h>
#include "timer.cuh"
#include "flash_attention/10_flash_attn_v2.cuh"
#include "paged_attention/11_paged_attn.cuh"
#include "paged_attention/12_gqa.cuh"

// Build paged KV cache from contiguous K/V [B, H_kv, N, d]
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

int main() {
    const int d = 64;
    const int block_size = 16;
    const float peak_bw = 192.0f;

    printf("SLICK PagedAttention + GQA Benchmark\n");
    printf("GPU: GTX 1650 Ti | CUDA 11.8 | FP32\n");
    printf("Block size: %d | Head dim: %d\n", block_size, d);
    printf("================================================\n\n");

    // --- K11 vs K10 benchmark ---
    printf("--- K11 PagedAttention vs K10 FlashAttention (causal) ---\n\n");
    printf("%-10s %5s %5s %6s  %10s %10s %10s\n",
           "Config", "B", "H", "N", "K10(us)", "K11(us)", "Overhead");
    printf("---------------------------------------------------------------\n");

    struct MHAConfig { int B, H, N; const char* desc; };
    MHAConfig mha_configs[] = {
        {1, 8,  128, "Small"},
        {1, 8,  256, "Medium"},
        {1, 12, 256, "GPT-2 sm"},
        {1, 12, 512, "GPT-2 md"},
        {2, 8,  256, "Batch=2"},
    };

    for (auto& c : mha_configs) {
        int total_qkv = c.B * c.H * c.N * d;
        int blocks_per_seq = (c.N + block_size - 1) / block_size;
        int num_phys = c.B * blocks_per_seq;
        int cache_size = num_phys * block_size * c.H * d;

        std::vector<float> h_K(total_qkv), h_V(total_qkv);
        std::vector<float> h_k_cache(cache_size), h_v_cache(cache_size);
        std::vector<int> h_bt(c.B * blocks_per_seq), h_cl(c.B);
        srand(42);
        for (int i = 0; i < total_qkv; ++i) {
            h_K[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            h_V[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
        build_paged_kv_cache(h_K.data(), h_V.data(), c.B, c.H, c.N, d, block_size,
                             h_k_cache.data(), h_v_cache.data(), h_bt.data(), h_cl.data());

        float *d_Q, *d_K, *d_V, *d_O, *d_kc, *d_vc;
        int *d_bt, *d_cl;
        CUDA_CHECK(cudaMalloc(&d_Q, total_qkv * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, total_qkv * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V, total_qkv * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O, total_qkv * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_kc, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_vc, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bt, c.B * blocks_per_seq * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cl, c.B * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_Q, h_K.data(), total_qkv * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), total_qkv * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), total_qkv * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kc, h_k_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vc, h_v_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bt, h_bt.data(), c.B * blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cl, h_cl.data(), c.B * sizeof(int), cudaMemcpyHostToDevice));

        // K10 benchmark
        for (int w = 0; w < 3; ++w)
            run_flash_attn_v2(c.B, c.H, c.N, d, d_Q, d_K, d_V, d_O, true);
        CUDA_CHECK(cudaDeviceSynchronize());
        GpuTimer timer;
        timer.tic();
        for (int r = 0; r < 10; ++r)
            run_flash_attn_v2(c.B, c.H, c.N, d, d_Q, d_K, d_V, d_O, true);
        float k10_us = timer.toc() / 10.0f * 1000.0f;

        // K11 benchmark
        for (int w = 0; w < 3; ++w)
            run_paged_attn(c.B, c.H, c.N, d, d_Q, d_kc, d_vc,
                           d_bt, d_cl, c.N, block_size, blocks_per_seq, d_O, true);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.tic();
        for (int r = 0; r < 10; ++r)
            run_paged_attn(c.B, c.H, c.N, d, d_Q, d_kc, d_vc,
                           d_bt, d_cl, c.N, block_size, blocks_per_seq, d_O, true);
        float k11_us = timer.toc() / 10.0f * 1000.0f;

        printf("%-10s %5d %5d %6d  %10.1f %10.1f %9.2fx\n",
               c.desc, c.B, c.H, c.N, k10_us, k11_us, k11_us / k10_us);

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
        cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_bt); cudaFree(d_cl);
    }

    // --- K12 GQA sweep ---
    printf("\n--- K12 GQA Sweep (causal, B=1, d=%d) ---\n\n", d);
    printf("%-6s %6s %6s %6s  %10s\n",
           "H_q", "H_kv", "Group", "N", "K12(us)");
    printf("----------------------------------------------\n");

    struct GQAConfig { int H_q, H_kv, N; };
    GQAConfig gqa_configs[] = {
        {8,  8, 256},   // GROUP_SIZE=1 (MHA)
        {8,  4, 256},   // GROUP_SIZE=2
        {8,  2, 256},   // GROUP_SIZE=4
        {8,  1, 256},   // GROUP_SIZE=8 (MQA)
        {16, 4, 256},   // GROUP_SIZE=4
        {32, 4, 256},   // GROUP_SIZE=8
        {32, 4, 512},   // GROUP_SIZE=8, longer seq
    };

    for (auto& c : gqa_configs) {
        int B = 1;
        int group = c.H_q / c.H_kv;
        int total_q = B * c.H_q * c.N * d;
        int blocks_per_seq = (c.N + block_size - 1) / block_size;
        int num_phys = B * blocks_per_seq;
        int cache_size = num_phys * block_size * c.H_kv * d;

        std::vector<float> h_K(B * c.H_kv * c.N * d), h_V(B * c.H_kv * c.N * d);
        std::vector<float> h_k_cache(cache_size), h_v_cache(cache_size);
        std::vector<int> h_bt(B * blocks_per_seq), h_cl(B);
        srand(42);
        for (size_t i = 0; i < h_K.size(); ++i) {
            h_K[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            h_V[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
        build_paged_kv_cache(h_K.data(), h_V.data(), B, c.H_kv, c.N, d, block_size,
                             h_k_cache.data(), h_v_cache.data(), h_bt.data(), h_cl.data());

        float *d_Q, *d_O, *d_kc, *d_vc;
        int *d_bt, *d_cl;
        CUDA_CHECK(cudaMalloc(&d_Q, total_q * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O, total_q * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_kc, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_vc, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bt, B * blocks_per_seq * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cl, B * sizeof(int)));

        std::vector<float> h_Q(total_q);
        for (int i = 0; i < total_q; ++i)
            h_Q[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), total_q * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kc, h_k_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vc, h_v_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bt, h_bt.data(), B * blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cl, h_cl.data(), B * sizeof(int), cudaMemcpyHostToDevice));

        for (int w = 0; w < 3; ++w)
            run_gqa_paged_attn(B, c.H_q, c.H_kv, c.N, d, d_Q, d_kc, d_vc,
                               d_bt, d_cl, c.N, block_size, blocks_per_seq, d_O, true);
        CUDA_CHECK(cudaDeviceSynchronize());
        GpuTimer timer;
        timer.tic();
        for (int r = 0; r < 10; ++r)
            run_gqa_paged_attn(B, c.H_q, c.H_kv, c.N, d, d_Q, d_kc, d_vc,
                               d_bt, d_cl, c.N, block_size, blocks_per_seq, d_O, true);
        float k12_us = timer.toc() / 10.0f * 1000.0f;

        printf("%-6d %6d %6d %6d  %10.1f\n",
               c.H_q, c.H_kv, group, c.N, k12_us);

        cudaFree(d_Q); cudaFree(d_O);
        cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_bt); cudaFree(d_cl);
    }

    printf("\nFLOPs formula: 4 * B * H_q * N^2 * d (QK^T + PV matmuls)\n");
    printf("Peak BW: %.0f GB/s\n", peak_bw);

    return 0;
}

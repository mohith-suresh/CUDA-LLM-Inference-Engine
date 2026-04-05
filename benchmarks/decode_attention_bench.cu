// benchmarks/decode_attention_bench.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "timer.cuh"
#include "paged_attention/11_paged_attn.cuh"
#include "paged_attention/12_gqa.cuh"
#include "decode/13_decode_attn.cuh"

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

    printf("SLICK Decode Attention Benchmark\n");
    printf("GPU: GTX 1650 Ti | CUDA 11.8 | FP32\n");
    printf("Block size: %d | Head dim: %d\n", block_size, d);
    printf("================================================\n\n");

    printf("--- K13 Decode Attention vs K11 PagedAttention (N=1, non-causal) ---\n\n");
    printf("%-6s %5s %5s %6s  %10s %10s %10s\n",
           "Config", "B", "H", "ctx", "K11(us)", "K13(us)", "Speedup");
    printf("--------------------------------------------------------------\n");

    struct Config { int B, H_q, H_kv, ctx; const char* desc; };
    Config configs[] = {
        {1,  8,  8,  128,  "Small"},
        {1,  8,  8,  256,  "Med"},
        {1,  8,  8,  512,  "Large"},
        {1,  8,  8,  1024, "XL"},
        {4,  8,  8,  256,  "Bat=4"},
        {8,  8,  8,  512,  "Bat=8"},
        {1,  16, 4,  256,  "GQA4"},
        {1,  32, 8,  512,  "GQA4L"},
    };

    for (auto& c : configs) {
        int q_size = c.B * c.H_q * 1 * d;
        int kv_total = c.B * c.H_kv * c.ctx * d;
        int blocks_per_seq = (c.ctx + block_size - 1) / block_size;
        int num_phys = c.B * blocks_per_seq;
        int cache_size = num_phys * block_size * c.H_kv * d;
        int ws_size = c.B * c.H_q * DA_MAX_SPLITS * (d + 2);

        std::vector<float> h_Q(q_size), h_K(kv_total), h_V(kv_total);
        std::vector<float> h_k_cache(cache_size), h_v_cache(cache_size);
        std::vector<int> h_bt(c.B * blocks_per_seq), h_cl(c.B);

        srand(42);
        for (int i = 0; i < q_size; ++i)
            h_Q[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (int i = 0; i < kv_total; ++i) {
            h_K[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            h_V[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
        build_paged_kv_cache(h_K.data(), h_V.data(), c.B, c.H_kv, c.ctx, d,
                             block_size, h_k_cache.data(), h_v_cache.data(),
                             h_bt.data(), h_cl.data());

        float *d_Q, *d_O, *d_kc, *d_vc, *d_ws;
        int *d_bt, *d_cl;
        CUDA_CHECK(cudaMalloc(&d_Q, q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O, q_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_kc, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_vc, cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ws, ws_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bt, c.B * blocks_per_seq * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cl, c.B * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), q_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kc, h_k_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vc, h_v_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bt, h_bt.data(), c.B * blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cl, h_cl.data(), c.B * sizeof(int), cudaMemcpyHostToDevice));

        // K11 with N=1 (non-causal for decode)
        auto bench_k11 = [&]() {
            if (c.H_q == c.H_kv)
                run_paged_attn(c.B, c.H_q, 1, d, d_Q, d_kc, d_vc,
                               d_bt, d_cl, c.ctx, block_size, blocks_per_seq, d_O, false);
            else
                run_gqa_paged_attn(c.B, c.H_q, c.H_kv, 1, d, d_Q, d_kc, d_vc,
                                   d_bt, d_cl, c.ctx, block_size, blocks_per_seq, d_O, false);
        };

        for (int w = 0; w < 3; ++w) bench_k11();
        CUDA_CHECK(cudaDeviceSynchronize());
        GpuTimer timer;
        timer.tic();
        for (int r = 0; r < 20; ++r) bench_k11();
        float k11_us = timer.toc() / 20.0f * 1000.0f;

        // K13
        for (int w = 0; w < 3; ++w)
            run_decode_attn(c.B, c.H_q, c.H_kv, d, d_Q, d_kc, d_vc,
                            d_bt, d_cl, c.ctx, block_size, blocks_per_seq, d_O, d_ws);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.tic();
        for (int r = 0; r < 20; ++r)
            run_decode_attn(c.B, c.H_q, c.H_kv, d, d_Q, d_kc, d_vc,
                            d_bt, d_cl, c.ctx, block_size, blocks_per_seq, d_O, d_ws);
        float k13_us = timer.toc() / 20.0f * 1000.0f;

        printf("%-6s %5d %5d %6d  %10.1f %10.1f %9.2fx\n",
               c.desc, c.B, c.H_q, c.ctx, k11_us, k13_us, k11_us / k13_us);

        cudaFree(d_Q); cudaFree(d_O);
        cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_ws);
        cudaFree(d_bt); cudaFree(d_cl);
    }

    printf("\nK13 uses split-K parallelism: multiple blocks per (batch, head)\n");
    printf("K11 uses single block per (batch, head) with N=1 (Br=64 tile mostly empty)\n");

    return 0;
}

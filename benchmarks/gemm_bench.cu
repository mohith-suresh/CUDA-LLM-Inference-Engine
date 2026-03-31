#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include "timer.cuh"
#include "validator.cuh"
#include "gemm/01_naive.cuh"
#include "gemm/02_coalesced.cuh"
#include "gemm/03_shared_tiling.cuh"
#include "gemm/04_reg_tiling_1d.cuh"
#include "gemm/05_reg_tiling_2d.cuh"
#include "gemm/06_vectorized.cuh"
#include "gemm/07_double_buffered.cuh"

typedef void (*GemmFn)(int, int, int, const float*, const float*, float*);

struct KernelInfo {
    const char* name;
    GemmFn fn;
    int bm, bn, bk;   // block tile dimensions (0 = no tiling)
    int tm, tn;        // thread tile dimensions
    int smem_bytes;
    int threads;
};

int main() {
    int sizes[] = {256, 512, 1024, 2048};
    int num_sizes = 4;

    // name, fn, BM, BN, BK, TM, TN, smem_bytes, threads
    KernelInfo kernels[] = {
        {"01 Naive",          run_sgemm_naive,          32, 32,  0, 1, 1,     0, 1024},
        {"02 Coalesced",      run_sgemm_coalesced,      32, 32,  0, 1, 1,     0, 1024},
        {"03 Shared Tiling",  run_sgemm_shared_tiling,  32, 32, 32, 1, 1,  8192, 1024},
        {"04 1D Reg Tiling",  run_sgemm_reg_tiling_1d,  64, 64,  8, 8, 1,  4096,  512},
        {"05 2D Reg Tiling",  run_sgemm_reg_tiling_2d, 128,128,  8, 8, 8,  8192,  256},
        {"06 Vectorized",     run_sgemm_vectorized,    128,128,  8, 8, 8,  8192,  256},
        {"07 Double Buffered",run_sgemm_double_buffered,128,128, 8, 8, 8, 16384,  256},
    };
    int num_kernels = sizeof(kernels) / sizeof(kernels[0]);

    CublasValidator validator;

    printf("SLICK GEMM Benchmark\n");
    printf("GPU: GTX 1650 Ti | CUDA 10.1 | FP32\n");
    printf("========================================\n\n");

    for (int s = 0; s < num_sizes; ++s) {
        int M = sizes[s], N = sizes[s], K = sizes[s];
        int size_A = M * K;
        int size_B = K * N;
        int size_C = M * N;

        float *d_A, *d_B, *d_C, *d_ref;
        CUDA_CHECK(cudaMalloc(&d_A,   size_A * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B,   size_B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C,   size_C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ref, size_C * sizeof(float)));

        init_random_matrix(d_A, size_A, 42);
        init_random_matrix(d_B, size_B, 137);

        // cuBLAS reference
        validator.sgemm(M, N, K, d_A, d_B, d_ref);
        CUDA_CHECK(cudaDeviceSynchronize());

        printf("Matrix Size: %dx%d\n", M, M);
        printf("%-25s %10s %8s %15s\n", "Kernel", "GFLOPS", "Status", "Max Error");
        printf("--------------------------------------------------------------\n");

        for (int ki = 0; ki < num_kernels; ++ki) {
            CUDA_CHECK(cudaMemset(d_C, 0, size_C * sizeof(float)));

            float avg_ms = benchmark_gemm(kernels[ki].fn, M, N, K,
                                          d_A, d_B, d_C);
            float gflops = compute_gflops(M, N, K, avg_ms);

            float err = max_error(d_C, d_ref, size_C);
            // FP32 accumulation error grows with K; use K * machine_eps as tolerance
            float tol = K * 1.2e-7f;
            bool pass = err < tol;

            printf("%-25s %10.2f %8s %15.2e\n",
                   kernels[ki].name, gflops,
                   pass ? "PASS" : "FAIL", err);
        }
        printf("\n");

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaFree(d_ref));
    }

    // ==========================================
    // Roofline Analysis (analytical)
    // ==========================================
    // GTX 1650 Ti specs:
    //   1024 CUDA cores @ 2100 MHz max = 2 * 1024 * 2.1 = 4300 GFLOPS peak FP32
    //   GDDR6 128-bit bus @ 12 Gbps = 192 GB/s peak DRAM bandwidth
    //   Ridge point = 4300 / 192 = 22.4 FLOP/byte
    const float peak_gflops = 4300.0f;
    const float peak_bw = 192.0f;  // GB/s
    const float ridge = peak_gflops / peak_bw;

    printf("========================================\n");
    printf("ROOFLINE ANALYSIS (Analytical)\n");
    printf("Peak FP32: %.0f GFLOPS | Peak BW: %.0f GB/s | Ridge: %.1f FLOP/byte\n",
           peak_gflops, peak_bw, ridge);
    printf("========================================\n\n");

    {
        int M = 2048, N = 2048, K = 2048;
        int size_A = M * K, size_B = K * N, size_C = M * N;
        double min_bytes = (double)(size_A + size_B + size_C) * 4.0;

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));
        init_random_matrix(d_A, size_A, 42);
        init_random_matrix(d_B, size_B, 137);

        printf("Matrix: %dx%d | Min DRAM: %.1f MB\n\n", M, M, min_bytes / (1024.0 * 1024.0));
        printf("%-20s %7s %7s %8s %10s %9s  %-9s %s\n",
               "Kernel", "GFLOPS", "%Peak", "AI", "BW(GB/s)", "%BW", "Bound", "Details");
        printf("------------------------------------------------------------------------------------------------------\n");

        for (int ki = 0; ki < num_kernels; ++ki) {
            CUDA_CHECK(cudaMemset(d_C, 0, size_C * sizeof(float)));

            float avg_ms = benchmark_gemm(kernels[ki].fn, M, N, K,
                                          d_A, d_B, d_C, 3, 20);
            float gflops = compute_gflops(M, N, K, avg_ms);
            float pct_peak = gflops / peak_gflops * 100.0f;

            // Achieved bandwidth (lower bound: assumes min data movement)
            float achieved_bw = (float)(min_bytes / (avg_ms * 1e6));
            float pct_bw = achieved_bw / peak_bw * 100.0f;

            // Theoretical arithmetic intensity from tile dimensions
            float ai;
            if (kernels[ki].bk > 0) {
                // AI = 2*BM*BN*BK / ((BM*BK + BK*BN) * 4)
                float flops = 2.0f * kernels[ki].bm * kernels[ki].bn * kernels[ki].bk;
                float bytes = (kernels[ki].bm * kernels[ki].bk +
                               kernels[ki].bk * kernels[ki].bn) * 4.0f;
                ai = flops / bytes;
            } else {
                // No tiling: each output reads full K from A and B
                // 2K FLOPs per (K+K)*4 bytes = 0.25 FLOP/byte
                ai = 0.25f;
            }

            float mem_roof = ai * peak_bw;
            const char* bound;
            char details[128];

            if (ai < ridge) {
                bound = "MEMORY";
                float eff = gflops / mem_roof * 100.0f;
                snprintf(details, sizeof(details), "%.0f%% of mem ceiling (%.0f GFLOPS)", eff, mem_roof);
            } else {
                bound = "COMPUTE";
                snprintf(details, sizeof(details), "%.1f%% of compute ceiling", pct_peak);
            }

            printf("%-20s %7.0f %6.1f%% %8.1f %10.1f %8.1f%%  %-9s %s\n",
                   kernels[ki].name, gflops, pct_peak, ai,
                   achieved_bw, pct_bw, bound, details);
        }

        printf("\n");
        printf("AI = Arithmetic Intensity (FLOP/byte, theoretical from tile dims)\n");
        printf("BW = Achieved bandwidth assuming minimum data movement (read A,B + write C)\n");
        printf("Bound = MEMORY if AI < ridge (%.1f), COMPUTE if AI >= ridge\n", ridge);
        printf("\n");

        // Kernel properties summary
        printf("Kernel Properties\n");
        printf("%-20s %5s %5s %5s %4s %4s %8s %8s %6s\n",
               "Kernel", "BM", "BN", "BK", "TM", "TN", "SMEM", "Threads", "AI");
        printf("----------------------------------------------------------------------\n");
        for (int ki = 0; ki < num_kernels; ++ki) {
            float ai = kernels[ki].bk > 0
                ? 2.0f * kernels[ki].bm * kernels[ki].bn * kernels[ki].bk /
                  ((kernels[ki].bm * kernels[ki].bk + kernels[ki].bk * kernels[ki].bn) * 4.0f)
                : 0.25f;
            printf("%-20s %5d %5d %5d %4d %4d %7dB %8d %6.1f\n",
                   kernels[ki].name,
                   kernels[ki].bm, kernels[ki].bn, kernels[ki].bk,
                   kernels[ki].tm, kernels[ki].tn,
                   kernels[ki].smem_bytes, kernels[ki].threads, ai);
        }

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    return 0;
}

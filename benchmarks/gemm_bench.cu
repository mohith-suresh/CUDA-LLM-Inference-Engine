#include <cstdio>
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
};

int main() {
    int sizes[] = {256, 512, 1024, 2048};
    int num_sizes = 4;

    KernelInfo kernels[] = {
        {"01 Naive",          run_sgemm_naive},
        {"02 Coalesced",      run_sgemm_coalesced},
        {"03 Shared Tiling",  run_sgemm_shared_tiling},
        {"04 1D Reg Tiling",  run_sgemm_reg_tiling_1d},
        {"05 2D Reg Tiling",  run_sgemm_reg_tiling_2d},
        {"06 Vectorized",      run_sgemm_vectorized},
        {"07 Double Buffered", run_sgemm_double_buffered},
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

    return 0;
}

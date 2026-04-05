// benchmarks/int8_gemm_bench.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "timer.cuh"
#include "quantization/14_int8_gemm.cuh"

int main() {
    printf("SLICK INT8 GEMM (dp4a) Benchmark\n");
    printf("GPU: GTX 1650 Ti | CUDA 11.8 | INT8 dp4a\n");
    printf("Layout: NT (A row-major, B^T row-major, both int8x4 packed)\n");
    printf("Tile: BM=%d BN=%d BK=%d | Thread tile: TM=%d TN=%d\n",
           I8_BM, I8_BN, I8_BK, I8_TM, I8_TN);
    printf("================================================\n\n");

    cublasHandle_t handle;
    cublasCreate(&handle);

    printf("%-6s  %10s %10s %10s %10s\n",
           "Size", "K14(us)", "FP32(us)", "K14 GOPS", "INT8/FP32");
    printf("-----------------------------------------------------------\n");

    int sizes[] = {256, 512, 1024, 2048};

    for (int S : sizes) {
        int M = S, N = S, K = S;
        int sizeA = M * K;
        int sizeB = K * N;

        // Generate FP32 data
        std::vector<float> h_A(sizeA), h_B(sizeB), h_BT(N * K);
        srand(42);
        for (int i = 0; i < sizeA; ++i)
            h_A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (int i = 0; i < sizeB; ++i)
            h_B[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        // Transpose B [K, N] -> B^T [N, K]
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                h_BT[n * K + k] = h_B[k * N + n];

        // Device allocs
        float *d_A, *d_BT, *d_C;
        int32_t *d_A_packed, *d_BT_packed;
        float *d_scale_A, *d_scale_B;
        CUDA_CHECK(cudaMalloc(&d_A, sizeA * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_BT, N * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_A_packed, M * (K / 4) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_BT_packed, N * (K / 4) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_scale_A, M * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scale_B, N * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_BT, h_BT.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));

        // Quantize (not timed — separate kernel)
        run_quantize_fp32_to_int8(M, K, d_A, d_A_packed, d_scale_A);
        run_quantize_fp32_to_int8(N, K, d_BT, d_BT_packed, d_scale_B);
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- K14 benchmark ---
        for (int w = 0; w < 3; ++w)
            run_int8_gemm(M, N, K, d_A_packed, d_BT_packed, d_scale_A, d_scale_B, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());

        GpuTimer timer;
        int repeats = 10;
        timer.tic();
        for (int r = 0; r < repeats; ++r)
            run_int8_gemm(M, N, K, d_A_packed, d_BT_packed, d_scale_A, d_scale_B, d_C);
        float k14_us = timer.toc() / repeats * 1000.0f;

        // --- FP32 cuBLAS SGEMM baseline ---
        float *d_B_fp32, *d_C_fp32;
        CUDA_CHECK(cudaMalloc(&d_B_fp32, sizeB * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_fp32, M * N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_B_fp32, h_B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));

        float alpha_f = 1.0f, beta_f = 0.0f;
        // cuBLAS col-major: C^T(N,M) = B^T(N,K) @ A^T(K,M)
        for (int w = 0; w < 3; ++w)
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        N, M, K, &alpha_f,
                        d_B_fp32, K, d_A, K,
                        &beta_f, d_C_fp32, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.tic();
        for (int r = 0; r < repeats; ++r)
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        N, M, K, &alpha_f,
                        d_B_fp32, K, d_A, K,
                        &beta_f, d_C_fp32, N);
        float fp32_us = timer.toc() / repeats * 1000.0f;

        // INT8 GOPS: 2*M*N*K operations
        double ops = 2.0 * M * N * K;
        float k14_gops = (float)(ops / (k14_us * 1e3));

        printf("%-6d  %10.1f %10.1f %10.1f %9.2fx\n",
               S, k14_us, fp32_us, k14_gops, fp32_us / k14_us);

        cudaFree(d_A); cudaFree(d_BT); cudaFree(d_C);
        cudaFree(d_A_packed); cudaFree(d_BT_packed);
        cudaFree(d_scale_A); cudaFree(d_scale_B);
        cudaFree(d_B_fp32); cudaFree(d_C_fp32);
    }

    cublasDestroy(handle);

    printf("\nGOPS = 2*M*N*K / time (giga int8 ops/sec)\n");
    printf("INT8/FP32 > 1 means K14 INT8 is faster than cuBLAS FP32 SGEMM\n");
    printf("Quantization time not included in K14 timing\n");

    return 0;
}

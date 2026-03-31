#include <cuda_runtime.h>
#include "gemm/01_naive.cuh"

__global__ void sgemm_naive_kernel(int M, int N, int K,
                                   const float* A, const float* B, float* C) {
    // threadIdx.x maps to row → adjacent threads access non-adjacent B elements
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void run_sgemm_naive(int M, int N, int K,
                     const float* A, const float* B, float* C) {
    dim3 block(32, 32);
    dim3 grid((M + 31) / 32, (N + 31) / 32);
    sgemm_naive_kernel<<<grid, block>>>(M, N, K, A, B, C);
}

#include <cuda_runtime.h>
#include "gemm/02_coalesced.cuh"

__global__ void sgemm_coalesced_kernel(int M, int N, int K,
                                       const float* A, const float* B, float* C) {
    // threadIdx.x maps to col → adjacent threads read adjacent B[k*N + col] (coalesced)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void run_sgemm_coalesced(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    sgemm_coalesced_kernel<<<grid, block>>>(M, N, K, A, B, C);
}

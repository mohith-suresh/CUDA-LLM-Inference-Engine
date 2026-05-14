// Step 2 — Global memory coalescing
// ----------------------------------------------------------------------------
// Identical math to step 1. The ONLY change: threadIdx.x now drives `col`.
//
// Now a warp (consecutive threadIdx.x) touches consecutive COLUMNS, so:
//   B[k*N + col]  -> consecutive addresses -> 1 coalesced 128-byte transaction
//   C[row*N + col]-> consecutive addresses -> coalesced
//   A[row*K + k]  -> same address for all 32 threads -> broadcast (free)
//
// Same arithmetic intensity as step 1, but ~10x faster. Coalescing is about
// transaction EFFICIENCY, not about moving less data.
#pragma once
#include "common.cuh"

__global__ void coalesced_kernel(int M, int N, int K, float alpha,
                                 const float* A, const float* B, float beta, float* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // <-- threadIdx.x drives col (coalesced)
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

inline void run_coalesced(int M, int N, int K, float alpha,
                          const float* A, const float* B, float beta, float* C) {
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    coalesced_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

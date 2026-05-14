// Step 1 — Naive
// ----------------------------------------------------------------------------
// GEMM:  C = alpha * (A @ B) + beta * C.
// One thread computes one output element C[row,col]. It walks the full K
// dimension, reading a whole row of A and a whole column of B from global
// memory. No reuse, no coalescing — this is the baseline.
//
// The deliberately-bad part: we map threadIdx.x -> row. A warp is 32 threads
// with consecutive threadIdx.x, so consecutive threads touch consecutive ROWS.
// That makes A[row*K+k] and C[row*N+col] strided by K and N across the warp
// (uncoalesced). Step 2 fixes exactly this by mapping threadIdx.x -> col.
#pragma once
#include "common.cuh"

__global__ void naive_kernel(int M, int N, int K, float alpha,
                             const float* A, const float* B, float beta, float* C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;   // <-- threadIdx.x drives row (uncoalesced)
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

inline void run_naive(int M, int N, int K, float alpha,
                      const float* A, const float* B, float beta, float* C) {
    dim3 block(32, 32);
    dim3 grid((M + 31) / 32, (N + 31) / 32);
    naive_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

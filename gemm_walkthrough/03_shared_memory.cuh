// Step 3 — Shared-memory tiling (cache blocking)
// ----------------------------------------------------------------------------
// Reuse! A block cooperatively loads a TILE x TILE chunk of A and of B into
// shared memory, then every thread reuses those cached values TILE times
// before loading the next chunk. Global-memory traffic drops ~TILE-fold.
//
// Here every phase (load A, load B, compute) uses the SAME 32x32 shape, so a
// 2D thread block (threadIdx.x, threadIdx.y) is the natural, clean choice.
// From step 4 on, the phases tile the threads into DIFFERENT shapes, so we
// switch to a 1D linear index and derive coordinates with / and % (the
// standard GEMM idiom).
//
// NOTE: from step 3 onward the kernels assume M, N, K are multiples of the tile
// sizes (see README). This keeps the index math clean and free of bounds checks.
#pragma once
#include "common.cuh"

template <int TILE>
__global__ void shared_kernel(int M, int N, int K, float alpha,
                              const float* A, const float* B, float beta, float* C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    // Slide a TILE-wide window along the K dimension.
    for (int kTile = 0; kTile < K; kTile += TILE) {
        // Each thread loads one element of A and one of B into shared memory.
        As[threadIdx.y][threadIdx.x] = A[row * K + (kTile + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(kTile + threadIdx.y) * N + col];
        __syncthreads();                       // tile fully loaded

        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();                       // done before overwriting the tile
    }

    C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

inline void run_shared(int M, int N, int K, float alpha,
                       const float* A, const float* B, float beta, float* C) {
    constexpr int TILE = 32;
    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, M / TILE);
    shared_kernel<TILE><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

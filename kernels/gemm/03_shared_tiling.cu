#include <cuda_runtime.h>
#include "gemm/03_shared_tiling.cuh"

#define TILE_SIZE 32

__global__ void sgemm_shared_tiling_kernel(int M, int N, int K,
                                           const float* A, const float* B, float* C) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load A tile: A[row][t*TILE + threadIdx.x]
        int a_col = t * TILE_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;

        // Load B tile: B[t*TILE + threadIdx.y][col]
        int b_row = t * TILE_SIZE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // Partial dot product from this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void run_sgemm_shared_tiling(int M, int N, int K,
                             const float* A, const float* B, float* C) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);
    sgemm_shared_tiling_kernel<<<grid, block>>>(M, N, K, A, B, C);
}

#include <cuda_runtime.h>
#include "gemm/04_reg_tiling_1d.cuh"

#define BM 64
#define BN 64
#define BK 8
#define TM 8

__global__ void sgemm_reg_tiling_1d_kernel(int M, int N, int K,
                                           const float* A, const float* B, float* C) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Block position in output matrix
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    // Thread indexing: 512 threads total, linearized
    // Each thread computes TM=8 elements along M for one column
    const int tid = threadIdx.x;
    const int thread_row = tid / BN;        // 0..7 (BM/TM = 8 rows of threads)
    const int thread_col = tid % BN;        // 0..63

    // Starting position in C for this thread's micro-column
    const int row_base = block_row * BM + thread_row * TM;
    const int col = block_col * BN + thread_col;

    // Accumulators in registers
    float accum[TM] = {0.0f};

    // Tile loading indices: 512 threads load BM*BK=512 A elements, BK*BN=512 B elements
    // Each thread loads 1 element of A and 1 element of B
    const int a_tid_row = tid / BK;         // 0..63
    const int a_tid_col = tid % BK;         // 0..7
    const int b_tid_row = tid / BN;         // 0..7
    const int b_tid_col = tid % BN;         // 0..63

    const int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        // Load A tile: As[BM][BK]
        int a_row = block_row * BM + a_tid_row;
        int a_col = t * BK + a_tid_col;
        As[a_tid_row][a_tid_col] = (a_row < M && a_col < K)
            ? A[a_row * K + a_col] : 0.0f;

        // Load B tile: Bs[BK][BN]
        int b_row = t * BK + b_tid_row;
        int b_col = block_col * BN + b_tid_col;
        Bs[b_tid_row][b_tid_col] = (b_row < K && b_col < N)
            ? B[b_row * N + b_col] : 0.0f;

        __syncthreads();

        // Compute: each thread accumulates TM=8 results
        for (int k = 0; k < BK; ++k) {
            float b_val = Bs[k][thread_col];
            for (int tm = 0; tm < TM; ++tm) {
                accum[tm] += As[thread_row * TM + tm][k] * b_val;
            }
        }

        __syncthreads();
    }

    // Write results
    for (int tm = 0; tm < TM; ++tm) {
        int out_row = row_base + tm;
        if (out_row < M && col < N) {
            C[out_row * N + col] = accum[tm];
        }
    }
}

void run_sgemm_reg_tiling_1d(int M, int N, int K,
                             const float* A, const float* B, float* C) {
    // BM/TM * BN = 8 * 64 = 512 threads per block
    dim3 block(BM / TM * BN);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_reg_tiling_1d_kernel<<<grid, block>>>(M, N, K, A, B, C);
}

#undef BM
#undef BN
#undef BK
#undef TM

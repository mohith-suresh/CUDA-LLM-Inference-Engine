#include <cuda_runtime.h>
#include "gemm/05_reg_tiling_2d.cuh"

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_reg_tiling_2d_kernel(int M, int N, int K,
                                           const float* A, const float* B, float* C) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    // 256 threads: 16x16 logical grid of threads
    const int tid = threadIdx.x;
    const int thread_row = tid / (BN / TN);  // 0..15
    const int thread_col = tid % (BN / TN);  // 0..15

    // Starting position in C for this thread's 8x8 micro-tile
    const int row_base = block_row * BM + thread_row * TM;
    const int col_base = block_col * BN + thread_col * TN;

    // 64 accumulators for the 8x8 micro-tile
    float accum[TM][TN] = {{0.0f}};
    float a_reg[TM];
    float b_reg[TN];

    // Loading indices: 256 threads load BM*BK = 1024 A elements (4 per thread)
    // and BK*BN = 1024 B elements (4 per thread)
    const int a_elements = (BM * BK) / (BM / TM * BN / TN);  // 4
    const int b_elements = (BK * BN) / (BM / TM * BN / TN);  // 4

    const int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        // Load A tile: As[128][8] — each thread loads 4 elements
        for (int i = 0; i < a_elements; ++i) {
            int linear = tid * a_elements + i;  // 0..1023
            int a_s_row = linear / BK;          // 0..127
            int a_s_col = linear % BK;          // 0..7
            int a_g_row = block_row * BM + a_s_row;
            int a_g_col = t * BK + a_s_col;
            As[a_s_row][a_s_col] = (a_g_row < M && a_g_col < K)
                ? A[a_g_row * K + a_g_col] : 0.0f;
        }

        // Load B tile: Bs[8][128] — each thread loads 4 elements
        for (int i = 0; i < b_elements; ++i) {
            int linear = tid * b_elements + i;  // 0..1023
            int b_s_row = linear / BN;          // 0..7
            int b_s_col = linear % BN;          // 0..127
            int b_g_row = t * BK + b_s_row;
            int b_g_col = block_col * BN + b_s_col;
            Bs[b_s_row][b_s_col] = (b_g_row < K && b_g_col < N)
                ? B[b_g_row * N + b_g_col] : 0.0f;
        }

        __syncthreads();

        // Compute: outer product accumulation
        for (int k = 0; k < BK; ++k) {
            // Load A column into registers
            for (int tm = 0; tm < TM; ++tm) {
                a_reg[tm] = As[thread_row * TM + tm][k];
            }
            // Load B row into registers
            for (int tn = 0; tn < TN; ++tn) {
                b_reg[tn] = Bs[k][thread_col * TN + tn];
            }
            // Outer product
            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; ++tn) {
                    accum[tm][tn] += a_reg[tm] * b_reg[tn];
                }
            }
        }

        __syncthreads();
    }

    // Write results
    for (int tm = 0; tm < TM; ++tm) {
        for (int tn = 0; tn < TN; ++tn) {
            int out_row = row_base + tm;
            int out_col = col_base + tn;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = accum[tm][tn];
            }
        }
    }
}

void run_sgemm_reg_tiling_2d(int M, int N, int K,
                             const float* A, const float* B, float* C) {
    // (BM/TM) * (BN/TN) = 16 * 16 = 256 threads
    dim3 block((BM / TM) * (BN / TN));
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_reg_tiling_2d_kernel<<<grid, block>>>(M, N, K, A, B, C);
}

#undef BM
#undef BN
#undef BK
#undef TM
#undef TN

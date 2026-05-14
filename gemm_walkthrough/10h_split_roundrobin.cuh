// Step 10h — column-split round-robin: give each thread a SQUARE-ish region
// ----------------------------------------------------------------------------
// The plain round-robin (10f/10f2) stacks a thread's tiles into a 32 x MT_N
// vertical strip (all tiles share a column) -> worst aspect ratio: 32+4 = 36
// shared loads/k and lots of A-registers -> low occupancy.
//
// Fix (this file): split the block columns into NSPLIT halves and round-robin
// EACH half separately. Now a thread collects tiles from every column-half, so
// its outputs span C_GROUPS = NSPLIT distinct column-groups AND R_GROUPS row-
// groups -> an effective (R_GROUPS*MT_M) x (C_GROUPS*MT_N) register tile.
//
//   NSPLIT=2, MT_M=8, MT_N=4, 128 threads  ->  R_GROUPS=2, C_GROUPS=2
//   effective tile = 16 x 8  ->  16+8 = 24 shared loads/k  (== warptiling)
//
// This is the same reuse warptiling gets, reached from the round-robin framing:
// spreading the thread's work in 2D (rows AND cols) instead of a 1D strip.
#pragma once
#include "common.cuh"

template <int BM, int BN, int BK, int MT_M, int MT_N, int NSPLIT, int NUM_THREADS, int MINB = 1>
__global__ void __launch_bounds__(NUM_THREADS, MINB)
split_roundrobin_kernel(int M, int N, int K, float alpha,
                        const float* A, const float* B, float beta, float* C) {
    static_assert(MT_N % 4 == 0, "tile width multiple of 4 for float4 stores");
    static_assert(BN % NSPLIT == 0, "columns split evenly");
    constexpr int HALF_COLS      = BN / NSPLIT;
    constexpr int TILE_COLS_HALF = HALF_COLS / MT_N;
    constexpr int TILE_ROWS      = BM / MT_M;
    constexpr int TILES_PER_HALF = TILE_ROWS * TILE_COLS_HALF;
    constexpr int R_GROUPS       = TILES_PER_HALF / NUM_THREADS;   // row-groups/thread
    constexpr int C_GROUPS       = NSPLIT;                          // col-groups/thread
    constexpr int NROWS          = R_GROUPS * MT_M;                 // effective tile height
    constexpr int NCOLS          = C_GROUPS * MT_N;                 // effective tile width
    static_assert(TILES_PER_HALF % NUM_THREADS == 0, "threads divide a half");
    static_assert(NUM_THREADS % TILE_COLS_HALF == 0, "share a column within a half");

    __shared__ float As[BK * BM];      // transposed As[k][m]
    __shared__ float Bs[BK * BN];

    const int blockRow0 = blockIdx.y * BM, blockCol0 = blockIdx.x * BN;
    A += blockRow0 * K;
    B += blockCol0;
    C += blockRow0 * N + blockCol0;

    // This thread's row-groups (from the per-half round-robin) and col-groups
    // (one per split). All tiles in a half share the same in-half column.
    const int colInHalf = (threadIdx.x % TILE_COLS_HALF) * MT_N;
    int rowGroup[R_GROUPS], colGroup[C_GROUPS];
    #pragma unroll
    for (int r = 0; r < R_GROUPS; ++r)
        rowGroup[r] = ((threadIdx.x + r * NUM_THREADS) / TILE_COLS_HALF) * MT_M;
    #pragma unroll
    for (int c = 0; c < C_GROUPS; ++c)
        colGroup[c] = c * HALF_COLS + colInHalf;

    float acc[NROWS * NCOLS] = {0.0f};

    constexpr int A_VEC_COLS = BK / 4, B_VEC_COLS = BN / 4;
    constexpr int A_VEC_CNT  = BM * A_VEC_COLS, B_VEC_CNT = BK * B_VEC_COLS;

    for (int kTile = 0; kTile < K; kTile += BK) {
        for (int vi = threadIdx.x; vi < A_VEC_CNT; vi += NUM_THREADS) {
            const int row = vi / A_VEC_COLS, col4 = vi % A_VEC_COLS;
            const float4 v = *reinterpret_cast<const float4*>(&A[row * K + col4 * 4]);
            As[(col4 * 4 + 0) * BM + row] = v.x;
            As[(col4 * 4 + 1) * BM + row] = v.y;
            As[(col4 * 4 + 2) * BM + row] = v.z;
            As[(col4 * 4 + 3) * BM + row] = v.w;
        }
        for (int vi = threadIdx.x; vi < B_VEC_CNT; vi += NUM_THREADS) {
            const int row = vi / B_VEC_COLS, col4 = vi % B_VEC_COLS;
            *reinterpret_cast<float4*>(&Bs[row * BN + col4 * 4]) =
                *reinterpret_cast<const float4*>(&B[row * N + col4 * 4]);
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float regA[NROWS], regB[NCOLS];
            #pragma unroll
            for (int r = 0; r < R_GROUPS; ++r)
                #pragma unroll
                for (int i = 0; i < MT_M; ++i)
                    regA[r * MT_M + i] = As[k * BM + rowGroup[r] + i];
            #pragma unroll
            for (int c = 0; c < C_GROUPS; ++c)
                #pragma unroll
                for (int j = 0; j < MT_N; ++j)
                    regB[c * MT_N + j] = Bs[k * BN + colGroup[c] + j];
            #pragma unroll
            for (int ai = 0; ai < NROWS; ++ai)
                #pragma unroll
                for (int bj = 0; bj < NCOLS; ++bj)
                    acc[ai * NCOLS + bj] += regA[ai] * regB[bj];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
    }

    // Epilogue: float4 stores, one per (row, col-group).
    #pragma unroll
    for (int r = 0; r < R_GROUPS; ++r)
        #pragma unroll
        for (int i = 0; i < MT_M; ++i) {
            const int grow = rowGroup[r] + i;
            #pragma unroll
            for (int c = 0; c < C_GROUPS; ++c)
                #pragma unroll
                for (int jj = 0; jj < MT_N; jj += 4) {
                    float* cptr = &C[grow * N + colGroup[c] + jj];
                    const float4 old = *reinterpret_cast<const float4*>(cptr);
                    const int base = (r * MT_M + i) * NCOLS + c * MT_N + jj;
                    float4 v;
                    v.x = alpha * acc[base + 0] + beta * old.x;
                    v.y = alpha * acc[base + 1] + beta * old.y;
                    v.z = alpha * acc[base + 2] + beta * old.z;
                    v.w = alpha * acc[base + 3] + beta * old.w;
                    *reinterpret_cast<float4*>(cptr) = v;
                }
        }
}

// 2 column-halves, 8x4 tiles round-robin per half -> effective 16x8 tile/thread.
inline void run_split2_roundrobin(int M, int N, int K, float alpha,
                                  const float* A, const float* B, float beta, float* C) {
    constexpr int BM = 128, BN = 128, BK = 16, NUM_THREADS = 128;
    dim3 block(NUM_THREADS);
    dim3 grid(N / BN, M / BM);
    split_roundrobin_kernel<BM, BN, BK, /*MT_M=*/8, /*MT_N=*/4, /*NSPLIT=*/2, NUM_THREADS>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

// Same design, but ask the compiler for 3 blocks/SM (caps regs at ~170) to lift
// occupancy from 16.7% -> 24% (matching warptiling's 3-block budget).
inline void run_split2_roundrobin_occ(int M, int N, int K, float alpha,
                                      const float* A, const float* B, float beta, float* C) {
    constexpr int BM = 128, BN = 128, BK = 16, NUM_THREADS = 128;
    dim3 block(NUM_THREADS);
    dim3 grid(N / BN, M / BM);
    split_roundrobin_kernel<BM, BN, BK, 8, 4, 2, NUM_THREADS, /*MINB=*/3>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

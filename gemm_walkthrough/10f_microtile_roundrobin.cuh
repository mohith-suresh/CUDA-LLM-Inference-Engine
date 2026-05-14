// Step 10f — MT_M x MT_N micro-tiles, ROUND-ROBIN assignment (not warp-organized)
// ----------------------------------------------------------------------------
// Divide the 128x128 block into MT_M x MT_N micro-tiles, linearize them top-left
// row-major, and hand tile i to thread (i % NUM_THREADS):
//
//   tiles 0..127 -> threads 0..127, then tile 128 -> thread 0, etc.
//
// Each thread owns NUM_TILES/NUM_THREADS scattered tiles. Provided NUM_THREADS is
// a multiple of the tile-column count, all of a thread's tiles share the SAME
// column-group, so B is loaded ONCE per k (not per tile). Each tile is MT_N-wide
// (a multiple of 4) so the C store is float4 and, since consecutive threads land
// on consecutive tile-columns, coalesced.
//
// This is the "flat cyclic" layout, deliberately NOT the compact warp tile of
// step 10. In these configs the tiles degenerate to a (scattered) 32 x MT_N strip
// per thread, so the ceiling is the contiguous 32x4 kernel (10g), not warptiling.
#pragma once
#include "common.cuh"

template <int BM, int BN, int BK, int MT_M, int MT_N, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
roundrobin_microtile_kernel(int M, int N, int K, float alpha,
                            const float* A, const float* B, float beta, float* C) {
    static_assert(MT_N % 4 == 0, "tile width must be a multiple of 4 for float4 stores");
    constexpr int TILE_ROWS = BM / MT_M;
    constexpr int TILE_COLS = BN / MT_N;
    constexpr int NUM_TILES = TILE_ROWS * TILE_COLS;
    constexpr int TPT       = NUM_TILES / NUM_THREADS;   // tiles per thread
    static_assert(NUM_TILES % NUM_THREADS == 0, "threads must divide micro-tiles");
    static_assert(NUM_THREADS % TILE_COLS == 0, "tiles must share a column-group (B once)");

    __shared__ float As[BK * BM];      // transposed As[k][m]
    __shared__ float Bs[BK * BN];

    const int blockRow0 = blockIdx.y * BM, blockCol0 = blockIdx.x * BN;
    A += blockRow0 * K;
    B += blockCol0;
    C += blockRow0 * N + blockCol0;

    // Round-robin micro-tile assignment: thread owns tiles { tid, tid+NT, ... }.
    int rowBase[TPT], colBase[TPT];
    #pragma unroll
    for (int s = 0; s < TPT; ++s) {
        const int mt = threadIdx.x + s * NUM_THREADS;
        rowBase[s] = (mt / TILE_COLS) * MT_M;
        colBase[s] = (mt % TILE_COLS) * MT_N;
    }

    float acc[TPT * MT_M * MT_N] = {0.0f};

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
            // B loaded ONCE per k — all of this thread's tiles share the column.
            float rb[MT_N];
            #pragma unroll
            for (int j = 0; j < MT_N; ++j) rb[j] = Bs[k * BN + colBase[0] + j];
            #pragma unroll
            for (int s = 0; s < TPT; ++s) {
                float ra[MT_M];
                #pragma unroll
                for (int i = 0; i < MT_M; ++i) ra[i] = As[k * BM + rowBase[s] + i];
                #pragma unroll
                for (int i = 0; i < MT_M; ++i)
                    #pragma unroll
                    for (int j = 0; j < MT_N; ++j)
                        acc[s * MT_M * MT_N + i * MT_N + j] += ra[i] * rb[j];
            }
        }
        __syncthreads();
        A += BK;
        B += BK * N;
    }

    // Epilogue: float4 stores per (tile, row). Consecutive threads -> consecutive
    // tile-columns -> coalesced.
    #pragma unroll
    for (int s = 0; s < TPT; ++s) {
        #pragma unroll
        for (int i = 0; i < MT_M; ++i) {
            #pragma unroll
            for (int j = 0; j < MT_N; j += 4) {
                float* cptr = &C[(rowBase[s] + i) * N + colBase[s] + j];
                const float4 old = *reinterpret_cast<const float4*>(cptr);
                const int base = s * MT_M * MT_N + i * MT_N + j;
                float4 v;
                v.x = alpha * acc[base + 0] + beta * old.x;
                v.y = alpha * acc[base + 1] + beta * old.y;
                v.z = alpha * acc[base + 2] + beta * old.z;
                v.w = alpha * acc[base + 3] + beta * old.w;
                *reinterpret_cast<float4*>(cptr) = v;
            }
        }
    }
}

// 4x4 micro-tiles: 1024 tiles / 128 threads = 8 tiles/thread.
inline void run_roundrobin_4x4(int M, int N, int K, float alpha,
                               const float* A, const float* B, float beta, float* C) {
    constexpr int BM = 128, BN = 128, BK = 16, NUM_THREADS = 128;
    dim3 block(NUM_THREADS);
    dim3 grid(N / BN, M / BM);
    roundrobin_microtile_kernel<BM, BN, BK, 4, 4, NUM_THREADS>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

// 8x4 micro-tiles: 512 tiles / 128 threads = 4 tiles/thread (bigger row-groups).
inline void run_roundrobin_8x4(int M, int N, int K, float alpha,
                               const float* A, const float* B, float beta, float* C) {
    constexpr int BM = 128, BN = 128, BK = 16, NUM_THREADS = 128;
    dim3 block(NUM_THREADS);
    dim3 grid(N / BN, M / BM);
    roundrobin_microtile_kernel<BM, BN, BK, 8, 4, NUM_THREADS>
        <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

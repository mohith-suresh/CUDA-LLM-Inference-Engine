#include <cuda_runtime.h>
#include "gemm/06_vectorized.cuh"

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_vectorized_kernel(int M, int N, int K,
                                        const float* A, const float* B, float* C) {
    // Transposed A storage for bank-conflict-free column access
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    const int tid = threadIdx.x;
    const int thread_row = tid / (BN / TN);  // 0..15
    const int thread_col = tid % (BN / TN);  // 0..15

    const int row_base = block_row * BM + thread_row * TM;
    const int col_base = block_col * BN + thread_col * TN;

    float accum[TM][TN] = {{0.0f}};
    float a_reg[TM];
    float b_reg[TN];

    // Loading with float4: 256 threads, each loads 4 floats per tile
    // A tile: BM×BK = 128×8 = 1024 floats = 256 float4s → 1 float4 per thread
    // B tile: BK×BN = 8×128 = 1024 floats = 256 float4s → 1 float4 per thread
    // A: load float4 along K dimension (row of 8 → 2 float4s, but we pack differently)
    // A has 128 rows × 8 cols. Group: each float4 covers 4 consecutive row elements in same col.
    // Actually: load A as 128×8, viewing as 256 float4s: linear_id → row/col mapping

    // For A: we load 4 elements along M (rows), store transposed into As[k][m]
    // 1024 elements / 256 threads = 4 elements per thread (scalar loads, transposed store)
    const int a_elements = 4;
    // For B: we load float4 along N (cols), 1024/4 = 256 float4 loads, 1 per thread
    // B row has 128 cols = 32 float4s, 8 rows → 256 float4s total

    const int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        // Load A tile with transposed store: A[m][k] → As[k][m]
        // Each thread loads 4 elements
        for (int i = 0; i < a_elements; ++i) {
            int linear = tid * a_elements + i;
            int a_s_row = linear / BK;   // m index: 0..127
            int a_s_col = linear % BK;   // k index: 0..7
            int a_g_row = block_row * BM + a_s_row;
            int a_g_col = t * BK + a_s_col;
            float val = (a_g_row < M && a_g_col < K)
                ? A[a_g_row * K + a_g_col] : 0.0f;
            As[a_s_col][a_s_row] = val;  // transposed store
        }

        // Load B tile with float4: B[k][n] → Bs[k][n]
        {
            int linear = tid;  // 0..255
            int b_s_row = linear / (BN / 4);  // 0..7 (each row = 32 float4s)
            int b_s_col = (linear % (BN / 4)) * 4;  // 0,4,8,...124
            int b_g_row = t * BK + b_s_row;
            int b_g_col = block_col * BN + b_s_col;

            if (b_g_row < K && b_g_col + 3 < N) {
                float4 tmp = reinterpret_cast<const float4*>(&B[b_g_row * N + b_g_col])[0];
                Bs[b_s_row][b_s_col]     = tmp.x;
                Bs[b_s_row][b_s_col + 1] = tmp.y;
                Bs[b_s_row][b_s_col + 2] = tmp.z;
                Bs[b_s_row][b_s_col + 3] = tmp.w;
            } else {
                for (int j = 0; j < 4; ++j) {
                    int col = b_g_col + j;
                    Bs[b_s_row][b_s_col + j] = (b_g_row < K && col < N)
                        ? B[b_g_row * N + col] : 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute: outer product with registers
        for (int k = 0; k < BK; ++k) {
            // Load from transposed As: As[k][m] — sequential access along m
            for (int tm = 0; tm < TM; ++tm) {
                a_reg[tm] = As[k][thread_row * TM + tm];
            }
            for (int tn = 0; tn < TN; ++tn) {
                b_reg[tn] = Bs[k][thread_col * TN + tn];
            }
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

void run_sgemm_vectorized(int M, int N, int K,
                          const float* A, const float* B, float* C) {
    dim3 block((BM / TM) * (BN / TN));  // 256
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_vectorized_kernel<<<grid, block>>>(M, N, K, A, B, C);
}

#undef BM
#undef BN
#undef BK
#undef TM
#undef TN

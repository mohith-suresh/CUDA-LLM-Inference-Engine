#include <cuda_runtime.h>
#include "gemm/07_double_buffered.cuh"

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

// Load A tile: A[m][k] → As[buf][k][m] (transposed for bank-conflict-free access)
__device__ __forceinline__
void load_a_tile(const float* A, float As[][BK][BM],
                 int buf, int block_row, int t, int tid,
                 int M, int K) {
    const int a_elements = 4;
    for (int i = 0; i < a_elements; ++i) {
        int linear = tid * a_elements + i;
        int a_s_row = linear / BK;   // m: 0..127
        int a_s_col = linear % BK;   // k: 0..7
        int a_g_row = block_row * BM + a_s_row;
        int a_g_col = t * BK + a_s_col;
        float val = (a_g_row < M && a_g_col < K)
            ? A[a_g_row * K + a_g_col] : 0.0f;
        As[buf][a_s_col][a_s_row] = val;
    }
}

// Load B tile: B[k][n] → Bs[buf][k][n] with float4
__device__ __forceinline__
void load_b_tile(const float* B, float Bs[][BK][BN],
                 int buf, int block_col, int t, int tid,
                 int K, int N) {
    int linear = tid;
    int b_s_row = linear / (BN / 4);
    int b_s_col = (linear % (BN / 4)) * 4;
    int b_g_row = t * BK + b_s_row;
    int b_g_col = block_col * BN + b_s_col;

    if (b_g_row < K && b_g_col + 3 < N) {
        float4 tmp = reinterpret_cast<const float4*>(&B[b_g_row * N + b_g_col])[0];
        Bs[buf][b_s_row][b_s_col]     = tmp.x;
        Bs[buf][b_s_row][b_s_col + 1] = tmp.y;
        Bs[buf][b_s_row][b_s_col + 2] = tmp.z;
        Bs[buf][b_s_row][b_s_col + 3] = tmp.w;
    } else {
        for (int j = 0; j < 4; ++j) {
            int col = b_g_col + j;
            Bs[buf][b_s_row][b_s_col + j] = (b_g_row < K && col < N)
                ? B[b_g_row * N + col] : 0.0f;
        }
    }
}

// Compute outer product from shared memory buffer
__device__ __forceinline__
void compute_tile(float As[][BK][BM], float Bs[][BK][BN],
                  int buf, int thread_row, int thread_col,
                  float accum[TM][TN]) {
    float a_reg[TM];
    float b_reg[TN];

    for (int k = 0; k < BK; ++k) {
        for (int tm = 0; tm < TM; ++tm) {
            a_reg[tm] = As[buf][k][thread_row * TM + tm];
        }
        for (int tn = 0; tn < TN; ++tn) {
            b_reg[tn] = Bs[buf][k][thread_col * TN + tn];
        }
        for (int tm = 0; tm < TM; ++tm) {
            for (int tn = 0; tn < TN; ++tn) {
                accum[tm][tn] += a_reg[tm] * b_reg[tn];
            }
        }
    }
}

__global__ void sgemm_double_buffered_kernel(int M, int N, int K,
                                             const float* A, const float* B, float* C) {
    // Double-buffered shared memory
    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    const int tid = threadIdx.x;
    const int thread_row = tid / (BN / TN);
    const int thread_col = tid % (BN / TN);

    const int row_base = block_row * BM + thread_row * TM;
    const int col_base = block_col * BN + thread_col * TN;

    float accum[TM][TN] = {{0.0f}};

    const int num_tiles = (K + BK - 1) / BK;

    // Load first tile into buffer 0
    load_a_tile(A, As, 0, block_row, 0, tid, M, K);
    load_b_tile(B, Bs, 0, block_col, 0, tid, K, N);
    __syncthreads();

    // Main loop: compute on current buffer while loading into next
    for (int t = 0; t < num_tiles - 1; ++t) {
        int cur = t & 1;
        int nxt = 1 - cur;

        // Prefetch next tile into alternate buffer
        load_a_tile(A, As, nxt, block_row, t + 1, tid, M, K);
        load_b_tile(B, Bs, nxt, block_col, t + 1, tid, K, N);

        // Compute on current buffer
        compute_tile(As, Bs, cur, thread_row, thread_col, accum);

        __syncthreads();
    }

    // Compute final tile
    compute_tile(As, Bs, (num_tiles - 1) & 1, thread_row, thread_col, accum);

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

void run_sgemm_double_buffered(int M, int N, int K,
                               const float* A, const float* B, float* C) {
    dim3 block((BM / TM) * (BN / TN));  // 256
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_double_buffered_kernel<<<grid, block>>>(M, N, K, A, B, C);
}

#undef BM
#undef BN
#undef BK
#undef TM
#undef TN

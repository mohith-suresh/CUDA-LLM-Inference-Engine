#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include "softmax/09_warp_reduce.cuh"
#include "timer.cuh"

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8

// Device: warp-level merge reduction of (max, sum_exp) pairs
__device__ __forceinline__
void warp_reduce_md(float& m, float& d) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(0xFFFFFFFF, m, offset);
        float d2 = __shfl_down_sync(0xFFFFFFFF, d, offset);
        float new_m = fmaxf(m, m2);
        d = d * __expf(m - new_m) + d2 * __expf(m2 - new_m);
        m = new_m;
    }
}

__global__
void softmax_warp_reduce_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int N_rows, int N_cols) {
    int row = blockIdx.x * WARPS_PER_BLOCK + threadIdx.y;
    if (row >= N_rows) return;

    const float* row_in = input + row * N_cols;
    float* row_out = output + row * N_cols;
    int lane = threadIdx.x;

    // Pass 1: accumulate local (max, sum_exp)
    float m = -FLT_MAX;
    float d = 0.0f;

    for (int col = lane; col < N_cols; col += WARP_SIZE) {
        float x = row_in[col];
        float old_m = m;
        m = fmaxf(m, x);
        d = d * __expf(old_m - m) + __expf(x - m);
    }

    // Warp-level reduction via shuffle
    warp_reduce_md(m, d);

    // Broadcast from lane 0
    float m_global = __shfl_sync(0xFFFFFFFF, m, 0);
    float d_global = __shfl_sync(0xFFFFFFFF, d, 0);

    // Pass 2: normalize
    for (int col = lane; col < N_cols; col += WARP_SIZE) {
        float x = row_in[col];
        row_out[col] = __expf(x - m_global) / d_global;
    }
}

void run_softmax_warp_reduce(const float* input, float* output,
                              int N_rows, int N_cols) {
    int grid_rows = (N_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(grid_rows);
    dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
    softmax_warp_reduce_kernel<<<grid, block>>>(input, output, N_rows, N_cols);
}

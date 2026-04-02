#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include "softmax/08_fused_online.cuh"
#include "timer.cuh"

#define BLOCK_SIZE 256

// Device: merge two (max, sum_exp) pairs
__device__ __forceinline__
void merge_pair(float m1, float d1, float m2, float d2,
                float& m_out, float& d_out) {
    m_out = fmaxf(m1, m2);
    d_out = d1 * __expf(m1 - m_out) + d2 * __expf(m2 - m_out);
}

__global__
void softmax_fused_online_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int N_rows, int N_cols) {
    int row = blockIdx.x;
    if (row >= N_rows) return;

    const float* row_in = input + row * N_cols;
    float* row_out = output + row * N_cols;
    int tid = threadIdx.x;

    // Pass 1: accumulate local (max, sum_exp)
    float m = -FLT_MAX;
    float d = 0.0f;

    for (int col = tid; col < N_cols; col += BLOCK_SIZE) {
        float x = row_in[col];
        float old_m = m;
        m = fmaxf(m, x);
        d = d * __expf(old_m - m) + __expf(x - m);
    }

    // Shared memory reduction of (m, d) pairs
    __shared__ float s_m[BLOCK_SIZE];
    __shared__ float s_d[BLOCK_SIZE];
    s_m[tid] = m;
    s_d[tid] = d;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float m1 = s_m[tid], d1 = s_d[tid];
            float m2 = s_m[tid + stride], d2 = s_d[tid + stride];
            merge_pair(m1, d1, m2, d2, s_m[tid], s_d[tid]);
        }
        __syncthreads();
    }

    float m_global = s_m[0];
    float d_global = s_d[0];

    // Pass 2: normalize
    for (int col = tid; col < N_cols; col += BLOCK_SIZE) {
        float x = row_in[col];
        row_out[col] = __expf(x - m_global) / d_global;
    }
}

void run_softmax_fused_online(const float* input, float* output,
                               int N_rows, int N_cols) {
    dim3 grid(N_rows);
    dim3 block(BLOCK_SIZE);
    softmax_fused_online_kernel<<<grid, block>>>(input, output, N_rows, N_cols);
}

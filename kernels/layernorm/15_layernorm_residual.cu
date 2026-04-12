// kernels/layernorm/15_layernorm_residual.cu
#include "layernorm/15_layernorm_residual.cuh"

#define LN_BLOCK 256

__global__ void layernorm_residual_kernel(
    int cols,
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    float* __restrict__ residual_out,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* x_row = x + row * cols;
    const float* res_row = residual + row * cols;
    float* out_row = out + row * cols;
    float* res_out_row = residual_out + row * cols;

    extern __shared__ float smem[];

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    for (int c = tid; c < cols; c += LN_BLOCK) {
        float y = x_row[c] + res_row[c];
        res_out_row[c] = y;
        local_sum += y;
        local_sum_sq += y * y;
    }

    smem[tid] = local_sum;
    __syncthreads();
    for (int s = LN_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float mean = smem[0] / cols;
    __syncthreads();

    smem[tid] = local_sum_sq;
    __syncthreads();
    for (int s = LN_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float var = smem[0] / cols - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int c = tid; c < cols; c += LN_BLOCK) {
        float y = res_out_row[c];
        out_row[c] = gamma[c] * (y - mean) * inv_std + beta[c];
    }
}

__global__ void layernorm_kernel(
    int cols,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* x_row = x + row * cols;
    float* out_row = out + row * cols;

    extern __shared__ float smem[];

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int c = tid; c < cols; c += LN_BLOCK) {
        float v = x_row[c];
        local_sum += v;
        local_sum_sq += v * v;
    }

    smem[tid] = local_sum;
    __syncthreads();
    for (int s = LN_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float mean = smem[0] / cols;
    __syncthreads();

    smem[tid] = local_sum_sq;
    __syncthreads();
    for (int s = LN_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float var = smem[0] / cols - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int c = tid; c < cols; c += LN_BLOCK) {
        out_row[c] = gamma[c] * (x_row[c] - mean) * inv_std + beta[c];
    }
}

void run_layernorm_residual(int rows, int cols,
                             const float* x,
                             const float* residual,
                             const float* gamma,
                             const float* beta,
                             float* out,
                             float* residual_out,
                             float eps) {
    dim3 grid(rows);
    dim3 block(LN_BLOCK);
    int smem = LN_BLOCK * sizeof(float);
    layernorm_residual_kernel<<<grid, block, smem>>>(
        cols, x, residual, gamma, beta, out, residual_out, eps);
}

void run_layernorm(int rows, int cols,
                    const float* x,
                    const float* gamma,
                    const float* beta,
                    float* out,
                    float eps) {
    dim3 grid(rows);
    dim3 block(LN_BLOCK);
    int smem = LN_BLOCK * sizeof(float);
    layernorm_kernel<<<grid, block, smem>>>(cols, x, gamma, beta, out, eps);
}

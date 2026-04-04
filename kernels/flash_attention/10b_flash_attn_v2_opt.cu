// kernels/flash_attention/10b_flash_attn_v2_opt.cu
#include <cuda_runtime.h>
#include "flash_attention/10b_flash_attn_v2_opt.cuh"
#include "timer.cuh"

void run_flash_attn_v2_opt(int B, int H, int N, int d,
                           const float* Q, const float* K, const float* V,
                           float* O, bool causal) {
    CUDA_CHECK(cudaMemset(O, 0, (size_t)B * H * N * d * sizeof(float)));
}

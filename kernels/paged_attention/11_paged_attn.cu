// kernels/paged_attention/11_paged_attn.cu
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include "paged_attention/11_paged_attn.cuh"
#include "timer.cuh"

void run_paged_attn(int B, int H, int N, int d,
                    const float* Q,
                    const float* k_cache, const float* v_cache,
                    const int* block_table, const int* context_lens,
                    int max_context_len, int block_size,
                    int num_blocks_per_seq,
                    float* O, bool causal) {
    // Stub - to be implemented in Task 3
}

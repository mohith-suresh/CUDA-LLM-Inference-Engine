// kernels/paged_attention/11_paged_attn.cu
#include "paged_attention/11_paged_attn.cuh"

void run_paged_attn(int B, int H, int N, int d,
                    const float* Q,
                    const float* k_cache, const float* v_cache,
                    const int* block_table, const int* context_lens,
                    int max_context_len, int block_size,
                    int num_blocks_per_seq,
                    float* O, bool causal) {
    float scale = 1.0f / sqrtf((float)d);
    int num_q_tiles = (N + PA_BR - 1) / PA_BR;
    dim3 grid(B, H, num_q_tiles);
    dim3 block(PA_NTHREADS);

    // H_kv = H for MHA (GROUP_SIZE=1)
    if (causal)
        paged_attn_kernel<1, true><<<grid, block>>>(
            N, d, scale, Q, k_cache, v_cache,
            block_table, context_lens,
            block_size, num_blocks_per_seq, H, O);
    else
        paged_attn_kernel<1, false><<<grid, block>>>(
            N, d, scale, Q, k_cache, v_cache,
            block_table, context_lens,
            block_size, num_blocks_per_seq, H, O);
}

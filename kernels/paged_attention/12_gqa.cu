// kernels/paged_attention/12_gqa.cu
#include "paged_attention/11_paged_attn.cuh"
#include "paged_attention/12_gqa.cuh"

template <int GROUP_SIZE>
static void launch_gqa(int B, int H_q, int H_kv, int N, int d, float scale,
                       const float* Q,
                       const float* k_cache, const float* v_cache,
                       const int* block_table, const int* context_lens,
                       int block_size, int num_blocks_per_seq,
                       float* O, bool causal) {
    int num_q_tiles = (N + PA_BR - 1) / PA_BR;
    dim3 grid(B, H_q, num_q_tiles);
    dim3 block(PA_NTHREADS);

    if (causal)
        paged_attn_kernel<GROUP_SIZE, true><<<grid, block>>>(
            N, d, scale, Q, k_cache, v_cache,
            block_table, context_lens,
            block_size, num_blocks_per_seq, H_kv, O);
    else
        paged_attn_kernel<GROUP_SIZE, false><<<grid, block>>>(
            N, d, scale, Q, k_cache, v_cache,
            block_table, context_lens,
            block_size, num_blocks_per_seq, H_kv, O);
}

void run_gqa_paged_attn(int B, int H_q, int H_kv, int N, int d,
                        const float* Q,
                        const float* k_cache, const float* v_cache,
                        const int* block_table, const int* context_lens,
                        int max_context_len, int block_size,
                        int num_blocks_per_seq,
                        float* O, bool causal) {
    float scale = 1.0f / sqrtf((float)d);
    int group_size = H_q / H_kv;

    switch (group_size) {
        case 1: launch_gqa<1>(B, H_q, H_kv, N, d, scale, Q, k_cache, v_cache,
                    block_table, context_lens, block_size,
                    num_blocks_per_seq, O, causal); break;
        case 2: launch_gqa<2>(B, H_q, H_kv, N, d, scale, Q, k_cache, v_cache,
                    block_table, context_lens, block_size,
                    num_blocks_per_seq, O, causal); break;
        case 4: launch_gqa<4>(B, H_q, H_kv, N, d, scale, Q, k_cache, v_cache,
                    block_table, context_lens, block_size,
                    num_blocks_per_seq, O, causal); break;
        case 8: launch_gqa<8>(B, H_q, H_kv, N, d, scale, Q, k_cache, v_cache,
                    block_table, context_lens, block_size,
                    num_blocks_per_seq, O, causal); break;
        default: break;
    }
}

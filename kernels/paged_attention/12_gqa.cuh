// kernels/paged_attention/12_gqa.cuh
#pragma once

// GQA PagedAttention: multiple Q heads share fewer KV heads
// group_size = H_q / H_kv. Dispatches to template GROUP_SIZE={1,2,4,8}.
void run_gqa_paged_attn(int B, int H_q, int H_kv, int N, int d,
                        const float* Q,
                        const float* k_cache, const float* v_cache,
                        const int* block_table, const int* context_lens,
                        int max_context_len, int block_size,
                        int num_blocks_per_seq,
                        float* O, bool causal);

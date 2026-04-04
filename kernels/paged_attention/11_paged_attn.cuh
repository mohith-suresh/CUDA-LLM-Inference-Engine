// kernels/paged_attention/11_paged_attn.cuh
#pragma once

// PagedAttention: FlashAttention-2 with block-table KV cache indirection
// KV cache layout: [num_physical_blocks][BLOCK_SIZE][H_kv][d]
// Block table: [B][max_blocks_per_seq] maps logical -> physical block index
void run_paged_attn(int B, int H, int N, int d,
                    const float* Q,
                    const float* k_cache, const float* v_cache,
                    const int* block_table, const int* context_lens,
                    int max_context_len, int block_size,
                    int num_blocks_per_seq,
                    float* O, bool causal);

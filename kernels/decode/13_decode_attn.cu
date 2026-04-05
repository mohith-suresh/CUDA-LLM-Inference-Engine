// kernels/decode/13_decode_attn.cu
#include "decode/13_decode_attn.cuh"

void run_decode_attn(int B, int H_q, int H_kv, int d,
                     const float* Q,
                     const float* k_cache, const float* v_cache,
                     const int* block_table, const int* context_lens,
                     int max_context_len, int block_size,
                     int num_blocks_per_seq,
                     float* O,
                     float* workspace) {
    // TODO: implement split-K decode attention
}

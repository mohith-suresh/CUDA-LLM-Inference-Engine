// kernels/decode/13_decode_attn.cuh
#pragma once
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// ============================================================
// Decode Attention constants
// ============================================================
#define DA_HD 64          // Head dimension
#define DA_NTHREADS 256   // Threads per block
#define DA_MAX_SPLITS 16  // Maximum number of KV splits

// Host wrapper: split-K decode attention over paged KV cache
// Q: [B, H_q, 1, d], O: [B, H_q, 1, d]
// workspace: [B, H_q, max_splits, d+2] (o_partial + m + l per split)
void run_decode_attn(int B, int H_q, int H_kv, int d,
                     const float* Q,
                     const float* k_cache, const float* v_cache,
                     const int* block_table, const int* context_lens,
                     int max_context_len, int block_size,
                     int num_blocks_per_seq,
                     float* O,
                     float* workspace);

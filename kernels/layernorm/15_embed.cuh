// kernels/layernorm/15_embed.cuh
#pragma once
#include <cuda_runtime.h>

// Embedding lookup: out[i][j] = wte[token_ids[i]][j] + wpe[start_pos+i][j]
void run_embedding(const int* token_ids, int seq_len, int start_pos, int d_model,
                   const float* wte, const float* wpe, float* out);

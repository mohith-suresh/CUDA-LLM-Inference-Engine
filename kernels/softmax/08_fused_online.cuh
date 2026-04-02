#pragma once

// Fused Online Softmax: one block per row, shared memory reduction
// Online algorithm: single-pass (max, sum_exp) accumulation, then normalize
void run_softmax_fused_online(const float* input, float* output, int N_rows, int N_cols);

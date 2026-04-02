#pragma once

// Warp Reduce Softmax: one warp per row, __shfl_down_sync reduction
// 8 rows per block (block dim = 32x8), zero shared memory for reductions
void run_softmax_warp_reduce(const float* input, float* output, int N_rows, int N_cols);

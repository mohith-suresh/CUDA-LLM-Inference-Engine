#pragma once

// Double Buffered GEMM: 2x shared memory, overlap load and compute
void run_sgemm_double_buffered(int M, int N, int K,
                               const float* A, const float* B, float* C);

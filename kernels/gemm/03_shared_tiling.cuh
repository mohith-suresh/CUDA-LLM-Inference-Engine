#pragma once

// Shared memory tiling GEMM: 32x32 tiles reduce global memory traffic
void run_sgemm_shared_tiling(int M, int N, int K,
                             const float* A, const float* B, float* C);

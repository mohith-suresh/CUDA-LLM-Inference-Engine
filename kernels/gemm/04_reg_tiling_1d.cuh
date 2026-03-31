#pragma once

// 1D Register Tiling GEMM: TM=8, each thread computes 8 elements along M
void run_sgemm_reg_tiling_1d(int M, int N, int K,
                             const float* A, const float* B, float* C);

#pragma once

// 2D Register Tiling GEMM: TM=TN=8, 8x8 micro-tile per thread
void run_sgemm_reg_tiling_2d(int M, int N, int K,
                             const float* A, const float* B, float* C);

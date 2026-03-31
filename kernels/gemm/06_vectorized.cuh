#pragma once

// Vectorized GEMM: float4 loads for 128-bit memory transactions
void run_sgemm_vectorized(int M, int N, int K,
                          const float* A, const float* B, float* C);

#pragma once

// Naive GEMM: 1 thread computes 1 element of C
// threadIdx.x → row (non-coalesced B reads — intentionally slow)
void run_sgemm_naive(int M, int N, int K,
                     const float* A, const float* B, float* C);

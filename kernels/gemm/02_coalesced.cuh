#pragma once

// Coalesced GEMM: threadIdx.x → col so adjacent threads read adjacent B elements
void run_sgemm_coalesced(int M, int N, int K,
                         const float* A, const float* B, float* C);

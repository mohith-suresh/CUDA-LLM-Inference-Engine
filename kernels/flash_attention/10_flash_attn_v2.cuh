// kernels/flash_attention/10_flash_attn_v2.cuh
#pragma once

// FlashAttention-2 forward pass: fused Q@K^T + softmax + P@V
// One thread block (256 threads) per (batch, head) pair
// Tiling: Br=64, Bc=32, d=64. Online softmax with warp shuffle reductions.
void run_flash_attn_v2(int B, int H, int N, int d,
                       const float* Q, const float* K, const float* V,
                       float* O, bool causal);

// kernels/flash_attention/10b_flash_attn_v2_opt.cuh
#pragma once

// Optimized FlashAttention-2 forward pass
// Grid-parallelized across Q tiles: grid(num_q_tiles, H, B)
// Tiling: Br=64, Bc=64, d=64. float4 loads, register-only P via warp shuffle.
void run_flash_attn_v2_opt(int B, int H, int N, int d,
                           const float* Q, const float* K, const float* V,
                           float* O, bool causal);

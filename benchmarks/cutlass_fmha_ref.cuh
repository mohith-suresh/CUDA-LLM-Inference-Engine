// benchmarks/cutlass_fmha_ref.cuh
// CUTLASS FMHA reference wrapper for benchmarking FlashAttention
// Uses CUTLASS example 41 (fused_multi_head_attention) kernel on Sm75 + FP32
#pragma once

#include "kernel_forward.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Instantiate CUTLASS FMHA for float, Sm75, head_dim <= 64
using CutlassFMHA = AttentionKernel<
    float,                    // scalar_t
    cutlass::arch::Sm75,      // ArchTag
    true,                     // isAligned (Q,K,V 128-bit aligned)
    64,                       // kQueriesPerBlock
    64,                       // kKeysPerBlock
    64,                       // kMaxK (head_dim)
    false,                    // kSupportsDropout
    false                     // kSupportsBias
>;

// Transpose BMHK -> BHNK on the host for validation
static void transpose_bmhk_to_bhnk(const float* bmhk, float* bhnk,
                                    int B, int H, int N, int d) {
    for (int b = 0; b < B; ++b)
        for (int n = 0; n < N; ++n)
            for (int h = 0; h < H; ++h)
                for (int k = 0; k < d; ++k) {
                    int src = b * (N * H * d) + n * (H * d) + h * d + k;
                    int dst = b * (H * N * d) + h * (N * d) + n * d + k;
                    bhnk[dst] = bmhk[src];
                }
}

// Run CUTLASS FMHA
// Input Q, K, V: [B, H, N, d] (our BHNK layout) in device memory
// Output O: [B, N, H, d] (CUTLASS BMHK layout) in device memory
//   - use transpose_bmhk_to_bhnk() to convert for validation
static void run_cutlass_fmha(int B, int H, int N, int d,
                              const float* Q, const float* K, const float* V,
                              float* O, bool causal) {
    using Attention = CutlassFMHA;
    typename Attention::Params p;

    p.query_ptr = const_cast<float*>(Q);
    p.key_ptr   = const_cast<float*>(K);
    p.value_ptr = const_cast<float*>(V);
    p.output_ptr = O;
    p.logsumexp_ptr = nullptr;
    p.output_accum_ptr = nullptr;

    if (Attention::kNeedsOutputAccumulatorBuffer) {
        cudaMalloc(&p.output_accum_ptr,
                   (size_t)B * H * N * d * sizeof(typename Attention::output_accum_t));
    }

    p.scale = 1.0f / sqrtf((float)d);

    p.num_heads   = H;
    p.num_batches = B;
    p.head_dim       = d;
    p.head_dim_value = d;
    p.num_queries = N;
    p.num_keys    = N;

    if (causal) {
        p.custom_mask_type = Attention::CausalFromTopLeft;
    }

    // Map our BHNK layout to CUTLASS's BMHK stride convention:
    // In BHNK: offset(b,h,n,k) = b*(H*N*d) + h*(N*d) + n*d + k
    // CUTLASS reads: ptr + b*strideB + n*strideM + h*strideH + k
    p.q_strideH = N * d;        // head stride in BHNK
    p.k_strideH = N * d;
    p.v_strideH = N * d;
    p.q_strideM = d;            // seq-position stride in BHNK
    p.k_strideM = d;
    p.v_strideM = d;
    p.q_strideB = (int64_t)H * N * d;
    p.k_strideB = (int64_t)H * N * d;
    p.v_strideB = (int64_t)H * N * d;

    // Output is always BMHK: offset(b,n,h,k) = b*(N*H*d) + n*(H*d) + h*d + k
    p.o_strideM = H * d;

    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) {
        cudaFuncSetAttribute(kernel_fn,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes);
    }

    if (!Attention::check_supported(p)) {
        fprintf(stderr, "CUTLASS FMHA: kernel does not support these inputs "
                "(B=%d H=%d N=%d d=%d)\n", B, H, N, d);
        return;
    }

    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);

    if (Attention::kNeedsOutputAccumulatorBuffer && p.output_accum_ptr) {
        cudaFree(p.output_accum_ptr);
    }
}

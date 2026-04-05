// kernels/decode/13_decode_attn.cuh
#pragma once
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// ============================================================
// Decode Attention constants
// ============================================================
#define DA_HD 64          // Head dimension
#define DA_NTHREADS 256   // Threads per block (8 warps)
#define DA_MAX_SPLITS 16  // Maximum number of KV splits
#define DA_WARP_SIZE 32

// ============================================================
// Pass 1: Partial attention kernel
// Each block processes a range of KV blocks for one (batch, head) pair.
// Grid: (B, H_q, num_splits)
//
// For each KV token in its assigned range:
//   s = dot(q, k) * scale  (warp-parallel across d)
//   online softmax: track m_partial, l_partial
//   o_partial += exp(s - m) * v
//
// Output to workspace: [B, H_q, num_splits, d+2]
//   workspace[...][0..d-1] = o_partial (unnormalized)
//   workspace[...][d]      = m_partial (max score)
//   workspace[...][d+1]    = l_partial (sum of exp)
// ============================================================
template <int GROUP_SIZE>
__global__ __launch_bounds__(DA_NTHREADS)
void decode_attn_partial_kernel(
    int d, float scale,
    const float* __restrict__ Q,           // [B, H_q, 1, d]
    const float* __restrict__ k_cache,     // [num_phys_blocks, block_size, H_kv, d]
    const float* __restrict__ v_cache,     // [num_phys_blocks, block_size, H_kv, d]
    const int* __restrict__ block_table,   // [B, num_blocks_per_seq]
    const int* __restrict__ context_lens,  // [B]
    int block_size, int num_blocks_per_seq,
    int H_kv, int num_splits, int blocks_per_split,
    float* __restrict__ workspace)         // [B, H_q, num_splits, d+2]
{
    const int batch  = blockIdx.x;
    const int q_head = blockIdx.y;
    const int split  = blockIdx.z;
    const int H_q    = gridDim.y;
    const int kv_head = q_head / GROUP_SIZE;
    const int tid = threadIdx.x;

    const int ctx_len = context_lens[batch];
    const int num_kv_blocks = (ctx_len + block_size - 1) / block_size;

    // This split's KV block range
    const int blk_start = split * blocks_per_split;
    const int blk_end = min(blk_start + blocks_per_split, num_kv_blocks);
    if (blk_start >= num_kv_blocks) return;

    const int* bt = block_table + batch * num_blocks_per_seq;

    // Load query vector into shared memory [d] (float4 vectorized)
    __shared__ float q_smem[DA_HD];
    const float* Q_bh = Q + (static_cast<long long>(batch) * H_q + q_head) * d;
    for (int i = tid; i < d / 4; i += DA_NTHREADS) {
        float4 val = *reinterpret_cast<const float4*>(&Q_bh[i * 4]);
        q_smem[i * 4 + 0] = val.x;
        q_smem[i * 4 + 1] = val.y;
        q_smem[i * 4 + 2] = val.z;
        q_smem[i * 4 + 3] = val.w;
    }
    __syncthreads();

    // Each warp handles different KV tokens, all threads in warp
    // cooperate on the dot product across d.
    // With 8 warps, 8 KV tokens processed in parallel per iteration.
    const int warp_id = tid / DA_WARP_SIZE;
    const int lane_id = tid % DA_WARP_SIZE;
    const int num_warps = DA_NTHREADS / DA_WARP_SIZE;  // 8

    // Per-warp accumulators (in registers)
    float m_acc = -FLT_MAX;  // running max
    float l_acc = 0.0f;       // running sum of exp
    float o_acc[DA_HD / DA_WARP_SIZE];  // each lane owns d/32 = 2 elements (for d=64)
    const int elems_per_lane = d / DA_WARP_SIZE;  // 64/32 = 2

    #pragma unroll
    for (int e = 0; e < elems_per_lane; ++e)
        o_acc[e] = 0.0f;

    // Iterate over KV tokens in this split's range
    // Each warp takes every num_warps-th token
    for (int blk_idx = blk_start; blk_idx < blk_end; ++blk_idx) {
        int phys_block = bt[blk_idx];
        int tokens_in_block = min(block_size, ctx_len - blk_idx * block_size);

        for (int t = warp_id; t < tokens_in_block; t += num_warps) {
            // Compute dot(q, k) across d — each lane handles elems_per_lane elements
            int cache_base = ((phys_block * block_size + t) * H_kv + kv_head) * d;

            float dot = 0.0f;
            #pragma unroll
            for (int e = 0; e < elems_per_lane; ++e) {
                int dim = lane_id * elems_per_lane + e;
                dot += q_smem[dim] * k_cache[cache_base + dim];
            }

            // Warp-level reduction for dot product
            #pragma unroll
            for (int offset = DA_WARP_SIZE / 2; offset >= 1; offset >>= 1)
                dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
            float s = __shfl_sync(0xFFFFFFFF, dot, 0) * scale;

            // Online softmax update
            float m_new = fmaxf(m_acc, s);
            float alpha = __expf(m_acc - m_new);
            float p = __expf(s - m_new);

            // Rescale existing accumulator
            #pragma unroll
            for (int e = 0; e < elems_per_lane; ++e)
                o_acc[e] *= alpha;
            l_acc = l_acc * alpha + p;
            m_acc = m_new;

            // Accumulate p * v
            #pragma unroll
            for (int e = 0; e < elems_per_lane; ++e) {
                int dim = lane_id * elems_per_lane + e;
                o_acc[e] += p * v_cache[cache_base + dim];
            }
        }
    }

    // Reduce across warps in shared memory
    // Each warp writes its (o_acc[], m_acc, l_acc) to smem, then warp 0 merges
    __shared__ float warp_m[8];       // max per warp
    __shared__ float warp_l[8];       // sum per warp
    __shared__ float warp_o[8][DA_HD]; // partial o per warp

    // Each lane writes its owned elements
    #pragma unroll
    for (int e = 0; e < elems_per_lane; ++e)
        warp_o[warp_id][lane_id * elems_per_lane + e] = o_acc[e];

    if (lane_id == 0) {
        warp_m[warp_id] = m_acc;
        warp_l[warp_id] = l_acc;
    }
    __syncthreads();

    // Warp 0 merges all warp results
    if (warp_id == 0) {
        float merged_m = warp_m[0];
        float merged_l = warp_l[0];
        float merged_o[DA_HD / DA_WARP_SIZE];

        #pragma unroll
        for (int e = 0; e < elems_per_lane; ++e)
            merged_o[e] = warp_o[0][lane_id * elems_per_lane + e];

        for (int w = 1; w < num_warps; ++w) {
            float w_m = warp_m[w];
            float w_l = warp_l[w];
            if (w_l == 0.0f) continue;  // warp processed no tokens

            float m_new = fmaxf(merged_m, w_m);
            float alpha = __expf(merged_m - m_new);
            float beta  = __expf(w_m - m_new);

            #pragma unroll
            for (int e = 0; e < elems_per_lane; ++e)
                merged_o[e] = merged_o[e] * alpha
                            + warp_o[w][lane_id * elems_per_lane + e] * beta;

            merged_l = merged_l * alpha + w_l * beta;
            merged_m = m_new;
        }

        // Write to workspace: [B, H_q, num_splits, d+2]
        int ws_offset = ((batch * H_q + q_head) * num_splits + split) * (d + 2);
        #pragma unroll
        for (int e = 0; e < elems_per_lane; ++e)
            workspace[ws_offset + lane_id * elems_per_lane + e] = merged_o[e];

        if (lane_id == 0) {
            workspace[ws_offset + d]     = merged_m;
            workspace[ws_offset + d + 1] = merged_l;
        }
    }
}

// ============================================================
// Pass 2: Reduction kernel
// Merges partial results from all splits using online softmax correction.
// Grid: (B * H_q), one block per (batch, head) pair
// ============================================================
__global__ __launch_bounds__(DA_NTHREADS)
void decode_attn_reduce_kernel(
    int d, int H_q, int num_splits,
    const float* __restrict__ workspace,   // [B, H_q, num_splits, d+2]
    float* __restrict__ O)                 // [B, H_q, 1, d]
{
    const int bh = blockIdx.x;             // flattened (batch, head)
    const int batch = bh / H_q;
    const int head  = bh % H_q;
    const int tid = threadIdx.x;

    if (tid >= d) return;  // only first d threads do work

    const int stride = d + 2;
    int ws_base = (static_cast<long long>(batch) * H_q + head) * num_splits * stride;

    // Read first split
    float m_acc = workspace[ws_base + d];
    float l_acc = workspace[ws_base + d + 1];
    float o_acc = workspace[ws_base + tid];

    // Merge remaining splits
    for (int s = 1; s < num_splits; ++s) {
        int offset = ws_base + s * stride;
        float m_s = workspace[offset + d];
        float l_s = workspace[offset + d + 1];
        float o_s = workspace[offset + tid];

        if (l_s == 0.0f) continue;  // empty split

        float m_new = fmaxf(m_acc, m_s);
        float alpha = __expf(m_acc - m_new);
        float beta  = __expf(m_s - m_new);

        o_acc = o_acc * alpha + o_s * beta;
        l_acc = l_acc * alpha + l_s * beta;
        m_acc = m_new;
    }

    // Finalize: normalize by l
    float inv_l = (l_acc > 0.0f) ? 1.0f / l_acc : 0.0f;
    int o_offset = (static_cast<long long>(batch) * H_q + head) * d;
    O[o_offset + tid] = o_acc * inv_l;
}

// Host wrapper declaration
void run_decode_attn(int B, int H_q, int H_kv, int d,
                     const float* Q,
                     const float* k_cache, const float* v_cache,
                     const int* block_table, const int* context_lens,
                     int max_context_len, int block_size,
                     int num_blocks_per_seq,
                     float* O,
                     float* workspace);

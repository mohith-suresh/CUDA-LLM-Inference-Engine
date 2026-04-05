// kernels/decode/13_decode_attn.cu
#include "decode/13_decode_attn.cuh"
#include <algorithm>

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

template <int GROUP_SIZE>
static void launch_decode_attn(int B, int H_q, int H_kv, int d, float scale,
                               const float* Q,
                               const float* k_cache, const float* v_cache,
                               const int* block_table, const int* context_lens,
                               int max_context_len, int block_size,
                               int num_blocks_per_seq,
                               float* O, float* workspace) {
    int max_kv_blocks = (max_context_len + block_size - 1) / block_size;
    int num_splits = std::clamp(max_kv_blocks / 4, 1, DA_MAX_SPLITS);
    int blocks_per_split = (max_kv_blocks + num_splits - 1) / num_splits;

    // Pass 1: partial attention
    dim3 grid1(B, H_q, num_splits);
    dim3 block1(DA_NTHREADS);
    decode_attn_partial_kernel<GROUP_SIZE><<<grid1, block1>>>(
        d, scale, Q, k_cache, v_cache,
        block_table, context_lens,
        block_size, num_blocks_per_seq,
        H_kv, num_splits, blocks_per_split,
        workspace);

    // Pass 2: reduction
    dim3 grid2(B * H_q);
    dim3 block2(DA_NTHREADS);
    decode_attn_reduce_kernel<<<grid2, block2>>>(
        d, H_q, num_splits, workspace, O);
}

void run_decode_attn(int B, int H_q, int H_kv, int d,
                     const float* Q,
                     const float* k_cache, const float* v_cache,
                     const int* block_table, const int* context_lens,
                     int max_context_len, int block_size,
                     int num_blocks_per_seq,
                     float* O, float* workspace) {
    float scale = 1.0f / sqrtf((float)d);
    int group_size = H_q / H_kv;

    switch (group_size) {
        case 1: launch_decode_attn<1>(B, H_q, H_kv, d, scale, Q, k_cache, v_cache,
                    block_table, context_lens, max_context_len, block_size,
                    num_blocks_per_seq, O, workspace); break;
        case 2: launch_decode_attn<2>(B, H_q, H_kv, d, scale, Q, k_cache, v_cache,
                    block_table, context_lens, max_context_len, block_size,
                    num_blocks_per_seq, O, workspace); break;
        case 4: launch_decode_attn<4>(B, H_q, H_kv, d, scale, Q, k_cache, v_cache,
                    block_table, context_lens, max_context_len, block_size,
                    num_blocks_per_seq, O, workspace); break;
        case 8: launch_decode_attn<8>(B, H_q, H_kv, d, scale, Q, k_cache, v_cache,
                    block_table, context_lens, max_context_len, block_size,
                    num_blocks_per_seq, O, workspace); break;
        default: break;
    }
}

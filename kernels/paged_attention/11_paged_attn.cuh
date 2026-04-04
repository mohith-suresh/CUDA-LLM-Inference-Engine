// kernels/paged_attention/11_paged_attn.cuh
#pragma once
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// ============================================================
// Tile dimensions for PagedAttention
// ============================================================
#define PA_BR 64        // Q tile rows
#define PA_BC 16        // KV tile = one physical block (BLOCK_SIZE)
#define PA_HD 64        // Head dimension
#define PA_NTHREADS 256 // 8 warps

// Thread tiles for S = Q @ K^T  (64 x 16)
// 256 threads -> 16 thread_rows x 16 thread_cols
// Each thread: TM_S=4 rows x TN_S=1 col -> covers 64 x 16
#define PA_TM_S 4
#define PA_TN_S 1

// Thread tiles for O accumulation (64 x 64)
// Each thread: TM_O=4 rows x TN_O=4 cols -> covers 64 x 64
#define PA_TM_O 4
#define PA_TN_O 4

// ============================================================
// Kernel template: <GROUP_SIZE, CAUSAL>
// GROUP_SIZE=1 for MHA (K11), >1 for GQA (K12)
// ============================================================
template <int GROUP_SIZE, bool CAUSAL>
__global__ __launch_bounds__(PA_NTHREADS)
void paged_attn_kernel(int N, int d, float scale,
                       const float* __restrict__ Q,
                       const float* __restrict__ k_cache,
                       const float* __restrict__ v_cache,
                       const int* __restrict__ block_table,
                       const int* __restrict__ context_lens,
                       int block_size, int num_blocks_per_seq,
                       int H_kv,
                       float* __restrict__ O) {
    // Grid: (B, H_q, num_q_tiles)
    const int batch  = blockIdx.x;
    const int q_head = blockIdx.y;
    const int q_tile = blockIdx.z;
    const int H_q    = gridDim.y;
    const int kv_head = q_head / GROUP_SIZE;

    const int ctx_len = context_lens[batch];
    const int num_kv_blocks = (ctx_len + block_size - 1) / block_size;

    const int q_start = q_tile * PA_BR;
    if (q_start >= N) return;

    // Pointers for this batch+head
    const float* Q_bh = Q + (static_cast<long long>(batch) * H_q + q_head) * N * d;
    float*       O_bh = O + (static_cast<long long>(batch) * H_q + q_head) * N * d;
    const int*   bt   = block_table + batch * num_blocks_per_seq;

    const int tid = threadIdx.x;
    const int thread_row = tid / 16;   // 0..15
    const int thread_col = tid % 16;   // 0..15

    const int lane = tid & 31;
    const int half_leader = (lane < 16) ? 0 : 16;

    // Shared memory (padded to avoid bank conflicts)
    __shared__ float Q_smem[PA_BR][PA_HD + 1];    // 64 x 65
    __shared__ float KV_smem[PA_BC][PA_HD + 1];   // 16 x 65
    __shared__ float P_smem[PA_BR][PA_BC + 1];    // 64 x 17

    // Load Q tile [Br x d] into shared memory
    for (int idx = tid; idx < PA_BR * PA_HD; idx += PA_NTHREADS) {
        int r = idx / PA_HD;
        int c = idx % PA_HD;
        int gr = q_start + r;
        Q_smem[r][c] = (gr < N) ? Q_bh[gr * d + c] : 0.0f;
    }

    // Initialize O accumulator and softmax state
    float O_acc[PA_TM_O][PA_TN_O];
    float m_i[PA_TM_O];
    float l_i[PA_TM_O];

    #pragma unroll
    for (int tm = 0; tm < PA_TM_O; ++tm) {
        m_i[tm] = -FLT_MAX;
        l_i[tm] = 0.0f;
        #pragma unroll
        for (int tn = 0; tn < PA_TN_O; ++tn)
            O_acc[tm][tn] = 0.0f;
    }

    __syncthreads();  // Q_smem ready

    // ================= Inner loop: KV blocks =================
    for (int blk_idx = 0; blk_idx < num_kv_blocks; ++blk_idx) {
        int kv_start = blk_idx * block_size;

        // Causal tile skip: entire KV block is past the diagonal
        if (CAUSAL && kv_start > q_start + PA_BR - 1) break;

        int phys_block = bt[blk_idx];

        // --- Load K from paged cache into KV_smem ---
        // k_cache layout: [num_phys_blocks][block_size][H_kv][d]
        for (int idx = tid; idx < PA_BC * PA_HD; idx += PA_NTHREADS) {
            int r = idx / PA_HD;   // token within block (0..15)
            int c = idx % PA_HD;   // dim (0..63)
            int seq_pos = kv_start + r;
            if (seq_pos < ctx_len && r < block_size) {
                int cache_idx = ((phys_block * block_size + r) * H_kv + kv_head) * d + c;
                KV_smem[r][c] = k_cache[cache_idx];
            } else {
                KV_smem[r][c] = 0.0f;
            }
        }
        __syncthreads();  // KV_smem (K) ready

        // --- S = Q @ K^T * scale  [Br x Bc] ---
        float S[PA_TM_S][PA_TN_S];
        #pragma unroll
        for (int tm = 0; tm < PA_TM_S; ++tm)
            #pragma unroll
            for (int tn = 0; tn < PA_TN_S; ++tn)
                S[tm][tn] = 0.0f;

        for (int k = 0; k < PA_HD; ++k) {
            float q_frag[PA_TM_S];
            #pragma unroll
            for (int tm = 0; tm < PA_TM_S; ++tm)
                q_frag[tm] = Q_smem[thread_row * PA_TM_S + tm][k];

            #pragma unroll
            for (int tn = 0; tn < PA_TN_S; ++tn) {
                float k_val = KV_smem[thread_col * PA_TN_S + tn][k];
                #pragma unroll
                for (int tm = 0; tm < PA_TM_S; ++tm)
                    S[tm][tn] += q_frag[tm] * k_val;
            }
        }

        #pragma unroll
        for (int tm = 0; tm < PA_TM_S; ++tm)
            #pragma unroll
            for (int tn = 0; tn < PA_TN_S; ++tn)
                S[tm][tn] *= scale;

        // Boundary + causal mask
        #pragma unroll
        for (int tm = 0; tm < PA_TM_S; ++tm) {
            int gr = q_start + thread_row * PA_TM_S + tm;
            #pragma unroll
            for (int tn = 0; tn < PA_TN_S; ++tn) {
                int gc = kv_start + thread_col * PA_TN_S + tn;
                bool masked = (gc >= ctx_len);
                if (CAUSAL) masked = masked || (gr < gc);
                if (masked) S[tm][tn] = -FLT_MAX;
            }
        }

        // --- Row-wise softmax via half-warp shuffle ---
        // TN_S=1: each thread has 1 col value. 16 lanes cover 16 cols.
        float m_ij[PA_TM_S], l_ij[PA_TM_S];

        #pragma unroll
        for (int tm = 0; tm < PA_TM_S; ++tm) {
            float local_m = S[tm][0];

            // Half-warp max reduction (16 lanes)
            #pragma unroll
            for (int offset = 8; offset >= 1; offset >>= 1)
                local_m = fmaxf(local_m,
                                __shfl_down_sync(0xFFFFFFFF, local_m, offset));
            m_ij[tm] = __shfl_sync(0xFFFFFFFF, local_m, half_leader);

            // exp(S - m_ij)
            #pragma unroll
            for (int tn = 0; tn < PA_TN_S; ++tn)
                S[tm][tn] = __expf(S[tm][tn] - m_ij[tm]);

            float local_l = S[tm][0];

            // Half-warp sum reduction
            #pragma unroll
            for (int offset = 8; offset >= 1; offset >>= 1)
                local_l += __shfl_down_sync(0xFFFFFFFF, local_l, offset);
            l_ij[tm] = __shfl_sync(0xFFFFFFFF, local_l, half_leader);
        }

        // --- Online rescaling of O accumulator ---
        #pragma unroll
        for (int tm = 0; tm < PA_TM_O; ++tm) {
            float m_new = fmaxf(m_i[tm], m_ij[tm]);
            float alpha = __expf(m_i[tm] - m_new);
            float beta  = __expf(m_ij[tm] - m_new);

            #pragma unroll
            for (int tn = 0; tn < PA_TN_O; ++tn)
                O_acc[tm][tn] *= alpha;

            l_i[tm] = l_i[tm] * alpha + l_ij[tm] * beta;
            m_i[tm] = m_new;

            // Scale P by beta
            #pragma unroll
            for (int tn = 0; tn < PA_TN_S; ++tn)
                S[tm][tn] *= beta;
        }

        // --- Write P_scaled to P_smem ---
        #pragma unroll
        for (int tm = 0; tm < PA_TM_S; ++tm)
            #pragma unroll
            for (int tn = 0; tn < PA_TN_S; ++tn)
                P_smem[thread_row * PA_TM_S + tm]
                      [thread_col * PA_TN_S + tn] = S[tm][tn];
        __syncthreads();  // P_smem ready

        // --- Load V from paged cache into KV_smem ---
        for (int idx = tid; idx < PA_BC * PA_HD; idx += PA_NTHREADS) {
            int r = idx / PA_HD;
            int c = idx % PA_HD;
            int seq_pos = kv_start + r;
            if (seq_pos < ctx_len && r < block_size) {
                int cache_idx = ((phys_block * block_size + r) * H_kv + kv_head) * d + c;
                KV_smem[r][c] = v_cache[cache_idx];
            } else {
                KV_smem[r][c] = 0.0f;
            }
        }
        __syncthreads();  // KV_smem (V) ready

        // --- O += P @ V  [Br x Bc] @ [Bc x d] ---
        for (int k = 0; k < PA_BC; ++k) {
            float p_frag[PA_TM_O];
            #pragma unroll
            for (int tm = 0; tm < PA_TM_O; ++tm)
                p_frag[tm] = P_smem[thread_row * PA_TM_O + tm][k];

            #pragma unroll
            for (int tn = 0; tn < PA_TN_O; ++tn) {
                float v_val = KV_smem[k][thread_col * PA_TN_O + tn];
                #pragma unroll
                for (int tm = 0; tm < PA_TM_O; ++tm)
                    O_acc[tm][tn] += p_frag[tm] * v_val;
            }
        }

        __syncthreads();  // Ensure P@V reads complete before next block load

    }  // end KV block loop

    // --- Write O to HBM (finalize: O /= l) ---
    #pragma unroll
    for (int tm = 0; tm < PA_TM_O; ++tm) {
        int gr = q_start + thread_row * PA_TM_O + tm;
        if (gr < N) {
            float inv_l = (l_i[tm] > 0.0f) ? 1.0f / l_i[tm] : 0.0f;
            #pragma unroll
            for (int tn = 0; tn < PA_TN_O; ++tn) {
                int gc = thread_col * PA_TN_O + tn;
                O_bh[gr * d + gc] = O_acc[tm][tn] * inv_l;
            }
        }
    }
}

// Host wrapper declaration (GROUP_SIZE=1, standard MHA)
void run_paged_attn(int B, int H, int N, int d,
                    const float* Q,
                    const float* k_cache, const float* v_cache,
                    const int* block_table, const int* context_lens,
                    int max_context_len, int block_size,
                    int num_blocks_per_seq,
                    float* O, bool causal);

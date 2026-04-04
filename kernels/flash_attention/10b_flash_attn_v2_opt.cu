// kernels/flash_attention/10b_flash_attn_v2_opt.cu
// Optimized FlashAttention-2: grid-parallel, float4, BC=64, register-only P
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include "flash_attention/10b_flash_attn_v2_opt.cuh"
#include "timer.cuh"

// ============================================================
// Tile dimensions
// ============================================================
#define BR 64        // Q tile rows
#define BC 64        // KV tile cols (doubled from K10's 32)
#define HD 64        // Head dimension
#define NTHREADS 256 // 16x16 thread grid
#define PAD 4        // Shared memory padding for float4 alignment

// Thread tile sizes — same 4 rows in both GEMMs (enables register-only P)
#define TM   4  // rows per thread: 16 thread_rows x 4 = 64
#define TN_S 4  // S cols per thread: 16 thread_cols x 4 = 64 = BC
#define TN_O 4  // O cols per thread: 16 thread_cols x 4 = 64 = HD

// ============================================================
// Kernel
// ============================================================
template <bool CAUSAL>
__global__ __launch_bounds__(NTHREADS)
void flash_attn_v2_opt_kernel(int N, int d, float scale,
                              const float* __restrict__ Q,
                              const float* __restrict__ K,
                              const float* __restrict__ V,
                              float* __restrict__ O) {
    // Grid: (num_q_tiles, H, B)
    const int qi        = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int bh = batch_idx * gridDim.y + head_idx;

    const float* Q_bh = Q + bh * N * d;
    const float* K_bh = K + bh * N * d;
    const float* V_bh = V + bh * N * d;
    float*       O_bh = O + bh * N * d;

    const int tid = threadIdx.x;
    const int thread_row = tid / 16;   // 0..15
    const int thread_col = tid % 16;   // 0..15

    // Half-warp info for shuffle reductions
    const int lane    = tid & 31;
    const int half    = lane / 16;             // 0 or 1
    const unsigned half_mask = half == 0 ? 0x0000FFFFu : 0xFFFF0000u;
    const int half_leader = half * 16;         // lane 0 or lane 16

    // Shared memory (padded +4 for float4-aligned rows)
    __shared__ float Q_smem[BR][HD + PAD];    // 64 x 68 = 17,408 B
    __shared__ float KV_smem[BC][HD + PAD];   // 64 x 68 = 17,408 B
                                               // Total:    34,816 B

    const int q_start = qi * BR;
    const int num_kv_tiles = (N + BC - 1) / BC;

    // --- Load Q tile (BR x d) into Q_smem via float4 ---
    for (int idx = tid; idx < BR * (HD / 4); idx += NTHREADS) {
        int r  = idx / (HD / 4);
        int c4 = idx % (HD / 4);
        int gr = q_start + r;
        float4 val;
        if (gr < N) {
            val = reinterpret_cast<const float4*>(Q_bh + gr * d)[c4];
        } else {
            val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        reinterpret_cast<float4*>(&Q_smem[r][c4 * 4])[0] = val;
    }

    // Initialize O accumulator and softmax state
    float O_acc[TM][TN_O];
    float m_i[TM];
    float l_i[TM];

    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        m_i[tm] = -FLT_MAX;
        l_i[tm] = 0.0f;
        #pragma unroll
        for (int tn = 0; tn < TN_O; ++tn)
            O_acc[tm][tn] = 0.0f;
    }

    __syncthreads();  // Q_smem ready

    // ================= Inner loop: KV tiles =================
    for (int kj = 0; kj < num_kv_tiles; ++kj) {
        const int kv_start = kj * BC;

        // Causal early exit
        if (CAUSAL && kv_start > q_start + BR - 1) break;

        // --- Step 1: Load K_j (BC x d) into KV_smem via float4 ---
        for (int idx = tid; idx < BC * (HD / 4); idx += NTHREADS) {
            int r  = idx / (HD / 4);
            int c4 = idx % (HD / 4);
            int gr = kv_start + r;
            float4 val;
            if (gr < N) {
                val = reinterpret_cast<const float4*>(K_bh + gr * d)[c4];
            } else {
                val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            reinterpret_cast<float4*>(&KV_smem[r][c4 * 4])[0] = val;
        }
        __syncthreads();

        // --- Step 2: S = Q @ K^T * scale  (TM x TN_S = 4x4 per thread) ---
        float S[TM][TN_S];
        #pragma unroll
        for (int tm = 0; tm < TM; ++tm)
            #pragma unroll
            for (int tn = 0; tn < TN_S; ++tn)
                S[tm][tn] = 0.0f;

        for (int k = 0; k < HD; ++k) {
            float q_frag[TM];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm)
                q_frag[tm] = Q_smem[thread_row * TM + tm][k];

            #pragma unroll
            for (int tn = 0; tn < TN_S; ++tn) {
                float k_val = KV_smem[thread_col * TN_S + tn][k];
                #pragma unroll
                for (int tm = 0; tm < TM; ++tm)
                    S[tm][tn] += q_frag[tm] * k_val;
            }
        }

        // Scale
        #pragma unroll
        for (int tm = 0; tm < TM; ++tm)
            #pragma unroll
            for (int tn = 0; tn < TN_S; ++tn)
                S[tm][tn] *= scale;

        // Boundary + causal mask
        #pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
            const int gr = q_start + thread_row * TM + tm;
            #pragma unroll
            for (int tn = 0; tn < TN_S; ++tn) {
                const int gc = kv_start + thread_col * TN_S + tn;
                bool masked = (gc >= N);
                if (CAUSAL) masked = masked || (gr < gc);
                if (masked) S[tm][tn] = -FLT_MAX;
            }
        }

        // --- Step 3: Online softmax via half-warp shuffle ---
        float m_ij[TM], l_ij[TM];

        #pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
            // Local max across TN_S=4 columns
            float local_m = S[tm][0];
            #pragma unroll
            for (int tn = 1; tn < TN_S; ++tn)
                local_m = fmaxf(local_m, S[tm][tn]);

            // Half-warp max reduction (16 lanes)
            #pragma unroll
            for (int offset = 8; offset >= 1; offset >>= 1)
                local_m = fmaxf(local_m,
                                __shfl_down_sync(0xFFFFFFFF, local_m, offset));
            m_ij[tm] = __shfl_sync(0xFFFFFFFF, local_m, half_leader);

            // exp(S - max) -> P values in S registers
            #pragma unroll
            for (int tn = 0; tn < TN_S; ++tn)
                S[tm][tn] = __expf(S[tm][tn] - m_ij[tm]);

            // Local sum
            float local_l = S[tm][0];
            #pragma unroll
            for (int tn = 1; tn < TN_S; ++tn)
                local_l += S[tm][tn];

            // Half-warp sum reduction
            #pragma unroll
            for (int offset = 8; offset >= 1; offset >>= 1)
                local_l += __shfl_down_sync(0xFFFFFFFF, local_l, offset);
            l_ij[tm] = __shfl_sync(0xFFFFFFFF, local_l, half_leader);
        }

        // --- Online rescaling of O accumulator ---
        #pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
            float m_new = fmaxf(m_i[tm], m_ij[tm]);
            float alpha = __expf(m_i[tm] - m_new);
            float beta  = __expf(m_ij[tm] - m_new);

            #pragma unroll
            for (int tn = 0; tn < TN_O; ++tn)
                O_acc[tm][tn] *= alpha;

            l_i[tm] = l_i[tm] * alpha + l_ij[tm] * beta;
            m_i[tm] = m_new;

            // Scale P by beta for P@V
            #pragma unroll
            for (int tn = 0; tn < TN_S; ++tn)
                S[tm][tn] *= beta;
        }

        // --- Step 4: Load V_j (BC x d) into KV_smem via float4 ---
        __syncthreads();  // done reading K from KV_smem

        for (int idx = tid; idx < BC * (HD / 4); idx += NTHREADS) {
            int r  = idx / (HD / 4);
            int c4 = idx % (HD / 4);
            int gr = kv_start + r;
            float4 val;
            if (gr < N) {
                val = reinterpret_cast<const float4*>(V_bh + gr * d)[c4];
            } else {
                val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            reinterpret_cast<float4*>(&KV_smem[r][c4 * 4])[0] = val;
        }
        __syncthreads();  // V ready in KV_smem

        // --- Step 5: O += P @ V via warp-shuffle broadcast (no P_smem) ---
        // Each thread holds S[TM][TN_S] = P values for its 4 rows x 4 KV-cols.
        // Full P row has BC=64 values across 16 threads (4 each).
        // Broadcast via __shfl_sync across the half-warp.

        for (int src = 0; src < 16; ++src) {
            // Broadcast 4 P values from source thread
            float p_bcast[TM][TN_S];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm)
                #pragma unroll
                for (int tn_s = 0; tn_s < TN_S; ++tn_s)
                    p_bcast[tm][tn_s] = __shfl_sync(half_mask, S[tm][tn_s],
                                                     src + half_leader);

            // Accumulate: for each of the 4 KV rows this source owns
            #pragma unroll
            for (int tn_s = 0; tn_s < TN_S; ++tn_s) {
                int kv_row = src * TN_S + tn_s;  // 0..63
                float v_frag[TN_O];
                #pragma unroll
                for (int tn = 0; tn < TN_O; ++tn)
                    v_frag[tn] = KV_smem[kv_row][thread_col * TN_O + tn];

                #pragma unroll
                for (int tm = 0; tm < TM; ++tm) {
                    float p_val = p_bcast[tm][tn_s];
                    #pragma unroll
                    for (int tn = 0; tn < TN_O; ++tn)
                        O_acc[tm][tn] += p_val * v_frag[tn];
                }
            }
        }

        __syncthreads();  // ensure P@V reads done before next K load

    }  // end inner loop

    // --- Write O to global memory via float4 ---
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        const int gr = q_start + thread_row * TM + tm;
        if (gr < N) {
            float inv_l = (l_i[tm] > 0.0f) ? 1.0f / l_i[tm] : 0.0f;
            float4 out_val;
            out_val.x = O_acc[tm][0] * inv_l;
            out_val.y = O_acc[tm][1] * inv_l;
            out_val.z = O_acc[tm][2] * inv_l;
            out_val.w = O_acc[tm][3] * inv_l;
            reinterpret_cast<float4*>(O_bh + gr * d)[thread_col] = out_val;
        }
    }
}

// ============================================================
// Host wrapper
// ============================================================
void run_flash_attn_v2_opt(int B, int H, int N, int d,
                           const float* Q, const float* K, const float* V,
                           float* O, bool causal) {
    float scale = 1.0f / sqrtf((float)d);
    int num_q_tiles = (N + BR - 1) / BR;
    dim3 grid(num_q_tiles, H, B);
    dim3 block(NTHREADS);

    if (causal)
        flash_attn_v2_opt_kernel<true><<<grid, block>>>(N, d, scale, Q, K, V, O);
    else
        flash_attn_v2_opt_kernel<false><<<grid, block>>>(N, d, scale, Q, K, V, O);
}

#undef BR
#undef BC
#undef HD
#undef NTHREADS
#undef PAD
#undef TM
#undef TN_S
#undef TN_O

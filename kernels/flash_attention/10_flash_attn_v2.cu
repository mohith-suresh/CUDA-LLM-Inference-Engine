// kernels/flash_attention/10_flash_attn_v2.cu
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include "flash_attention/10_flash_attn_v2.cuh"
#include "timer.cuh"

// ============================================================
// Tile dimensions
// ============================================================
#define BR 64        // Q tile rows (outer loop)
#define BC 32        // KV tile rows (inner loop)
#define HD 64        // Head dimension (not tiled)
#define NTHREADS 256 // Threads per block (8 warps)

// Thread tile sizes for S = Q @ K^T  (BR x BC = 64 x 32)
#define TM_S 4  // rows per thread: 16 thread_rows x 4 = 64
#define TN_S 2  // cols per thread: 16 thread_cols x 2 = 32

// Thread tile sizes for O accumulation (BR x HD = 64 x 64)
#define TM_O 4  // rows per thread: 16 thread_rows x 4 = 64
#define TN_O 4  // cols per thread: 16 thread_cols x 4 = 64

// ============================================================
// Kernel
// ============================================================
template <bool CAUSAL>
__global__ __launch_bounds__(NTHREADS)
void flash_attn_v2_kernel(int N, int d, float scale,
                          const float* __restrict__ Q,
                          const float* __restrict__ K,
                          const float* __restrict__ V,
                          float* __restrict__ O) {
    // Each block handles one (batch, head) pair
    const int bh = blockIdx.x;
    const float* Q_bh = Q + bh * N * d;
    const float* K_bh = K + bh * N * d;
    const float* V_bh = V + bh * N * d;
    float*       O_bh = O + bh * N * d;

    const int tid = threadIdx.x;
    const int thread_row = tid / 16;   // 0..15
    const int thread_col = tid % 16;   // 0..15

    // Warp layout: warp k has tid [32k, 32k+31]
    //   lanes 0-15  -> thread_row 2k,   thread_cols 0-15
    //   lanes 16-31 -> thread_row 2k+1, thread_cols 0-15
    const int lane = tid & 31;
    const int half_leader = (lane < 16) ? 0 : 16;

    // Shared memory (padded +1 per row to avoid bank conflicts)
    __shared__ float Q_smem[BR][HD + 1];    // 64 x 65 = 16,640 B
    __shared__ float KV_smem[BC][HD + 1];   // 32 x 65 =  8,320 B
    __shared__ float P_smem[BR][BC + 1];    // 64 x 33 =  8,448 B
                                             // Total:    33,408 B

    const int num_q_tiles  = (N + BR - 1) / BR;
    const int num_kv_tiles = (N + BC - 1) / BC;

    // ===================== Outer loop: Q tiles =====================
    for (int qi = 0; qi < num_q_tiles; ++qi) {
        const int q_start = qi * BR;

        // --- Load Q_i (BR x d) into Q_smem ---
        for (int idx = tid; idx < BR * HD; idx += NTHREADS) {
            const int r = idx / HD;
            const int c = idx % HD;
            const int gr = q_start + r;
            Q_smem[r][c] = (gr < N) ? Q_bh[gr * d + c] : 0.0f;
        }

        // Initialize O accumulator and softmax state in registers
        float O_acc[TM_O][TN_O];
        float m_i[TM_O];   // running row max
        float l_i[TM_O];   // running row sum_exp

        #pragma unroll
        for (int tm = 0; tm < TM_O; ++tm) {
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

            // Causal tile skip: entire tile above diagonal
            if (CAUSAL && kv_start > (qi + 1) * BR - 1) break;

            // --- Step 1: Load K_j (BC x d) into KV_smem ---
            for (int idx = tid; idx < BC * HD; idx += NTHREADS) {
                const int r = idx / HD;
                const int c = idx % HD;
                const int gr = kv_start + r;
                KV_smem[r][c] = (gr < N) ? K_bh[gr * d + c] : 0.0f;
            }
            __syncthreads();  // KV_smem (K) ready

            // --- Step 2: S = Q @ K^T * scale (BR x BC in registers) ---
            float S[TM_S][TN_S];
            #pragma unroll
            for (int tm = 0; tm < TM_S; ++tm)
                #pragma unroll
                for (int tn = 0; tn < TN_S; ++tn)
                    S[tm][tn] = 0.0f;

            for (int k = 0; k < HD; ++k) {
                float q_frag[TM_S];
                #pragma unroll
                for (int tm = 0; tm < TM_S; ++tm)
                    q_frag[tm] = Q_smem[thread_row * TM_S + tm][k];

                #pragma unroll
                for (int tn = 0; tn < TN_S; ++tn) {
                    float k_val = KV_smem[thread_col * TN_S + tn][k];
                    #pragma unroll
                    for (int tm = 0; tm < TM_S; ++tm)
                        S[tm][tn] += q_frag[tm] * k_val;
                }
            }

            // Scale by 1/sqrt(d)
            #pragma unroll
            for (int tm = 0; tm < TM_S; ++tm)
                #pragma unroll
                for (int tn = 0; tn < TN_S; ++tn)
                    S[tm][tn] *= scale;

            // Boundary + causal mask: set S = -inf for invalid positions
            #pragma unroll
            for (int tm = 0; tm < TM_S; ++tm) {
                const int gr = q_start + thread_row * TM_S + tm;
                #pragma unroll
                for (int tn = 0; tn < TN_S; ++tn) {
                    const int gc = kv_start + thread_col * TN_S + tn;
                    bool masked = (gc >= N);
                    if (CAUSAL) masked = masked || (gr < gc);
                    if (masked) S[tm][tn] = -FLT_MAX;
                }
            }

            // --- Step 3: Row-wise softmax via half-warp shuffle ---
            float m_ij[TM_S], l_ij[TM_S];

            #pragma unroll
            for (int tm = 0; tm < TM_S; ++tm) {
                // Local max across TN_S=2 columns
                float local_m = fmaxf(S[tm][0], S[tm][1]);

                // Warp shuffle max reduction (16 lanes in half-warp)
                #pragma unroll
                for (int offset = 8; offset >= 1; offset >>= 1)
                    local_m = fmaxf(local_m,
                                    __shfl_down_sync(0xFFFFFFFF, local_m, offset));
                m_ij[tm] = __shfl_sync(0xFFFFFFFF, local_m, half_leader);

                // exp(S - m_ij) -> P values (stored back in S)
                #pragma unroll
                for (int tn = 0; tn < TN_S; ++tn)
                    S[tm][tn] = __expf(S[tm][tn] - m_ij[tm]);

                // Local sum of P values
                float local_l = S[tm][0] + S[tm][1];

                // Warp shuffle sum reduction
                #pragma unroll
                for (int offset = 8; offset >= 1; offset >>= 1)
                    local_l += __shfl_down_sync(0xFFFFFFFF, local_l, offset);
                l_ij[tm] = __shfl_sync(0xFFFFFFFF, local_l, half_leader);
            }

            // --- Online rescaling of O accumulator ---
            #pragma unroll
            for (int tm = 0; tm < TM_O; ++tm) {
                float m_new = fmaxf(m_i[tm], m_ij[tm]);
                float alpha = __expf(m_i[tm] - m_new);   // rescale old O
                float beta  = __expf(m_ij[tm] - m_new);  // scale new P

                #pragma unroll
                for (int tn = 0; tn < TN_O; ++tn)
                    O_acc[tm][tn] *= alpha;

                l_i[tm] = l_i[tm] * alpha + l_ij[tm] * beta;
                m_i[tm] = m_new;

                // Scale P by beta for the P@V accumulation
                #pragma unroll
                for (int tn = 0; tn < TN_S; ++tn)
                    S[tm][tn] *= beta;
            }

            // --- Step 4: Write P_scaled to P_smem ---
            #pragma unroll
            for (int tm = 0; tm < TM_S; ++tm)
                #pragma unroll
                for (int tn = 0; tn < TN_S; ++tn)
                    P_smem[thread_row * TM_S + tm]
                          [thread_col * TN_S + tn] = S[tm][tn];
            __syncthreads();  // P_smem ready

            // --- Step 5: Load V_j (BC x d) into KV_smem (overwrites K) ---
            for (int idx = tid; idx < BC * HD; idx += NTHREADS) {
                const int r = idx / HD;
                const int c = idx % HD;
                const int gr = kv_start + r;
                KV_smem[r][c] = (gr < N) ? V_bh[gr * d + c] : 0.0f;
            }
            __syncthreads();  // KV_smem (V) ready

            // --- Step 6: O += P @ V  (BR x BC @ BC x d -> BR x d) ---
            for (int k = 0; k < BC; ++k) {
                float p_frag[TM_O];
                #pragma unroll
                for (int tm = 0; tm < TM_O; ++tm)
                    p_frag[tm] = P_smem[thread_row * TM_O + tm][k];

                #pragma unroll
                for (int tn = 0; tn < TN_O; ++tn) {
                    float v_val = KV_smem[k][thread_col * TN_O + tn];
                    #pragma unroll
                    for (int tm = 0; tm < TM_O; ++tm)
                        O_acc[tm][tn] += p_frag[tm] * v_val;
                }
            }

            __syncthreads();  // Ensure P@V reads complete before next K load

        }  // end inner loop

        // --- Write O_i to HBM (finalize: O /= l) ---
        #pragma unroll
        for (int tm = 0; tm < TM_O; ++tm) {
            const int gr = q_start + thread_row * TM_O + tm;
            if (gr < N) {
                float inv_l = (l_i[tm] > 0.0f) ? 1.0f / l_i[tm] : 0.0f;
                #pragma unroll
                for (int tn = 0; tn < TN_O; ++tn) {
                    const int gc = thread_col * TN_O + tn;
                    O_bh[gr * d + gc] = O_acc[tm][tn] * inv_l;
                }
            }
        }

        __syncthreads();  // All threads done before next Q tile load

    }  // end outer loop
}

// ============================================================
// Host wrapper
// ============================================================
void run_flash_attn_v2(int B, int H, int N, int d,
                       const float* Q, const float* K, const float* V,
                       float* O, bool causal) {
    float scale = 1.0f / sqrtf((float)d);
    dim3 grid(B * H);
    dim3 block(NTHREADS);

    if (causal)
        flash_attn_v2_kernel<true><<<grid, block>>>(N, d, scale, Q, K, V, O);
    else
        flash_attn_v2_kernel<false><<<grid, block>>>(N, d, scale, Q, K, V, O);
}

#undef BR
#undef BC
#undef HD
#undef NTHREADS
#undef TM_S
#undef TN_S
#undef TM_O
#undef TN_O

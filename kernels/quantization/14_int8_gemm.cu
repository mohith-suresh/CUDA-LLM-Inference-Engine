// kernels/quantization/14_int8_gemm.cu
#include "quantization/14_int8_gemm.cuh"

// ============================================================
// Quantization kernel: FP32 -> INT8 (per-row symmetric)
// One block per row. Threads cooperatively:
//   Phase 1: find max(|row|) via shared memory reduction
//   Phase 2: quantize + pack 4 int8s into int32
// ============================================================
__global__ void quantize_fp32_to_int8_kernel(
    int cols,
    const float* __restrict__ input,     // [rows, cols]
    int32_t* __restrict__ output_packed, // [rows, cols/4]
    float* __restrict__ scales)          // [rows]
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row_ptr = input + row * cols;

    extern __shared__ float smem[];

    float local_max = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = fabsf(row_ptr[i]);
        if (val > local_max) local_max = val;
    }
    smem[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    float row_max = smem[0];
    float scale = (row_max > 0.0f) ? row_max / 127.0f : 1.0f;
    float inv_scale = 1.0f / scale;

    if (tid == 0) scales[row] = scale;
    __syncthreads();

    int packed_cols = cols / 4;
    int32_t* out_row = output_packed + row * packed_cols;

    for (int i = tid; i < packed_cols; i += blockDim.x) {
        int base = i * 4;
        int8_t q0 = static_cast<int8_t>(fminf(fmaxf(rintf(row_ptr[base + 0] * inv_scale), -128.0f), 127.0f));
        int8_t q1 = static_cast<int8_t>(fminf(fmaxf(rintf(row_ptr[base + 1] * inv_scale), -128.0f), 127.0f));
        int8_t q2 = static_cast<int8_t>(fminf(fmaxf(rintf(row_ptr[base + 2] * inv_scale), -128.0f), 127.0f));
        int8_t q3 = static_cast<int8_t>(fminf(fmaxf(rintf(row_ptr[base + 3] * inv_scale), -128.0f), 127.0f));

        int32_t packed = 0;
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(q0)));
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(q1)) << 8);
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(q2)) << 16);
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(q3)) << 24);
        out_row[i] = packed;
    }
}

// ============================================================
// INT8 GEMM kernel via __dp4a() — templated epilogue
// ============================================================
template <I8Epilogue EPILOGUE>
__global__ __launch_bounds__(I8_NTHREADS)
void int8_gemm_dp4a_kernel(
    int M, int N, int K,
    const int32_t* __restrict__ A_packed,
    const int32_t* __restrict__ BT_packed,
    const float* __restrict__ scale_A,
    const float* __restrict__ scale_B,
    float* __restrict__ C,
    const float* __restrict__ bias,
    const float* __restrict__ residual)
{
    const int bm = blockIdx.y;
    const int bn = blockIdx.x;
    const int tid = threadIdx.x;
    const int thread_row = tid / (I8_BN / I8_TN);
    const int thread_col = tid % (I8_BN / I8_TN);

    const int K4 = K / 4;
    const int BK4 = I8_BK / 4;

    __shared__ int32_t A_smem[I8_BM][BK4 + 1];
    __shared__ int32_t BT_smem[I8_BN][BK4 + 1];

    int32_t acc[I8_TM][I8_TN];
    #pragma unroll
    for (int tm = 0; tm < I8_TM; ++tm)
        #pragma unroll
        for (int tn = 0; tn < I8_TN; ++tn)
            acc[tm][tn] = 0;

    for (int k_tile = 0; k_tile < K4; k_tile += BK4) {
        {
            int load_idx = tid;
            int r = load_idx / BK4;
            int c = load_idx % BK4;
            int global_row = bm * I8_BM + r;
            int global_col = k_tile + c;
            if (global_row < M && global_col < K4)
                A_smem[r][c] = A_packed[global_row * K4 + global_col];
            else
                A_smem[r][c] = 0;
        }
        {
            int load_idx = tid;
            int r = load_idx / BK4;
            int c = load_idx % BK4;
            int global_row = bn * I8_BN + r;
            int global_col = k_tile + c;
            if (global_row < N && global_col < K4)
                BT_smem[r][c] = BT_packed[global_row * K4 + global_col];
            else
                BT_smem[r][c] = 0;
        }
        __syncthreads();

        #pragma unroll
        for (int k4 = 0; k4 < BK4; ++k4) {
            int32_t a_frag[I8_TM];
            #pragma unroll
            for (int tm = 0; tm < I8_TM; ++tm)
                a_frag[tm] = A_smem[thread_row * I8_TM + tm][k4];

            int32_t b_frag[I8_TN];
            #pragma unroll
            for (int tn = 0; tn < I8_TN; ++tn)
                b_frag[tn] = BT_smem[thread_col * I8_TN + tn][k4];

            #pragma unroll
            for (int tm = 0; tm < I8_TM; ++tm)
                #pragma unroll
                for (int tn = 0; tn < I8_TN; ++tn)
                    acc[tm][tn] = __dp4a(a_frag[tm], b_frag[tn], acc[tm][tn]);
        }
        __syncthreads();
    }

    // Epilogue: dequantize + fused ops
    #pragma unroll
    for (int tm = 0; tm < I8_TM; ++tm) {
        int gr = bm * I8_BM + thread_row * I8_TM + tm;
        if (gr >= M) continue;
        float sa = scale_A[gr];

        #pragma unroll
        for (int tn = 0; tn < I8_TN; ++tn) {
            int gc = bn * I8_BN + thread_col * I8_TN + tn;
            if (gc >= N) continue;

            float val = static_cast<float>(acc[tm][tn]) * sa * scale_B[gc];

            if constexpr (EPILOGUE == I8Epilogue::BiasOnly) {
                val += bias[gc];
            } else if constexpr (EPILOGUE == I8Epilogue::BiasGELU) {
                val += bias[gc];
                float x3 = val * val * val;
                val = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * x3)));
            } else if constexpr (EPILOGUE == I8Epilogue::BiasResidual) {
                val += bias[gc];
                val += residual[gr * N + gc];
            }

            C[gr * N + gc] = val;
        }
    }
}

void run_quantize_fp32_to_int8(int rows, int cols,
                                const float* input,
                                int32_t* output_packed,
                                float* scales) {
    int threads = 256;
    int smem = threads * sizeof(float);
    quantize_fp32_to_int8_kernel<<<rows, threads, smem>>>(
        cols, input, output_packed, scales);
}

void run_int8_gemm(int M, int N, int K,
                   const int32_t* A_packed,
                   const int32_t* BT_packed,
                   const float* scale_A,
                   const float* scale_B,
                   float* C) {
    dim3 grid((N + I8_BN - 1) / I8_BN, (M + I8_BM - 1) / I8_BM);
    dim3 block(I8_NTHREADS);
    int8_gemm_dp4a_kernel<I8Epilogue::Plain><<<grid, block>>>(
        M, N, K, A_packed, BT_packed, scale_A, scale_B, C, nullptr, nullptr);
}

void run_int8_gemm_bias(int M, int N, int K,
                        const int32_t* A_packed,
                        const int32_t* BT_packed,
                        const float* scale_A,
                        const float* scale_B,
                        const float* bias,
                        float* C) {
    dim3 grid((N + I8_BN - 1) / I8_BN, (M + I8_BM - 1) / I8_BM);
    dim3 block(I8_NTHREADS);
    int8_gemm_dp4a_kernel<I8Epilogue::BiasOnly><<<grid, block>>>(
        M, N, K, A_packed, BT_packed, scale_A, scale_B, C, bias, nullptr);
}

void run_int8_gemm_bias_gelu(int M, int N, int K,
                              const int32_t* A_packed,
                              const int32_t* BT_packed,
                              const float* scale_A,
                              const float* scale_B,
                              const float* bias,
                              float* C) {
    dim3 grid((N + I8_BN - 1) / I8_BN, (M + I8_BM - 1) / I8_BM);
    dim3 block(I8_NTHREADS);
    int8_gemm_dp4a_kernel<I8Epilogue::BiasGELU><<<grid, block>>>(
        M, N, K, A_packed, BT_packed, scale_A, scale_B, C, bias, nullptr);
}

void run_int8_gemm_bias_residual(int M, int N, int K,
                                  const int32_t* A_packed,
                                  const int32_t* BT_packed,
                                  const float* scale_A,
                                  const float* scale_B,
                                  const float* bias,
                                  const float* residual,
                                  float* C) {
    dim3 grid((N + I8_BN - 1) / I8_BN, (M + I8_BM - 1) / I8_BM);
    dim3 block(I8_NTHREADS);
    int8_gemm_dp4a_kernel<I8Epilogue::BiasResidual><<<grid, block>>>(
        M, N, K, A_packed, BT_packed, scale_A, scale_B, C, bias, residual);
}

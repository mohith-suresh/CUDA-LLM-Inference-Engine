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
// INT8 GEMM kernel via __dp4a()
// NT layout: A [M, K/4] and B^T [N, K/4], both int8x4 packed as int32
// Tiled: BM=64, BN=64, BK=16 (int8 elements) = 4 int32 packed values
// Register tiling: TM=4, TN=4 — each thread owns 4x4 int32 accumulators
//
// __dp4a(a, b, c): treats a,b as packed int8x4
//   c += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
//
// Epilogue: dequantize with per-row scales
//   C_fp32[i][j] = C_int32[i][j] * scale_A[i] * scale_B[j]
// ============================================================
__global__ __launch_bounds__(I8_NTHREADS)
void int8_gemm_dp4a_kernel(
    int M, int N, int K,
    const int32_t* __restrict__ A_packed,   // [M, K/4]
    const int32_t* __restrict__ BT_packed,  // [N, K/4]
    const float* __restrict__ scale_A,      // [M]
    const float* __restrict__ scale_B,      // [N]
    float* __restrict__ C)                  // [M, N]
{
    const int bm = blockIdx.y;   // tile row
    const int bn = blockIdx.x;   // tile col
    const int tid = threadIdx.x;
    const int thread_row = tid / (I8_BN / I8_TN);  // 0..15
    const int thread_col = tid % (I8_BN / I8_TN);  // 0..15

    const int K4 = K / 4;     // packed dimension
    const int BK4 = I8_BK / 4; // 4 int32s per K-tile

    // Shared memory for A tile [BM][BK/4] and B^T tile [BN][BK/4]
    // +1 padding to avoid bank conflicts
    __shared__ int32_t A_smem[I8_BM][BK4 + 1];
    __shared__ int32_t BT_smem[I8_BN][BK4 + 1];

    // Register accumulators: TM x TN = 4x4 int32
    int32_t acc[I8_TM][I8_TN];
    #pragma unroll
    for (int tm = 0; tm < I8_TM; ++tm)
        #pragma unroll
        for (int tn = 0; tn < I8_TN; ++tn)
            acc[tm][tn] = 0;

    // Tile loop over K dimension
    for (int k_tile = 0; k_tile < K4; k_tile += BK4) {
        // Cooperative load: A tile [BM][BK4]
        // Total elements = BM * BK4 = 64 * 4 = 256 = NTHREADS (one element each)
        {
            int load_idx = tid;  // tid in [0, 255]
            int r = load_idx / BK4;       // 0..63
            int c = load_idx % BK4;       // 0..3
            int global_row = bm * I8_BM + r;
            int global_col = k_tile + c;
            if (global_row < M && global_col < K4)
                A_smem[r][c] = A_packed[global_row * K4 + global_col];
            else
                A_smem[r][c] = 0;
        }

        // Cooperative load: B^T tile [BN][BK4]
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

        // Compute: dp4a over BK4 packed int32 values
        #pragma unroll
        for (int k4 = 0; k4 < BK4; ++k4) {
            // Load A fragments: TM rows
            int32_t a_frag[I8_TM];
            #pragma unroll
            for (int tm = 0; tm < I8_TM; ++tm)
                a_frag[tm] = A_smem[thread_row * I8_TM + tm][k4];

            // Load B^T fragments: TN rows
            int32_t b_frag[I8_TN];
            #pragma unroll
            for (int tn = 0; tn < I8_TN; ++tn)
                b_frag[tn] = BT_smem[thread_col * I8_TN + tn][k4];

            // dp4a: 4 int8 MADs per call
            #pragma unroll
            for (int tm = 0; tm < I8_TM; ++tm)
                #pragma unroll
                for (int tn = 0; tn < I8_TN; ++tn)
                    acc[tm][tn] = __dp4a(a_frag[tm], b_frag[tn], acc[tm][tn]);
        }

        __syncthreads();
    }

    // Epilogue: dequantize and write C
    #pragma unroll
    for (int tm = 0; tm < I8_TM; ++tm) {
        int gr = bm * I8_BM + thread_row * I8_TM + tm;
        if (gr >= M) continue;
        float sa = scale_A[gr];

        #pragma unroll
        for (int tn = 0; tn < I8_TN; ++tn) {
            int gc = bn * I8_BN + thread_col * I8_TN + tn;
            if (gc >= N) continue;
            float sb = scale_B[gc];
            C[gr * N + gc] = static_cast<float>(acc[tm][tn]) * sa * sb;
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
    int8_gemm_dp4a_kernel<<<grid, block>>>(
        M, N, K, A_packed, BT_packed, scale_A, scale_B, C);
}

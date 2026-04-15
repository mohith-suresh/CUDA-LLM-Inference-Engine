// src/gpt2_engine.cu
#include "gpt2_engine.cuh"
#include "timer.cuh"
#include "quantization/14_int8_gemm.cuh"
#include "layernorm/15_layernorm_residual.cuh"
#include "layernorm/15_embed.cuh"
#include "decode/13_decode_attn.cuh"
#include "flash_attention/10_flash_attn_v2.cuh"
#include "sampler.cuh"

#include <fstream>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>
#include <cmath>

static float* load_bin_f32(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Failed to open %s\n", path.c_str()); exit(1); }
    std::vector<float> buf(count);
    f.read(reinterpret_cast<char*>(buf.data()), count * sizeof(float));
    float* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ptr, buf.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    return d_ptr;
}

static int32_t* load_bin_i32(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Failed to open %s\n", path.c_str()); exit(1); }
    std::vector<int32_t> buf(count);
    f.read(reinterpret_cast<char*>(buf.data()), count * sizeof(int32_t));
    int32_t* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_ptr, buf.data(), count * sizeof(int32_t), cudaMemcpyHostToDevice));
    return d_ptr;
}

static GPT2Config parse_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "Failed to open %s\n", path.c_str()); exit(1); }
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    GPT2Config c;
    auto get_int = [&](const char* key) -> int {
        auto pos = content.find(std::string("\"") + key + "\"");
        pos = content.find(":", pos);
        return atoi(content.c_str() + pos + 1);
    };
    c.n_layers = get_int("n_layers");
    c.n_heads = get_int("n_heads");
    c.d_model = get_int("d_model");
    c.d_ff = get_int("d_ff");
    c.vocab_size = get_int("vocab_size");
    c.max_seq_len = get_int("max_seq_len");
    c.d_head = get_int("d_head");
    return c;
}

static QuantWeight load_quant(const std::string& dir, const char* name, int N, int K) {
    QuantWeight w;
    w.N = N;
    w.K = K;
    w.packed = load_bin_i32(dir + "/" + name + "_weight.bin", (size_t)N * (K / 4));
    w.scale = load_bin_f32(dir + "/" + name + "_scale.bin", N);
    return w;
}

void GPT2Engine::load_config() {
    config_ = parse_config(model_dir_ + "/config.json");
}

void GPT2Engine::load_weights() {
    int d = config_.d_model;
    int ff = config_.d_ff;
    int V = config_.vocab_size;
    int L = config_.max_seq_len;

    weights_.wte = load_bin_f32(model_dir_ + "/wte.bin", (size_t)V * d);
    weights_.wpe = load_bin_f32(model_dir_ + "/wpe.bin", (size_t)L * d);
    weights_.ln_f_gamma = load_bin_f32(model_dir_ + "/ln_f_gamma.bin", d);
    weights_.ln_f_beta = load_bin_f32(model_dir_ + "/ln_f_beta.bin", d);

    for (int i = 0; i < config_.n_layers; ++i) {
        char layer_dir[256];
        snprintf(layer_dir, sizeof(layer_dir), "%s/layer_%02d", model_dir_.c_str(), i);
        auto& lw = weights_.layers[i];

        lw.ln1_gamma = load_bin_f32(std::string(layer_dir) + "/ln1_gamma.bin", d);
        lw.ln1_beta = load_bin_f32(std::string(layer_dir) + "/ln1_beta.bin", d);

        lw.qkv = load_quant(layer_dir, "qkv", 3 * d, d);
        lw.qkv_bias = load_bin_f32(std::string(layer_dir) + "/qkv_bias.bin", 3 * d);

        lw.out = load_quant(layer_dir, "out", d, d);
        lw.out_bias = load_bin_f32(std::string(layer_dir) + "/out_bias.bin", d);

        lw.ln2_gamma = load_bin_f32(std::string(layer_dir) + "/ln2_gamma.bin", d);
        lw.ln2_beta = load_bin_f32(std::string(layer_dir) + "/ln2_beta.bin", d);

        lw.ffn_up = load_quant(layer_dir, "ffn_up", ff, d);
        lw.ffn_up_bias = load_bin_f32(std::string(layer_dir) + "/ffn_up_bias.bin", ff);

        lw.ffn_down = load_quant(layer_dir, "ffn_down", d, ff);
        lw.ffn_down_bias = load_bin_f32(std::string(layer_dir) + "/ffn_down_bias.bin", d);
    }

    printf("Loaded GPT-2 weights: %d layers, d=%d, ff=%d, vocab=%d\n",
           config_.n_layers, d, ff, V);
}

void GPT2Engine::alloc_workspace() {
    int d = config_.d_model;
    int ff = config_.d_ff;
    int V = config_.vocab_size;
    int L = config_.max_seq_len;
    int H = config_.n_heads;
    int dh = config_.d_head;
    int n_layers = config_.n_layers;

    CUDA_CHECK(cudaMalloc(&d_x_, (size_t)L * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual_, (size_t)L * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qkv_, (size_t)L * 3 * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_out_, (size_t)L * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn_hidden_, (size_t)L * ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits_, (size_t)V * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_q_buf_, (size_t)L * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_buf_, (size_t)L * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_buf_, (size_t)L * d * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_q_split_, (size_t)L * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_split_, (size_t)L * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_split_, (size_t)L * d * sizeof(float)));

    int max_rows = L;
    int max_cols = ff;
    CUDA_CHECK(cudaMalloc(&d_act_packed_, (size_t)max_rows * (max_cols / 4) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_act_scale_, (size_t)max_rows * sizeof(float)));

    block_size_ = 16;
    int max_blocks = (L + block_size_ - 1) / block_size_;
    num_phys_blocks_ = max_blocks + 4;
    blocks_allocated_ = 0;
    current_pos_ = 0;

    size_t cache_size = (size_t)n_layers * num_phys_blocks_ * block_size_ * H * dh * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_k_cache_, cache_size));
    CUDA_CHECK(cudaMalloc(&d_v_cache_, cache_size));

    CUDA_CHECK(cudaMalloc(&d_block_tables_, max_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_context_lens_, sizeof(int)));

    size_t ws_size = (size_t)H * DA_MAX_SPLITS * (dh + 2) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_decode_workspace_, ws_size));
}

static FP32Weight load_fp32_weight(const std::string& dir, const char* name,
                                    int K, int N, float* bias_shared) {
    FP32Weight w;
    w.K = K;
    w.N = N;
    w.data = load_bin_f32(dir + "/" + name + "_weight_fp32.bin", (size_t)K * N);
    w.bias = bias_shared;
    return w;
}

void GPT2Engine::load_weights_fp32() {
    int d = config_.d_model;
    int ff = config_.d_ff;
    for (int i = 0; i < config_.n_layers; ++i) {
        char ld[256];
        snprintf(ld, sizeof(ld), "%s/layer_%02d", model_dir_.c_str(), i);
        std::string ldir(ld);
        auto& fw = weights_.layers_fp32[i];
        auto& lw = weights_.layers[i];
        fw.qkv      = load_fp32_weight(ldir, "qkv",      d,  3 * d, lw.qkv_bias);
        fw.out      = load_fp32_weight(ldir, "out",      d,  d,     lw.out_bias);
        fw.ffn_up   = load_fp32_weight(ldir, "ffn_up",   d,  ff,    lw.ffn_up_bias);
        fw.ffn_down = load_fp32_weight(ldir, "ffn_down", ff, d,     lw.ffn_down_bias);
    }
    printf("Loaded FP32 weights for cuBLAS backend\n");
}

GPT2Engine::GPT2Engine(const std::string& model_dir, InferenceBackend backend)
    : model_dir_(model_dir), backend_(backend), cublas_ready_(false) {
    load_config();
    load_weights();
    if (backend_ == InferenceBackend::CUBLAS_FP32) {
        load_weights_fp32();
        cublasCreate(&cublas_handle_);
        cublas_ready_ = true;
    }
    alloc_workspace();
}

void GPT2Engine::free_all() {
    cudaFree(d_x_); cudaFree(d_residual_);
    cudaFree(d_qkv_); cudaFree(d_attn_out_);
    cudaFree(d_ffn_hidden_); cudaFree(d_logits_);
    cudaFree(d_q_buf_); cudaFree(d_k_buf_); cudaFree(d_v_buf_);
    cudaFree(d_q_split_); cudaFree(d_k_split_); cudaFree(d_v_split_);
    cudaFree(d_act_packed_); cudaFree(d_act_scale_);
    cudaFree(d_k_cache_); cudaFree(d_v_cache_);
    cudaFree(d_block_tables_); cudaFree(d_context_lens_);
    cudaFree(d_decode_workspace_);

    cudaFree(weights_.wte); cudaFree(weights_.wpe);
    cudaFree(weights_.ln_f_gamma); cudaFree(weights_.ln_f_beta);
    for (int i = 0; i < config_.n_layers; ++i) {
        auto& lw = weights_.layers[i];
        cudaFree(lw.ln1_gamma); cudaFree(lw.ln1_beta);
        cudaFree(lw.qkv.packed); cudaFree(lw.qkv.scale); cudaFree(lw.qkv_bias);
        cudaFree(lw.out.packed); cudaFree(lw.out.scale); cudaFree(lw.out_bias);
        cudaFree(lw.ln2_gamma); cudaFree(lw.ln2_beta);
        cudaFree(lw.ffn_up.packed); cudaFree(lw.ffn_up.scale); cudaFree(lw.ffn_up_bias);
        cudaFree(lw.ffn_down.packed); cudaFree(lw.ffn_down.scale); cudaFree(lw.ffn_down_bias);
    }

    if (backend_ == InferenceBackend::CUBLAS_FP32) {
        for (int i = 0; i < config_.n_layers; ++i) {
            auto& fw = weights_.layers_fp32[i];
            cudaFree(fw.qkv.data);
            cudaFree(fw.out.data);
            cudaFree(fw.ffn_up.data);
            cudaFree(fw.ffn_down.data);
        }
        if (cublas_ready_) cublasDestroy(cublas_handle_);
    }
}

GPT2Engine::~GPT2Engine() {
    free_all();
}

// ============================================================
// Transpose helpers for flash-attn layout
//   token-major [M, H, d] <-> head-major [H, M, d]
// ============================================================
__global__ void transpose_mhd_to_hmd_kernel(
    const float* __restrict__ in, float* __restrict__ out,
    int M, int H, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * H * d;
    if (idx >= total) return;
    int dim   = idx % d;
    int rem   = idx / d;
    int h     = rem % H;
    int m     = rem / H;
    // in:  [m, h, dim]
    // out: [h, m, dim]
    out[(h * M + m) * d + dim] = in[idx];
}

__global__ void transpose_hmd_to_mhd_kernel(
    const float* __restrict__ in, float* __restrict__ out,
    int M, int H, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * M * d;
    if (idx >= total) return;
    int dim   = idx % d;
    int rem   = idx / d;
    int m     = rem % M;
    int h     = rem / M;
    // in:  [h, m, dim]
    // out: [m, h, dim]
    out[(m * H + h) * d + dim] = in[idx];
}

static void transpose_mhd_to_hmd(const float* in, float* out, int M, int H, int d) {
    int total = M * H * d;
    int block = 256;
    int grid = (total + block - 1) / block;
    transpose_mhd_to_hmd_kernel<<<grid, block>>>(in, out, M, H, d);
}

static void transpose_hmd_to_mhd(const float* in, float* out, int M, int H, int d) {
    int total = M * H * d;
    int block = 256;
    int grid = (total + block - 1) / block;
    transpose_hmd_to_mhd_kernel<<<grid, block>>>(in, out, M, H, d);
}

// ============================================================
// QKV stripe deinterleave: [M, 3*d] -> Q[M,d], K[M,d], V[M,d]
// HF c_attn packs Q||K||V along last dim per token.
// ============================================================
__global__ void qkv_split_kernel(
    const float* __restrict__ qkv,  // [M, 3*d]
    float* __restrict__ Q,
    float* __restrict__ K,
    float* __restrict__ V,
    int M, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * d;
    if (idx >= total) return;
    int m = idx / d;
    int i = idx % d;
    int src = m * 3 * d;
    Q[m * d + i] = qkv[src + i];
    K[m * d + i] = qkv[src + d + i];
    V[m * d + i] = qkv[src + 2 * d + i];
}

static void run_qkv_split(const float* qkv, float* Q, float* K, float* V, int M, int d) {
    int total = M * d;
    int block = 256;
    int grid = (total + block - 1) / block;
    qkv_split_kernel<<<grid, block>>>(qkv, Q, K, V, M, d);
}

// ============================================================
// KV cache write: copy [seq_len, H, d] at positions [start_pos, start_pos+seq_len)
// into paged cache [num_phys_blocks, block_size, H, d] via block_table.
// ============================================================
__global__ void kv_cache_write_kernel(
    const float* __restrict__ K_src,
    const float* __restrict__ V_src,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const int* __restrict__ block_table,
    int seq_len, int start_pos, int n_heads, int d_head, int block_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * n_heads * d_head;
    if (idx >= total) return;

    int t = idx / (n_heads * d_head);
    int rem = idx % (n_heads * d_head);
    int h = rem / d_head;
    int d = rem % d_head;

    int abs_pos = start_pos + t;
    int block_idx = abs_pos / block_size;
    int block_off = abs_pos % block_size;
    int phys = block_table[block_idx];

    int cache_idx = ((phys * block_size + block_off) * n_heads + h) * d_head + d;
    k_cache[cache_idx] = K_src[(t * n_heads + h) * d_head + d];
    v_cache[cache_idx] = V_src[(t * n_heads + h) * d_head + d];
}

// ============================================================
// cuBLAS backend helpers: bias add / bias+GELU / bias+residual
// ============================================================
__global__ void bias_add_kernel(float* __restrict__ C, const float* __restrict__ bias,
                                 int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    C[idx] += bias[idx % N];
}

__global__ void bias_gelu_kernel(float* __restrict__ C, const float* __restrict__ bias,
                                  int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    float v = C[idx] + bias[idx % N];
    float v3 = v * v * v;
    C[idx] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v3)));
}

__global__ void bias_residual_kernel(float* __restrict__ C, const float* __restrict__ bias,
                                      const float* __restrict__ residual, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    C[idx] += bias[idx % N] + residual[idx];
}

static void run_bias_add(float* C, const float* bias, int M, int N) {
    int total = M * N;
    bias_add_kernel<<<(total + 255) / 256, 256>>>(C, bias, M, N);
}

static void run_bias_gelu(float* C, const float* bias, int M, int N) {
    int total = M * N;
    bias_gelu_kernel<<<(total + 255) / 256, 256>>>(C, bias, M, N);
}

static void run_bias_residual(float* C, const float* bias, const float* residual,
                               int M, int N) {
    int total = M * N;
    bias_residual_kernel<<<(total + 255) / 256, 256>>>(C, bias, residual, M, N);
}

// Vocab projection (FP32): logits[j] = dot(hidden, wte[j])
__global__ void vocab_proj_kernel(
    const float* __restrict__ hidden,
    const float* __restrict__ wte,
    float* __restrict__ logits,
    int d, int V)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= V) return;
    float sum = 0.0f;
    const float* wj = wte + (size_t)j * d;
    for (int k = 0; k < d; ++k) sum += hidden[k] * wj[k];
    logits[j] = sum;
}

static void run_vocab_proj(const float* hidden, const float* wte,
                           float* logits, int d, int V) {
    int block = 256;
    int grid = (V + block - 1) / block;
    vocab_proj_kernel<<<grid, block>>>(hidden, wte, logits, d, V);
}

// ============================================================
// Engine methods
// ============================================================
void GPT2Engine::embed(const int* d_token_ids, int seq_len, int start_pos, float* out) {
    run_embedding(d_token_ids, seq_len, start_pos, config_.d_model,
                  weights_.wte, weights_.wpe, out);
}

void GPT2Engine::ensure_kv_blocks(int total_tokens) {
    int needed = (total_tokens + block_size_ - 1) / block_size_;
    int max_blocks = (config_.max_seq_len + block_size_ - 1) / block_size_;
    while (blocks_allocated_ < needed && blocks_allocated_ < max_blocks) {
        int phys = blocks_allocated_;
        CUDA_CHECK(cudaMemcpy(d_block_tables_ + blocks_allocated_, &phys,
                              sizeof(int), cudaMemcpyHostToDevice));
        blocks_allocated_++;
    }
}

void GPT2Engine::kv_cache_append(int layer, const float* K, const float* V,
                                  int seq_len, int start_pos) {
    int H = config_.n_heads;
    int dh = config_.d_head;
    size_t layer_offset = (size_t)layer * num_phys_blocks_ * block_size_ * H * dh;

    int total = seq_len * H * dh;
    int block = 256;
    int grid = (total + block - 1) / block;
    kv_cache_write_kernel<<<grid, block>>>(
        K, V,
        d_k_cache_ + layer_offset, d_v_cache_ + layer_offset,
        d_block_tables_,
        seq_len, start_pos, H, dh, block_size_);
}

void GPT2Engine::forward_layer(int layer, int seq_len, bool is_prefill) {
    int d  = config_.d_model;
    int H  = config_.n_heads;
    int dh = config_.d_head;
    int ff = config_.d_ff;
    auto& lw = weights_.layers[layer];
    int M = seq_len;

    // --- Attention sub-block (pre-norm + residual) ---
    // residual_save = x
    CUDA_CHECK(cudaMemcpy(d_residual_, d_x_,
                          (size_t)M * d * sizeof(float), cudaMemcpyDeviceToDevice));
    // x = LN(x)
    run_layernorm(M, d, d_x_, lw.ln1_gamma, lw.ln1_beta, d_x_);

    // QKV = x @ W_qkv^T + bias   [M, 3d]
    run_quantize_fp32_to_int8(M, d, d_x_, d_act_packed_, d_act_scale_);
    run_int8_gemm_bias(M, 3 * d, d, d_act_packed_, lw.qkv.packed,
                       d_act_scale_, lw.qkv.scale, lw.qkv_bias, d_qkv_);

    // Layout: d_qkv_ is [M, 3*d] per-token stripe Q||K||V (HF c_attn).
    // Deinterleave into separate [M, d] buffers for downstream use.
    run_qkv_split(d_qkv_, d_q_split_, d_k_split_, d_v_split_, M, d);
    float* Q_mhd = d_q_split_;
    float* K_mhd = d_k_split_;
    float* V_mhd = d_v_split_;

    // current start position
    int start_pos = current_pos_;

    // Write K, V to paged KV cache (layout [M, H, dh] matches cache)
    kv_cache_append(layer, K_mhd, V_mhd, M, start_pos);

    size_t layer_off = (size_t)layer * num_phys_blocks_ * block_size_ * H * dh;
    float* layer_k = d_k_cache_ + layer_off;
    float* layer_v = d_v_cache_ + layer_off;

    if (is_prefill) {
        // Transpose Q,K,V: [M,H,d] -> [H,M,d] for flash attn
        transpose_mhd_to_hmd(Q_mhd, d_q_buf_, M, H, dh);
        transpose_mhd_to_hmd(K_mhd, d_k_buf_, M, H, dh);
        transpose_mhd_to_hmd(V_mhd, d_v_buf_, M, H, dh);

        // flash attn output [1, H, M, d] into d_k_buf_ (reuse) as staging
        run_flash_attn_v2(1, H, M, dh, d_q_buf_, d_k_buf_, d_v_buf_, d_v_buf_, true);
        // Transpose back: [H,M,d] -> [M,H,d]
        transpose_hmd_to_mhd(d_v_buf_, d_attn_out_, M, H, dh);
    } else {
        // Decode: Q layout [1,H,1,d] == [H,d] in memory — Q_mhd already works
        int ctx = current_pos_ + 1;
        CUDA_CHECK(cudaMemcpy(d_context_lens_, &ctx, sizeof(int), cudaMemcpyHostToDevice));
        int max_blocks_seq = (ctx + block_size_ - 1) / block_size_;
        run_decode_attn(1, H, H, dh,
                        Q_mhd,
                        layer_k, layer_v,
                        d_block_tables_, d_context_lens_,
                        ctx, block_size_, max_blocks_seq,
                        d_attn_out_, d_decode_workspace_);
    }

    // Output projection: attn_out @ W_o^T + bias + residual -> x
    run_quantize_fp32_to_int8(M, d, d_attn_out_, d_act_packed_, d_act_scale_);
    run_int8_gemm_bias_residual(M, d, d, d_act_packed_, lw.out.packed,
                                 d_act_scale_, lw.out.scale, lw.out_bias,
                                 d_residual_, d_x_);

    // --- FFN sub-block (pre-norm + residual) ---
    CUDA_CHECK(cudaMemcpy(d_residual_, d_x_,
                          (size_t)M * d * sizeof(float), cudaMemcpyDeviceToDevice));
    run_layernorm(M, d, d_x_, lw.ln2_gamma, lw.ln2_beta, d_x_);

    // FFN up: [M,d] -> [M,ff] + bias + GELU
    run_quantize_fp32_to_int8(M, d, d_x_, d_act_packed_, d_act_scale_);
    run_int8_gemm_bias_gelu(M, ff, d, d_act_packed_, lw.ffn_up.packed,
                             d_act_scale_, lw.ffn_up.scale, lw.ffn_up_bias, d_ffn_hidden_);

    // FFN down: [M,ff] -> [M,d] + bias + residual
    run_quantize_fp32_to_int8(M, ff, d_ffn_hidden_, d_act_packed_, d_act_scale_);
    run_int8_gemm_bias_residual(M, d, ff, d_act_packed_, lw.ffn_down.packed,
                                 d_act_scale_, lw.ffn_down.scale, lw.ffn_down_bias,
                                 d_residual_, d_x_);
}

void GPT2Engine::forward_layer_cublas(int layer, int seq_len, bool is_prefill) {
    int d  = config_.d_model;
    int H  = config_.n_heads;
    int dh = config_.d_head;
    int ff = config_.d_ff;
    auto& lw = weights_.layers[layer];
    auto& fw = weights_.layers_fp32[layer];
    int M = seq_len;
    float alpha = 1.0f, beta = 0.0f;

    // --- Attention sub-block ---
    CUDA_CHECK(cudaMemcpy(d_residual_, d_x_,
                          (size_t)M * d * sizeof(float), cudaMemcpyDeviceToDevice));
    run_layernorm(M, d, d_x_, lw.ln1_gamma, lw.ln1_beta, d_x_);

    // QKV = x @ W_qkv + bias    [M, 3d] = [M,d] @ [d, 3d]
    cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                3 * d, M, d, &alpha,
                fw.qkv.data, 3 * d,
                d_x_, d,
                &beta, d_qkv_, 3 * d);
    run_bias_add(d_qkv_, lw.qkv_bias, M, 3 * d);

    run_qkv_split(d_qkv_, d_q_split_, d_k_split_, d_v_split_, M, d);
    float* Q_mhd = d_q_split_;
    float* K_mhd = d_k_split_;
    float* V_mhd = d_v_split_;

    int start_pos = current_pos_;
    kv_cache_append(layer, K_mhd, V_mhd, M, start_pos);

    size_t layer_off = (size_t)layer * num_phys_blocks_ * block_size_ * H * dh;
    float* layer_k = d_k_cache_ + layer_off;
    float* layer_v = d_v_cache_ + layer_off;

    if (is_prefill) {
        transpose_mhd_to_hmd(Q_mhd, d_q_buf_, M, H, dh);
        transpose_mhd_to_hmd(K_mhd, d_k_buf_, M, H, dh);
        transpose_mhd_to_hmd(V_mhd, d_v_buf_, M, H, dh);
        run_flash_attn_v2(1, H, M, dh, d_q_buf_, d_k_buf_, d_v_buf_, d_v_buf_, true);
        transpose_hmd_to_mhd(d_v_buf_, d_attn_out_, M, H, dh);
    } else {
        int ctx = current_pos_ + 1;
        CUDA_CHECK(cudaMemcpy(d_context_lens_, &ctx, sizeof(int), cudaMemcpyHostToDevice));
        int max_blocks_seq = (ctx + block_size_ - 1) / block_size_;
        run_decode_attn(1, H, H, dh,
                        Q_mhd, layer_k, layer_v,
                        d_block_tables_, d_context_lens_,
                        ctx, block_size_, max_blocks_seq,
                        d_attn_out_, d_decode_workspace_);
    }

    // Output projection: attn_out @ W_o + bias + residual  -> x   [M,d]=[M,d]@[d,d]
    cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                d, M, d, &alpha,
                fw.out.data, d,
                d_attn_out_, d,
                &beta, d_x_, d);
    run_bias_residual(d_x_, lw.out_bias, d_residual_, M, d);

    // --- FFN sub-block ---
    CUDA_CHECK(cudaMemcpy(d_residual_, d_x_,
                          (size_t)M * d * sizeof(float), cudaMemcpyDeviceToDevice));
    run_layernorm(M, d, d_x_, lw.ln2_gamma, lw.ln2_beta, d_x_);

    // FFN up: [M,d] -> [M,ff]
    cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                ff, M, d, &alpha,
                fw.ffn_up.data, ff,
                d_x_, d,
                &beta, d_ffn_hidden_, ff);
    run_bias_gelu(d_ffn_hidden_, lw.ffn_up_bias, M, ff);

    // FFN down: [M,ff] -> [M,d]
    cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                d, M, ff, &alpha,
                fw.ffn_down.data, d,
                d_ffn_hidden_, ff,
                &beta, d_x_, d);
    run_bias_residual(d_x_, lw.ffn_down_bias, d_residual_, M, d);
}

void GPT2Engine::forward_final_ln(int seq_len, float* out_last_token) {
    int d = config_.d_model;
    const float* last = d_x_ + (size_t)(seq_len - 1) * d;
    run_layernorm(1, d, last, weights_.ln_f_gamma, weights_.ln_f_beta, out_last_token);
}

void GPT2Engine::forward_logits(const float* x_last, float* out_logits) {
    run_vocab_proj(x_last, weights_.wte, out_logits, config_.d_model, config_.vocab_size);
}

void GPT2Engine::reset_session() {
    blocks_allocated_ = 0;
    current_pos_ = 0;
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_context_lens_, &zero, sizeof(int), cudaMemcpyHostToDevice));
}

void GPT2Engine::generate(const std::vector<int>& prompt_tokens,
                           int max_new_tokens,
                           float temperature,
                           int top_k,
                           bool greedy,
                           std::function<void(int)> token_callback,
                           InferenceMetrics& metrics,
                           bool reset_kv) {
    int prompt_len = (int)prompt_tokens.size();
    int d = config_.d_model;
    int V = config_.vocab_size;

    metrics = InferenceMetrics{};
    metrics.max_tokens = max_new_tokens;
    metrics.generating = true;

    // Upload prompt tokens to a small device buffer
    int* d_tokens;
    int max_tokens_buf = std::max(prompt_len, 1);
    CUDA_CHECK(cudaMalloc(&d_tokens, max_tokens_buf * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tokens, prompt_tokens.data(),
                          prompt_len * sizeof(int), cudaMemcpyHostToDevice));

    if (reset_kv) reset_session();

    // Scratch for single-token hidden state
    float* d_last_hidden;
    CUDA_CHECK(cudaMalloc(&d_last_hidden, d * sizeof(float)));

    GpuTimer total_timer, step_timer;
    std::vector<float> itl_samples;

    // === Prefill ===
    total_timer.tic();
    step_timer.tic();

    if (current_pos_ == 0) {
        // Fresh session: fused flash-attn prefill over all prompt tokens.
        ensure_kv_blocks(prompt_len);
        embed(d_tokens, prompt_len, 0, d_x_);

        for (int l = 0; l < config_.n_layers; ++l) {
            if (backend_ == InferenceBackend::CUBLAS_FP32)
                forward_layer_cublas(l, prompt_len, /*is_prefill=*/true);
            else
                forward_layer(l, prompt_len, /*is_prefill=*/true);
        }

        current_pos_ = prompt_len;
        CUDA_CHECK(cudaMemcpy(d_context_lens_, &current_pos_, sizeof(int), cudaMemcpyHostToDevice));
        forward_final_ln(prompt_len, d_last_hidden);
    } else {
        // Continuation: feed each new prompt token through decode path so it
        // attends to the existing KV cache, growing the cache linearly.
        for (int i = 0; i < prompt_len; ++i) {
            ensure_kv_blocks(current_pos_ + 1);
            CUDA_CHECK(cudaMemcpy(d_tokens, &prompt_tokens[i], sizeof(int),
                                  cudaMemcpyHostToDevice));
            embed(d_tokens, 1, current_pos_, d_x_);
            for (int l = 0; l < config_.n_layers; ++l) {
                if (backend_ == InferenceBackend::CUBLAS_FP32)
                    forward_layer_cublas(l, 1, /*is_prefill=*/false);
                else
                    forward_layer(l, 1, /*is_prefill=*/false);
            }
            current_pos_++;
            CUDA_CHECK(cudaMemcpy(d_context_lens_, &current_pos_, sizeof(int), cudaMemcpyHostToDevice));
        }
        forward_final_ln(1, d_last_hidden);
    }

    forward_logits(d_last_hidden, d_logits_);
    CUDA_CHECK(cudaDeviceSynchronize());

    int next_token;
    if (greedy)
        next_token = sample_argmax(d_logits_, V);
    else
        next_token = sample_top_k(d_logits_, V, top_k, temperature, 42u);

    metrics.ttft_ms = step_timer.toc();

    // === Decode loop ===
    int tokens_generated = 0;
    unsigned int sample_seed = 42u;
    float total_decode_ms = 0.0f;

    for (int step = 0; step < max_new_tokens; ++step) {
        if (next_token == 50256) break;

        token_callback(next_token);
        tokens_generated++;
        metrics.tokens_generated = tokens_generated;

        step_timer.tic();

        ensure_kv_blocks(current_pos_ + 1);

        CUDA_CHECK(cudaMemcpy(d_tokens, &next_token, sizeof(int), cudaMemcpyHostToDevice));
        embed(d_tokens, 1, current_pos_, d_x_);

        for (int l = 0; l < config_.n_layers; ++l) {
            if (backend_ == InferenceBackend::CUBLAS_FP32)
                forward_layer_cublas(l, 1, /*is_prefill=*/false);
            else
                forward_layer(l, 1, /*is_prefill=*/false);
        }

        current_pos_++;
        CUDA_CHECK(cudaMemcpy(d_context_lens_, &current_pos_, sizeof(int), cudaMemcpyHostToDevice));

        forward_final_ln(1, d_last_hidden);
        forward_logits(d_last_hidden, d_logits_);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (greedy)
            next_token = sample_argmax(d_logits_, V);
        else
            next_token = sample_top_k(d_logits_, V, top_k, temperature, ++sample_seed);

        float step_ms = step_timer.toc();
        itl_samples.push_back(step_ms);
        total_decode_ms += step_ms;

        metrics.tpot_ms = total_decode_ms / tokens_generated;
        metrics.tps = 1000.0f * tokens_generated / total_decode_ms;
        metrics.itl_min_ms = *std::min_element(itl_samples.begin(), itl_samples.end());
        metrics.itl_max_ms = *std::max_element(itl_samples.begin(), itl_samples.end());
        if (itl_samples.size() >= 20) {
            auto sorted = itl_samples;
            std::sort(sorted.begin(), sorted.end());
            metrics.itl_p95_ms = sorted[(int)(sorted.size() * 0.95)];
        } else {
            metrics.itl_p95_ms = metrics.itl_max_ms;
        }

        int max_blocks = (config_.max_seq_len + block_size_ - 1) / block_size_;
        metrics.kv_cache_pct = 100.0f * blocks_allocated_ / max_blocks;

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        metrics.vram_used_mb = (total_mem - free_mem) / (1024.0f * 1024.0f);
        metrics.vram_total_mb = total_mem / (1024.0f * 1024.0f);

        // MBU: theoretical 192 GB/s on GTX 1650 Ti, ~140MB INT8 weights per token
        float theo_max_tps = 192e9f / 140e6f;
        metrics.mbu = metrics.tps / theo_max_tps;
    }

    metrics.end_to_end_ms = total_timer.toc();
    metrics.generating = false;

    cudaFree(d_tokens);
    cudaFree(d_last_hidden);
}

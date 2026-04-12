// src/gpt2_engine.cu
#include "gpt2_engine.cuh"
#include "timer.cuh"
#include "quantization/14_int8_gemm.cuh"
#include "layernorm/15_layernorm_residual.cuh"
#include "decode/13_decode_attn.cuh"

#include <fstream>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>

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

GPT2Engine::GPT2Engine(const std::string& model_dir) : model_dir_(model_dir) {
    load_config();
    load_weights();
    alloc_workspace();
}

void GPT2Engine::free_all() {
    cudaFree(d_x_); cudaFree(d_residual_);
    cudaFree(d_qkv_); cudaFree(d_attn_out_);
    cudaFree(d_ffn_hidden_); cudaFree(d_logits_);
    cudaFree(d_q_buf_); cudaFree(d_k_buf_); cudaFree(d_v_buf_);
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
}

GPT2Engine::~GPT2Engine() {
    free_all();
}

// Forward pass + generate implemented in Task 8.
void GPT2Engine::generate(const std::vector<int>& /*prompt*/,
                          int /*max_new*/,
                          float /*temperature*/,
                          int /*top_k*/,
                          bool /*greedy*/,
                          std::function<void(int)> /*cb*/,
                          InferenceMetrics& /*m*/) {
    fprintf(stderr, "GPT2Engine::generate not yet implemented (Task 8)\n");
}

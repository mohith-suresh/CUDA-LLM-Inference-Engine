// src/gpt2_engine.cuh
#pragma once
#include "gpt2_types.cuh"
#include <string>
#include <vector>
#include <functional>

class GPT2Engine {
public:
    GPT2Engine(const std::string& model_dir);
    ~GPT2Engine();

    void generate(const std::vector<int>& prompt_tokens,
                  int max_new_tokens,
                  float temperature,
                  int top_k,
                  bool greedy,
                  std::function<void(int token_id)> token_callback,
                  InferenceMetrics& metrics);

    const GPT2Config& config() const { return config_; }

private:
    GPT2Config config_;
    GPT2Weights weights_;
    std::string model_dir_;

    // Activation workspace
    float* d_x_;
    float* d_residual_;
    float* d_qkv_;
    float* d_attn_out_;
    float* d_ffn_hidden_;
    float* d_logits_;

    // Runtime quant workspace (activation quantization per-row)
    int32_t* d_act_packed_;
    float* d_act_scale_;

    // Paged KV cache
    float* d_k_cache_;     // [n_layers, num_phys_blocks, block_size, n_heads, d_head]
    float* d_v_cache_;
    int* d_block_tables_;  // [max_blocks]
    int* d_context_lens_;  // [1]
    int num_phys_blocks_;
    int blocks_allocated_;
    int block_size_;
    int current_pos_;      // total tokens in KV cache

    // Decode attn workspace
    float* d_decode_workspace_;

    // Scratch buffers for Q K V split / reshape during attention
    float* d_q_buf_;   // [seq, n_heads, d_head]
    float* d_k_buf_;
    float* d_v_buf_;

    void load_config();
    void load_weights();
    void alloc_workspace();
    void free_all();

    // Forward helpers
    void embed(const int* host_token_ids, int seq_len, int start_pos, float* out);
    void forward_layer(int layer, int seq_len, bool is_prefill);
    void forward_final_ln(int seq_len, float* out_last_token);
    void forward_logits(const float* x_last, float* out_logits);
    int sample_token(float* d_logits, float temperature, int top_k, bool greedy);

    // KV cache helpers
    void kv_cache_append(int layer, const float* K, const float* V, int seq_len, int start_pos);
    void ensure_kv_blocks(int total_tokens);
};

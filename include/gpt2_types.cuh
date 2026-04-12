// include/gpt2_types.cuh
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

struct GPT2Config {
    int n_layers;
    int n_heads;
    int d_model;
    int d_ff;
    int vocab_size;
    int max_seq_len;
    int d_head;
};

struct QuantWeight {
    int32_t* packed;    // [N, K/4] int8x4 packed as int32
    float* scale;       // [N] per-row scales
    int N;
    int K;
};

struct LayerWeights {
    float* ln1_gamma;
    float* ln1_beta;

    QuantWeight qkv;
    float* qkv_bias;

    QuantWeight out;
    float* out_bias;

    float* ln2_gamma;
    float* ln2_beta;

    QuantWeight ffn_up;
    float* ffn_up_bias;

    QuantWeight ffn_down;
    float* ffn_down_bias;
};

struct GPT2Weights {
    float* wte;
    float* wpe;
    float* ln_f_gamma;
    float* ln_f_beta;
    LayerWeights layers[12];
};

struct InferenceMetrics {
    float ttft_ms;
    float tpot_ms;
    float itl_min_ms;
    float itl_max_ms;
    float itl_p95_ms;
    float tps;
    int tokens_generated;
    int max_tokens;
    float kv_cache_pct;
    float vram_used_mb;
    float vram_total_mb;
    float mbu;
    float end_to_end_ms;
    bool generating;
};

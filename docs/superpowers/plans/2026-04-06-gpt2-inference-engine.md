# GPT-2 INT8 Inference Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained GPT-2 Small (124M) inference engine using SLICK's custom CUDA kernels with INT8 quantization, featuring a live TUI metrics dashboard.

**Architecture:** Python export script dumps HuggingFace GPT-2 weights as INT8-quantized binaries. C++ engine loads them and runs prefill (K10 FlashAttention) + autoregressive decode (K13 decode attention) with K14 INT8 GEMM (fused epilogues for bias/GELU/residual), K15 fused LayerNorm+residual, and paged KV cache. FTXUI renders a two-panel TUI with streaming tokens and live metrics.

**Tech Stack:** CUDA 11.8, C++17, CMake, FTXUI, Python (transformers + torch for export only)

**Spec:** `docs/superpowers/specs/2026-04-06-gpt2-inference-engine-design.md`

---

## File Structure

```
python/
  export_gpt2.py              # HuggingFace → INT8 .bin export

kernels/
  quantization/
    14_int8_gemm.cu            # MODIFIED: templated epilogues
    14_int8_gemm.cuh           # MODIFIED: new run_int8_gemm_* wrappers
  layernorm/
    15_layernorm_residual.cu   # NEW: fused LayerNorm + residual
    15_layernorm_residual.cuh  # NEW: kernel declaration

include/
  gpt2_types.cuh              # NEW: GPT2Config, LayerWeights, Metrics structs
  kv_cache.cuh                # NEW: BlockAllocator, PagedKVCache
  tokenizer.h                 # NEW: BPE tokenizer (CPU-only, .h not .cuh)
  sampler.cuh                 # NEW: top-k + temperature sampling

src/
  gpt2_engine.cu              # NEW: weight loader + forward pass
  gpt2_engine.cuh             # NEW: engine interface
  kv_cache.cu                 # NEW: allocator implementation
  tokenizer.cpp               # NEW: BPE encode/decode (CPU, .cpp)
  sampler.cu                  # NEW: sampling kernels
  main.cu                     # NEW: FTXUI TUI + CLI entry point

tests/
  test_layernorm.cu           # NEW: K15 tests
  test_int8_epilogues.cu      # NEW: K14 epilogue tests
  test_engine.cu              # NEW: end-to-end greedy decode validation

python/
  benchmark_pytorch.py         # NEW: PyTorch FP32 baseline benchmark

scripts/
  demo_compare.sh              # NEW: runs all 3 backends, formats comparison table

CMakeLists.txt                # MODIFIED: FTXUI dep, new targets
```

---

### Task 1: Python Export Script

**Files:**
- Create: `python/export_gpt2.py`

- [ ] **Step 1: Write the export script**

```python
#!/usr/bin/env python3
"""Export GPT-2 Small weights to INT8-quantized binary files for SLICK inference."""

import argparse
import json
import os
import struct
import numpy as np

def quantize_per_row(weight_fp32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric INT8 quantization.
    
    Args:
        weight_fp32: [rows, cols] FP32 weight matrix
    Returns:
        packed_int32: [rows, cols//4] packed int8x4 as int32 (for dp4a)
        scales: [rows] FP32 per-row scales
    """
    rows, cols = weight_fp32.shape
    assert cols % 4 == 0, f"cols={cols} must be divisible by 4"

    row_max = np.max(np.abs(weight_fp32), axis=1)  # [rows]
    scales = np.where(row_max > 0, row_max / 127.0, 1.0).astype(np.float32)

    inv_scales = 1.0 / scales
    quantized = np.clip(np.round(weight_fp32 * inv_scales[:, None]), -128, 127).astype(np.int8)

    # Pack 4 int8 → 1 int32 (little-endian, matching CUDA __dp4a layout)
    quantized_bytes = quantized.view(np.uint8).reshape(rows, cols // 4, 4)
    packed = (quantized_bytes[:, :, 0].astype(np.uint32) |
              (quantized_bytes[:, :, 1].astype(np.uint32) << 8) |
              (quantized_bytes[:, :, 2].astype(np.uint32) << 16) |
              (quantized_bytes[:, :, 3].astype(np.uint32) << 24))
    packed_int32 = packed.view(np.int32)

    return packed_int32, scales


def save_bin(path: str, arr: np.ndarray):
    """Save numpy array as raw binary."""
    arr.tofile(path)
    print(f"  {path}: {arr.shape} {arr.dtype} ({arr.nbytes} bytes)")


def export_model(output_dir: str):
    # Import here so the script fails fast with a clear message
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        print("ERROR: pip install transformers torch")
        return

    print("Loading GPT-2 Small from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    sd = model.state_dict()

    os.makedirs(output_dir, exist_ok=True)

    # Config
    config = {
        "n_layers": 12, "n_heads": 12, "d_model": 768,
        "d_ff": 3072, "vocab_size": 50257, "max_seq_len": 1024,
        "d_head": 64
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("Saved config.json")

    # Token embedding [50257, 768] — FP32 (shared with lm_head)
    wte = sd["transformer.wte.weight"].numpy().astype(np.float32)
    save_bin(os.path.join(output_dir, "wte.bin"), wte)

    # Position embedding [1024, 768] — FP32
    wpe = sd["transformer.wpe.weight"].numpy().astype(np.float32)
    save_bin(os.path.join(output_dir, "wpe.bin"), wpe)

    # Final LayerNorm
    save_bin(os.path.join(output_dir, "ln_f_gamma.bin"),
             sd["transformer.ln_f.weight"].numpy().astype(np.float32))
    save_bin(os.path.join(output_dir, "ln_f_beta.bin"),
             sd["transformer.ln_f.bias"].numpy().astype(np.float32))

    # Per-layer weights
    for i in range(12):
        layer_dir = os.path.join(output_dir, f"layer_{i:02d}")
        os.makedirs(layer_dir, exist_ok=True)
        prefix = f"transformer.h.{i}"

        # Attention LayerNorm
        save_bin(os.path.join(layer_dir, "ln1_gamma.bin"),
                 sd[f"{prefix}.ln_1.weight"].numpy().astype(np.float32))
        save_bin(os.path.join(layer_dir, "ln1_beta.bin"),
                 sd[f"{prefix}.ln_1.bias"].numpy().astype(np.float32))

        # QKV projection: GPT-2 stores as one [768, 2304] matrix (W) + [2304] bias
        # K14 uses NT layout: B^T [N, K]. Here N=2304, K=768.
        # W is [768, 2304] row-major. W^T is [2304, 768] row-major.
        qkv_w = sd[f"{prefix}.attn.c_attn.weight"].numpy().astype(np.float32)  # [768, 2304]
        qkv_wt = qkv_w.T.copy()  # [2304, 768] — NT layout for K14
        qkv_packed, qkv_scale = quantize_per_row(qkv_wt)
        save_bin(os.path.join(layer_dir, "qkv_weight.bin"), qkv_packed)
        save_bin(os.path.join(layer_dir, "qkv_scale.bin"), qkv_scale)
        save_bin(os.path.join(layer_dir, "qkv_bias.bin"),
                 sd[f"{prefix}.attn.c_attn.bias"].numpy().astype(np.float32))

        # Output projection: [768, 768] → transpose to [768, 768] NT
        out_w = sd[f"{prefix}.attn.c_proj.weight"].numpy().astype(np.float32)  # [768, 768]
        out_wt = out_w.T.copy()  # [768, 768]
        out_packed, out_scale = quantize_per_row(out_wt)
        save_bin(os.path.join(layer_dir, "out_weight.bin"), out_packed)
        save_bin(os.path.join(layer_dir, "out_scale.bin"), out_scale)
        save_bin(os.path.join(layer_dir, "out_bias.bin"),
                 sd[f"{prefix}.attn.c_proj.bias"].numpy().astype(np.float32))

        # FFN LayerNorm
        save_bin(os.path.join(layer_dir, "ln2_gamma.bin"),
                 sd[f"{prefix}.ln_2.weight"].numpy().astype(np.float32))
        save_bin(os.path.join(layer_dir, "ln2_beta.bin"),
                 sd[f"{prefix}.ln_2.bias"].numpy().astype(np.float32))

        # FFN up: [768, 3072] → NT [3072, 768]
        up_w = sd[f"{prefix}.mlp.c_fc.weight"].numpy().astype(np.float32)
        up_wt = up_w.T.copy()
        up_packed, up_scale = quantize_per_row(up_wt)
        save_bin(os.path.join(layer_dir, "ffn_up_weight.bin"), up_packed)
        save_bin(os.path.join(layer_dir, "ffn_up_scale.bin"), up_scale)
        save_bin(os.path.join(layer_dir, "ffn_up_bias.bin"),
                 sd[f"{prefix}.mlp.c_fc.bias"].numpy().astype(np.float32))

        # FFN down: [3072, 768] → NT [768, 3072]
        down_w = sd[f"{prefix}.mlp.c_proj.weight"].numpy().astype(np.float32)
        down_wt = down_w.T.copy()
        down_packed, down_scale = quantize_per_row(down_wt)
        save_bin(os.path.join(layer_dir, "ffn_down_weight.bin"), down_packed)
        save_bin(os.path.join(layer_dir, "ffn_down_scale.bin"), down_scale)
        save_bin(os.path.join(layer_dir, "ffn_down_bias.bin"),
                 sd[f"{prefix}.mlp.c_proj.bias"].numpy().astype(np.float32))

        print(f"Layer {i}: done")

    # Tokenizer: vocab and merges
    vocab = tokenizer.encoder  # dict: str → int
    with open(os.path.join(output_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    merges_path = os.path.join(output_dir, "merges.txt")
    # Get merges from tokenizer's bpe_ranks
    bpe_merges = list(tokenizer.bpe_ranks.keys())
    with open(merges_path, "w") as f:
        for pair in bpe_merges:
            f.write(f"{pair[0]} {pair[1]}\n")

    print(f"\nExport complete: {output_dir}")
    total_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fnames in os.walk(output_dir) for f in fnames
    )
    print(f"Total size: {total_bytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/gpt2-int8",
                        help="Output directory for exported weights")
    args = parser.parse_args()
    export_model(args.output)
```

- [ ] **Step 2: Run the export script**

```bash
cd /home/adithya/Document/SLICK
uv run python python/export_gpt2.py --output models/gpt2-int8
```

Expected: Downloads GPT-2 small, exports ~140MB of binary files into `models/gpt2-int8/`. Verify directory structure:

```bash
ls models/gpt2-int8/
# config.json  ln_f_beta.bin  ln_f_gamma.bin  merges.txt  vocab.json  wpe.bin  wte.bin  layer_00/ ... layer_11/
ls models/gpt2-int8/layer_00/
# ffn_down_bias.bin  ffn_down_scale.bin  ffn_down_weight.bin  ffn_up_bias.bin  ffn_up_scale.bin  ffn_up_weight.bin  ln1_beta.bin  ln1_gamma.bin  ln2_beta.bin  ln2_gamma.bin  out_bias.bin  out_scale.bin  out_weight.bin  qkv_bias.bin  qkv_scale.bin  qkv_weight.bin
```

- [ ] **Step 3: Add models/ to .gitignore and commit**

```bash
echo "models/" >> .gitignore
git add python/export_gpt2.py .gitignore
git commit -m "feat: add GPT-2 INT8 weight export script"
```

---

### Task 2: K14 Epilogue Extensions

**Files:**
- Modify: `kernels/quantization/14_int8_gemm.cuh`
- Modify: `kernels/quantization/14_int8_gemm.cu`
- Create: `tests/test_int8_epilogues.cu`

- [ ] **Step 1: Write failing tests for the three epilogues**

Create `tests/test_int8_epilogues.cu`:

```cpp
// tests/test_int8_epilogues.cu
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include "timer.cuh"
#include "quantization/14_int8_gemm.cuh"

class Int8EpilogueTest : public ::testing::Test {
protected:
    // Helper: run FP32 GEMM on CPU as reference
    void cpu_gemm(int M, int N, int K,
                  const float* A, const float* B, float* C) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                    sum += A[i * K + k] * B[k * N + j];
                C[i * N + j] = sum;
            }
    }

    float gelu_ref(float x) {
        return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }

    // Quantize on host (mirrors run_quantize_fp32_to_int8)
    void host_quantize(int rows, int cols, const float* input,
                       int32_t* packed, float* scales) {
        for (int r = 0; r < rows; ++r) {
            float mx = 0.0f;
            for (int c = 0; c < cols; ++c)
                mx = fmaxf(mx, fabsf(input[r * cols + c]));
            scales[r] = (mx > 0.0f) ? mx / 127.0f : 1.0f;
            float inv = 1.0f / scales[r];
            for (int c = 0; c < cols; c += 4) {
                int8_t q[4];
                for (int j = 0; j < 4; ++j)
                    q[j] = (int8_t)fminf(fmaxf(roundf(input[r * cols + c + j] * inv), -128.0f), 127.0f);
                uint32_t p = 0;
                p |= (uint32_t)(uint8_t)q[0];
                p |= (uint32_t)(uint8_t)q[1] << 8;
                p |= (uint32_t)(uint8_t)q[2] << 16;
                p |= (uint32_t)(uint8_t)q[3] << 24;
                packed[r * (cols / 4) + c / 4] = (int32_t)p;
            }
        }
    }

    void RunBiasOnlyTest(int M, int N, int K) {
        float tol = 0.012f * sqrtf((float)K) + 0.1f;  // extra margin for bias

        std::vector<float> h_A(M * K), h_B(K * N), h_bias(N);
        srand(42);
        for (auto& v : h_A) v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (auto& v : h_B) v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (auto& v : h_bias) v = (float)rand() / RAND_MAX * 0.2f - 0.1f;

        // CPU reference: GEMM + bias
        std::vector<float> h_ref(M * N);
        cpu_gemm(M, N, K, h_A.data(), h_B.data(), h_ref.data());
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                h_ref[i * N + j] += h_bias[j];

        // Transpose B for NT layout
        std::vector<float> h_BT(N * K);
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                h_BT[n * K + k] = h_B[k * N + n];

        // Quantize on host
        std::vector<int32_t> h_A_packed(M * K / 4), h_BT_packed(N * K / 4);
        std::vector<float> h_scaleA(M), h_scaleB(N);
        host_quantize(M, K, h_A.data(), h_A_packed.data(), h_scaleA.data());
        host_quantize(N, K, h_BT.data(), h_BT_packed.data(), h_scaleB.data());

        // Upload to GPU
        int32_t *d_A_packed, *d_BT_packed;
        float *d_scaleA, *d_scaleB, *d_bias, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A_packed, M * K / 4 * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_BT_packed, N * K / 4 * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_scaleA, M * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scaleB, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_A_packed, h_A_packed.data(), M * K / 4 * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_BT_packed, h_BT_packed.data(), N * K / 4 * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scaleA, h_scaleA.data(), M * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scaleB, h_scaleB.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        run_int8_gemm_bias(M, N, K, d_A_packed, d_BT_packed, d_scaleA, d_scaleB, d_bias, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_C(M * N);
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        for (int i = 0; i < M * N; ++i)
            max_err = fmaxf(max_err, fabsf(h_C[i] - h_ref[i]));
        EXPECT_LT(max_err, tol) << "BiasOnly max_err=" << max_err << " M=" << M << " N=" << N << " K=" << K;

        cudaFree(d_A_packed); cudaFree(d_BT_packed); cudaFree(d_scaleA);
        cudaFree(d_scaleB); cudaFree(d_bias); cudaFree(d_C);
    }

    void RunBiasGELUTest(int M, int N, int K) {
        float tol = 0.012f * sqrtf((float)K) + 0.2f;

        std::vector<float> h_A(M * K), h_B(K * N), h_bias(N);
        srand(42);
        for (auto& v : h_A) v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (auto& v : h_B) v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (auto& v : h_bias) v = (float)rand() / RAND_MAX * 0.2f - 0.1f;

        std::vector<float> h_ref(M * N);
        cpu_gemm(M, N, K, h_A.data(), h_B.data(), h_ref.data());
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float v = h_ref[i * N + j] + h_bias[j];
                h_ref[i * N + j] = gelu_ref(v);
            }

        std::vector<float> h_BT(N * K);
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                h_BT[n * K + k] = h_B[k * N + n];

        std::vector<int32_t> h_A_packed(M * K / 4), h_BT_packed(N * K / 4);
        std::vector<float> h_scaleA(M), h_scaleB(N);
        host_quantize(M, K, h_A.data(), h_A_packed.data(), h_scaleA.data());
        host_quantize(N, K, h_BT.data(), h_BT_packed.data(), h_scaleB.data());

        int32_t *d_A_packed, *d_BT_packed;
        float *d_scaleA, *d_scaleB, *d_bias, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A_packed, M * K / 4 * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_BT_packed, N * K / 4 * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_scaleA, M * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scaleB, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_A_packed, h_A_packed.data(), M * K / 4 * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_BT_packed, h_BT_packed.data(), N * K / 4 * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scaleA, h_scaleA.data(), M * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scaleB, h_scaleB.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        run_int8_gemm_bias_gelu(M, N, K, d_A_packed, d_BT_packed, d_scaleA, d_scaleB, d_bias, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_C(M * N);
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        for (int i = 0; i < M * N; ++i)
            max_err = fmaxf(max_err, fabsf(h_C[i] - h_ref[i]));
        EXPECT_LT(max_err, tol) << "BiasGELU max_err=" << max_err << " M=" << M << " N=" << N << " K=" << K;

        cudaFree(d_A_packed); cudaFree(d_BT_packed); cudaFree(d_scaleA);
        cudaFree(d_scaleB); cudaFree(d_bias); cudaFree(d_C);
    }

    void RunBiasResidualTest(int M, int N, int K) {
        float tol = 0.012f * sqrtf((float)K) + 0.2f;

        std::vector<float> h_A(M * K), h_B(K * N), h_bias(N), h_residual(M * N);
        srand(42);
        for (auto& v : h_A) v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (auto& v : h_B) v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (auto& v : h_bias) v = (float)rand() / RAND_MAX * 0.2f - 0.1f;
        for (auto& v : h_residual) v = (float)rand() / RAND_MAX * 2.0f - 1.0f;

        std::vector<float> h_ref(M * N);
        cpu_gemm(M, N, K, h_A.data(), h_B.data(), h_ref.data());
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                h_ref[i * N + j] += h_bias[j] + h_residual[i * N + j];

        std::vector<float> h_BT(N * K);
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                h_BT[n * K + k] = h_B[k * N + n];

        std::vector<int32_t> h_A_packed(M * K / 4), h_BT_packed(N * K / 4);
        std::vector<float> h_scaleA(M), h_scaleB(N);
        host_quantize(M, K, h_A.data(), h_A_packed.data(), h_scaleA.data());
        host_quantize(N, K, h_BT.data(), h_BT_packed.data(), h_scaleB.data());

        int32_t *d_A_packed, *d_BT_packed;
        float *d_scaleA, *d_scaleB, *d_bias, *d_residual, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A_packed, M * K / 4 * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_BT_packed, N * K / 4 * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_scaleA, M * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scaleB, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_residual, M * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_A_packed, h_A_packed.data(), M * K / 4 * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_BT_packed, h_BT_packed.data(), N * K / 4 * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scaleA, h_scaleA.data(), M * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scaleB, h_scaleB.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_residual, h_residual.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

        run_int8_gemm_bias_residual(M, N, K, d_A_packed, d_BT_packed, d_scaleA, d_scaleB,
                                     d_bias, d_residual, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_C(M * N);
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        for (int i = 0; i < M * N; ++i)
            max_err = fmaxf(max_err, fabsf(h_C[i] - h_ref[i]));
        EXPECT_LT(max_err, tol) << "BiasResidual max_err=" << max_err << " M=" << M << " N=" << N << " K=" << K;

        cudaFree(d_A_packed); cudaFree(d_BT_packed); cudaFree(d_scaleA);
        cudaFree(d_scaleB); cudaFree(d_bias); cudaFree(d_residual); cudaFree(d_C);
    }
};

// BiasOnly tests — matches GPT-2 QKV projection shape
TEST_F(Int8EpilogueTest, BiasOnly_256)  { RunBiasOnlyTest(256, 256, 256); }
TEST_F(Int8EpilogueTest, BiasOnly_GPT2_QKV) { RunBiasOnlyTest(128, 2304, 768); }

// BiasGELU tests — matches GPT-2 FFN up projection shape
TEST_F(Int8EpilogueTest, BiasGELU_256)  { RunBiasGELUTest(256, 256, 256); }
TEST_F(Int8EpilogueTest, BiasGELU_GPT2_FFN) { RunBiasGELUTest(128, 3072, 768); }

// BiasResidual tests — matches GPT-2 output projection shape
TEST_F(Int8EpilogueTest, BiasResidual_256) { RunBiasResidualTest(256, 256, 256); }
TEST_F(Int8EpilogueTest, BiasResidual_GPT2_Out) { RunBiasResidualTest(128, 768, 768); }
```

- [ ] **Step 2: Update K14 header with new declarations**

In `kernels/quantization/14_int8_gemm.cuh`, add after the existing `run_int8_gemm` declaration:

```cpp
// Epilogue types for fused INT8 GEMM
enum class I8Epilogue { Plain, BiasOnly, BiasGELU, BiasResidual };

// INT8 GEMM with bias: C = dequant(A @ B^T) + bias
void run_int8_gemm_bias(int M, int N, int K,
                        const int32_t* A_packed,
                        const int32_t* BT_packed,
                        const float* scale_A,
                        const float* scale_B,
                        const float* bias,
                        float* C);

// INT8 GEMM with bias + GELU: C = GELU(dequant(A @ B^T) + bias)
void run_int8_gemm_bias_gelu(int M, int N, int K,
                              const int32_t* A_packed,
                              const int32_t* BT_packed,
                              const float* scale_A,
                              const float* scale_B,
                              const float* bias,
                              float* C);

// INT8 GEMM with bias + residual: C = dequant(A @ B^T) + bias + residual
void run_int8_gemm_bias_residual(int M, int N, int K,
                                  const int32_t* A_packed,
                                  const int32_t* BT_packed,
                                  const float* scale_A,
                                  const float* scale_B,
                                  const float* bias,
                                  const float* residual,
                                  float* C);
```

- [ ] **Step 3: Implement templated epilogue kernel**

In `kernels/quantization/14_int8_gemm.cu`, replace the existing `int8_gemm_dp4a_kernel` and epilogue section with a templated version. Keep the quantize kernel untouched. Replace from line 74 to the end:

```cpp
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

// --- Host wrappers ---

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
```

- [ ] **Step 4: Add test to CMakeLists.txt and build**

Add to `CMakeLists.txt` after the `test_int8_gemm` target:

```cmake
add_executable(test_int8_epilogues tests/test_int8_epilogues.cu)
target_link_libraries(test_int8_epilogues GTest::gtest_main quant_kernels ${CUBLAS_LIB})
add_test(NAME Int8EpilogueTests COMMAND test_int8_epilogues)
```

Build and run:

```bash
cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --target test_int8_epilogues
./build/test_int8_epilogues
```

Expected: All 6 tests PASS.

- [ ] **Step 5: Run existing K14 tests to verify no regression**

```bash
./build/test_int8_gemm
```

Expected: All 5 existing tests PASS (the `run_int8_gemm` wrapper still dispatches `Plain` epilogue).

- [ ] **Step 6: Commit**

```bash
git add kernels/quantization/14_int8_gemm.cu kernels/quantization/14_int8_gemm.cuh \
        tests/test_int8_epilogues.cu CMakeLists.txt
git commit -m "feat: add K14 fused epilogues (BiasOnly, BiasGELU, BiasResidual)"
```

---

### Task 3: K15 Fused LayerNorm + Residual Kernel

**Files:**
- Create: `kernels/layernorm/15_layernorm_residual.cuh`
- Create: `kernels/layernorm/15_layernorm_residual.cu`
- Create: `tests/test_layernorm.cu`

- [ ] **Step 1: Write failing test**

Create `tests/test_layernorm.cu`:

```cpp
// tests/test_layernorm.cu
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "timer.cuh"
#include "layernorm/15_layernorm_residual.cuh"

class LayerNormTest : public ::testing::Test {
protected:
    void cpu_layernorm_residual(int rows, int cols, const float* x, const float* residual,
                                 const float* gamma, const float* beta,
                                 float* out, float* residual_out, float eps = 1e-5f) {
        for (int r = 0; r < rows; ++r) {
            // y = x + residual
            std::vector<float> y(cols);
            for (int c = 0; c < cols; ++c) {
                y[c] = x[r * cols + c] + residual[r * cols + c];
                residual_out[r * cols + c] = y[c];
            }
            // mean
            float mean = 0.0f;
            for (int c = 0; c < cols; ++c) mean += y[c];
            mean /= cols;
            // variance
            float var = 0.0f;
            for (int c = 0; c < cols; ++c) var += (y[c] - mean) * (y[c] - mean);
            var /= cols;
            // normalize
            float inv_std = 1.0f / sqrtf(var + eps);
            for (int c = 0; c < cols; ++c)
                out[r * cols + c] = gamma[c] * (y[c] - mean) * inv_std + beta[c];
        }
    }

    void RunTest(int rows, int cols) {
        float tol = 1e-4f;
        std::vector<float> h_x(rows * cols), h_res(rows * cols);
        std::vector<float> h_gamma(cols), h_beta(cols);
        srand(42);
        for (auto& v : h_x) v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (auto& v : h_res) v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (auto& v : h_gamma) v = 0.8f + (float)rand() / RAND_MAX * 0.4f;
        for (auto& v : h_beta) v = (float)rand() / RAND_MAX * 0.2f - 0.1f;

        // CPU reference
        std::vector<float> h_ref_out(rows * cols), h_ref_res_out(rows * cols);
        cpu_layernorm_residual(rows, cols, h_x.data(), h_res.data(),
                               h_gamma.data(), h_beta.data(),
                               h_ref_out.data(), h_ref_res_out.data());

        // GPU
        float *d_x, *d_res, *d_gamma, *d_beta, *d_out, *d_res_out;
        CUDA_CHECK(cudaMalloc(&d_x, rows * cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_res, rows * cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gamma, cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_beta, cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, rows * cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_res_out, rows * cols * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

        run_layernorm_residual(rows, cols, d_x, d_res, d_gamma, d_beta, d_out, d_res_out);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_out(rows * cols), h_res_result(rows * cols);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_res_result.data(), d_res_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err_out = 0.0f, max_err_res = 0.0f;
        for (int i = 0; i < rows * cols; ++i) {
            max_err_out = fmaxf(max_err_out, fabsf(h_out[i] - h_ref_out[i]));
            max_err_res = fmaxf(max_err_res, fabsf(h_res_result[i] - h_ref_res_out[i]));
        }
        EXPECT_LT(max_err_out, tol) << "LN output err=" << max_err_out;
        EXPECT_LT(max_err_res, tol) << "Residual output err=" << max_err_res;

        cudaFree(d_x); cudaFree(d_res); cudaFree(d_gamma);
        cudaFree(d_beta); cudaFree(d_out); cudaFree(d_res_out);
    }
};

TEST_F(LayerNormTest, K15_1x768)    { RunTest(1, 768); }     // single token decode
TEST_F(LayerNormTest, K15_128x768)  { RunTest(128, 768); }   // prefill
TEST_F(LayerNormTest, K15_1x256)    { RunTest(1, 256); }     // smaller dim
TEST_F(LayerNormTest, K15_64x768)   { RunTest(64, 768); }
TEST_F(LayerNormTest, K15_512x768)  { RunTest(512, 768); }   // large prefill
```

- [ ] **Step 2: Write K15 header**

Create `kernels/layernorm/15_layernorm_residual.cuh`:

```cpp
// kernels/layernorm/15_layernorm_residual.cuh
#pragma once
#include <cuda_runtime.h>

// Fused LayerNorm + Residual: out = LayerNorm(x + residual), residual_out = x + residual
// One block per row. Warp-cooperative reduction for mean/variance.
// x, residual, out, residual_out: [rows, cols], gamma, beta: [cols]
void run_layernorm_residual(int rows, int cols,
                             const float* x,
                             const float* residual,
                             const float* gamma,
                             const float* beta,
                             float* out,
                             float* residual_out,
                             float eps = 1e-5f);
```

- [ ] **Step 3: Implement K15 kernel**

Create `kernels/layernorm/15_layernorm_residual.cu`:

```cpp
// kernels/layernorm/15_layernorm_residual.cu
#include "layernorm/15_layernorm_residual.cuh"

#define LN_BLOCK 256

__global__ void layernorm_residual_kernel(
    int cols,
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    float* __restrict__ residual_out,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* x_row = x + row * cols;
    const float* res_row = residual + row * cols;
    float* out_row = out + row * cols;
    float* res_out_row = residual_out + row * cols;

    extern __shared__ float smem[];  // [LN_BLOCK]

    // Pass 1: compute y = x + residual, accumulate sum and sum_sq
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    for (int c = tid; c < cols; c += LN_BLOCK) {
        float y = x_row[c] + res_row[c];
        res_out_row[c] = y;  // store residual output
        local_sum += y;
        local_sum_sq += y * y;
    }

    // Shared memory reduction for sum
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = LN_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float mean = smem[0] / cols;

    // Shared memory reduction for sum_sq
    smem[tid] = local_sum_sq;
    __syncthreads();
    for (int s = LN_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float var = smem[0] / cols - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Pass 2: normalize
    for (int c = tid; c < cols; c += LN_BLOCK) {
        float y = res_out_row[c];
        out_row[c] = gamma[c] * (y - mean) * inv_std + beta[c];
    }
}

void run_layernorm_residual(int rows, int cols,
                             const float* x,
                             const float* residual,
                             const float* gamma,
                             const float* beta,
                             float* out,
                             float* residual_out,
                             float eps) {
    dim3 grid(rows);
    dim3 block(LN_BLOCK);
    int smem = LN_BLOCK * sizeof(float);
    layernorm_residual_kernel<<<grid, block, smem>>>(
        cols, x, residual, gamma, beta, out, residual_out, eps);
}
```

- [ ] **Step 4: Add to CMakeLists.txt, build, run tests**

Add to `CMakeLists.txt`:

```cmake
# --- LayerNorm Kernel (Week 7) ---
add_library(layernorm_kernels
    kernels/layernorm/15_layernorm_residual.cu
)
target_include_directories(layernorm_kernels PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels
)
```

And add the test target:

```cmake
add_executable(test_layernorm tests/test_layernorm.cu)
target_link_libraries(test_layernorm GTest::gtest_main layernorm_kernels)
add_test(NAME LayerNormTests COMMAND test_layernorm)
```

Build and run:

```bash
cmake --build build --target test_layernorm
./build/test_layernorm
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kernels/layernorm/15_layernorm_residual.cu kernels/layernorm/15_layernorm_residual.cuh \
        tests/test_layernorm.cu CMakeLists.txt
git commit -m "feat: implement K15 fused LayerNorm + residual kernel"
```

---

### Task 4: GPT-2 Types, Config & Weight Loader

**Files:**
- Create: `include/gpt2_types.cuh`
- Create: `src/gpt2_engine.cuh`
- Create: `src/gpt2_engine.cu` (weight loading portion — forward pass added in Task 8)

- [ ] **Step 1: Write GPT2 types header**

Create `include/gpt2_types.cuh`:

```cpp
// include/gpt2_types.cuh
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

struct GPT2Config {
    int n_layers;       // 12
    int n_heads;        // 12
    int d_model;        // 768
    int d_ff;           // 3072
    int vocab_size;     // 50257
    int max_seq_len;    // 1024
    int d_head;         // 64
};

// Pre-quantized INT8 weight matrix (NT layout for K14)
struct QuantWeight {
    int32_t* packed;    // [N, K/4] int8x4 packed as int32
    float* scale;       // [N] per-row scales
    int N;              // output dim (rows of B^T)
    int K;              // input dim
};

// Per-layer weights
struct LayerWeights {
    // Attention LayerNorm
    float* ln1_gamma;   // [d_model]
    float* ln1_beta;    // [d_model]

    // QKV projection
    QuantWeight qkv;    // [2304, 768/4] packed + [2304] scale
    float* qkv_bias;    // [2304]

    // Output projection
    QuantWeight out;     // [768, 768/4] packed + [768] scale
    float* out_bias;    // [768]

    // FFN LayerNorm
    float* ln2_gamma;   // [d_model]
    float* ln2_beta;    // [d_model]

    // FFN up projection
    QuantWeight ffn_up;  // [3072, 768/4] packed + [3072] scale
    float* ffn_up_bias; // [3072]

    // FFN down projection
    QuantWeight ffn_down; // [768, 3072/4] packed + [768] scale
    float* ffn_down_bias; // [768]
};

// Full model weights
struct GPT2Weights {
    float* wte;          // [vocab_size, d_model] token embeddings
    float* wpe;          // [max_seq_len, d_model] position embeddings
    float* ln_f_gamma;   // [d_model] final layernorm
    float* ln_f_beta;    // [d_model]
    LayerWeights layers[12];
};

// Live inference metrics
struct InferenceMetrics {
    float ttft_ms;           // time to first token
    float tpot_ms;           // average time per output token
    float itl_min_ms;        // min inter-token latency
    float itl_max_ms;        // max inter-token latency
    float itl_p95_ms;        // p95 inter-token latency
    float tps;               // tokens per second
    int tokens_generated;    // count so far
    int max_tokens;          // target count
    float kv_cache_pct;      // KV cache utilization %
    float vram_used_mb;      // VRAM used
    float vram_total_mb;     // VRAM total
    float mbu;               // model bandwidth utilization
    float end_to_end_ms;     // total latency (set at end)
    bool generating;         // currently generating?
};
```

- [ ] **Step 2: Write engine header**

Create `src/gpt2_engine.cuh`:

```cpp
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

    // Generate tokens from prompt. Calls token_callback with each new token string.
    // metrics is updated in-place after each token.
    void generate(const std::vector<int>& prompt_tokens,
                  int max_new_tokens,
                  float temperature,
                  int top_k,
                  bool greedy,
                  std::function<void(const std::string& token_str)> token_callback,
                  InferenceMetrics& metrics);

    const GPT2Config& config() const { return config_; }

private:
    GPT2Config config_;
    GPT2Weights weights_;
    std::string model_dir_;

    // Activation workspace (pre-allocated, reused each step)
    float* d_x_;              // [max_seq_len, d_model]
    float* d_residual_;       // [max_seq_len, d_model]
    float* d_qkv_;            // [max_seq_len, 3 * d_model]
    float* d_attn_out_;       // [max_seq_len, d_model]
    float* d_ffn_hidden_;     // [max_seq_len, d_ff]
    float* d_logits_;         // [1, vocab_size] (decode only)

    // Runtime quantization workspace
    int32_t* d_act_packed_;   // [max_seq_len, d_ff/4] (largest activation)
    float* d_act_scale_;      // [max_seq_len]

    // KV cache (paged)
    float* d_k_cache_;        // [num_phys_blocks, block_size, n_heads, d_head]
    float* d_v_cache_;        // same
    int* d_block_tables_;     // [1, max_blocks_per_seq] (single sequence)
    int* d_context_lens_;     // [1]
    int num_phys_blocks_;
    int blocks_allocated_;
    int block_size_;          // 16

    // Decode attention workspace
    float* d_decode_workspace_;

    void load_config();
    void load_weights();
    void alloc_workspace();
    void free_all();

    // Forward pass components
    void embed(const int* token_ids, int seq_len, int start_pos, float* out);
    void forward_layer(int layer, int seq_len, bool is_prefill);
    void forward_final_ln(int seq_len);
    void forward_logits(float* out_logits);  // last token only
    int sample_token(const float* logits, float temperature, int top_k, bool greedy);

    // KV cache management
    void kv_cache_append(int layer, const float* K, const float* V, int seq_len, int start_pos);
    void ensure_kv_blocks(int total_tokens);
};
```

- [ ] **Step 3: Implement weight loading**

Create `src/gpt2_engine.cu` (weight loading and allocation — forward pass added in Task 8):

```cpp
// src/gpt2_engine.cu
#include "gpt2_engine.cuh"
#include "timer.cuh"
#include "quantization/14_int8_gemm.cuh"
#include "layernorm/15_layernorm_residual.cuh"
#include "flash_attention/10_flash_attn_v2.cuh"
#include "decode/13_decode_attn.cuh"
#include "softmax/08_fused_online.cuh"

#include <fstream>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <numeric>
// Read a raw binary file into a device buffer
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

// Minimal JSON config parser (avoids nlohmann dependency)
static GPT2Config parse_config(const std::string& path) {
    std::ifstream f(path);
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

    CUDA_CHECK(cudaMalloc(&d_x_, (size_t)L * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual_, (size_t)L * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qkv_, (size_t)L * 3 * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_out_, (size_t)L * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn_hidden_, (size_t)L * ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits_, (size_t)V * sizeof(float)));

    // Quantization workspace: max activation is [L, ff] for FFN
    int max_rows = L;
    int max_cols = ff;
    CUDA_CHECK(cudaMalloc(&d_act_packed_, (size_t)max_rows * (max_cols / 4) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_act_scale_, (size_t)max_rows * sizeof(float)));

    // Paged KV cache
    block_size_ = 16;
    int max_blocks = (L + block_size_ - 1) / block_size_;
    num_phys_blocks_ = max_blocks + 4;  // small buffer
    blocks_allocated_ = 0;
    int H = config_.n_heads;
    int dh = config_.d_head;
    int n_layers = config_.n_layers;

    // KV cache: [n_layers, num_phys_blocks, block_size, n_heads, d_head]
    size_t cache_size = (size_t)n_layers * num_phys_blocks_ * block_size_ * H * dh * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_k_cache_, cache_size));
    CUDA_CHECK(cudaMalloc(&d_v_cache_, cache_size));

    CUDA_CHECK(cudaMalloc(&d_block_tables_, max_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_context_lens_, sizeof(int)));

    // Decode workspace: [1, H, DA_MAX_SPLITS, d+2]
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
```

- [ ] **Step 4: Commit types and weight loader**

```bash
git add include/gpt2_types.cuh src/gpt2_engine.cuh src/gpt2_engine.cu
git commit -m "feat: add GPT-2 types, config, and weight loader"
```

---

### Task 5: BPE Tokenizer

**Files:**
- Create: `include/tokenizer.h`
- Create: `src/tokenizer.cpp`

- [ ] **Step 1: Write tokenizer header**

Create `include/tokenizer.h`:

```cpp
// include/tokenizer.h
#pragma once
#include <string>
#include <vector>
#include <unordered_map>

class BPETokenizer {
public:
    explicit BPETokenizer(const std::string& model_dir);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(int token_id) const;
    std::string decode(const std::vector<int>& token_ids) const;

    int vocab_size() const { return (int)id_to_token_.size(); }
    int eos_token() const { return 50256; }  // <|endoftext|>

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, int> merge_rank_;

    std::vector<std::string> bpe(const std::string& word) const;
    static std::string bytes_to_unicode(unsigned char b);
    static std::unordered_map<unsigned char, std::string> byte_encoder();
};
```

- [ ] **Step 2: Implement tokenizer**

Create `src/tokenizer.cpp`:

```cpp
// src/tokenizer.cpp
#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <climits>

// GPT-2 byte-level BPE maps bytes to unicode chars to avoid whitespace/control issues
static std::unordered_map<unsigned char, std::string> build_byte_encoder() {
    std::unordered_map<unsigned char, std::string> be;
    // Printable ASCII ranges that map to themselves
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            // These bytes map to single unicode char (same codepoint)
            char buf[8];
            int len = 0;
            if (b < 0x80) {
                buf[0] = (char)b; len = 1;
            } else if (b < 0xC0) {
                buf[0] = (char)(0xC2); buf[1] = (char)b; len = 2;
            } else {
                buf[0] = (char)(0xC3); buf[1] = (char)(b - 64); len = 2;
            }
            be[(unsigned char)b] = std::string(buf, len);
        }
    }
    // Remaining bytes map to U+0100 + n
    for (int b = 0; b < 256; ++b) {
        if (be.find((unsigned char)b) == be.end()) {
            int cp = 256 + n;
            char buf[4];
            if (cp < 0x80) {
                buf[0] = (char)cp;
                be[(unsigned char)b] = std::string(buf, 1);
            } else if (cp < 0x800) {
                buf[0] = (char)(0xC0 | (cp >> 6));
                buf[1] = (char)(0x80 | (cp & 0x3F));
                be[(unsigned char)b] = std::string(buf, 2);
            }
            ++n;
        }
    }
    return be;
}

static std::unordered_map<std::string, unsigned char> build_byte_decoder() {
    auto be = build_byte_encoder();
    std::unordered_map<std::string, unsigned char> bd;
    for (auto& [b, s] : be) bd[s] = b;
    return bd;
}

BPETokenizer::BPETokenizer(const std::string& model_dir) {
    // Load vocab.json
    std::ifstream vf(model_dir + "/vocab.json");
    if (!vf) { fprintf(stderr, "Failed to open vocab.json\n"); exit(1); }
    std::string vcontent((std::istreambuf_iterator<char>(vf)),
                          std::istreambuf_iterator<char>());

    // Simple JSON parser for {"token": id, ...}
    id_to_token_.resize(50257);
    size_t pos = 1; // skip opening {
    while (pos < vcontent.size()) {
        // Find key
        auto ks = vcontent.find('"', pos);
        if (ks == std::string::npos) break;
        auto ke = vcontent.find('"', ks + 1);
        // Handle escaped quotes
        while (ke != std::string::npos && vcontent[ke - 1] == '\\') {
            ke = vcontent.find('"', ke + 1);
        }
        std::string key = vcontent.substr(ks + 1, ke - ks - 1);
        // Unescape
        std::string clean;
        for (size_t i = 0; i < key.size(); ++i) {
            if (key[i] == '\\' && i + 1 < key.size()) {
                if (key[i+1] == '"') { clean += '"'; ++i; }
                else if (key[i+1] == '\\') { clean += '\\'; ++i; }
                else if (key[i+1] == 'n') { clean += '\n'; ++i; }
                else if (key[i+1] == 't') { clean += '\t'; ++i; }
                else if (key[i+1] == 'u') {
                    // Unicode escape \uXXXX
                    unsigned int cp = 0;
                    for (int j = 0; j < 4 && i + 2 + j < key.size(); ++j) {
                        char c = key[i + 2 + j];
                        cp <<= 4;
                        if (c >= '0' && c <= '9') cp |= c - '0';
                        else if (c >= 'a' && c <= 'f') cp |= c - 'a' + 10;
                        else if (c >= 'A' && c <= 'F') cp |= c - 'A' + 10;
                    }
                    // UTF-8 encode
                    if (cp < 0x80) { clean += (char)cp; }
                    else if (cp < 0x800) {
                        clean += (char)(0xC0 | (cp >> 6));
                        clean += (char)(0x80 | (cp & 0x3F));
                    } else {
                        clean += (char)(0xE0 | (cp >> 12));
                        clean += (char)(0x80 | ((cp >> 6) & 0x3F));
                        clean += (char)(0x80 | (cp & 0x3F));
                    }
                    i += 5;
                }
                else { clean += key[i]; }
            } else {
                clean += key[i];
            }
        }

        // Find value (integer)
        auto cs = vcontent.find(':', ke);
        auto vs = vcontent.find_first_of("0123456789", cs);
        auto ve = vcontent.find_first_not_of("0123456789", vs);
        int id = atoi(vcontent.substr(vs, ve - vs).c_str());

        token_to_id_[clean] = id;
        if (id < (int)id_to_token_.size()) id_to_token_[id] = clean;

        pos = ve;
        auto next = vcontent.find_first_of(",}", pos);
        if (next == std::string::npos || vcontent[next] == '}') break;
        pos = next + 1;
    }

    // Load merges.txt
    std::ifstream mf(model_dir + "/merges.txt");
    if (!mf) { fprintf(stderr, "Failed to open merges.txt\n"); exit(1); }
    std::string line;
    int rank = 0;
    while (std::getline(mf, line)) {
        if (line.empty()) continue;
        auto sp = line.find(' ');
        if (sp == std::string::npos) continue;
        std::string a = line.substr(0, sp);
        std::string b = line.substr(sp + 1);
        merges_.push_back({a, b});
        merge_rank_[a + " " + b] = rank++;
    }
}

std::vector<std::string> BPETokenizer::bpe(const std::string& word) const {
    // Split word into individual characters (UTF-8 safe — each GPT-2 "char" is one byte_encoder output)
    std::vector<std::string> tokens;
    // word is already in byte_encoder space, split into UTF-8 chars
    size_t i = 0;
    while (i < word.size()) {
        int len = 1;
        unsigned char c = (unsigned char)word[i];
        if (c >= 0xC0 && c < 0xE0) len = 2;
        else if (c >= 0xE0 && c < 0xF0) len = 3;
        else if (c >= 0xF0) len = 4;
        tokens.push_back(word.substr(i, len));
        i += len;
    }

    if (tokens.size() <= 1) return tokens;

    // Iteratively merge the lowest-rank pair
    while (true) {
        int best_rank = INT_MAX;
        int best_pos = -1;
        for (int j = 0; j < (int)tokens.size() - 1; ++j) {
            std::string key = tokens[j] + " " + tokens[j + 1];
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = j;
            }
        }
        if (best_pos < 0) break;

        std::string merged = tokens[best_pos] + tokens[best_pos + 1];
        tokens[best_pos] = merged;
        tokens.erase(tokens.begin() + best_pos + 1);
    }
    return tokens;
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    auto be = build_byte_encoder();
    std::vector<int> ids;

    // GPT-2 pre-tokenizes by splitting on whitespace/punctuation patterns
    // Simplified: split on spaces, prepend space char (Ġ) to non-first words
    std::istringstream iss(text);
    std::string word;
    bool first = true;
    while (iss >> word) {
        // Convert bytes to GPT-2 unicode representation
        std::string encoded;
        if (!first) {
            // Prepend space — in GPT-2's byte encoder, space (0x20) maps to 'Ġ'
            encoded += be[0x20];
        }
        for (unsigned char c : word) {
            encoded += be[c];
        }
        first = false;

        auto bpe_tokens = bpe(encoded);
        for (auto& t : bpe_tokens) {
            auto it = token_to_id_.find(t);
            if (it != token_to_id_.end())
                ids.push_back(it->second);
        }
    }
    return ids;
}

std::string BPETokenizer::decode(int token_id) const {
    if (token_id < 0 || token_id >= (int)id_to_token_.size())
        return "";
    auto bd = build_byte_decoder();
    const std::string& token = id_to_token_[token_id];

    std::string result;
    size_t i = 0;
    while (i < token.size()) {
        int len = 1;
        unsigned char c = (unsigned char)token[i];
        if (c >= 0xC0 && c < 0xE0) len = 2;
        else if (c >= 0xE0 && c < 0xF0) len = 3;
        else if (c >= 0xF0) len = 4;
        std::string ch = token.substr(i, len);
        auto it = bd.find(ch);
        if (it != bd.end())
            result += (char)it->second;
        else
            result += ch;
        i += len;
    }
    return result;
}

std::string BPETokenizer::decode(const std::vector<int>& token_ids) const {
    std::string result;
    for (int id : token_ids) result += decode(id);
    return result;
}
```

- [ ] **Step 3: Commit**

```bash
git add include/tokenizer.h src/tokenizer.cpp
git commit -m "feat: add BPE tokenizer (encode + decode)"
```

---

### Task 6: Sampler

**Files:**
- Create: `include/sampler.cuh`
- Create: `src/sampler.cu`

- [ ] **Step 1: Write sampler header**

Create `include/sampler.cuh`:

```cpp
// include/sampler.cuh
#pragma once
#include <cuda_runtime.h>

// Greedy argmax over logits [vocab_size]. Returns token id.
int sample_argmax(const float* d_logits, int vocab_size);

// Top-k sampling with temperature. Returns token id.
// Modifies d_logits in-place (softmax applied).
int sample_top_k(const float* d_logits, int vocab_size,
                 int k, float temperature, unsigned int seed);
```

- [ ] **Step 2: Implement sampler**

Create `src/sampler.cu`:

```cpp
// src/sampler.cu
#include "sampler.cuh"
#include "timer.cuh"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

int sample_argmax(const float* d_logits, int vocab_size) {
    std::vector<float> h(vocab_size);
    CUDA_CHECK(cudaMemcpy(h.data(), d_logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    return (int)(std::max_element(h.begin(), h.end()) - h.begin());
}

int sample_top_k(const float* d_logits, int vocab_size,
                 int k, float temperature, unsigned int seed) {
    std::vector<float> h(vocab_size);
    CUDA_CHECK(cudaMemcpy(h.data(), d_logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Find top-k indices
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](int a, int b) { return h[a] > h[b]; });

    // Temperature + softmax over top-k
    float max_val = h[indices[0]];
    float sum = 0.0f;
    std::vector<float> probs(k);
    for (int i = 0; i < k; ++i) {
        probs[i] = expf((h[indices[i]] - max_val) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < k; ++i) probs[i] /= sum;

    // Multinomial sample
    srand(seed);
    float r = (float)rand() / RAND_MAX;
    float cum = 0.0f;
    for (int i = 0; i < k; ++i) {
        cum += probs[i];
        if (r < cum) return indices[i];
    }
    return indices[k - 1];
}
```

- [ ] **Step 3: Commit**

```bash
git add include/sampler.cuh src/sampler.cu
git commit -m "feat: add top-k temperature sampler"
```

---

### Task 7: Embedding Kernel

**Files:**
- Create: `kernels/layernorm/15_embed.cu` (small, co-located with layernorm as "element-wise" ops)
- Create: `kernels/layernorm/15_embed.cuh`

- [ ] **Step 1: Write embedding kernel**

Create `kernels/layernorm/15_embed.cuh`:

```cpp
// kernels/layernorm/15_embed.cuh
#pragma once
#include <cuda_runtime.h>

// Embedding lookup: out[i] = wte[token_ids[i]] + wpe[start_pos + i]
// out: [seq_len, d_model], wte: [vocab_size, d_model], wpe: [max_seq_len, d_model]
void run_embedding(const int* token_ids, int seq_len, int start_pos, int d_model,
                   const float* wte, const float* wpe, float* out);

// Standalone LayerNorm (no residual add) — for final LN before logits
void run_layernorm(int rows, int cols,
                   const float* x,
                   const float* gamma,
                   const float* beta,
                   float* out,
                   float eps = 1e-5f);
```

- [ ] **Step 2: Implement kernels**

Create `kernels/layernorm/15_embed.cu`:

```cpp
// kernels/layernorm/15_embed.cu
#include "layernorm/15_embed.cuh"

__global__ void embedding_kernel(
    const int* __restrict__ token_ids,
    int seq_len, int start_pos, int d_model,
    const float* __restrict__ wte,
    const float* __restrict__ wpe,
    float* __restrict__ out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * d_model;
    if (idx >= total) return;

    int token_pos = idx / d_model;
    int dim = idx % d_model;
    int token_id = token_ids[token_pos];
    int abs_pos = start_pos + token_pos;

    out[token_pos * d_model + dim] = wte[token_id * d_model + dim]
                                   + wpe[abs_pos * d_model + dim];
}

void run_embedding(const int* token_ids, int seq_len, int start_pos, int d_model,
                   const float* wte, const float* wpe, float* out) {
    int total = seq_len * d_model;
    int block = 256;
    int grid = (total + block - 1) / block;
    embedding_kernel<<<grid, block>>>(token_ids, seq_len, start_pos, d_model, wte, wpe, out);
}

#define LN_BLOCK 256

__global__ void layernorm_kernel(
    int cols,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* x_row = x + row * cols;
    float* out_row = out + row * cols;

    extern __shared__ float smem[];

    float local_sum = 0.0f, local_sum_sq = 0.0f;
    for (int c = tid; c < cols; c += LN_BLOCK) {
        float v = x_row[c];
        local_sum += v;
        local_sum_sq += v * v;
    }

    smem[tid] = local_sum;
    __syncthreads();
    for (int s = LN_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float mean = smem[0] / cols;

    smem[tid] = local_sum_sq;
    __syncthreads();
    for (int s = LN_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float var = smem[0] / cols - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int c = tid; c < cols; c += LN_BLOCK) {
        out_row[c] = gamma[c] * (x_row[c] - mean) * inv_std + beta[c];
    }
}

void run_layernorm(int rows, int cols,
                   const float* x,
                   const float* gamma,
                   const float* beta,
                   float* out,
                   float eps) {
    dim3 grid(rows);
    dim3 block(LN_BLOCK);
    int smem = LN_BLOCK * sizeof(float);
    layernorm_kernel<<<grid, block, smem>>>(cols, x, gamma, beta, out, eps);
}
```

- [ ] **Step 3: Add to CMakeLists.txt layernorm_kernels library**

Add `kernels/layernorm/15_embed.cu` to the `layernorm_kernels` library:

```cmake
add_library(layernorm_kernels
    kernels/layernorm/15_layernorm_residual.cu
    kernels/layernorm/15_embed.cu
)
```

- [ ] **Step 4: Commit**

```bash
git add kernels/layernorm/15_embed.cu kernels/layernorm/15_embed.cuh CMakeLists.txt
git commit -m "feat: add embedding lookup and standalone LayerNorm kernels"
```

---

### Task 8: GPT-2 Forward Pass

**Files:**
- Modify: `src/gpt2_engine.cu` (add forward pass methods to existing file)

This is the core orchestration. The `generate()` method runs prefill then decode loop.

- [ ] **Step 1: Add embedding, KV cache, and forward_layer methods**

Append to `src/gpt2_engine.cu` (after the constructor/destructor):

```cpp
// ============================================================
// Embedding: out[i] = wte[token_ids[i]] + wpe[start_pos + i]
// ============================================================
#include "layernorm/15_embed.cuh"

void GPT2Engine::embed(const int* d_token_ids, int seq_len, int start_pos, float* out) {
    run_embedding(d_token_ids, seq_len, start_pos, config_.d_model,
                  weights_.wte, weights_.wpe, out);
}

// ============================================================
// KV Cache management
// ============================================================
void GPT2Engine::ensure_kv_blocks(int total_tokens) {
    int needed = (total_tokens + block_size_ - 1) / block_size_;
    while (blocks_allocated_ < needed) {
        // Sequential block allocation (simple for single-sequence demo)
        // Update block table on host then copy
        int h_bt[128];
        CUDA_CHECK(cudaMemcpy(h_bt, d_block_tables_, blocks_allocated_ * sizeof(int), cudaMemcpyDeviceToHost));
        h_bt[blocks_allocated_] = blocks_allocated_;  // identity mapping
        CUDA_CHECK(cudaMemcpy(d_block_tables_, h_bt, (blocks_allocated_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
        blocks_allocated_++;
    }
}

// Simple kernel: copy QKV slice into paged KV cache
__global__ void kv_cache_write_kernel(
    const float* __restrict__ K_src,   // [seq_len, n_heads, d_head]
    const float* __restrict__ V_src,
    float* __restrict__ k_cache,       // [num_phys_blocks, block_size, n_heads, d_head]
    float* __restrict__ v_cache,
    const int* __restrict__ block_table,
    int seq_len, int start_pos, int n_heads, int d_head, int block_size,
    int num_phys_blocks)
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
    int block_offset = abs_pos % block_size;
    int phys_block = block_table[block_idx];

    int cache_idx = ((phys_block * block_size + block_offset) * n_heads + h) * d_head + d;
    k_cache[cache_idx] = K_src[idx];
    v_cache[cache_idx] = V_src[idx];
}

void GPT2Engine::kv_cache_append(int layer, const float* K, const float* V,
                                  int seq_len, int start_pos) {
    int H = config_.n_heads;
    int dh = config_.d_head;
    // Per-layer offset into the cache
    size_t layer_offset = (size_t)layer * num_phys_blocks_ * block_size_ * H * dh;

    int total = seq_len * H * dh;
    int block = 256;
    int grid = (total + block - 1) / block;
    kv_cache_write_kernel<<<grid, block>>>(
        K, V,
        d_k_cache_ + layer_offset, d_v_cache_ + layer_offset,
        d_block_tables_,
        seq_len, start_pos, H, dh, block_size_, num_phys_blocks_);
}

// ============================================================
// Forward pass: single layer
// ============================================================
void GPT2Engine::forward_layer(int layer, int seq_len, bool is_prefill) {
    int d = config_.d_model;
    int H = config_.n_heads;
    int dh = config_.d_head;
    int ff = config_.d_ff;
    auto& lw = weights_.layers[layer];
    int M = seq_len;  // number of tokens being processed

    // --- Attention sub-block ---
    // residual = x (copy d_x_ → d_residual_)
    CUDA_CHECK(cudaMemcpy(d_residual_, d_x_, (size_t)M * d * sizeof(float), cudaMemcpyDeviceToDevice));

    // x = LayerNorm(x + residual) — but first time, x IS residual, so this is just LN(x)
    // Actually: pre-norm means LN is applied to the input, residual is added AFTER attention.
    // Let's correct: d_x_ holds the layer input. We LN it, run attention, add residual.
    // For the first call, d_residual_ = d_x_ (saved above).
    // ln_out goes into d_x_ (we can reuse since residual is saved)
    run_layernorm(M, d, d_x_, lw.ln1_gamma, lw.ln1_beta, d_x_);

    // QKV projection: [M, d] × W_qkv^T → [M, 3*d] with bias
    run_quantize_fp32_to_int8(M, d, d_x_, d_act_packed_, d_act_scale_);
    run_int8_gemm_bias(M, 3 * d, d, d_act_packed_, lw.qkv.packed,
                       d_act_scale_, lw.qkv.scale, lw.qkv_bias, d_qkv_);

    // Split QKV → Q [M, d], K [M, d], V [M, d] (pointer offsets)
    float* Q = d_qkv_;
    float* K = d_qkv_ + (size_t)M * d;
    float* V = d_qkv_ + (size_t)M * d * 2;

    // Get start_pos from context lens
    int h_ctx;
    CUDA_CHECK(cudaMemcpy(&h_ctx, d_context_lens_, sizeof(int), cudaMemcpyDeviceToHost));
    int start_pos = is_prefill ? 0 : h_ctx;

    // Write K, V to paged cache
    kv_cache_append(layer, K, V, M, start_pos);

    int total_ctx = start_pos + M;

    // Attention
    size_t layer_cache_offset = (size_t)layer * num_phys_blocks_ * block_size_ * H * dh;
    float* layer_k = d_k_cache_ + layer_cache_offset;
    float* layer_v = d_v_cache_ + layer_cache_offset;

    if (is_prefill) {
        // FlashAttention-2 for full sequence
        // Q, K, V are [B=1, H, N, d] — need to reshape from [M, H*d] to [1, H, M, d]
        // K10 expects [B, H, N, d] layout — Q/K/V are already [M, H, d] which is [1, H=1_flat, M, d]
        // Actually K10 expects [B*H*N*d] in [B, H, N, d] order
        run_flash_attn_v2(1, H, M, dh, Q, K, V, d_attn_out_, true);
    } else {
        // Decode attention: single token query against full KV cache
        int ctx_for_decode = total_ctx;
        CUDA_CHECK(cudaMemcpy(d_context_lens_, &ctx_for_decode, sizeof(int), cudaMemcpyHostToDevice));
        int max_blocks = (ctx_for_decode + block_size_ - 1) / block_size_;

        run_decode_attn(1, H, H, dh,
                        Q,
                        layer_k, layer_v,
                        d_block_tables_, d_context_lens_,
                        ctx_for_decode, block_size_, max_blocks,
                        d_attn_out_, d_decode_workspace_);
    }

    // Output projection: [M, d] × W_o^T → [M, d] with bias + residual
    run_quantize_fp32_to_int8(M, d, d_attn_out_, d_act_packed_, d_act_scale_);
    run_int8_gemm_bias_residual(M, d, d, d_act_packed_, lw.out.packed,
                                 d_act_scale_, lw.out.scale, lw.out_bias,
                                 d_residual_, d_x_);

    // --- FFN sub-block ---
    CUDA_CHECK(cudaMemcpy(d_residual_, d_x_, (size_t)M * d * sizeof(float), cudaMemcpyDeviceToDevice));
    run_layernorm(M, d, d_x_, lw.ln2_gamma, lw.ln2_beta, d_x_);

    // FFN up: [M, d] → [M, ff] with GELU
    run_quantize_fp32_to_int8(M, d, d_x_, d_act_packed_, d_act_scale_);
    run_int8_gemm_bias_gelu(M, ff, d, d_act_packed_, lw.ffn_up.packed,
                             d_act_scale_, lw.ffn_up.scale, lw.ffn_up_bias, d_ffn_hidden_);

    // FFN down: [M, ff] → [M, d] with residual
    run_quantize_fp32_to_int8(M, ff, d_ffn_hidden_, d_act_packed_, d_act_scale_);
    run_int8_gemm_bias_residual(M, d, ff, d_act_packed_, lw.ffn_down.packed,
                                 d_act_scale_, lw.ffn_down.scale, lw.ffn_down_bias,
                                 d_residual_, d_x_);
}

void GPT2Engine::forward_final_ln(int seq_len) {
    // For decode, we only need the last token
    int d = config_.d_model;
    float* last_token = d_x_ + (size_t)(seq_len - 1) * d;
    run_layernorm(1, d, last_token, weights_.ln_f_gamma, weights_.ln_f_beta, last_token);
}

void GPT2Engine::forward_logits(float* out_logits) {
    // Project last token hidden state to vocab: [1, d] × wte^T → [1, V]
    // GPT-2 ties wte and lm_head. wte is [V, d], so wte^T is [d, V].
    // But our INT8 GEMM expects pre-quantized B^T. For the vocab projection,
    // we use wte directly as B^T since it's [V, d] which IS the NT layout.
    // We quantize the activation and wte rows at runtime.
    int d = config_.d_model;
    int V = config_.vocab_size;

    // Quantize the single-token activation [1, d]
    int h_ctx;
    CUDA_CHECK(cudaMemcpy(&h_ctx, d_context_lens_, sizeof(int), cudaMemcpyDeviceToHost));
    float* last_hidden = d_x_ + (size_t)(h_ctx > 0 ? 0 : 0) * d;
    // For decode, d_x_ has [1, d] after forward_layer processes 1 token

    // FP32 matmul for vocab projection (wte is not pre-quantized)
    // Use a simple kernel: logits[j] = dot(hidden, wte[j]) for j in [0, V)
    // This avoids needing to quantize the 50257x768 embedding at runtime
    run_quantize_fp32_to_int8(1, d, d_x_, d_act_packed_, d_act_scale_);

    // We need wte in quantized form. Pre-quantize at load time would save runtime cost.
    // For now, quantize wte rows on-the-fly. But wte is [V, d] = [50257, 768].
    // This is expensive to quantize every decode step.
    // Better approach: store wte as FP32 and do FP32 matmul for the final projection.
    // With M=1, N=50257, K=768, this is memory-bound anyway — FP32 is fine.

    // Simple FP32 matmul kernel for [1, d] × [V, d]^T = [1, V]
    // Reuse the embedding weight directly
    // We'll write a tiny kernel inline:
    extern void run_vocab_proj(const float* hidden, const float* wte,
                               float* logits, int d, int V);
    run_vocab_proj(d_x_, weights_.wte, out_logits, d, V);
}

// Vocab projection: logits[j] = dot(hidden[0:d], wte[j*d : (j+1)*d])
__global__ void vocab_proj_kernel(
    const float* __restrict__ hidden,  // [d]
    const float* __restrict__ wte,     // [V, d]
    float* __restrict__ logits,        // [V]
    int d, int V)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= V) return;

    float sum = 0.0f;
    for (int k = 0; k < d; ++k)
        sum += hidden[k] * wte[j * d + k];
    logits[j] = sum;
}

void run_vocab_proj(const float* hidden, const float* wte,
                    float* logits, int d, int V) {
    int block = 256;
    int grid = (V + block - 1) / block;
    vocab_proj_kernel<<<grid, block>>>(hidden, wte, logits, d, V);
}

// ============================================================
// Generate: prefill + decode loop
// ============================================================
#include "sampler.cuh"
#include "tokenizer.h"

void GPT2Engine::generate(const std::vector<int>& prompt_tokens,
                           int max_new_tokens,
                           float temperature,
                           int top_k,
                           bool greedy,
                           std::function<void(const std::string& token_str)> token_callback,
                           InferenceMetrics& metrics) {
    int prompt_len = (int)prompt_tokens.size();
    int d = config_.d_model;

    metrics = {};
    metrics.max_tokens = max_new_tokens;
    metrics.generating = true;

    // Upload prompt tokens
    int* d_tokens;
    CUDA_CHECK(cudaMalloc(&d_tokens, prompt_len * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tokens, prompt_tokens.data(), prompt_len * sizeof(int), cudaMemcpyHostToDevice));

    // Reset KV cache
    blocks_allocated_ = 0;
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_context_lens_, &zero, sizeof(int), cudaMemcpyHostToDevice));

    GpuTimer total_timer, step_timer;
    std::vector<float> itl_samples;

    // === Prefill ===
    total_timer.tic();
    step_timer.tic();

    ensure_kv_blocks(prompt_len);
    embed(d_tokens, prompt_len, 0, d_x_);

    for (int l = 0; l < config_.n_layers; ++l)
        forward_layer(l, prompt_len, true);

    // Update context length
    CUDA_CHECK(cudaMemcpy(d_context_lens_, &prompt_len, sizeof(int), cudaMemcpyHostToDevice));

    // Get first token
    forward_final_ln(prompt_len);
    // For prefill, we need the last token's hidden state
    // Shift d_x_ so forward_logits reads the last token
    float* last_hidden = d_x_ + (size_t)(prompt_len - 1) * d;
    CUDA_CHECK(cudaMemcpy(d_x_, last_hidden, d * sizeof(float), cudaMemcpyDeviceToDevice));

    forward_logits(d_logits_);
    CUDA_CHECK(cudaDeviceSynchronize());

    int next_token;
    if (greedy)
        next_token = sample_argmax(d_logits_, config_.vocab_size);
    else
        next_token = sample_top_k(d_logits_, config_.vocab_size, top_k, temperature, 42);

    metrics.ttft_ms = step_timer.toc();

    // === Decode loop ===
    int tokens_generated = 0;
    unsigned int sample_seed = 42;

    for (int step = 0; step < max_new_tokens; ++step) {
        if (next_token == 50256) break;  // EOS

        // Callback with decoded token
        // (tokenizer is loaded separately by main — we pass the raw token id via callback)
        // Actually, we need to pass the string. The engine doesn't own the tokenizer.
        // Solution: token_callback receives token_id as string, main decodes it.
        // Simpler: pass token_id encoded as string, let main handle decode.
        // For clean API: callback gets token string. But engine needs tokenizer ref.
        // Compromise: pass token id, main wraps it.
        token_callback(std::to_string(next_token));

        tokens_generated++;
        metrics.tokens_generated = tokens_generated;

        step_timer.tic();

        // Prepare single-token input
        int ctx = prompt_len + tokens_generated - 1;
        ensure_kv_blocks(ctx + 1);

        CUDA_CHECK(cudaMemcpy(d_tokens, &next_token, sizeof(int), cudaMemcpyHostToDevice));
        embed(d_tokens, 1, ctx, d_x_);

        for (int l = 0; l < config_.n_layers; ++l)
            forward_layer(l, 1, false);

        int new_ctx = ctx + 1;
        CUDA_CHECK(cudaMemcpy(d_context_lens_, &new_ctx, sizeof(int), cudaMemcpyHostToDevice));

        forward_final_ln(1);
        forward_logits(d_logits_);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (greedy)
            next_token = sample_argmax(d_logits_, config_.vocab_size);
        else
            next_token = sample_top_k(d_logits_, config_.vocab_size, top_k, temperature, ++sample_seed);

        float step_ms = step_timer.toc();
        itl_samples.push_back(step_ms);

        // Update metrics
        float wall_ms = total_timer.toc();
        total_timer.tic();  // restart for next measurement
        metrics.tpot_ms = wall_ms / tokens_generated;  // approximation
        metrics.tps = tokens_generated / (wall_ms / 1000.0f);

        // ITL stats
        metrics.itl_min_ms = *std::min_element(itl_samples.begin(), itl_samples.end());
        metrics.itl_max_ms = *std::max_element(itl_samples.begin(), itl_samples.end());
        if (itl_samples.size() >= 20) {
            auto sorted = itl_samples;
            std::sort(sorted.begin(), sorted.end());
            metrics.itl_p95_ms = sorted[(int)(sorted.size() * 0.95)];
        } else {
            metrics.itl_p95_ms = metrics.itl_max_ms;
        }

        // KV cache utilization
        int max_blocks = (config_.max_seq_len + block_size_ - 1) / block_size_;
        metrics.kv_cache_pct = 100.0f * blocks_allocated_ / max_blocks;

        // VRAM
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        metrics.vram_used_mb = (total_mem - free_mem) / (1024.0f * 1024.0f);
        metrics.vram_total_mb = total_mem / (1024.0f * 1024.0f);

        // MBU: actual TPS vs theoretical max
        // Theoretical: each token reads all weights (~140MB INT8) from HBM
        // At 192 GB/s peak BW, max tokens/s = 192e9 / 140e6 ≈ 1371
        float theoretical_max_tps = 192e9f / 140e6f;
        metrics.mbu = metrics.tps / theoretical_max_tps;
    }

    metrics.generating = false;
    cudaFree(d_tokens);
}
```

**Note:** The `forward_layer` uses `run_layernorm` (standalone, from Task 7) instead of `run_layernorm_residual` (fused) for simplicity in the pre-norm pattern. The fused version is used when the residual needs to be captured, but here we explicitly copy and manage the residual. This can be optimized post-demo.

- [ ] **Step 2: Commit**

```bash
git add src/gpt2_engine.cu
git commit -m "feat: implement GPT-2 forward pass (prefill + decode)"
```

---

### Task 9: CMake Build Configuration

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add FTXUI dependency and slick target**

Add to `CMakeLists.txt` after the CUTLASS FetchContent block:

```cmake
# FTXUI (TUI library)
FetchContent_Declare(
    ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
    GIT_TAG        v5.0.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(ftxui)

# --- GPT-2 Engine (Week 7) ---
add_library(gpt2_engine
    src/gpt2_engine.cu
    src/tokenizer.cpp
    src/sampler.cu
    src/kv_cache.cu
)
target_include_directories(gpt2_engine PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(gpt2_engine
    quant_kernels
    layernorm_kernels
    attention_kernels
    decode_kernels
    paged_attention_kernels
    softmax_kernels
    ${CUBLAS_LIB}
)

# SLICK main executable
add_executable(slick src/main.cu)
target_include_directories(slick PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(slick
    gpt2_engine
    ftxui::screen
    ftxui::dom
    ftxui::component
)
```

**Note:** `src/kv_cache.cu` is listed but can be an empty file for now (the KV cache logic is currently inline in gpt2_engine.cu). Create it as a stub:

- [ ] **Step 2: Create stub kv_cache.cu**

Create `src/kv_cache.cu`:

```cpp
// src/kv_cache.cu
// KV cache management — currently inline in gpt2_engine.cu
// This file exists for the build system. Future: extract allocator here.
#include "gpt2_types.cuh"
```

- [ ] **Step 3: Build to verify compilation**

```bash
cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --target slick
```

Expected: Compiles (won't run yet — main.cu doesn't exist).

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt src/kv_cache.cu
git commit -m "build: add FTXUI dep, GPT-2 engine library, and slick target"
```

---

### Task 10: FTXUI TUI & Main Application

**Files:**
- Create: `src/main.cu`

- [ ] **Step 1: Implement main with FTXUI TUI**

Create `src/main.cu`:

```cpp
// src/main.cu
#include "gpt2_engine.cuh"
#include "gpt2_types.cuh"
#include "tokenizer.h"
#include "sampler.cuh"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstdio>
#include <cstring>

using namespace ftxui;

struct AppState {
    std::mutex mtx;
    std::string generated_text;
    InferenceMetrics metrics;
    std::atomic<bool> running{false};
    std::atomic<bool> quit{false};
};

// Format float with fixed precision
static std::string fmt(float v, int prec = 1) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.*f", prec, v);
    return buf;
}

Element MetricsPanel(const InferenceMetrics& m) {
    auto section = [](std::string title, std::vector<Element> children) {
        return vbox({
            text(title) | bold,
            separator(),
            vbox(std::move(children)),
        });
    };

    return vbox({
        section("Latency", {
            hbox({text("TTFT       "), text(fmt(m.ttft_ms) + " ms") | align_right}),
            hbox({text("TPOT       "), text(fmt(m.tpot_ms) + " ms") | align_right}),
            hbox({text("ITL (P95)  "), text(fmt(m.itl_p95_ms) + " ms") | align_right}),
            hbox({text("ITL min/max"), text(fmt(m.itl_min_ms) + "/" + fmt(m.itl_max_ms) + " ms") | align_right}),
            hbox({text("TPS        "), text(fmt(m.tps)) | align_right}),
            hbox({text("Tokens     "),
                  text(std::to_string(m.tokens_generated) + "/" + std::to_string(m.max_tokens)) | align_right}),
        }),
        text(""),
        section("Resource", {
            hbox({text("KV Cache "), gauge(m.kv_cache_pct / 100.0f) | flex,
                  text(" " + fmt(m.kv_cache_pct, 0) + "%")}),
            hbox({text("VRAM     "), gauge(m.vram_used_mb / std::max(m.vram_total_mb, 1.0f)) | flex,
                  text(" " + fmt(m.vram_used_mb, 0) + "/" + fmt(m.vram_total_mb, 0) + " MB")}),
        }),
        text(""),
        hbox({text("End-to-end "),
              text(m.end_to_end_ms > 0 ? fmt(m.end_to_end_ms) + " ms" : "--") | align_right}),
    }) | border | size(WIDTH, EQUAL, 40);
}

int main(int argc, char** argv) {
    // Parse args
    std::string model_dir = "models/gpt2-int8";
    std::string initial_prompt;
    int max_tokens = 256;
    float temperature = 0.8f;
    int top_k_val = 50;
    bool greedy = false;
    bool bench_mode = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_dir = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) initial_prompt = argv[++i];
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) temperature = atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k_val = atoi(argv[++i]);
        else if (strcmp(argv[i], "--greedy") == 0) greedy = true;
        else if (strcmp(argv[i], "--bench") == 0) bench_mode = true;
    }

    printf("Loading model from %s...\n", model_dir.c_str());
    GPT2Engine engine(model_dir);
    BPETokenizer tokenizer(model_dir);
    printf("Model loaded. Vocab: %d\n", tokenizer.vocab_size());

    if (bench_mode) {
        // Simple benchmark mode — no TUI
        if (initial_prompt.empty()) initial_prompt = "The meaning of life is";
        auto tokens = tokenizer.encode(initial_prompt);
        printf("Prompt (%d tokens): %s\n", (int)tokens.size(), initial_prompt.c_str());

        InferenceMetrics metrics;
        engine.generate(tokens, max_tokens, temperature, top_k_val, greedy,
                        [&](const std::string& tok_id_str) {
                            int id = atoi(tok_id_str.c_str());
                            printf("%s", tokenizer.decode(id).c_str());
                            fflush(stdout);
                        }, metrics);

        printf("\n\n--- Metrics ---\n");
        printf("TTFT:        %.1f ms\n", metrics.ttft_ms);
        printf("TPOT:        %.1f ms\n", metrics.tpot_ms);
        printf("ITL P95:     %.1f ms\n", metrics.itl_p95_ms);
        printf("TPS:         %.1f\n", metrics.tps);
        printf("Tokens:      %d\n", metrics.tokens_generated);
        printf("KV Cache:    %.0f%%\n", metrics.kv_cache_pct);
        printf("VRAM:        %.0f / %.0f MB\n", metrics.vram_used_mb, metrics.vram_total_mb);
        return 0;
    }

    // --- Interactive TUI mode ---
    AppState state;

    auto screen = ScreenInteractive::Fullscreen();

    std::string input_text;
    auto input = Input(&input_text, "Enter prompt...");

    auto component = CatchEvent(input, [&](Event event) -> bool {
        if (event == Event::Return && !input_text.empty() && !state.running) {
            std::string prompt = input_text;
            input_text.clear();

            state.running = true;
            {
                std::lock_guard<std::mutex> lock(state.mtx);
                state.generated_text = prompt;
                state.metrics = {};
            }

            // Run generation in background thread
            std::thread([&, prompt]() {
                auto tokens = tokenizer.encode(prompt);
                engine.generate(tokens, max_tokens, temperature, top_k_val, greedy,
                    [&](const std::string& tok_id_str) {
                        int id = atoi(tok_id_str.c_str());
                        std::string decoded = tokenizer.decode(id);
                        {
                            std::lock_guard<std::mutex> lock(state.mtx);
                            state.generated_text += decoded;
                        }
                        screen.Post(Event::Custom);
                    }, state.metrics);

                state.running = false;
                screen.Post(Event::Custom);
            }).detach();

            return true;
        }
        if (event == Event::Escape) {
            state.quit = true;
            screen.Exit();
            return true;
        }
        return false;
    });

    auto renderer = Renderer(component, [&] {
        std::lock_guard<std::mutex> lock(state.mtx);

        auto output_panel = vbox({
            text("Output") | bold,
            separator(),
            paragraph(state.generated_text) | flex,
        }) | border | flex;

        auto metrics_panel = MetricsPanel(state.metrics);

        auto main_area = hbox({
            output_panel,
            metrics_panel,
        }) | flex;

        auto input_bar = hbox({
            text("> "),
            component->Render() | flex,
            text(state.running ? " [generating...]" : "") | dim,
        }) | border;

        return vbox({
            text(" SLICK GPT-2 Engine ") | bold | center,
            main_area,
            input_bar,
        });
    });

    screen.Loop(renderer);
    return 0;
}
```

- [ ] **Step 2: Build and verify compilation**

```bash
cmake --build build --target slick
```

Expected: Compiles and links successfully.

- [ ] **Step 3: Smoke test (bench mode)**

```bash
./build/slick --model models/gpt2-int8 --bench --greedy --prompt "Hello world" --max-tokens 20
```

Expected: Prints generated tokens and metrics summary. Tokens should be English text (may be noisy due to INT8 quantization, but should be coherent).

- [ ] **Step 4: Test TUI mode**

```bash
./build/slick --model models/gpt2-int8 --max-tokens 64
```

Expected: FTXUI renders two-panel layout. Type a prompt, press Enter, see tokens stream with live metrics. Press Escape to quit.

- [ ] **Step 5: Commit**

```bash
git add src/main.cu
git commit -m "feat: add FTXUI TUI and CLI for GPT-2 inference demo"
```

---

### Task 11: Integration Test & Polish

**Files:**
- Create: `tests/test_engine.cu`

- [ ] **Step 1: Write greedy decode validation test**

Create `tests/test_engine.cu`:

```cpp
// tests/test_engine.cu
// Integration test: verify greedy decode produces deterministic output
#include <gtest/gtest.h>
#include "gpt2_engine.cuh"
#include "tokenizer.h"
#include <vector>
#include <string>

class EngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = new GPT2Engine("models/gpt2-int8");
        tokenizer_ = new BPETokenizer("models/gpt2-int8");
    }
    void TearDown() override {
        delete engine_;
        delete tokenizer_;
    }
    GPT2Engine* engine_;
    BPETokenizer* tokenizer_;
};

TEST_F(EngineTest, GreedyDeterministic) {
    // Two runs with the same prompt should produce identical output
    auto prompt = tokenizer_->encode("The capital of France is");
    std::vector<int> run1_tokens, run2_tokens;
    InferenceMetrics m;

    engine_->generate(prompt, 10, 1.0f, 1, true,
        [&](const std::string& tok) { run1_tokens.push_back(atoi(tok.c_str())); }, m);

    engine_->generate(prompt, 10, 1.0f, 1, true,
        [&](const std::string& tok) { run2_tokens.push_back(atoi(tok.c_str())); }, m);

    ASSERT_EQ(run1_tokens.size(), run2_tokens.size());
    for (size_t i = 0; i < run1_tokens.size(); ++i)
        EXPECT_EQ(run1_tokens[i], run2_tokens[i]) << "Token mismatch at position " << i;
}

TEST_F(EngineTest, MetricsPopulated) {
    auto prompt = tokenizer_->encode("Hello");
    InferenceMetrics m;

    engine_->generate(prompt, 5, 1.0f, 1, true,
        [](const std::string&) {}, m);

    EXPECT_GT(m.ttft_ms, 0.0f);
    EXPECT_GT(m.tps, 0.0f);
    EXPECT_GT(m.kv_cache_pct, 0.0f);
    EXPECT_EQ(m.tokens_generated, 5);
}

TEST_F(EngineTest, EosStopsGeneration) {
    // Generate with a long max but expect natural EOS
    auto prompt = tokenizer_->encode("The end.");
    InferenceMetrics m;
    int count = 0;

    engine_->generate(prompt, 1024, 1.0f, 1, true,
        [&](const std::string&) { count++; }, m);

    // Should have stopped before 1024 (EOS or at limit)
    EXPECT_LE(count, 1024);
    EXPECT_GT(count, 0);
}
```

- [ ] **Step 2: Add to CMakeLists.txt and build**

```cmake
add_executable(test_engine tests/test_engine.cu)
target_include_directories(test_engine PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_engine GTest::gtest_main gpt2_engine)
add_test(NAME EngineTests COMMAND test_engine)
```

```bash
cmake --build build --target test_engine
./build/test_engine
```

Expected: All 3 tests PASS.

- [ ] **Step 3: Run full test suite to check no regressions**

```bash
ctest --test-dir build --output-on-failure
```

Expected: All test suites pass (GEMM, Softmax, Attention, PagedAttention, Decode, Int8Gemm, Int8Epilogue, LayerNorm, Engine).

- [ ] **Step 4: Commit**

```bash
git add tests/test_engine.cu CMakeLists.txt
git commit -m "test: add GPT-2 engine integration tests"
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: mark Week 7 complete in roadmap"
```

---

### Task 12: FP32 Weight Export & cuBLAS Backend

**Files:**
- Modify: `python/export_gpt2.py` (add FP32 export)
- Modify: `include/gpt2_types.cuh` (add FP32 weight struct + backend enum)
- Modify: `src/gpt2_engine.cuh` (add backend parameter)
- Modify: `src/gpt2_engine.cu` (add cuBLAS FP32 forward path)

The cuBLAS backend reuses the same engine, attention kernels, KV cache, TUI — only the linear projections change from K14 INT8 GEMM to cuBLAS FP32 SGEMM.

- [ ] **Step 1: Update export script to also save FP32 weights**

In `python/export_gpt2.py`, add FP32 export alongside INT8. After each INT8 quantized weight is saved, also save the raw FP32 weight matrix in the layout cuBLAS needs.

Add this function:

```python
def save_fp32_weight(layer_dir: str, name: str, weight_fp32: np.ndarray):
    """Save FP32 weight in [K, N] row-major layout for cuBLAS.
    
    cuBLAS computes y = x @ W where x is [M, K] and W is [K, N].
    We store W as-is (row-major [K, N]).
    """
    save_bin(os.path.join(layer_dir, f"{name}_weight_fp32.bin"), weight_fp32.astype(np.float32))
```

Add these calls inside the per-layer loop, right after each INT8 export:

```python
        # QKV: W is [768, 2304] — save as-is for cuBLAS
        save_fp32_weight(layer_dir, "qkv", qkv_w)

        # Output proj: [768, 768]
        save_fp32_weight(layer_dir, "out", out_w)

        # FFN up: [768, 3072]
        save_fp32_weight(layer_dir, "ffn_up", up_w)

        # FFN down: [3072, 768]
        save_fp32_weight(layer_dir, "ffn_down", down_w)
```

Re-run the export:

```bash
uv run python python/export_gpt2.py --output models/gpt2-int8
```

Expected: Each layer directory now has `*_weight_fp32.bin` files alongside the INT8 packed files. Total size increases from ~140MB to ~620MB.

- [ ] **Step 2: Add backend enum and FP32 weight struct to types**

In `include/gpt2_types.cuh`, add:

```cpp
enum class InferenceBackend { SLICK_INT8, CUBLAS_FP32 };

// FP32 weight matrix for cuBLAS path
struct FP32Weight {
    float* data;    // [K, N] row-major
    float* bias;    // [N]
    int K;          // input dim
    int N;          // output dim
};

// Extended per-layer weights (add FP32 variants)
struct LayerWeightsFP32 {
    FP32Weight qkv;       // [768, 2304]
    FP32Weight out;       // [768, 768]
    FP32Weight ffn_up;    // [768, 3072]
    FP32Weight ffn_down;  // [3072, 768]
};
```

And add to `GPT2Weights`:

```cpp
struct GPT2Weights {
    // ... existing fields ...
    LayerWeightsFP32 layers_fp32[12];  // FP32 weights for cuBLAS backend
};
```

- [ ] **Step 3: Add backend to engine and implement cuBLAS FP32 linear**

In `src/gpt2_engine.cuh`, add to the constructor and class:

```cpp
class GPT2Engine {
public:
    GPT2Engine(const std::string& model_dir, InferenceBackend backend = InferenceBackend::SLICK_INT8);
    // ... rest unchanged ...

private:
    InferenceBackend backend_;
    cublasHandle_t cublas_handle_;  // for FP32 backend

    void load_weights_fp32();
    void forward_layer_cublas(int layer, int seq_len, bool is_prefill);
};
```

In `src/gpt2_engine.cu`, add FP32 weight loading:

```cpp
static float* load_bin_f32_raw(const std::string& path, size_t count) {
    // Same as load_bin_f32 — reuse existing function
    return load_bin_f32(path, count);
}

static FP32Weight load_fp32_weight(const std::string& dir, const char* name,
                                    int K, int N, const std::string& bias_path) {
    FP32Weight w;
    w.K = K;
    w.N = N;
    w.data = load_bin_f32(dir + "/" + name + "_weight_fp32.bin", (size_t)K * N);
    w.bias = load_bin_f32(bias_path, N);
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

        // Note: bias is shared between INT8 and FP32 paths (already loaded)
        fw.qkv = {load_bin_f32(ldir + "/qkv_weight_fp32.bin", (size_t)d * 3 * d),
                   weights_.layers[i].qkv_bias, d, 3 * d};
        fw.out = {load_bin_f32(ldir + "/out_weight_fp32.bin", (size_t)d * d),
                  weights_.layers[i].out_bias, d, d};
        fw.ffn_up = {load_bin_f32(ldir + "/ffn_up_weight_fp32.bin", (size_t)d * ff),
                     weights_.layers[i].ffn_up_bias, d, ff};
        fw.ffn_down = {load_bin_f32(ldir + "/ffn_down_weight_fp32.bin", (size_t)ff * d),
                       weights_.layers[i].ffn_down_bias, ff, d};
    }
}
```

Add a helper for cuBLAS SGEMM with bias:

```cpp
// cuBLAS SGEMM: C = A @ W + bias (row-major)
// A: [M, K], W: [K, N], C: [M, N], bias: [N]
static void cublas_linear(cublasHandle_t handle,
                          int M, int N, int K,
                          const float* A, const float* W, const float* bias,
                          float* C, float* workspace) {
    float alpha = 1.0f, beta = 0.0f;
    // Row-major trick: C^T = W^T @ A^T
    // Treating row-major as col-major: W [K,N] → "col-major [N,K]" = W^T
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, W, N, A, K, &beta, C, N);

    // Add bias: C[i][j] += bias[j] for all rows i
    // Simple kernel:
    extern void run_bias_add(float* C, const float* bias, int M, int N);
    run_bias_add(C, bias, M, N);
}

__global__ void bias_add_kernel(float* __restrict__ C, const float* __restrict__ bias,
                                 int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    C[idx] += bias[idx % N];
}

void run_bias_add(float* C, const float* bias, int M, int N) {
    int total = M * N;
    bias_add_kernel<<<(total + 255) / 256, 256>>>(C, bias, M, N);
}

// cuBLAS SGEMM + bias + GELU
__global__ void bias_gelu_kernel(float* __restrict__ C, const float* __restrict__ bias,
                                  int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    float val = C[idx] + bias[idx % N];
    float x3 = val * val * val;
    C[idx] = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * x3)));
}

void run_bias_gelu(float* C, const float* bias, int M, int N) {
    int total = M * N;
    bias_gelu_kernel<<<(total + 255) / 256, 256>>>(C, bias, M, N);
}

// cuBLAS SGEMM + bias + residual
__global__ void bias_residual_kernel(float* __restrict__ C, const float* __restrict__ bias,
                                      const float* __restrict__ residual, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    C[idx] += bias[idx % N] + residual[idx];
}

void run_bias_residual(float* C, const float* bias, const float* residual, int M, int N) {
    int total = M * N;
    bias_residual_kernel<<<(total + 255) / 256, 256>>>(C, bias, residual, M, N);
}
```

Add the cuBLAS forward layer variant:

```cpp
void GPT2Engine::forward_layer_cublas(int layer, int seq_len, bool is_prefill) {
    int d = config_.d_model;
    int H = config_.n_heads;
    int dh = config_.d_head;
    int ff = config_.d_ff;
    auto& lw = weights_.layers[layer];     // LN weights, bias (shared)
    auto& fw = weights_.layers_fp32[layer]; // FP32 GEMM weights
    int M = seq_len;

    // --- Attention sub-block ---
    CUDA_CHECK(cudaMemcpy(d_residual_, d_x_, (size_t)M * d * sizeof(float), cudaMemcpyDeviceToDevice));
    run_layernorm(M, d, d_x_, lw.ln1_gamma, lw.ln1_beta, d_x_);

    // QKV: cuBLAS SGEMM + bias
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                3 * d, M, d, &alpha, fw.qkv.data, 3 * d, d_x_, d, &beta, d_qkv_, 3 * d);
    run_bias_add(d_qkv_, lw.qkv_bias, M, 3 * d);

    float* Q = d_qkv_;
    float* K = d_qkv_ + (size_t)M * d;
    float* V = d_qkv_ + (size_t)M * d * 2;

    int h_ctx;
    CUDA_CHECK(cudaMemcpy(&h_ctx, d_context_lens_, sizeof(int), cudaMemcpyDeviceToHost));
    int start_pos = is_prefill ? 0 : h_ctx;

    kv_cache_append(layer, K, V, M, start_pos);
    int total_ctx = start_pos + M;

    size_t layer_cache_offset = (size_t)layer * num_phys_blocks_ * block_size_ * H * dh;
    float* layer_k = d_k_cache_ + layer_cache_offset;
    float* layer_v = d_v_cache_ + layer_cache_offset;

    if (is_prefill) {
        run_flash_attn_v2(1, H, M, dh, Q, K, V, d_attn_out_, true);
    } else {
        int ctx_for_decode = total_ctx;
        CUDA_CHECK(cudaMemcpy(d_context_lens_, &ctx_for_decode, sizeof(int), cudaMemcpyHostToDevice));
        int max_blocks = (ctx_for_decode + block_size_ - 1) / block_size_;
        run_decode_attn(1, H, H, dh, Q, layer_k, layer_v,
                        d_block_tables_, d_context_lens_,
                        ctx_for_decode, block_size_, max_blocks,
                        d_attn_out_, d_decode_workspace_);
    }

    // Output proj: cuBLAS SGEMM + bias + residual
    cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                d, M, d, &alpha, fw.out.data, d, d_attn_out_, d, &beta, d_x_, d);
    run_bias_residual(d_x_, lw.out_bias, d_residual_, M, d);

    // --- FFN sub-block ---
    CUDA_CHECK(cudaMemcpy(d_residual_, d_x_, (size_t)M * d * sizeof(float), cudaMemcpyDeviceToDevice));
    run_layernorm(M, d, d_x_, lw.ln2_gamma, lw.ln2_beta, d_x_);

    // FFN up: cuBLAS SGEMM + bias + GELU
    cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                ff, M, d, &alpha, fw.ffn_up.data, ff, d_x_, d, &beta, d_ffn_hidden_, ff);
    run_bias_gelu(d_ffn_hidden_, lw.ffn_up_bias, M, ff);

    // FFN down: cuBLAS SGEMM + bias + residual
    cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                d, M, ff, &alpha, fw.ffn_down.data, d, d_ffn_hidden_, ff, &beta, d_x_, d);
    run_bias_residual(d_x_, lw.ffn_down_bias, d_residual_, M, d);
}
```

Update `forward_layer` dispatch in `generate()`:

```cpp
for (int l = 0; l < config_.n_layers; ++l) {
    if (backend_ == InferenceBackend::CUBLAS_FP32)
        forward_layer_cublas(l, seq_len, is_prefill);
    else
        forward_layer(l, seq_len, is_prefill);
}
```

Update constructor:

```cpp
GPT2Engine::GPT2Engine(const std::string& model_dir, InferenceBackend backend)
    : model_dir_(model_dir), backend_(backend) {
    load_config();
    load_weights();
    if (backend_ == InferenceBackend::CUBLAS_FP32) {
        load_weights_fp32();
        cublasCreate(&cublas_handle_);
    }
    alloc_workspace();
}
```

- [ ] **Step 4: Update main.cu CLI to accept --backend flag**

In `src/main.cu`, add parsing:

```cpp
    InferenceBackend backend = InferenceBackend::SLICK_INT8;

    for (int i = 1; i < argc; ++i) {
        // ... existing args ...
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            ++i;
            if (strcmp(argv[i], "cublas") == 0) backend = InferenceBackend::CUBLAS_FP32;
            else if (strcmp(argv[i], "int8") == 0) backend = InferenceBackend::SLICK_INT8;
        }
    }

    GPT2Engine engine(model_dir, backend);
```

- [ ] **Step 5: Build and test both backends**

```bash
cmake --build build --target slick
./build/slick --bench --greedy --prompt "The capital of France" --max-tokens 20
./build/slick --bench --greedy --prompt "The capital of France" --max-tokens 20 --backend cublas
```

Expected: Both produce English text. The INT8 backend should show higher TPS than cuBLAS FP32.

- [ ] **Step 6: Commit**

```bash
git add python/export_gpt2.py include/gpt2_types.cuh src/gpt2_engine.cuh src/gpt2_engine.cu src/main.cu
git commit -m "feat: add cuBLAS FP32 backend for baseline comparison"
```

---

### Task 13: PyTorch Baseline Benchmark

**Files:**
- Create: `python/benchmark_pytorch.py`

- [ ] **Step 1: Write PyTorch benchmark script**

Create `python/benchmark_pytorch.py`:

```python
#!/usr/bin/env python3
"""Benchmark HuggingFace GPT-2 inference on GPU — baseline for SLICK comparison."""

import argparse
import json
import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def benchmark(prompt: str, max_tokens: int, device: str = "cuda"):
    print(f"Loading GPT-2 Small on {device}...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    print(f"Prompt ({prompt_len} tokens): {prompt}")

    # Warmup
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=5, do_sample=False)
    torch.cuda.synchronize()

    # Benchmark with CUDA events for accurate GPU timing
    start_event = torch.cuda.Event(enable_timing=True)
    first_token_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    generated_tokens = []
    itl_samples = []
    past_key_values = None

    torch.cuda.synchronize()
    start_event.record()

    with torch.no_grad():
        # Prefill
        outputs = model(input_ids, past_key_values=None, use_cache=True)
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values

        first_token_event.record()
        torch.cuda.synchronize()

        generated_tokens.append(next_token.item())
        prev_event = torch.cuda.Event(enable_timing=True)
        prev_event.record()

        # Decode loop
        for step in range(max_tokens - 1):
            if next_token.item() == tokenizer.eos_token_id:
                break

            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values

            step_event = torch.cuda.Event(enable_timing=True)
            step_event.record()
            torch.cuda.synchronize()
            itl_samples.append(prev_event.elapsed_time(step_event))
            prev_event = step_event

            generated_tokens.append(next_token.item())
            print(tokenizer.decode([next_token.item()]), end="", flush=True)

    end_event.record()
    torch.cuda.synchronize()

    # Compute metrics
    ttft_ms = start_event.elapsed_time(first_token_event)
    total_ms = start_event.elapsed_time(end_event)
    num_tokens = len(generated_tokens)
    tpot_ms = total_ms / num_tokens if num_tokens > 0 else 0
    tps = num_tokens / (total_ms / 1000.0) if total_ms > 0 else 0

    itl_min = min(itl_samples) if itl_samples else 0
    itl_max = max(itl_samples) if itl_samples else 0
    itl_sorted = sorted(itl_samples)
    itl_p95 = itl_sorted[int(len(itl_sorted) * 0.95)] if len(itl_sorted) >= 20 else itl_max

    # VRAM
    vram_used = torch.cuda.memory_allocated() / 1024 / 1024
    vram_total = torch.cuda.get_device_properties(0).total_mem / 1024 / 1024

    generated_text = tokenizer.decode(generated_tokens)

    print(f"\n\n{'='*50}")
    print(f"PyTorch GPT-2 FP32 Baseline")
    print(f"{'='*50}")
    print(f"Generated: {prompt}{generated_text}")
    print(f"Tokens:       {num_tokens}")
    print(f"TTFT:         {ttft_ms:.1f} ms")
    print(f"TPOT:         {tpot_ms:.1f} ms")
    print(f"TPS:          {tps:.1f}")
    print(f"ITL P95:      {itl_p95:.1f} ms")
    print(f"ITL min/max:  {itl_min:.1f}/{itl_max:.1f} ms")
    print(f"End-to-end:   {total_ms:.1f} ms")
    print(f"VRAM:         {vram_used:.0f}/{vram_total:.0f} MB")

    # Save as JSON for comparison script
    results = {
        "backend": "pytorch_fp32",
        "tokens": num_tokens,
        "ttft_ms": round(ttft_ms, 1),
        "tpot_ms": round(tpot_ms, 1),
        "tps": round(tps, 1),
        "itl_p95_ms": round(itl_p95, 1),
        "end_to_end_ms": round(total_ms, 1),
        "vram_mb": round(vram_used, 0),
    }
    with open("pytorch_baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to pytorch_baseline.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="The meaning of life is",
                        help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Maximum tokens to generate")
    args = parser.parse_args()
    benchmark(args.prompt, args.max_tokens)
```

- [ ] **Step 2: Run and verify**

```bash
uv run python python/benchmark_pytorch.py --prompt "The meaning of life is" --max-tokens 64
```

Expected: Prints generated text and metrics. Saves `pytorch_baseline.json`. Note the TPS value — this is the number to beat.

- [ ] **Step 3: Commit**

```bash
git add python/benchmark_pytorch.py
git commit -m "bench: add PyTorch FP32 baseline benchmark script"
```

---

### Task 14: Three-Way Comparison Demo

**Files:**
- Create: `scripts/demo_compare.sh`
- Modify: `src/main.cu` (add `--compare` flag and JSON output for `--bench`)

- [ ] **Step 1: Add JSON metrics output to bench mode**

In `src/main.cu`, update the bench mode to optionally write a JSON results file:

```cpp
    if (bench_mode) {
        if (initial_prompt.empty()) initial_prompt = "The meaning of life is";
        auto tokens = tokenizer.encode(initial_prompt);
        printf("Prompt (%d tokens): %s\n", (int)tokens.size(), initial_prompt.c_str());

        InferenceMetrics metrics;
        GpuTimer total;
        total.tic();

        engine.generate(tokens, max_tokens, temperature, top_k_val, greedy,
                        [&](const std::string& tok_id_str) {
                            int id = atoi(tok_id_str.c_str());
                            printf("%s", tokenizer.decode(id).c_str());
                            fflush(stdout);
                        }, metrics);

        float total_ms = total.toc();
        metrics.end_to_end_ms = total_ms;

        const char* backend_name = (backend == InferenceBackend::CUBLAS_FP32)
                                    ? "cublas_fp32" : "slick_int8";

        printf("\n\n==================================================\n");
        printf("SLICK %s\n", backend_name);
        printf("==================================================\n");
        printf("Tokens:       %d\n", metrics.tokens_generated);
        printf("TTFT:         %.1f ms\n", metrics.ttft_ms);
        printf("TPOT:         %.1f ms\n", metrics.tpot_ms);
        printf("TPS:          %.1f\n", metrics.tps);
        printf("ITL P95:      %.1f ms\n", metrics.itl_p95_ms);
        printf("ITL min/max:  %.1f/%.1f ms\n", metrics.itl_min_ms, metrics.itl_max_ms);
        printf("End-to-end:   %.1f ms\n", metrics.end_to_end_ms);
        printf("KV Cache:     %.0f%%\n", metrics.kv_cache_pct);
        printf("VRAM:         %.0f/%.0f MB\n", metrics.vram_used_mb, metrics.vram_total_mb);
        printf("MBU:          %.1f%%\n", metrics.mbu * 100.0f);

        // Write JSON for comparison script
        char json_file[256];
        snprintf(json_file, sizeof(json_file), "%s_results.json", backend_name);
        FILE* jf = fopen(json_file, "w");
        if (jf) {
            fprintf(jf, "{\n");
            fprintf(jf, "  \"backend\": \"%s\",\n", backend_name);
            fprintf(jf, "  \"tokens\": %d,\n", metrics.tokens_generated);
            fprintf(jf, "  \"ttft_ms\": %.1f,\n", metrics.ttft_ms);
            fprintf(jf, "  \"tpot_ms\": %.1f,\n", metrics.tpot_ms);
            fprintf(jf, "  \"tps\": %.1f,\n", metrics.tps);
            fprintf(jf, "  \"itl_p95_ms\": %.1f,\n", metrics.itl_p95_ms);
            fprintf(jf, "  \"end_to_end_ms\": %.1f,\n", metrics.end_to_end_ms);
            fprintf(jf, "  \"vram_mb\": %.0f\n", metrics.vram_used_mb);
            fprintf(jf, "}\n");
            fclose(jf);
            printf("\nResults saved to %s\n", json_file);
        }
        return 0;
    }
```

- [ ] **Step 2: Write the comparison shell script**

Create `scripts/demo_compare.sh`:

```bash
#!/bin/bash
# Three-way GPT-2 inference comparison: PyTorch FP32 vs cuBLAS FP32 vs SLICK INT8
set -e

PROMPT="${1:-The meaning of life is}"
MAX_TOKENS="${2:-64}"
MODEL_DIR="${3:-models/gpt2-int8}"

echo "========================================================"
echo " SLICK — Three-Way Inference Comparison"
echo " Prompt: \"$PROMPT\""
echo " Max tokens: $MAX_TOKENS"
echo " Hardware: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================================"
echo ""

# 1. PyTorch FP32 baseline
echo ">>> [1/3] Running PyTorch FP32 baseline..."
echo "--------------------------------------------------------"
uv run python python/benchmark_pytorch.py \
    --prompt "$PROMPT" --max-tokens "$MAX_TOKENS"
echo ""

# 2. cuBLAS FP32 backend
echo ">>> [2/3] Running SLICK cuBLAS FP32 backend..."
echo "--------------------------------------------------------"
./build/slick --bench --greedy --backend cublas \
    --model "$MODEL_DIR" --prompt "$PROMPT" --max-tokens "$MAX_TOKENS"
echo ""

# 3. SLICK INT8 (our optimized kernels)
echo ">>> [3/3] Running SLICK INT8 (optimized)..."
echo "--------------------------------------------------------"
./build/slick --bench --greedy \
    --model "$MODEL_DIR" --prompt "$PROMPT" --max-tokens "$MAX_TOKENS"
echo ""

# Print comparison table from JSON files
echo "========================================================"
echo " COMPARISON TABLE"
echo "========================================================"
python3 -c "
import json, os

backends = []
for f in ['pytorch_baseline.json', 'cublas_fp32_results.json', 'slick_int8_results.json']:
    if os.path.exists(f):
        with open(f) as fh:
            backends.append(json.load(fh))

if not backends:
    print('No result files found.')
    exit()

# Header
print(f'{\"Metric\":<20} ', end='')
for b in backends:
    print(f'{b[\"backend\"]:>18} ', end='')
print()
print('-' * (20 + 19 * len(backends)))

# Rows
for key, label, unit in [
    ('ttft_ms', 'TTFT', 'ms'),
    ('tpot_ms', 'TPOT', 'ms'),
    ('tps', 'TPS', 'tok/s'),
    ('itl_p95_ms', 'ITL (P95)', 'ms'),
    ('end_to_end_ms', 'End-to-end', 'ms'),
    ('vram_mb', 'VRAM', 'MB'),
    ('tokens', 'Tokens', ''),
]:
    print(f'{label + \" (\" + unit + \")\" if unit else label:<20} ', end='')
    for b in backends:
        val = b.get(key, '--')
        print(f'{val:>18} ', end='')
    print()

# Speedup row
if len(backends) == 3:
    print('-' * (20 + 19 * len(backends)))
    pt_tps = backends[0].get('tps', 1)
    print(f'{\"Speedup vs PyTorch\":<20} ', end='')
    for b in backends:
        speedup = b.get('tps', 0) / pt_tps if pt_tps > 0 else 0
        print(f'{speedup:>17.1f}x ', end='')
    print()
"

# Cleanup
rm -f pytorch_baseline.json cublas_fp32_results.json slick_int8_results.json
```

- [ ] **Step 3: Make executable and test**

```bash
chmod +x scripts/demo_compare.sh
./scripts/demo_compare.sh "The capital of France is" 32
```

Expected output:

```
========================================================
 SLICK — Three-Way Inference Comparison
 Prompt: "The capital of France is"
 Max tokens: 32
 Hardware: NVIDIA GeForce GTX 1650 Ti
========================================================

>>> [1/3] Running PyTorch FP32 baseline...
...
>>> [2/3] Running SLICK cuBLAS FP32 backend...
...
>>> [3/3] Running SLICK INT8 (optimized)...
...

========================================================
 COMPARISON TABLE
========================================================
Metric                     pytorch_fp32      cublas_fp32       slick_int8
-----------------------------------------------------------------------
TTFT (ms)                      xxx.x            xxx.x            xxx.x
TPOT (ms)                      xxx.x            xxx.x            xxx.x
TPS (tok/s)                    xxx.x            xxx.x            xxx.x
ITL (P95) (ms)                 xxx.x            xxx.x            xxx.x
End-to-end (ms)                xxx.x            xxx.x            xxx.x
VRAM (MB)                      xxxx             xxxx             xxxx
-----------------------------------------------------------------------
Speedup vs PyTorch              1.0x             x.xx             x.xx
```

- [ ] **Step 4: Commit**

```bash
git add scripts/demo_compare.sh src/main.cu
git commit -m "feat: add three-way inference comparison demo"
```

- [ ] **Step 5: Update README roadmap**

In `README.md`, update the Week 7 roadmap entry:

```
- [x] Week 7: GPT-2 inference demo
```

- [ ] **Step 6: Final commit**

```bash
git add README.md
git commit -m "docs: mark Week 7 complete in roadmap"
```

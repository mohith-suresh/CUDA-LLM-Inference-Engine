# GPT-2 INT8 Inference Engine — Design Spec

## Overview

Week 7 capstone: a self-contained GPT-2 Small (124M) inference engine using SLICK's custom CUDA kernels with INT8 quantization. Interactive TUI demo with live metrics streaming.

**Hardware target:** GTX 1650 Ti, CC 7.5, 4GB VRAM, CUDA 11.8

## Architecture

Three components:

1. **`export_gpt2.py`** — One-time Python script. Downloads GPT-2-small from HuggingFace, quantizes all linear weights to INT8 with per-row symmetric scales, exports raw `.bin` files + tokenizer data.
2. **`gpt2_engine`** — C++ inference engine. Loads exported weights, orchestrates the GPT-2 forward pass using existing SLICK kernels with fused epilogues.
3. **`slick` (main)** — FTXUI-based TUI. Two-panel layout: streaming token output + live metrics dashboard. Interactive prompt input.

## Kernel Mapping

| Operation | Kernel | Notes |
|-----------|--------|-------|
| QKV / output / FFN projections | K14 (INT8 GEMM) | Templated epilogues: BiasOnly, BiasGELU, BiasResidual |
| Prefill attention | K10 (FlashAttention-2) | Full sequence, causal mask |
| Decode attention | K13 (decode attention) | Single-token, paged KV cache, split-K |
| Paged KV cache structure | K11 (block table) | block_size=16 |
| LayerNorm + residual | K15 (new) | Fused pre-norm + residual add, vectorized |
| Final logit softmax | K08 (fused online) | Row-wise softmax over vocab |

### K14 Epilogue Extensions

The existing INT8 GEMM kernel (K14) gains three templated epilogue variants. The GEMM tile math is unchanged — only the store path differs:

- **`Epilogue::BiasOnly`** — dequantize INT32 accumulators to FP32, add bias vector, store
- **`Epilogue::BiasGELU`** — dequantize, add bias, apply GELU activation, store
- **`Epilogue::BiasResidual`** — dequantize, add bias, load residual from global memory, add, store

This eliminates standalone element-wise kernels for bias, GELU, and residual add — these ops execute in-register on the compute-bound GEMM output at zero extra memory traffic.

### K15: Fused LayerNorm + Residual (New Kernel)

Pre-norm GPT-2 applies `x = LayerNorm(residual + x)` before each sub-block. Single kernel:

1. Load `x` and `residual` via float4 vectorized loads
2. Compute `y = x + residual` (store residual output for later use)
3. Warp-cooperative reduction for mean and variance over d=768
4. Normalize: `output = gamma * (y - mean) / sqrt(var + eps) + beta`

Memory-bound (AI ~2 FLOP/byte), but fusing residual+LN into one pass halves the traffic vs two separate kernels. float4 vectorized loads to approach peak bandwidth.

## GPT-2 Forward Pass (Per Layer)

GPT-2 Small config: 12 layers, 12 heads, d_model=768, d_head=64, d_ff=3072, vocab=50257

```
Input x [B, N, 768]

  residual = x
  x = K15_LayerNorm_Residual(x, residual, gamma1, beta1)
  QKV = K14_BiasOnly(x, W_qkv)                       # [B,N,768] x [768,2304]
  Q, K, V = split(QKV, dim=-1, chunks=3)             # pointer arithmetic
  Prefill:  O = K10_FlashAttn(Q, K, V, causal=true)  # full sequence
  Decode:   O = K13_DecodeAttn(q, KV_cache, block_table) # single token
  x = K14_BiasResidual(O, W_o, residual)              # [B,N,768] x [768,768]

  residual = x
  x = K15_LayerNorm_Residual(x, residual, gamma2, beta2)
  x = K14_BiasGELU(x, W_up)                           # [B,N,768] x [768,3072]
  x = K14_BiasResidual(x, W_down, residual)            # [B,N,3072] x [3072,768]

Output x [B, N, 768]

Final layer:
  x = LayerNorm(x)
  logits = K14_BiasOnly(x, W_vocab)                    # [B,1,768] x [768,50257]
  probs = K08_Softmax(logits)
  token = sample(probs)
```

**Kernel launches per layer:** 4x K14 + 2x K15 + 1x attention = 7
**Total per token:** 12 layers x 7 + 3 (final LN + vocab proj + softmax) = **87 kernel launches**

## KV Cache & Memory Management

**Paged KV cache using K11's block table structure:**

- Block size: 16 tokens (matches K11/K13 Bc=16 tiling)
- Max blocks per sequence: 1024 / 16 = 64 blocks
- Per-block memory: 2 (K+V) x 12 heads x 16 tokens x 64 dims x 4 bytes = 96KB
- Full context (1024 tokens): 64 x 96KB = 6MB per sequence

**Block allocator:** Pre-allocated pool with free-list. Allocate on prefill + each decode step crossing a block boundary. Free all on sequence completion.

**Memory budget (1024 context):**

| Component | Size |
|-----------|------|
| INT8 weights + scales | ~140MB |
| KV cache pool (1 seq) | 6MB |
| Activations workspace | ~25MB (single-token decode) |
| FTXUI + overhead | ~20MB |
| **Total** | ~190MB |
| **Free VRAM** | ~3.8GB |

## Metrics Collection

All latency metrics use CUDA events (cudaEventRecord/cudaEventElapsedTime) for GPU-accurate timing.

| Metric | Computation | Update Frequency |
|--------|-------------|------------------|
| TTFT | Event diff: prompt submit to first token emitted | Once per generation |
| TPOT | Running average of per-token decode time | Each token |
| ITL | Per-token event diff; track min, max, P95 | Each token |
| TPS | tokens_generated / wall_time | Each token |
| End-to-end latency | Event diff: prompt submit to last token | Once per generation |
| KV cache utilization | allocated_blocks / pool_size x 100% | Each token |
| VRAM usage | cudaMemGetInfo (used / total) | Each token |
| MBU | actual TPS / theoretical max TPS | Each token |

## TUI Layout (FTXUI)

```
+-- SLICK GPT-2 Engine --------------------------------------------------------+
|                                                                               |
|  +- Output --------------------------+  +- Metrics -------------------------+ |
|  |                                   |  |                                   | |
|  |  Once upon a time there was       |  |  TTFT        42.3 ms             | |
|  |  a small dragon who lived         |  |  TPOT        18.7 ms             | |
|  |  in a cave beneath the            |  |  ITL (P95)   21.2 ms             | |
|  |  mountain. Every day he           |  |  TPS          53.4               | |
|  |  would _                          |  |  Tokens      47/256              | |
|  |                                   |  |                                  | |
|  |                                   |  |  --- Resource ----------------   | |
|  |                                   |  |  KV Cache  ####....  47%        | |
|  |                                   |  |  VRAM      ##......  19%        | |
|  |                                   |  |  MBU                 34%        | |
|  |                                   |  |                                  | |
|  |                                   |  |  --- Latency -----------------  | |
|  |                                   |  |  End-to-end   --                | |
|  |                                   |  |  ITL min/max  14/23 ms          | |
|  +-----------------------------------+  +----------------------------------+ |
|                                                                               |
|  > Enter prompt: _                                                            |
+-------------------------------------------------------------------------------+
```

- Left panel: scrolling generated text with blinking cursor
- Right panel: live metrics with FTXUI gauge components for KV cache and VRAM
- Bottom: interactive prompt input
- Refresh: FTXUI ScreenInteractive loop, metrics update each token

## Tokenizer

Full BPE encode + decode in C++, loaded from HuggingFace-exported files.

**Export script writes:**
- `vocab.json` — token_id to byte string mapping (50257 entries)
- `merges.txt` — BPE merge pairs with rank ordering

**C++ tokenizer implements:**
- `encode(string) -> vector<int>` — BPE merge loop: split to bytes, greedily merge adjacent pairs by rank
- `decode(int) -> string` — direct vocab table lookup

CPU-only, runs once per prompt (encode) and once per generated token (decode). Microsecond-scale, not a bottleneck.

## Weight Export (Python)

`python/export_gpt2.py`:

1. Load GPT-2-small via `transformers.AutoModelForCausalLM`
2. For each linear weight matrix:
   - Compute per-row scale: `scale[i] = max(|row[i]|) / 127`
   - Quantize: `W_int8[i] = round(W_fp32[i] / scale[i])`, clamp to [-128, 127]
   - Pack 4 int8 values into int32 for dp4a (K-dim contiguous, NT layout matching K14)
   - Write `{name}_weight.bin` (packed int32) and `{name}_scale.bin` (float32)
3. For each bias, LayerNorm gamma/beta, embedding table:
   - Write as raw FP32 `.bin` files (no quantization)
4. Export `vocab.json` and `merges.txt` from tokenizer
5. Write `config.json` with model dimensions

**Output directory structure:**
```
models/gpt2-int8/
  config.json
  vocab.json
  merges.txt
  wte.bin                    # token embedding [50257, 768] FP32
  wpe.bin                    # position embedding [1024, 768] FP32
  ln_f_gamma.bin             # final layernorm
  ln_f_beta.bin
  layer_00/
    ln1_gamma.bin, ln1_beta.bin
    qkv_weight.bin, qkv_scale.bin, qkv_bias.bin
    out_weight.bin, out_scale.bin, out_bias.bin
    ln2_gamma.bin, ln2_beta.bin
    ffn_up_weight.bin, ffn_up_scale.bin, ffn_up_bias.bin
    ffn_down_weight.bin, ffn_down_scale.bin, ffn_down_bias.bin
  layer_01/
    ...
  layer_11/
    ...
```

## File Structure

**New files:**
```
python/
  export_gpt2.py

kernels/
  layernorm/
    15_layernorm_residual.cu
    15_layernorm_residual.cuh

include/
  gpt2_types.cuh             # GPT2Config, LayerWeights, KVBlock, Metrics structs
  kv_cache.cuh               # Block allocator + paged cache manager
  gpt2_engine.cuh            # Forward pass orchestrator
  tokenizer.cuh              # BPE encode/decode
  sampler.cuh                # Top-k / temperature sampling

src/
  gpt2_engine.cu
  kv_cache.cu
  tokenizer.cu
  sampler.cu
  main.cu                    # FTXUI app, CLI parsing, main loop
```

**Modified files:**
- `kernels/quantization/14_int8_gemm.cu/.cuh` — add templated epilogues (BiasOnly, BiasGELU, BiasResidual)
- `CMakeLists.txt` — add FTXUI FetchContent, new libraries and executable targets

## CLI Interface

```bash
# One-time export
python python/export_gpt2.py --output models/gpt2-int8/

# Interactive TUI
./build/slick --model models/gpt2-int8/ --max-tokens 256

# Single prompt mode
./build/slick --model models/gpt2-int8/ --prompt "Once upon a time" --max-tokens 512

# Benchmark mode (no TUI, metrics to stdout)
./build/slick --model models/gpt2-int8/ --bench --prompt "The meaning of life" --max-tokens 128
```

## Sampling

Temperature-scaled top-k sampling:
1. K08 softmax over logits (50257-dim)
2. Top-k selection (default k=50) via partial sort on GPU
3. Temperature scaling (default T=0.8)
4. Multinomial sample from filtered distribution

Simple argmax (greedy) mode via `--greedy` flag for deterministic benchmarking.

## Success Criteria

1. Generates coherent GPT-2 text matching HuggingFace reference output (greedy mode, first 20 tokens match)
2. All metrics (TTFT, TPOT, ITL, TPS, KV cache util, VRAM) display correctly in TUI
3. TTFT < 500ms for prompts up to 128 tokens
4. Sustained decode at > 30 tokens/sec (TPOT < 33ms)
5. No OOM up to 1024 context length
6. Clean demo-ready TUI suitable for screen recording

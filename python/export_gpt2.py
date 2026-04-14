#!/usr/bin/env python3
"""Export GPT-2 Small weights to INT8-quantized binary files for SLICK inference."""

import argparse
import json
import os
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
    quantized = np.clip(np.round(weight_fp32 * inv_scales[:, None]), -128, 127).astype(
        np.int8
    )

    # Pack 4 int8 -> 1 int32 (little-endian, matching CUDA __dp4a layout)
    quantized_bytes = quantized.view(np.uint8).reshape(rows, cols // 4, 4)
    packed = (
        quantized_bytes[:, :, 0].astype(np.uint32)
        | (quantized_bytes[:, :, 1].astype(np.uint32) << 8)
        | (quantized_bytes[:, :, 2].astype(np.uint32) << 16)
        | (quantized_bytes[:, :, 3].astype(np.uint32) << 24)
    )
    packed_int32 = packed.view(np.int32)

    return packed_int32, scales


def save_bin(path: str, arr: np.ndarray):
    arr.tofile(path)
    print(f"  {path}: {arr.shape} {arr.dtype} ({arr.nbytes} bytes)")


def save_fp32_weight(layer_dir: str, name: str, weight_fp32: np.ndarray):
    """Save FP32 weight in [K, N] row-major for cuBLAS SGEMM.

    cuBLAS y = x @ W where x is [M, K] and W is [K, N]. Store W as-is.
    """
    save_bin(
        os.path.join(layer_dir, f"{name}_weight_fp32.bin"),
        weight_fp32.astype(np.float32),
    )


def export_model(output_dir: str):
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

    config = {
        "n_layers": 12,
        "n_heads": 12,
        "d_model": 768,
        "d_ff": 3072,
        "vocab_size": 50257,
        "max_seq_len": 1024,
        "d_head": 64,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("Saved config.json")

    wte = sd["transformer.wte.weight"].numpy().astype(np.float32)
    save_bin(os.path.join(output_dir, "wte.bin"), wte)

    wpe = sd["transformer.wpe.weight"].numpy().astype(np.float32)
    save_bin(os.path.join(output_dir, "wpe.bin"), wpe)

    save_bin(
        os.path.join(output_dir, "ln_f_gamma.bin"),
        sd["transformer.ln_f.weight"].numpy().astype(np.float32),
    )
    save_bin(
        os.path.join(output_dir, "ln_f_beta.bin"),
        sd["transformer.ln_f.bias"].numpy().astype(np.float32),
    )

    for i in range(12):
        layer_dir = os.path.join(output_dir, f"layer_{i:02d}")
        os.makedirs(layer_dir, exist_ok=True)
        prefix = f"transformer.h.{i}"

        save_bin(
            os.path.join(layer_dir, "ln1_gamma.bin"),
            sd[f"{prefix}.ln_1.weight"].numpy().astype(np.float32),
        )
        save_bin(
            os.path.join(layer_dir, "ln1_beta.bin"),
            sd[f"{prefix}.ln_1.bias"].numpy().astype(np.float32),
        )

        # QKV: HF stores [768, 2304] row-major. K14 NT layout wants [N=2304, K=768].
        qkv_w = sd[f"{prefix}.attn.c_attn.weight"].numpy().astype(np.float32)
        qkv_wt = qkv_w.T.copy()
        qkv_packed, qkv_scale = quantize_per_row(qkv_wt)
        save_bin(os.path.join(layer_dir, "qkv_weight.bin"), qkv_packed)
        save_bin(os.path.join(layer_dir, "qkv_scale.bin"), qkv_scale)
        save_bin(
            os.path.join(layer_dir, "qkv_bias.bin"),
            sd[f"{prefix}.attn.c_attn.bias"].numpy().astype(np.float32),
        )
        save_fp32_weight(layer_dir, "qkv", qkv_w)

        out_w = sd[f"{prefix}.attn.c_proj.weight"].numpy().astype(np.float32)
        out_wt = out_w.T.copy()
        out_packed, out_scale = quantize_per_row(out_wt)
        save_bin(os.path.join(layer_dir, "out_weight.bin"), out_packed)
        save_bin(os.path.join(layer_dir, "out_scale.bin"), out_scale)
        save_bin(
            os.path.join(layer_dir, "out_bias.bin"),
            sd[f"{prefix}.attn.c_proj.bias"].numpy().astype(np.float32),
        )
        save_fp32_weight(layer_dir, "out", out_w)

        save_bin(
            os.path.join(layer_dir, "ln2_gamma.bin"),
            sd[f"{prefix}.ln_2.weight"].numpy().astype(np.float32),
        )
        save_bin(
            os.path.join(layer_dir, "ln2_beta.bin"),
            sd[f"{prefix}.ln_2.bias"].numpy().astype(np.float32),
        )

        up_w = sd[f"{prefix}.mlp.c_fc.weight"].numpy().astype(np.float32)
        up_wt = up_w.T.copy()
        up_packed, up_scale = quantize_per_row(up_wt)
        save_bin(os.path.join(layer_dir, "ffn_up_weight.bin"), up_packed)
        save_bin(os.path.join(layer_dir, "ffn_up_scale.bin"), up_scale)
        save_bin(
            os.path.join(layer_dir, "ffn_up_bias.bin"),
            sd[f"{prefix}.mlp.c_fc.bias"].numpy().astype(np.float32),
        )
        save_fp32_weight(layer_dir, "ffn_up", up_w)

        down_w = sd[f"{prefix}.mlp.c_proj.weight"].numpy().astype(np.float32)
        down_wt = down_w.T.copy()
        down_packed, down_scale = quantize_per_row(down_wt)
        save_bin(os.path.join(layer_dir, "ffn_down_weight.bin"), down_packed)
        save_bin(os.path.join(layer_dir, "ffn_down_scale.bin"), down_scale)
        save_bin(
            os.path.join(layer_dir, "ffn_down_bias.bin"),
            sd[f"{prefix}.mlp.c_proj.bias"].numpy().astype(np.float32),
        )
        save_fp32_weight(layer_dir, "ffn_down", down_w)

        print(f"Layer {i}: done")

    import shutil
    from huggingface_hub import hf_hub_download

    shutil.copy(
        hf_hub_download("gpt2", "vocab.json"), os.path.join(output_dir, "vocab.json")
    )
    shutil.copy(
        hf_hub_download("gpt2", "merges.txt"), os.path.join(output_dir, "merges.txt")
    )

    print(f"\nExport complete: {output_dir}")
    total_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fnames in os.walk(output_dir)
        for f in fnames
    )
    print(f"Total size: {total_bytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="models/gpt2-int8",
        help="Output directory for exported weights",
    )
    args = parser.parse_args()
    export_model(args.output)

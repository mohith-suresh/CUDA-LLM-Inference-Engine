#!/usr/bin/env python3
"""Benchmark HuggingFace GPT-2 inference on GPU — baseline for SLICK comparison."""

import argparse
import json

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def benchmark(prompt: str, max_tokens: int, device: str = "cuda") -> dict:
    print(f"Loading GPT-2 Small on {device}...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    print(f"Prompt ({prompt_len} tokens): {prompt}")

    # Warmup
    with torch.no_grad():
        _ = model(input_ids, use_cache=True)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    first_token_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    generated_tokens: list[int] = []
    itl_samples: list[float] = []

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
        for _ in range(max_tokens - 1):
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

    ttft_ms = start_event.elapsed_time(first_token_event)
    total_ms = start_event.elapsed_time(end_event)
    num_tokens = len(generated_tokens)
    tpot_ms = total_ms / num_tokens if num_tokens > 0 else 0.0
    tps = num_tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0

    itl_min = min(itl_samples) if itl_samples else 0.0
    itl_max = max(itl_samples) if itl_samples else 0.0
    itl_sorted = sorted(itl_samples)
    itl_p95 = (
        itl_sorted[int(len(itl_sorted) * 0.95)] if len(itl_sorted) >= 20 else itl_max
    )

    vram_used = torch.cuda.memory_allocated() / 1024 / 1024
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024

    generated_text = tokenizer.decode(generated_tokens)

    print(f"\n\n{'=' * 50}")
    print("PyTorch GPT-2 FP32 Baseline")
    print(f"{'=' * 50}")
    print(f"Generated: {prompt}{generated_text}")
    print(f"Tokens:       {num_tokens}")
    print(f"TTFT:         {ttft_ms:.1f} ms")
    print(f"TPOT:         {tpot_ms:.1f} ms")
    print(f"TPS:          {tps:.1f}")
    print(f"ITL P95:      {itl_p95:.1f} ms")
    print(f"ITL min/max:  {itl_min:.1f}/{itl_max:.1f} ms")
    print(f"End-to-end:   {total_ms:.1f} ms")
    print(f"VRAM:         {vram_used:.0f}/{vram_total:.0f} MB")

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
    print("\nResults saved to pytorch_baseline.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", default="The meaning of life is", help="Input prompt"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="Maximum tokens to generate"
    )
    args = parser.parse_args()
    benchmark(args.prompt, args.max_tokens)

#!/bin/bash
# Three-way GPT-2 inference comparison:
#   PyTorch FP32 baseline  vs  SLICK cuBLAS FP32  vs  SLICK INT8
set -e

PROMPT="${1:-The meaning of life is}"
MAX_TOKENS="${2:-64}"
MODEL_DIR="${3:-models/gpt2-int8}"

echo "========================================================"
echo " SLICK — Three-Way Inference Comparison"
echo " Prompt: \"$PROMPT\""
echo " Max tokens: $MAX_TOKENS"
echo " Hardware: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "========================================================"
echo ""

# 1. PyTorch FP32 baseline
echo ">>> [1/3] PyTorch FP32 baseline"
echo "--------------------------------------------------------"
uv run python python/benchmark_pytorch.py \
    --prompt "$PROMPT" --max-tokens "$MAX_TOKENS"
echo ""

# 2. SLICK cuBLAS FP32 backend
echo ">>> [2/3] SLICK cuBLAS FP32 backend"
echo "--------------------------------------------------------"
./build/slick --bench --temperature 0.8 --top-k 40 --backend cublas \
    --model "$MODEL_DIR" --prompt "$PROMPT" --max-tokens "$MAX_TOKENS"
echo ""

# 3. SLICK INT8 (optimized kernels)
echo ">>> [3/3] SLICK INT8 (optimized)"
echo "--------------------------------------------------------"
./build/slick --bench --temperature 0.8 --top-k 40 --backend int8 \
    --model "$MODEL_DIR" --prompt "$PROMPT" --max-tokens "$MAX_TOKENS"
echo ""

# Comparison table from JSON
echo "========================================================"
echo " COMPARISON TABLE"
echo "========================================================"
python3 - <<'PYEOF'
import json, os

files = ['pytorch_baseline.json', 'cublas_fp32_results.json', 'slick_int8_results.json']
backends = []
for f in files:
    if os.path.exists(f):
        with open(f) as fh:
            backends.append(json.load(fh))

if not backends:
    print('No result files found.')
    raise SystemExit

print(f'{"Metric":<20} ', end='')
for b in backends:
    print(f'{b["backend"]:>18} ', end='')
print()
print('-' * (20 + 19 * len(backends)))

rows = [
    ('ttft_ms', 'TTFT', 'ms'),
    ('tpot_ms', 'TPOT', 'ms'),
    ('tps', 'TPS', 'tok/s'),
    ('itl_p95_ms', 'ITL P95', 'ms'),
    ('end_to_end_ms', 'End-to-end', 'ms'),
    ('vram_mb', 'VRAM', 'MB'),
    ('tokens', 'Tokens', ''),
]
for key, label, unit in rows:
    name = f'{label} ({unit})' if unit else label
    print(f'{name:<20} ', end='')
    for b in backends:
        val = b.get(key, '--')
        print(f'{val:>18} ', end='')
    print()

if len(backends) >= 2 and backends[0].get('backend') == 'pytorch_fp32':
    print('-' * (20 + 19 * len(backends)))
    pt_tps = backends[0].get('tps', 0) or 1e-9
    print(f'{"Speedup vs PyTorch":<20} ', end='')
    for b in backends:
        speedup = (b.get('tps', 0) or 0) / pt_tps
        print(f'{speedup:>17.2f}x ', end='')
    print()
PYEOF

# Cleanup
rm -f pytorch_baseline.json cublas_fp32_results.json slick_int8_results.json

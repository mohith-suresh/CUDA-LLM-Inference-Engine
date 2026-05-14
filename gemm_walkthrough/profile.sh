#!/usr/bin/env bash
# Profile the GEMM kernels with Nsight Systems + Nsight Compute and export
# everything to text/CSV that can be analyzed off-box.
#
#   ./profile.sh            # size 2048
#   ./profile.sh 1024       # smaller = faster ncu replay
#   SUDO=sudo ./profile.sh  # if ncu needs perf-counter permission (ERR_NVGPUCTRPERM)
#
# Afterward, copy the whole  prof/  directory back:
#   scp -r <user>@<vm>:.../gemm_walkthrough/prof ./gemm_walkthrough/
set -euo pipefail

SIZE="${1:-2048}"
SUDO="${SUDO:-}"
OUT=prof
mkdir -p "$OUT"

# Kernel-name filter (our kernels only — skips cuBLAS internals).
KREGEX="regex:naive_kernel|coalesced_kernel|shared_kernel|blocktiling_1d_kernel|blocktiling_2d_kernel|vectorized_kernel|bank_conflicts_kernel|bank_extra_col_kernel|warptiling_kernel|double_buffered_kernel"

echo "=== Building ==="
make gemm profile

echo
echo "=== Nsight Systems (timeline + kernel-time summary) on ./gemm $SIZE ==="
nsys profile -f true -o "$OUT/gemm_sys" ./gemm "$SIZE"
# Kernel-time summary, per-launch trace, and API overhead -> CSV.
nsys stats --report cuda_gpu_kern_sum --format csv --output "$OUT/sys" "$OUT/gemm_sys.nsys-rep" || \
nsys stats --report gpukernsum       --format csv --output "$OUT/sys" "$OUT/gemm_sys.nsys-rep" || true
nsys stats --report cuda_gpu_trace   --format csv --output "$OUT/sys" "$OUT/gemm_sys.nsys-rep" || true
nsys stats --report cuda_api_sum     --format csv --output "$OUT/sys" "$OUT/gemm_sys.nsys-rep" || true

echo
echo "=== Nsight Compute (per-kernel, full metric set) on ./profile $SIZE ==="
echo "    (this replays each kernel many times; use ./profile.sh 1024 if slow)"
$SUDO ncu --set full \
     --kernel-name-base demangled \
     -k "$KREGEX" \
     -f -o "$OUT/gemm_ncu" \
     ./profile "$SIZE"

echo
echo "=== Exporting ncu report to CSV ==="
# Raw = every metric per kernel (best for analysis); details = the sectioned view.
ncu --import "$OUT/gemm_ncu.ncu-rep" --csv --page raw     > "$OUT/gemm_ncu_raw.csv"
ncu --import "$OUT/gemm_ncu.ncu-rep" --csv --page details > "$OUT/gemm_ncu_details.csv"

echo
echo "Done. Everything is in ./$OUT/ :"
ls -la "$OUT"
echo
echo "Copy it back with:  scp -r <user>@<vm>:\$PWD/$OUT ./gemm_walkthrough/"

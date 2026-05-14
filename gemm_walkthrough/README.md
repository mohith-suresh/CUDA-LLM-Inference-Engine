# GEMM Walkthrough — a clean re-implementation of the CUDA-MMM ladder

A from-scratch SGEMM optimization ladder, following the steps in Simon Boehm's
["How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance"](https://siboehm.com/articles/22/CUDA-MMM),
rewritten for **readability**: consistent naming, one idea per file, a comment on
every non-obvious line, and a cuBLAS-validated benchmark harness.

Every kernel computes true **GEMM**:

```
C = alpha * (A @ B) + beta * C      (row-major, A[M,K], B[K,N], C[M,N])
```

Each step changes exactly one thing relative to the previous.

## Build & run

```bash
make                 # builds ./gemm and ./autotune for sm_75 (Turing: T4, 1650 Ti)
make run             # build + run the ladder (sizes 1024, 2048, 4096)
./gemm 2048          # single size
make ARCH=sm_86 run  # A10 / Ampere
make autotune-run    # step 9: sweep tile configs, print the winner
```

Requires CUDA + cuBLAS. The harness checks correctness with **alpha=1, beta=0.5**
and a **random initial C** (so the beta path is actually exercised, not bypassed),
then times each kernel at the standard alpha=1, beta=0.

## The steps

| File | Boehm # | The one change | Idea |
|------|--------|----------------|------|
| `01_naive.cuh`          | 1  | `threadIdx.x -> row` | 1 thread / output, no reuse |
| `02_coalesced.cuh`      | 2  | `threadIdx.x -> col` | warp hits consecutive addresses |
| `03_shared_memory.cuh`  | 3  | stage tiles in smem  | reuse each element `TILE`× |
| `04_blocktiling_1d.cuh` | 4  | `TM` outputs/thread  | one B value feeds `TM` FMAs (registers) |
| `05_blocktiling_2d.cuh` | 5  | `TM×TN` micro-tile   | outer product: `TM*TN` FMAs / `TM+TN` loads |
| `06_vectorized.cuh`     | 6  | float4 + transposed As | wider loads, conflict-free A |
| `07_bank_conflicts.cuh` | 7  | swizzled `Bs` layout | make the `regB` read stride-1 → conflict-free |
| `08_bank_extra_col.cuh` | 8  | pad `Bs` rows by 5   | shift bank alignment (simpler alternative to 7) |
| `09_autotune.cu`        | 9  | sweep tile configs   | pick the best params for this GPU (separate driver) |
| `10_warptiling.cuh`     | 10 | add a warp-level tile | make all 3 hierarchy levels explicit |
| `11_double_buffered.cuh`| 11 | 2× smem, prefetch    | overlap next-tile load with compute |

## Conventions (readable + standard, transferable to other kernels)

- **`blockRow` / `blockCol`** name `blockIdx.y` / `blockIdx.x`. Each matrix
  pointer is then advanced to the **top-left corner of this block's tile**
  (`A += blockRow*BM*K`, etc.), so all later indexing is relative to the tile —
  the standard CUTLASS/cuBLAS base-pointer idiom.
- **`threadRow` / `threadCol`** are derived from a **1D `threadIdx.x`** via `/`
  and `%`. From step 4 on, the load-A, load-B, and compute phases tile the same
  threads into *different* 2D shapes, so a 1D index (reshaped per phase) is
  cleaner than a fixed 2D block. Step 3 keeps a 2D block because all its phases
  share one 32×32 shape.
- **`loadRowA/loadColA`, `regA/regB`, `acc`** — consistent across every file.
- Row-major addressing everywhere: element `(r,c)` of an `R×C` matrix is `r*C + c`.

## What was changed from the article's code (for clarity)

- Descriptive names instead of `cRow`, `dotIdx`, `innerRowA`, `tmp`.
- One concept per file, each with a header comment stating the single change.
- Consistent skeleton across kernels: *load → sync → compute → sync → store*.
- cuBLAS reference wrapped in `common.cuh` with the row-major↔column-major
  transpose identity fully explained.

## Assumptions

- Steps 3–11 assume `M`, `N`, `K` are **multiples of the tile sizes** (true for
  1024/2048/4096), which removes boundary `if`s so the algorithm reads cleanly.
  Steps 1–2 keep bounds checks since they are the introductory kernels.
- Steps 7 & 8 are specialized to the 128/128/8/8/8 tile (TN=8, BN/TN=16), matching
  step 6; they are ported faithfully from Boehm's source.
- Step 10's default tile config is Boehm's A6000 tuning (128 accumulators/thread).
  On smaller GPUs it may spill registers — shrink `WM/WN/TM/TN` in `run_warptiling`.
  Correctness is unaffected.

## Note on numbers

Absolute GFLOP/s depends entirely on the GPU. The **shape** of the ladder — the
big jumps at coalescing and 1D blocktiling, the crossover to compute-bound around
2D blocktiling, and diminishing returns after vectorization — is what to study.
Bank-conflict gains (7/8) and warptiling (10) were large on Boehm's A6000; measure
them on your own GPU with `ncu`.

The public README headline uses the cleaned A10 result from `WALKTHROUGH.md`:
the 10h+ split round-robin ablation reaches **96.5% of cuBLAS** at 4096³
(about 12.8 TFLOP/s, 10.8 ms). Raw Nsight Compute reports are intentionally not
checked in; the walkthrough keeps the reproducible source plus the summarized
measurements.

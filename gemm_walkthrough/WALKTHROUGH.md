# SGEMM Optimization Walkthrough — technique · profiler observation · result

One section per kernel: **what the technique is**, **the Nsight Compute observation that motivated the next
step**, and **the measured result** (GFLOP/s · % of cuBLAS · kernel duration). This is the "improve one at a
time" logic — each step targets exactly the metric the profiler flagged, then re-profiles and watches the
bottleneck move.

- **Benchmark:** `C = alpha·(A@B) + beta·C`, FP32, **4096³**, NVIDIA **A10**, validated vs cuBLAS (all PASS,
  rel_err ~1e-6). cuBLAS reference = **13,238 GFLOP/s ≈ 10.38 ms**.
- **Duration** = wall-clock kernel time = `2·4096³ / GFLOP·s⁻¹` (lower is the whole point). It's the metric
  that matters — GFLOP/s and duration are the same story inverted.

## Summary — the ladder at a glance
| step | kernel | GFLOP/s | % cuBLAS | **duration** | ×faster vs prev |
|---|---|---|---|---|---|
| 1 | Naive | 233.8 | 1.8% | **587.9 ms** | — |
| 2 | Coalesced | 1,445.9 | 10.9% | **95.1 ms** | ×6.2 |
| 3 | Shared Memory | 2,032.5 | 15.4% | **67.6 ms** | ×1.41 |
| 4 | 1D Blocktiling | 5,280.4 | 39.9% | **26.0 ms** | ×2.60 |
| 5 | 2D Blocktiling | 8,363.1 | 63.2% | **16.4 ms** | ×1.58 |
| 6 | Vectorized (float4) | 11,675.9 | 88.2% | **11.8 ms** | ×1.40 |
| 7 | Bank Conflicts (swizzle) | 10,467.9 | 79.1% | 13.1 ms | ×0.90 (regress) |
| 8 | Bank Conflicts (padding) | 10,659.2 | 80.5% | 12.9 ms | — |
| 10 | **Warptiling** | **12,695.0** | **95.9%** | **10.8 ms** | ×1.09 |
| 11 | Double Buffered | 11,667.0 | 88.1% | 11.8 ms | ×0.92 (regress) |
| 0 | cuBLAS (reference) | 13,238.6 | 100% | 10.4 ms | — |

**Cumulative: naive → warptiling = ×54 faster (587.9 ms → 10.8 ms).**

---

## Step 1 — Naive · `01_naive.cuh`
- **Technique:** one thread per output C[i][j], loops over K reading A and B straight from global memory.
  `threadIdx.x` maps to the *row* → a warp reads scattered columns.
- **Profiler observation:** Memory SOL ~99%, but drilling in → **L1/TEX ~99% while DRAM ~2.4%** (memory-bound,
  not bandwidth-bound); **bytes/sector = 4/32** (only 1/8 of each transaction used); top stall **LG Throttle**;
  "Uncoalesced Global Accesses" in Source Counters. → the classic uncoalesced signature.
- **Result:** 233.8 GFLOP/s · **1.8%** · **587.9 ms.**
- **→ Next:** map `threadIdx.x → column` so a warp reads contiguous addresses (coalesce).

## Step 2 — Global memory coalescing · `02_coalesced.cuh`
- **Technique:** remap threads so consecutive threads read consecutive columns → contiguous 128-byte loads.
- **Profiler observation:** `bytes/sector 4/32 → ~26/32`, **sectors/request 32 → 2.5** (A-broadcast[1] +
  B-coalesced[4]). But now **Compute SOL ≈ Memory SOL ≈ Mem Pipes Busy ≈ 95%** (LSU/L1-pipe bound) and
  **L1/TEX hit rate 95%**, DRAM idle (13%). The 95% L1 hit = the loads are *redundant re-reads*; per-access is
  already optimal.
- **Result:** 1,445.9 GFLOP/s · **10.9%** · **95.1 ms** (×6.2 — the single biggest jump).
- **→ Next:** can't improve a load, so cut the *count* → stage tiles in shared memory.

## Step 3 — Shared-memory tiling · `03_shared_memory.cuh`
- **Technique:** cooperatively load a BK-wide tile of A and B into shared memory once, `__syncthreads`,
  compute the partial products from shared; slide across K.
- **Profiler observation:** DRAM traffic collapses, but still **LSU/shared-bound**; the **FMA pipe is nearly
  idle** and the roofline sits at ~15% of peak. Each thread still produces one output → arithmetic intensity
  ≈ 1 (one FMA per shared-load pair). The FMA units starve while the LSU does all the work.
- **Result:** 2,032.5 GFLOP/s · **15.4%** · **67.6 ms** (×1.41).
- **→ Next:** raise reuse — compute multiple outputs per thread (register tiling).

## Step 4 — 1D block tiling · `04_blocktiling_1d.cuh`
- **Technique:** each thread computes **TM outputs** (a register column); load one B value, reuse it across
  TM FMAs.
- **Profiler observation:** **Compute SOL = Memory SOL = Mem Pipes Busy = 68** (four-way tie → **LSU-pipe
  bound**); **FMA pipe only ~21%**; Memory Tables show **Shared-Load instructions 100.6M vs Global 8.5M
  (~12×)** — the LSU is saturated by shared loads (~1.8 FMA per load), still too load-heavy.
- **Result:** 5,280.4 GFLOP/s · **39.9%** · **26.0 ms** (×2.60).
- **→ Next:** an outer product gives TM·TN FMAs for TM+TN loads → drain the LSU.

## Step 5 — 2D block tiling · `05_blocktiling_2d.cuh`
- **Technique:** each thread computes a **TM×TN register micro-tile** via an outer product (loads TM of A +
  TN of B per k, does TM·TN FMAs).
- **Profiler observation:** LSU drained — **Mem Pipes Busy 68 → 47**, highest pipe flips **LSU → FMA (37%)**.
  But the top bound is now **Memory SOL = Mem Busy 76** (internal shared datapath), with a **3.2-way bank
  conflict** and **48% excess wavefronts**; DRAM idle; occupancy ~30% (96 regs) but ILP hides it
  (Warp-Cycles/Issued-Instr 6.99).
- **Result:** 8,363.1 GFLOP/s · **63.2%** · **16.4 ms** (×1.58).
  *(Isolation kernels `05b` transpose-As and `05c` conflict-free bump this to ~70% — see those files.)*
- **→ Next:** still too many *scalar* memory instructions on the LSU → 128-bit float4 loads to cut the count ¼.

## Step 6 — Vectorized (float4) + transposed As · `06_vectorized.cuh`
- **Technique:** load A/B with `float4` (128-bit) — one `LD.128` replaces four `LD.32`; transpose As in shared
  so the loads stay contiguous.
- **Profiler observation:** **Mem Pipes Busy 51 → 33** (LSU instruction count cut — float4's whole job),
  **Compute SOL 62**, **IPC 2.0 → 2.5**. But **Mem Busy still 82** (float4 does *not* fix bank conflicts,
  40% excess wf), the **global store is 16/32** (uncoalesced), and the **FMA pipe is only 62%** (not saturated).
  *(Note the DRAM-throughput rose 9.6→15.1, but that's pure time-compression — coalescing was already optimal.)*
- **Result:** 11,675.9 GFLOP/s · **88.2%** · **11.8 ms** (×1.40).
- **→ Next:** FMA still underfed → warp-level tile (reuse + conflict-free warp reads + occupancy sweet spot).

## Steps 7 & 8 — Resolve bank conflicts · `07_bank_conflicts.cuh` (swizzle), `08_bank_extra_col.cuh` (padding)
- **Technique:** remove the shared-memory bank conflicts by swizzling the Bs layout (7) or padding an extra
  column (8).
- **Profiler observation / result:** **79–80% · ~13 ms — a *regression* vs step 6 (88%).** At 4096, float4's
  instruction-count win already dominates the conflict cost, and these older-structure kernels don't carry it
  forward. Instructive negative result: fixing conflicts is *not* always the highest-leverage move.

## Step 10 — Warptiling · `10_warptiling.cuh`
- **Technique:** a **three-level tile** (block → 64×64 warp tile → thread), so the 32 lanes of a warp reuse
  shared fragments and their reads are coalesced broadcasts.
- **Profiler observation:** **Compute SOL 71 (FMA pipe 69%, now the top)**, **Mem Busy 82 → 57** (far fewer
  conflicts), global **store 32/32**, roofline **65%**, **occupancy 24% (168 regs, 3 blocks/SM)**, IPC 2.84.
  The FMA pipe is finally the limiter — the mark of a well-optimized compute-bound GEMM. Wins at only 24%
  occupancy → **ILP + reuse, not occupancy.**
- **Result:** 12,695.0 GFLOP/s · **95.9%** · **10.8 ms** (×1.09) — essentially matching cuBLAS (10.4 ms).

## Step 11 — Double buffering · `11_double_buffered.cuh`
- **Technique:** two shared buffers — prefetch the next K-tile while computing the current one (software
  pipelining).
- **Profiler observation / result:** **88.1% · 11.8 ms — a *regression* vs warptiling.** The kernel is
  compute-bound with idle DRAM → there's no global-load latency to overlap, so the prefetch buys nothing and
  the extra shared memory hurts. Another instructive negative: latency-hiding only pays off when latency-bound.

---

## Beyond the standard ladder — reverse-engineering warptiling (custom ablation)
These kernels (`10b`–`10h`, `12`) rebuild warptiling's advantage from a plain flat register-tiled kernel,
one lever at a time, and the endpoint **beats it**.

| kernel | technique | %cuBLAS | duration | driving metric |
|---|---|---|---|---|
| `10b` Flat 8×16 | baseline flat tile | 68.3% | 15.2 ms | uncoalesced store 16/32 + 11-way read conflict |
| `10d` interleaved | interleave cols → coalesced store | 83.2% | 12.5 ms | store 32/32 + conflict-free reads |
| `10g` Flat 32×4 | 1-float4-wide tile | 90.4% | 11.5 ms | native coalescing (but strip: 36 loads/k) |
| `10h` Split 16×8 | square-ish reuse | 93.0% | 11.2 ms | **Mem Pipes 35→26** (24 loads/k, FMA/load 3.56→5.33) |
| `10h+` Split, 3 blk/SM | `__launch_bounds__(128,3)` | **96.5%** | **10.8 ms** | **registers 193→168 (0 spills) → occ 16.7→24%** |
| `12` warptile + cp.async | real 2-stage cp.async DB | 85.6% | 12.4 ms | compute-bound → nothing to overlap; regress |

**The decomposition:** `68.3% (flat) → 83.2% (coalesced stores) → 90.4% (1-float4-wide tile) → 93.0%
(16×8 reuse) → 96.5% (register cap → 3 blocks/SM)`. The controlled ablations show that mapping, reuse,
and occupancy are separate levers — and warptiling is simply the corner that gets them right.

---

## The one-line method (every step)
> Read the **SOL bars** → find the saturated unit → confirm in its section (Memory Workload / Warp Stalls /
> Instruction Mix) → the fix targets *that unit* → re-profile and watch the bottleneck **move**. It marched:
> DRAM-uncoalesced (L1) → LSU/L1 pipe (redundant loads) → LSU pipe (shared loads) → Mem Busy (bank conflicts)
> → FMA-underfed (reuse/occupancy). Never a generic "AI is low."

*Results source: A10 local profiling run summarized here. Raw Nsight Compute CSV/`.ncu-rep` artifacts are not
checked in; durations are wall-clock, computed from the measured GFLOP/s.*

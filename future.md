# Future Optimizations — K11/K12 PagedAttention

Current: 97 regs, 25KB smem, 2 blocks/SM, 0 spills. Peak 1.43 TFLOPS (33% FP32).

## 1. Double-Buffer KV Blocks (~10-15%)

Load the next physical block's K while computing S=Q@K^T on the current block. The `__syncthreads()` after K load and before V load serializes load+compute phases. Use 2x KV_smem buffers and alternate: compute S with buffer[i%2] while loading K into buffer[(i+1)%2].

**Risk:** Doubles KV_smem from 4KB to 8KB (total ~29KB). Still fits 2 blocks/SM under 64KB smem config (2x29KB=58KB < 64KB). Register count may increase from loop state tracking — verify no spills.

## 2. P_smem Elimination via Warp Shuffle

Replace P_smem[64][17] write+read with `__shfl_sync` broadcast: each thread holds P[tm][tc] in S registers, broadcast to all 16 columns via `p = __shfl_sync(0xFFFFFFFF, S[tm][0], half_leader + k)`.

**Tested and reverted.** Saves 4KB smem (25KB→21KB) but changes occupancy from 2→3 blocks/SM on SM75, which slowed the kernel by 10-15%. Would need to be combined with `__launch_bounds__(256, 2)` or smem padding to maintain 2 blocks/SM. The compiler also used 17 fewer registers (97→80) with shorter S live ranges, contributing to the occupancy shift.

**Approach if revisited:** Use `__launch_bounds__(PA_NTHREADS, 2)` to hint the compiler to target 2 blocks/SM, which would cap register usage at 128/thread and prevent 3-block scheduling.

## 3. Block Table Prefetch for Multi-Batch

The B=2 config shows 15% regression vs K10. The block table indirection (`bt[blk_idx]`) adds a dependent global memory load before each KV block. Prefetch `bt[blk_idx+1]` into a register at the start of each iteration to overlap the indirection latency with S computation.

```cuda
int next_phys = (blk_idx + 1 < num_kv_blocks) ? bt[blk_idx + 1] : 0;
// ... compute with phys_block ...
phys_block = next_phys;
```

## 4. GQA Cooperative KV Loading

For GROUP_SIZE=8, 8 Q-head CTAs independently load identical KV blocks. L2 cache provides some reuse (~5% speedup at GROUP_SIZE 1→8), but explicit scheduling could guarantee it:

- **Persistent kernel:** Launch one CTA per SM, loop over (batch, q_head, q_tile) work items. Q heads sharing a KV head are processed back-to-back on the same SM, guaranteeing L2/smem locality.
- **CTA clustering (SM90+):** Not available on SM75/Turing.

**Impact on SM75:** Marginal — Turing L2 is only 2MB, and the kernel is compute-bound at the tile level. More impactful on Ampere/Hopper with larger L2 caches.

## 5. Software-Pipelined GEMM (ILP)

Preload k+1 fragments while computing with k in both S=Q@K^T and O+=P@V loops. Pattern:

```cuda
float q_next[TM], k_next;
// prefetch k=0
for (int k = 0; k < d; ++k) {
    // prefetch k+1
    q_next = Q_smem[...][k+1];
    k_next = KV_smem[...][k+1];
    // compute with k
    S += q_cur * k_cur;
    q_cur = q_next; k_cur = k_next;
}
```

Estimated ~10-15% from overlapping smem loads with FMAs. Same technique described in K10b optimization notes.

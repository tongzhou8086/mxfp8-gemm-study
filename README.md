# 12 - Production B200 GEMM kernels (before the CuTe rewrite)

This chapter captures **two production BF16 GEMM kernels for B200 (Blackwell,
SM100)**, rendered from the `mmcomposer` code generator, as the *reference
point* we will re-implement in CuTe DSL. Everything here is hand-tuned CUDA C++
with raw PTX; the goal of the next chapters is to reproduce this behavior in
CuTe and compare.

Both kernels are the same warp-specialized, persistent, 2-CTA-cluster design.
They differ in exactly three knobs — `BN`, `NS`, and the TMEM buffering mode —
and those three are not independent: `BN` decides how many TMEM accumulators
fit, and the SMEM budget then caps `NS`.

| Kernel | `BN` | `NS` | TMEM accumulators | TMA store buffers | dir |
|--------|-----:|-----:|-------------------|------------------:|-----|
| **Double-TMEM** | 256 | 6 | **2** (double-buffer) | 2 | `bn256-double-tmem/` |
| **Single-TMEM** | 512 | 4 | **1** (single-buffer) | 2 | `bn512-single-tmem/` |

Each directory holds a fully-specialized, branch-free `kernel.cu` (no `#if`, no
knob `if constexpr` — reads like hand-written code for that one combo) plus a
self-contained `launcher.py`.

There is also a first ThunderKittens API port:

- `bn256-double-tmem-tk/` keeps the BN256/NS6 persistent schedule and launch ABI,
  but moves the low-level tcgen05/TMEM/sync utilities to TK wrappers so it can be
  benchmarked against the raw baseline locally.
- `bn256-double-tmem-tk-tma/` is the next BN256 port. It keeps the same schedule,
  but also moves the A/B TMA loads and D TMA stores to TK `gl`/`st` TMA
  primitives.
- `bn256-double-tmem-tk-tma-managed/` is the same TK-TMA port, but uses TK's
  default managed `tensor_allocator` so TMEM allocation/deallocation is handled
  by the allocator constructor/destructor.

## Measured baseline (B200, `sm_100a`, BF16, M=N=K=4096)

Compiled by the launchers with `nvcc` and benchmarked on a B200. These are the
**reference targets** for the CuTe rewrite:

| Kernel | rel err | time | throughput |
|--------|--------:|-----:|-----------:|
| **BN256 double-TMEM** (NS6, 2 store buf) | 0.29% | 91.1 µs | **1508.6 TFLOPS** |
| **BN512 single-TMEM** (NS4, 2 store buf) | 0.29% | 97.4 µs | **1411.4 TFLOPS** |

At 4096³ (a small-/medium-K, overlap-limited shape) the double-buffered BN256
kernel wins, exactly as the presentation predicts: the next-tile TMEM overlap
beats BN512's higher arithmetic intensity. BN512 pulls ahead only at large K,
where its 171 flop/byte AI amortizes the single-buffer drain stall.
(Both `launcher.py` are pinned to M=N=K=4096; edit the shape loop near the
bottom to sweep other sizes.)

## What "production kernel" means here

Both kernels share the full production feature set:

- **Persistent grid** (`PERSISTENT=1`) — grid = #SMs, each CTA walks a run of output tiles.
- **2-CTA cluster MMA** (`TWO_CTA=1`, `cta_group::2`) — the MMA spans `M = 2×128` across the cluster; each CTA owns only `BN/2` columns of B, halving its B SMEM.
- **TMA load** — `cp.async.bulk.tensor.2d …cta_group::2` into an `NS`-slot SMEM ring, 128B-swizzled for conflict-free MMA reads.
- **tcgen05 MMA into TMEM** — `tcgen05.mma.cta_group::2.kind::f16`, operands named by SMEM matrix descriptors.
- **Pipelined TMA-store epilogue** (`EPILOGUE_TMA_PIPELINED=1`, `TMA_STORE_STAGES=2`) — drain TMEM→regs (`tcgen05.ld`), downcast fp32→bf16 into a swizzled `128×64` SMEM chunk, then `cp.async.bulk` store; 2 store buffers so one chunk's store overlaps the next chunk's SMEM write.
- **Epilogue overlaps the next tile's compute** (`EPILOGUE_OVERLAP=1`) — see below.

## Warp specialization (identical in both)

`LAUNCH_THREADS = (NUM_WARPS + 4) × 32`. Four "stream" warps + `NUM_WARPS`
epilogue warps (here `NUM_WARPS=4`). Each role is a wait→work→signal loop, and
the mbarriers *are* the signal arrows between them:

| Warp | ID | Waits on | Work | Signals |
|------|----|----------|------|---------|
| **TMA load** | `warp_id == 0` (elected lane) | `mbar_compute_buffer_free[slot]` | TMA-load A + B into ring slot | `mbar_compute_data_ready[slot]` |
| **MMA** | `cta_rank==0 && warp_id==1` (elected lane) | `mbar_compute_data_ready[slot]` | `tcgen05.mma` chain into TMEM | `mbar_compute_buffer_free[slot]`, `mbar_tmem_data_ready[buf]` |
| **Epilogue** | `4 ≤ warp_id < NUM_WARPS+4` | `mbar_tmem_data_ready[buf]` | drain TMEM→SMEM→HBM (TMA store) | `mbar_tmem_buffer_free[buf]` |

Signaling uses async hardware phase-flips: the TMA engine flips
`compute_data_ready` when the expected byte count lands; `tcgen05.commit` flips
`tmem_data_ready` when the tile's MMAs retire. Phase parity is tracked per slot
in `compute_*_phase[NS]` / `tmem_*_phase[buf]` arrays.

## The one real difference: TMEM double- vs single-buffering

B200 TMEM is a fixed **128 × 512 FP32** array per SM. The accumulator width in
TMEM columns equals `BN`:

- **`BN=256`** uses 256 of 512 columns → **two accumulators fit** →
  double-buffer (`buf 0` / `buf 1`). The MMA fills the *next* tile into `buf 1`
  while the epilogue drains the *previous* tile from `buf 0` → engines fully
  overlap. `SINGLE_TMEM_ACCUM=0`, and `mbar_tmem_*[2]`.
- **`BN=512`** uses all 512 columns → **only one accumulator fits** →
  single-buffer. Every tile pays a fixed TMEM-reuse delay: the MMA must wait for
  the epilogue to drain the sole buffer before starting the next tile.
  `SINGLE_TMEM_ACCUM=1`, and `mbar_tmem_*[1]`.

Trade-off (from the presentation): `BN=512` has higher arithmetic intensity
(`AI = 256·BN/(256+BN)` → 171 vs 128 flop/byte for BN256) but loses the
next-tile overlap, so it wins on **large K** (delay amortized) while the
double-buffered `BN=256` wins on **small/medium K** (overlap-limited).

## Sizing the SMEM ring (why NS=6 for BN256, NS=4 for BN512)

Dynamic SMEM holds the `NS`-slot compute ring **plus** the disjoint TMA-store
staging (disjoint because, with overlap, the store of tile *i* runs while the
K-loop of tile *i+1* is filling the ring). B200 usable SMEM ≈ **227 KB/CTA**.

Per 2-CTA slot: `A = BM·BK·2 = 16 KB` + `B_local = (BN/2)·BK·2`.

| Kernel | slot | ring `NS×slot` | store `2×16 KB` | total |
|--------|-----:|---------------:|----------------:|------:|
| BN256 | 32 KB | 6 × 32 = **192 KB** | 32 KB | **224 KB** ✅ |
| BN512 | 48 KB | 4 × 48 = **192 KB** | 32 KB | **224 KB** ✅ |

Both land at exactly 224 KB, under the ~227 KB cap. (Note: `mmcomposer`'s
`validate_config` reports BN256/NS6/store2 as 225 KB and flags it — that extra
~1 KB is its conservative `__align__(1024)` padding against a 224 KB *soft* cap;
the real allocation fits, and this is the config the autotuner ships, e.g. the
SwiGLU K=768 case. NS=7 at BN256, or store buffers >2, would genuinely overflow.)

## How these were rendered

`mmcomposer/mmcomposer/kernels/tier3_cluster_swizzle/kernel.cu` is a template
(`#if` knob branches + `constexpr` value lines). `generate_kernel(config)`
splices shared fragments, resolves every `#if` against the config, and
substitutes the constants, yielding the branch-free `.cu` here:

```python
import sys; sys.path.insert(0, "/data/home/tong/projects/mmcomposer")
from mmcomposer.codegen import generate_kernel, generate_host

COMMON = dict(skeleton="tier3_cluster_swizzle", BM=128, BK=64,
              GROUP_SIZE_M=8, NUM_WARPS=4, TCGEN05_LD_WIDTH=8,
              EPILOGUE_OVERLAP=1, EPILOGUE_SPLIT=0, PERSISTENT=1, TWO_CTA=1,
              EPILOGUE_L1_NO_ALLOC=0, EPILOGUE_TMA_PIPELINED=1, TMA_STORE_STAGES=2)

cfg_bn256 = dict(COMMON, BN=256, NS=6, SINGLE_TMEM_ACCUM=0)  # double-TMEM
cfg_bn512 = dict(COMMON, BN=512, NS=4, SINGLE_TMEM_ACCUM=1)  # single-TMEM
```

The `bn512-single-tmem/kernel.cu` is byte-identical to mmcomposer's checked-in
golden `tier3_overlap_bn512_single_tmem.cu`. The `bn256-double-tmem/kernel.cu`
is a new NS=6 pipelined-store point (mmcomposer's goldens only pin NS=3).

## Running

Each `launcher.py` is self-contained (needs `torch`, `cuda-python`, and `nvcc`
on PATH, and a B200). It compiles the sibling `kernel.cu` and benchmarks it:

```bash
cd bn256-double-tmem && python launcher.py   # kernel.cu must sit alongside
```

The TK variant additionally needs a ThunderKittens checkout:

```bash
cd bn256-double-tmem-tk && python launcher.py
cd bn256-double-tmem-tk-tma && python launcher.py
cd bn256-double-tmem-tk-tma-managed && python launcher.py
TK_INCLUDE=/path/to/ThunderKittens/include python launcher.py
```

## Next: the CuTe rewrite

These are the reference. The following chapters re-implement each kernel in CuTe
DSL — the persistent scheduler, the `cute.arch` TMA/tcgen05/mbarrier primitives,
TMEM buffering, and the overlapped epilogue — and compare against these
hand-written PTX baselines.

## Files

- `bn256-double-tmem/` — `kernel.cu` (570 lines) + `launcher.py`; BN256, NS6, double-TMEM, 2 store buffers.
- `bn256-double-tmem-tk/` — first TK API variant of the BN256 kernel; same schedule and Python ABI.
- `bn256-double-tmem-tk-tma/` — second TK API variant; uses TK globals/shared tiles and TK TMA primitives.
- `bn256-double-tmem-tk-tma-managed/` — same as TK-TMA, but with managed TK TMEM allocation.
- `bn512-single-tmem/` — `kernel.cu` (590 lines) + `launcher.py`; BN512, NS4, single-TMEM, 2 store buffers.

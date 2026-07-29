# BN256 MXFP8 CUDA Raw-TMA Variant

This variant peels one more TK layer away from
`../bn256-mxfp8-cuda-mma-tma`.

The goal is instruction parity with the TK implementation.  The CUDA code
should spell out the same PTX instructions TK emits, not replace them with
equivalent-looking CUDA idioms.

It is self-contained CUDA/PTX.  It uses plain CUDA structs and inline PTX for
A/B operand loads, MXFP8 scale movement, TMEM allocation, mbarriers, the MXFP8
MMA itself, raw TMA descriptor globals, and BF16 TMA stores:

- `load_fp8_operand_tile_from_gmem_to_smem`: raw 5D TMA load / multicast for
  the 128B-swizzled FP8 A/B tiles
- `SmemTile`: typed shared-memory tile storage with compile-time shape and
  swizzle metadata
- `TmemTile`: typed tensor-memory address wrapper with compile-time shape and
  TMEM subtile address arithmetic
- `ScaleAtom`: raw `uint8_t data[32][16]` in SMEM
- `load_scale_atom_from_gmem_to_smem`: raw 4D TMA load / multicast
- `copy_scale_atom_from_smem_to_tmem`: raw `tcgen05.cp`
- `issue_mxfp8_mma2_abt_128x256x128`: raw
  `tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X`
- `mxfp8_mma2_abt_256x256_desc`: low-level `tcgen05.mma` instruction
  descriptor encoding
- `tcgen05_alloc_g2` / `tcgen05_dealloc_g2`: raw TMEM allocation/free
- raw `uint64_t mbar_*` shared-memory barriers plus explicit parity phase
  variables at the wait sites
- `mbarrier_*` helpers: raw `mbarrier.*` synchronization
- `elect_sync`: raw `elect.sync` warp election
- five explicit raw by-value `CUtensorMap` kernel parameters:
  `A_tmap`, `A_sc_tmap`, `B_tmap`, `B_sc_tmap`, `D_tmap`
- `prefetch_tma_descriptor`: optional `prefetch.tensormap` hint for those
  128B descriptors, not a tensor-data load
- `store_bf16_tile_from_smem_to_gmem`: raw
  `cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group`
- `SharedMemoryAllocator1024`: plain 1024-byte-aligned dynamic SMEM allocation
- `cluster_cta_rank` / `cluster_sync`: raw `%cluster_ctarank` and
  `barrier.cluster.*.aligned`

Do not replace `elect.sync` with a simple lane-0 predicate.  It is a warp-level
election/reconvergence primitive, and the lane-0 shortcut measured much slower
for this persistent pipeline even though it was functionally correct.

The MMA helper is intentionally fixed to this tutorial shape.  It builds the
K-major shared-memory descriptors for A and B, issues four 32B K chunks with
scale-factor ids `0,1,2,3`, and then commits the SMEM-buffer-free mbarrier for
both CTAs.

For the value operands, this wrapper mirrors TK's swizzled-TMA coordinate
mapping.  The logical tile coordinate `{row_tile, k_tile}` becomes:

```text
TMA coords {0, row_tile * tile_rows, k_tile * (tile_cols / 128), 0, 0}
```

For this kernel, both A and B value tiles have `tile_cols=128`, so the K
coordinate is just `k_tile`.

The scale layout is unchanged:

- GMEM stores each scale atom as a host-packed `[32,16]` E8M0 tile.
- TMA uses `TMA_SWIZZLE_NONE`.
- SMEM stores each scale atom as row-major contiguous 512B:
  row 0 bytes `0..15`, row 1 bytes `16..31`, ..., row 31 bytes `496..511`.
- `tcgen05.cp.cta_group::2.32x128b.warpx4` copies that `32 x 16B` source
  atom into the MMA-visible TMEM scale tile.

For `BN=256`, SFB still uses two independent 512B atoms:

```text
b_sc_smem[slot][0] -> TMEM cols slot*32 +  0..15
b_sc_smem[slot][1] -> TMEM cols slot*32 + 16..31
```

The SFB load behavior mirrors the TK wrapper:

- CTA 0 loads SFB atom 0 and multicasts it to both CTAs.
- CTA 1 loads SFB atom 1 and multicasts it to both CTAs.
- Both transactions complete into CTA 0's mbarrier via `mapa.shared::cluster`.

So each CTA still ends up with both SFB atoms locally, but without redundant
GMEM loads.

Run:

```bash
SHAPE_N=1024 python launcher.py
SHAPE_N=4096 python launcher.py
```

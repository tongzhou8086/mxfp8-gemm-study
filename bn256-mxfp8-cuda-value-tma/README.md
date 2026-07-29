# BN256 MXFP8 CUDA Value-TMA Variant

This variant peels one more TK layer away from
`../bn256-mxfp8-cuda-scale-tma`.

It keeps TK for:

- tensor-memory allocation
- `tcgen05.mma` wrapper calls
- mbarrier helpers
- BF16 epilogue and TMA stores

It uses plain CUDA wrappers and inline PTX for A/B operand loads and the MXFP8
scale path:

- `load_fp8_operand_tile_from_gmem_to_smem`: raw 5D TMA load / multicast for
  the 128B-swizzled FP8 A/B tiles
- `ScaleAtom`: raw `uint8_t data[32][16]` in SMEM
- `load_scale_atom_from_gmem_to_smem`: raw 4D TMA load / multicast
- `copy_scale_atom_from_smem_to_tmem`: raw `tcgen05.cp`

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

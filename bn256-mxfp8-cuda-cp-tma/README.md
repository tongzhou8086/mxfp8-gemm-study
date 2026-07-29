# BN256 MXFP8 CUDA tcgen05.cp Variant

This variant starts from `../bn256-mxfp8-tk-tma`, but replaces the TK
`load_mxnv_scale_async2` wrapper with explicit CUDA inline PTX for the scale
copy:

```cpp
tcgen05.cp.cta_group::2.32x128b.warpx4
```

The input scale layout is unchanged:

- GMEM scale atoms are host-packed as `[32,16]` E8M0 tiles.
- TMA loads each scale atom into SMEM with `TMA_SWIZZLE_NONE`.
- SMEM source atoms are non-swizzled, row-major contiguous 512B objects:
  row 0 bytes `0..15`, row 1 bytes `16..31`, ..., row 31 bytes `496..511`.
- SFA uses one 512B atom per `BM=128, BK=128` stage.
- SFB uses two 512B atoms per `BN=256, BK=128` stage.

This is intentionally not a full no-TK GEMM yet.  It still uses TK for TMA
descriptors, tensor-memory allocation, MMA wrappers, barriers, and the epilogue.
The purpose is to isolate and demonstrate the hardware scale-copy requirement:
the `[32,16]` SMEM scale atom is a plain row-major source for `tcgen05.cp`.

Run:

```bash
python launcher.py
SHAPE_N=1024 python launcher.py
```

# BN256 MXFP8 TK TMA Variant

This is the first local MXFP8 adaptation of the BN256 2-CTA B200 GEMM pipeline.
It keeps the persistent cluster schedule and TK TMA path, but switches the MMA
mainloop to block-scaled MXFP8:

- A is E4M3 `[M, K]`; B is E4M3 `[N, K]` and is consumed via `ABt`.
- A/B scale tensors are E8M0 in the tcgen05 `[outer/128, K/128, 32, 16]` layout.
- Warp 0 loads A, B, SFA, and SFB to SMEM.
- CTA 0 warp 1 copies SFA/SFB from SMEM to TMEM with `load_mxnv_scale_async2`,
  then issues `mm2_ABt` / `mma2_ABt`.
- Warps 4-7 drain FP32 TMEM to BF16 output through the existing staged TMA-store epilogue.

The BF16 double-TMEM variant uses two 256-column FP32 accumulators. MXFP8 also
needs TMEM columns for scale-factor tensors, so this first variant uses one
256-column accumulator and places scales at the pinned TK-style offsets:

```text
out_tm  @ 0
A_sc_tm @ 256
B_sc_tm @ 384
```

Run:

```bash
python launcher.py
TK_INCLUDE=/data/home/tong/projects/ThunderKittens/include python launcher.py
```

Last local B200 smoke checks:

```text
OK  MXFP8  M=N=K=1024   rel err=0.16%
OK  MXFP8  M=N=K=4096   rel err=0.29%
```

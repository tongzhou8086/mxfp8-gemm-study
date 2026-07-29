# BN256 Double-TMEM TK Variant

This directory is the first ThunderKittens API variant of
`../bn256-double-tmem`.

The goal is not to retune the kernel. It keeps the original BN256/NS6 schedule:

- persistent 2-CTA clusters
- warp 0 TMA load producer
- CTA 0 warp 1 tcgen05 MMA issuer
- warps 4-7 TMEM-to-SMEM-to-HBM epilogue
- two TMEM accumulator buffers
- two TMA-store staging buffers

What changed:

- includes ThunderKittens via `kittens.cuh`
- uses `kittens::cluster_ctarank`
- uses `kittens::semaphore`, `init_semaphore`, `wait`, and cluster arrive/expect helpers
- uses `kittens::detail::tcgen05::{st_st,commit}` for MMA issue and completion signaling
- uses `kittens::tensor_allocator<1, 2, false>` for the 512-column TMEM allocation
- keeps the original raw `CUtensorMap` launcher ABI and raw TMA load/store calls for now

Run:

```bash
python launcher.py
TK_INCLUDE=/data/home/tong/projects/ThunderKittens/include python launcher.py
```

Last local B200 check:

```text
OK  M=N=K=4096   rel err=0.29%   105.4 us/call   1303.5 TFLOPS
```

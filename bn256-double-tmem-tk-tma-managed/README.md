# BN256 Double-TMEM TK TMA Managed Variant

This is the managed-allocator variant of `../bn256-double-tmem-tk-tma`.

Compared with `../bn256-double-tmem-tk-tma`, this version leaves the TMA path
unchanged but switches TMEM allocation from unmanaged to managed:

```cpp
kittens::tensor_allocator<1, CTA_GROUP> tmem_allocator{};
```

That uses the default `managed = true`. The allocator constructor provisions
TMEM and stores the address in `tmem_allocator.addr`; the destructor deprovisions
TMEM when the variable goes out of scope.

The TMA path is still TK:

- A/B global tensors are passed as TK `gl` objects inside a by-value globals blob.
- A/B/D shared staging uses TK `st_bf` tiles allocated by `tma_swizzle_allocator`.
- A loads use `kittens::tma::cluster::load_async`.
- B loads remain two 64-column panels, matching the raw kernel's load sequence.
- D stores use `kittens::tma::store_async`.

The pipeline schedule is intentionally unchanged:

- persistent 2-CTA clusters
- warp 0 TMA load producer
- CTA 0 warp 1 tcgen05 MMA issuer
- warps 4-7 TMEM-to-SMEM-to-HBM epilogue
- two TMEM accumulator buffers
- two TMA-store staging buffers

The launcher keeps the original benchmark's B tensor layout as `K x N`.
It packs the TK `gl<bf16, 1, 1, -1, -1, tile>` objects from Python using the
layout checked against C++:

```text
sizeof(gl)=384, raw_ptr=0, rows=16, cols=24, tma_desc=128
sizeof(globals)=1152
```

Run:

```bash
python launcher.py
TK_INCLUDE=/data/home/tong/projects/ThunderKittens/include python launcher.py
```

Last local B200 check:

```text
OK  M=N=K=4096   rel err=0.29%   95.1 us/call   1444.7 TFLOPS
```

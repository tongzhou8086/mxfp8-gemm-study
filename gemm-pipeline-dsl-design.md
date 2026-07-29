# TilePipe: Schedule-Level GEMM DSL for Hopper and Blackwell

## One-Sentence Pitch

TilePipe is a schedule-level GEMM DSL where users write the pipeline, and the compiler writes the indexing.

TilePipe lets users describe the pipeline schedule over tiles without writing mechanical indexing or layout code.

## Core Idea

TilePipe is designed around a simple boundary:

- The user controls the GEMM pipeline schedule.
- The compiler owns tile indexing, layout calculation, predicates, and target-specific lowering.

This makes the language **schedule-explicit but index-implicit**. It keeps the low-level details that matter for performance, such as warp roles, buffering, barriers, TMA/MMA ordering, and epilogue staging, while removing mechanical code that does not express the pipeline schedule.

## Motivation

High-performance Hopper and Blackwell GEMM kernels require a large amount of code that is not conceptually part of the GEMM schedule:

- block and thread ID arithmetic
- CTA tile coordinate mapping
- CTA swizzling and grouped-M scheduling
- persistent-grid traversal
- boundary predicates
- TMA descriptor construction
- SMEM layout and swizzle selection
- TMEM address allocation
- MXFP8 scale tensor packing
- target-specific instruction glue

These details are essential for correctness and performance, but they are not the main thing a kernel author wants to reason about when designing a pipeline.

The DSL should let the user express:

- how many pipeline stages exist
- which warp or warpgroup produces data
- which warp or warpgroup consumes data
- when barriers are waited on or arrived on
- when TMA loads happen
- when MMA instructions are issued
- when TMEM is drained
- how epilogue stores are staged

The compiler should derive the indexing and physical layout details from the problem metadata, schedule metadata, and target backend.

## Design Boundary

### User-Written

The user writes:

- problem shape and tensor roles
- data types and compute type
- CTA tile shape
- cluster shape
- pipeline stage counts
- explicit barrier objects
- warp or warpgroup role structure
- high-level TMA, MMA, TMEM load, and store operations
- epilogue schedule

### Compiler-Generated

The compiler generates:

- thread ID and block ID usage
- CTA tile coordinates
- persistent-grid loops
- CTA swizzling / grouped-M traversal
- global-memory offsets
- SMEM offsets
- TMEM addresses
- TMA descriptors
- SMEM layouts
- TMEM layouts
- mbarrier phase toggling
- boundary predicates
- zero-fill behavior
- target-specific inline assembly or CUDA/TK/CuTe calls
- MXFP8 scale physical layout

## Non-Goals

The DSL is not trying to be a fully declarative matrix algebra language.

It intentionally exposes pipeline scheduling concepts such as barriers and warp roles. Those are part of the performance design. The language only hides details that are orthogonal to the schedule, especially index arithmetic and layout plumbing.

The first version should not support arbitrary device helper functions. The kernel body should be built from a small set of compiler-known operations so the compiler can reason about dependencies, barriers, memory layouts, and target lowering.

## Example: BF16 Blackwell GEMM

This example is intentionally high-level. It describes the schedule but avoids block IDs, thread IDs, global offsets, SMEM offsets, TMEM addresses, CTA swizzle math, and persistent-grid traversal.

```text
.metadata

problem {
    C[M, N] = A[M, K] @ B[K, N]
}

tensor A {
    shape  = (M, K)
    stride = (K, 1)
    dtype  = bf16
}

tensor B {
    shape  = (K, N)
    stride = (N, 1)
    dtype  = bf16
}

tensor C {
    shape  = (M, N)
    stride = (N, 1)
    dtype  = bf16
}

schedule {
    target        = sm100
    cta_tile      = (128, 256, 64)
    cluster       = (2, 1)
    smem_stages   = 6
    tmem_buffers  = 2
    store_buffers = 2
    mma_layout    = ABt
}

.kernel

mbarrier comp_data_ready[smem_stages]
mbarrier comp_buffer_free[smem_stages]
mbarrier tmem_data_ready[tmem_buffers]
mbarrier tmem_buffer_free[tmem_buffers]

num_k_tiles = ceil_div(K, BK)

warp 0 {
    for k_tile in 0..num_k_tiles {
        slot = k_tile % smem_stages

        wait_mbar(comp_buffer_free[slot])

        tma_load(A, tile=(BM, BK), stage=slot)
        tma_load(B, tile=(BK, BN), stage=slot)

        arrive_on_bytes(
            bytes(A[BM, BK]) + bytes(B[BK, BN]),
            comp_data_ready[slot]
        )
    }
}

warp 1 {
    acc = flip_tmem_buffer()

    wait_mbar(tmem_buffer_free[acc])

    for k_tile in 0..num_k_tiles {
        slot = k_tile % smem_stages

        wait_mbar(comp_data_ready[slot])

        issue_mma_k_tile(
            tile  = (BM, BN, BK),
            stage = slot
        )

        arrive_on_mma_done(comp_buffer_free[slot])
    }

    arrive_on_mma_done(tmem_data_ready[acc])
}

warp 4..8 {
    acc = flip_tmem_buffer()

    wait_mbar(tmem_data_ready[acc])

    for chunk in 0..4 {
        data = tmem_load(acc, chunk=chunk)
        wait_tmem_load()

        if chunk == 3 {
            arrive_mbar(tmem_buffer_free[acc])
        }

        tma_store(C, data, chunk=chunk)
    }
}
```

Here `issue_mma_k_tile` is a schedule event. On Blackwell, the compiler lowers it to MMA instructions that read the staged A/B tiles from SMEM, accumulate into the current TMEM accumulator, and reset-vs-accumulate according to whether this is the first K tile.

## Example: MXFP8 Blackwell GEMM

The user should express MXFP8 semantically:

```text
tensor A {
    shape  = (M, K)
    stride = (K, 1)
    dtype  = mxfp8(e4m3, scale=e8m0, block=32)
}

tensor B {
    shape  = (K, N)
    stride = (N, 1)
    dtype  = mxfp8(e4m3, scale=e8m0, block=32)
}

tensor C {
    shape = (M, N)
    dtype = bf16
}
```

The user should not manually express the Blackwell scale plumbing:

```text
logical scale:  [rows, K / 32]
physical scale: [rows / 128, K / 128, 32, 16]
SMEM tile:      st_fp8e8m0<32, 16, false>
TMEM tile:      full_tt_fp8e8m0<...>
copy op:        tcgen05.cp.cta_group::*.32x128b.warpx4
```

Those are target-specific layout-lowering details. The compiler should generate them when lowering an MXFP8 GEMM to Blackwell.

The pipeline can still expose scale movement at the schedule level:

```text
warp 0 {
    for k_tile in 0..num_k_tiles {
        slot = k_tile % smem_stages

        wait_mbar(comp_buffer_free[slot])

        tma_load(A.values, stage=slot)
        tma_load(B.values, stage=slot)
        tma_load(A.scales, stage=slot)
        tma_load(B.scales, stage=slot)

        arrive_on_bytes(bytes(stage_payload), comp_data_ready[slot])
    }
}

warp 1 {
    acc = flip_tmem_buffer()

    for k_tile in 0..num_k_tiles {
        slot = k_tile % smem_stages

        wait_mbar(comp_data_ready[slot])

        prepare_mxfp8_scales(stage=slot)

        issue_mma_k_tile(
            tile  = (BM, BN, BK),
            stage = slot
        )

        arrive_on_mma_done(comp_buffer_free[slot])
    }
}
```

Here `prepare_mxfp8_scales` is still a pipeline event, but the exact SMEM-to-TMEM scale movement, TMEM scale layout, and instruction sequence are backend-owned.

## Built-In Operations

The first version should provide a small set of built-ins:

```text
ceil_div(x, y)
wait_mbar(barrier)
arrive_mbar(barrier)
arrive_on_bytes(bytes, barrier)
arrive_on_mma_done(barrier)

tma_load(tensor, tile, stage)
tma_store(tensor, data, chunk)

issue_mma_k_tile(tile, stage)
prepare_mxfp8_scales(stage)
tmem_load(buffer, chunk)
wait_tmem_load()
flip_tmem_buffer()
bytes(tile)
```

The compiler should reject arbitrary function calls in `.kernel`. This keeps the kernel analyzable and makes it possible to lower barriers, memory dependencies, and target-specific instruction sequences correctly.

## Target Model

The same high-level schedule should be lowerable to different backend models, but not every backend supports every operation.

### Hopper / SM90

Likely lowering:

- TMA for global-to-shared movement
- WGMMA for matrix multiply
- no Blackwell TMEM accumulator model
- no `tcgen05`
- MXFP8 support depends on available target instructions and should not be assumed

### Blackwell / SM100+

Likely lowering:

- TMA for global-to-shared movement
- TMEM allocation for accumulators
- `tcgen05.mma` for MMA
- `tcgen05.ld` for epilogue TMEM drain
- `tcgen05.cp` for MXFP8/NVFP4 scale movement into TMEM
- target-specific scale tensor physical layouts

The frontend syntax can be shared, but the backend legality checks and generated code will differ substantially.

## Compiler Options

Some choices should be compiler or schedule options rather than source-level indexing code:

```text
pdc --target=sm100 --use-persistent-grid --gsm=8 gemm_b200.gpd
```

Candidate options:

```text
--target=sm90|sm100|sm103
--use-persistent-grid
--gsm=<value>
--cluster-m=<value>
--cluster-n=<value>
--smem-stages=<value>
--tmem-buffers=<value>
--store-buffers=<value>
--epilogue-warps=<value>
--emit=cuda|tk|cute
```

The persistent-grid loop and grouped-M / CTA-swizzle mapping are mechanical transformations. They should not pollute the source-level kernel schedule.

## Compilation Workflow

Given:

```text
gemm_b200.gpd
```

Compile with:

```text
pdc --target=sm100 --use-persistent-grid --gsm=8 gemm_b200.gpd
```

The compiler emits:

```text
gemm_b200_kernel.cu
gemm_b200_host.py
```

The generated host code should be directly callable from Python. A first implementation can use `cuda-python` bindings and NVRTC/NVCC compilation. Later versions could emit an importable C++/Python extension.

## Lowering Responsibilities

For a BF16 Blackwell GEMM, the compiler lowers:

```text
tma_load(A)
tma_load(B)
issue_mma_k_tile(...)
tmem_load(...)
tma_store(C)
```

into:

- TMA tensor-map descriptors
- SMEM tile allocation
- mbarrier initialization and phase tracking
- `tcgen05.mma` issue sequence
- TMEM allocator usage
- `tcgen05.ld` epilogue drain
- TMA store pipeline
- boundary predicates

For an MXFP8 Blackwell GEMM, the compiler additionally lowers:

- logical scale tensors
- scale prepacking requirements
- scale TMA descriptors
- SMEM scale tile layout
- TMEM scale allocation
- `tcgen05.cp` scale-copy instructions
- MXFP8 `tcgen05.mma` descriptors

## Design Principle

The DSL should not hide performance-critical scheduling.

It should hide the mechanical implementation burden needed to realize that schedule on a specific GPU target.

In short:

```text
Keep:    pipeline design
Remove:  indexing and layout plumbing
```

## Open Questions

- Should MXFP8 scale prepacking be represented as a generated preprocessing kernel, a host-side tensor transform, or a required input layout contract?
- Should the DSL allow both warp-level and warpgroup-level roles?
- Should barrier phase variables be explicit or compiler-generated?
- Should boundary behavior be implicit for all tensors, or should users choose between predicate, zero-fill, and unchecked modes?
- Should the compiler emit raw CUDA, ThunderKittens, CuTe DSL, or multiple backends?
- How much of epilogue mapping should be explicit before the DSL becomes too low-level?
- Should autotuning be part of `pdc`, or should `pdc` only generate one concrete schedule?

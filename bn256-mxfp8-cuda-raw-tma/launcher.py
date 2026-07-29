"""Self-contained MXFP8 matmul kernel launcher.

Variant: bn256-mxfp8-cuda-raw-tma
Config: BM=128 BN=256 BK=128 NS=5 GROUP_SIZE_M=16 NUM_WARPS=4

Run with:  python <this file>.py   (kernel.cu must sit alongside it)
Requires:  torch, numpy, cuda-python (cuda.bindings), nvcc on PATH.

This variant implements A/B value TMA, MXFP8 scale movement, TMEM allocation,
mbarriers, MXFP8 MMA, raw TMA descriptor globals, and BF16 TMA stores with
explicit CUDA inline PTX:
    cp.async.bulk.tensor.5d...  for GMEM->SMEM FP8 value TMA
    cp.async.bulk.tensor.4d...  for GMEM->SMEM scale TMA
    tcgen05.cp.cta_group::2.32x128b.warpx4
    tcgen05.mma.cta_group::2.kind::mxf8f6f4...
    tcgen05.alloc/dealloc and mbarrier.* synchronization
    cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group for D store
"""

import ctypes
import os
import subprocess
import sys

import numpy as np
from cuda.bindings import driver, nvrtc, runtime


# ── ctypes binding for cuTensorMapEncodeTiled ───────────────────────────────
#
# cuda-python's wrapper for this function has shifted across versions
# (strict typed scalars, evolving return-value conventions).  Calling
# the C entry point directly via ctypes is more stable and the
# descriptor-build site reads much more cleanly.

_libcuda = ctypes.CDLL("libcuda.so", mode=ctypes.RTLD_GLOBAL)
_cuTensorMapEncodeTiled = _libcuda.cuTensorMapEncodeTiled
_cuTensorMapEncodeTiled.restype = ctypes.c_int
_cuTensorMapEncodeTiled.argtypes = [
    ctypes.c_void_p,                  # CUtensorMap* (output)
    ctypes.c_int,                     # dtype
    ctypes.c_uint32,                  # rank
    ctypes.c_void_p,                  # globalAddress
    ctypes.POINTER(ctypes.c_uint64),  # globalDim
    ctypes.POINTER(ctypes.c_uint64),  # globalStrides
    ctypes.POINTER(ctypes.c_uint32),  # boxDim
    ctypes.POINTER(ctypes.c_uint32),  # elementStrides
    ctypes.c_int,                     # interleave
    ctypes.c_int,                     # swizzle
    ctypes.c_int,                     # l2 promotion
    ctypes.c_int,                     # oob fill
]


# Mirrors the CUtensorMap* enums from cuda.h.  Numeric values stable.
TMA_UINT8       = 0
TMA_UINT16      = 1
TMA_UINT32      = 2
TMA_INT32       = 3
TMA_UINT64      = 4
TMA_INT64       = 5
TMA_FLOAT16     = 6
TMA_FLOAT32     = 7
TMA_FLOAT64     = 8
TMA_BFLOAT16    = 9

TMA_INTERLEAVE_NONE = 0
TMA_INTERLEAVE_16B  = 1
TMA_INTERLEAVE_32B  = 2

TMA_SWIZZLE_NONE = 0
TMA_SWIZZLE_32B  = 1
TMA_SWIZZLE_64B  = 2
TMA_SWIZZLE_128B = 3

TMA_L2_NONE = 0
TMA_L2_64B  = 1
TMA_L2_128B = 2
TMA_L2_256B = 3

TMA_OOB_NONE = 0
TMA_OOB_NAN_REQUEST_ZERO_FMA = 1


def encode_tensor_map(
    *,
    dtype: int,
    rank: int,
    gptr: int,
    global_dim,
    box_dim,
    element_strides,
    global_strides=None,
    interleave: int = TMA_INTERLEAVE_NONE,
    swizzle: int = TMA_SWIZZLE_NONE,
    l2_promotion: int = TMA_L2_NONE,
    oob_fill: int = TMA_OOB_NONE,
) -> np.ndarray:
    """Build a 128-byte CUtensorMap via libcuda's cuTensorMapEncodeTiled.

    All shape/stride args are Python sequences of ints:

      global_dim       — length `rank`, innermost-first
      global_strides   — length `rank - 1`, in BYTES, outer-dim strides;
                         pass None or [] for 1D
      box_dim          — length `rank`, innermost-first
      element_strides  — length `rank`, typically all 1s

    Returns a numpy uint8 array of length 128 holding the descriptor.
    Pass it to a kernel as a by-value 128-byte struct argument.
    """
    if global_strides is None:
        global_strides = []
    assert len(global_dim)      == rank,     f"global_dim must have {rank} entries"
    assert len(box_dim)         == rank,     f"box_dim must have {rank} entries"
    assert len(element_strides) == rank,     f"element_strides must have {rank} entries"
    assert len(global_strides)  == rank - 1, f"global_strides must have {rank - 1} entries"

    tmap = np.zeros(128, dtype=np.uint8)
    gdim_arr = (ctypes.c_uint64 * rank)(*global_dim)
    bdim_arr = (ctypes.c_uint32 * rank)(*box_dim)
    estr_arr = (ctypes.c_uint32 * rank)(*element_strides)
    gstr_arr = ((ctypes.c_uint64 * (rank - 1))(*global_strides)
                if rank > 1 else None)

    err = _cuTensorMapEncodeTiled(
        tmap.ctypes.data,
        dtype,
        rank,
        gptr,
        gdim_arr,
        gstr_arr,
        bdim_arr,
        estr_arr,
        interleave,
        swizzle,
        l2_promotion,
        oob_fill,
    )
    if err != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: CUresult={err}")
    return tmap


# ── Error checking ──────────────────────────────────────────────────────────

def cu(result):
    """Unwrap a cuda-python `(err, *rest)` return tuple, raising on error.

    cuda-python's driver / nvrtc / runtime bindings all return a leading
    error code followed by any out-parameters.  This helper checks the
    error and returns the rest (a single value if just one, else a
    tuple, or None if no out-params).
    """
    err, *rest = result
    if isinstance(err, driver.CUresult):
        if err != driver.CUresult.CUDA_SUCCESS:
            _, name = driver.cuGetErrorName(err)
            raise RuntimeError(f"CUDA driver error: {name.decode()}")
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"NVRTC error: {err}")
    elif isinstance(err, runtime.cudaError_t):
        if err != runtime.cudaError_t.cudaSuccess:
            _, name = runtime.cudaGetErrorName(err)
            raise RuntimeError(f"CUDA runtime error: {name.decode()}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")
    if not rest:
        return None
    if len(rest) == 1:
        return rest[0]
    return tuple(rest)


# ── Device + context ────────────────────────────────────────────────────────

def init_cuda(device_id: int = 0):
    """Init the CUDA driver, pick a device, and bring up its primary context.

    Returns (device, ctx).  We use the **primary context** rather than
    `cuCtxCreate` because (a) its signature has shifted across recent
    cuda-python versions, and (b) the primary context plays nicely with
    anything else using CUDA in the same process (PyTorch, etc.).

    Caller is responsible for releasing the primary context with
    `cu(driver.cuDevicePrimaryCtxRelease(device))` at the end.
    """
    cu(driver.cuInit(0))
    device = cu(driver.cuDeviceGet(device_id))
    ctx = cu(driver.cuDevicePrimaryCtxRetain(device))
    cu(driver.cuCtxSetCurrent(ctx))
    return device, ctx


def compute_arch(device) -> str:
    """Return e.g. 'sm_100a' for the current device (suitable for NVRTC)."""
    major = cu(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device))
    minor = cu(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device))
    return f"sm_{major}{minor}a"


# ── nvcc compile (with mtime-based cubin cache) ─────────────────────────────

def compile_kernel(src_path: str, device, kernels: list, extra_opts: list = None):
    """Compile a .cu file via `nvcc --cubin` and resolve named kernels.

    The cubin is cached on disk next to the .cu file (suffix
    `_sm_XYZa.cubin`).  Re-compile happens only when the .cu's mtime
    is newer than the cubin's — so repeated runs are fast.

    nvcc is used (rather than NVRTC) so the kernel can include
    standard CUDA headers (`<cuda.h>`, `<cuda_bf16.h>`, `<cstdint>`,
    etc.) without manual workarounds.

    Returns (module, {kernel_name: CUfunction}).
    """
    arch = compute_arch(device)
    cubin_path = src_path[:-3] + f"_{arch}.cubin"

    needs_rebuild = (not os.path.exists(cubin_path)
                     or os.path.getmtime(src_path) > os.path.getmtime(cubin_path))
    if needs_rebuild:
        print(f"[nvcc] compiling {os.path.basename(src_path)} → "
              f"{os.path.basename(cubin_path)} ... ",
              end="", flush=True)
        nvcc = os.environ.get("NVCC", "nvcc")
        cmd = [
            nvcc,
            f"-arch={arch}",
            "-O3",
            "--std=c++20",
            "--expt-relaxed-constexpr",
            "--extended-lambda",
            "--cubin",
        ]
        if extra_opts:
            cmd.extend(extra_opts)
        cmd += [src_path, "-o", cubin_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write(r.stderr)
            raise RuntimeError(f"nvcc failed (exit {r.returncode})")
        print("done", flush=True)

    with open(cubin_path, "rb") as f:
        cubin = f.read()
    module = cu(driver.cuModuleLoadData(cubin))
    fns = {k: cu(driver.cuModuleGetFunction(module, k.encode()))
           for k in kernels}
    return module, fns


# ── Memory I/O ──────────────────────────────────────────────────────────────

def htod(host: np.ndarray) -> int:
    """Allocate device memory matching `host.nbytes` and copy host → device."""
    d = cu(driver.cuMemAlloc(host.nbytes))
    cu(driver.cuMemcpyHtoD(d, host.ctypes.data, host.nbytes))
    return d


def dtoh(d: int, nbytes: int, dtype) -> np.ndarray:
    """Allocate a host ndarray of `dtype` and copy `nbytes` from `d`."""
    out = np.empty(nbytes // np.dtype(dtype).itemsize, dtype=dtype)
    cu(driver.cuMemcpyDtoH(out.ctypes.data, d, nbytes))
    return out


# ── Launch ──────────────────────────────────────────────────────────────────

def launch(kernel, *, grid, block, shared: int, args: list, stream: int = 0,
           sync: bool = True):
    """Launch a kernel.

    `args` is a list of ctypes objects (one per kernel parameter).
    For by-value structs, pass a `(ctypes.c_byte * N).from_buffer_copy(bytes)`.
    For pointers, pass a `ctypes.c_void_p(int_address)`.

    `sync=True` (default) blocks the host until the kernel completes —
    convenient for chapter examples that read C right after launching.
    Timing harnesses should pass `sync=False` and synchronize once at
    the end of a batch; otherwise the per-launch sync (~5–10 µs round
    trip) inflates small-shape timings significantly.
    """
    arg_ptrs = (ctypes.c_void_p * len(args))(
        *[ctypes.addressof(a) for a in args]
    )
    cu(driver.cuLaunchKernel(
        kernel,
        *grid,
        *block,
        shared,
        stream,
        arg_ptrs,
        0,    # extra (unused)
    ))
    if sync:
        cu(driver.cuCtxSynchronize())


# ── Timing ──────────────────────────────────────────────────────────────────

def time_kernel_us(call_fn, warmup_ms=None, rep_ms=None) -> float:
    """Median per-call time of `call_fn` in microseconds.

    Thin wrapper around `triton.testing.do_bench`.  Used by every
    chapter's main.py to time kernel launches consistently:

    - do_bench flushes a 256 MB scratch buffer through L2 between
      samples, so each measurement reflects cold-cache first-call
      behaviour (not warm-L2 steady-state).
    - It picks its own iter count adaptively to fit `rep_ms`, so fast
      and slow kernels are timed over comparable wall-clock budgets.
    - It returns quantiles so we get median, min, max for free.

    `call_fn` should perform exactly one kernel launch (or operation)
    per call.  Pass `sync=False` to your `launch(...)` inside `call_fn`
    — do_bench handles the synchronization itself.

    Triton is available as a transitive dependency of PyTorch on any
    GPU install, so this doesn't add a new top-level dependency.
    """
    import triton.testing  # lazy import — only chapters that time pay the cost
    if warmup_ms is None:
        warmup_ms = int(os.environ.get("WARMUP_MS", "20"))
    if rep_ms is None:
        rep_ms = int(os.environ.get("REP_MS", "200"))
    ms_med, _, _ = triton.testing.do_bench(
        call_fn,
        warmup=warmup_ms,
        rep=rep_ms,
        quantiles=(0.5, 0.0, 1.0),
    )
    return ms_med * 1000.0


# ===== tier launcher =====

# ─────────────────────────────────────────────────────────────────────
# Tier 3 — warp-specialized + 2-CTA cluster MMA (`cta_group::2`): two
# CTAs cooperate in one tcgen05.mma, each owning BN/2 columns of B and
# BM rows.  Generalized variable-warp epilogue, CTA-swizzle tunable.
#
# This launcher is a fragment: mmcomposer prepends the self-contained
# runtime preamble above it, so the downloaded file runs on its own:
#     python <this file>.py
# Requires: torch, numpy, cuda-python (`cuda.bindings`), and `nvcc` on PATH.
# ─────────────────────────────────────────────────────────────────────

import os
import ctypes

import torch
from cuda.bindings import driver


# ── User-tunable constants (mirror kernel.cu — mmcomposer keeps in sync) ──
BM, BN, BK = 128, 256, 128
NS           = 5
GROUP_SIZE_M = 16
NUM_WARPS    = 4
PERSISTENT   = 1
TCGEN05_LD_WIDTH = 8
EPILOGUE_OVERLAP = 1
EPILOGUE_SPLIT = 0
EPILOGUE_L1_NO_ALLOC = 0
EPILOGUE_TMA_PIPELINED = 1
SINGLE_TMEM_ACCUM = 1
SEGMENTED_PANELS = 0        # 1 = BN512 segmented panel schedule (SEG = NS)
TWO_CTA      = 1            # 1 = 2-CTA cluster MMA; 0 = single-CTA (grid/SMEM degenerate)

CTA_GROUP    = 2 if TWO_CTA else 1
BN_LOCAL     = BN // CTA_GROUP
FP8_BYTES    = 1
BF16_BYTES   = 2
STORE_N      = 64
TMA_STORE_STAGES = 3
# Overlap uses two stream warps in warpgroup 0 plus NUM_WARPS epilogue
# warps starting at warp 4, matching the Tier 2 overlap convention.
THREADS      = (NUM_WARPS + 4) * 32 if EPILOGUE_OVERLAP else NUM_WARPS * 32
A_SLOT_BYTES = BM       * BK * FP8_BYTES
B_SLOT_BYTES = BN_LOCAL * BK * FP8_BYTES
SF_TILE_BYTES = 32 * 16 * FP8_BYTES
SLOT_BYTES   = A_SLOT_BYTES + B_SLOT_BYTES + SF_TILE_BYTES * (1 + CTA_GROUP)
if EPILOGUE_OVERLAP and EPILOGUE_TMA_PIPELINED:
    EPI_BYTES = BM * STORE_N * BF16_BYTES * TMA_STORE_STAGES
else:
    EPI_LD    = ((BN // 2 + 8) if (EPILOGUE_OVERLAP and EPILOGUE_SPLIT) else (BN + 8))
    EPI_BYTES = BM * EPI_LD * BF16_BYTES
if SEGMENTED_PANELS:
    # Segmented panel schedule: SMEM = [ A ring | B ring | C_store ].
    # A ring = SEG+1 slots; B ring budget-fills the 14-tile (225 KB) budget.
    SEG_NA = NS + 1
    SEG_NB = 14 - TMA_STORE_STAGES - SEG_NA
    SEG_B_SLOT_BYTES = (BN // 2 // CTA_GROUP) * BK * FP8_BYTES
    SHARED_BYTES = SEG_NA * A_SLOT_BYTES + SEG_NB * SEG_B_SLOT_BYTES + EPI_BYTES + 1024
elif EPILOGUE_OVERLAP:
    SHARED_BYTES = NS * SLOT_BYTES + EPI_BYTES + 4096
else:
    SHARED_BYTES = max(NS * SLOT_BYTES, EPI_BYTES) + 4096
HERE         = os.path.dirname(os.path.abspath(__file__))

CUDA_TENSOR_MAP_SIZE = 128


def encode_swizzled_tile_tensor_map(*, gptr: int, rows: int, cols: int,
                                    tile_rows: int, tile_cols: int,
                                    elem_bytes: int, dtype: int) -> np.ndarray:
    """Build the 5D CUtensorMap layout used by the 128B-swizzled SMEM tiles."""
    swizzle_elements = 128 // elem_bytes
    assert tile_cols % swizzle_elements == 0
    return encode_tensor_map(
        dtype=dtype,
        rank=5,
        gptr=gptr,
        global_dim=[
            swizzle_elements,
            rows,
            (cols + swizzle_elements - 1) // swizzle_elements,
            1,
            1,
        ],
        global_strides=[
            cols * elem_bytes,
            128,
            rows * cols * elem_bytes,
            rows * cols * elem_bytes,
        ],
        box_dim=[
            swizzle_elements,
            tile_rows,
            tile_cols // swizzle_elements,
            1,
            1,
        ],
        element_strides=[1, 1, 1, 1, 1],
        swizzle=TMA_SWIZZLE_128B,
    )


def encode_scale_tensor_map(*, gptr: int, outer: int, k_tiles: int) -> np.ndarray:
    """Build the 4D CUtensorMap for packed `[outer, k_tiles, 32, 16]` scales."""
    return encode_tensor_map(
        dtype=TMA_UINT8,
        rank=4,
        gptr=gptr,
        global_dim=[16, 32, k_tiles, outer],
        global_strides=[
            16 * FP8_BYTES,
            32 * 16 * FP8_BYTES,
            k_tiles * 32 * 16 * FP8_BYTES,
        ],
        box_dim=[16, 32, 1, 1],
        element_strides=[1, 1, 1, 1],
        swizzle=TMA_SWIZZLE_NONE,
    )


def build_tma_descriptors(A, A_sc, B, B_sc, C, M: int, N: int, K: int):
    """Build the five by-value CUtensorMap descriptors consumed by the kernel."""
    a_tma = encode_swizzled_tile_tensor_map(
        gptr=A.data_ptr(), rows=M, cols=K, tile_rows=BM, tile_cols=BK,
        elem_bytes=FP8_BYTES, dtype=TMA_UINT8)
    a_sc_tma = encode_scale_tensor_map(
        gptr=A_sc.data_ptr(), outer=M // BM, k_tiles=K // BK)
    b_tma = encode_swizzled_tile_tensor_map(
        gptr=B.data_ptr(), rows=N, cols=K, tile_rows=BN_LOCAL, tile_cols=BK,
        elem_bytes=FP8_BYTES, dtype=TMA_UINT8)
    b_sc_tma = encode_scale_tensor_map(
        gptr=B_sc.data_ptr(), outer=N // BN_LOCAL, k_tiles=K // BK)
    d_tma = encode_swizzled_tile_tensor_map(
        gptr=C.data_ptr(), rows=M, cols=N, tile_rows=BM, tile_cols=STORE_N,
        elem_bytes=BF16_BYTES, dtype=TMA_BFLOAT16)

    return a_tma, a_sc_tma, b_tma, b_sc_tma, d_tma


def by_value_tensor_map_arg(tensor_map: np.ndarray):
    """Convert one 128B CUtensorMap into a ctypes by-value kernel argument."""
    return (ctypes.c_byte * CUDA_TENSOR_MAP_SIZE).from_buffer_copy(tensor_map.tobytes())


device, ctx = init_cuda()
NUM_SMS = cu(driver.cuDeviceGetAttribute(
    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device))

mod, fns = compile_kernel(os.path.join(HERE, "kernel.cu"),
                          device, kernels=["matmul_cluster"])
kernel = fns["matmul_cluster"]

cu(driver.cuFuncSetAttribute(
    kernel,
    driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    SHARED_BYTES))


MXFP8_DEST_MAX = 448.0
MXFP8_BIAS = 127.0


def mxfp8_quantize(v: torch.Tensor):
    """BF16/FP32 [rows, cols] -> E4M3 values plus unswizzled E8M0 scales."""
    assert v.dim() == 2
    rows, cols = v.shape
    assert cols % 32 == 0
    vf = v.to(torch.float32)
    block_amax = torch.amax(vf.abs().view(rows, cols // 32, 32), dim=-1)
    exp = torch.clamp(torch.ceil(torch.log2(block_amax / MXFP8_DEST_MAX)),
                      min=-127.0, max=127.0)
    exp = torch.where(block_amax == 0, torch.zeros_like(exp), exp)
    e8m0 = (exp + MXFP8_BIAS).to(torch.uint8)
    scale = (2.0 ** exp).repeat_interleave(32, dim=-1)
    q = (vf / scale).to(torch.float8_e4m3fn)
    return q.contiguous(), e8m0.contiguous()


def mxfp8_dequant(q: torch.Tensor, e8m0: torch.Tensor) -> torch.Tensor:
    exp = e8m0.to(torch.float32) - MXFP8_BIAS
    scale = (2.0 ** exp).repeat_interleave(32, dim=-1)
    return q.to(torch.float32) * scale


def scale_swizzle(e8m0: torch.Tensor) -> torch.Tensor:
    """[rows, cols/32] uint8 -> [rows/128, cols/128, 32, 16] uint8."""
    assert e8m0.dtype == torch.uint8 and e8m0.dim() == 2
    rows, n_32 = e8m0.shape
    assert rows % 128 == 0
    assert (n_32 * 32) % 128 == 0
    v = e8m0.reshape(rows // 128, 128, n_32 // 4, 4)
    v = v.transpose(1, 2).reshape(rows // 128, n_32 // 4, 4, 32, 4)
    v = v.transpose(-2, -3).reshape(rows // 128, n_32 // 4, 32, 16)
    return v.contiguous()


def setup(M, N, K):
    torch.manual_seed(0)
    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    A, A_sc_unsw = mxfp8_quantize(A_bf16)
    B, B_sc_unsw = mxfp8_quantize(B_bf16)
    A_sc = scale_swizzle(A_sc_unsw)
    B_sc = scale_swizzle(B_sc_unsw)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    A_tmap, A_sc_tmap, B_tmap, B_sc_tmap, D_tmap = build_tma_descriptors(
        A, A_sc, B, B_sc, C, M, N, K)
    arg_a_tmap = by_value_tensor_map_arg(A_tmap)
    arg_a_sc_tmap = by_value_tensor_map_arg(A_sc_tmap)
    arg_b_tmap = by_value_tensor_map_arg(B_tmap)
    arg_b_sc_tmap = by_value_tensor_map_arg(B_sc_tmap)
    arg_d_tmap = by_value_tensor_map_arg(D_tmap)
    arg_c = ctypes.c_void_p(C.data_ptr())
    args = [
        arg_a_tmap,
        arg_a_sc_tmap,
        arg_b_tmap,
        arg_b_sc_tmap,
        arg_d_tmap,
        arg_c,
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
    ]

    grid_m_clusters = M // (CTA_GROUP * BM)
    grid_n          = N // BN
    if PERSISTENT:
        grid = (NUM_SMS - NUM_SMS % CTA_GROUP, 1, 1)
    else:
        grid = (grid_m_clusters * grid_n * CTA_GROUP, 1, 1)
    return A, A_sc_unsw, B, B_sc_unsw, C, grid, args


SHAPE_N = int(os.environ.get("SHAPE_N", "4096"))
SHAPE_M = int(os.environ.get("SHAPE_M", str(SHAPE_N)))
SHAPE_K = int(os.environ.get("SHAPE_K", str(SHAPE_N)))

for (M, N, K) in [(SHAPE_M, SHAPE_N, SHAPE_K)]:
    A, A_sc_unsw, B, B_sc_unsw, C, grid, args = setup(M, N, K)
    flops = 2.0 * M * N * K

    C.zero_()
    launch(kernel, grid=grid, block=(THREADS, 1, 1),
           shared=SHARED_BYTES, args=args)
    C_ref = (mxfp8_dequant(A, A_sc_unsw) @ mxfp8_dequant(B, B_sc_unsw).t()).to(torch.bfloat16)
    rel = (C.float() - C_ref.float()).abs().max().item() / C_ref.float().abs().max().item()
    ok = "OK" if rel < 5e-2 else "FAIL"

    us = time_kernel_us(lambda: launch(
        kernel, grid=grid, block=(THREADS, 1, 1),
        shared=SHARED_BYTES, args=args, sync=False))
    tf = flops / (us * 1e-6) / 1e12

    print(f"{ok}  MXFP8 CUDA-RAW-TMA    M,N,K={M:>5},{N:>5},{K:>5}   grid={grid[0]} CTAs ({CTA_GROUP}-CTA clusters)   rel err={rel:.2%}")
    print(f"     BM={BM} BN={BN} BK={BK} NS={NS} GSM={GROUP_SIZE_M} NW={NUM_WARPS} "
          f"PERSISTENT={PERSISTENT} "
          f"EPILOGUE_OVERLAP={EPILOGUE_OVERLAP} EPILOGUE_SPLIT={EPILOGUE_SPLIT}   "
          f"EPILOGUE_TMA_PIPELINED={EPILOGUE_TMA_PIPELINED}   "
          f"TMA_STORE_STAGES={TMA_STORE_STAGES}   "
          f"SINGLE_TMEM_ACCUM={SINGLE_TMEM_ACCUM}   "
          f"{us:7.1f} us/call   "
          f"{tf:6.1f} TFLOPS")
    print()


cu(driver.cuModuleUnload(mod))
cu(driver.cuDevicePrimaryCtxRelease(device))

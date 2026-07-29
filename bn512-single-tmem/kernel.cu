#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

// ── User-tunable constants (the webui substitutes these) ────────────
constexpr int BM           = 128;
constexpr int BN           = 512;
constexpr int BK           = 64;
constexpr int NS           = 4;       // multi-stage SMEM ring depth
constexpr int GROUP_SIZE_M = 8;       // CTA-swizzle chunk (1 = no swizzle)
constexpr int NUM_WARPS    = 4;       // total warps per CTA
constexpr int TCGEN05_LD_WIDTH = 8;  // TMEM->reg epilogue load width: 8 or 16 (32-bit elems per lane)
constexpr int EPILOGUE_OVERLAP = 1;  // 1 = persistent 2-CTA cluster + epilogue/K-loop overlap
constexpr int EPILOGUE_SPLIT   = 0;  // 1 = split overlapped int4 writeback into two half-BN passes
constexpr int EPILOGUE_TMA_PIPELINED = 1;  // 1 = chunked staged TMA-store overlap epilogue
constexpr int SINGLE_TMEM_ACCUM = 1;  // 1 = overlap path synchronizes epilogue drain before reusing one TMEM accumulator
constexpr int SEGMENTED_PANELS = 0;  // 1 = BN512 segmented panel schedule (SEG = NS k-tiles per segment)
constexpr int TWO_CTA          = 1;  // 1 = 2-CTA cluster MMA (cta_group::2); 0 = single-CTA

// ── Derived constants (do not edit) ─────────────────────────────────
constexpr int MMA_K     = 16;
constexpr int BF16_BYTES = 2;
constexpr int K_MMAS    = BK / MMA_K;        // 4

constexpr int CTA_GROUP        = TWO_CTA ? 2 : 1;    // 2-CTA cluster vs single-CTA
constexpr int BN_LOCAL         = BN / CTA_GROUP;     // per-CTA N width of B (=BN single-CTA)
constexpr int SWIZZLE_ROW_BYTES = 128;               // one 128B-swizzle atom row
constexpr int STORE_N          = 64;                 // TMA-store chunk width
constexpr int TMA_STORE_STAGES = 2;                  // TMA-store SMEM buffers

// Per-stage SMEM per CTA: A = BM*BK*2 = 16 KB; B = BN_LOCAL*BK*2 = 16 KB.
// Total 32 KB / stage / CTA — half of ch07's 48 KB / stage / CTA.
constexpr int A_SLOT_BYTES = BM       * BK * BF16_BYTES;       // 16 KB
constexpr int B_SLOT_BYTES = BN_LOCAL * BK * BF16_BYTES;       // 16 KB
constexpr int SLOT_BYTES   = A_SLOT_BYTES + B_SLOT_BYTES;      // 32 KB / slot


// ── Important: dynamic SMEM is used in TWO non-overlapping phases ──
//
// 1.  During the K-loop, the kernel uses `NS * SLOT_BYTES` bytes —
//     NS slots × (A + B) per slot — as the multi-stage ring buffer.
// 2.  During the epilogue, the same dynamic SMEM is REINTERPRETED as
//     a `[BM][BN+8]` BF16 staging buffer for the coalesced writeback
//     (see ch07).  Its size is `EPILOGUE_STAGING_BYTES` below.
//
// The two phases never overlap in time (`all_mmas_done` separates
// them), so SMEM can be reused.  But the launcher MUST size the
// dynamic SMEM allocation to the MAX of the two phases:
//
//     shared_bytes = max(NS * SLOT_BYTES, EPILOGUE_STAGING_BYTES)
//                  + padding for __align__(1024)
//
// In ch07 (single-CTA) the K-loop term always dominated, so we never
// had to think about this.  In ch08, the per-CTA B-slot SMEM cost
// drops from 32 KB to 16 KB (cluster splits B), which means at low
// NS the K-loop SMEM can fall *below* the staging buffer's needs.
// Specifically, at NS=2, `NS * SLOT_BYTES = 64 KB < 67584 B` and the
// staging dominates.  Allocate too little dynamic SMEM and the
// epilogue scribbles past it → CUDA_ERROR_ILLEGAL_ADDRESS.
//
// See `shared_for()` in `main.py` for the launcher-side computation,
// and the README's "Sizing the dynamic SMEM" subsection for the
// full discussion.
constexpr int WARP_SIZE = 32;
constexpr int THREADS   = NUM_WARPS * WARP_SIZE;  // epilogue worker threads
constexpr int LAUNCH_THREADS = (NUM_WARPS + 4) * WARP_SIZE;


// ── helpers ─────────────────────────────────────────────────────────
// ---- elementwise epilogue (EDL): per-element fp32 map before bf16 store ----
__device__ __forceinline__ float mmc_epi(float x) { return x; }

__device__ __forceinline__ bool elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t .reg .pred px;\n\t"
        "elect.sync _|px, %1;\n\t"
        "@px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred) : "r"(0xFFFFFFFF));
    return pred;
}

// TMA load — `.cta_group::2` is the key new modifier.  The tx-count
// is bookkept against a cluster-wide mbar (the peer-CTA arrival is
// what makes both CTAs' arrivals count toward CTA 0's SMEM-compute-full
// mbar).  Without it, peer-CTA loads silently fail to advance the
// mbar and the kernel deadlocks.
__device__ __forceinline__ void tma_2d_load_g2(
    uint32_t smem_dst, const void* tmap, int x, int y, uint32_t mbar
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar) : "memory");
}


// A's descriptor — MN-major, unchanged from earlier chapters.
__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    uint64_t b = ((SBO)              >> 4) & 0x3FFFULL;
    return a | (b << 32) | (1ULL << 46) | (2ULL << 61);   // SWIZZLE_128B
}

// K-major B descriptor (same as ch06/07).
__device__ __forceinline__ uint64_t make_desc_K_major(
    uint32_t smem_addr, int lbo_bytes
) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a   = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    uint64_t lbo = ((uint64_t)lbo_bytes >> 4) & 0x3FFFULL;
    uint64_t b   = ((SBO)               >> 4) & 0x3FFFULL;
    return a | (lbo << 16) | (b << 32) | (1ULL << 46) | (2ULL << 61);
}

// idesc with M = CTA_GROUP * BM (cluster spans both CTAs in M),
// bit 16 = 1 (B is K-major).
__device__ __forceinline__ uint32_t make_idesc_bf16_cluster(int m, int n) {
    uint32_t d = 0;
    d |= (1u << 4);                                    // c_format = F32
    d |= (1u << 7);                                    // a_format = BF16
    d |= (1u << 10);                                   // b_format = BF16
    d |= (1u << 16);                                   // B is K-major
    d |= (((uint32_t)(n >> 3) & 0x3F) << 17);          // n_dim
    d |= (((uint32_t)(m >> 4) & 0x1F) << 24);          // m_dim
    return d;
}


// ── tcgen05 MMA wrappers (cta_group::2 cluster / cta_group::1 single) ─
// Same names + signatures under both TWO_CTA arms so the call sites (and the
// MMA_ISSUE macro) are identical — TWO_CTA=1 renders byte-for-byte as the
// cluster tier; TWO_CTA=0 swaps in the single-CTA cta_group::1 instructions.
__device__ __forceinline__ void tcgen05_alloc_g2(uint32_t smem_dst, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_dst), "r"(n_cols) : "memory");
}
__device__ __forceinline__ void tcgen05_dealloc_g2(uint32_t taddr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(taddr), "r"(n_cols) : "memory");
}
__device__ __forceinline__ void tcgen05_mma_g2(
    uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
    uint32_t idesc, bool enable_d
) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "setp.ne.b32 P, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, P;\n\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
           "r"((uint32_t)enable_d) : "memory");
}
// Multicast commit: arrives on the supplied mbar in every CTA whose bit is set in
// the mask.  cta_mask = (1 << CTA_GROUP) - 1 = 0b11 → both CTAs.
__device__ __forceinline__ void signal_on_mma_completion(uint32_t smem_bar, int16_t cta_mask) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
        "[%0], %1;"
        :: "r"(smem_bar), "h"(cta_mask) : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}
__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
// ── tcgen05.ld width helpers (building block) ───────────────────────
// mvp_core splices these at the TCGEN05_LD marker in every tier, so the
// TMEM->register load width (TCGEN05_LD_WIDTH = 8/16 32-bit elems per lane)
// is one knob with the asm in a single place.  Wider = fewer ld + fewer
// wait_ld syncs (more registers, but we're SMEM-occupancy-bound so it's free).
// The epilogue picks the variant via `#if` (resolved at generation time).

__device__ __forceinline__ void tcgen05_ld_32x32b_x8(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        :
          "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
          "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7])
        : "r"(taddr));
}

__device__ __forceinline__ void tcgen05_ld_32x32b_x16(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        :
          "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
          "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7]),
          "=f"(out[8]), "=f"(out[9]), "=f"(out[10]), "=f"(out[11]),
          "=f"(out[12]), "=f"(out[13]), "=f"(out[14]), "=f"(out[15])
        : "r"(taddr));
}


// ── mbarrier helpers ────────────────────────────────────────────────
__device__ __forceinline__ void mbarrier_init(uint32_t mb, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mb), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive_no_tx(uint32_t mb) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(mb) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive_no_tx_cluster(uint32_t mb) {
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                 :: "r"(mb) : "memory");
}
__device__ __forceinline__ void signal_on_bytes_loaded(uint32_t mb, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(mb), "r"(bytes) : "memory");
}
__device__ __forceinline__ void wait_phase(uint32_t mb, uint32_t phase) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE_%=;\n\t bra WAIT_%=;\n\t DONE_%=:\n\t }"
        :: "r"(mb), "r"(phase) : "memory");
}


// ── Kernel (NS, GROUP_SIZE_M are file-level constexpr knobs) ────────
// ── TMA store helpers (pipelined TMA-store epilogue) ────────────────
__device__ __forceinline__ void tma_2d_store(
    const void* tmap, uint32_t smem_src, int x, int y
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
        "[%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_src) : "memory");
}
__device__ __forceinline__ void tma_commit_group() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void tma_wait_group() {
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory");
}

// MMA-issue building block (shared fragment).  The cluster tier uses the
// g2 (cta_group::2) MMA instruction.
#define MMA_ISSUE(t, a, b, i, e) tcgen05_mma_g2((t), (a), (b), (i), (e))
// ── MMA-issue chain (building block) ────────────────────────────────
// Issues the K_MMAS tcgen05 MMAs for one K-tile (slot) into the
// accumulator at `taddr`.  mvp_core stitches this into every tier at the
// MMA-chain marker, so the descriptor math + K-step loop live in exactly
// one place.  The only per-tier variation is the MMA instruction
// itself, supplied just before the marker as:
//   MMA_ISSUE(taddr, a_desc, b_desc, idesc, enable_d)
// → tcgen05_mma (single-CTA) or tcgen05_mma_g2 (2-CTA cluster).
__device__ __forceinline__ void issue_mma_chain(
    uint32_t taddr, uint32_t a_base_slot, uint32_t b_base_slot,
    uint32_t idesc, bool first_k_tile)
{
    #pragma unroll
    for (int kk = 0; kk < K_MMAS; kk++) {
        const uint64_t a_desc = make_desc(a_base_slot + kk * MMA_K * BF16_BYTES);
        const uint64_t b_desc = make_desc_K_major(
            b_base_slot + kk * MMA_K * SWIZZLE_ROW_BYTES, BK * SWIZZLE_ROW_BYTES);
        const bool first_ever = first_k_tile && (kk == 0);
        MMA_ISSUE(taddr, a_desc, b_desc, idesc, !first_ever);
    }
}
#undef MMA_ISSUE

__device__ __forceinline__ void matmul_cluster_impl(
    const CUtensorMap* A_tmap,
    const CUtensorMap* B_tmap,
    const CUtensorMap* C_tmap_ptr,
    __nv_bfloat16* __restrict__ C_ptr,
    int M, int N, int K
) {
    // ── Per-cluster + per-CTA tile coords ───────────────────────────
    //
    // Grid is ceil(M / (CTA_GROUP*BM)) * ceil(N / BN) flat CTA ids.  Each
    // *pair* of CTAs forms one cluster; cta_rank picks which CTA in
    // the pair owns which half.  Ragged edge tiles are clipped by TMA.
    //
    // bid (the cluster id derived from blockIdx.x / CTA_GROUP) is what
    // we'd normally call the grid coordinate; the cluster handles a
    // 2*BM × BN output tile.
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Tile coords (the GSM chunked-walk swizzle) are computed PER-TILE
    // inside each path's persistent loop below — both the overlap and the
    // non-overlap branch derive (cluster_m, cluster_n) from their own
    // cluster id, so there are no tile-specific coords at this scope.

    // ── SMEM (per CTA — B is now half-width) ────────────────────────
    extern __shared__ __align__(1024) char smem[];
    const uint32_t SMEM_BASE = (uint32_t)__cvta_generic_to_shared(smem);
    auto A_base = [SMEM_BASE](int s) -> uint32_t {
        return SMEM_BASE + s * SLOT_BYTES;
    };
    auto B_base = [SMEM_BASE](int s) -> uint32_t {
        return SMEM_BASE + s * SLOT_BYTES + A_SLOT_BYTES;
    };

    __shared__ uint64_t mbar_compute_data_ready[NS];
    __shared__ uint64_t mbar_compute_buffer_free[NS];
    __shared__ uint64_t all_mmas_done;
    __shared__ uint32_t tmem_addr_holder[1];

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane    = tid % WARP_SIZE;

    {
        // Persistent cluster pipeline: both CTAs stream A/B, CTA 0 issues
        // cta_group::2 MMA into a two-buffer TMEM accumulator, and every CTA
        // drains its own BM x BN output half while the next cluster tile runs.
        __shared__ uint64_t mbar_tmem_data_ready[2];
        __shared__ uint64_t mbar_tmem_buffer_free[2];
        // Pipelined TMA-store mode keeps the K-loop ring intact and reserves
        // compact 128B-swizzled SMEM buffers for chunked TMA stores.
        constexpr int STORE_BUF_BYTES = BM * STORE_N * BF16_BYTES;
        const uint32_t STORE_SMEM_BASE = SMEM_BASE + NS * SLOT_BYTES;
        auto C_store = reinterpret_cast<__nv_bfloat16*>(smem + NS * SLOT_BYTES);

        if (warp_id == 0) {
            tcgen05_alloc_g2((uint32_t)__cvta_generic_to_shared(tmem_addr_holder), BN);
        }
        if (warp_id == 0 && elect_sync()) {
            #pragma unroll
            for (int s = 0; s < NS; s++) {
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_compute_data_ready[s]), CTA_GROUP);
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_compute_buffer_free[s]), 1);
                mbarrier_arrive_no_tx((uint32_t)__cvta_generic_to_shared(&mbar_compute_buffer_free[s]));
            }
            #pragma unroll
            for (int b = 0; b < 2; b++) {
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_tmem_data_ready[b]), 1);
                mbarrier_init((uint32_t)__cvta_generic_to_shared(&mbar_tmem_buffer_free[b]), CTA_GROUP);
                uint32_t tmem_buffer_free_cta0 =
                    ((uint32_t)__cvta_generic_to_shared(&mbar_tmem_buffer_free[b])) & 0xFEFFFFFFu;
                mbarrier_arrive_no_tx_cluster(tmem_buffer_free_cta0);
            }
            asm volatile("fence.mbarrier_init.release.cluster;");
        }

        asm volatile("barrier.cluster.arrive.release.aligned;");
        asm volatile("barrier.cluster.wait.acquire.aligned;");

        const uint32_t taddr = tmem_addr_holder[0];
        constexpr int BN_PANEL = 256;
        constexpr int BN_PANEL_LOCAL = BN_PANEL / CTA_GROUP;
        const uint32_t idesc = make_idesc_bf16_cluster(CTA_GROUP * BM, BN_PANEL);
        const int num_k = K / BK;
        constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;

        // ceil-div tile counts: a ragged M/N launches partial edge tiles whose
        // TMA box is clipped out of bounds (zero-fill on load, masked on store).
        const int grid_m_clusters = (M + CTA_GROUP * BM - 1) / (CTA_GROUP * BM);
        const int grid_n          = (N + BN - 1) / BN;
        const int num_cluster_in_group = GROUP_SIZE_M * grid_n;
        const int num_clusters = grid_m_clusters * grid_n;
        const int cluster_pid = (int)blockIdx.x / CTA_GROUP;
        const int cluster_stride = (int)gridDim.x / CTA_GROUP;
        const int num_my = (cluster_pid >= num_clusters) ? 0
                         : (num_clusters - cluster_pid + cluster_stride - 1) / cluster_stride;

        auto map_off = [&](int ti, int& base_m, int& base_n, int& local_m, int& local_n) {
            int tile = cluster_pid + ti * cluster_stride;
            int group = tile / num_cluster_in_group;
            int first = group * GROUP_SIZE_M;
            int gsm_i = min(grid_m_clusters - first, GROUP_SIZE_M);
            int cm = first + (tile % gsm_i);
            int cn = (tile % num_cluster_in_group) / gsm_i;
            base_m = cm * (CTA_GROUP * BM);
            base_n = cn * BN;
            local_m = base_m + cta_rank * BM;
            local_n = base_n + cta_rank * BN_LOCAL;
        };

        if (warp_id == 0 && elect_sync()) {
            uint32_t compute_buffer_free_phase[NS] = {};
            long gk = 0;
            for (int ti = 0; ti < num_my; ti++) {
                int base_m, base_n, local_m, local_n;
                map_off(ti, base_m, base_n, local_m, local_n);
                for (int k = 0; k < num_k; k++) {
                    int slot = gk % NS;
                    uint32_t compute_buffer_free_addr =
                        (uint32_t)__cvta_generic_to_shared(&mbar_compute_buffer_free[slot]);
                    uint32_t compute_data_ready_cta0 =
                        ((uint32_t)__cvta_generic_to_shared(&mbar_compute_data_ready[slot])) & 0xFEFFFFFFu;
                    wait_phase(compute_buffer_free_addr, compute_buffer_free_phase[slot]);
                    tma_2d_load_g2(A_base(slot), A_tmap, k * BK, local_m, compute_data_ready_cta0);
                    #pragma unroll
                    for (int panel = 0; panel < 2; panel++) {
                        #pragma unroll
                        for (int n = 0; n < BN_PANEL_LOCAL; n += 64) {
                            tma_2d_load_g2(
                                B_base(slot) + (panel * BN_PANEL_LOCAL + n) * BK * BF16_BYTES,
                                B_tmap,
                                base_n + panel * BN_PANEL + cta_rank * BN_PANEL_LOCAL + n,
                                k * BK,
                                compute_data_ready_cta0);
                        }
                    }
                    signal_on_bytes_loaded(compute_data_ready_cta0, SLOT_BYTES);
                    compute_buffer_free_phase[slot] ^= 1;
                    gk++;
                }
            }
        } else if (cta_rank == 0 && warp_id == 1 && elect_sync()) {
            uint32_t compute_data_ready_phase[NS] = {};
            uint32_t tmem_buffer_free_phase[2] = {};
            long gk = 0;
            for (int ti = 0; ti < num_my; ti++) {
                const int buf = 0;
                uint32_t d_tmem = taddr;
                wait_phase((uint32_t)__cvta_generic_to_shared(&mbar_tmem_buffer_free[buf]),
                                    tmem_buffer_free_phase[buf]);
                tmem_buffer_free_phase[buf] ^= 1;
                for (int k = 0; k < num_k; k++) {
                    int slot = gk % NS;
                    uint32_t compute_data_ready_addr =
                        (uint32_t)__cvta_generic_to_shared(&mbar_compute_data_ready[slot]);
                    uint32_t compute_buffer_free_addr =
                        (uint32_t)__cvta_generic_to_shared(&mbar_compute_buffer_free[slot]);
                    wait_phase(compute_data_ready_addr, compute_data_ready_phase[slot]);
                    tcgen05_fence_after_thread_sync();
                    issue_mma_chain(d_tmem,
                                    A_base(slot),
                                    B_base(slot),
                                    idesc,
                                    /*first_k_tile=*/ k == 0);
                    issue_mma_chain(d_tmem + BN_PANEL,
                                    A_base(slot),
                                    B_base(slot) + BN_PANEL_LOCAL * BK * BF16_BYTES,
                                    idesc,
                                    /*first_k_tile=*/ k == 0);
                    signal_on_mma_completion(compute_buffer_free_addr, cta_mask);
                    compute_data_ready_phase[slot] ^= 1;
                    gk++;
                }
                signal_on_mma_completion((uint32_t)__cvta_generic_to_shared(&mbar_tmem_data_ready[buf]), cta_mask);
            }
        } else if (warp_id >= 4 && warp_id < NUM_WARPS + 4) {
            // Contract for the shared overlap-drain fragment: cluster tier writes
            // this CTA's BM x BN output half (local_m / base_n) and releases the
            // TMEM buffer with a CTA-0-masked cluster arrive.
#define EPI_OUT_ROW                 local_m
#define EPI_OUT_COL_BASE            base_n
#define signal_sync(buf)   do { uint32_t _f = ((uint32_t)__cvta_generic_to_shared(&mbar_tmem_buffer_free[buf])) & 0xFEFFFFFFu; mbarrier_arrive_no_tx_cluster(_f); } while (0)
            constexpr int ROW_STRIPS    = BM / 32;
            constexpr int COL_GROUPS    = NUM_WARPS / ROW_STRIPS;
            constexpr int COLS_PER_WARP = BN / COL_GROUPS;
            constexpr int EPI_THREADS   = NUM_WARPS * 32;
            const int ew = warp_id - 4;
            const int row_warp = ew % ROW_STRIPS;
            const int col_warp = ew / ROW_STRIPS;
            const int my_row = row_warp * 32 + lane;
            const int col_base = col_warp * COLS_PER_WARP;
            const int etid = ew * 32 + lane;
            uint32_t full[2] = {};
            for (int ti = 0; ti < num_my; ti++) {
                int base_m, base_n, local_m, local_n;
                const int buf = 0;
                map_off(ti, base_m, base_n, local_m, local_n);
                wait_phase((uint32_t)__cvta_generic_to_shared(&mbar_tmem_data_ready[buf]), full[buf]);
                full[buf] ^= 1;
                tcgen05_fence_after_thread_sync();
                const uint32_t trow =
                    (taddr + buf * BN) + ((uint32_t)(cta_rank * BM + row_warp * 32) << 16);
                constexpr int LDW = TCGEN05_LD_WIDTH;

                // ── Overlap epilogue drain (TMEM → SMEM → GMEM), shared ─────
                // Spliced into the overlap epilogue-warp loop of every warp-spec
                // tier's drain marker, right after `trow` (the tier-specific TMEM
                // lane base) and `LDW` are in scope.  The skeleton supplies three
                // contract macros for the per-tier bits:
                //   EPI_OUT_ROW                  this CTA's GMEM row base
                //   EPI_OUT_COL_BASE             this CTA's GMEM column base
                //   signal_sync(buf)    release the drained TMEM buffer
                // EPILOGUE_TMA_PIPELINED picks the Paul-v6-style path:
                // chunk BN into STORE_N=64 columns, stage each chunk into one
                // of TMA_STORE_STAGES compact swizzled SMEM buffers, and
                // launch TMA stores.
                // EPILOGUE_SPLIT (constexpr) picks the two-pass half-BN writeback,
                // which stages one BN/2 column panel at a time (EPI_STAGE_COLS=BN/2)
                // so the epilogue SMEM shrinks enough for an extra K-loop stage.
                //
                // EPILOGUE_L1_NO_ALLOC (knob): the write-once C store bypasses L1
                // allocation (`st...L1::no_allocate`) so it doesn't evict A/B from
                // L1.  Measured win when the epilogue is exposed (low K), null at
                // high K — so it's a sweep knob, not always-on.
                {
                    constexpr int LOADS_PER_CHUNK = STORE_N / 8;
                    constexpr int LOADS_PER_WARP = LOADS_PER_CHUNK / COL_GROUPS;
                    constexpr int NUM_CHUNKS = BN / STORE_N;
                    static_assert(STORE_N == 64, "pipelined TMA store assumes STORE_N=64");
                    static_assert(NUM_CHUNKS * STORE_N == BN, "BN must divide into STORE_N chunks");
                    static_assert(LOADS_PER_WARP * COL_GROUPS == LOADS_PER_CHUNK,
                                  "STORE_N/8 chunks must divide across column warp groups");
                    int store_stage = 0;

                    #pragma unroll
                    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                        float t[LOADS_PER_WARP][8];
                        #pragma unroll
                        for (int n = 0; n < LOADS_PER_WARP; n++) {
                            const int local_n = col_warp * LOADS_PER_WARP + n;
                            tcgen05_ld_32x32b_x8(trow + (uint32_t)(chunk * STORE_N + local_n * 8), t[n]);
                        }
                        tcgen05_wait_ld();


                        // The TMEM->reg load above doesn't touch the store buffer, so the
                        // free-store-slot wait below is deferred to just before the buffer write
                        // (and stays before the bar.sync so every warp observes the ew==0 wait).
                        if (chunk == NUM_CHUNKS - 1)
                            tcgen05_fence_before_thread_sync();

                        if (ew == 0)
                            tma_wait_group<TMA_STORE_STAGES - 1>();

                        asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));

                        if (chunk == NUM_CHUNKS - 1) {
                            if (ew == 0 && elect_sync())
                                signal_sync(buf);
                        }

                        #pragma unroll
                        for (int n = 0; n < LOADS_PER_WARP; n++) {
                            __nv_bfloat162 pk[4];
                            #pragma unroll
                            for (int i = 0; i < 4; i++)
                                pk[i] = __floats2bfloat162_rn(mmc_epi(t[n][2 * i]), mmc_epi(t[n][2 * i + 1]));
                            const int local_n = col_warp * LOADS_PER_WARP + n;
                            const int swizzled_n = local_n ^ (my_row & 7);
                            __nv_bfloat16* write_ptr =
                                C_store + store_stage * BM * STORE_N + my_row * STORE_N + swizzled_n * 8;
                            *reinterpret_cast<int4*>(write_ptr) = *reinterpret_cast<int4*>(pk);
                        }

                        __syncwarp();
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));

                        if (ew == 0 && elect_sync()) {
                            const uint32_t src = STORE_SMEM_BASE + store_stage * STORE_BUF_BYTES;
                            tma_2d_store(C_tmap_ptr, src,
                                         EPI_OUT_COL_BASE + chunk * STORE_N, EPI_OUT_ROW);
                            tma_commit_group();
                        }

                        store_stage ^= 1;
                    }
                }
            }
            if (ew == 0)
                tma_wait_group<0>();
            asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));
#undef EPI_OUT_ROW
#undef EPI_OUT_COL_BASE
#undef signal_sync
        }

        __syncthreads();
        if (warp_id == 0 && elect_sync()) {
            tcgen05_dealloc_g2(taddr, BN);
        }
        return;
    }
}


// ── Single entry symbol — NS and GROUP_SIZE_M are baked in from the
// constexpr knobs at the top of the file (the webui substitutes them).
extern "C" __global__ __cluster_dims__(CTA_GROUP, 1, 1) __launch_bounds__(LAUNCH_THREADS, 1)
void matmul_cluster(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ CUtensorMap C_tmap,
    __nv_bfloat16* C_ptr, int M, int N, int K
)
{
    matmul_cluster_impl(&A_tmap, &B_tmap, &C_tmap, C_ptr, M, N, K
    );
}

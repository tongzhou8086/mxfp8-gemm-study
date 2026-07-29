#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "kittens.cuh"

// First MXFP8 variant: keep the BN256 2-CTA persistent pipeline shape, but use
// a single TMEM accumulator so scale-factor TMEM has room.
constexpr int BM           = 128;
constexpr int BN           = 256;
constexpr int BK           = 128;
constexpr int NS           = 5;
constexpr int GROUP_SIZE_M = 8;
constexpr int NUM_WARPS    = 4;
constexpr int CTA_GROUP    = 2;
constexpr int BN_LOCAL     = BN / CTA_GROUP;
constexpr int STORE_N      = 64;
constexpr int TMA_STORE_STAGES = 2;

constexpr int WARP_SIZE = 32;
constexpr int THREADS   = NUM_WARPS * WARP_SIZE;
constexpr int LAUNCH_THREADS = (NUM_WARPS + 4) * WARP_SIZE;

using a_tile = kittens::st_fp8e4m3<BM, BK>;
using b_tile = kittens::st_fp8e4m3<BN_LOCAL, BK>;
using sf_tile = kittens::st_fp8e8m0<32, 16, false>;
using d_tile = kittens::st_bf<BM, STORE_N>;

struct alignas(16) ScaleAtom {
    uint8_t data[32][16];
};
static_assert(sizeof(ScaleAtom) == 32 * 16);

using a_gl = kittens::gl<kittens::fp8e4m3, 1, 1, -1, -1, a_tile>;
using b_gl = kittens::gl<kittens::fp8e4m3, 1, 1, -1, -1, b_tile>;
using sf_gl = kittens::gl<kittens::fp8e8m0, -1, -1, 32, 16, sf_tile>;
using d_gl = kittens::gl<kittens::bf16, 1, 1, -1, -1, d_tile>;

struct TkMxfp8Globals {
    a_gl A;       // A[M, K], FP8 E4M3
    sf_gl A_sc;   // swizzled E8M0 scales [M/128, K/128, 32, 16]
    b_gl B;       // B[N, K], FP8 E4M3, consumed as B^T by ABt MMA
    sf_gl B_sc;   // swizzled E8M0 scales [N/128, K/128, 32, 16]
    d_gl D;       // D[M, N], BF16
};

__device__ __forceinline__ float mmc_epi(float x) { return x; }

__device__ __forceinline__ bool elect_sync() {
    // Use the warp-level election primitive, not a lane-0 predicate.  The
    // explicit lane check is logically similar here but generates slower
    // control flow in the persistent pipeline.
    uint32_t elected = 0;
    asm volatile(
        "{.reg .pred P;\n"
        " elect.sync _|P, %1;\n"
        " selp.u32 %0, 1, 0, P;}\n"
        : "+r"(elected)
        : "r"(0xFFFFFFFF));
    return static_cast<bool>(elected);
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;\n");
}

__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;\n");
}

__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__ void tcgen05_ld_32x32b_x8(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        :
          "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
          "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7])
        : "r"(taddr));
}

namespace cuda_tcgen05 {

__device__ __forceinline__ uint32_t shared_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint32_t shared_u32_in_cta(const void* ptr, int cta_rank) {
    uint32_t local_addr = shared_u32(ptr);
    uint32_t mapped_addr;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(mapped_addr)
        : "r"(local_addr), "r"(cta_rank));
    return mapped_addr;
}

__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) {
    return (x & 0x3FFFF) >> 4;
}

__device__ __forceinline__ uint64_t make_scale_smem_desc(const void* smem_ptr) {
    // Matches the descriptor TK builds for tcgen05.cp on a non-swizzled
    // 32x16 E8M0 source tile.  The 128 offsets are the ISA descriptor
    // units used by the 32x128b copy shape.
    uint64_t desc = matrix_descriptor_encode(reinterpret_cast<uint64_t>(smem_ptr));
    desc |= 1ull << 46;  // Blackwell shared-memory descriptor bit.
    desc |= matrix_descriptor_encode(128ull) << 16;
    desc |= matrix_descriptor_encode(128ull) << 32;
    return desc;
}

template <typename SharedTile>
__device__ __forceinline__ void load_fp8_operand_tile_from_gmem_to_smem(
    SharedTile& dst,
    const CUtensorMap* operand_tensor_map,
    int row_start,
    int k_tile,
    kittens::semaphore& completion_barrier,
    uint16_t destination_cta_mask,
    int completion_barrier_cta = 0
) {
    static_assert(SharedTile::swizzle, "FP8 operand TMA tiles must use the 128B swizzled layout");
    static_assert(SharedTile::swizzle_bytes == 128, "This wrapper mirrors TK's 128B swizzled TMA path");
    constexpr int swizzle_elements =
        SharedTile::swizzle_bytes / sizeof(typename SharedTile::dtype);
    static_assert(SharedTile::cols % swizzle_elements == 0);
    constexpr int k_tma_step = SharedTile::cols / swizzle_elements;

    const uint32_t dst_addr = shared_u32(&dst);
    const uint32_t mbar_addr = shared_u32_in_cta(&completion_barrier, completion_barrier_cta);
    const uint64_t tma_addr = reinterpret_cast<uint64_t>(operand_tensor_map);
    const int k_swizzle_tile = k_tile * k_tma_step;

    // Mirrors kittens::tma::cluster::load_async for a 128B-swizzled ST tile
    // with axis=ROW.  The CUtensorMap has innermost-first dimensions:
    //   [128B swizzle group, logical rows, logical K / 128B, depth, batch].
    // So a logical tile at rows row_start..row_start+rows-1 and K stage k
    // is requested with TMA coords {0, row_start, k_swizzle_tile, 0, 0}.
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::cluster"
        " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
        :
        : "r"(dst_addr), "l"(tma_addr), "r"(mbar_addr),
          "n"(0), "r"(row_start), "r"(k_swizzle_tile), "r"(0), "r"(0),
          "h"(destination_cta_mask)
        : "memory");
}

__device__ __forceinline__ void load_scale_atom_from_gmem_to_smem(
    ScaleAtom& dst,
    const CUtensorMap* scale_tensor_map,
    int outer_tile,
    int k_tile,
    kittens::semaphore& completion_barrier,
    uint16_t destination_cta_mask,
    int completion_barrier_cta = 0
) {
    const uint32_t dst_addr = shared_u32(&dst.data[0][0]);
    const uint32_t mbar_addr = shared_u32_in_cta(&completion_barrier, completion_barrier_cta);
    const uint64_t tma_addr = reinterpret_cast<uint64_t>(scale_tensor_map);

    // The scale CUtensorMap is 4D with innermost-first dimensions:
    //   [16, 32, K/128, outer].
    // Therefore the requested tile coordinate is:
    //   {col=0, row=0, k_tile, outer_tile}.
    //
    // On SM100, the cta_group::2 form is what lets the transaction complete
    // into CTA 0's mbarrier even when CTA 1 issues the load.  The destination
    // mask controls whether the payload lands in one CTA or is multicast.
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::cluster"
        " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
        :
        : "r"(dst_addr), "l"(tma_addr), "r"(mbar_addr),
          "r"(0), "r"(0), "r"(k_tile), "r"(outer_tile), "h"(destination_cta_mask)
        : "memory");
}

__device__ __forceinline__ void copy_scale_atom_from_smem_to_tmem(
    uint32_t tmem_addr,
    const ScaleAtom& src
) {
    uint64_t st_desc = make_scale_smem_desc(&src.data[0][0]);
    asm volatile(
        "{tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;}"
        :: "r"(tmem_addr), "l"(st_desc));
}

template <typename SharedTile>
__device__ __forceinline__ uint64_t make_k_major_operand_smem_desc(
    const SharedTile& tile
) {
    static_assert(SharedTile::swizzle, "tcgen05 operand tiles must be swizzled");
    static_assert(SharedTile::swizzle_bytes == 128, "This kernel uses 128B swizzled FP8 operand tiles");
    static_assert(sizeof(typename SharedTile::dtype) == 1, "This helper is specialized for FP8 operands");

    uint64_t desc = matrix_descriptor_encode(reinterpret_cast<uint64_t>(&tile.data[0]));
    desc |= 1ull << 46;  // Blackwell shared-memory descriptor bit.
    desc |= matrix_descriptor_encode(16ull) << 16;    // ignored in K-major mode
    desc |= matrix_descriptor_encode(1024ull) << 32;  // 128B swizzle stride
    desc |= 1ull << 62;  // 128B swizzle mode
    return desc;
}

template <typename SharedTile>
__device__ __forceinline__ uint64_t k_major_operand_chunk_desc(
    uint64_t base_desc,
    int chunk_idx
) {
    static_assert(SharedTile::swizzle_bytes == 128);
    constexpr int tile_row_dim = 16;
    const int byte_offset =
        (chunk_idx % 4) * 32 +
        (chunk_idx / 4) * (SharedTile::rows / tile_row_dim) * 2048;
    return base_desc + matrix_descriptor_encode(static_cast<uint64_t>(byte_offset));
}

template <int ScaleFactorId>
__device__ __forceinline__ constexpr uint32_t mxfp8_mma2_abt_256x256_desc() {
    static_assert(ScaleFactorId >= 0 && ScaleFactorId < 4);
    constexpr int mma_m = CTA_GROUP * BM;
    constexpr int mma_n = BN;

    uint32_t desc = 0;
    desc |= 0b00 << 0;                 // dense, no sparsity
    desc |= 0b0 << 2;                  // dense
    desc |= 0b0 << 3;                  // no saturate
    desc |= ScaleFactorId << 4;        // B scale-factor id
    desc |= 0b000 << 7;                // A is E4M3
    desc |= 0b000 << 10;               // B is E4M3
    desc |= 0b0 << 13;                 // do not negate A
    desc |= 0b0 << 14;                 // do not negate B
    desc |= 0b0 << 15;                 // block-scaled MMA: no transpose bits
    desc |= 0b0 << 16;
    desc |= (mma_n >> 3) << 17;        // N dimension
    desc |= 0b1 << 23;                 // E8M0 scales
    desc |= 0b000 << 24;
    desc |= (mma_m >> 7) << 27;        // M dimension
    desc |= ScaleFactorId << 29;       // A scale-factor id
    desc |= 0b0u << 31;                // MXFP8 K chunk is 32B
    return desc;
}

template <int Accumulate>
__device__ __forceinline__ void issue_mxfp8_mma2_abt_chunk(
    uint32_t dst_tmem_addr,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t a_scale_tmem_addr,
    uint32_t b_scale_tmem_addr,
    uint32_t instruction_desc
) {
    static_assert(Accumulate == 0 || Accumulate == 1);
    asm volatile(
        "{.reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X "
        "[%0], %1, %2, %3, [%4], [%5], p;}\n"
        :
        : "r"(dst_tmem_addr), "l"(a_desc), "l"(b_desc), "r"(instruction_desc),
          "r"(a_scale_tmem_addr), "r"(b_scale_tmem_addr), "n"(Accumulate));
}

__device__ __forceinline__ void commit_mma_arrive_cta_group_2(
    kittens::semaphore& sem,
    uint16_t cta_mask
) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n"
        :: "l"(__cvta_generic_to_shared(&sem)), "h"(cta_mask));
}

template <int Accumulate, typename AccumTensor, typename AScaleTensor, typename BScaleTensor>
__device__ __forceinline__ void issue_mxfp8_mma2_abt_128x256x128(
    const AccumTensor& dst,
    const a_tile& a,
    const b_tile& b,
    const AScaleTensor& a_scale,
    const BScaleTensor& b_scale,
    kittens::semaphore& smem_buffer_free
) {
    static_assert(BM == 128 && BN == 256 && BK == 128);
    static_assert(CTA_GROUP == 2 && BN_LOCAL == 128);

    constexpr uint32_t idescs[4] = {
        mxfp8_mma2_abt_256x256_desc<0>(),
        mxfp8_mma2_abt_256x256_desc<1>(),
        mxfp8_mma2_abt_256x256_desc<2>(),
        mxfp8_mma2_abt_256x256_desc<3>(),
    };
    const uint64_t a_base = make_k_major_operand_smem_desc(a);
    const uint64_t b_base = make_k_major_operand_smem_desc(b);

    issue_mxfp8_mma2_abt_chunk<Accumulate>(
        dst.addr,
        k_major_operand_chunk_desc<a_tile>(a_base, 0),
        k_major_operand_chunk_desc<b_tile>(b_base, 0),
        a_scale.addr, b_scale.addr, idescs[0]);

    #pragma unroll
    for (int sfid = 1; sfid < 4; ++sfid) {
        issue_mxfp8_mma2_abt_chunk<1>(
            dst.addr,
            k_major_operand_chunk_desc<a_tile>(a_base, sfid),
            k_major_operand_chunk_desc<b_tile>(b_base, sfid),
            a_scale.addr, b_scale.addr, idescs[sfid]);
    }

    commit_mma_arrive_cta_group_2(smem_buffer_free, static_cast<uint16_t>((1 << CTA_GROUP) - 1));
}

struct TensorMemoryAllocatorCtaGroup2 {
    static constexpr int cols = 512;
    uint32_t addr = 0;

    __device__ __forceinline__ void provision(uint32_t& shared_addr) {
        asm volatile(
            "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;\n"
            :: "l"(reinterpret_cast<uint64_t>(&shared_addr)), "n"(cols));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n");
    }

    __device__ __forceinline__ void set_addr(uint32_t tmem_addr) {
        addr = tmem_addr;
    }

    template <typename TensorTile>
    __device__ __forceinline__ TensorTile allocate(int col_offset) const {
        return TensorTile(addr + col_offset);
    }

    __device__ __forceinline__ void deprovision() const {
        asm volatile(
            "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;\n"
            :: "r"(addr), "n"(cols));
    }
};

} // namespace cuda_tcgen05

__device__ __forceinline__ void mbarrier_init(kittens::semaphore& sem, int count) {
    const uint32_t mbar_addr = cuda_tcgen05::shared_u32(&sem);
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(mbar_addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_no_tx(kittens::semaphore& sem) {
    const uint32_t mbar_addr = cuda_tcgen05::shared_u32(&sem);
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
        :: "r"(mbar_addr)
        : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_no_tx_cluster(kittens::semaphore& sem) {
    const uint32_t mbar_addr = cuda_tcgen05::shared_u32_in_cta(&sem, 0);
    asm volatile(
        "mbarrier.arrive.shared::cluster.b64 _, [%0], %1;\n"
        :: "r"(mbar_addr), "r"(1)
        : "memory");
}

__device__ __forceinline__ void signal_on_bytes_loaded(kittens::semaphore& sem, int bytes) {
    const uint32_t mbar_addr = cuda_tcgen05::shared_u32_in_cta(&sem, 0);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;\n"
        :: "r"(mbar_addr), "r"(bytes));
}

__device__ __forceinline__ void signal_on_mma_completion(
    kittens::semaphore& sem, uint16_t cta_mask
) {
    cuda_tcgen05::commit_mma_arrive_cta_group_2(sem, cta_mask);
}

__device__ __forceinline__ void wait_phase(kittens::semaphore& sem, uint32_t phase) {
    const uint32_t mbar_addr = cuda_tcgen05::shared_u32(&sem);
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1 bra.uni DONE;\n"
        "bra.uni LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_addr), "r"(phase));
}

template <int N>
__device__ __forceinline__ void tma_wait_group() {
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory");
}

__device__ __forceinline__ void matmul_cluster_impl(
    const TkMxfp8Globals& g,
    __nv_bfloat16*,
    int M, int N, int K
) {
    int cta_rank = kittens::cluster_ctarank();

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<a_tile>();
        g.A_sc.template prefetch_tma<sf_tile>();
        g.B.template prefetch_tma<b_tile>();
        g.B_sc.template prefetch_tma<sf_tile>();
        g.D.template prefetch_tma<d_tile>();
    }

    extern __shared__ int __shm[];
    kittens::tma_swizzle_allocator al((int*)&__shm[0]);

    a_tile (&a_smem)[NS] = al.allocate<a_tile, NS>();
    b_tile (&b_smem)[NS] = al.allocate<b_tile, NS>();
    ScaleAtom (&a_sc_smem)[NS] = al.allocate<ScaleAtom, NS>();
    ScaleAtom (&b_sc_smem)[NS][CTA_GROUP] = al.allocate<ScaleAtom, NS, CTA_GROUP>();
    d_tile (&d_smem)[TMA_STORE_STAGES] = al.allocate<d_tile, TMA_STORE_STAGES>();

    __shared__ kittens::semaphore mbar_compute_data_ready[NS];
    __shared__ kittens::semaphore mbar_compute_buffer_free[NS];
    __shared__ uint32_t tmem_addr_holder[1];

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane    = tid % WARP_SIZE;
    cuda_tcgen05::TensorMemoryAllocatorCtaGroup2 tmem_allocator{};

    __shared__ kittens::semaphore mbar_tmem_data_ready;
    __shared__ kittens::semaphore mbar_tmem_buffer_free;

    if (warp_id == 0) {
        tmem_allocator.provision(tmem_addr_holder[0]);
    }
    if (warp_id == 0 && elect_sync()) {
        #pragma unroll
        for (int s = 0; s < NS; s++) {
            mbarrier_init(mbar_compute_data_ready[s], CTA_GROUP);
            mbarrier_init(mbar_compute_buffer_free[s], 1);
            mbarrier_arrive_no_tx(mbar_compute_buffer_free[s]);
        }
        mbarrier_init(mbar_tmem_data_ready, 1);
        mbarrier_init(mbar_tmem_buffer_free, CTA_GROUP);
        mbarrier_arrive_no_tx_cluster(mbar_tmem_buffer_free);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    kittens::everyone::tma::cluster::sync();

    const uint32_t taddr = tmem_addr_holder[0];
    const int num_k = K / BK;
    constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;

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
        int remaining_m_groups = grid_m_clusters - first;
        int gsm_i = remaining_m_groups < GROUP_SIZE_M ? remaining_m_groups : GROUP_SIZE_M;
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
                wait_phase(mbar_compute_buffer_free[slot], compute_buffer_free_phase[slot]);

                cuda_tcgen05::load_fp8_operand_tile_from_gmem_to_smem(
                    a_smem[slot],
                    g.A.template get_tma<a_tile, kittens::dim::ROW>(),
                    local_m, k,
                    mbar_compute_data_ready[slot],
                    static_cast<uint16_t>(1 << cta_rank), 0);
                cuda_tcgen05::load_fp8_operand_tile_from_gmem_to_smem(
                    b_smem[slot],
                    g.B.template get_tma<b_tile, kittens::dim::ROW>(),
                    local_n, k,
                    mbar_compute_data_ready[slot],
                    static_cast<uint16_t>(1 << cta_rank), 0);
                cuda_tcgen05::load_scale_atom_from_gmem_to_smem(
                    a_sc_smem[slot],
                    g.A_sc.template get_tma<sf_tile, kittens::dim::ROW>(),
                    local_m / BM, k,
                    mbar_compute_data_ready[slot],
                    static_cast<uint16_t>(1 << cta_rank));

                cuda_tcgen05::load_scale_atom_from_gmem_to_smem(
                    b_sc_smem[slot][cta_rank],
                    g.B_sc.template get_tma<sf_tile, kittens::dim::ROW>(),
                    local_n / BN_LOCAL, k,
                    mbar_compute_data_ready[slot],
                    static_cast<uint16_t>(cta_mask));

                constexpr int slot_bytes =
                    sizeof(a_tile) + sizeof(b_tile) + sizeof(ScaleAtom) * (1 + CTA_GROUP);
                signal_on_bytes_loaded(mbar_compute_data_ready[slot], slot_bytes);
                compute_buffer_free_phase[slot] ^= 1;
                gk++;
            }
        }
    } else if (cta_rank == 0 && warp_id == 1 && elect_sync()) {
        uint32_t compute_data_ready_phase[NS] = {};
        uint32_t tmem_buffer_free_phase = 0;
        long gk = 0;

        tmem_allocator.set_addr(taddr);
        auto out_tm = tmem_allocator.template allocate<kittens::full_tt_fl<BN>>(0);
        auto a_sc_tm = tmem_allocator.template allocate<kittens::full_tt_fp8e8m0<16 * NS>>(256);
        auto b_sc_tm = tmem_allocator.template allocate<kittens::full_tt_fp8e8m0<32 * NS>>(384);

        for (int ti = 0; ti < num_my; ti++) {
            wait_phase(mbar_tmem_buffer_free, tmem_buffer_free_phase);
            tmem_buffer_free_phase ^= 1;
            tcgen05_fence_after_thread_sync();

            for (int k = 0; k < num_k; k++) {
                int slot = gk % NS;
                wait_phase(mbar_compute_data_ready[slot], compute_data_ready_phase[slot]);

                auto a_sc_stage = a_sc_tm.template subtile<kittens::full_tt_fp8e8m0<16>>(slot * 16);
                auto b_sc_stage_0 = b_sc_tm.template subtile<kittens::full_tt_fp8e8m0<16>>(slot * 32);
                auto b_sc_stage_1 = b_sc_tm.template subtile<kittens::full_tt_fp8e8m0<16>>(slot * 32 + 16);
                cuda_tcgen05::copy_scale_atom_from_smem_to_tmem(
                    a_sc_stage.addr, a_sc_smem[slot]);
                cuda_tcgen05::copy_scale_atom_from_smem_to_tmem(
                    b_sc_stage_0.addr, b_sc_smem[slot][0]);
                cuda_tcgen05::copy_scale_atom_from_smem_to_tmem(
                    b_sc_stage_1.addr, b_sc_smem[slot][1]);

                auto b_sc_stage = b_sc_tm.template subtile<kittens::full_tt_fp8e8m0<32>>(slot * 32);
                if (k == 0) {
                    cuda_tcgen05::issue_mxfp8_mma2_abt_128x256x128<0>(
                        out_tm, a_smem[slot], b_smem[slot],
                        a_sc_stage, b_sc_stage, mbar_compute_buffer_free[slot]);
                } else {
                    cuda_tcgen05::issue_mxfp8_mma2_abt_128x256x128<1>(
                        out_tm, a_smem[slot], b_smem[slot],
                        a_sc_stage, b_sc_stage, mbar_compute_buffer_free[slot]);
                }

                compute_data_ready_phase[slot] ^= 1;
                gk++;
            }
            signal_on_mma_completion(mbar_tmem_data_ready, cta_mask);
        }
    } else if (warp_id >= 4 && warp_id < NUM_WARPS + 4) {
        constexpr int ROW_STRIPS    = BM / 32;
        constexpr int COL_GROUPS    = NUM_WARPS / ROW_STRIPS;
        constexpr int COLS_PER_WARP = BN / COL_GROUPS;
        constexpr int EPI_THREADS   = NUM_WARPS * 32;
        const int ew = warp_id - 4;
        const int row_warp = ew % ROW_STRIPS;
        const int col_warp = ew / ROW_STRIPS;
        const int my_row = row_warp * 32 + lane;
        uint32_t full = 0;

        for (int ti = 0; ti < num_my; ti++) {
            int base_m, base_n, local_m, local_n;
            map_off(ti, base_m, base_n, local_m, local_n);
            wait_phase(mbar_tmem_data_ready, full);
            full ^= 1;
            tcgen05_fence_after_thread_sync();

            const uint32_t trow = taddr + ((uint32_t)(cta_rank * BM + row_warp * 32) << 16);
            constexpr int LOADS_PER_CHUNK = STORE_N / 8;
            constexpr int LOADS_PER_WARP = LOADS_PER_CHUNK / COL_GROUPS;
            constexpr int NUM_CHUNKS = BN / STORE_N;
            static_assert(STORE_N == 64);
            static_assert(LOADS_PER_WARP * COL_GROUPS == LOADS_PER_CHUNK);
            int store_stage = 0;

            #pragma unroll
            for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                float t[LOADS_PER_WARP][8];
                #pragma unroll
                for (int n = 0; n < LOADS_PER_WARP; n++) {
                    const int local_n = col_warp * LOADS_PER_WARP + n;
                    tcgen05_ld_32x32b_x8(
                        trow + (uint32_t)(chunk * STORE_N + local_n * 8), t[n]);
                }
                tcgen05_wait_ld();

                if (chunk == NUM_CHUNKS - 1) {
                    tcgen05_fence_before_thread_sync();
                    if (ew == 0 && elect_sync()) {
                        mbarrier_arrive_no_tx_cluster(mbar_tmem_buffer_free);
                    }
                }

                if (ew == 0) {
                    tma_wait_group<TMA_STORE_STAGES - 1>();
                }
                asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));

                #pragma unroll
                for (int n = 0; n < LOADS_PER_WARP; n++) {
                    __nv_bfloat162 pk[4];
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        pk[i] = __floats2bfloat162_rn(
                            mmc_epi(t[n][2 * i]), mmc_epi(t[n][2 * i + 1]));
                    }
                    const int local_n = col_warp * LOADS_PER_WARP + n;
                    const int swizzled_n = local_n ^ (my_row & 7);
                    __nv_bfloat16* write_ptr =
                        &d_smem[store_stage].data[0] + my_row * STORE_N + swizzled_n * 8;
                    *reinterpret_cast<int4*>(write_ptr) = *reinterpret_cast<int4*>(pk);
                }

                __syncwarp();
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));

                if (ew == 0 && elect_sync()) {
                    kittens::tma::store_async(
                        g.D, d_smem[store_stage],
                        {local_m / BM, (base_n + chunk * STORE_N) / STORE_N});
                }
                store_stage ^= 1;
            }
        }
        if (ew == 0) {
            tma_wait_group<0>();
        }
        asm volatile("bar.sync 1, %0;" :: "n"(EPI_THREADS));
    }

    __syncthreads();
    if (warp_id == 0) {
        tmem_allocator.set_addr(taddr);
        tmem_allocator.deprovision();
    }
}

extern "C" __global__ __cluster_dims__(CTA_GROUP, 1, 1) __launch_bounds__(LAUNCH_THREADS, 1)
void matmul_cluster(
    const __grid_constant__ TkMxfp8Globals g,
    __nv_bfloat16* C_ptr, int M, int N, int K
) {
    matmul_cluster_impl(g, C_ptr, M, N, K);
}

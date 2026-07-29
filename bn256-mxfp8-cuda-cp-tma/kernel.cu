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
    return kittens::warp::elect_leader();
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
    kittens::tensor_after_thread_sync();
}

__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
    kittens::tensor_before_thread_sync();
}

__device__ __forceinline__ void tcgen05_wait_ld() {
    kittens::tensor_load_wait();
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

__device__ __forceinline__ void cp_scale_32x16b_cta_group_2(
    uint32_t tmem_addr,
    const sf_tile& src
) {
    static_assert(sf_tile::rows == 32 && sf_tile::cols == 16);
    static_assert(!sf_tile::swizzle);
    uint64_t st_desc = make_scale_smem_desc(&src.data[0]);
    asm volatile(
        "{tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;}"
        :: "r"(tmem_addr), "l"(st_desc));
}

} // namespace cuda_tcgen05

__device__ __forceinline__ void mbarrier_init(kittens::semaphore& sem, int count) {
    kittens::init_semaphore(sem, count);
}

__device__ __forceinline__ void mbarrier_arrive_no_tx(kittens::semaphore& sem) {
    kittens::arrive(sem);
}

__device__ __forceinline__ void mbarrier_arrive_no_tx_cluster(kittens::semaphore& sem) {
    kittens::tma::cluster::arrive(sem, 0);
}

__device__ __forceinline__ void signal_on_bytes_loaded(kittens::semaphore& sem, int bytes) {
    kittens::tma::cluster::expect_bytes(sem, bytes, 0);
}

__device__ __forceinline__ void signal_on_mma_completion(
    kittens::semaphore& sem, uint16_t cta_mask
) {
    kittens::detail::tcgen05::commit<CTA_GROUP>(sem, cta_mask);
}

__device__ __forceinline__ void wait_phase(kittens::semaphore& sem, uint32_t phase) {
    kittens::wait(sem, phase);
}

template <int N>
__device__ __forceinline__ void tma_wait_group() {
    kittens::tma::store_async_wait<N>();
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
    sf_tile (&a_sc_smem)[NS] = al.allocate<sf_tile, NS>();
    sf_tile (&b_sc_smem)[NS][CTA_GROUP] = al.allocate<sf_tile, NS, CTA_GROUP>();
    d_tile (&d_smem)[TMA_STORE_STAGES] = al.allocate<d_tile, TMA_STORE_STAGES>();

    __shared__ kittens::semaphore mbar_compute_data_ready[NS];
    __shared__ kittens::semaphore mbar_compute_buffer_free[NS];
    __shared__ uint32_t tmem_addr_holder[1];

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane    = tid % WARP_SIZE;
    kittens::tensor_allocator<1, CTA_GROUP, false> tmem_allocator{};

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

                kittens::tma::cluster::load_async(
                    a_smem[slot], g.A, {local_m / BM, k},
                    mbar_compute_data_ready[slot],
                    static_cast<uint16_t>(1 << cta_rank), 0);
                kittens::tma::cluster::load_async(
                    b_smem[slot], g.B, {local_n / BN_LOCAL, k},
                    mbar_compute_data_ready[slot],
                    static_cast<uint16_t>(1 << cta_rank), 0);
                kittens::tma::cluster::load_async(
                    a_sc_smem[slot], g.A_sc, {local_m / BM, k, 0, 0},
                    mbar_compute_data_ready[slot],
                    static_cast<uint16_t>(1 << cta_rank), 0);
                kittens::tma::cluster::load_async(
                    b_sc_smem[slot][cta_rank], g.B_sc, {local_n / BN_LOCAL, k, 0, 0},
                    mbar_compute_data_ready[slot],
                    static_cast<uint16_t>(cta_mask), 0);

                constexpr int slot_bytes =
                    sizeof(a_tile) + sizeof(b_tile) + sizeof(sf_tile) * (1 + CTA_GROUP);
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
                cuda_tcgen05::cp_scale_32x16b_cta_group_2(
                    a_sc_stage.addr, a_sc_smem[slot]);
                cuda_tcgen05::cp_scale_32x16b_cta_group_2(
                    b_sc_stage_0.addr, b_sc_smem[slot][0]);
                cuda_tcgen05::cp_scale_32x16b_cta_group_2(
                    b_sc_stage_1.addr, b_sc_smem[slot][1]);

                auto b_sc_stage = b_sc_tm.template subtile<kittens::full_tt_fp8e8m0<32>>(slot * 32);
                if (k == 0) {
                    kittens::mm2_ABt(out_tm, a_smem[slot], b_smem[slot],
                                     a_sc_stage, b_sc_stage, mbar_compute_buffer_free[slot]);
                } else {
                    kittens::mma2_ABt(out_tm, a_smem[slot], b_smem[slot],
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

#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/scheduler/gemm.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// SM100 (Blackwell) FUSED BF16 GEMM + PUSH-RING ReduceScatter (single-node NVLink).
//
// Unlike the push+combine kernel (sm100_bf16_gemm_reduce_scatter.cuh, which stores every
// tile to a per-rank slot then does a serial owner-side combine), this kernel folds the
// reduction INTO a bidirectional ring transfer, INSIDE the GEMM epilogue:
//
//   Ring direction i -> (i-1). For tile owner c on rank i:  d=(i-c+R)%R.
//     START  (d==R-1): running_sum = my own tile           -> TMA-store to down(i) ring slot[c]
//     MIDDLE (0<d<R-1): wait upstream flag -> load upstream partial + my own tile -> add
//                        -> TMA-store to down(i) ring slot[c]
//     END    (d==0,i==c): wait upstream flag -> load partial + own -> add -> write OUTPUT
//   Writer: TMA-store -> tma_store_wait -> threadfence_system -> red_add_rel_sys(down flag).
//   Reader (w3): ld_acq_sys spin on my flag. Deadlock-free: per-tile dep chain START->..->END
//   is a linear DAG; combined with per-SM in-order execution it stays acyclic (verified by
//   tests/sim_ring_deadlock.py for R=2/4/8, all M, both schedules).
//
// 3 warpgroups (384 threads = 12 warps):
//   WG0 producer (w0-3, 48 regs): w0 TMA-load A/B; w1 MMA; w2 TMEM alloc; w3 partial TMA-load
//       (pulls the upstream-written partial from THIS rank's ring recv slot into partial_buf).
//   WG1 epilogue (w4-7, 200 regs): TMEM -> smem (STSM_T transpose) ONLY. Does NOT write HBM;
//       writes into epi_smem (double-buffered), then hands off to WG2.
//   WG2 ring     (w8-11, 200 regs): read epi_smem(own) + partial_buf(upstream) -> add in-place
//       into epi_smem -> TMA-store to down peer ring slot (or own output) -> set down flag.
//
// Symmetric buffer layout (identical on every rank):
//   [ barrier region (32 B) ]
//   [ output   : m_per_rank * N * sizeof(bf16) ]            (END writes here, owner==rank)
//   [ ring recv: kNumRanks * m_per_rank * N * sizeof(bf16) ](upstream writes running-sum here)
//   [ flags    : num_m_blocks * num_n_blocks * int ]        (per-tile ready flag)
//
// swap-AB: token(M) is UMMA N-dim (BLOCK_M, 16..256); hidden(N) is UMMA M-dim (BLOCK_N=128).
// Owner segments padded to a multiple of BLOCK_M so each tile belongs to exactly one owner.
template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K_,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages_,
          uint32_t kNumSMs,
          uint32_t kNumRanks,
          uint64_t kTensorCoreUtilControl>
CUTLASS_GLOBAL void __launch_bounds__(384, 1)
sm100_bf16_gemm_reduce_scatter_ring_impl(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                                         const uint32_t rank,
                                         const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                                         const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                                         const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                                         const __grid_constant__ cute::TmaDescriptor tensor_map_partial_load,
                                         const __grid_constant__ cute::TmaDescriptor tensor_map_ring_store_down,
                                         const __grid_constant__ cute::TmaDescriptor tensor_map_out) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using cd_dtype_t = cutlass::bfloat16_t;   // epilogue staging + comm are BF16
    constexpr uint32_t kNumGroups = 1;
    constexpr uint32_t kNumMulticast = 1;
    constexpr bool kIsMulticastOnA = false;
    constexpr GemmType kGemmType = GemmType::Normal;

    // No BLOCK_K merge for the ring kernel (kept simple; stages are decided host-side).
    constexpr uint32_t BLOCK_K = BLOCK_K_;
    constexpr uint32_t kNumStages = kNumStages_;

    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator1Sm;

    // MMA configs (swap-AB: UMMA_N = BLOCK_M = token tile; UMMA_M = 128 = hidden rows)
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = BLOCK_M;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_K_ == 64, "Invalid block K");
    DG_STATIC_ASSERT(BLOCK_N == LAYOUT_AD_M, "swap-AB requires BLOCK_N == 128");

    // Epilogue configs: 2-stage TMEM accumulator (comm is the slow side; >2 gives no gain and
    // frees TMEM cols so UMMA_N=BLOCK_M can reach 256: 2*256=512 = TMEM col cap).
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;   // epi_smem double-buffer
    constexpr uint32_t kNumPartialStages  = 2;   // partial_buf double-buffer
    DG_STATIC_ASSERT(kNumEpilogueStages * UMMA_N <= 512, "TMEM accumulator columns exceed 512");

    // swap-AB store granularity: 16-row token blocks.
    constexpr uint32_t STORE_BLOCK_M = 16;
    constexpr uint32_t STORE_BLOCK_N = BLOCK_N;
    // Epilogue (WG1) + ring (WG2) each have 4 warps = 128 threads.
    constexpr uint32_t kNumEpilogueThreads = 128;
    constexpr uint32_t kNumRingThreads     = 128;
    constexpr uint32_t kNumProducerThreads = 128;
    DG_STATIC_ASSERT(kNumProducerThreads + kNumEpilogueThreads + kNumRingThreads == 384, "Must be 384 threads");

    // Register allocation (verified: (48+200+200)*128 = 57344 <= 64512).
    constexpr uint32_t kNumProducerRegisters = 48;
    constexpr uint32_t kNumEpilogueRegisters = 200;
    constexpr uint32_t kNumRingRegisters     = 200;
    DG_STATIC_ASSERT(kNumProducerRegisters * kNumProducerThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads +
                     kNumRingRegisters     * kNumRingThreads <= 64512, "Too many registers");

    // Shared memory sizes.
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);        // epi_smem
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_PARTIAL_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);   // partial_buf
    constexpr uint32_t SMEM_PARTIAL_SIZE = SMEM_PARTIAL_SIZE_PER_STAGE * kNumPartialStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(cutlass::bfloat16_t);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_PARTIAL_SIZE % 1024 == 0 and
                     SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0,
                     "Shared memory must be aligned to 1024 bytes");

    // Real tensor memory size and offsets.
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * UMMA_N;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols>();
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Utils
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors.
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_partial_load);
        cute::prefetch_tma_descriptor(&tensor_map_ring_store_down);
        cute::prefetch_tma_descriptor(&tensor_map_out);
    }

    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Symmetric buffer regions.
    const uint32_t m_per_rank = shape_m / kNumRanks;
    const uint64_t shard_elems = static_cast<uint64_t>(m_per_rank) * shape_n;
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, /*num_experts=*/kNumRanks,
        /*num_max_tokens_per_rank=*/0u, /*num_topk=*/1u);
    auto* out_base = reinterpret_cast<nv_bfloat16*>(
        static_cast<uint8_t*>(sym_buffer.get_base_ptr()) + layout::Workspace::kNumBarrierSignalBytes);
    auto* ring_base = reinterpret_cast<nv_bfloat16*>(
        reinterpret_cast<uint8_t*>(out_base) + shard_elems * sizeof(nv_bfloat16));
    auto* flag_base = reinterpret_cast<int*>(
        reinterpret_cast<uint8_t*>(ring_base) + kNumRanks * shard_elems * sizeof(nv_bfloat16));

    // Ring neighbour: where I STORE my running-sum.
    const uint32_t down = (rank + kNumRanks - 1) % kNumRanks;

    // ================= shared memory carve-up =================
    // [ epi_smem (CD) ][ partial_buf ][ A stages ][ B stages ][ barriers ][ tmem ptr ]
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_partial = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_PARTIAL_SIZE_PER_STAGE);
    });
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(
            smem_buffer + SMEM_CD_SIZE + SMEM_PARTIAL_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(
            smem_buffer + SMEM_CD_SIZE + SMEM_PARTIAL_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // Barriers region.
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        smem_buffer + SMEM_CD_SIZE + SMEM_PARTIAL_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    // A/B pipeline barriers.
    auto full_barriers       = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers      = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    // TMEM accumulator barriers.
    auto tmem_full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + i; });
    auto tmem_empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + kNumEpilogueStages + i; });
    // epi_smem hand-off barriers: WG1 signals "epi ready" (full), WG2 signals "epi consumed" (empty).
    auto epi_full_barriers   = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + kNumEpilogueStages * 2 + i; });
    auto epi_empty_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + kNumEpilogueStages * 2 + kNumTMAStoreStages + i; });
    // partial_buf hand-off barriers: w3 signals "partial ready" (full), WG2 signals "partial consumed" (empty).
    auto part_full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + kNumEpilogueStages * 2 + kNumTMAStoreStages * 2 + i; });
    auto part_empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + kNumEpilogueStages * 2 + kNumTMAStoreStages * 2 + kNumPartialStages + i; });

    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(
        barrier_start_ptr + kNumStages * 2 + kNumEpilogueStages * 2 + kNumTMAStoreStages * 2 + kNumPartialStages * 2);

    // Initialize barriers.
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(kNumMulticast);
            empty_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumEpilogueThreads);   // WG1 releases the accumulator
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumTMAStoreStages; ++ i) {
            epi_full_barriers[i]->init(kNumEpilogueThreads);     // all WG1 threads arrive after STSM
            epi_empty_barriers[i]->init(kNumRingThreads);        // all WG2 threads arrive after consume
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumPartialStages; ++ i) {
            part_full_barriers[i]->init(1);                      // w3 (one thread) signals partial loaded
            part_empty_barriers[i]->init(kNumRingThreads);       // all WG2 threads release partial
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    cudaGridDependencySynchronize();

    // ================= reset per-tile flags + cross-rank barrier =================
    // Each MIDDLE/END consumer waits flag[tile]==0 -> 1, so flags must start at 0 each launch.
    const uint32_t num_m_blocks = shape_m / BLOCK_M;                    // owner-aligned padded M
    const uint32_t num_n_blocks = shape_n / BLOCK_N;
    const uint32_t num_tile_flags = num_m_blocks * num_n_blocks;
    for (uint32_t f = thread_idx + sm_idx * blockDim.x; f < num_tile_flags; f += kNumSMs * blockDim.x)
        flag_base[f] = 0;
    comm::grid_sync<kNumSMs, /*kGridSyncIndex=*/1>(workspace, sm_idx, thread_idx, [&]() { __syncthreads(); });
    comm::nvlink_barrier<kNumRanks, kNumSMs, 384, /*kGridSyncIndex=*/2, /*kTag=*/0>(
        workspace, sym_buffer, sm_idx, thread_idx, [&]() { __syncthreads(); });

    // Block scheduler (shared shape across roles).
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs>(
        shape_m, shape_n, shape_k, nullptr);

    // Pipeline and TMA phases (A/B).
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = (stage_idx + 1) % kNumStages;
        phase ^= stage_idx == 0;
    };

    // Stagger: rank i starts with owner (rank+1)%R (its own START owner), so upstream stays
    // exactly one step ahead. Bijection over owners -> every tile still computed once.
    auto stagger_owner = [&](const uint32_t& raw_m_block) -> uint32_t {
        const uint32_t bpo = m_per_rank / BLOCK_M;              // blocks per owner (exact)
        const uint32_t owner_raw = raw_m_block / bpo;
        const uint32_t off = raw_m_block % bpo;
        const uint32_t c = (owner_raw + rank + 1) % kNumRanks;
        return c * bpo + off;
    };

    // ============================================================================
    // WG0 producer (warps 0-3): TMA-load A/B (w0), MMA (w1), TMEM alloc (w2), partial load (w3)
    // ============================================================================
    if (warp_idx < 4) {
        cutlass::arch::warpgroup_reg_dealloc<kNumProducerRegisters>();

        if (warp_idx == 0 and cute::elect_one_sync()) {
            // ---- w0: TMA-load A/B (GEMM inputs) ----
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                m_block_idx = stagger_owner(m_block_idx);
                const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                    uint32_t m_idx = scheduler.template get_global_idx<true, sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
                    uint32_t n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), sched::IndexType::MN>(shape_n, BLOCK_N, n_block_idx, m_block_idx);
                    DG_STATIC_ASSERT(kMajorA == cute::UMMA::Major::K, "Invalid major");
                    uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), sched::IndexType::K>(shape_k, BLOCK_K, k_block_idx, m_block_idx);
                    uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), sched::IndexType::K>(shape_k, BLOCK_K, k_block_idx, m_block_idx);
                    if constexpr (kMajorA == cute::UMMA::Major::K)
                        tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t, false>(
                            &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_a_idx, m_idx, kNumMulticast, 0);
                    if constexpr (kMajorB == cute::UMMA::Major::K)
                        tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t, false>(
                            &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_b_idx, n_idx, kNumMulticast, 0);
                    if constexpr (kMajorB == cute::UMMA::Major::MN)
                        tma::copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, cutlass::bfloat16_t, false>(
                            &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], n_idx, k_b_idx, kNumMulticast, 0);
                    constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
                    full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast);
                }
            }
        } else if (warp_idx == 1 and is_leader_cta) {
            // ---- w1: MMA issue ----
            auto instr_desc = cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                          UMMA_M, UMMA_N, kMajorB, kMajorA>();
            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            auto a_desc = mma::sm100::make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
            uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
            uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;
            DG_STATIC_ASSERT((UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                             "Invalid MMA instruction shape");
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                m_block_idx = stagger_owner(m_block_idx);
                auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
                auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
                ptx::tcgen05_after_thread_sync();
                auto umma_arrive = [](const uint64_t* barrier) { cutlass::arch::umma_arrive(barrier); };
                auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                    umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                    __syncwarp();
                };
                const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    ptx::tcgen05_after_thread_sync();
                    using mma_t = ptx::SM100_MMA_F16BF16_SS;
                    const auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
                    const auto a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
                    const auto b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
                    if (cute::elect_one_sync()) {
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                            a_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t>(a_desc_base_lo, 0, k * UMMA_K);
                            b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t>(b_desc_base_lo, 0, k * UMMA_K);
                            mma_t::fma(b_desc, a_desc, accum_stage_idx * UMMA_N, k_block_idx > 0 or k > 0, runtime_instr_desc);
                        }
                    }
                    __syncwarp();
                    empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
                }
            }
        } else if (warp_idx == 3) {
            // ---- w3: partial TMA-load. For each MIDDLE/END tile, wait my flag, then TMA-load
            //          the upstream-written partial from MY ring recv slot[c] into partial_buf. ----
            uint32_t part_stage_idx = 0, part_phase = 0;
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                m_block_idx = stagger_owner(m_block_idx);
                const uint32_t base_m_idx = m_block_idx * BLOCK_M;
                const uint32_t base_n_idx = n_block_idx * BLOCK_N;
                const uint32_t owner = base_m_idx / m_per_rank;
                const uint32_t d = (rank - owner + kNumRanks) % kNumRanks;
                const bool is_start = (d == kNumRanks - 1);
                const uint32_t tile_flag_idx = m_block_idx * num_n_blocks + n_block_idx;   // padded-M block index
                // START tiles have no upstream partial; skip (WG2 copies own tile directly).
                if (not is_start) {
                    const uint32_t m_local = base_m_idx - owner * m_per_rank;
                    // Wait the whole tile's flag (upstream set it once, after storing all 16-row blocks).
                    if (lane_idx == 0) {
                        while (ptx::ld_acq_sys(flag_base + tile_flag_idx) == 0) {}
                    }
                    __syncwarp();
                    // TMA-load each 16-row block of the tile into partial_buf (double-buffered).
                    #pragma unroll 1
                    for (uint32_t s = 0; s < BLOCK_M / STORE_BLOCK_M; ++ s) {
                        part_empty_barriers[part_stage_idx]->wait(part_phase ^ 1);
                        if (cute::elect_one_sync()) {
                            // ring recv slot[owner] row = owner*m_per_rank + m_local + s*16.
                            const uint32_t g_m = owner * m_per_rank + m_local + s * STORE_BLOCK_M;
                            tma::copy<STORE_BLOCK_N, STORE_BLOCK_M, kSwizzleCDMode, cd_dtype_t, false>(
                                &tensor_map_partial_load, part_full_barriers[part_stage_idx], smem_partial[part_stage_idx],
                                base_n_idx, g_m, 1, 0);
                            part_full_barriers[part_stage_idx]->arrive_and_expect_tx(SMEM_PARTIAL_SIZE_PER_STAGE);
                        }
                        __syncwarp();
                        part_stage_idx = (part_stage_idx + 1) % kNumPartialStages;
                        part_phase ^= part_stage_idx == 0;
                    }
                }
            }
        }
    // ============================================================================
    // WG1 epilogue (warps 4-7): TMEM -> epi_smem (STSM_T transpose) ONLY.
    // ============================================================================
    } else if (warp_idx < 8) {
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();
        const uint32_t epilogue_warp_idx = warp_idx - 4;
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        uint32_t epi_stage_idx = 0, epi_phase = 0;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            m_block_idx = stagger_owner(m_block_idx);
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();
            const auto tmem_base_addr = accum_stage_idx * UMMA_N;

            // For each 16-row block: TMEM_LOAD -> STSM_T transpose -> epi_smem (double-buffered).
            #pragma unroll 1
            for (uint32_t s = 0; s < BLOCK_M / STORE_BLOCK_M; ++ s) {
                // Wait WG2 to have consumed the epi_smem stage we're about to fill.
                epi_empty_barriers[epi_stage_idx]->wait(epi_phase ^ 1);

                constexpr uint32_t kNumBankGroupBytes = 16;
                constexpr uint32_t kNumSwizzleAtomRows = 8;
                constexpr uint32_t STORE_BLOCK_N_ATOM = kSwizzleCDMode / sizeof(cd_dtype_t);
                #pragma unroll
                for (uint32_t i = 0; i < STORE_BLOCK_M / kNumSwizzleAtomRows; ++ i) {
                    uint32_t tmem_addr = tmem_base_addr + s * STORE_BLOCK_M + i * kNumSwizzleAtomRows;
                    uint32_t values[kNumSwizzleAtomRows];
                    DG_STATIC_ASSERT(STORE_BLOCK_N_ATOM % 32 == 0, "Invalid block sizes");
                    constexpr uint32_t kNumWarpsPerAtom = STORE_BLOCK_N_ATOM / 32;
                    uint32_t outer_atom_offset = (epilogue_warp_idx / kNumWarpsPerAtom) * STORE_BLOCK_M * kSwizzleCDMode;
                    uint32_t inner_atom_offset = i * kNumSwizzleAtomRows * kSwizzleCDMode;
                    auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[epi_stage_idx]) + outer_atom_offset + inner_atom_offset;
                    cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr, values[0], values[1], values[2], values[3]);
                    cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000, values[4], values[5], values[6], values[7]);
                    cutlass::arch::fence_view_async_tmem_load();
                    uint32_t row = lane_idx % 8;
                    uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                    auto smem_ptr = smem_base_ptr + row * (kNumBankGroupBytes * 8) + (col ^ row) * kNumBankGroupBytes;
                    ptx::SM90_U32x4_STSM_T<int>::copy(math::cast_into_bf16_and_pack(values[0], values[1]),
                                                      math::cast_into_bf16_and_pack(values[2], values[3]),
                                                      math::cast_into_bf16_and_pack(values[4], values[5]),
                                                      math::cast_into_bf16_and_pack(values[6], values[7]),
                                                      smem_ptr);
                }
                // Release the accumulator on the last 16-row block (mirrors base kernel).
                if (s == BLOCK_M / STORE_BLOCK_M - 1) {
                    ptx::tcgen05_before_thread_sync();
                    tmem_empty_barriers[accum_stage_idx]->arrive();
                }
                cute::tma_store_fence();   // make STSM writes visible before WG2 reads epi_smem
                // Signal WG2: epi_smem[epi_stage_idx] is ready.
                epi_full_barriers[epi_stage_idx]->arrive();
                epi_stage_idx = (epi_stage_idx + 1) % kNumTMAStoreStages;
                epi_phase ^= epi_stage_idx == 0;
            }
        }
    // ============================================================================
    // WG2 ring (warps 8-11): read epi_smem(own) + partial_buf(upstream) -> add in-place ->
    // TMA-store to down peer ring slot (or own output) -> set down flag.
    // ============================================================================
    } else {
        cutlass::arch::warpgroup_reg_alloc<kNumRingRegisters>();
        const uint32_t ring_warp_idx = warp_idx - 8;

        constexpr uint32_t STORE_BLOCK_N_ATOM = kSwizzleCDMode / sizeof(cd_dtype_t);
        uint32_t epi_stage_idx = 0, epi_phase = 0;
        uint32_t part_stage_idx = 0, part_phase = 0;

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            m_block_idx = stagger_owner(m_block_idx);
            const uint32_t base_m_idx = m_block_idx * BLOCK_M;
            const uint32_t base_n_idx = n_block_idx * BLOCK_N;
            const uint32_t owner = base_m_idx / m_per_rank;
            const uint32_t m_local = base_m_idx - owner * m_per_rank;
            const uint32_t d = (rank - owner + kNumRanks) % kNumRanks;
            const bool is_start = (d == kNumRanks - 1);
            const bool is_end   = (d == 0);
            const uint32_t tile_flag_idx = m_block_idx * num_n_blocks + n_block_idx;

            // Delayed-release store pipeline. Issue block s's store from epi_smem[cur], then
            // tma_store_wait<kNumTMAStoreStages-1> so at most (stages-1) stores stay in flight
            // (adjacent stores overlap on NVLink instead of draining one-by-one). Release the
            // PREVIOUS block's stage (whose store has now completed) back to WG1. Drain fully
            // ONCE at tile end before flagging downstream, so the whole tile is visible.
            bool pending_valid = false;
            uint32_t pending_stage = 0;
            const uint32_t num_stores = BLOCK_M / STORE_BLOCK_M;
            #pragma unroll 1
            for (uint32_t s = 0; s < num_stores; ++ s) {
                const uint32_t cur_stage = epi_stage_idx;
                // Wait own tile (epi_smem) from WG1.
                epi_full_barriers[cur_stage]->wait(epi_phase);
                // Accumulate upstream partial into epi_smem in place (MIDDLE/END). START: epi_smem
                // already holds the own tile (running_sum = own), no add.
                if (not is_start) {
                    part_full_barriers[part_stage_idx]->wait(part_phase);
                    // Both buffers share the identical 128B-swizzle [16,128] layout -> raw uint4 add.
                    constexpr uint32_t kNumVecs = (STORE_BLOCK_M * STORE_BLOCK_N) / 8;   // 16*128/8 = 256
                    auto* dst = reinterpret_cast<uint4*>(smem_cd[cur_stage]);
                    auto* src = reinterpret_cast<const uint4*>(smem_partial[part_stage_idx]);
                    #pragma unroll 4
                    for (uint32_t v = ring_warp_idx * 32 + lane_idx; v < kNumVecs; v += kNumRingThreads) {
                        uint4 a = ptx::ld_shared(dst + v);
                        uint4 b = ptx::ld_shared(src + v);
                        const nv_bfloat16* pa = reinterpret_cast<const nv_bfloat16*>(&a);
                        const nv_bfloat16* pb = reinterpret_cast<const nv_bfloat16*>(&b);
                        nv_bfloat16 r[8];
                        #pragma unroll
                        for (uint32_t e = 0; e < 8; ++ e)
                            r[e] = __hadd(pa[e], pb[e]);
                        uint4 rr = *reinterpret_cast<const uint4*>(r);
                        ptx::st_shared(dst + v, rr.x, rr.y, rr.z, rr.w);
                    }
                }
                // Fence epi_smem writes, then rendezvous: all threads done writing + partial fully read.
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(kNumRingThreads, 1);
                if (not is_start) {
                    part_empty_barriers[part_stage_idx]->arrive();     // release partial_buf to w3
                    part_stage_idx = (part_stage_idx + 1) % kNumPartialStages;
                    part_phase ^= part_stage_idx == 0;
                }
                // Issue this block's store (pipelined).
                if (ring_warp_idx == 0 and cute::elect_one_sync()) {
                    const uint32_t g_m_local = m_local + s * STORE_BLOCK_M;
                    const auto& tmap = is_end ? tensor_map_out : tensor_map_ring_store_down;
                    const uint32_t row = is_end ? g_m_local : (owner * m_per_rank + g_m_local);
                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_N / STORE_BLOCK_N_ATOM; ++ i) {
                        auto smem_ptr = smem_cd[cur_stage] + i * STORE_BLOCK_M * STORE_BLOCK_N_ATOM;
                        cute::SM90_TMA_STORE_2D::copy(&tmap, smem_ptr, base_n_idx + i * STORE_BLOCK_N_ATOM, row);
                    }
                    cute::tma_store_arrive();
                    cute::tma_store_wait<kNumTMAStoreStages - 1>();    // <= stages-1 in flight
                }
                // After wait<stages-1>, the PREVIOUS block's store has completed -> release its stage.
                cutlass::arch::NamedBarrier::sync(kNumRingThreads, 2);
                if (pending_valid)
                    epi_empty_barriers[pending_stage]->arrive();
                pending_stage = cur_stage;
                pending_valid = true;
                epi_stage_idx = (epi_stage_idx + 1) % kNumTMAStoreStages;
                epi_phase ^= epi_stage_idx == 0;
            }

            // Drain the last in-flight store, release the last stage, THEN flag downstream so the
            // full tile is visible. (END writes own output -> no flag.)
            if (ring_warp_idx == 0 and cute::elect_one_sync())
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumRingThreads, 2);
            if (pending_valid)
                epi_empty_barriers[pending_stage]->arrive();
            if (not is_end and ring_warp_idx == 0 and cute::elect_one_sync()) {
                __threadfence_system();
                ptx::red_add_rel_sys(sym_buffer.map(flag_base + tile_flag_idx, down), 1);
            }
        }
    }

    // ================= epilogue: free TMEM, final cross-rank barrier =================
    __syncthreads();
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);
    __threadfence_system();
    // Final barrier so all peers' ring stores / output writes are globally visible.
    comm::nvlink_barrier<kNumRanks, kNumSMs, 384, /*kGridSyncIndex=*/0, /*kTag=*/1>(
        workspace, sym_buffer, sm_idx, thread_idx, [&]() { __syncthreads(); });
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop

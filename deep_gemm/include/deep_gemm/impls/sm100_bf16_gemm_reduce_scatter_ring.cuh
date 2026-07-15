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

    // MMA configs (NON-swap / standard: token(M) = UMMA_M = 128 rows; hidden(N) = UMMA_N = BLOCK_N).
    // TMEM accumulator is [token, hidden] = same order as the HBM output -> the epilogue TMEM_LOAD
    // yields token-major registers and writes smem with a plain st_shared (NO STSM_T transpose).
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;   // token rows
    constexpr uint32_t UMMA_N = BLOCK_N;                       // hidden cols (free dim)
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_K_ == 64, "Invalid block K");
    // BLOCK_M is LOCKED to 128 (== LAYOUT_AD_M): token tile is the whole owner-aligned 128-row block
    // (m_per_rank padded to 128), so each tile is single-owner and the segment/flag logic is unchanged.
    DG_STATIC_ASSERT(BLOCK_M == 128, "fused ring locks BLOCK_M == 128 (token tile = owner-aligned)");
    DG_STATIC_ASSERT(BLOCK_N == 128, "fused ring uses BLOCK_N == 128 (hidden tile)");

    // Epilogue configs. NO separate epi_smem: WG1 adds the upstream partial IN REGISTERS (non-swap
    // is token-major, so TMEM_LOAD values and the partial share layout) and writes the running-sum
    // straight back into partial_buf; WG2 then just TMA-stores it (pure send). partial_buf is the
    // ONE staging buffer (2 stages) shared by w3(fill upstream) -> WG1(add own, write sum) ->
    // WG2(send). Removing epi_smem frees smem for a deeper A/B load pipeline. TMEM kept at 4.
    constexpr uint32_t kNumEpilogueStages = 4;   // TMEM accumulator depth (MMA look-ahead)
    constexpr uint32_t kNumPartialStages  = 2;   // partial_buf depth (w3/WG1/WG2 pipeline)
    DG_STATIC_ASSERT(kNumEpilogueStages * UMMA_N <= 512, "TMEM accumulator columns exceed 512");

    // Store granularity = WHOLE tile: 128 token rows (== BLOCK_M) x BLOCK_N hidden.
    constexpr uint32_t STORE_BLOCK_M = BLOCK_M;     // 128 token rows
    constexpr uint32_t STORE_BLOCK_N = BLOCK_N;     // 128 hidden cols
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

    // Shared memory sizes. Only partial_buf now (no epi_smem).
    constexpr uint32_t SMEM_PARTIAL_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);   // partial_buf
    constexpr uint32_t SMEM_PARTIAL_SIZE = SMEM_PARTIAL_SIZE_PER_STAGE * kNumPartialStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(cutlass::bfloat16_t);
    DG_STATIC_ASSERT(SMEM_PARTIAL_SIZE % 1024 == 0 and
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
    // [ partial_buf ][ A stages ][ B stages ][ barriers ][ tmem ptr ]   (NO separate epi_smem)
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_partial = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_PARTIAL_SIZE_PER_STAGE);
    });
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(
            smem_buffer + SMEM_PARTIAL_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(
            smem_buffer + SMEM_PARTIAL_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // Barriers region. partial_buf is a 3-stage-of-life pipeline on ONE buffer:
    //   part_full : w3 filled upstream  -> WG1        (init 1  : one TMA producer)
    //   epi_full  : WG1 wrote running-sum -> WG2      (init 128: all WG1 threads)
    //   part_empty: WG2 sent it -> w3 (refill)        (init 128: all WG2 threads)
    // Each group has kNumPartialStages entries.
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        smem_buffer + SMEM_PARTIAL_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers       = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers      = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    auto tmem_full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + i; });
    auto tmem_empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + kNumEpilogueStages + i; });
    const uint32_t kPartBase = kNumStages * 2 + kNumEpilogueStages * 2;
    auto part_full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kPartBase + i; });
    auto epi_full_barriers   = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kPartBase + kNumPartialStages + i; });
    auto part_empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kPartBase + kNumPartialStages * 2 + i; });

    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(
        barrier_start_ptr + kPartBase + kNumPartialStages * 3);

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
        for (uint32_t i = 0; i < kNumPartialStages; ++ i) {
            part_full_barriers[i]->init(1);                      // w3 (one thread) signals partial loaded
            epi_full_barriers[i]->init(kNumEpilogueThreads);     // all WG1 threads signal running-sum ready
            part_empty_barriers[i]->init(kNumRingThreads);       // all WG2 threads release partial
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    cudaGridDependencySynchronize();

    // ================= reset PER-OWNER-SEGMENT flags + counters + cross-rank barrier =================
    // Coarse per-segment flag protocol (flux-style): instead of a flag per tile (hundreds of
    // system fences), there is ONE flag per owner c (R total). Upstream (i+1) sets rank i's
    // flag[c] exactly ONCE, after storing ALL of segment c's tiles; rank i (w3) waits it ONCE.
    // A device-scope counter[c] (rank-local) counts how many of segment c's tiles THIS rank has
    // stored; the SM that bumps it to seg_tiles does the single threadfence_system + release flag.
    //   flag region layout: [ flags: R int ][ counters: R int ]
    const uint32_t num_m_blocks = shape_m / BLOCK_M;                    // owner-aligned padded M
    const uint32_t num_n_blocks = shape_n / BLOCK_N;
    auto* counter_base = flag_base + kNumRanks;                        // R flags, then R counters
    const uint32_t bpo = m_per_rank / BLOCK_M;                         // blocks per owner (exact)
    const uint32_t seg_tiles = bpo * num_n_blocks;                     // tiles per owner segment
    const uint32_t total_tiles = num_m_blocks * num_n_blocks;          // = R * seg_tiles
    if (sm_idx == 0 and thread_idx < 2 * kNumRanks)
        flag_base[thread_idx] = 0;                                     // zero R flags + R counters
    comm::grid_sync<kNumSMs, /*kGridSyncIndex=*/1>(workspace, sm_idx, thread_idx, [&]() { __syncthreads(); });
    comm::nvlink_barrier<kNumRanks, kNumSMs, 384, /*kGridSyncIndex=*/2, /*kTag=*/0>(
        workspace, sym_buffer, sm_idx, thread_idx, [&]() { __syncthreads(); });

    // Keep a scheduler object only for get_global_idx (Normal GEMM: block_idx*block_size, stateless).
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

    // SEGMENT-ORDERED tile enumeration (replaces the L2-swizzle scheduler order). All 5 roles
    // iterate the SAME sequence: for staggered owner segment t=0..R-1 (c=(rank+1+t)%R), the
    // segment's (off,nb) tiles are handed grid-strided to SMs; a whole segment is enumerated
    // before the next. This makes the per-segment flag deadlock-free (verified by
    // tests/sim_ring_deadlock.py --seg-flag --seg-order for R=2/4/8, all M). Whole segment c
    // lives at m_block [c*bpo, c*bpo+bpo); BLOCK_M=128 padding keeps every tile single-owner.
    auto seg_tile = [&](const uint32_t& it, uint32_t& mb, uint32_t& nb_out, uint32_t& owner) -> bool {
        const uint64_t gbi = static_cast<uint64_t>(it) * kNumSMs + sm_idx;
        if (gbi >= total_tiles) return false;
        const uint32_t t = static_cast<uint32_t>(gbi / seg_tiles);
        const uint32_t in_seg = static_cast<uint32_t>(gbi % seg_tiles);
        const uint32_t c = (rank + 1 + t) % kNumRanks;
        owner = c;
        mb = c * bpo + in_seg / num_n_blocks;      // owner segment base + m-offset
        nb_out = in_seg % num_n_blocks;
        return true;
    };

    // ============================================================================
    // WG0 producer (warps 0-3): TMA-load A/B (w0), MMA (w1), TMEM alloc (w2), partial load (w3)
    // ============================================================================
    if (warp_idx < 4) {
        cutlass::arch::warpgroup_reg_dealloc<kNumProducerRegisters>();

        if (warp_idx == 0 and cute::elect_one_sync()) {
            // ---- w0: TMA-load A/B (GEMM inputs) ----
            uint32_t seg_owner;
            for (uint32_t it = 0; seg_tile(it, m_block_idx, n_block_idx, seg_owner); ++ it) {
                const auto num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    empty_barriers[stage_idx]->wait(phase ^ 1);
#ifdef DG_RS_SKIP_AB_LOAD
                    // ABLATION: skip the A/B TMA-load; MMA runs on stale smem (garbage math, but
                    // isolates whether the GEMM-input load transport paces the kernel).
                    full_barriers[stage_idx]->arrive_and_expect_tx(0);
#else
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
#endif
                }
            }
        } else if (warp_idx == 1 and is_leader_cta) {
            // ---- w1: MMA issue ----
            // NON-swap operand order: A then B (swap-AB would be kMajorB, kMajorA).
            auto instr_desc = cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                          UMMA_M, UMMA_N, kMajorA, kMajorB>();
            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            auto a_desc = mma::sm100::make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
            uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
            uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;
            DG_STATIC_ASSERT((UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                             "Invalid MMA instruction shape");
            uint32_t seg_owner;
            for (uint32_t it = 0; seg_tile(it, m_block_idx, n_block_idx, seg_owner); ++ it) {
                auto accum_stage_idx = it % kNumEpilogueStages;
                auto accum_phase_idx = (it / kNumEpilogueStages) & 1;
                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
                ptx::tcgen05_after_thread_sync();
                auto umma_arrive = [](const uint64_t* barrier) { cutlass::arch::umma_arrive(barrier); };
                auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                    umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                    __syncwarp();
                };
                const auto num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    ptx::tcgen05_after_thread_sync();
#ifndef DG_RS_SKIP_MMA
                    using mma_t = ptx::SM100_MMA_F16BF16_SS;
                    const auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
                    const auto a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
                    const auto b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
                    if (cute::elect_one_sync()) {
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                            a_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t>(a_desc_base_lo, 0, k * UMMA_K);
                            b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t>(b_desc_base_lo, 0, k * UMMA_K);
                            mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N, k_block_idx > 0 or k > 0, runtime_instr_desc);
                        }
                    }
                    __syncwarp();
#else
                    // ABLATION: skip UMMA (TMEM holds garbage). Keep barrier handshakes so the
                    // WG1/WG2 pipeline still flows -> isolates pure comm cost (no MMA compute).
                    __syncwarp();
#endif
                    empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
                }
            }
        } else if (warp_idx == 3) {
            // ---- w3: partial TMA-load. EVERY tile cycles the partial_buf stage (so w3/WG1/WG2
            //          barrier counts stay consistent). Per SEGMENT wait the ONE upstream flag once
            //          (MIDDLE/END). For a non-START tile TMA-load the upstream partial into
            //          partial_buf; for a START tile just arrive part_full with 0 bytes (WG1 will
            //          write the own tile with no add). ----
            uint32_t part_stage_idx = 0, part_phase = 0;
            int cur_owner = -1;
            bool cur_is_start = false;
            uint32_t seg_owner;
            for (uint32_t it = 0; seg_tile(it, m_block_idx, n_block_idx, seg_owner); ++ it) {
                const uint32_t base_m_idx = m_block_idx * BLOCK_M;
                const uint32_t base_n_idx = n_block_idx * BLOCK_N;
                const uint32_t owner = seg_owner;
                const uint32_t d = (rank - owner + kNumRanks) % kNumRanks;
                if (static_cast<int>(owner) != cur_owner) {
                    cur_owner = static_cast<int>(owner);
                    cur_is_start = (d == kNumRanks - 1);
#ifndef DG_RS_W3_NOLOAD
                    if (not cur_is_start) {
                        if (lane_idx == 0)
                            while (ptx::ld_acq_sys(flag_base + owner) == 0) {}   // per-segment flag
                        __syncwarp();
                    }
#endif
                }
                part_empty_barriers[part_stage_idx]->wait(part_phase ^ 1);
                if (cute::elect_one_sync()) {
#ifdef DG_RS_W3_NOLOAD
                    // ABLATION: skip cross-rank flag-wait + TMA-load; keep the barrier handshake so
                    // the w3/WG1/WG2 partial pipeline still flows (partial holds stale data -> wrong
                    // output, speed only). Isolates whether w3's load/flag-wait paces the kernel.
                    part_full_barriers[part_stage_idx]->arrive_and_expect_tx(0);
#else
                    if (not cur_is_start) {
                        const uint32_t m_local = base_m_idx - owner * m_per_rank;
                        const uint32_t g_m = owner * m_per_rank + m_local;
                        tma::copy<STORE_BLOCK_N, STORE_BLOCK_M, kSwizzleCDMode, cd_dtype_t, false>(
                            &tensor_map_partial_load, part_full_barriers[part_stage_idx], smem_partial[part_stage_idx],
                            base_n_idx, g_m, 1, 0);
                        part_full_barriers[part_stage_idx]->arrive_and_expect_tx(SMEM_PARTIAL_SIZE_PER_STAGE);
                    } else {
                        // START: no upstream to load; just signal the buffer is available to WG1.
                        part_full_barriers[part_stage_idx]->arrive_and_expect_tx(0);
                    }
#endif
                }
                __syncwarp();
                part_stage_idx = (part_stage_idx + 1) % kNumPartialStages;
                part_phase ^= part_stage_idx == 0;
            }
        }
    // ============================================================================
    // WG1 epilogue (warps 4-7): TMEM -> epi_smem, NON-swap (token-major) via plain st_shared,
    // NO transpose. TMEM accum is [token,hidden] = HBM order, so TMEM_LOAD gives token-major regs.
    // ============================================================================
    } else if (warp_idx < 8) {
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();
        const uint32_t epilogue_warp_idx = warp_idx - 4;
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);   // 8 bf16
        constexpr uint32_t STORE_BLOCK_N_ATOM = kSwizzleCDMode / sizeof(cd_dtype_t);          // 64 (swizzle atom)
        DG_STATIC_ASSERT(STORE_BLOCK_N % STORE_BLOCK_N_ATOM == 0, "Invalid swizzle atom");
        DG_STATIC_ASSERT(STORE_BLOCK_N_ATOM % kNumElemsPerBankGroup == 0, "Invalid swizzle");

        uint32_t part_stage_idx = 0, part_phase = 0;
        int cur_owner = -1;
        bool cur_is_start = false;
        uint32_t seg_owner;
        for (uint32_t it = 0; seg_tile(it, m_block_idx, n_block_idx, seg_owner); ++ it) {
            const uint32_t owner = seg_owner;
            const uint32_t d = (rank - owner + kNumRanks) % kNumRanks;
            if (static_cast<int>(owner) != cur_owner) {
                cur_owner = static_cast<int>(owner);
                cur_is_start = (d == kNumRanks - 1);
            }
            const bool is_start = cur_is_start;
            auto accum_stage_idx = it % kNumEpilogueStages;
            auto accum_phase_idx = (it / kNumEpilogueStages) & 1;
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);   // wait MMA
            ptx::tcgen05_after_thread_sync();
            const auto tmem_base_addr = accum_stage_idx * UMMA_N;
            // Wait w3 to have filled partial_buf (upstream loaded, or START -> empty buffer ready).
            part_full_barriers[part_stage_idx]->wait(part_phase);

            // Whole [128,128] tile: TMEM_LOAD own (token-major, NO transpose), add the upstream
            // partial read from the SAME swizzle address in partial_buf (MIDDLE/END; the upstream
            // was TMA-loaded there in the identical CD swizzle layout, so ld_shared at smem_ptr gives
            // the 8 bf16 matching the 8 token-major 'values'), then st_shared the running-sum back
            // into partial_buf. START: no add. Mirrors sm100_store_cd.cuh addressing (s x i atoms).
            auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_partial[part_stage_idx]);
            #pragma unroll
            for (uint32_t s = 0; s < STORE_BLOCK_N / STORE_BLOCK_N_ATOM; ++ s) {
                auto atom_base = smem_base_ptr + s * STORE_BLOCK_M * STORE_BLOCK_N_ATOM * sizeof(cd_dtype_t);
                #pragma unroll
                for (uint32_t i = 0; i < STORE_BLOCK_N_ATOM / kNumElemsPerBankGroup; ++ i) {
                    auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                    constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                    auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                    auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                    col ^= row % (kSwizzleCDMode / 16);
                    uint32_t tmem_addr = tmem_base_addr + s * STORE_BLOCK_N_ATOM + i * kNumElemsPerBankGroup;
                    auto smem_ptr = atom_base + epilogue_warp_idx * 32 * kSwizzleCDMode
                                              + row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
                    uint32_t values[8];
                    cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                        values[0], values[1], values[2], values[3],
                        values[4], values[5], values[6], values[7]);
                    cutlass::arch::fence_view_async_tmem_load();
                    // cast own -> 4 packed u32 (8 bf16)
                    uint32_t own[4] = {
                        static_cast<uint32_t>(math::cast_into_bf16_and_pack(values[0], values[1])),
                        static_cast<uint32_t>(math::cast_into_bf16_and_pack(values[2], values[3])),
                        static_cast<uint32_t>(math::cast_into_bf16_and_pack(values[4], values[5])),
                        static_cast<uint32_t>(math::cast_into_bf16_and_pack(values[6], values[7])) };
                    if (not is_start) {
                        // Add upstream partial (already in partial_buf at this swizzle addr).
                        uint4 up = ptx::ld_shared(reinterpret_cast<const uint4*>(smem_ptr));
                        const uint32_t upw[4] = { up.x, up.y, up.z, up.w };
                        #pragma unroll
                        for (uint32_t p = 0; p < 4; ++ p) {
                            __nv_bfloat162 a2 = *reinterpret_cast<const __nv_bfloat162*>(&own[p]);
                            __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(&upw[p]);
                            __nv_bfloat162 r2 = __hadd2(a2, b2);
                            own[p] = *reinterpret_cast<const uint32_t*>(&r2);
                        }
                    }
                    ptx::st_shared(smem_ptr, own[0], own[1], own[2], own[3]);
                }
            }
            // Release the accumulator; signal WG2 the running-sum is ready in partial_buf.
            ptx::tcgen05_before_thread_sync();
            tmem_empty_barriers[accum_stage_idx]->arrive();
            cute::tma_store_fence();   // make st_shared writes visible before WG2 TMA-stores partial_buf
            epi_full_barriers[part_stage_idx]->arrive();
            part_stage_idx = (part_stage_idx + 1) % kNumPartialStages;
            part_phase ^= part_stage_idx == 0;
        }
    // ============================================================================
    // WG2 ring (warps 8-11): PURE SEND. WG1 already wrote the running-sum (own+upstream) into
    // partial_buf, so WG2 just TMA-stores it to the down peer ring slot (or own output for END),
    // drains, releases partial_buf back to w3, and counts toward the per-segment flag.
    // ============================================================================
    } else {
        cutlass::arch::warpgroup_reg_alloc<kNumRingRegisters>();
        const uint32_t ring_warp_idx = warp_idx - 8;

        constexpr uint32_t STORE_BLOCK_N_ATOM = kSwizzleCDMode / sizeof(cd_dtype_t);
        uint32_t part_stage_idx = 0, part_phase = 0;

        // Per-SEGMENT flag: each stored tile bumps a device-scope counter[c]; the SM that makes
        // counter[c] reach seg_tiles does ONE __threadfence_system + release the downstream flag[c].
        // (deadlock-free under segment-ordered enumeration; sim_ring_deadlock --seg-flag --seg-order.)
        int cur_owner = -1;
        bool cur_is_end = false;
        uint32_t seg_local_cnt = 0;
        auto flush_segment = [&](uint32_t c, bool is_end) {
            cutlass::arch::NamedBarrier::sync(kNumRingThreads, 3);
            if (not is_end and ring_warp_idx == 0 and cute::elect_one_sync() and seg_local_cnt > 0) {
                __threadfence_system();
                int prev = atomicAdd(counter_base + c, static_cast<int>(seg_local_cnt));
                if (prev + static_cast<int>(seg_local_cnt) == static_cast<int>(seg_tiles))
                    ptx::red_add_rel_sys(sym_buffer.map(flag_base + c, down), 1);
            }
            seg_local_cnt = 0;
        };

        uint32_t seg_owner;
        for (uint32_t it = 0; seg_tile(it, m_block_idx, n_block_idx, seg_owner); ++ it) {
            const uint32_t base_n_idx = n_block_idx * BLOCK_N;
            const uint32_t owner = seg_owner;
            const uint32_t m_local = m_block_idx * BLOCK_M - owner * m_per_rank;
            const uint32_t d = (rank - owner + kNumRanks) % kNumRanks;
            if (static_cast<int>(owner) != cur_owner) {
                if (cur_owner >= 0)
                    flush_segment(static_cast<uint32_t>(cur_owner), cur_is_end);
                cur_owner = static_cast<int>(owner);
                cur_is_end = (d == 0);
            }
            const bool is_end = cur_is_end;

            // Wait WG1 to have written the running-sum into partial_buf[part_stage_idx].
            epi_full_barriers[part_stage_idx]->wait(part_phase);
            // Issue + drain the whole-[128,128] store (down peer ring slot, or own output for END).
            if (ring_warp_idx == 0 and cute::elect_one_sync()) {
                const auto& tmap = is_end ? tensor_map_out : tensor_map_ring_store_down;
                const uint32_t row = is_end ? m_local : (owner * m_per_rank + m_local);
                #pragma unroll
                for (uint32_t i = 0; i < STORE_BLOCK_N / STORE_BLOCK_N_ATOM; ++ i) {
                    auto smem_ptr = smem_partial[part_stage_idx] + i * STORE_BLOCK_M * STORE_BLOCK_N_ATOM;
                    cute::SM90_TMA_STORE_2D::copy(&tmap, smem_ptr, base_n_idx + i * STORE_BLOCK_N_ATOM, row);
                }
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();     // store done before releasing partial_buf
            }
            // Release partial_buf back to w3; count this stored tile toward the segment.
            cutlass::arch::NamedBarrier::sync(kNumRingThreads, 2);
            part_empty_barriers[part_stage_idx]->arrive();
            ++ seg_local_cnt;
            part_stage_idx = (part_stage_idx + 1) % kNumPartialStages;
            part_phase ^= part_stage_idx == 0;
        }
        // Flush the last segment (flag).
        if (cur_owner >= 0)
            flush_segment(static_cast<uint32_t>(cur_owner), cur_is_end);
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

#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/scheduler/gemm.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// SM100 (Blackwell) fused BF16 GEMM + ReduceScatter (single-node NVLink).
//
// Single fused kernel, three phases:
//   * Phase-1 GEMM: the epilogue stores the partial as **BF16** into a local scratch.
//   * Phase-2 PUSH: every thread pushes each BF16 tile into the owner rank's per-rank
//     scratch SLOT with a *plain vectorized store* (NO remote atomic). Owner rank `o`
//     receives `kNumRanks` slots, slot `j` holding rank `j`'s contribution to `o`'s rows.
//   * Phase-3 COMBINE: owner reads its `kNumRanks` local slots and sums them in FP32,
//     writing the final `[m_per_rank, N]` shard into a dedicated output region.
//
// Two design choices keep it fast: (a) BF16 over NVLink (half the bytes), and
// (b) plain stores instead of per-element `red.add.f32` remote atomics (the reduction
// is a cheap *local* pass). Owner-side FP32 accumulation gives better accuracy than a
// BF16 NCCL reduce-scatter. Matches the mega-MoE dispatch(plain remote write) +
// combine(local reduce) idiom.
//
// Symmetric buffer layout (identical on every rank):
//   [ barrier region (32 B) ]
//   [ output   : m_per_rank * N * sizeof(out_dtype_t) ]
//   [ scratch  : kNumRanks * m_per_rank * N * sizeof(bf16) ]   (kNumRanks slots)
//
// Restrictions: FP32 or BF16 output, no swap-AB, no multicast, Normal GEMM only.
template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K_,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages_,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumSMs,
          uint32_t kNumRanks,
          uint64_t kTensorCoreUtilControl>
CUTLASS_GLOBAL void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_bf16_gemm_reduce_scatter_impl(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                                       const uint32_t rank,
                                       nv_bfloat16* local_scratch,
                                       const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                                       const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                                       const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                                       const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    // Fixed configuration for the fused reduce-scatter path
    using cd_dtype_t = cutlass::bfloat16_t; // local scratch + epilogue output are BF16
                                            // (epilogue requires cutlass::bfloat16_t, bit-compatible with nv_bfloat16)
    using out_dtype_t = float;              // final combined output (FP32 for crisp verification)
    constexpr uint32_t kNumGroups = 1;
    constexpr uint32_t kNumMulticast = 1;
    constexpr bool kIsMulticastOnA = false;
    constexpr bool kSwapAB = false;
    constexpr bool kWithAccumulation = false;
    constexpr GemmType kGemmType = GemmType::Normal;

    // Enlarge `BLOCK_K` for some cases (same merge trick as the dense kernel)
    constexpr bool kDoMergeStages =
        kNumStages_ >= 8 and
        kMajorA == cute::UMMA::Major::K and kMajorB == cute::UMMA::Major::K;
    constexpr uint32_t kNumMinStages = 8;
    constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages_ / kNumMinStages : 1;
    constexpr uint32_t BLOCK_K = BLOCK_K_ * kNumStagesPerMerge;
    constexpr uint32_t kNumStages = kNumStages_ / kNumStagesPerMerge;

    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator1Sm;

    // MMA Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = BLOCK_N;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_K_ == 64, "Invalid block K");
    DG_STATIC_ASSERT(BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M, "Invalid block size");

    // Epilogue configs
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t STORE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // Shared memory sizes (CD is now BF16)
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(cutlass::bfloat16_t);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0,
                     "Shared memory of A/B must be aligned to 1024 bytes");

    static constexpr uint32_t UMMA_A_SIZE_PER_STAGE = math::constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(nv_bfloat16);
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory out of bound for UMMA");

    // Real tensor memory size and offsets
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * UMMA_N;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols>();
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Utils
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    // Overwrite shape constants if the compiler gives
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Symmetric buffer layout: [barrier][output fp32][scratch: kNumRanks slots bf16]
    const uint32_t m_per_rank = shape_m / kNumRanks;
    const uint64_t shard_elems = static_cast<uint64_t>(m_per_rank) * shape_n;
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, /*num_experts=*/kNumRanks,
        /*num_max_tokens_per_rank=*/0u, /*num_topk=*/1u);
    auto* out_base = reinterpret_cast<out_dtype_t*>(
        static_cast<uint8_t*>(sym_buffer.get_base_ptr()) + layout::Workspace::kNumBarrierSignalBytes);
    // Scratch slots begin right after the FP32 output region
    auto* scratch_base = reinterpret_cast<nv_bfloat16*>(
        reinterpret_cast<uint8_t*>(out_base) + shard_elems * sizeof(out_dtype_t));

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // D/A/B shared memory
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers              = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers             = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto tmem_full_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_empty_barriers        = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages + i); });
    auto tensor_core_full_barrier   = barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2;

    // Fill the tensor memory pointer
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2 + 1);

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(kNumMulticast);
            empty_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }
        if constexpr (kTensorCoreUtilControl < 100)
            tensor_core_full_barrier->init(1);

        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs>(
        shape_m, shape_n, shape_k, nullptr);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0, tensor_core_phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = (stage_idx + 1) % kNumStages;
        phase ^= stage_idx == 0;
    };

    // ================= Phase 1: GEMM into local BF16 scratch =================
    if (warp_idx == 0 and cute::elect_one_sync()) {
        // TMA load warp
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                uint32_t m_idx = scheduler.template get_global_idx<true, sched::IndexType::MN>(
                    shape_m, BLOCK_M, m_block_idx);
                uint32_t n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), sched::IndexType::MN>(
                    shape_n, BLOCK_N, n_block_idx, m_block_idx);

                DG_STATIC_ASSERT(kMajorA == cute::UMMA::Major::K, "Invalid major");
                uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), sched::IndexType::K>(
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);
                uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), sched::IndexType::K>(
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);

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
        // MMA issue warp
        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                      UMMA_M, UMMA_N, kMajorA, kMajorB>();

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;
        auto a_desc = mma::sm100::make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            auto umma_arrive = [](const uint64_t* barrier) {
                cutlass::arch::umma_arrive(barrier);
            };
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
                        uint32_t atom_k_idx = k * UMMA_K / BLOCK_ATOM_K;
                        a_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t>(
                                        a_desc_base_lo, atom_k_idx * LOAD_BLOCK_M * BLOCK_ATOM_K, k * UMMA_K % BLOCK_ATOM_K);
                        b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t>(
                                        b_desc_base_lo, atom_k_idx * LOAD_BLOCK_N * BLOCK_ATOM_K, k * UMMA_K % BLOCK_ATOM_K);
                        mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N,
                                   k_block_idx > 0 or k > 0, runtime_instr_desc);
                    }
                }
                __syncwarp();

                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);

                DG_STATIC_ASSERT(kTensorCoreUtilControl > 0, "Invalid tensor utilization control");
                if constexpr (kTensorCoreUtilControl < 100) {
                    umma_arrive(reinterpret_cast<uint64_t*>(tensor_core_full_barrier));
                    __syncwarp();
                    tensor_core_full_barrier->wait(tensor_core_phase);
                    tensor_core_phase ^= 1;
                    constexpr static uint64_t kNumUMMACycles = (2ull * UMMA_M * UMMA_N * BLOCK_K) / 8192ull;
                    constexpr static uint64_t kNumDummyCycles = (100ull - kTensorCoreUtilControl) * kNumUMMACycles / kTensorCoreUtilControl;
                    const auto start_clock = clock64();
                    if (cute::elect_one_sync())
                        while (clock64() - start_clock < kNumDummyCycles) {}
                    __syncwarp();
                }
            }
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        // Epilogue warp group: TMA-store BF16 partial into local scratch (fast local store).
        // The cross-rank push happens in a separate bulk phase with high memory-level
        // parallelism (better than blocking the epilogue on high-latency remote stores).
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        uint32_t tma_stage_idx = 0;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            const auto tmem_base_addr = accum_stage_idx * UMMA_N;
            const auto base_m_idx = scheduler.template get_global_idx<true, sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
            const auto base_n_idx = n_block_idx * BLOCK_N;

            epilogue::sm100_store_cd<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                kGemmType, kWithAccumulation,
                cd_dtype_t, epilogue::transform::EpilogueIdentity>
            (smem_cd, tma_stage_idx, tmem_base_addr,
             base_m_idx, base_n_idx, scheduler.current_group_idx,
             epilogue_warp_idx, lane_idx,
             tmem_empty_barriers[accum_stage_idx],
             tensor_map_cd);
        }
    }

    // Ensure this rank's TMA stores into local scratch are all issued and completed
    if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        if (cute::elect_one_sync())
            cute::tma_store_wait<0>();
    }
    __syncthreads();

    // Deallocate tensor memory
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

    // ================= grid sync: all SMs finished local scratch =================
    comm::grid_sync<kNumSMs, /*kGridSyncIndex=*/0>(
        workspace, sm_idx, thread_idx, [&]() { __syncthreads(); });

    // ================= Phase 2: PUSH local BF16 scratch into owner slots =================
    // Plain vectorized store (NO atomic), grid-strided over the whole [shape_m, shape_n]
    // scratch. High memory-level parallelism: each thread keeps kNumUnroll independent
    // uint4 loads in flight before storing, to hide latency at low occupancy.
    constexpr uint32_t kNumThreads = kNumNonEpilogueThreads + kNumEpilogueThreads;
    constexpr uint32_t kNumElemsPerVec = 8;                        // 8 bf16 == 16 B == uint4
    constexpr uint32_t kNumUnroll = 8;
    const uint64_t total_vecs = (static_cast<uint64_t>(shape_m) * shape_n) / kNumElemsPerVec;
    const uint32_t vecs_per_row = shape_n / kNumElemsPerVec;
    const uint64_t stride_vecs = static_cast<uint64_t>(kNumSMs) * kNumThreads;
    const uint64_t slot_stride_vecs = shard_elems / kNumElemsPerVec;
    const uint64_t unroll_stride = stride_vecs * kNumUnroll;
    const uint64_t base_v = static_cast<uint64_t>(sm_idx) * kNumThreads + thread_idx;
    for (uint64_t vb = base_v; vb < total_vecs; vb += unroll_stride) {
        uint4 reg[kNumUnroll];
        // Issue all loads first (independent -> maximizes in-flight memory ops)
        #pragma unroll
        for (uint32_t u = 0; u < kNumUnroll; ++ u) {
            const uint64_t v = vb + static_cast<uint64_t>(u) * stride_vecs;
            if (v < total_vecs)
                reg[u] = *(reinterpret_cast<const uint4*>(local_scratch) + v);
        }
        // Then scatter to remote owners
        #pragma unroll
        for (uint32_t u = 0; u < kNumUnroll; ++ u) {
            const uint64_t v = vb + static_cast<uint64_t>(u) * stride_vecs;
            if (v < total_vecs) {
                const uint32_t row = static_cast<uint32_t>(v / vecs_per_row);
                const uint32_t vcol = static_cast<uint32_t>(v % vecs_per_row);
                const uint32_t owner_rank = row / m_per_rank;
                const uint32_t m_local = row - owner_rank * m_per_rank;
                uint4* dst = reinterpret_cast<uint4*>(scratch_base) +
                             static_cast<uint64_t>(rank) * slot_stride_vecs +
                             static_cast<uint64_t>(m_local) * vecs_per_row + vcol;
                *sym_buffer.map(dst, owner_rank) = reg[u];
            }
        }
    }

    // ================= cross-rank NVLink barrier =================
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads,
                         /*kGridSyncIndex=*/0, /*kTag=*/0>(
        workspace, sym_buffer, sm_idx, thread_idx,
        [&]() { __syncthreads(); });

    // ================= Phase 3: COMBINE local slots -> FP32 output =================
    // This rank owns rows [rank*m_per_rank, (rank+1)*m_per_rank). Its `kNumRanks` scratch
    // slots hold every rank's contribution. Sum in FP32. Unrolled for memory-level
    // parallelism: load all `kNumRanks` slots for kNumCombineUnroll vecs, then reduce.
    const uint64_t out_total_vecs = shard_elems / kNumElemsPerVec;
    constexpr uint32_t kNumCombineUnroll = 4;
    const uint64_t combine_unroll_stride = stride_vecs * kNumCombineUnroll;
    for (uint64_t vb = base_v; vb < out_total_vecs; vb += combine_unroll_stride) {
        float acc[kNumCombineUnroll][kNumElemsPerVec];
        #pragma unroll
        for (uint32_t u = 0; u < kNumCombineUnroll; ++ u)
            #pragma unroll
            for (uint32_t e = 0; e < kNumElemsPerVec; ++ e)
                acc[u][e] = 0.0f;

        #pragma unroll 1
        for (uint32_t j = 0; j < kNumRanks; ++ j) {
            uint4 reg[kNumCombineUnroll];
            #pragma unroll
            for (uint32_t u = 0; u < kNumCombineUnroll; ++ u) {
                const uint64_t v = vb + static_cast<uint64_t>(u) * stride_vecs;
                if (v < out_total_vecs)
                    reg[u] = *(reinterpret_cast<const uint4*>(scratch_base) +
                               static_cast<uint64_t>(j) * slot_stride_vecs + v);
            }
            #pragma unroll
            for (uint32_t u = 0; u < kNumCombineUnroll; ++ u) {
                const nv_bfloat16* bf = reinterpret_cast<const nv_bfloat16*>(&reg[u]);
                #pragma unroll
                for (uint32_t e = 0; e < kNumElemsPerVec; ++ e)
                    acc[u][e] += __bfloat162float(bf[e]);
            }
        }

        #pragma unroll
        for (uint32_t u = 0; u < kNumCombineUnroll; ++ u) {
            const uint64_t v = vb + static_cast<uint64_t>(u) * stride_vecs;
            if (v < out_total_vecs) {
                float4* out_ptr = reinterpret_cast<float4*>(out_base) + v * 2;
                out_ptr[0] = make_float4(acc[u][0], acc[u][1], acc[u][2], acc[u][3]);
                out_ptr[1] = make_float4(acc[u][4], acc[u][5], acc[u][6], acc[u][7]);
            }
        }
    }

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop

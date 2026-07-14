#pragma once

#include <cute/atom/copy_traits_sm100.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

namespace deep_gemm::epilogue {

// swap-AB epilogue for the fused BF16 GEMM + ReduceScatter kernel.
//
// Differs from `sm100_store_cd_swap_ab` in exactly one place: the destination of each
// 16-row (`STORE_BLOCK_M`) TMA store is chosen PER STORE BLOCK by the token->owner map,
// instead of a single tile-wide tensor map. Because owner segments are padded to a
// multiple of `STORE_BLOCK_M` (16) rows, each 16-row block lands wholly inside one
// owner's peer slot (no straddling). Phantom tail blocks (`owner >= kNumRanks`, produced
// when the M_pad tiling overruns the real owner region) are skipped.
//
// `base_m_idx` is the GLOBAL padded token index of the tile start (owner segments laid
// out contiguously: owner `o` occupies rows [o*m_per_rank_pad, o*m_per_rank_pad+m_per_rank)).
template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t kSwizzleCDMode,
          uint32_t kNumTMAStoreStages,
          uint32_t kNumUMMAStoreThreads,
          uint32_t kNumRanks,
          typename cd_dtype_t,
          typename epilogue_type_t,
          typename pattern_cd_t>
CUTLASS_DEVICE void
sm100_store_cd_swap_ab_rs(const utils::PatternVisitor<pattern_cd_t>& smem_cd, uint32_t& tma_stage_idx,
                          const uint32_t& tmem_base_addr,
                          const uint32_t& base_m_idx, const uint32_t& base_n_idx,
                          const uint32_t& effective_m, const uint32_t& m_per_rank_pad,
                          const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
                          const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier,
                          const layout::CdTmaMaps<kNumRanks>& tensor_map_cd_owners) {
    // NOTES: The epilogue requires a full warpgroup to read all 128 TMEM rows,
    //          implying STORE_BLOCK_N must be 128.
    DG_STATIC_ASSERT(STORE_BLOCK_N == 128, "STORE_BLOCK_N must be 128 to match TMEM rows");
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "RS store path is BF16 only");

    // TMA checks
    constexpr uint32_t STORE_BLOCK_N_ATOM = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumSwizzleAtomRows = 8;
    DG_STATIC_ASSERT(kSwizzleCDMode == 128, "TMA D must be 128B swizzled");
    DG_STATIC_ASSERT(BLOCK_M % STORE_BLOCK_M == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(STORE_BLOCK_M % kNumSwizzleAtomRows == 0, "Invalid swizzling");
    DG_STATIC_ASSERT(STORE_BLOCK_N % STORE_BLOCK_N_ATOM == 0, "Invalid swizzling");

    // Share store pipeline between blocks
    auto advance_store_pipeline = [&]() {
        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    };

    // Iterate over M (token) blocks; each 16-row block is routed to its own owner
    const auto num_stores = effective_m / STORE_BLOCK_M;
    for (uint32_t s = 0; s < num_stores; ++ s, advance_store_pipeline()) {
        // Token->owner routing for this 16-row block (owner boundaries are 16-aligned,
        // so the whole block belongs to exactly one owner). Phantom tail => skip store.
        const uint32_t global_m = base_m_idx + s * STORE_BLOCK_M;
        const uint32_t owner     = global_m / m_per_rank_pad;
        const uint32_t m_local   = global_m - owner * m_per_rank_pad;
        const bool     do_store  = owner < kNumRanks;


        // Wait shared memory to be released
        if (epilogue_warp_idx == 0)
            cute::tma_store_wait<kNumTMAStoreStages - 1>();
        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

        // Store into shared memory
        #pragma unroll
        for (uint32_t i = 0; i < STORE_BLOCK_M / kNumSwizzleAtomRows; ++ i) {
            uint32_t tmem_addr = tmem_base_addr +
                                 s * STORE_BLOCK_M +            // Store stage offset
                                 i * kNumSwizzleAtomRows;       // In-block offset
            uint32_t values[kNumSwizzleAtomRows];

            // Warps cooperatively write an atomic block to shared memory
            DG_STATIC_ASSERT(STORE_BLOCK_N_ATOM % 32 == 0, "Invalid block sizes");
            constexpr uint32_t kNumWarpsPerAtom = STORE_BLOCK_N_ATOM / 32;
            uint32_t outer_atom_offset = (epilogue_warp_idx / kNumWarpsPerAtom) * STORE_BLOCK_M * kSwizzleCDMode;
            uint32_t inner_atom_offset = i * kNumSwizzleAtomRows * kSwizzleCDMode;
            auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) + outer_atom_offset + inner_atom_offset;

            // Load from TMEM using `.16x256b` shape to satisfy STSM layout requirements
            // Start from lane index 0
            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                   values[0], values[1], values[2], values[3]);
            // Start from lane index 16
            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                   values[4], values[5], values[6], values[7]);
            cutlass::arch::fence_view_async_tmem_load();

            // Destination shared memory address
            uint32_t row = lane_idx % 8;
            uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
            auto smem_ptr = smem_base_ptr + row * (kNumBankGroupBytes * 8)
                                          + (col ^ row) * kNumBankGroupBytes;

            // Store matrix with transposition
            ptx::SM90_U32x4_STSM_T<int>::copy(math::cast_into_bf16_and_pack(values[0], values[1]),
                                              math::cast_into_bf16_and_pack(values[2], values[3]),
                                              math::cast_into_bf16_and_pack(values[4], values[5]),
                                              math::cast_into_bf16_and_pack(values[6], values[7]),
                                              smem_ptr);
        }

        // Notify tensor memory empty (only at the leader CTA) arrival ASAP
        // NOTES: only the last stage needs to do this
        if (s == num_stores - 1) {
            ptx::tcgen05_before_thread_sync();
            tmem_empty_barrier->arrive(0u);
        }

        // Synchronize all threads and issue TMA
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
        if (do_store and epilogue_warp_idx == 0 and cute::elect_one_sync()) {
            const auto& tensor_map_cd = tensor_map_cd_owners.maps[owner];
            #pragma unroll
            for (uint32_t i = 0; i < STORE_BLOCK_N / STORE_BLOCK_N_ATOM; ++ i) {
                auto smem_ptr = smem_cd[tma_stage_idx] + i * STORE_BLOCK_M * STORE_BLOCK_N_ATOM;
                uint32_t m_idx = m_local;                                                     // token row inside owner slot
                uint32_t n_idx = epilogue_type_t::template apply_index_n<STORE_BLOCK_N_ATOM>(base_n_idx + i * STORE_BLOCK_N_ATOM);

                // Fused reduce-scatter is always Normal GEMM (2D), plain store (no accumulation)
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_cd, smem_ptr, n_idx, m_idx);
            }
            cute::tma_store_arrive();
        }
        __syncwarp();
    }
}

} // namespace deep_gemm::epilogue

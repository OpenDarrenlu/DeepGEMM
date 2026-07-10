#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

namespace deep_gemm {

// Standalone reduce-scatter COMM kernel, PULL-based 2-shot.
//
// Companion to a plain GEMM that wrote this rank's BF16 [shape_m, shape_n] partial into
// the symmetric buffer's `partial` region. This kernel:
//   1. NVLink barrier: ensure every rank's GEMM partial is written.
//   2. PULL + REDUCE: each rank owns rows [rank*m_per_rank, (rank+1)*m_per_rank). For its
//      shard, it reads the corresponding rows from ALL kNumRanks peers' partials (over
//      NVLink for remote, local for self), sums in FP32, writes the shard once.
//   3. NVLink barrier: so peers don't overwrite partials before everyone finished reading.
//
// vs. the push+slot+combine design (A-a), PULL avoids the local scratch DRAM round-trip:
// no slot storage written then re-read. Per rank: read (R-1)/R * mn * 2B remote + 1/R local,
// write shard once. This is the minimal traffic for a reduce-scatter.
//
// Symmetric buffer layout: [barrier (32 B)][FP32 output m_per_rank*n][BF16 partial shape_m*n]
template <uint32_t SHAPE_N, uint32_t kNumBlocks, uint32_t kNumRanks, uint32_t kNumThreads,
          uint32_t kMinBlocksPerSM = 8>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads, kMinBlocksPerSM)
sm100_reduce_scatter_comm_pull_impl(uint32_t shape_m, uint32_t shape_n,
                                    const uint32_t rank,
                                    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;

    const uint32_t block_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;

    const uint32_t m_per_rank = shape_m / kNumRanks;
    const uint64_t shard_elems = static_cast<uint64_t>(m_per_rank) * shape_n;
    const uint64_t partial_elems = static_cast<uint64_t>(shape_m) * shape_n;
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumRanks, 0u, 1u);
    // BF16 output shard (matches sglang's bf16 reduce-scatter output; half the write BW of fp32)
    auto* out_base = reinterpret_cast<nv_bfloat16*>(
        static_cast<uint8_t*>(sym_buffer.get_base_ptr()) + layout::Workspace::kNumBarrierSignalBytes);
    // Partial region: this rank's full [shape_m, shape_n] BF16 GEMM output
    auto* partial_base = reinterpret_cast<nv_bfloat16*>(
        reinterpret_cast<uint8_t*>(out_base) + shard_elems * sizeof(nv_bfloat16));

    constexpr uint32_t kNumElemsPerVec = 8;                        // uint4 == 8 bf16
    const uint32_t vecs_per_row = shape_n / kNumElemsPerVec;
    const uint64_t stride_vecs = static_cast<uint64_t>(kNumBlocks) * kNumThreads;
    const uint64_t base_v = static_cast<uint64_t>(block_idx) * kNumThreads + thread_idx;

    // ---- barrier: all ranks' GEMM partials written ----
    // This rank's GEMM completed before this kernel launched (same stream), so the
    // prologue grid-sync is unnecessary; we only need the cross-rank signal + a grid-sync
    // afterwards to guarantee every peer's partial is visible before we read it.
    comm::nvlink_barrier<kNumRanks, kNumBlocks, kNumThreads, /*kGridSyncIndex=*/0, /*kTag=*/0>(
        workspace, sym_buffer, block_idx, thread_idx, [&]() { __syncthreads(); },
        /*sync_prologue=*/false, /*sync_epilogue=*/true);

    // ---- PULL + REDUCE ----
    // This rank owns rows [rank*m_per_rank, (rank+1)*m_per_rank). Those rows live at
    // offset `rank*m_per_rank*shape_n` inside every peer's partial region. Issue ALL
    // kNumRanks * kNumUnroll remote loads before consuming them (max memory-level
    // parallelism to hide NVLink latency), then reduce in FP32.
    const uint64_t owner_row_offset_vecs = static_cast<uint64_t>(rank) * m_per_rank * vecs_per_row;
    const uint64_t out_total_vecs = shard_elems / kNumElemsPerVec;
    constexpr uint32_t kNumUnroll = 2;
    const uint64_t unroll_stride = stride_vecs * kNumUnroll;

    // Precompute peer partial base pointers (this rank's shard rows) once
    const uint4* peer_ptr[kNumRanks];
    #pragma unroll
    for (uint32_t j = 0; j < kNumRanks; ++ j)
        peer_ptr[j] = reinterpret_cast<const uint4*>(sym_buffer.map(partial_base, j)) + owner_row_offset_vecs;

    for (uint64_t vb = base_v; vb < out_total_vecs; vb += unroll_stride) {
        uint4 reg[kNumRanks][kNumUnroll];
        // Issue ALL remote loads first (kNumRanks * kNumUnroll independent -> deep MLP)
        #pragma unroll
        for (uint32_t j = 0; j < kNumRanks; ++ j) {
            #pragma unroll
            for (uint32_t u = 0; u < kNumUnroll; ++ u) {
                const uint64_t v = vb + static_cast<uint64_t>(u) * stride_vecs;
                if (v < out_total_vecs)
                    reg[j][u] = *(peer_ptr[j] + v);
            }
        }
        // Reduce in FP32
        #pragma unroll
        for (uint32_t u = 0; u < kNumUnroll; ++ u) {
            const uint64_t v = vb + static_cast<uint64_t>(u) * stride_vecs;
            if (v < out_total_vecs) {
                float acc[kNumElemsPerVec];
                #pragma unroll
                for (uint32_t e = 0; e < kNumElemsPerVec; ++ e)
                    acc[e] = 0.0f;
                #pragma unroll
                for (uint32_t j = 0; j < kNumRanks; ++ j) {
                    const nv_bfloat16* bf = reinterpret_cast<const nv_bfloat16*>(&reg[j][u]);
                    #pragma unroll
                    for (uint32_t e = 0; e < kNumElemsPerVec; ++ e)
                        acc[e] += __bfloat162float(bf[e]);
                }
                uint4 packed;
                packed.x = math::cast_into_bf16_and_pack(acc[0], acc[1]);
                packed.y = math::cast_into_bf16_and_pack(acc[2], acc[3]);
                packed.z = math::cast_into_bf16_and_pack(acc[4], acc[5]);
                packed.w = math::cast_into_bf16_and_pack(acc[6], acc[7]);
                *(reinterpret_cast<uint4*>(out_base) + v) = packed;
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

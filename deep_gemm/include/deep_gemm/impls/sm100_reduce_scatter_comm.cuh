#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

namespace deep_gemm {

// Standalone reduce-scatter COMM kernel (push + NVLink barrier + local combine).
//
// Companion to a plain GEMM that produced this rank's BF16 [shape_m, shape_n] partial
// in `local_scratch`. This kernel:
//   1. PUSH: grid-strided plain vectorized store of local scratch into the owner rank's
//      per-rank scratch slot (NO atomic), MLP-unrolled.
//   2. NVLink barrier (grid sync + cross-rank signal + grid sync).
//   3. COMBINE: sum kNumRanks bf16 slots in FP32 -> FP32 output.
//
// Because it uses ~no shared memory, it runs at full occupancy (unlike the fused kernel
// which is capped at 1 block/SM by the GEMM's 230 KB smem), so the memory-bound push +
// combine saturate DRAM/NVLink bandwidth.
//
// Symmetric buffer layout: [barrier (32 B)][FP32 output m_per_rank*n][BF16 scratch kNumRanks*m_per_rank*n]
template <uint32_t SHAPE_N, uint32_t kNumSMs, uint32_t kNumRanks, uint32_t kNumThreads,
          uint32_t kMinBlocksPerSM = 6>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads, kMinBlocksPerSM)
sm100_reduce_scatter_comm_impl(uint32_t shape_m, uint32_t shape_n,
                               const uint32_t rank,
                               const nv_bfloat16* __restrict__ local_scratch,
                               const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;

    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;

    const uint32_t m_per_rank = shape_m / kNumRanks;
    const uint64_t shard_elems = static_cast<uint64_t>(m_per_rank) * shape_n;
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumRanks, 0u, 1u);
    auto* out_base = reinterpret_cast<float*>(
        static_cast<uint8_t*>(sym_buffer.get_base_ptr()) + layout::Workspace::kNumBarrierSignalBytes);
    auto* scratch_base = reinterpret_cast<nv_bfloat16*>(
        reinterpret_cast<uint8_t*>(out_base) + shard_elems * sizeof(float));

    constexpr uint32_t kNumElemsPerVec = 8;                        // uint4 == 8 bf16
    constexpr uint32_t kNumUnroll = 4;
    const uint64_t total_vecs = (static_cast<uint64_t>(shape_m) * shape_n) / kNumElemsPerVec;
    const uint32_t vecs_per_row = shape_n / kNumElemsPerVec;
    const uint64_t stride_vecs = static_cast<uint64_t>(kNumSMs) * kNumThreads;
    const uint64_t slot_stride_vecs = shard_elems / kNumElemsPerVec;
    const uint64_t base_v = static_cast<uint64_t>(sm_idx) * kNumThreads + thread_idx;

    // ---- PUSH ----
    const uint64_t unroll_stride = stride_vecs * kNumUnroll;
    for (uint64_t vb = base_v; vb < total_vecs; vb += unroll_stride) {
        uint4 reg[kNumUnroll];
        #pragma unroll
        for (uint32_t u = 0; u < kNumUnroll; ++ u) {
            const uint64_t v = vb + static_cast<uint64_t>(u) * stride_vecs;
            if (v < total_vecs)
                reg[u] = *(reinterpret_cast<const uint4*>(local_scratch) + v);
        }
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

    // ---- NVLink barrier ----
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, /*kGridSyncIndex=*/0, /*kTag=*/0>(
        workspace, sym_buffer, sm_idx, thread_idx, [&]() { __syncthreads(); });

    // ---- COMBINE ----
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

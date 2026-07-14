#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

namespace deep_gemm {

// Standalone PUSH-RING reduce-scatter COMM kernel (validate the ring logic before fusing
// it into the GEMM epilogue). Companion to a plain GEMM that produced this rank's BF16
// [shape_m, shape_n] partial in `local_scratch` (owner `o`'s contribution = rows
// [o*m_per_rank, (o+1)*m_per_rank)).
//
// Ring reduce-scatter (direction i -> (i-1), R = num_ranks):
//   Each rank i, for step t = 0..R-1, handles owner c = (i+1+t) % R (staggered start).
//     d = (i-c+R)%R ; hop = R-1-d ; up=(i+1)%R ; down=(i-1+R)%R
//     START  (d==R-1): running_sum = my partial[c]                -> push to down's ring slot[c]
//     MIDDLE (0<d<R-1): wait my flag[c] -> read my ring slot[c] + my partial[c] -> add
//                       -> push to down's ring slot[c]
//     END    (d==0,i==c): wait my flag[c] -> read my ring slot[c] + my partial[c] -> add
//                       -> write my OUTPUT
//   After the store, the writer fences (system) and sets the DOWNSTREAM rank's flag[c].
//   Deadlock-free: dep (i,c)@hop h -> (up(i),c)@hop h-1 (strictly smaller) => linear DAG.
//
// Symmetric buffer layout (identical on every rank):
//   [ barrier region (32 B) ]
//   [ output   : m_per_rank * N * sizeof(bf16) ]
//   [ ring recv: kNumRanks * m_per_rank * N * sizeof(bf16) ]   (kNumRanks owner slots)
//   [ flags    : kNumRanks * int ]                             (per-owner ready flag)
//
// NOTE: this standalone kernel processes each owner segment as a whole (grid-strided over
// its elements) with a grid_sync between ring steps, so the per-owner flag is enough (no
// per-tile flag needed). The fused version will need per-tile flags.
template <uint32_t SHAPE_N, uint32_t kNumSMs, uint32_t kNumRanks, uint32_t kNumThreads>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads, 4)
sm100_reduce_scatter_ring_impl(uint32_t shape_m, uint32_t shape_n,
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
    // Regions in this rank's symmetric buffer.
    auto* out_base = reinterpret_cast<nv_bfloat16*>(
        static_cast<uint8_t*>(sym_buffer.get_base_ptr()) + layout::Workspace::kNumBarrierSignalBytes);
    auto* ring_base = reinterpret_cast<nv_bfloat16*>(
        reinterpret_cast<uint8_t*>(out_base) + shard_elems * sizeof(nv_bfloat16));
    auto* flag_base = reinterpret_cast<int*>(
        reinterpret_cast<uint8_t*>(ring_base) + kNumRanks * shard_elems * sizeof(nv_bfloat16));

    // Ring neighbours.
    const uint32_t down = (rank + kNumRanks - 1) % kNumRanks;   // where I STORE my running-sum

    // Vectorized element addressing (uint4 = 8 bf16 = 16 B).
    constexpr uint32_t kNumElemsPerVec = 8;
    const uint64_t total_vecs = shard_elems / kNumElemsPerVec;
    const uint64_t stride_vecs = static_cast<uint64_t>(kNumSMs) * kNumThreads;
    const uint64_t base_v = static_cast<uint64_t>(sm_idx) * kNumThreads + thread_idx;
    constexpr uint32_t kNumUnroll = 4;
    const uint64_t unroll_stride = stride_vecs * kNumUnroll;

    // Helper: this rank's local partial for owner `c` (rows [c*m_per_rank, ...)).
    auto my_partial_vec = [&](uint32_t c) {
        return reinterpret_cast<const uint4*>(local_scratch) + static_cast<uint64_t>(c) * total_vecs;
    };
    // Helper: my ring recv slot for owner `c`.
    auto my_ring_vec = [&](uint32_t c) {
        return reinterpret_cast<uint4*>(ring_base) + static_cast<uint64_t>(c) * total_vecs;
    };
    // Helper: DOWNSTREAM rank's ring recv slot for owner `c` (peer pointer).
    auto down_ring_vec = [&](uint32_t c) {
        auto* p = reinterpret_cast<uint4*>(ring_base) + static_cast<uint64_t>(c) * total_vecs;
        return reinterpret_cast<uint4*>(sym_buffer.map(reinterpret_cast<nv_bfloat16*>(p), down));
    };
    // Output slot (owner == rank).
    auto my_out_vec = [&]() { return reinterpret_cast<uint4*>(out_base); };

    // Set the DOWNSTREAM rank's flag for owner c (release, system scope).
    // Called AFTER a grid_sync (all my blocks finished writing dst), so a single writer
    // (block 0, thread 0) setting the flag once is sufficient and correct.
    auto set_down_flag = [&](uint32_t c) {
        __threadfence_system();
        if (sm_idx == 0 and thread_idx == 0)
            ptx::red_add_rel_sys(sym_buffer.map(flag_base + c, down), 1);
    };
    // Wait until MY flag for owner c has been set by my upstream (acquire, system scope).
    auto wait_my_flag = [&](uint32_t c) {
        if (thread_idx == 0) {
            while (ptx::ld_acq_sys(flag_base + c) == 0) {}
        }
        __syncthreads();
    };

    auto grid_sync = [&]() {
        comm::grid_sync<kNumSMs, /*kGridSyncIndex=*/0>(
            workspace, sm_idx, thread_idx, [&]() { __syncthreads(); });
    };

    // ================= reset flags (safe for repeated launches on the same buffer) =================
    // Every consumer waits on flag[c]==0 -> 1; flags must start at 0 each launch. Zero them,
    // then a cross-rank NVLink barrier so no rank begins the ring (and starts setting a peer's
    // flag) before every rank has finished zeroing its own flags.
    if (sm_idx == 0 and thread_idx < kNumRanks)
        flag_base[thread_idx] = 0;
    grid_sync();
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, /*kGridSyncIndex=*/1, /*kTag=*/0>(
        workspace, sym_buffer, sm_idx, thread_idx, [&]() { __syncthreads(); });

    // ================= ring steps =================
    #pragma unroll 1
    for (uint32_t t = 0; t < kNumRanks; ++ t) {
        const uint32_t c = (rank + 1 + t) % kNumRanks;   // owner handled this step
        const uint32_t d = (rank - c + kNumRanks) % kNumRanks;
        const bool is_start = (d == kNumRanks - 1);
        const bool is_end   = (d == 0);

        const uint4* upstream = is_start ? nullptr : my_ring_vec(c);   // upstream sum in MY buffer
        const uint4* mine     = my_partial_vec(c);
        uint4* dst            = is_end ? my_out_vec() : down_ring_vec(c);

        if (not is_start)
            wait_my_flag(c);

        // running_sum = mine (+ upstream if not start) -> dst
        for (uint64_t vb = base_v; vb < total_vecs; vb += unroll_stride) {
            #pragma unroll
            for (uint32_t u = 0; u < kNumUnroll; ++ u) {
                const uint64_t v = vb + static_cast<uint64_t>(u) * stride_vecs;
                if (v >= total_vecs) continue;
                uint4 acc = mine[v];
                if (not is_start) {
                    uint4 up = upstream[v];
                    const nv_bfloat16* a = reinterpret_cast<const nv_bfloat16*>(&acc);
                    const nv_bfloat16* b = reinterpret_cast<const nv_bfloat16*>(&up);
                    nv_bfloat16 r[kNumElemsPerVec];
                    #pragma unroll
                    for (uint32_t e = 0; e < kNumElemsPerVec; ++ e)
                        r[e] = __hadd(a[e], b[e]);
                    acc = *reinterpret_cast<const uint4*>(r);
                }
                dst[v] = acc;
            }
        }

        // Make my store visible, then tell downstream its flag[c] is ready.
        // (END writes to own output: no downstream flag needed.)
        if (not is_end) {
            grid_sync();                 // ensure ALL blocks finished writing dst before flag
            set_down_flag(c);
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop

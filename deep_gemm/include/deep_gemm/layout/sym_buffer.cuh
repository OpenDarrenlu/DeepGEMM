#pragma once

#include <cuda.h>

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::layout {

constexpr static uint32_t kNumMaxRanks = 72;

// A fixed-size array of TMA CD descriptors, one per owner rank, passed to the fused
// reduce-scatter kernel as a `__grid_constant__` param so the epilogue can TMA-store a
// tile directly into the owner rank's peer symmetric slot (verified: SM100 TMA can write
// peer memory over NVLink). Sized to the max so the host-side launcher `Args` is not
// templated; the kernel receives only its `kNumRanks` prefix.
template <uint32_t kNumRanks = kNumMaxRanks>
struct CdTmaMaps {
    CUtensorMap maps[kNumRanks];
};

template <uint32_t kNumRanks = kNumMaxRanks>
struct SymBuffer {
    int64_t base;
    int64_t offsets[kNumMaxRanks];
    uint32_t rank_idx;

    DG_STATIC_ASSERT(kNumRanks <= kNumMaxRanks, "Too many ranks");

    SymBuffer() = default;

    template <typename Container>
    explicit SymBuffer(const Container& c, const uint32_t& rank_idx): rank_idx(rank_idx) {
        const auto size = static_cast<uint32_t>(c.size());
        base = c[rank_idx];
        for (uint32_t i = 0; i < kNumMaxRanks; ++ i)
            offsets[i] = i < size ? (c[i] - base) : 0;
    }

#if defined(__CUDA_ARCH__) or defined(__CLION_IDE__)
    template <typename ptr_t = void*>
    CUTLASS_DEVICE ptr_t get_base_ptr() const {
        return reinterpret_cast<ptr_t>(base);
    }

    template <typename ptr_t>
    CUTLASS_DEVICE ptr_t map(const ptr_t& ptr, const uint32_t& dst_rank_idx) const {
        int64_t mapped_ptr = offsets[dst_rank_idx] + reinterpret_cast<int64_t>(ptr);
        return *reinterpret_cast<ptr_t*>(&mapped_ptr);
    }
#endif
};

} // namespace deep_gemm::layout

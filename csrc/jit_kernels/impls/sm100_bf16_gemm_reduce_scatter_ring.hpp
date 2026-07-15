#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>   // layout::Workspace::kNumBarrierSignalBytes

namespace deep_gemm {

class SM100BF16GemmReduceScatterRingRuntime final: public LaunchRuntime<SM100BF16GemmReduceScatterRingRuntime> {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        LaunchArgs launch_args;

        int num_ranks;
        uint32_t rank;
        layout::SymBuffer<> sym_buffer;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_partial_load;    // over THIS rank's ring recv region [R*m_per_rank, n]
        CUtensorMap tensor_map_ring_store_down;  // over DOWN peer's ring recv region [R*m_per_rank, n]
        CUtensorMap tensor_map_out;              // over THIS rank's output region [m_per_rank, n]
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_gemm_reduce_scatter_ring.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_gemm_reduce_scatter_ring_impl<
        {}, {},
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {},
        {},
        {},
        {}
    >);
}};
)",
        to_string(args.gemm_desc.major_a), to_string(args.gemm_desc.major_b),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode, args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_sms,
        args.num_ranks,
        args.gemm_desc.tc_util);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.rank,
            args.sym_buffer,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_partial_load, args.tensor_map_ring_store_down, args.tensor_map_out));
    }
};

// Fused BF16 GEMM + PUSH-RING ReduceScatter (BF16 comm, reduction folded into a bidirectional
// ring transfer inside the GEMM epilogue).
//
// `out_sym_buffer` symmetric layout (identical on every rank):
//   [ barrier (32 B) ][ BF16 output: m_per_rank*n ][ BF16 ring recv: R * m_per_rank * n ]
//   [ int flags: num_m_blocks * num_n_blocks ]
// `sym_buffer_ptrs`: all peers' base pointers into that symmetric allocation.
static void sm100_bf16_gemm_reduce_scatter_ring(const torch::Tensor& a,
                                                const torch::Tensor& b,
                                                const torch::Tensor& out_sym_buffer,
                                                const std::vector<int64_t>& sym_buffer_ptrs,
                                                const int& rank,
                                                const int& m, const int& n, const int& k,
                                                const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                const std::string& compiled_dims) {
    const auto [m_a, k_a] = get_shape<2>(a);
    const auto [n_b, k_b] = get_shape<2>(b);
    DG_HOST_ASSERT(k_a == k_b);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16 and b.scalar_type() == torch::kBFloat16);

    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 10 and "reduce_scatter_ring only supports SM100");
    DG_HOST_ASSERT(m % num_ranks == 0);
    // Stagger owner'=(owner+rank+1)%R is a rotation (bijection) for any R>=2; power-of-2 not required.
    DG_HOST_ASSERT(num_ranks >= 2);

    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = torch::kBFloat16,
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims
    };

    // Single-CTA swap-AB layout with a deterministic BLOCK_M (matches the Python wrapper's pad):
    //   chosen_block_m = min(256, m_per_rank).  BLOCK_N locked to 128.
    const int chosen_block_m = std::min(256, m / num_ranks);
    const auto all_candidates = SM100ArchSpec::get_layout_candidates(desc);
    std::vector<Layout> candidates;
    for (const auto& c: all_candidates) {
        if (c.swap_ab and c.get_cluster_size() == 1 and c.block_n == 128 and c.block_m == chosen_block_m)
            candidates.push_back(c);
    }
    DG_HOST_ASSERT(not candidates.empty() and "No single-CTA swap-AB BLOCK_N=128 config at chosen BLOCK_M");

    auto layout = candidates[0];
    auto layout_info = SM100ArchSpec::get_layout_info(desc, layout);
    for (int i = 1; i < static_cast<int>(candidates.size()); ++ i) {
        const auto candidate_info = SM100ArchSpec::get_layout_info(desc, candidates[i]);
        if (SM100ArchSpec::compare(candidate_info, layout_info))
            layout = candidates[i], layout_info = candidate_info;
    }
    const auto storage_config = SM100ArchSpec::get_storage_config(desc, layout);
    auto config = GemmConfig {
        .layout = layout,
        .storage_config = storage_config,
        .pipeline_config = SM100ArchSpec::get_pipeline_config(desc, layout, storage_config),
        .launch_config = SM100ArchSpec::get_launch_config(desc, layout)
    };
    DG_HOST_ASSERT((m / num_ranks) % config.layout.block_m == 0 and
                   "owner segment must be a multiple of BLOCK_M (pad in the wrapper)");
    // 3 warpgroups = 384 threads (WG0 producer / WG1 epilogue / WG2 ring).
    config.launch_config.num_threads = 384;

    // Recompute the A/B pipeline depth for THIS kernel's smem layout, which adds a
    // double-buffered partial_buf and extra barrier sets on top of the base epi_smem/A/B.
    // Device layout (bytes): [ epi_smem (CD, 2 stages) ][ partial_buf (2 stages) ]
    //   [ A: ns stages ][ B: ns stages ][ barriers: (2*ns + 2*2 + 2*2 + 2*2) * 8 ][ tmem_ptr: 4 ].
    {
        constexpr int kElem = 2;                    // bf16
        constexpr int kStoreBlockM = 16, kStoreBlockN = 128;
        constexpr int kEpiStages = 2, kTMAStoreStages = 2, kPartialStages = 2;
        const int block_m = config.layout.block_m, block_n = config.layout.block_n, block_k = config.layout.block_k;
        const int smem_cd      = kStoreBlockM * kStoreBlockN * kElem * kTMAStoreStages;   // epi_smem
        const int smem_partial = kStoreBlockM * kStoreBlockN * kElem * kPartialStages;    // partial_buf
        const int smem_a_per   = block_m * block_k * kElem;
        const int smem_b_per   = block_n * block_k * kElem;
        const int smem_fixed   = smem_cd + smem_partial + 4 /*tmem_ptr*/ + 1024 /*align slack*/;
        auto device_smem = [&](int ns) {
            const int barriers = (2 * ns + kEpiStages * 2 + kTMAStoreStages * 2 + kPartialStages * 2) * 8;
            return smem_fixed + ns * (smem_a_per + smem_b_per) + barriers;
        };
        int ns = 1;
        while (ns + 1 <= 32 and device_smem(ns + 1) <= SM100ArchSpec::smem_capacity)
            ns += 1;
        DG_HOST_ASSERT(device_smem(ns) <= SM100ArchSpec::smem_capacity and "smem overflow");
        config.pipeline_config.num_stages = ns;
        config.pipeline_config.smem_size = device_smem(ns);
    }


    const int m_per_rank = m / num_ranks;
    const int64_t barrier_bytes = static_cast<int64_t>(layout::Workspace::kNumBarrierSignalBytes);
    const int64_t out_bytes = static_cast<int64_t>(m_per_rank) * n * static_cast<int64_t>(sizeof(nv_bfloat16));

    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m, config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n, config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), 1,
                                              config.storage_config.swizzle_b_mode);

    // Region base pointers.
    auto* my_base   = reinterpret_cast<uint8_t*>(sym_buffer_ptrs[rank]);
    auto* down_base = reinterpret_cast<uint8_t*>(sym_buffer_ptrs[(rank + num_ranks - 1) % num_ranks]);
    auto* my_out_ptr   = reinterpret_cast<void*>(my_base + barrier_bytes);
    auto* my_ring_ptr  = reinterpret_cast<void*>(my_base + barrier_bytes + out_bytes);
    auto* down_ring_ptr = reinterpret_cast<void*>(down_base + barrier_bytes + out_bytes);

    // partial-load: read THIS rank's ring recv [R*m_per_rank, n], 16-row store blocks.
    const auto tensor_map_partial_load = make_tma_cd_desc_raw(my_ring_ptr, torch::kBFloat16,
                                                              num_ranks * m_per_rank, n,
                                                              config.storage_config.store_block_m,
                                                              config.storage_config.store_block_n,
                                                              n, config.storage_config.swizzle_cd_mode);
    // ring-store: write DOWN peer's ring recv [R*m_per_rank, n].
    const auto tensor_map_ring_store_down = make_tma_cd_desc_raw(down_ring_ptr, torch::kBFloat16,
                                                                 num_ranks * m_per_rank, n,
                                                                 config.storage_config.store_block_m,
                                                                 config.storage_config.store_block_n,
                                                                 n, config.storage_config.swizzle_cd_mode);
    // out: write THIS rank's output [m_per_rank, n].
    const auto tensor_map_out = make_tma_cd_desc_raw(my_out_ptr, torch::kBFloat16,
                                                     m_per_rank, n,
                                                     config.storage_config.store_block_m,
                                                     config.storage_config.store_block_n,
                                                     n, config.storage_config.swizzle_cd_mode);

    const SM100BF16GemmReduceScatterRingRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .num_ranks = num_ranks,
        .rank = static_cast<uint32_t>(rank),
        .sym_buffer = layout::SymBuffer<>(sym_buffer_ptrs, static_cast<uint32_t>(rank)),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_partial_load = tensor_map_partial_load,
        .tensor_map_ring_store_down = tensor_map_ring_store_down,
        .tensor_map_out = tensor_map_out
    };
    const auto code = SM100BF16GemmReduceScatterRingRuntime::generate(args);
    const auto runtime = compiler->build("sm100_bf16_gemm_reduce_scatter_ring", code);
    SM100BF16GemmReduceScatterRingRuntime::launch(runtime, args);
}

} // namespace deep_gemm

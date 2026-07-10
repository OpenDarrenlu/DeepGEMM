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

namespace deep_gemm {

class SM100BF16GemmReduceScatterRuntime final: public LaunchRuntime<SM100BF16GemmReduceScatterRuntime> {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        LaunchArgs launch_args;

        int num_ranks;
        uint32_t rank;
        void* local_scratch;
        layout::SymBuffer<> sym_buffer;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_gemm_reduce_scatter.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_gemm_reduce_scatter_impl<
        {}, {},
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {},
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
        args.gemm_config.launch_config.num_non_epilogue_threads, args.gemm_config.launch_config.num_epilogue_threads,
        args.gemm_config.launch_config.num_sms,
        args.num_ranks,
        args.gemm_desc.tc_util);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.rank,
            args.local_scratch,
            args.sym_buffer,
            args.tensor_map_a, args.tensor_map_b, args.tensor_map_cd));
    }
};

// Fused BF16 GEMM + ReduceScatter (BF16 comm + plain remote store + local FP32 combine).
//
// `local_scratch`: this rank's BF16 [m, n] partial (phase-1 GEMM output).
// `out_sym_buffer`: symmetric buffer laid out as
//   [ barrier (32 B) ][ FP32 output: m_per_rank*n ][ BF16 scratch: num_ranks * m_per_rank * n ]
// `sym_buffer_ptrs`: all peers' base pointers into that symmetric allocation.
static void sm100_bf16_gemm_reduce_scatter(const torch::Tensor& a,
                                              const torch::Tensor& b,
                                              const torch::Tensor& local_scratch,
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
    DG_HOST_ASSERT(local_scratch.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(local_scratch.numel() == static_cast<int64_t>(m) * n);

    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 10 and "reduce_scatter only supports SM100");
    DG_HOST_ASSERT(m % num_ranks == 0);

    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = torch::kBFloat16,           // <-- BF16 CD staging
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims
    };

    // Restrict to single-CTA non-swap-AB layouts (matching the kernel), pick the best.
    const auto all_candidates = SM100ArchSpec::get_layout_candidates(desc);
    std::vector<Layout> candidates;
    for (const auto& c: all_candidates) {
        if (not c.swap_ab and c.get_cluster_size() == 1 and
            ceil_div(m, c.block_m) % num_ranks == 0)
            candidates.push_back(c);
    }
    DG_HOST_ASSERT(not candidates.empty() and "No single-CTA non-swap-AB config for this shape");

    auto layout = candidates[0];
    auto layout_info = SM100ArchSpec::get_layout_info(desc, layout);
    for (int i = 1; i < static_cast<int>(candidates.size()); ++ i) {
        const auto candidate_info = SM100ArchSpec::get_layout_info(desc, candidates[i]);
        if (SM100ArchSpec::compare(candidate_info, layout_info))
            layout = candidates[i], layout_info = candidate_info;
    }
    const auto config = GemmConfig {
        .layout = layout,
        .storage_config = SM100ArchSpec::get_storage_config(desc, layout),
        .pipeline_config = SM100ArchSpec::get_pipeline_config(desc, layout, SM100ArchSpec::get_storage_config(desc, layout)),
        .launch_config = SM100ArchSpec::get_launch_config(desc, layout)
    };
    DG_HOST_ASSERT(m % (config.layout.block_m * num_ranks) == 0);

    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), 1,
                                              config.storage_config.swizzle_b_mode);
    // Local BF16 [m, n] scratch for the phase-1 TMA store
    const auto tensor_map_cd = make_tma_cd_desc(local_scratch, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(local_scratch.stride(-2)), 1,
                                                config.storage_config.swizzle_cd_mode);

    const SM100BF16GemmReduceScatterRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .num_ranks = num_ranks,
        .rank = static_cast<uint32_t>(rank),
        .local_scratch = local_scratch.data_ptr(),
        .sym_buffer = layout::SymBuffer<>(sym_buffer_ptrs, static_cast<uint32_t>(rank)),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd
    };
    const auto code = SM100BF16GemmReduceScatterRuntime::generate(args);
    const auto runtime = compiler->build("sm100_bf16_gemm_reduce_scatter", code);
    SM100BF16GemmReduceScatterRuntime::launch(runtime, args);
}

} // namespace deep_gemm

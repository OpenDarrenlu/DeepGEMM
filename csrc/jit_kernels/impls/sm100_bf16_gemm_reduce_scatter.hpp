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
        layout::CdTmaMaps<> tensor_map_cd_owners;   // one CD map per owner (peer slot)
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
            args.tensor_map_a, args.tensor_map_b, args.tensor_map_cd_owners));
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
    // XOR rank-swizzle in the kernel is a bijection over owners only for power-of-2 ranks.
    DG_HOST_ASSERT((num_ranks & (num_ranks - 1)) == 0 and "num_ranks must be a power of 2");

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

    // Restrict to single-CTA swap-AB layouts (matching the kernel), with a DETERMINISTIC
    // BLOCK_M that the Python wrapper also computes (so tiles are single-owner without any
    // cross-language handshake). `m` arrives already padded, so each owner segment
    // m_per_rank = m/num_ranks is an exact multiple of the chosen BLOCK_M.
    //   chosen_block_m = min(256, m_per_rank)  == deep_gemm.comm.rs_block_m(real_m_per_rank)
    // (both round the real per-rank count up to a multiple of 16, then cap at 256; here
    // m_per_rank is already that padded multiple, so min(256, m_per_rank) reproduces it).
    // swap-AB: token(M) -> UMMA N-dim = BLOCK_M (16..256), hidden(N) -> UMMA M-dim = BLOCK_N = 128.
    // BLOCK_M<=256 from the 2-stage TMEM cap (2*BLOCK_M <= 512). Owner segments padded to a
    // multiple of BLOCK_M so 16-row store blocks stay within one owner.
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
    // NOTES: the RS swap-AB epilogue stores in 16-row (umma_step_n) blocks, so
    // storage_config.store_block_m is 16 (from get_storage_config) — matching the device
    // kernel's STORE_BLOCK_M=16. The CD tensor-map smem box and host pipeline smem accounting
    // both use this 16, consistent with the device's SMEM_CD_SIZE = 16*128*2*kNumTMAStoreStages.
    const auto config = GemmConfig {
        .layout = layout,
        .storage_config = storage_config,
        .pipeline_config = SM100ArchSpec::get_pipeline_config(desc, layout, storage_config),
        .launch_config = SM100ArchSpec::get_launch_config(desc, layout)
    };
    // `m` is already padded (by the Python wrapper) so each owner's segment
    // `m_per_rank = m / num_ranks` is an exact multiple of BLOCK_M (single-owner tiles).
    DG_HOST_ASSERT((m / num_ranks) % config.layout.block_m == 0 and
                   "owner segment must be a multiple of BLOCK_M (pad in the wrapper)");

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
    // Per-owner CD tensor maps: the epilogue TMA-stores each 16-row (STORE_BLOCK_M) token
    // block directly into the OWNER rank's peer symmetric slot (owner segments are 16-aligned,
    // so each block lands wholly in one owner). The descriptor smem box MUST match the device's
    // SMEM_CD staging: [store_block_m = 16 rows, swizzle/elem cols]. `m_per_rank` is the PADDED
    // per-rank row count (m is padded by the wrapper to a multiple of BLOCK_M, itself a multiple
    // of 16), so the box [m_local, m_local+16) never overruns the owner slot.
    // Slot region base in owner `o`'s buffer =
    //   sym_buffer_ptrs[o] + BARRIER_BYTES + fp32_out_bytes + rank * slot_bytes.
    const int m_per_rank = m / num_ranks;
    const int64_t barrier_bytes = static_cast<int64_t>(layout::Workspace::kNumBarrierSignalBytes);
    const int64_t fp32_out_bytes = static_cast<int64_t>(m_per_rank) * n * static_cast<int64_t>(sizeof(float));
    const int64_t slot_bytes = static_cast<int64_t>(m_per_rank) * n * static_cast<int64_t>(sizeof(nv_bfloat16));
    layout::CdTmaMaps<> tensor_map_cd_owners{};
    for (int o = 0; o < num_ranks; ++ o) {
        auto* slot_ptr = reinterpret_cast<void*>(
            sym_buffer_ptrs[o] + barrier_bytes + fp32_out_bytes + static_cast<int64_t>(rank) * slot_bytes);
        // Raw peer pointer (owner o's buffer, this rank's slot) -> TMA CD descriptor.
        tensor_map_cd_owners.maps[o] = make_tma_cd_desc_raw(slot_ptr, torch::kBFloat16, m_per_rank, n,
                                                            config.storage_config.store_block_m,
                                                            config.storage_config.store_block_n,
                                                            n,
                                                            config.storage_config.swizzle_cd_mode);
    }

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
        .tensor_map_cd_owners = tensor_map_cd_owners
    };
    const auto code = SM100BF16GemmReduceScatterRuntime::generate(args);
    const auto runtime = compiler->build("sm100_bf16_gemm_reduce_scatter", code);
    SM100BF16GemmReduceScatterRuntime::launch(runtime, args);
}

} // namespace deep_gemm

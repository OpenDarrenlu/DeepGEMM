#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"

#include <deep_gemm/layout/sym_buffer.cuh>

namespace deep_gemm {

class SM100ReduceScatterCommPullRuntime final: public LaunchRuntime<SM100ReduceScatterCommPullRuntime> {
public:
    struct Args {
        int shape_m, shape_n;
        int num_blocks, num_threads, num_ranks;
        uint32_t rank;
        layout::SymBuffer<> sym_buffer;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_reduce_scatter_comm_pull.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_reduce_scatter_comm_pull_impl<
        {}, {}, {}, {}
    >);
}};
)",
        args.shape_n, args.num_blocks, args.num_ranks, args.num_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            static_cast<uint32_t>(args.shape_m), static_cast<uint32_t>(args.shape_n),
            args.rank, args.sym_buffer));
    }
};

static void sm100_reduce_scatter_comm_pull(const torch::Tensor& out_sym_buffer,
                                           const std::vector<int64_t>& sym_buffer_ptrs,
                                           const int& rank,
                                           const int& m, const int& n) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    DG_HOST_ASSERT(m % num_ranks == 0);
    DG_HOST_ASSERT(n % 8 == 0);

    constexpr int kNumThreads = 256;
    constexpr int kBlocksPerSM = 8;
    const int num_sms = device_runtime->get_num_sms();
    const int num_blocks = num_sms * kBlocksPerSM;

    const SM100ReduceScatterCommPullRuntime::Args args = {
        .shape_m = m, .shape_n = n,
        .num_blocks = num_blocks, .num_threads = kNumThreads, .num_ranks = num_ranks,
        .rank = static_cast<uint32_t>(rank),
        .sym_buffer = layout::SymBuffer<>(sym_buffer_ptrs, static_cast<uint32_t>(rank)),
        .launch_args = LaunchArgs(num_blocks, kNumThreads, /*smem=*/0, /*cluster=*/1, /*enable_pdl=*/false)
    };
    const auto code = SM100ReduceScatterCommPullRuntime::generate(args);
    const auto runtime = compiler->build("sm100_reduce_scatter_comm_pull", code);
    SM100ReduceScatterCommPullRuntime::launch(runtime, args);
}

} // namespace deep_gemm

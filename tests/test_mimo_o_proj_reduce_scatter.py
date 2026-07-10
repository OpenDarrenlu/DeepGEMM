"""
Fused BF16 o_proj GEMM + ReduceScatter for MiMo-V2-Flash under DP-attention.

This mirrors the *exact* communication path SGLang takes for MiMo-V2-Flash when
launched with `--tp-size 8 --dp-size 2 --enable-dp-attention`:

    attn_tp_size = tp_size / dp_size = 8 / 2 = 4
    attn_dp_size = 2                              (two independent DP groups)

In `sglang/srt/models/mimo_v2_flash.py`, `MiMoV2Attention.o_proj` is a
`RowParallelLinear(total_num_heads * v_head_dim -> hidden_size, tp_size=attn_tp_size,
reduce_results=False)`. Because `reduce_results=False`, each attention-TP rank only
produces a *partial* `[num_tokens, hidden_size]`. The cross-rank reduction happens
later in `LayerCommunicator.prepare_mlp -> _scatter_hidden_states_and_residual`, which
calls `attn_tp_reduce_scatter_tensor(...)` — a reduce-scatter over the 4-rank
attention-TP group. Rank r ends up with rows [r * m/4, (r+1) * m/4) of the summed
output, which then feeds `post_attention_layernorm`.

So per attention-TP rank the fused op is:

    a_r = attn_output shard   : [m, k]   bf16   (k = num_heads * v_head_dim / attn_tp = 2048)
    b_r = o_proj weight shard : [n, k]   bf16   (n = hidden_size = 4096)
    P_r = a_r @ b_r.T         : [m, n]   fp32   (partial, NOT reduced by o_proj)
    out = reduce_scatter_M(sum_r P_r) : [m/4, n] fp32  (this rank's shard)

which is precisely `deep_gemm.bf16_gemm_reduce_scatter(a_r, b_r, rs_buffer)` with the
`ReduceScatterBuffer` built on the 4-rank attention-TP subgroup.

Shapes are taken from `/preset-models/config.json`:
    hidden_size            = 4096
    num_attention_heads    = 64
    v_head_dim             = 128
    -> o_proj in-features  = 64 * 128 = 8192, sharded by attn_tp=4 -> k = 2048
    -> o_proj out-features = hidden_size = 4096 = n
"""

import argparse
import os
import random
import sys

import torch
import torch.distributed as dist

# Prefer the source tree over any installed `deep_gemm` in site-packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import deep_gemm
from deep_gemm.utils.dist import dist_print, init_dist


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def build_attn_tp_subgroups(world_size: int, attn_tp_size: int):
    """Create every attention-TP subgroup on every rank (dist.new_group is collective),
    and return this rank's subgroup plus its (dp_group_idx, rank_in_group)."""
    assert world_size % attn_tp_size == 0, 'world size must be divisible by attn_tp_size'
    num_dp_groups = world_size // attn_tp_size
    global_rank = dist.get_rank()

    my_group = None
    my_dp_idx = global_rank // attn_tp_size
    for dp_idx in range(num_dp_groups):
        ranks = list(range(dp_idx * attn_tp_size, (dp_idx + 1) * attn_tp_size))
        # NOTES: every rank must call `new_group` for *every* subgroup with identical args
        subgroup = dist.new_group(ranks)
        if dp_idx == my_dp_idx:
            my_group = subgroup
    return my_group, my_dp_idx, my_group.rank(), num_dp_groups


# noinspection PyShadowingNames
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    global_rank, world_size, _ = init_dist(local_rank, num_local_ranks)

    attn_tp_size = args.attn_tp_size
    assert world_size % attn_tp_size == 0, \
        f'world_size={world_size} must be divisible by attn_tp_size={attn_tp_size}'

    attn_tp_group, dp_group_idx, rank_in_group, num_dp_groups = \
        build_attn_tp_subgroups(world_size, attn_tp_size)

    # Distinct data per global rank; keep it reproducible
    torch.manual_seed(global_rank)
    random.seed(global_rank)

    m, n, k = args.m, args.n, args.k
    assert m % attn_tp_size == 0, 'num_tokens (M) must be divisible by attn_tp_size'
    m_per_rank = m // attn_tp_size

    if global_rank == 0:
        print(f'>>> world_size={world_size}, attn_tp_size={attn_tp_size}, '
              f'attn_dp_size={num_dp_groups} | per-rank GEMM: [{m},{k}] @ [{n},{k}].T -> '
              f'reduce-scatter -> [{m_per_rank},{n}] fp32', flush=True)

    # One symmetric reduce-scatter buffer per 4-rank attention-TP group
    rs_buffer = deep_gemm.ReduceScatterBuffer(attn_tp_group, m, n)
    rs_buffer_pull = deep_gemm.ReduceScatterBufferPull(attn_tp_group, m, n)

    for test_idx in range(args.num_tests):
        # `a`: this rank's attention-output shard [m (tokens), k (heads*v_head_dim/attn_tp)]
        # `b`: this rank's o_proj weight shard    [n (hidden), k]  (NT: out = a @ b.T)
        a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda') * 0.1
        b = torch.randn((n, k), dtype=torch.bfloat16, device='cuda') * 0.1

        # Reference: local partial, summed across the *subgroup*, sliced to this rank's shard
        local_partial = a.float() @ b.float().T
        full = local_partial.clone()
        dist.all_reduce(full, op=dist.ReduceOp.SUM, group=attn_tp_group)
        ref = full[rank_in_group * m_per_rank:(rank_in_group + 1) * m_per_rank]

        # Fused kernel (push + local combine) and pull-based split kernel
        out = deep_gemm.bf16_gemm_reduce_scatter(a, b, rs_buffer)
        out_pull = deep_gemm.bf16_gemm_reduce_scatter_pull(a, b, rs_buffer_pull)

        diff = calc_diff(out, ref)
        diff_pull = calc_diff(out_pull.float(), ref)
        ok = diff < 1e-2 and diff_pull < 1e-2  # bf16 comm -> looser tol
        dist_print(f'[test {test_idx}] global_rank={global_rank} dp_group={dp_group_idx} '
                   f'rank_in_group={rank_in_group} diff={diff:.3e} '
                   f'diff_pull={diff_pull:.3e} {"OK" if ok else "FAILED"}')
        assert ok, f'rank {global_rank}: mismatch (diff={diff:.3e} pull={diff_pull:.3e})'

    dist.barrier()
    if global_rank == 0:
        print('All MiMo o_proj reduce-scatter correctness tests passed!')

    rs_buffer.destroy()
    rs_buffer_pull.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test fused MiMo-V2-Flash o_proj BF16 GEMM + ReduceScatter (SM100, DP-attention)')
    parser.add_argument('--num-processes', type=int, default=8,
                        help='Total GPUs / world size (mimo launch uses tp-size 8)')
    parser.add_argument('--attn-tp-size', type=int, default=4,
                        help='Attention TP group size = tp_size / dp_size (8/2 = 4 for mimo)')
    parser.add_argument('--m', type=int, default=4096,
                        help='Num tokens (global M within a group); must be divisible by attn_tp_size')
    parser.add_argument('--n', type=int, default=4096, help='hidden_size (o_proj out-features)')
    parser.add_argument('--k', type=int, default=2048,
                        help='o_proj per-rank contraction = num_heads*v_head_dim/attn_tp (64*128/4)')
    parser.add_argument('--num-tests', type=int, default=5, help='Correctness iterations')
    args = parser.parse_args()

    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)

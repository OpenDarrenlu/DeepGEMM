"""
Accurate benchmark report: fused  vs  baseline (sglang path).

Scope (per the mimo dp2+tp8 topology, attn_tp_size=4):
  * fused     = deep_gemm fused BF16 GEMM + ReduceScatter, single kernel
                (bf16 push to owner slots + local fp32 combine).
  * baseline  = sglang path: deep_gemm bf16 GEMM  ->  NCCL reduce_scatter_tensor (bf16),
                the two run back-to-back (what sglang does today).

Both produce this rank's [num_tokens/attn_tp, N] shard. We sweep the number of tokens
(M) and report steady-state per-call wall time (CUDA events, group-synchronized) plus
the speedup the fused kernel delivers over baseline.

Fixed shape from /preset-models/config.json (o_proj): N = hidden_size = 4096,
K = num_heads*v_head_dim/attn_tp = 64*128/4 = 2048.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import deep_gemm
from deep_gemm import _C
from deep_gemm.utils.dist import init_dist


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return (1 - 2 * (x * y).sum() / denom).item()


def build_attn_tp_subgroup(world_size: int, attn_tp_size: int):
    assert world_size % attn_tp_size == 0
    num_dp = world_size // attn_tp_size
    gr = dist.get_rank()
    my = gr // attn_tp_size
    grp = None
    for d in range(num_dp):
        g = dist.new_group(list(range(d * attn_tp_size, (d + 1) * attn_tp_size)))
        if d == my:
            grp = g
    return grp, my, grp.rank(), num_dp


def bench_group(fn, group, warmups: int, iters: int) -> float:
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')
    cache.zero_()
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1e3  # seconds


# noinspection PyShadowingNames
def run(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    global_rank, world_size, _ = init_dist(local_rank, num_local_ranks)
    attn_tp = args.attn_tp_size
    group, dp_idx, rk, num_dp = build_attn_tp_subgroup(world_size, attn_tp)
    torch.manual_seed(global_rank)

    n, k = args.n, args.k
    token_list = [int(x) for x in args.tokens.split(',')]

    if global_rank == 0:
        print('\n' + '=' * 78, flush=True)
        print('Fused  vs  baseline (GEMM + NCCL reduce-scatter)  |  SM100', flush=True)
        print(f'world={world_size}  attn_tp={attn_tp}  dp_groups={num_dp}  '
              f'N(hidden)={n}  K(per-rank)={k}', flush=True)
        print(f'output shard per rank = [tokens/{attn_tp}, {n}]  '
              f'(warmups={args.num_warmups}, iters={args.num_tests})', flush=True)
        print('=' * 78, flush=True)
        hdr = (f'{"tokens":>8} {"tok/rank":>9} | {"fused (us)":>10} {"baseline(us)":>13} '
               f'| {"speedup":>8} | {"fused diff":>10} {"base diff":>10}')
        print(hdr, flush=True)
        print('-' * 78, flush=True)

    for m in token_list:
        if m % attn_tp != 0:
            continue
        m_per_rank = m // attn_tp

        a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda') * 0.1
        b = torch.randn((n, k), dtype=torch.bfloat16, device='cuda') * 0.1

        # Reference: subgroup all-reduce then slice
        full = a.float() @ b.float().T
        dist.all_reduce(full, op=dist.ReduceOp.SUM, group=group)
        ref = full[rk * m_per_rank:(rk + 1) * m_per_rank].contiguous()

        # ---- fused kernel ----
        rs_aa = deep_gemm.ReduceScatterBuffer(group, m, n)
        aa_scratch = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

        def aa_call():
            _C.bf16_gemm_reduce_scatter_nt(
                a, b, aa_scratch, rs_aa.buffer, rs_aa.buffer_ptrs, rs_aa.rank, 'nk')

        out_aa = deep_gemm.bf16_gemm_reduce_scatter(a, b, rs_aa)
        diff_aa = calc_diff(out_aa, ref)

        # ---- baseline: GEMM (bf16) + NCCL reduce_scatter_tensor (bf16) ----
        partial = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        shard = torch.empty((m_per_rank, n), dtype=torch.bfloat16, device='cuda')

        def base_call():
            _C.bf16_gemm_nt(a, b, partial, None, 'nk')
            dist.reduce_scatter_tensor(shard, partial, op=dist.ReduceOp.SUM, group=group)

        base_call()
        diff_base = calc_diff(shard.float(), ref)

        dist.barrier(group=group)
        t_aa = bench_group(aa_call, group, args.num_warmups, args.num_tests)
        t_base = bench_group(base_call, group, args.num_warmups, args.num_tests)

        if global_rank == 0:
            print(f'{m:>8} {m_per_rank:>9} | {t_aa*1e6:>10.1f} {t_base*1e6:>13.1f} '
                  f'| {t_base/t_aa:>7.2f}x | {diff_aa:>10.1e} {diff_base:>10.1e}', flush=True)

        rs_aa.destroy()
        del a, b, full, ref, aa_scratch, partial, shard

    if global_rank == 0:
        print('-' * 78, flush=True)
        print('fused: single fused kernel (push + local FP32 combine).  baseline: deep_gemm GEMM + NCCL reduce-scatter '
              '(= sglang today).', flush=True)
        print('diff = 1 - cosine-sim vs fp32 all-reduce reference (lower is better).\n', flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Accurate fused vs baseline report (SM100)')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--attn-tp-size', type=int, default=4)
    parser.add_argument('--tokens', type=str,
                        default='512,1024,1536,2048,4096,8192',
                        help='comma-separated token counts (global M within a group)')
    parser.add_argument('--n', type=int, default=4096, help='hidden_size')
    parser.add_argument('--k', type=int, default=2048, help='num_heads*v_head_dim/attn_tp')
    parser.add_argument('--num-warmups', type=int, default=20)
    parser.add_argument('--num-tests', type=int, default=100)
    args = parser.parse_args()

    torch.multiprocessing.spawn(run, args=(args.num_processes, args), nprocs=args.num_processes)

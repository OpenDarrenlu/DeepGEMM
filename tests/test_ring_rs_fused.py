"""Correctness test for the FULLY FUSED BF16 GEMM + PUSH-RING reduce-scatter kernel
(bf16_gemm_reduce_scatter_ring_fused). Compares vs the fp32 all-reduce reference shard.
TIGHT-TIMEOUT friendly: a cross-rank ring/flag bug = 8-GPU hang, so always run under
`timeout` with a short budget."""
import argparse, sys
sys.path.insert(0, '/sgl-workspace/DeepGEMM')
import torch
import torch.distributed as dist
import deep_gemm
from deep_gemm.utils.dist import init_dist, dist_print


def calc_diff(x, y):
    x, y = x.double(), y.double()
    d = (x * x + y * y).sum()
    return (1 - 2 * (x * y).sum() / d).item()


def build_attn_tp_subgroups(world, tp):
    groups = []
    for base in range(0, world, tp):
        ranks = list(range(base, base + tp))
        groups.append((ranks, dist.new_group(ranks)))
    return groups


def test(local_rank, nlocal, args):
    gr, world, _ = init_dist(local_rank, nlocal)
    tp = args.attn_tp_size
    groups = build_attn_tp_subgroups(world, tp)
    my = next((rk, g) for rk, g in groups if gr in rk)
    ranks, grp = my
    rank_in_group = grp.rank()
    n, k = args.n, args.k
    tokens = [int(x) for x in args.tokens.split(',')]

    for m in tokens:
        m_per_rank = m // tp
        rs = deep_gemm.ReduceScatterRingFusedBuffer(grp, m, n)
        for it in range(args.num_tests):
            a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda') * 0.1
            b = torch.randn((n, k), dtype=torch.bfloat16, device='cuda') * 0.1
            full = (a.float() @ b.float().T)
            dist.all_reduce(full, op=dist.ReduceOp.SUM, group=grp)
            ref = full[rank_in_group * m_per_rank:(rank_in_group + 1) * m_per_rank]

            out = deep_gemm.bf16_gemm_reduce_scatter_ring_fused(a, b, rs)
            diff = calc_diff(out.float(), ref)
            ok = diff < 1e-2
            dist_print(f'[fused-ring M={m} it={it}] gr={gr} in_grp={rank_in_group} '
                       f'diff={diff:.3e} {"OK" if ok else "FAILED"}')
            assert ok, f'rank {gr}: fused-ring mismatch M={m} diff={diff:.3e}'
        rs.destroy()
    dist.barrier()
    if gr == 0:
        print('All FUSED-ring reduce-scatter correctness tests passed!', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num-processes', type=int, default=8)
    p.add_argument('--attn-tp-size', type=int, default=4)
    p.add_argument('--tokens', type=str, default='2048')
    p.add_argument('--n', type=int, default=4096)
    p.add_argument('--k', type=int, default=2048)
    p.add_argument('--num-tests', type=int, default=2)
    args = p.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)

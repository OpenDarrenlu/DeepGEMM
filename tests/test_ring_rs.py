"""Correctness test for the standalone PUSH-RING reduce-scatter comm kernel.
Compares bf16_gemm_reduce_scatter_ring vs the fp32 all-reduce reference.
Tight-timeout friendly (ring deadlock = hang)."""
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
    m, n, k = args.m, args.n, args.k
    m_per_rank = m // tp

    rs = deep_gemm.ReduceScatterRingBuffer(grp, m, n)
    for it in range(args.num_tests):
        a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda') * 0.1
        b = torch.randn((n, k), dtype=torch.bfloat16, device='cuda') * 0.1
        # reference: subgroup all-reduce of local partial, then this rank's shard
        full = (a.float() @ b.float().T)
        dist.all_reduce(full, op=dist.ReduceOp.SUM, group=grp)
        ref = full[rank_in_group * m_per_rank:(rank_in_group + 1) * m_per_rank]

        out = deep_gemm.bf16_gemm_reduce_scatter_ring(a, b, rs)
        diff = calc_diff(out.float(), ref)
        ok = diff < 1e-2
        dist_print(f'[ring test {it}] gr={gr} in_grp={rank_in_group} diff={diff:.3e} {"OK" if ok else "FAILED"}')
        assert ok, f'rank {gr}: ring mismatch diff={diff:.3e}'
    dist.barrier()
    if gr == 0:
        print('All ring reduce-scatter correctness tests passed!', flush=True)
    rs.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num-processes', type=int, default=8)
    p.add_argument('--attn-tp-size', type=int, default=4)
    p.add_argument('--m', type=int, default=4096)
    p.add_argument('--n', type=int, default=4096)
    p.add_argument('--k', type=int, default=2048)
    p.add_argument('--num-tests', type=int, default=2)
    args = p.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)

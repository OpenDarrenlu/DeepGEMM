"""Bench the standalone comm kernels at M=8192: RING vs push+combine vs NCCL RS.
All operate on a pre-computed local partial (GEMM excluded) — pure comm comparison."""
import argparse, sys
sys.path.insert(0, '/sgl-workspace/DeepGEMM')
import torch
import torch.distributed as dist
import deep_gemm
from deep_gemm import _C
from deep_gemm.utils.dist import init_dist


def build_grp(world, tp):
    return next(g for rk, g in
                [(list(range(b, b+tp)), dist.new_group(list(range(b, b+tp)))) for b in range(0, world, tp)]
                if dist.get_rank() in rk)


def bench(fn, group, warmup, iters):
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda'); cache.zero_()
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); dist.barrier(group=group)
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    t = torch.tensor([s.elapsed_time(e)/iters], device='cuda')
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return t.item()*1e3


def main(local_rank, nlocal, args):
    gr, world, _ = init_dist(local_rank, nlocal)
    grp = build_grp(world, args.attn_tp_size)
    m, n, tp = args.m, args.n, args.attn_tp_size
    mpr = m // tp
    partial = torch.randn((m, n), dtype=torch.bfloat16, device='cuda') * 0.1

    # RING comm
    ring = deep_gemm.ReduceScatterRingBuffer(grp, m, n)
    def ring_call(): _C.reduce_scatter_ring(partial, ring.buffer, ring.buffer_ptrs, ring.rank)
    # NCCL RS
    shard = torch.empty((mpr, n), dtype=torch.bfloat16, device='cuda')
    def nccl_call(): dist.reduce_scatter_tensor(shard, partial, op=dist.ReduceOp.SUM, group=grp)

    ring_call(); nccl_call()
    tr = bench(ring_call, grp, args.warmup, args.iters)
    tn = bench(nccl_call, grp, args.warmup, args.iters)
    if gr == 0:
        print(f'\n===== comm-only M={m} (per-rank {mpr}), n={n} =====', flush=True)
        print(f'  RING          = {tr:8.1f} us', flush=True)
        print(f'  NCCL RS       = {tn:8.1f} us', flush=True)
        print(f'  ring/nccl = {tr/tn:.2f}x', flush=True)
    ring.destroy()
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num-processes', type=int, default=8)
    p.add_argument('--attn-tp-size', type=int, default=4)
    p.add_argument('--m', type=int, default=8192)
    p.add_argument('--n', type=int, default=4096)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iters', type=int, default=60)
    args = p.parse_args()
    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes)

"""Bench the FULLY FUSED push-ring GEMM+RS (single kernel) vs:
  - sglang baseline: deep_gemm.bf16_gemm_nt + NCCL reduce_scatter_tensor (2 ops, no overlap)
  - split-ring: deep_gemm.bf16_gemm_nt + standalone ring comm (2 kernels, no overlap)
8-GPU attn_tp=4, per-attention-TP-group. The fused kernel's win is GEMM/comm overlap.
Always run under `timeout` (a ring bug = hang)."""
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
    s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    t = torch.tensor([s.elapsed_time(e)/iters], device='cuda')
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group); return t.item()*1e3


def main(local_rank, nlocal, args):
    gr, world, _ = init_dist(local_rank, nlocal)
    grp = build_grp(world, args.attn_tp_size)
    n, k, tp = args.n, args.k, args.attn_tp_size
    tokens = [int(x) for x in args.tokens.split(',')]
    if gr == 0:
        print(f'\n=== FUSED push-ring vs sglang(GEMM+NCCL) vs split-ring | n={n} k={k} tp={tp} ===', flush=True)
        print(f'{"M":>7} {"fused(us)":>10} {"sglang(us)":>11} {"split(us)":>10} '
              f'{"fս/sg":>7} {"fս/spl":>7}', flush=True)
    for m in tokens:
        mpr = m // tp
        a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda') * 0.1
        b = torch.randn((n, k), dtype=torch.bfloat16, device='cuda') * 0.1

        # FUSED push-ring (single kernel)
        fused = deep_gemm.ReduceScatterRingFusedBuffer(grp, m, n)
        def fused_call(): deep_gemm.bf16_gemm_reduce_scatter_ring_fused(a, b, fused)

        # sglang baseline: GEMM + NCCL reduce_scatter
        partial = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        shard = torch.empty((mpr, n), dtype=torch.bfloat16, device='cuda')
        def sglang_call():
            deep_gemm.bf16_gemm_nt(a, b, partial)
            dist.reduce_scatter_tensor(shard, partial, op=dist.ReduceOp.SUM, group=grp)

        # split-ring: GEMM + standalone ring comm (2 kernels)
        ring = deep_gemm.ReduceScatterRingBuffer(grp, m, n)
        scratch = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        def split_call():
            deep_gemm.bf16_gemm_nt(a, b, scratch)
            _C.reduce_scatter_ring(scratch, ring.buffer, ring.buffer_ptrs, ring.rank)

        fused_call(); sglang_call(); split_call()
        tf = bench(fused_call, grp, args.warmup, args.iters)
        ts = bench(sglang_call, grp, args.warmup, args.iters)
        tp_ = bench(split_call, grp, args.warmup, args.iters)
        if gr == 0:
            print(f'{m:>7} {tf:>10.1f} {ts:>11.1f} {tp_:>10.1f} {ts/tf:>6.2f}x {tp_/tf:>6.2f}x', flush=True)
        fused.destroy(); ring.destroy(); del a, b, partial, shard, scratch
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num-processes', type=int, default=8)
    p.add_argument('--attn-tp-size', type=int, default=4)
    p.add_argument('--tokens', type=str, default='512,1024,2048,4096,8192')
    p.add_argument('--n', type=int, default=4096)
    p.add_argument('--k', type=int, default=2048)
    p.add_argument('--warmup', type=int, default=15)
    p.add_argument('--iters', type=int, default=50)
    args = p.parse_args()
    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes)

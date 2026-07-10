import torch
from typing import Optional

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load DeepGEMM comm kernels, please check your PyTorch version: {exception}')

from .. import _C


class ReduceScatterBuffer:
    """Symmetric buffer for the fused BF16 GEMM + ReduceScatter kernel.

    Layout (single symmetric allocation, shared across ranks over NVLink):
        [ barrier region (32 B) ]
        [ FP32 output : m_per_rank x n ]
        [ BF16 scratch: num_ranks x m_per_rank x n ]   (num_ranks slots)

    Semantics: each rank writes its BF16 partial contribution into the owner rank's
    scratch SLOT `rank` with a plain remote store (no atomic); the owner then sums its
    `num_ranks` slots in FP32 into the output region. Owner-side FP32 accumulation gives
    better accuracy than a BF16 NCCL reduce-scatter.
    """

    # Must match `layout::Workspace::kNumBarrierSignalBytes`
    BARRIER_BYTES = 32

    def __init__(self, group: 'dist.ProcessGroup', m: int, n: int):
        assert m % group.size() == 0, 'M must be divisible by world size'
        self.group = group
        self.world_size = group.size()
        self.rank = group.rank()
        self.m = m
        self.n = n
        self.m_per_rank = m // self.world_size

        shard_elems = self.m_per_rank * n
        num_out_bytes = shard_elems * torch.float32.itemsize
        num_scratch_bytes = self.world_size * shard_elems * torch.bfloat16.itemsize
        num_bytes = self.BARRIER_BYTES + num_out_bytes + num_scratch_bytes
        self._out_bytes = num_out_bytes

        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

    @property
    def buffer_ptrs(self):
        # Per-rank base pointers into the symmetric allocation
        return self.handle.buffer_ptrs

    @property
    def output(self) -> torch.Tensor:
        # FP32 [m_per_rank, n] view of the output region
        data = self.buffer[self.BARRIER_BYTES:self.BARRIER_BYTES + self._out_bytes]
        return data.view(torch.float32).view(self.m_per_rank, self.n)

    def zero_(self):
        # Clear barrier + scratch (output is fully overwritten by the combine pass).
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None


def bf16_gemm_reduce_scatter(a: torch.Tensor,
                             b: torch.Tensor,
                             rs_buffer: ReduceScatterBuffer,
                             compiled_dims: str = 'nk') -> torch.Tensor:
    """Fused BF16 GEMM (`a @ b.T`) + ReduceScatter over single-node NVLink (single kernel).

    `a`: `[M, K]` bf16, `b`: `[N, K]` bf16 (NT layout). Every rank passes its own partial
    `a`/`b`; the output is reduced (summed) across ranks and scattered along M. Returns
    this rank's `[M // world_size, N]` FP32 shard. BF16 over NVLink + plain stores (no
    remote atomics) + owner-side FP32 combine. Fastest at small/medium M (decode).
    """
    # Fresh accumulation target every call
    rs_buffer.zero_()
    m, _ = a.shape
    n, _ = b.shape
    # Local BF16 [M, N] scratch for the phase-1 GEMM output before the push
    local_scratch = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    _C.bf16_gemm_reduce_scatter_nt(
        a, b, local_scratch, rs_buffer.buffer, rs_buffer.buffer_ptrs, rs_buffer.rank, compiled_dims)
    # Make peers' pushes visible (the kernel's NVLink barrier already syncs, but
    # a host-side barrier keeps the Python-level contract simple)
    rs_buffer.group.barrier()
    return rs_buffer.output


class ReduceScatterBufferPull:
    """Symmetric buffer for the PULL-based split reduce-scatter (fastest at large M).

    Layout: [ barrier (32 B) ][ BF16 output m_per_rank*n ][ BF16 partial shape_m*n ]

    The GEMM writes this rank's full [m, n] BF16 partial into the `partial` region
    (symmetric), then the pull comm kernel reads all peers' partials for this rank's
    shard rows, sums in FP32, and writes the output region.
    """

    BARRIER_BYTES = 32

    def __init__(self, group: 'dist.ProcessGroup', m: int, n: int):
        assert m % group.size() == 0, 'M must be divisible by world size'
        self.group = group
        self.world_size = group.size()
        self.rank = group.rank()
        self.m = m
        self.n = n
        self.m_per_rank = m // self.world_size

        shard_elems = self.m_per_rank * n
        self._out_bytes = shard_elems * torch.bfloat16.itemsize
        num_partial_bytes = m * n * torch.bfloat16.itemsize
        num_bytes = self.BARRIER_BYTES + self._out_bytes + num_partial_bytes

        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

    @property
    def buffer_ptrs(self):
        return self.handle.buffer_ptrs

    @property
    def output(self) -> torch.Tensor:
        data = self.buffer[self.BARRIER_BYTES:self.BARRIER_BYTES + self._out_bytes]
        return data.view(torch.bfloat16).view(self.m_per_rank, self.n)

    @property
    def partial(self) -> torch.Tensor:
        # This rank's full [m, n] BF16 GEMM output region (symmetric)
        start = self.BARRIER_BYTES + self._out_bytes
        data = self.buffer[start:]
        return data.view(torch.bfloat16).view(self.m, self.n)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None


def bf16_gemm_reduce_scatter_pull(a: torch.Tensor,
                                  b: torch.Tensor,
                                  rs_buffer: ReduceScatterBufferPull,
                                  compiled_dims: str = 'nk') -> torch.Tensor:
    """PULL-based split RS: plain BF16 GEMM into the symmetric partial region, then a
    pull comm kernel (barrier + read-all-peers + fp32 reduce + write shard).

    Returns this rank's `[M//world_size, N]` BF16 shard (fp32 accumulation internally,
    so accuracy exceeds sglang's bf16 NCCL reduce-scatter). No local scratch round-trip.
    Fastest at large M (prefill).
    """
    from .. import bf16_gemm_nt
    m, _ = a.shape
    n, _ = b.shape
    bf16_gemm_nt(a, b, rs_buffer.partial)
    _C.reduce_scatter_comm_pull(rs_buffer.buffer, rs_buffer.buffer_ptrs, rs_buffer.rank, m, n)
    rs_buffer.group.barrier()
    return rs_buffer.output


# Crossover between the two fused reduce-scatter kernels (per attention-TP-group M).
# Below this, the fused push+combine kernel wins; at/above, pull (read-all-peers) wins.
# Measured on SM100 (M504), n=4096 k=2048, 4-rank attn-TP group.
RS_PULL_CROSSOVER_M = 2048


class ReduceScatterBufferAuto:
    """Holds both a fused buffer (small M) and a pull buffer (large M); dispatches by M.

    Use when M varies at runtime (decode vs prefill). Allocates both symmetric buffers
    once (sized for the max M). For a fixed M, prefer the specific kernel directly.
    """

    def __init__(self, group: 'dist.ProcessGroup', m: int, n: int):
        self.group = group
        self.m, self.n = m, n
        self._fused = ReduceScatterBuffer(group, m, n)
        self._pull = ReduceScatterBufferPull(group, m, n)

    def destroy(self):
        self._fused.destroy()
        self._pull.destroy()


def bf16_gemm_reduce_scatter_auto(a: torch.Tensor,
                                  b: torch.Tensor,
                                  rs_buffer: ReduceScatterBufferAuto,
                                  compiled_dims: str = 'nk') -> torch.Tensor:
    """Dispatch to the faster RS kernel by M: the fused push+combine kernel for small M
    (decode), pull for large M (prefill). Returns this rank's `[M//world_size, N]` shard
    (fp32 for the fused kernel, bf16 for pull)."""
    m = a.shape[0]
    if m < RS_PULL_CROSSOVER_M:
        return bf16_gemm_reduce_scatter(a, b, rs_buffer._fused, compiled_dims)
    return bf16_gemm_reduce_scatter_pull(a, b, rs_buffer._pull, compiled_dims)


def bf16_gemm_reduce_scatter_split(a: torch.Tensor,
                                   b: torch.Tensor,
                                   rs_buffer: ReduceScatterBuffer,
                                   compiled_dims: str = 'nk') -> torch.Tensor:
    """Split-kernel RS: plain BF16 GEMM -> bf16 scratch, then a standalone lightweight
    comm kernel (push + barrier + local FP32 combine) that runs at full occupancy.

    Uses the same buffer layout as `ReduceScatterBuffer`. Returns this rank's
    `[M//world_size, N]` FP32 shard.
    """
    from .. import bf16_gemm_nt
    rs_buffer.zero_()
    m, _ = a.shape
    n, _ = b.shape
    local_scratch = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    bf16_gemm_nt(a, b, local_scratch)
    _C.reduce_scatter_comm(local_scratch, rs_buffer.buffer, rs_buffer.buffer_ptrs, rs_buffer.rank)
    rs_buffer.group.barrier()
    return rs_buffer.output

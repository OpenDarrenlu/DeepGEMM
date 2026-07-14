// Ground-truth probe: can a TMA bulk-tensor store (cp.async.bulk.tensor.2d.global.shared)
// write into a PEER GPU's memory over NVLink/P2P on SM100?
//
// Single process, 2 GPUs. Enable P2P. Build a CUtensorMap whose global address is GPU1's
// buffer, run a kernel on GPU0 that fills smem and TMA-stores to that map, then read GPU1
// back on the host and check.
//
// Build: nvcc -arch=sm_100a -o /tmp/tma_peer_probe tests/tma_peer_probe.cu -lcuda
// Run:   /tmp/tma_peer_probe

#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 1; } } while(0)
#define DCHECK(x) do { CUresult e = (x); if (e != CUDA_SUCCESS) { const char* s; cuGetErrorString(e,&s); \
    printf("CU driver error %s at %s:%d\n", s, __FILE__, __LINE__); return 1; } } while(0)

// 2D TMA store: smem tile [BM, BN] fp32 -> global via tensor map at coord (0,0)
__global__ void tma_store_kernel(const __grid_constant__ CUtensorMap tmap) {
    // shared mem tile, 128B-aligned for TMA
    extern __shared__ __align__(128) float smem[];
    constexpr int BM = 8, BN = 32;         // 8 rows x 32 cols fp32 = 1KB, inner 128B
    int tid = threadIdx.x;
    // fill smem with a recognizable pattern: value = row*1000 + col
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int r = i / BN, c = i % BN;
        smem[i] = float(r * 1000 + c);
    }
    __syncthreads();
    // fence smem before TMA reads it
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();

    if (tid == 0) {
        uint64_t smem_int = static_cast<uint64_t>(__cvta_generic_to_shared(smem));
        // cp.async.bulk.tensor.2d.global.shared: dst = tensormap at {coord_n=0, coord_m=0}
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
            :: "l"(&tmap), "l"(smem_int), "r"(0), "r"(0) : "memory");
        asm volatile("cp.async.bulk.commit_group;");
        asm volatile("cp.async.bulk.wait_group 0;");
    }
    __syncthreads();
}

int main() {
    int ndev = 0; CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < 2) { printf("SKIP: need >=2 GPUs, have %d\n", ndev); return 0; }

    int canAccess01 = 0, canAccess10 = 0;
    CHECK(cudaDeviceCanAccessPeer(&canAccess01, 0, 1));
    CHECK(cudaDeviceCanAccessPeer(&canAccess10, 1, 0));
    printf("P2P 0->1=%d 1->0=%d\n", canAccess01, canAccess10);
    if (!canAccess01) { printf("SKIP: no P2P 0->1\n"); return 0; }

    constexpr int BM = 8, BN = 32;
    constexpr int M = 8, N = 32;                 // global buffer same size as one tile
    size_t bytes = size_t(M) * N * sizeof(float);

    // Allocate on GPU1 (the peer / destination)
    CHECK(cudaSetDevice(1));
    float* dst_gpu1 = nullptr;
    CHECK(cudaMalloc(&dst_gpu1, bytes));
    CHECK(cudaMemset(dst_gpu1, 0, bytes));

    // GPU0 enables access to GPU1
    CHECK(cudaSetDevice(0));
    cudaError_t pe = cudaDeviceEnablePeerAccess(1, 0);
    if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled) {
        printf("enablePeerAccess failed: %s\n", cudaGetErrorString(pe)); return 1;
    }

    // Build a tensor map on GPU0 whose GLOBAL ADDRESS is the peer (GPU1) pointer.
    CUtensorMap tmap;
    uint64_t gdims[2]   = { (uint64_t)N, (uint64_t)M };          // inner=N, outer=M
    uint64_t gstride[1] = { (uint64_t)N * sizeof(float) };       // row stride bytes
    uint32_t sdims[2]   = { (uint32_t)BN, (uint32_t)BM };
    uint32_t estride[2] = { 1, 1 };
    DCHECK(cuTensorMapEncodeTiled(
        &tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2,
        (void*)dst_gpu1,                 // <-- PEER pointer as TMA global base
        gdims, gstride, sdims, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    // Launch on GPU0
    CHECK(cudaSetDevice(0));
    tma_store_kernel<<<1, 256, BM * BN * sizeof(float)>>>(tmap);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Read GPU1 back
    float* host = (float*)malloc(bytes);
    CHECK(cudaSetDevice(1));
    CHECK(cudaMemcpy(host, dst_gpu1, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) {
            float exp = float(r * 1000 + c);
            float got = host[r * N + c];
            if (got != exp) {
                if (errors < 8) printf("  mismatch [%d,%d] exp %.0f got %.0f\n", r, c, exp, got);
                errors++;
            }
        }
    printf("sample got[0..4] = %.0f %.0f %.0f %.0f\n", host[0], host[1], host[2], host[3]);
    printf("sample got[row1] = %.0f (exp 1000)\n", host[N]);
    if (errors == 0) printf("RESULT: PASS — TMA store into peer memory WORKS\n");
    else             printf("RESULT: FAIL — %d mismatches\n", errors);
    free(host);
    return errors == 0 ? 0 : 2;
}

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "config.h"

namespace nanochat {

// RMSNorm: x / sqrt(mean(x^2) + eps)
// Each block handles one row (token)
__global__ void rmsnorm_kernel(nv_bfloat16* out, const nv_bfloat16* x, int n) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    const nv_bfloat16* x_row = x + row * n;
    nv_bfloat16* out_row = out + row * n;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = __bfloat162float(x_row[i]);
        sum_sq += val * val;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Store warp results to shared memory
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) {
        shared[warp_id] = sum_sq;
    }
    __syncthreads();

    // Final reduction in first warp
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (warp_id == 0) {
        sum_sq = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    // Broadcast RMS to all threads
    __shared__ float rms;
    if (threadIdx.x == 0) {
        rms = rsqrtf(sum_sq / n + 1e-6f);
    }
    __syncthreads();

    // Normalize
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = __bfloat162float(x_row[i]);
        out_row[i] = __float2bfloat16(val * rms);
    }
}

void rmsnorm(nv_bfloat16* out, const nv_bfloat16* x, int rows, int cols, cudaStream_t stream) {
    int threads = 256;
    int shared_mem = (threads / 32) * sizeof(float);
    rmsnorm_kernel<<<rows, threads, shared_mem, stream>>>(out, x, cols);
}

}  // namespace nanochat

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <vector>
#include "config.h"

namespace nanochat {

// Precomputed cos/sin tables stored in constant memory
__device__ float d_cos_table[ROPE_MAX_SEQ * HEAD_DIM / 2];
__device__ float d_sin_table[ROPE_MAX_SEQ * HEAD_DIM / 2];

// Initialize RoPE tables on host, copy to device
void init_rope_tables() {
    int half_dim = HEAD_DIM / 2;
    std::vector<float> cos_table(ROPE_MAX_SEQ * half_dim);
    std::vector<float> sin_table(ROPE_MAX_SEQ * half_dim);

    for (int pos = 0; pos < ROPE_MAX_SEQ; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / powf(ROPE_BASE, (2.0f * i) / HEAD_DIM);
            float angle = pos * freq;
            cos_table[pos * half_dim + i] = cosf(angle);
            sin_table[pos * half_dim + i] = sinf(angle);
        }
    }

    cudaMemcpyToSymbol(d_cos_table, cos_table.data(), cos_table.size() * sizeof(float));
    cudaMemcpyToSymbol(d_sin_table, sin_table.data(), sin_table.size() * sizeof(float));
}

// Apply RoPE to Q or K tensor
// Shape: [seq_len, n_heads, head_dim]
__global__ void rope_kernel(nv_bfloat16* x, int seq_len, int n_heads, int start_pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = HEAD_DIM / 2;
    int total = seq_len * n_heads * half_dim;

    if (idx >= total) return;

    int pair_idx = idx % half_dim;
    int head = (idx / half_dim) % n_heads;
    int pos = idx / (half_dim * n_heads);
    int actual_pos = start_pos + pos;

    // Get the pair of elements to rotate
    int base_idx = pos * n_heads * HEAD_DIM + head * HEAD_DIM;
    int i0 = base_idx + pair_idx;
    int i1 = base_idx + pair_idx + half_dim;

    float x0 = __bfloat162float(x[i0]);
    float x1 = __bfloat162float(x[i1]);

    float cos_val = d_cos_table[actual_pos * half_dim + pair_idx];
    float sin_val = d_sin_table[actual_pos * half_dim + pair_idx];

    // Rotation matching Python: y1 = x1*cos + x2*sin, y2 = -x1*sin + x2*cos
    x[i0] = __float2bfloat16(x0 * cos_val + x1 * sin_val);
    x[i1] = __float2bfloat16(-x0 * sin_val + x1 * cos_val);
}

void apply_rope(nv_bfloat16* x, int seq_len, int n_heads, int start_pos, cudaStream_t stream) {
    int half_dim = HEAD_DIM / 2;
    int total = seq_len * n_heads * half_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rope_kernel<<<blocks, threads, 0, stream>>>(x, seq_len, n_heads, start_pos);
}

// Fused RoPE + RMSNorm kernel
// Each block processes one (pos, head) pair = HEAD_DIM elements
// Applies RoPE rotation, then normalizes the result
// Uses shared memory to avoid redundant global memory access
__global__ void rope_rmsnorm_kernel(nv_bfloat16* x, int seq_len, int n_heads, int start_pos) {
    // Block ID maps to (pos, head)
    int block_id = blockIdx.x;
    int total_heads = seq_len * n_heads;
    if (block_id >= total_heads) return;

    int pos = block_id / n_heads;
    int head = block_id % n_heads;
    int actual_pos = start_pos + pos;
    constexpr int half_dim = HEAD_DIM / 2;  // 64

    // Base address for this head's data
    int base_idx = pos * n_heads * HEAD_DIM + head * HEAD_DIM;

    // Shared memory for rotated values and reduction
    __shared__ float rotated[HEAD_DIM];  // Store rotated values
    __shared__ float reduction[8];        // For warp reduction

    float sum_sq = 0.0f;

    // Phase 1: Load, apply RoPE, store to shared mem, accumulate sum_sq
    for (int pair = threadIdx.x; pair < half_dim; pair += blockDim.x) {
        int i0 = base_idx + pair;
        int i1 = base_idx + pair + half_dim;

        float x0 = __bfloat162float(x[i0]);
        float x1 = __bfloat162float(x[i1]);

        float cos_val = d_cos_table[actual_pos * half_dim + pair];
        float sin_val = d_sin_table[actual_pos * half_dim + pair];

        // Apply RoPE rotation
        float y0 = x0 * cos_val + x1 * sin_val;
        float y1 = -x0 * sin_val + x1 * cos_val;

        // Store to shared memory (avoid global write then read)
        rotated[pair] = y0;
        rotated[pair + half_dim] = y1;

        sum_sq += y0 * y0 + y1 * y1;
    }

    // Warp reduction for sum_sq
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    if (lane_id == 0) {
        reduction[warp_id] = sum_sq;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0) {
        sum_sq = (lane_id < num_warps) ? reduction[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
        if (lane_id == 0) {
            reduction[0] = rsqrtf(sum_sq / HEAD_DIM + 1e-6f);
        }
    }
    __syncthreads();

    float rms = reduction[0];

    // Phase 2: Normalize and write to global memory
    for (int pair = threadIdx.x; pair < half_dim; pair += blockDim.x) {
        int i0 = base_idx + pair;
        int i1 = base_idx + pair + half_dim;

        x[i0] = __float2bfloat16(rotated[pair] * rms);
        x[i1] = __float2bfloat16(rotated[pair + half_dim] * rms);
    }
}

void apply_rope_rmsnorm(nv_bfloat16* x, int seq_len, int n_heads, int start_pos, cudaStream_t stream) {
    int total_heads = seq_len * n_heads;
    int threads = 64;  // 64 threads for 64 pairs (HEAD_DIM/2)
    rope_rmsnorm_kernel<<<total_heads, threads, 0, stream>>>(x, seq_len, n_heads, start_pos);
}

}  // namespace nanochat

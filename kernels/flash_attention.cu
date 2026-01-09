#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>
#include "config.h"

namespace nanochat {

// Flash Attention: Fused Q@K^T + causal mask + softmax + @V
// Multi-warp version for better parallelism
//
// Each block handles one (query_pos, head) pair.
// Multiple warps process different KV ranges in parallel.
// Uses online softmax with proper cross-warp combination.

constexpr int THREADS_PER_BLOCK = 128;  // 4 warps
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;
constexpr int DIMS_PER_THREAD = HEAD_DIM / 32;  // 4 dims per thread within a warp

__global__ void flash_attention_kernel(
    nv_bfloat16* __restrict__ out,
    const nv_bfloat16* __restrict__ Q,
    const nv_bfloat16* __restrict__ K,
    const nv_bfloat16* __restrict__ V,
    int seq_q, int kv_len, int heads,
    float scale, int kv_offset
) {
    int block_id = blockIdx.x;
    int query_pos = block_id / heads;
    int head = block_id % heads;

    if (query_pos >= seq_q) return;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Causal mask limit
    int max_kv_pos = query_pos + kv_offset;

    // Shared memory
    __shared__ float s_q[HEAD_DIM];
    __shared__ float s_warp_max[WARPS_PER_BLOCK];
    __shared__ float s_warp_sum[WARPS_PER_BLOCK];
    __shared__ float s_warp_out[WARPS_PER_BLOCK][HEAD_DIM];
    __shared__ float s_global_max;
    __shared__ float s_global_sum;

    // Initialize shared memory to safe defaults (handles warps with no valid KV positions)
    if (threadIdx.x < WARPS_PER_BLOCK) {
        s_warp_max[threadIdx.x] = -FLT_MAX;
        s_warp_sum[threadIdx.x] = 0.0f;
    }
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            s_warp_out[w][d] = 0.0f;
        }
    }
    __syncthreads();

    // Load Q into shared memory
    int q_base = query_pos * heads * HEAD_DIM + head * HEAD_DIM;
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        s_q[d] = __bfloat162float(Q[q_base + d]);
    }
    __syncthreads();

    // Per-warp state
    float out_acc[DIMS_PER_THREAD] = {0.0f};
    float warp_max = -FLT_MAX;
    float warp_sum = 0.0f;

    // Divide KV positions among warps
    // Each warp processes every WARPS_PER_BLOCK-th position (interleaved for better load balance)
    for (int kv_pos = warp_id; kv_pos < kv_len; kv_pos += WARPS_PER_BLOCK) {
        if (kv_pos > max_kv_pos) continue;

        // Compute Q . K[kv_pos]
        int k_base = kv_pos * heads * HEAD_DIM + head * HEAD_DIM;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < DIMS_PER_THREAD; i++) {
            int d = lane_id * DIMS_PER_THREAD + i;
            dot += s_q[d] * __bfloat162float(K[k_base + d]);
        }

        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        float score = __shfl_sync(0xffffffff, dot, 0) * scale;

        // Online softmax update
        float new_max = fmaxf(warp_max, score);
        float exp_diff = expf(warp_max - new_max);
        float p = expf(score - new_max);

        warp_sum = warp_sum * exp_diff + p;
        #pragma unroll
        for (int i = 0; i < DIMS_PER_THREAD; i++) {
            out_acc[i] *= exp_diff;
        }
        warp_max = new_max;

        // Accumulate p * V[kv_pos]
        int v_base = kv_pos * heads * HEAD_DIM + head * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < DIMS_PER_THREAD; i++) {
            int d = lane_id * DIMS_PER_THREAD + i;
            out_acc[i] += p * __bfloat162float(V[v_base + d]);
        }
    }

    // Store per-warp results
    if (lane_id == 0) {
        s_warp_max[warp_id] = warp_max;
        s_warp_sum[warp_id] = warp_sum;
    }
    // Store per-warp output (each lane stores its dimensions)
    #pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; i++) {
        int d = lane_id * DIMS_PER_THREAD + i;
        s_warp_out[warp_id][d] = out_acc[i];
    }
    __syncthreads();

    // Thread 0 combines results from all warps
    if (threadIdx.x == 0) {
        // Find global max
        float global_max = s_warp_max[0];
        for (int w = 1; w < WARPS_PER_BLOCK; w++) {
            global_max = fmaxf(global_max, s_warp_max[w]);
        }
        s_global_max = global_max;

        // Compute global sum with rescaling
        float global_sum = 0.0f;
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            global_sum += s_warp_sum[w] * expf(s_warp_max[w] - global_max);
        }
        s_global_sum = global_sum;
    }
    __syncthreads();

    float global_max = s_global_max;
    float global_sum = s_global_sum;

    // Combine outputs: each thread handles some dimensions
    int out_base = query_pos * heads * HEAD_DIM + head * HEAD_DIM;
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        float combined = 0.0f;
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            float rescale = expf(s_warp_max[w] - global_max);
            combined += s_warp_out[w][d] * rescale;
        }
        out[out_base + d] = __float2bfloat16(combined / global_sum);
    }
}

void flash_attention(
    nv_bfloat16* out,
    const nv_bfloat16* Q,
    const nv_bfloat16* K,
    const nv_bfloat16* V,
    int seq_q, int kv_len, int heads,
    float scale, int kv_offset,
    cudaStream_t stream
) {
    int num_blocks = seq_q * heads;
    flash_attention_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        out, Q, K, V, seq_q, kv_len, heads, scale, kv_offset);
}

}  // namespace nanochat

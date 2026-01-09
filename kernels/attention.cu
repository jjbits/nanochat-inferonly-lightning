#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>
#include "config.h"

namespace nanochat {

// bf16 causal mask for all heads at once: scores[heads, rows, cols]
__global__ void apply_causal_mask_bf16_kernel(nv_bfloat16* scores, int heads, int rows, int cols, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * rows * cols;
    if (idx >= total) return;

    int col = idx % cols;
    int row = (idx / cols) % rows;

    if (col > row + offset) {
        scores[idx] = __float2bfloat16(-1e9f);
    }
}

void apply_causal_mask_bf16(nv_bfloat16* scores, int heads, int rows, int cols, int offset, cudaStream_t stream) {
    int total = heads * rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    apply_causal_mask_bf16_kernel<<<blocks, threads, 0, stream>>>(scores, heads, rows, cols, offset);
}

// bf16 softmax: x[rows, cols] -> out[rows, cols], one block per row
__global__ void softmax_bf16_kernel(nv_bfloat16* out, const nv_bfloat16* x, int cols) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    const nv_bfloat16* x_row = x + row * cols;
    nv_bfloat16* out_row = out + row * cols;

    // Find max
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, __bfloat162float(x_row[i]));
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    if (threadIdx.x % warpSize == 0) shared[threadIdx.x / warpSize] = max_val;
    __syncthreads();
    if (threadIdx.x < warpSize) {
        max_val = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x] : -FLT_MAX;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        if (threadIdx.x == 0) shared[0] = max_val;
    }
    __syncthreads();
    max_val = shared[0];

    // Sum of exp
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum_exp += expf(__bfloat162float(x_row[i]) - max_val);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    if (threadIdx.x % warpSize == 0) shared[threadIdx.x / warpSize] = sum_exp;
    __syncthreads();
    if (threadIdx.x < warpSize) {
        sum_exp = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
        }
        if (threadIdx.x == 0) shared[0] = sum_exp;
    }
    __syncthreads();
    float inv_sum = 1.0f / shared[0];

    // Normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = expf(__bfloat162float(x_row[i]) - max_val) * inv_sum;
        out_row[i] = __float2bfloat16(val);
    }
}

void softmax_bf16(nv_bfloat16* out, const nv_bfloat16* x, int rows, int cols, cudaStream_t stream) {
    int threads = 256;
    int shared_mem = (threads / 32) * sizeof(float);
    softmax_bf16_kernel<<<rows, threads, shared_mem, stream>>>(out, x, cols);
}

// Fused causal mask + softmax kernel
// scores layout: [heads, seq_q, kv_len]
// Each block handles one row (one head's one query position)
// Causal condition: col > row_in_seq + offset is masked
__global__ void causal_softmax_bf16_kernel(
    nv_bfloat16* out,
    const nv_bfloat16* scores,
    int heads, int seq_q, int kv_len, int offset
) {
    extern __shared__ float shared[];

    int block_id = blockIdx.x;  // Linear index over [heads * seq_q]
    int row_in_seq = block_id % seq_q;

    const nv_bfloat16* scores_row = scores + block_id * kv_len;
    nv_bfloat16* out_row = out + block_id * kv_len;

    // Causal mask: positions col > row_in_seq + offset are masked
    int max_valid_col = row_in_seq + offset;

    // Find max (with causal masking applied inline)
    float max_val = -FLT_MAX;
    for (int col = threadIdx.x; col < kv_len; col += blockDim.x) {
        if (col <= max_valid_col) {
            max_val = fmaxf(max_val, __bfloat162float(scores_row[col]));
        }
    }
    for (int off = warpSize / 2; off > 0; off /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, off));
    }
    if (threadIdx.x % warpSize == 0) shared[threadIdx.x / warpSize] = max_val;
    __syncthreads();
    if (threadIdx.x < warpSize) {
        max_val = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x] : -FLT_MAX;
        for (int off = warpSize / 2; off > 0; off /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, off));
        }
        if (threadIdx.x == 0) shared[0] = max_val;
    }
    __syncthreads();
    max_val = shared[0];

    // Sum of exp (masked positions contribute 0)
    float sum_exp = 0.0f;
    for (int col = threadIdx.x; col < kv_len; col += blockDim.x) {
        if (col <= max_valid_col) {
            sum_exp += expf(__bfloat162float(scores_row[col]) - max_val);
        }
    }
    for (int off = warpSize / 2; off > 0; off /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, off);
    }
    if (threadIdx.x % warpSize == 0) shared[threadIdx.x / warpSize] = sum_exp;
    __syncthreads();
    if (threadIdx.x < warpSize) {
        sum_exp = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x] : 0.0f;
        for (int off = warpSize / 2; off > 0; off /= 2) {
            sum_exp += __shfl_down_sync(0xffffffff, sum_exp, off);
        }
        if (threadIdx.x == 0) shared[0] = sum_exp;
    }
    __syncthreads();
    float inv_sum = 1.0f / shared[0];

    // Normalize and output (masked positions get 0)
    for (int col = threadIdx.x; col < kv_len; col += blockDim.x) {
        float val = 0.0f;
        if (col <= max_valid_col) {
            val = expf(__bfloat162float(scores_row[col]) - max_val) * inv_sum;
        }
        out_row[col] = __float2bfloat16(val);
    }
}

void causal_softmax_bf16(
    nv_bfloat16* out, const nv_bfloat16* scores,
    int heads, int seq_q, int kv_len, int offset, cudaStream_t stream
) {
    int total_rows = heads * seq_q;
    int threads = 256;
    int shared_mem = (threads / 32) * sizeof(float);
    causal_softmax_bf16_kernel<<<total_rows, threads, shared_mem, stream>>>(
        out, scores, heads, seq_q, kv_len, offset);
}

}  // namespace nanochat

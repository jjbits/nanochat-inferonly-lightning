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

}  // namespace nanochat

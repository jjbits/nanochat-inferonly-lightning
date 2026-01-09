#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include "config.h"

namespace nanochat {

// Embedding lookup: token_ids -> embeddings
__global__ void embedding_kernel(nv_bfloat16* out, const nv_bfloat16* weight, const int* tokens, int n_embd) {
    int token_idx = blockIdx.x;
    int token_id = tokens[token_idx];

    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        out[token_idx * n_embd + i] = weight[token_id * n_embd + i];
    }
}

void embedding(nv_bfloat16* out, const nv_bfloat16* weight, const int* tokens, int seq_len, int n_embd, cudaStream_t stream) {
    embedding_kernel<<<seq_len, 256, 0, stream>>>(out, weight, tokens, n_embd);
}

// ReLU squared: max(0, x)^2
__global__ void relu_squared_kernel(nv_bfloat16* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __bfloat162float(x[idx]);
        val = fmaxf(0.0f, val);
        x[idx] = __float2bfloat16(val * val);
    }
}

void relu_squared(nv_bfloat16* x, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_squared_kernel<<<blocks, threads, 0, stream>>>(x, n);
}

// Tanh soft cap for logits: cap * tanh(x / cap)
__global__ void tanh_cap_kernel(nv_bfloat16* x, int n, float cap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __bfloat162float(x[idx]);
        val = cap * tanhf(val / cap);
        x[idx] = __float2bfloat16(val);
    }
}

void tanh_cap(nv_bfloat16* x, int n, float cap, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    tanh_cap_kernel<<<blocks, threads, 0, stream>>>(x, n, cap);
}

// Residual add: out = a + b
__global__ void residual_add_kernel(nv_bfloat16* out, const nv_bfloat16* a, const nv_bfloat16* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __bfloat162float(a[idx]);
        float vb = __bfloat162float(b[idx]);
        out[idx] = __float2bfloat16(va + vb);
    }
}

void residual_add(nv_bfloat16* out, const nv_bfloat16* a, const nv_bfloat16* b, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads, 0, stream>>>(out, a, b, n);
}

}  // namespace nanochat

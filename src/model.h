#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <string>
#include <vector>
#include "config.h"
#include "tensor.h"

namespace nanochat {

// Kernel function declarations
void rmsnorm(nv_bfloat16* out, const nv_bfloat16* x, int rows, int cols, cudaStream_t stream);
void apply_rope(nv_bfloat16* x, int seq_len, int n_heads, int start_pos, cudaStream_t stream);
void apply_rope_rmsnorm(nv_bfloat16* x, int seq_len, int n_heads, int start_pos, cudaStream_t stream);
void init_rope_tables();
void embedding(nv_bfloat16* out, const nv_bfloat16* weight, const int* tokens, int seq_len, int n_embd, cudaStream_t stream);
void relu_squared(nv_bfloat16* x, int n, cudaStream_t stream);
void tanh_cap(nv_bfloat16* x, int n, float cap, cudaStream_t stream);
void residual_add(nv_bfloat16* out, const nv_bfloat16* a, const nv_bfloat16* b, int n, cudaStream_t stream);
// CUTLASS GEMM functions
void gemm_half(nv_bfloat16* C, const nv_bfloat16* A, const nv_bfloat16* B, int M, int N, int K, cudaStream_t stream);
void gemm_half_residual(nv_bfloat16* D, const nv_bfloat16* A, const nv_bfloat16* B, const nv_bfloat16* residual, int M, int N, int K, cudaStream_t stream);

// Flash Attention: fused Q@K^T + causal mask + softmax + @V
void flash_attention(nv_bfloat16* out, const nv_bfloat16* Q, const nv_bfloat16* K, const nv_bfloat16* V,
                     int seq_q, int kv_len, int heads, float scale, int kv_offset, cudaStream_t stream);

// Single transformer layer weights (RMSNorm is parameter-free)
struct LayerWeights {
    Tensor<nv_bfloat16> q_weight;        // [n_embd, n_embd]
    Tensor<nv_bfloat16> k_weight;        // [n_embd, n_embd]
    Tensor<nv_bfloat16> v_weight;        // [n_embd, n_embd]
    Tensor<nv_bfloat16> o_weight;        // [n_embd, n_embd]
    Tensor<nv_bfloat16> mlp_up_weight;   // [mlp_hidden, n_embd]
    Tensor<nv_bfloat16> mlp_down_weight; // [n_embd, mlp_hidden]
};

// Full model weights
struct ModelWeights {
    Tensor<nv_bfloat16> embed_weight;    // [vocab_size, n_embd]
    Tensor<nv_bfloat16> lm_head;         // [vocab_size, n_embd]
    std::vector<LayerWeights> layers;
    void load(const std::string& path);
};

// Scratch buffers for forward pass
struct ScratchBuffers {
    Tensor<nv_bfloat16> x;               // [seq, n_embd]
    Tensor<nv_bfloat16> x_norm;          // [seq, n_embd]
    Tensor<nv_bfloat16> q;               // [seq, n_heads, head_dim]
    Tensor<nv_bfloat16> k;               // [seq, n_kv_heads, head_dim]
    Tensor<nv_bfloat16> v;               // [seq, n_kv_heads, head_dim]
    Tensor<nv_bfloat16> attn_out;        // [seq, n_embd]
    Tensor<nv_bfloat16> mlp_hidden;      // [seq, mlp_hidden]
    Tensor<nv_bfloat16> mlp_out;         // [seq, n_embd]
    Tensor<nv_bfloat16> scores;         // [n_heads, seq, kv_len]
    Tensor<nv_bfloat16> scores_softmax;
    Tensor<nv_bfloat16> logits;          // [seq, vocab_size]
    Tensor<int> tokens;                  // [seq] - pre-allocated for forward
    void allocate(int max_seq);
};

// KV cache for a single layer
struct KVCache {
    Tensor<nv_bfloat16> k_cache;         // [max_seq, n_kv_heads, head_dim]
    Tensor<nv_bfloat16> v_cache;         // [max_seq, n_kv_heads, head_dim]
    int current_len = 0;
    int max_len = 0;
    void allocate(int max_seq);
    void grow(int new_max);
    void clear() { current_len = 0; }
};

class Model {
public:
    ModelWeights weights;
    std::vector<KVCache> kv_caches;
    ScratchBuffers scratch;
    cudaStream_t stream;

    Model();
    ~Model();
    void load(const std::string& weights_path);
    void forward(const int* tokens, int seq_len, int start_pos, nv_bfloat16* logits_out);
    // Evaluation: returns logits for ALL positions as float [seq_len * vocab_size]
    void forward_all(const int* tokens, int seq_len, float* logits_out);

private:
    void attention(int layer, int seq_len, int start_pos);
    void mlp(int layer, int seq_len);
};

}  // namespace nanochat

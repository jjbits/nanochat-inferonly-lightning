#include "model.h"
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace nanochat {

template <typename T>
static void load_tensor(Tensor<T>& tensor, std::ifstream& file, size_t count) {
    std::vector<T> host_data(count);
    file.read(reinterpret_cast<char*>(host_data.data()), count * sizeof(T));
    if (!file) throw std::runtime_error("Failed to read tensor data");
    tensor.from_host(host_data.data(), count);
}

void ModelWeights::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open weights file: " + path);

    load_tensor(embed_weight, file, VOCAB_SIZE * N_EMBD);

    layers.resize(N_LAYER);
    for (int i = 0; i < N_LAYER; i++) {
        auto& l = layers[i];
        load_tensor(l.q_weight, file, N_EMBD * N_EMBD);
        load_tensor(l.k_weight, file, N_EMBD * N_EMBD);
        load_tensor(l.v_weight, file, N_EMBD * N_EMBD);
        load_tensor(l.o_weight, file, N_EMBD * N_EMBD);
        load_tensor(l.mlp_up_weight, file, MLP_HIDDEN * N_EMBD);
        load_tensor(l.mlp_down_weight, file, N_EMBD * MLP_HIDDEN);
    }

    load_tensor(lm_head, file, VOCAB_SIZE * N_EMBD);
}

void ScratchBuffers::allocate(int max_seq) {
    x.allocate(max_seq * N_EMBD);
    x_norm.allocate(max_seq * N_EMBD);
    q.allocate(max_seq * N_HEAD * HEAD_DIM);
    k.allocate(max_seq * N_KV_HEAD * HEAD_DIM);
    v.allocate(max_seq * N_KV_HEAD * HEAD_DIM);
    attn_out.allocate(max_seq * N_EMBD);
    mlp_hidden.allocate(max_seq * MLP_HIDDEN);
    mlp_out.allocate(max_seq * N_EMBD);
    scores.allocate(N_HEAD * max_seq * SEQ_LEN);
    scores_softmax.allocate(N_HEAD * max_seq * SEQ_LEN);
    logits.allocate(max_seq * VOCAB_SIZE);
}

void KVCache::allocate(int max_seq) {
    max_len = max_seq;
    k_cache.allocate(max_seq * N_KV_HEAD * HEAD_DIM);
    v_cache.allocate(max_seq * N_KV_HEAD * HEAD_DIM);
    k_cache.zero();
    v_cache.zero();
}

void KVCache::grow(int new_max) {
    if (new_max <= max_len) return;

    Tensor<nv_bfloat16> new_k, new_v;
    new_k.allocate(new_max * N_KV_HEAD * HEAD_DIM);
    new_v.allocate(new_max * N_KV_HEAD * HEAD_DIM);
    new_k.zero();
    new_v.zero();

    if (current_len > 0) {
        CUDA_CHECK(cudaMemcpy(new_k.data, k_cache.data,
                              current_len * N_KV_HEAD * HEAD_DIM * sizeof(nv_bfloat16),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(new_v.data, v_cache.data,
                              current_len * N_KV_HEAD * HEAD_DIM * sizeof(nv_bfloat16),
                              cudaMemcpyDeviceToDevice));
    }

    k_cache = std::move(new_k);
    v_cache = std::move(new_v);
    max_len = new_max;
}

Model::Model() {
    CUDA_CHECK(cudaStreamCreate(&stream));
    init_rope_tables();
}

Model::~Model() {
    cudaStreamDestroy(stream);
}

void Model::load(const std::string& weights_path) {
    weights.load(weights_path);
    kv_caches.resize(N_LAYER);
    for (auto& cache : kv_caches) {
        cache.allocate(KV_CACHE_GROW);
    }
    scratch.allocate(SEQ_LEN);
}

void Model::attention(int layer, int seq_len, int start_pos) {
    auto& l = weights.layers[layer];
    auto& cache = kv_caches[layer];
    int kv_len = start_pos + seq_len;

    if (kv_len > cache.max_len) {
        cache.grow(((kv_len + KV_CACHE_GROW - 1) / KV_CACHE_GROW) * KV_CACHE_GROW);
    }

    // Pre-attention RMSNorm
    rmsnorm(scratch.x_norm.data, scratch.x.data, seq_len, N_EMBD, stream);

    // QKV projections: x[seq,n_embd] @ W[n_embd,n_embd]^T = out[seq,n_embd]
    gemm_half(scratch.q.data, scratch.x_norm.data, l.q_weight.data, seq_len, N_EMBD, N_EMBD, stream);
    gemm_half(scratch.k.data, scratch.x_norm.data, l.k_weight.data, seq_len, N_EMBD, N_EMBD, stream);
    gemm_half(scratch.v.data, scratch.x_norm.data, l.v_weight.data, seq_len, N_EMBD, N_EMBD, stream);

    // Fused RoPE + QK-norm (2 kernels instead of 4)
    apply_rope_rmsnorm(scratch.q.data, seq_len, N_HEAD, start_pos, stream);
    apply_rope_rmsnorm(scratch.k.data, seq_len, N_KV_HEAD, start_pos, stream);

    // Append to KV cache
    nv_bfloat16* k_cache_pos = cache.k_cache.data + start_pos * N_KV_HEAD * HEAD_DIM;
    nv_bfloat16* v_cache_pos = cache.v_cache.data + start_pos * N_KV_HEAD * HEAD_DIM;
    CUDA_CHECK(cudaMemcpyAsync(k_cache_pos, scratch.k.data,
                                seq_len * N_KV_HEAD * HEAD_DIM * sizeof(nv_bfloat16),
                                cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(v_cache_pos, scratch.v.data,
                                seq_len * N_KV_HEAD * HEAD_DIM * sizeof(nv_bfloat16),
                                cudaMemcpyDeviceToDevice, stream));
    cache.current_len = kv_len;

    // Flash Attention: fused Q@K^T + causal mask + softmax + @V (single kernel)
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    flash_attention(scratch.attn_out.data, scratch.q.data, cache.k_cache.data, cache.v_cache.data,
                    seq_len, kv_len, N_HEAD, scale, start_pos, stream);

    // Output projection + residual (fused)
    gemm_half_residual(scratch.x.data, scratch.attn_out.data, l.o_weight.data, scratch.x.data, seq_len, N_EMBD, N_EMBD, stream);
}

void Model::mlp(int layer, int seq_len) {
    auto& l = weights.layers[layer];

    // Pre-MLP RMSNorm
    rmsnorm(scratch.x_norm.data, scratch.x.data, seq_len, N_EMBD, stream);

    // Up projection: x[seq,n_embd] @ W[mlp_hidden,n_embd]^T = out[seq,mlp_hidden]
    gemm_half(scratch.mlp_hidden.data, scratch.x_norm.data, l.mlp_up_weight.data, seq_len, MLP_HIDDEN, N_EMBD, stream);
    relu_squared(scratch.mlp_hidden.data, seq_len * MLP_HIDDEN, stream);

    // Down projection + residual (fused)
    gemm_half_residual(scratch.x.data, scratch.mlp_hidden.data, l.mlp_down_weight.data, scratch.x.data, seq_len, N_EMBD, MLP_HIDDEN, stream);
}

void Model::forward(const int* tokens, int seq_len, int start_pos, nv_bfloat16* logits_out) {
    Tensor<int> d_tokens;
    d_tokens.allocate(seq_len);
    CUDA_CHECK(cudaMemcpyAsync(d_tokens.data, tokens, seq_len * sizeof(int), cudaMemcpyHostToDevice, stream));

    embedding(scratch.x.data, weights.embed_weight.data, d_tokens.data, seq_len, N_EMBD, stream);

    // RMSNorm after embedding
    rmsnorm(scratch.x.data, scratch.x.data, seq_len, N_EMBD, stream);

    for (int i = 0; i < N_LAYER; i++) {
        attention(i, seq_len, start_pos);
        mlp(i, seq_len);
    }

    // Final RMSNorm
    rmsnorm(scratch.x_norm.data, scratch.x.data, seq_len, N_EMBD, stream);

    // LM head: x[seq,n_embd] @ W[vocab_size,n_embd]^T = logits[seq,vocab_size]
    gemm_half(scratch.logits.data, scratch.x_norm.data, weights.lm_head.data, seq_len, VOCAB_SIZE, N_EMBD, stream);

    // Soft cap logits
    tanh_cap(scratch.logits.data, seq_len * VOCAB_SIZE, LOGIT_CAP, stream);

    // Return logits for last position
    nv_bfloat16* last_logits = scratch.logits.data + (seq_len - 1) * VOCAB_SIZE;
    CUDA_CHECK(cudaMemcpyAsync(logits_out, last_logits, VOCAB_SIZE * sizeof(nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace nanochat

#pragma once

namespace nanochat {

// Model architecture constants
constexpr int SEQ_LEN = 2048;
constexpr int VOCAB_SIZE = 65536;
constexpr int N_LAYER = 34;
constexpr int N_HEAD = 17;
constexpr int N_KV_HEAD = 17;
constexpr int N_EMBD = 2176;
constexpr int HEAD_DIM = N_EMBD / N_HEAD;  // 128
constexpr int MLP_HIDDEN = 4 * N_EMBD;     // 8704

// RoPE constants
constexpr float ROPE_BASE = 10000.0f;
constexpr int ROPE_MAX_SEQ = SEQ_LEN * 10;  // over-compute for safety

// Logit soft cap
constexpr float LOGIT_CAP = 15.0f;

// KV cache growth increment
constexpr int KV_CACHE_GROW = 1024;

}  // namespace nanochat

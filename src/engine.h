#pragma once

#include "model.h"
#include <vector>
#include <random>

namespace nanochat {

struct SamplingParams {
    float temperature = 1.0f;
    float top_p = 0.9f;
    int top_k = 50;
    int max_tokens = 256;
};

class Engine {
public:
    Engine();
    ~Engine();

    void load(const std::string& weights_path);
    void reset();

    // Generate next token from prompt tokens
    int generate_next(const std::vector<int>& prompt_tokens);

    // Generate multiple tokens, returns all generated token IDs
    std::vector<int> generate(const std::vector<int>& prompt_tokens, const SamplingParams& params);

    SamplingParams params;

private:
    Model model;
    Tensor<nv_bfloat16> logits_buf;
    std::vector<float> logits_host;
    std::mt19937 rng;
    int current_pos = 0;

    int sample(const float* logits, int vocab_size);
    void top_k_filter(std::vector<std::pair<float, int>>& probs, int k);
    void top_p_filter(std::vector<std::pair<float, int>>& probs, float p);
};

}  // namespace nanochat

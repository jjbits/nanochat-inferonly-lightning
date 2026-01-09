#include "engine.h"
#include <algorithm>
#include <cmath>

namespace nanochat {

Engine::Engine() : rng(std::random_device{}()) {
    logits_host.resize(VOCAB_SIZE);
}

Engine::~Engine() = default;

void Engine::load(const std::string& weights_path) {
    model.load(weights_path);
    logits_buf.allocate(VOCAB_SIZE);
}

void Engine::reset() {
    current_pos = 0;
    for (auto& cache : model.kv_caches) {
        cache.clear();
    }
}

int Engine::generate_next(const std::vector<int>& prompt_tokens) {
    int seq_len = prompt_tokens.size();

    model.forward(prompt_tokens.data(), seq_len, current_pos, logits_buf.data);
    current_pos += seq_len;

    // Copy logits to host and convert to float
    std::vector<nv_bfloat16> logits_bf16(VOCAB_SIZE);
    logits_buf.to_host(logits_bf16.data());
    for (int i = 0; i < VOCAB_SIZE; i++) {
        logits_host[i] = __bfloat162float(logits_bf16[i]);
    }

    return sample(logits_host.data(), VOCAB_SIZE);
}

std::vector<int> Engine::generate(const std::vector<int>& prompt_tokens, const SamplingParams& p) {
    params = p;
    reset();

    std::vector<int> output;
    std::vector<int> input = prompt_tokens;

    for (int i = 0; i < params.max_tokens; i++) {
        int next_token = generate_next(input);
        output.push_back(next_token);

        // EOS check (token 0 or 1 typically)
        if (next_token == 0 || next_token == 1) break;

        // Next iteration: only feed the new token
        input = {next_token};
    }

    return output;
}

void Engine::top_k_filter(std::vector<std::pair<float, int>>& probs, int k) {
    if (k <= 0 || k >= (int)probs.size()) return;

    std::partial_sort(probs.begin(), probs.begin() + k, probs.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
    probs.resize(k);
}

void Engine::top_p_filter(std::vector<std::pair<float, int>>& probs, float p) {
    if (p >= 1.0f) return;

    std::sort(probs.begin(), probs.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    float cumsum = 0.0f;
    size_t cutoff = probs.size();
    for (size_t i = 0; i < probs.size(); i++) {
        cumsum += probs[i].first;
        if (cumsum > p) {
            cutoff = i + 1;
            break;
        }
    }
    probs.resize(cutoff);
}

int Engine::sample(const float* logits, int vocab_size) {
    // Greedy decoding for temperature=0
    if (params.temperature == 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    // Apply temperature
    std::vector<std::pair<float, int>> probs(vocab_size);
    float max_logit = *std::max_element(logits, logits + vocab_size);

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float scaled = (logits[i] - max_logit) / params.temperature;
        probs[i] = {expf(scaled), i};
        sum += probs[i].first;
    }

    // Normalize
    for (auto& p : probs) p.first /= sum;

    // Top-k then top-p filtering
    top_k_filter(probs, params.top_k);
    top_p_filter(probs, params.top_p);

    // Renormalize after filtering
    sum = 0.0f;
    for (auto& p : probs) sum += p.first;
    for (auto& p : probs) p.first /= sum;

    // Sample from filtered distribution
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cumsum = 0.0f;
    for (auto& p : probs) {
        cumsum += p.first;
        if (r < cumsum) return p.second;
    }

    return probs.back().second;
}

}  // namespace nanochat

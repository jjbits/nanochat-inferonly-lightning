#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

extern "C" {
    int32_t rustbpe_load(const char* path);
    int32_t rustbpe_encode(const char* text, uint32_t* out_tokens, size_t max_tokens);
    int32_t rustbpe_decode(const uint32_t* tokens, size_t n_tokens, char* out_text, size_t max_len);
    int32_t rustbpe_special_token(const char* name);
    int32_t rustbpe_vocab_size();
    void rustbpe_free();
}

namespace nanochat {

class Tokenizer {
public:
    // Special token IDs
    static constexpr int BOS = 65527;
    static constexpr int USER_START = 65528;
    static constexpr int USER_END = 65529;
    static constexpr int ASSISTANT_START = 65530;
    static constexpr int ASSISTANT_END = 65531;
    static constexpr int PYTHON_START = 65532;
    static constexpr int PYTHON_END = 65533;
    static constexpr int OUTPUT_START = 65534;
    static constexpr int OUTPUT_END = 65535;

    Tokenizer() = default;
    ~Tokenizer() {
        if (loaded_) {
            rustbpe_free();
        }
    }

    // Non-copyable
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // Movable
    Tokenizer(Tokenizer&& other) noexcept : loaded_(other.loaded_) {
        other.loaded_ = false;
    }
    Tokenizer& operator=(Tokenizer&& other) noexcept {
        if (this != &other) {
            if (loaded_) rustbpe_free();
            loaded_ = other.loaded_;
            other.loaded_ = false;
        }
        return *this;
    }

    bool load(const std::string& path) {
        if (loaded_) {
            rustbpe_free();
            loaded_ = false;
        }
        if (rustbpe_load(path.c_str()) == 0) {
            loaded_ = true;
            return true;
        }
        return false;
    }

    bool is_loaded() const { return loaded_; }

    std::vector<int> encode(const std::string& text) const {
        if (!loaded_) return {};

        // Allocate buffer (rough estimate: 2 tokens per character max)
        std::vector<uint32_t> tokens(text.size() * 2 + 256);
        int n = rustbpe_encode(text.c_str(), tokens.data(), tokens.size());
        if (n < 0) return {};

        return std::vector<int>(tokens.begin(), tokens.begin() + n);
    }

    std::string decode(const std::vector<int>& tokens) const {
        if (!loaded_ || tokens.empty()) return "";

        std::vector<uint32_t> u32_tokens(tokens.begin(), tokens.end());
        std::vector<char> buf(tokens.size() * 16 + 256);
        int n = rustbpe_decode(u32_tokens.data(), u32_tokens.size(), buf.data(), buf.size());
        if (n < 0) return "";

        return std::string(buf.data(), n);
    }

    int special_token(const std::string& name) const {
        if (!loaded_) return -1;
        return rustbpe_special_token(name.c_str());
    }

    int vocab_size() const {
        if (!loaded_) return -1;
        return rustbpe_vocab_size();
    }

    // Build chat prompt: [BOS, USER_START, ...text..., USER_END, ASSISTANT_START]
    std::vector<int> encode_chat_prompt(const std::string& user_message) const {
        std::vector<int> tokens;
        tokens.push_back(BOS);
        tokens.push_back(USER_START);
        auto text_tokens = encode(user_message);
        tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
        tokens.push_back(USER_END);
        tokens.push_back(ASSISTANT_START);
        return tokens;
    }

private:
    bool loaded_ = false;
};

} // namespace nanochat

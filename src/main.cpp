#include "engine.h"
#include "model.h"
#include "tokenizer.h"
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <chrono>
#include <cstdio>

using namespace nanochat;

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <weights_path> [options]\n"
              << "Options:\n"
              << "  --tokenizer <path>     Path to tokenizer.json\n"
              << "  --prompt <text>        Text prompt (requires --tokenizer)\n"
              << "  --temperature <float>  Sampling temperature (default: 1.0)\n"
              << "  --top_p <float>        Top-p sampling (default: 0.9)\n"
              << "  --top_k <int>          Top-k sampling (default: 50)\n"
              << "  --max_tokens <int>     Max tokens to generate (default: 256)\n"
              << "  --eval                 Evaluation mode: read tokens from stdin, output logits\n"
              << "\n"
              << "Without --prompt, reads space-separated token IDs from stdin.\n"
              << "\n"
              << "Eval mode protocol:\n"
              << "  Input:  seq_len (4 bytes int32) + tokens (seq_len * 4 bytes int32)\n"
              << "  Output: logits (seq_len * vocab_size * 4 bytes float32)\n"
              << "  Send seq_len=0 to exit.\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string weights_path = argv[1];
    std::string tokenizer_path;
    std::string prompt;
    SamplingParams params;
    bool eval_mode = false;

    // Parse args
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--tokenizer" && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--temperature" && i + 1 < argc) {
            params.temperature = std::atof(argv[++i]);
        } else if (arg == "--top_p" && i + 1 < argc) {
            params.top_p = std::atof(argv[++i]);
        } else if (arg == "--top_k" && i + 1 < argc) {
            params.top_k = std::atoi(argv[++i]);
        } else if (arg == "--max_tokens" && i + 1 < argc) {
            params.max_tokens = std::atoi(argv[++i]);
        } else if (arg == "--eval") {
            eval_mode = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Evaluation mode: compute loss for continuation tokens
    if (eval_mode) {
        Model model;
        std::cerr << "Loading weights from " << weights_path << "...\n";
        model.load(weights_path);
        std::cerr << "Loaded. Ready for evaluation.\n";

        // Protocol: read seq_len, cont_start, tokens; write avg_loss, num_correct, num_tokens
        while (true) {
            int32_t seq_len, cont_start;
            if (fread(&seq_len, sizeof(int32_t), 1, stdin) != 1) break;
            if (seq_len <= 0) break;
            if (fread(&cont_start, sizeof(int32_t), 1, stdin) != 1) break;

            std::vector<int> tokens(seq_len);
            if (fread(tokens.data(), sizeof(int32_t), seq_len, stdin) != (size_t)seq_len) break;

            std::vector<float> logits(seq_len * VOCAB_SIZE);
            model.forward_all(tokens.data(), seq_len, logits.data());

            // Compute cross-entropy loss and argmax accuracy over continuation tokens
            // Loss at position i predicts token i+1
            double total_loss = 0.0;
            int32_t num_tokens = 0;
            int32_t num_correct = 0;
            for (int i = cont_start - 1; i < seq_len - 1; i++) {
                int target = tokens[i + 1];
                float* logits_row = &logits[i * VOCAB_SIZE];

                // Find max logit and argmax
                float max_logit = logits_row[0];
                int argmax = 0;
                for (int j = 1; j < VOCAB_SIZE; j++) {
                    if (logits_row[j] > max_logit) {
                        max_logit = logits_row[j];
                        argmax = j;
                    }
                }

                // Count correct predictions
                if (argmax == target) num_correct++;

                // Numerically stable cross-entropy
                double sum_exp = 0.0;
                for (int j = 0; j < VOCAB_SIZE; j++) {
                    sum_exp += std::exp(logits_row[j] - max_logit);
                }
                double log_prob = (logits_row[target] - max_logit) - std::log(sum_exp);
                total_loss -= log_prob;
                num_tokens++;
            }

            float avg_loss = (num_tokens > 0) ? (float)(total_loss / num_tokens) : 0.0f;
            fwrite(&avg_loss, sizeof(float), 1, stdout);
            fwrite(&num_correct, sizeof(int32_t), 1, stdout);
            fwrite(&num_tokens, sizeof(int32_t), 1, stdout);
            fflush(stdout);
        }
        return 0;
    }

    std::vector<int> prompt_tokens;
    Tokenizer tokenizer;
    bool use_tokenizer = !tokenizer_path.empty() && !prompt.empty();

    if (use_tokenizer) {
        // Text mode: load tokenizer and encode prompt
        if (!tokenizer.load(tokenizer_path)) {
            std::cerr << "Error: Failed to load tokenizer from " << tokenizer_path << "\n";
            return 1;
        }

        // Build chat format: [BOS, USER_START, ...text..., USER_END, ASSISTANT_START]
        prompt_tokens = tokenizer.encode_chat_prompt(prompt);
    } else {
        // Legacy mode: read token IDs from stdin
        std::string line;
        while (std::getline(std::cin, line)) {
            std::istringstream iss(line);
            int token;
            while (iss >> token) {
                prompt_tokens.push_back(token);
            }
        }
    }

    if (prompt_tokens.empty()) {
        std::cerr << "Error: No input provided\n";
        return 1;
    }

    // Debug: print tokens
    std::cerr << "Prompt tokens (" << prompt_tokens.size() << "): ";
    for (int t : prompt_tokens) std::cerr << t << " ";
    std::cerr << "\n";

    // Load and run
    Engine engine;
    std::cerr << "Loading weights from " << weights_path << "...\n";
    engine.load(weights_path);
    std::cerr << "Loaded. Generating...\n";

    engine.params = params;
    engine.reset();

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<int> input = prompt_tokens;
    std::vector<int> output_tokens;
    std::string prev_decoded;
    int token_count = 0;

    for (int i = 0; i < params.max_tokens; i++) {
        int next_token = engine.generate_next(input);
        token_count++;

        // Stop conditions
        if (next_token == Tokenizer::ASSISTANT_END ||
            next_token == Tokenizer::BOS ||
            next_token == 0 || next_token == 1) {
            break;
        }

        if (use_tokenizer) {
            // Streaming text output
            output_tokens.push_back(next_token);
            std::string decoded = tokenizer.decode(output_tokens);
            if (decoded.size() > prev_decoded.size()) {
                std::cout << decoded.substr(prev_decoded.size()) << std::flush;
                prev_decoded = decoded;
            }
        } else {
            // Legacy: output token IDs
            std::cout << next_token << std::endl;
        }

        input = {next_token};
    }

    if (use_tokenizer) {
        std::cout << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    std::cerr << "Generated " << token_count << " tokens in " << elapsed << "s ("
              << (token_count / elapsed) << " tok/s)\n";

    return 0;
}

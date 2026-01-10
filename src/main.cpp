#include "engine.h"
#include "tokenizer.h"
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <chrono>

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
              << "\n"
              << "Without --prompt, reads space-separated token IDs from stdin.\n";
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
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
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

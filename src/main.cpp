#include "engine.h"
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <chrono>

using namespace nanochat;

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <weights_path> [options]\n"
              << "Options:\n"
              << "  --temperature <float>  Sampling temperature (default: 1.0)\n"
              << "  --top_p <float>        Top-p sampling (default: 0.9)\n"
              << "  --top_k <int>          Top-k sampling (default: 50)\n"
              << "  --max_tokens <int>     Max tokens to generate (default: 256)\n"
              << "\n"
              << "Reads space-separated token IDs from stdin.\n"
              << "Outputs generated token IDs, one per line.\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string weights_path = argv[1];
    SamplingParams params;

    // Parse args
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--temperature" && i + 1 < argc) {
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

    // Read prompt tokens from stdin
    std::vector<int> prompt_tokens;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        int token;
        while (iss >> token) {
            prompt_tokens.push_back(token);
        }
    }

    if (prompt_tokens.empty()) {
        std::cerr << "Error: No input tokens provided\n";
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

    // Stream tokens one at a time
    std::vector<int> input = prompt_tokens;
    int token_count = 0;
    for (int i = 0; i < params.max_tokens; i++) {
        int next_token = engine.generate_next(input);
        token_count++;

        // Output immediately and flush
        std::cout << next_token << std::endl;  // endl flushes

        // EOS check
        if (next_token == 0 || next_token == 1) break;

        // Next iteration: only feed the new token
        input = {next_token};
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    std::cerr << "Generated " << token_count << " tokens in " << elapsed << "s ("
              << (token_count / elapsed) << " tok/s)\n";

    return 0;
}

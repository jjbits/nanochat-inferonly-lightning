#include <httplib.h>
#include <nlohmann/json.hpp>
#include "engine.h"
#include "tokenizer.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <chrono>
#include <netinet/tcp.h>
#include <cuda_runtime.h>
#include <mutex>
#include <memory>
#include <thread>

using json = nlohmann::json;
using namespace nanochat;

// Global tokenizer (loaded at startup) and engine (lazy init in handler thread)
Tokenizer tokenizer;
std::unique_ptr<Engine> engine;
std::mutex engine_mutex;
bool engine_loaded = false;

// Command line args
struct ServerArgs {
    std::string weights_path = "assets/weights/weights.bin";
    std::string tokenizer_path = "assets/tokenizer.json";
    int port = 8000;
    std::string host = "0.0.0.0";
    float temperature = 0.8f;
    float top_p = 0.9f;
    int top_k = 50;
    int max_tokens = 512;
};

ServerArgs args;

// Embedded UI HTML
const char* UI_HTML = R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NanoChat Lightning</title>
    <style>
        * { box-sizing: border-box; }
        html, body { height: 100%; margin: 0; }
        body {
            font-family: -apple-system, system-ui, "Segoe UI", Helvetica, Arial, sans-serif;
            background: #fff;
            color: #111;
            display: flex;
            flex-direction: column;
        }
        .header {
            padding: 1rem 1.5rem;
            border-bottom: 1px solid #e5e7eb;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .header h1 { font-size: 1.25rem; margin: 0; }
        .header span { color: #f59e0b; font-size: 0.875rem; }
        .new-btn {
            width: 32px; height: 32px;
            border: 1px solid #e5e7eb;
            border-radius: 0.5rem;
            background: #fff;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .new-btn:hover { background: #f3f4f6; }
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
        }
        .chat-wrapper {
            max-width: 48rem;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .message { display: flex; }
        .message.user { justify-content: flex-end; }
        .message.assistant { justify-content: flex-start; }
        .message-content {
            white-space: pre-wrap;
            line-height: 1.6;
            max-width: 80%;
            padding: 0.75rem 1rem;
            border-radius: 1rem;
        }
        .message.user .message-content {
            background: #f3f4f6;
        }
        .message.assistant .message-content {
            background: transparent;
        }
        .input-container {
            padding: 1rem;
            border-top: 1px solid #e5e7eb;
        }
        .input-wrapper {
            max-width: 48rem;
            margin: 0 auto;
            display: flex;
            gap: 0.75rem;
        }
        .chat-input {
            flex: 1;
            padding: 0.75rem 1rem;
            border: 1px solid #d1d5db;
            border-radius: 0.75rem;
            font-size: 1rem;
            resize: none;
            outline: none;
            min-height: 50px;
            max-height: 150px;
        }
        .chat-input:focus { border-color: #2563eb; }
        .send-btn {
            width: 50px; height: 50px;
            border: none;
            border-radius: 0.75rem;
            background: #111;
            color: #fff;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .send-btn:hover { background: #2563eb; }
        .send-btn:disabled { background: #e5e7eb; color: #9ca3af; cursor: not-allowed; }
        .typing::after {
            content: '...';
            animation: typing 1s infinite;
        }
        @keyframes typing {
            0%, 100% { opacity: 0.2; }
            50% { opacity: 1; }
        }
        .error { background: #fee2e2; color: #b91c1c; padding: 0.75rem; border-radius: 0.5rem; }
    </style>
</head>
<body>
    <div class="header">
        <button class="new-btn" onclick="newChat()" title="New Chat">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 5v14M5 12h14"/>
            </svg>
        </button>
        <h1>nanochat</h1>
        <span>lightning</span>
    </div>
    <div class="chat-container" id="chatContainer">
        <div class="chat-wrapper" id="chatWrapper"></div>
    </div>
    <div class="input-container">
        <div class="input-wrapper">
            <textarea id="chatInput" class="chat-input" placeholder="Ask anything..." rows="1"
                onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea>
            <button id="sendBtn" class="send-btn" onclick="send()">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                </svg>
            </button>
        </div>
    </div>
    <script>
        const chatWrapper = document.getElementById('chatWrapper');
        const chatContainer = document.getElementById('chatContainer');
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        let messages = [];
        let generating = false;

        chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        });

        function newChat() {
            messages = [];
            chatWrapper.innerHTML = '';
            chatInput.value = '';
            chatInput.focus();
        }

        function addMessage(role, content) {
            const div = document.createElement('div');
            div.className = 'message ' + role;
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            div.appendChild(contentDiv);
            chatWrapper.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return contentDiv;
        }

        async function send() {
            const text = chatInput.value.trim();
            if (!text || generating) return;

            generating = true;
            sendBtn.disabled = true;
            chatInput.value = '';
            chatInput.style.height = 'auto';

            messages.push({ role: 'user', content: text });
            addMessage('user', text);

            const assistantDiv = addMessage('assistant', '');
            assistantDiv.innerHTML = '<span class="typing"></span>';

            try {
                const response = await fetch('/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ messages, temperature: 0.8, top_k: 50, max_tokens: 512 })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullResponse = '';
                assistantDiv.textContent = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value);
                    for (const line of chunk.split('\n')) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.token) {
                                    fullResponse += data.token;
                                    assistantDiv.textContent = fullResponse;
                                    chatContainer.scrollTop = chatContainer.scrollHeight;
                                }
                            } catch (e) {}
                        }
                    }
                }

                messages.push({ role: 'assistant', content: fullResponse });
            } catch (error) {
                assistantDiv.innerHTML = '<div class="error">Error: ' + error.message + '</div>';
            } finally {
                generating = false;
                sendBtn.disabled = false;
            }
        }

        chatInput.focus();
    </script>
</body>
</html>)HTML";

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --weights <path>       Path to weights.bin (default: assets/weights/weights.bin)\n"
              << "  --tokenizer <path>     Path to tokenizer.json (default: assets/tokenizer.json)\n"
              << "  --port <int>           Server port (default: 8000)\n"
              << "  --host <str>           Host to bind (default: 0.0.0.0)\n"
              << "  --temperature <float>  Default temperature (default: 0.8)\n"
              << "  --top_p <float>        Default top-p (default: 0.9)\n"
              << "  --top_k <int>          Default top-k (default: 50)\n"
              << "  --max_tokens <int>     Default max tokens (default: 512)\n";
}

bool parse_args(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        } else if (arg == "--weights" && i + 1 < argc) {
            args.weights_path = argv[++i];
        } else if (arg == "--tokenizer" && i + 1 < argc) {
            args.tokenizer_path = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            args.port = std::atoi(argv[++i]);
        } else if (arg == "--host" && i + 1 < argc) {
            args.host = argv[++i];
        } else if (arg == "--temperature" && i + 1 < argc) {
            args.temperature = std::atof(argv[++i]);
        } else if (arg == "--top_p" && i + 1 < argc) {
            args.top_p = std::atof(argv[++i]);
        } else if (arg == "--top_k" && i + 1 < argc) {
            args.top_k = std::atoi(argv[++i]);
        } else if (arg == "--max_tokens" && i + 1 < argc) {
            args.max_tokens = std::atoi(argv[++i]);
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) {
        return 0;
    }

    // Load tokenizer
    std::cerr << "Loading tokenizer from " << args.tokenizer_path << "...\n";
    if (!tokenizer.load(args.tokenizer_path)) {
        std::cerr << "Error: Failed to load tokenizer\n";
        return 1;
    }
    std::cerr << "Tokenizer loaded (vocab size: " << tokenizer.vocab_size() << ")\n";

    // Engine will be loaded lazily on first request (in handler thread for CUDA context)
    std::cerr << "Engine will be loaded on first request...\n";

    httplib::Server server;

    // Disable Nagle's algorithm for immediate packet transmission
    server.set_socket_options([](socket_t sock) {
        int yes = 1;
        setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&yes), sizeof(yes));
    });

    // CORS middleware
    server.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // GET / - Serve UI
    server.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(UI_HTML, "text/html");
    });

    // GET /health
    server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        json response = {{"status", "ok"}, {"tokenizer_loaded", tokenizer.is_loaded()}};
        res.set_content(response.dump(), "application/json");
    });

    // POST /chat/completions - SSE streaming
    server.Post("/chat/completions", [](const httplib::Request& req, httplib::Response& res) {
        // Lazy init engine in handler thread (for CUDA context)
        {
            std::lock_guard<std::mutex> lock(engine_mutex);
            if (!engine_loaded) {
                cudaSetDevice(0);
                cudaFree(0);  // Force context creation in this thread
                std::cerr << "Loading weights from " << args.weights_path << "...\n";
                engine = std::make_unique<Engine>();
                engine->load(args.weights_path);
                engine_loaded = true;
                std::cerr << "Weights loaded.\n";
            }
        }

        try {
            auto request = json::parse(req.body);

            // Get parameters
            float temperature = request.value("temperature", args.temperature);
            float top_p = request.value("top_p", args.top_p);
            int top_k = request.value("top_k", args.top_k);
            int max_tokens = request.value("max_tokens", args.max_tokens);

            // Build conversation tokens
            std::vector<int> tokens = {Tokenizer::BOS};
            for (const auto& msg : request["messages"]) {
                std::string role = msg["role"];
                std::string content = msg["content"];

                if (role == "user") {
                    tokens.push_back(Tokenizer::USER_START);
                    auto text_tokens = tokenizer.encode(content);
                    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
                    tokens.push_back(Tokenizer::USER_END);
                } else if (role == "assistant") {
                    tokens.push_back(Tokenizer::ASSISTANT_START);
                    auto text_tokens = tokenizer.encode(content);
                    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
                    tokens.push_back(Tokenizer::ASSISTANT_END);
                }
            }
            tokens.push_back(Tokenizer::ASSISTANT_START);

            // Set engine parameters and reset
            engine->params.temperature = temperature;
            engine->params.top_p = top_p;
            engine->params.top_k = top_k;
            engine->params.max_tokens = max_tokens;
            engine->reset();

            // Generate tokens
            std::vector<int> input = tokens;
            std::vector<int> output_tokens;
            std::string prev_decoded;
            std::vector<std::string> events;

            for (int i = 0; i < max_tokens; i++) {
                int next = engine->generate_next(input);

                // Stop conditions
                if (next == Tokenizer::ASSISTANT_END ||
                    next == Tokenizer::BOS ||
                    next == 0 || next == 1) {
                    break;
                }

                output_tokens.push_back(next);
                std::string decoded = tokenizer.decode(output_tokens);

                // Emit new text when we have complete UTF-8
                if (decoded.size() > prev_decoded.size() &&
                    (decoded.back() & 0xC0) != 0x80) {
                    std::string delta = decoded.substr(prev_decoded.size());
                    json event_json = {{"token", delta}};
                    events.push_back("data: " + event_json.dump() + "\n\n");
                    prev_decoded = decoded;
                }

                input = {next};
            }
            events.push_back("data: {\"done\":true}\n\n");

            // Set response headers
            res.set_header("Cache-Control", "no-cache, no-store");
            res.set_header("X-Accel-Buffering", "no");

            // Concatenate all events and send as single response
            // (httplib doesn't support true SSE streaming)
            std::string body;
            for (const auto& event : events) {
                body += event;
            }
            res.set_content(body, "text/event-stream");

        } catch (const std::exception& e) {
            json error = {{"error", e.what()}};
            res.status = 400;
            res.set_content(error.dump(), "application/json");
        }
    });

    // Single-threaded mode: engine is lazily loaded in handler thread
    // and CUDA contexts are per-thread, so all requests must use same thread
    server.new_task_queue = [] { return new httplib::ThreadPool(1); };

    // Warmup thread: trigger engine loading before real requests
    std::thread warmup([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        httplib::Client cli("127.0.0.1", args.port);
        cli.Post("/chat/completions",
                 R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":1})",
                 "application/json");
        std::cerr << "Warmup complete.\n";
    });
    warmup.detach();

    std::cerr << "Server ready at http://" << args.host << ":" << args.port << "\n";
    server.listen(args.host, args.port);

    return 0;
}

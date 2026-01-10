#!/usr/bin/env python3
"""
Web chat server for nanochat-inferonly-lightning.
Identical to nanochat-inferonly server but uses C++ binary for inference.

Usage:
    python server.py
    python server.py --port 8080
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator

# Add parent nanochat-inferonly to path for tokenizer
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nanochat-inferonly"))
from tokenizer import Tokenizer

parser = argparse.ArgumentParser(description='NanoChat Lightning Web Server')
parser.add_argument('--port', type=int, default=8000, help='Port to run server on')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
parser.add_argument('--binary', type=str, default='../build/nanochat', help='Path to nanochat binary')
parser.add_argument('--weights', type=str, default='../assets/weights/weights.bin', help='Path to weights.bin')
parser.add_argument('--tokenizer', type=str, default='../../nanochat-inferonly/weights/weights_karpathy_d34_huggingface/tokenizer.pkl')
parser.add_argument('--temperature', type=float, default=0.8, help='Default temperature')
parser.add_argument('--top-k', type=int, default=50, help='Default top-k')
parser.add_argument('--top-p', type=float, default=0.9, help='Default top-p')
parser.add_argument('--max-tokens', type=int, default=512, help='Default max tokens')
args = parser.parse_args()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None


tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f"Tokenizer loaded (vocab size: {tokenizer.get_vocab_size()})")
    print(f"Using binary: {args.binary}")
    print(f"Using weights: {args.weights}")
    print(f"Server ready at http://localhost:{args.port}")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exact same HTML/JS as the working Python server
UI_HTML = '''<!DOCTYPE html>
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
                    for (const line of chunk.split('\\n')) {
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
</html>'''


@app.get("/")
async def root():
    return HTMLResponse(content=UI_HTML)


async def generate_stream(tokens: List[int], temperature: float, max_tokens: int, top_k: int):
    """Stream tokens from C++ binary."""
    import asyncio

    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    bos = tokenizer.get_bos_token_id()

    cmd = [
        args.binary, args.weights,
        "--temperature", str(temperature),
        "--top_p", str(args.top_p),
        "--top_k", str(top_k),
        "--max_tokens", str(max_tokens),
    ]

    # Use asyncio subprocess
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    # Send tokens to stdin
    token_str = " ".join(map(str, tokens)) + "\n"
    process.stdin.write(token_str.encode())
    await process.stdin.drain()
    process.stdin.close()
    await process.stdin.wait_closed()

    accumulated_tokens = []
    last_clean_text = ""

    # Read tokens line by line from stdout
    while True:
        line = await process.stdout.readline()
        if not line:
            break

        try:
            token = int(line.decode().strip())
        except ValueError:
            continue

        # Stop on special tokens
        if token == assistant_end or token == bos:
            break

        accumulated_tokens.append(token)
        current_text = tokenizer.decode(accumulated_tokens)

        # Only emit when we have complete UTF-8 (no replacement char)
        if not current_text.endswith('\ufffd'):
            new_text = current_text[len(last_clean_text):]
            if new_text:
                yield f"data: {json.dumps({'token': new_text}, ensure_ascii=False)}\n\n"
                last_clean_text = current_text

    await process.wait()
    yield f"data: {json.dumps({'done': True})}\n\n"


@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    temperature = request.temperature if request.temperature is not None else args.temperature
    max_tokens = request.max_tokens if request.max_tokens is not None else args.max_tokens
    top_k = request.top_k if request.top_k is not None else args.top_k

    # Build conversation tokens
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    conversation_tokens = [bos]
    for message in request.messages:
        if message.role == "user":
            conversation_tokens.append(user_start)
            conversation_tokens.extend(tokenizer.encode(message.content))
            conversation_tokens.append(user_end)
        elif message.role == "assistant":
            conversation_tokens.append(assistant_start)
            conversation_tokens.extend(tokenizer.encode(message.content))
            conversation_tokens.append(assistant_end)

    conversation_tokens.append(assistant_start)

    return StreamingResponse(
        generate_stream(conversation_tokens, temperature, max_tokens, top_k),
        media_type="text/event-stream"
    )


@app.get("/health")
async def health():
    return {"status": "ok", "tokenizer_loaded": tokenizer is not None}


if __name__ == "__main__":
    import uvicorn
    print(f"Starting NanoChat Lightning Web Server on port {args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

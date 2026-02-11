# nanochat-inferonly-lightning

High-performance inference only C++/CUDA [Nanochat](https://github.com/karpathy/nanochat) for a 2.2B parameter GPT-style transformer. Converted from [nanochat-inferonly](https://github.com/jjbits/nanochat-inferonly) (Python/PyTorch) with custom CUDA kernels and CUTLASS GEMMs.

**2.56x faster than PyTorch implementation (nanochat-inferonly)** — 108 tok/s vs 42.2 tok/s on RTX 4090.

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | ~2.2B (4GB bf16) |
| Layers | 34 |
| Attention Heads | 17 |
| Embedding Dim | 2176 |
| Head Dim | 128 |
| MLP Hidden | 8704 (4x) |
| Sequence Length | 2048 |
| Vocab Size | 65,536 |

**Features:** RoPE, RMSNorm, ReLU², QK-norm, soft-capped logits

## Forward Pass

```
tokens [seq]
   ↓
embedding → x [seq, 2176]
   ↓
rmsnorm(x) [seq, 2176] (in-place)
   ↓
┌───────────────────── Layer 0-33 ─────────────────────┐
│                                                      │
│  ┌─ Attention ────────────────────────────────────┐  │
│  │  rmsnorm(x) → x_norm [seq, 2176]               │  │
│  │  x_norm × Wq → Q [seq, 17, 128]                │  │
│  │  x_norm × Wk → K [seq, 17, 128]                │  │
│  │  x_norm × Wv → V [seq, 17, 128]                │  │
│  │  Q → RoPE → RMSNorm → Q [seq, 17, 128]         │  │
│  │  K → RoPE → RMSNorm → K [seq, 17, 128]         │  │
│  │  K → append to K cache [kv_len, 17, 128]       │  │
│  │  V → append to V cache [kv_len, 17, 128]       │  │
│  │  Q, K_cache, V_cache →                         │  │
│  │       Flash Attention → attn_out [seq, 2176]   │  │
│  │  attn_out × Wo → proj [seq, 2176]              │  │
│  └────────────────────────────────────────────────┘  │
│  x = proj + x [seq, 2176] (residual)                 │
│                                                      │
│  ┌─ MLP ──────────────────────────────────────────┐  │
│  │  rmsnorm(x) → x_norm [seq, 2176]               │  │
│  │  x_norm × W_up → hidden [seq, 8704]            │  │
│  │  ReLU²(hidden) [seq, 8704]                     │  │
│  │  hidden × W_down → mlp_out [seq, 2176]         │  │
│  └────────────────────────────────────────────────┘  │
│  x = mlp_out + x [seq, 2176] (residual)              │
│                                                      │
└──────────────────────────────────────────────────────┘
   ↓
rmsnorm(x) → x_norm [seq, 2176]
   ↓
x_norm × W_lm_head → logits [seq, 65536]
   ↓
tanh_cap(logits) [seq, 65536]
   ↓
logits[-1] → output [65536]
```

## Implementation Approach

**Custom CUDA kernels** for simple ops with full control:
- RMSNorm, RoPE, Embedding, ReLU², Tanh cap

**CUTLASS** for matrix multiplications with fusion:
- Epilogue fusion for residual adds (`D = A @ B + residual` in one kernel)

**rustbpe** for tokenization:
- BPE tokenizer via C FFI
- No Python dependency at runtime

## Performance Optimizations

| Optimization |Python Reference Model| C++ | Speedup Source |
|--------------|--------|-----|----------------|
| **Attention** | PyTorch SDPA | Custom Flash Attention with online softmax, multi-warp | Fuses Q@K + mask + softmax + @V into single kernel |
| **RoPE + QK-norm** | 4 separate ops | 1 fused kernel | Saves 68 kernel launches/forward |
| **Residual + GEMM** | `x = x + proj(y)` | CUTLASS epilogue fusion | Saves 68 kernel launches/forward |
| **Small-M GEMM** | Fixed 128×128 tiles | Adaptive 64×64 tiles for decode | 2x+ speedup for seq_len=1 |
| **Precision** | Mixed with conversions | bf16 throughout | No type conversion overhead |
| **Runtime** | Python + PyTorch | Direct CUDA calls | Lower latency per op |
| **Tokenizer** | tiktoken (Python) | rustbpe C FFI | No Python at runtime |

Extra post-implementation optimization, through profiling, results: [PROFILING.md](PROFILING.md)

### Kernel Launch Reduction

| Version | Kernels per Forward |
|---------|---------------------|
| Python (SDPA) | ~300+ |
| C++ | **~34** |

### Benchmark Results

| Metric | Python | C++ |
|--------|--------|-----|
| Inference (50 tokens) | 1.18s | **0.46s** |
| Speed | 42.2 tok/s | **108 tok/s** |
| Total (with load) | 8.9s | **~5.5s** |
| vs Python | baseline | **2.56x faster** |

### Model Validation (CORE Benchmark)

Validated implementation correctness using the [CORE benchmark](https://arxiv.org/abs/2406.11794) (HellaSwag, PIQA, ARC, WinoGrande, etc.):

| Implementation | CORE Score |
|----------------|------------|
| **C++ (this repo)** | **0.3630** |
| Python (nanochat) | 0.2219 |

Run evaluation:
```bash
# Install dependencies
uv pip install tiktoken pyyaml numpy

# Run CORE benchmark (100 examples per task, ~5 min)
.venv/bin/python utils/eval_core.py --max-per-task 100
```

## Project Structure

```
nanochat-inferonly-lightning/
├── src/
│   ├── config.h          # Model constants
│   ├── tensor.h          # GPU tensor wrapper
│   ├── tokenizer.h       # rustbpe C++ wrapper
│   ├── model.h/cpp       # Model forward pass
│   ├── engine.h/cpp      # Sampling and generation
│   └── main.cpp          # CLI interface
├── kernels/
│   ├── rmsnorm.cu        # RMSNorm kernel
│   ├── rope.cu           # RoPE + fused RoPE/RMSNorm
│   ├── flash_attention.cu# Custom Flash Attention
│   ├── ops.cu            # Embedding, ReLU², tanh cap
│   └── gemm.cu           # CUTLASS GEMM wrappers
├── utils/
│   ├── convert_weights.py    # .pt → binary weights
│   ├── export_tokenizer.py   # tiktoken → JSON
│   └── server/server.cpp     # C++ web server with SSE
├── extern/
│   └── rustbpe/          # BPE tokenizer (git submodule)
└── assets/
    ├── weights.bin       # Model weights (bf16)
    └── tokenizer.json    # BPE tokenizer
```

## Requirements

- **GPU:** NVIDIA Ampere or newer (RTX 30xx, RTX 40xx, A100, etc.) with 10GB+ VRAM
- **CUDA:** 12.0+
- **CMake:** 3.18+
- **Rust:** for rustbpe tokenizer

## Build

```bash

git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

## Download Weights

```bash
# Download from HuggingFace (4.2GB)
wget https://huggingface.co/jjbits/nanochat-inferonly-lightning/resolve/main/weights.bin -O assets/weights.bin
wget https://huggingface.co/jjbits/nanochat-inferonly-lightning/resolve/main/tokenizer.json -O assets/tokenizer.json
```

## Usage

### CLI Inference

```bash
./build/nanochat assets/weights.bin \
  --tokenizer assets/tokenizer.json \
  --prompt "What is 2+2?" \
  --max_tokens 50
```

### Web Server

```bash
./build/nanochat-server \
  --weights assets/weights.bin \
  --tokenizer assets/tokenizer.json \
  --port 8000
```

Opens a web UI with SSE streaming at `http://localhost:8000`.

## Docker

Alternatively, use Docker (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

```bash
# Download weights (once)
mkdir -p weights
wget https://huggingface.co/jjbits/nanochat-inferonly-lightning/resolve/main/weights.bin -O weights/weights.bin
wget https://huggingface.co/jjbits/nanochat-inferonly-lightning/resolve/main/tokenizer.json -O weights/tokenizer.json

# Run CLI
docker run --gpus all -v $(pwd)/weights:/weights joonhjung/nanochat-inferonly-lightning \
  --prompt "What is 2+2?" --max_tokens 50

# Run server
docker run --gpus all -p 8000:8000 -v $(pwd)/weights:/weights joonhjung/nanochat-inferonly-lightning server
```

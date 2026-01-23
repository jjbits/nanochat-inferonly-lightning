# GPU Profiling: nanochat-inferonly-lightning

Systematic profiling to identify and optimize performance bottlenecks.

## Tools

- **Nsight Systems (nsys)** — Timeline analysis, kernel launch patterns, memory transfers
- **Nsight Compute (ncu)** — Roofline analysis, SM utilization, memory throughput

## Baseline Profile

**Test:** 50-token generation on RTX 4090

```bash
nsys profile --stats=true ./build/nanochat assets/weights.bin \
  --tokenizer assets/tokenizer.json --prompt "Hello" --max_tokens 50
```

### Result: GEMM Dominates

| Kernel | GPU Time | % |
|--------|----------|---|
| CUTLASS GEMM | 898 ms | **97%** |
| Flash Attention | 12 ms | 1.3% |
| RMSNorm | 9 ms | 1.0% |
| RoPE+RMSNorm | 6 ms | 0.6% |
| ReLU² | 2 ms | 0.2% |

**Insight:** Any optimization not targeting GEMM has <3% potential impact.

## GEMM Analysis

Profiled GEMM kernels with roofline analysis:

```bash
ncu --set roofline -o profile_gemm ./build/nanochat ...
```

### Finding: Small Grid Size

| Grid | Blocks | SM Utilization | Compute Throughput |
|------|--------|----------------|-------------------|
| (8, 3, 1) | 24 | 0.09 waves | 6.2% |

RTX 4090 has 128 SMs. With only 24 blocks, **91% of SMs are idle**.

ncu message: *"This kernel grid is too small to fill the available resources"*

### Root Cause

During decode (seq_len=1), GEMM uses 128×128 tiles:
- Output: 1 × 2176 (M=1, N=2176)
- Tiles: ceil(1/128) × ceil(2176/128) = 1 × 17 = **17 blocks**

Only 17 blocks for 128 SMs = severe underutilization.

### Fix: Adaptive Tile Size

```cpp
if (M <= 16) {
    // Use 64×64 tiles → more blocks
} else {
    // Use 128×128 tiles → better for large M
}
```

With 64×64 tiles: ceil(2176/64) = **34 blocks** → 2x more parallelism.

### Result: 2.15x Speedup

| Metric | 128×128 tiles | 64×64 tiles |
|--------|---------------|-------------|
| Blocks | 17 | 34 |
| Compute throughput | 6% | 14% |
| Memory throughput | 17% | 81% |
| **Speed** | **50 tok/s** | **108 tok/s** |

Kernel is now memory-bound (81% DRAM) rather than SM-starved.

## Memory Analysis

Profiled memory operations:

```bash
nsys profile --stats=true -o profile_memory ./build/nanochat ...
```

### Transfer Breakdown

| Operation | Time/Token | Notes |
|-----------|------------|-------|
| D→H (logits) | 10 μs | bf16→float conversion on CPU |
| D→D (KV cache) | 69 μs | Already async, overlaps compute |

Both negligible vs 7.7ms GEMM time per token.

### Allocation Pattern

Found 296 cudaMalloc calls for 10-token inference:

| Source | Allocations |
|--------|-------------|
| Weight loading | 206 |
| KV caches | 68 |
| Scratch buffers | 12 |
| **d_tokens (per forward)** | **10** ← wasteful |

### Bug: Tensor Reuse

```cpp
// Before: always reallocates
void allocate(size_t n) {
    if (data) cudaFree(data);
    cudaMalloc(&data, n * sizeof(T));
}

// After: reuses same-size buffer
void allocate(size_t n) {
    if (size == n && data) return;
    if (data) cudaFree(data);
    cudaMalloc(&data, n * sizeof(T));
}
```

Fixed, though impact is minimal (allocations dominated by startup).

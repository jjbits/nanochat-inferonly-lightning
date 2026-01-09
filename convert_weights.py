#!/usr/bin/env python3
"""Convert PyTorch model weights to raw binary format for C++ inference."""

import argparse
import torch
from pathlib import Path

# Must match config.h
N_LAYER = 34
N_EMBD = 2176
MLP_HIDDEN = 8704
VOCAB_SIZE = 65536


def write_tensor(f, tensor, name, expected_shape=None):
    """Write tensor as contiguous bfloat16 binary."""
    t = tensor.detach().cpu().to(torch.bfloat16).contiguous()
    if expected_shape:
        assert t.shape == torch.Size(expected_shape), f"{name}: {t.shape} vs {expected_shape}"
    # bfloat16 needs special handling - convert to uint16 view for writing
    f.write(t.view(torch.uint16).numpy().tobytes())
    print(f"  {name}: {list(t.shape)}")


def convert(checkpoint_path: str, output_path: str):
    print(f"Loading {checkpoint_path}...")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    with open(output_path, "wb") as f:
        # Embedding: [vocab_size, n_embd]
        write_tensor(f, state["transformer.wte.weight"], "embed", (VOCAB_SIZE, N_EMBD))

        # Layers
        for i in range(N_LAYER):
            prefix = f"transformer.h.{i}."
            print(f"Layer {i}:")

            # Attention weights
            write_tensor(f, state[prefix + "attn.c_q.weight"], "  q", (N_EMBD, N_EMBD))
            write_tensor(f, state[prefix + "attn.c_k.weight"], "  k", (N_EMBD, N_EMBD))
            write_tensor(f, state[prefix + "attn.c_v.weight"], "  v", (N_EMBD, N_EMBD))
            write_tensor(f, state[prefix + "attn.c_proj.weight"], "  o", (N_EMBD, N_EMBD))

            # MLP weights
            write_tensor(f, state[prefix + "mlp.c_fc.weight"], "  mlp_up", (MLP_HIDDEN, N_EMBD))
            write_tensor(f, state[prefix + "mlp.c_proj.weight"], "  mlp_down", (N_EMBD, MLP_HIDDEN))

        # LM head: [vocab_size, n_embd]
        write_tensor(f, state["lm_head.weight"], "lm_head", (VOCAB_SIZE, N_EMBD))

    print(f"Done: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("-o", "--output", default="weights.bin")
    args = parser.parse_args()
    convert(args.checkpoint, args.output)

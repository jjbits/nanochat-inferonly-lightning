#!/usr/bin/env python3
"""Export tiktoken tokenizer.pkl to JSON format for Rust FFI.

Usage: python export_tokenizer.py <input.pkl> <output.json>

The output JSON contains:
- pattern: regex pattern for text splitting
- vocab: list of hex-encoded bytes, indexed by token ID
- encoder: dict of hex-encoded bytes -> token ID (for encoding lookups)
- special_tokens: dict of name -> token ID
"""

import pickle
import json
import sys


def export_tokenizer(pkl_path: str, output_path: str) -> None:
    with open(pkl_path, 'rb') as f:
        enc = pickle.load(f)

    # Get pattern
    pattern = enc._pat_str

    # Get vocab (token_id -> bytes) for decoding
    # NOTE: enc.token_byte_values() is NOT indexed by token ID!
    # We must invert _mergeable_ranks (bytes -> token_id) to get (token_id -> bytes)
    mergeable_ranks = enc._mergeable_ranks
    vocab = [None] * len(mergeable_ranks)
    for token_bytes, token_id in mergeable_ranks.items():
        vocab[token_id] = token_bytes.hex()
    assert all(v is not None for v in vocab), "Gap in vocab - missing token IDs"

    # Get encoder (bytes -> token_id) for encoding
    # This is the key for BPE: check if concatenated bytes exist
    encoder = {k.hex(): v for k, v in enc._mergeable_ranks.items()}

    # Special tokens
    special_tokens = {
        "<|bos|>": 65527,
        "<|user_start|>": 65528,
        "<|user_end|>": 65529,
        "<|assistant_start|>": 65530,
        "<|assistant_end|>": 65531,
        "<|python_start|>": 65532,
        "<|python_end|>": 65533,
        "<|output_start|>": 65534,
        "<|output_end|>": 65535,
    }

    # Export as JSON
    data = {
        "pattern": pattern,
        "vocab": vocab,
        "encoder": encoder,
        "special_tokens": special_tokens,
    }

    with open(output_path, 'w') as f:
        json.dump(data, f)

    print(f"Exported {len(vocab)} tokens to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.pkl> <output.json>", file=sys.stderr)
        sys.exit(1)
    export_tokenizer(sys.argv[1], sys.argv[2])

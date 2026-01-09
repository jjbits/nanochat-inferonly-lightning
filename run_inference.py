#!/usr/bin/env python3
"""Simple inference wrapper - handles tokenization, calls C++ binary."""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: pip install transformers")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to weights.bin")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer path")
    parser.add_argument("--binary", default="./build/nanochat", help="nanochat binary")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokens = tokenizer.encode(args.prompt)

    cmd = [
        args.binary, args.weights,
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--top_k", str(args.top_k),
        "--max_tokens", str(args.max_tokens),
    ]

    proc = subprocess.run(cmd, input=" ".join(map(str, tokens)), capture_output=True, text=True)

    if proc.returncode != 0:
        print(f"Error: {proc.stderr}", file=sys.stderr)
        sys.exit(1)

    output_tokens = [int(t) for t in proc.stdout.strip().split("\n") if t]
    print(args.prompt + tokenizer.decode(output_tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CORE Benchmark Evaluation for nanochat-inferonly-lightning.

This script evaluates the model on the CORE benchmark (including HellaSwag)
by interfacing with the C++ inference engine via subprocess.

Usage:
    python utils/eval_core.py --weights assets/weights.bin --tokenizer assets/tokenizer.pkl
"""

import os
import sys
import json
import yaml
import csv
import time
import random
import struct
import pickle
import zipfile
import tempfile
import shutil
import urllib.request
import subprocess
from pathlib import Path

# eval_bundle URL
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def get_cache_dir():
    """Get cache directory for eval bundle."""
    cache_dir = Path.home() / ".cache" / "nanochat-lightning"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def download_eval_bundle():
    """Download and extract eval_bundle if not already present."""
    cache_dir = get_cache_dir()
    eval_bundle_dir = cache_dir / "eval_bundle"

    if eval_bundle_dir.exists():
        print(f"eval_bundle already exists at {eval_bundle_dir}")
        return eval_bundle_dir

    print(f"Downloading eval_bundle from {EVAL_BUNDLE_URL}...")
    zip_path = cache_dir / "eval_bundle.zip"

    urllib.request.urlretrieve(EVAL_BUNDLE_URL, zip_path)
    print(f"Downloaded to {zip_path}")

    # Extract
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_dir = Path(tmpdir) / "eval_bundle"
        shutil.move(str(extracted_dir), str(eval_bundle_dir))

    # Clean up zip
    zip_path.unlink()
    print(f"Extracted eval_bundle to {eval_bundle_dir}")
    return eval_bundle_dir


class Tokenizer:
    """Wrapper around tiktoken for tokenization."""

    def __init__(self, pickle_path):
        import tiktoken
        with open(pickle_path, "rb") as f:
            self.enc = pickle.load(f)

    def encode(self, text):
        return self.enc.encode_ordinary(text)

    def decode(self, ids):
        return self.enc.decode(ids)


class ModelInterface:
    """Interface to C++ model via subprocess."""

    def __init__(self, binary_path, weights_path):
        self.binary_path = binary_path
        self.weights_path = weights_path
        self.proc = None
        self._start_process()

    def _start_process(self):
        """Start the C++ process in eval mode."""
        self.proc = subprocess.Popen(
            [self.binary_path, self.weights_path, "--eval"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        # Wait for ready message
        while True:
            line = self.proc.stderr.readline().decode()
            if "Ready for evaluation" in line:
                break
            if line:
                print(line.strip(), file=sys.stderr)

    def compute_loss(self, tokens, cont_start):
        """
        Compute average cross-entropy loss and argmax accuracy over continuation tokens.

        Args:
            tokens: list of token ids
            cont_start: index where continuation begins

        Returns:
            tuple: (avg_loss, num_correct, num_tokens)
        """
        seq_len = len(tokens)

        # Send: seq_len (int32) + cont_start (int32) + tokens (int32 array)
        self.proc.stdin.write(struct.pack('i', seq_len))
        self.proc.stdin.write(struct.pack('i', cont_start))
        self.proc.stdin.write(struct.pack(f'{seq_len}i', *tokens))
        self.proc.stdin.flush()

        # Receive: avg_loss (float32), num_correct (int32), num_tokens (int32)
        data = self.proc.stdout.read(12)
        if len(data) != 12:
            raise RuntimeError(f"Expected 12 bytes, got {len(data)}")

        avg_loss, num_correct, num_tokens = struct.unpack('fii', data)
        return avg_loss, num_correct, num_tokens

    def close(self):
        """Close the subprocess."""
        if self.proc:
            # Send seq_len=0 to signal exit
            self.proc.stdin.write(struct.pack('i', 0))
            self.proc.stdin.flush()
            self.proc.wait()
            self.proc = None

    def __del__(self):
        self.close()


def find_common_prefix_length(token_sequences):
    """Find the length of the common prefix across token sequences."""
    if not token_sequences:
        return 0
    min_len = min(len(seq) for seq in token_sequences)
    for i in range(min_len):
        token = token_sequences[0][i]
        if not all(seq[i] == token for seq in token_sequences):
            return i
    return min_len


def evaluate_multiple_choice_task(model, tokenizer, data, task_meta):
    """
    Evaluate a multiple choice task.

    For MC tasks, the context/query is the same but continuations differ.
    We compute the average loss over each continuation and pick the lowest.
    """
    correct = 0
    total = 0
    continuation_delimiter = task_meta.get('continuation_delimiter', ' ')

    for example in data:
        # MC tasks have: query (string), choices (list of strings), gold (int)
        query = example['query']
        choices = example['choices']
        gold = example['gold']

        # Tokenize all prompts to find common prefix
        all_tokens = []
        for choice in choices:
            prompt = query + continuation_delimiter + choice
            tokens = tokenizer.encode(prompt)
            all_tokens.append(tokens)

        # Find where the token sequences diverge
        cont_start = find_common_prefix_length(all_tokens)

        best_loss = float('inf')
        best_idx = -1

        for idx, tokens in enumerate(all_tokens):
            if cont_start >= len(tokens):
                continue

            # Compute loss over choice tokens
            avg_loss, _, _ = model.compute_loss(tokens, cont_start)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_idx = idx

        if best_idx == gold:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def evaluate_schema_task(model, tokenizer, data, task_meta):
    """
    Evaluate a schema-type task (like HellaSwag).

    For schema tasks, each example has multiple context options and one continuation.
    We compute the average loss over the continuation for each context,
    and predict the one with lowest loss.
    """
    correct = 0
    total = 0
    continuation_delimiter = task_meta.get('continuation_delimiter', ' ')

    for example in data:
        # Schema tasks have: context_options (list of strings), continuation (string), gold (int)
        context_options = example['context_options']
        continuation = example['continuation']
        gold = example['gold']

        best_loss = float('inf')
        best_idx = -1

        for idx, context in enumerate(context_options):
            # Build prompt: context + delimiter + continuation
            prompt = context + continuation_delimiter + continuation
            tokens = tokenizer.encode(prompt)

            # Tokenize just the context to find where continuation starts
            context_tokens = tokenizer.encode(context + continuation_delimiter)
            cont_start = len(context_tokens)

            if cont_start >= len(tokens):
                # Edge case: continuation is empty after tokenization
                continue

            # Compute loss over continuation tokens
            avg_loss, _, _ = model.compute_loss(tokens, cont_start)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_idx = idx

        if best_idx == gold:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def evaluate_language_modeling_task(model, tokenizer, data, task_meta):
    """
    Evaluate a language modeling task.

    For LM tasks, we check if the model's argmax predictions exactly match
    the continuation tokens. Each example has context and a single continuation.
    """
    correct = 0
    total = 0
    continuation_delimiter = task_meta.get('continuation_delimiter', ' ')

    for example in data:
        # LM tasks have: context (string), continuation (string)
        context = example['context']
        continuation = example['continuation']

        # Build prompt: context + delimiter + continuation
        # Strip context to avoid trailing whitespace issues
        prompt = context.strip() + continuation_delimiter + continuation
        tokens = tokenizer.encode(prompt)

        # Tokenize just the context to find where continuation starts
        context_tokens = tokenizer.encode(context.strip() + continuation_delimiter)
        cont_start = len(context_tokens)

        if cont_start >= len(tokens):
            continue

        # Check if all continuation tokens are correctly predicted
        _, num_correct, num_tokens = model.compute_loss(tokens, cont_start)

        if num_correct == num_tokens:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def evaluate_task(model, tokenizer, data, task_meta):
    """Dispatch to appropriate evaluation function based on task type."""
    task_type = task_meta['task_type']

    if task_type == 'multiple_choice':
        return evaluate_multiple_choice_task(model, tokenizer, data, task_meta)
    elif task_type == 'schema':
        return evaluate_schema_task(model, tokenizer, data, task_meta)
    elif task_type == 'language_modeling':
        return evaluate_language_modeling_task(model, tokenizer, data, task_meta)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='CORE Benchmark Evaluation')
    parser.add_argument('--weights', type=str, default='assets/weights.bin',
                        help='Path to weights file')
    parser.add_argument('--tokenizer', type=str, default='assets/tokenizer.pkl',
                        help='Path to tokenizer pickle file')
    parser.add_argument('--binary', type=str, default='build/nanochat',
                        help='Path to nanochat binary')
    parser.add_argument('--max-per-task', type=int, default=-1,
                        help='Max examples per task (-1 for all)')
    args = parser.parse_args()

    # Download eval bundle
    eval_bundle_dir = download_eval_bundle()

    # Load config
    config_path = eval_bundle_dir / "core.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # Load random baselines
    eval_meta_path = eval_bundle_dir / "eval_meta_data.csv"
    random_baselines = {}
    with open(eval_meta_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row['Eval Task']] = float(row['Random baseline'])

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = Tokenizer(args.tokenizer)

    # Start model
    print(f"Starting model from {args.binary} with weights {args.weights}...")
    model = ModelInterface(args.binary, args.weights)

    # Evaluate each task
    results = {}
    centered_results = {}

    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }

        print(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='', flush=True)

        # Load data
        data_path = eval_bundle_dir / "eval_data" / task_meta['dataset_uri']
        with open(data_path, 'r') as f:
            data = [json.loads(line.strip()) for line in f]

        # Shuffle and optionally limit
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if args.max_per_task > 0:
            data = data[:args.max_per_task]

        # Evaluate
        accuracy = evaluate_task(model, tokenizer, data, task_meta)

        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered

        elapsed = time.time() - start_time
        print(f"accuracy: {accuracy:.4f} | centered: {centered:.4f} | time: {elapsed:.2f}s")

    # Compute CORE metric
    core_metric = sum(centered_results.values()) / len(centered_results)

    print("\n" + "="*80)
    print("CORE Benchmark Results")
    print("="*80)
    print(f"{'Task':<35} {'Accuracy':<12} {'Centered':<12}")
    print("-"*60)
    for label in results:
        print(f"{label:<35} {results[label]:<12.4f} {centered_results[label]:<12.4f}")
    print("-"*60)
    print(f"{'CORE METRIC':<35} {'':<12} {core_metric:<12.4f}")
    print("="*80)
    print(f"\nNanochat reference CORE score: 0.2219")

    # Clean up
    model.close()

    return core_metric


if __name__ == '__main__':
    main()

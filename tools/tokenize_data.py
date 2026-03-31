#!/usr/bin/env python3
"""
Pre-tokenize a HuggingFace dataset into a flat uint16 binary file (tokens.bin).

Usage:
    python tools/tokenize_data.py
    python tools/tokenize_data.py --max_tokens_b 22.5 --num_proc 8
    python tools/tokenize_data.py --dataset_subset sample-100BT --out tokens.bin

    # Upload to R2 after tokenization (and delete local copy)
    python tools/tokenize_data.py --r2_bucket netra-data --r2_key data/tokens.bin
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    p = argparse.ArgumentParser(description="Pre-tokenize dataset to tokens.bin")
    p.add_argument("--tokenizer_path", type=str, default="tokenizer.json")
    p.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset_subset", type=str, default="sample-100BT")
    p.add_argument("--max_tokens_b", type=float, default=22.5,
                   help="Stop after this many billion tokens (0 = no limit)")
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--out", type=str, default="tokens.bin")
    p.add_argument("--streaming", action="store_true",
                   help="Use streaming mode (slower but needs less disk)")
    p.add_argument("--r2_bucket", type=str, default=os.environ.get("R2_BUCKET"),
                   help="Upload tokens.bin to R2 (default: $R2_BUCKET env var)")
    p.add_argument("--r2_key", type=str, default=None,
                   help="Remote key in R2 (default: same as --out filename)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output file")
    args = p.parse_args()

    from tools.r2 import check_connection
    check_connection()

    if os.path.exists(args.out) and not args.force:
        size_gb = os.path.getsize(args.out) / (1024**3)
        total_tokens = os.path.getsize(args.out) // 2
        print(f"{args.out} already exists ({total_tokens:,} tokens, {size_gb:.1f} GB) — skipping")
        print("Use --force to overwrite")
        return

    from tokenizers import Tokenizer
    from datasets import load_dataset

    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {args.tokenizer_path}. "
            "Train one first: python tools/train_tokenizer.py"
        )

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    eot_id = tokenizer.token_to_id("<|EOT|>")

    max_tokens = int(args.max_tokens_b * 1e9) if args.max_tokens_b > 0 else 0
    limit_str = f" (capped at {args.max_tokens_b:.1f}B tokens)" if max_tokens else ""
    print(f"Tokenizing {args.dataset_name}/{args.dataset_subset} → {args.out}{limit_str}")

    if args.streaming:
        _tokenize_streaming(tokenizer, eot_id, max_tokens, args)
    else:
        _tokenize_batch(tokenizer, eot_id, max_tokens, args)

    size_gb = os.path.getsize(args.out) / (1024**3)
    total_tokens = os.path.getsize(args.out) // 2
    print(f"Done: {total_tokens:,} tokens ({size_gb:.1f} GB) → {args.out}")

    if args.r2_bucket:
        from tools.r2 import upload
        key = args.r2_key or os.path.basename(args.out)
        print(f"Uploading to R2 ({size_gb:.1f} GB) …")
        upload(args.out, args.r2_bucket, key=key)
        print(f"Uploaded → r2://{args.r2_bucket}/{key}")


def _tokenize_batch(tokenizer, eot_id, max_tokens, args):
    """Download full dataset, then tokenize in parallel with HF .map()."""
    from datasets import load_dataset

    print(f"Loading {args.dataset_name}/{args.dataset_subset} "
          "(uses HF cache if previously downloaded) …")
    ds = load_dataset(args.dataset_name, args.dataset_subset, split="train")
    print(f"Loaded {len(ds):,} documents")

    def tokenize_batch(batch):
        all_ids = []
        encoded = tokenizer.encode_batch(batch["text"])
        for enc in encoded:
            ids = enc.ids
            if eot_id is not None:
                ids.append(eot_id)
            all_ids.extend(ids)
        return {"ids": [all_ids], "count": [len(all_ids)]}

    print(f"Tokenizing with {args.num_proc} workers …")
    tokenized = ds.map(
        tokenize_batch,
        batched=True,
        batch_size=1000,
        num_proc=args.num_proc,
        remove_columns=ds.column_names,
    )

    print("Writing tokens …")
    total_tokens = 0
    buf = np.empty(2**20, dtype=np.uint16)
    buf_pos = 0

    with open(args.out, "wb") as f:
        for i, row in enumerate(tokenized):
            for tok in row["ids"]:
                buf[buf_pos] = tok
                buf_pos += 1
                if buf_pos == len(buf):
                    f.write(buf.tobytes())
                    total_tokens += buf_pos
                    buf_pos = 0

            if max_tokens and (total_tokens + buf_pos) >= max_tokens:
                break

            if (i + 1) % 2_000_000 == 0:
                print(f"  {i+1:>12,} rows │ {(total_tokens + buf_pos) / 1e9:.2f}B tokens")

        if buf_pos > 0:
            f.write(buf[:buf_pos].tobytes())
            total_tokens += buf_pos

    if max_tokens and total_tokens >= max_tokens:
        print(f"Reached {args.max_tokens_b:.1f}B token cap — stopping early")


def _tokenize_streaming(tokenizer, eot_id, max_tokens, args):
    """Stream dataset and tokenize one document at a time (low disk usage)."""
    from datasets import load_dataset

    print("Streaming dataset …")
    ds = load_dataset(
        args.dataset_name, args.dataset_subset,
        split="train", streaming=True,
    )

    total_tokens = 0
    buf = np.empty(2**20, dtype=np.uint16)
    buf_pos = 0

    with open(args.out, "wb") as f:
        for i, row in enumerate(ds):
            ids = tokenizer.encode(row["text"]).ids
            if eot_id is not None:
                ids.append(eot_id)
            for t in ids:
                buf[buf_pos] = t
                buf_pos += 1
                if buf_pos == len(buf):
                    f.write(buf.tobytes())
                    total_tokens += buf_pos
                    buf_pos = 0

            if max_tokens and (total_tokens + buf_pos) >= max_tokens:
                break

            if (i + 1) % 500_000 == 0:
                print(f"  {i+1:>12,} docs │ {(total_tokens + buf_pos) / 1e9:.2f}B tokens")

        if buf_pos > 0:
            f.write(buf[:buf_pos].tobytes())
            total_tokens += buf_pos

    if max_tokens and total_tokens >= max_tokens:
        print(f"Reached {args.max_tokens_b:.1f}B token cap — stopping early")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train a BPE tokenizer on FineWeb data and save to disk.

Usage:
    python tools/train_tokenizer.py
    python tools/train_tokenizer.py --num_samples 1000000 --save_path tokenizer.json

    # Upload to R2 after training
    python tools/train_tokenizer.py --r2_bucket netra-data
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from netra import NetraTokenizer


def main():
    p = argparse.ArgumentParser(description="Train Netra BPE tokenizer")
    p.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset_subset", type=str, default="sample-10BT")
    p.add_argument("--vocab_size", type=int, default=32_000)
    p.add_argument("--num_samples", type=int, default=500_000)
    p.add_argument("--save_path", type=str, default="tokenizer.json")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing tokenizer")
    p.add_argument("--r2_bucket", type=str, default=os.environ.get("R2_BUCKET"),
                   help="Upload tokenizer to R2 (default: $R2_BUCKET env var)")
    p.add_argument("--r2_key", type=str, default=None,
                   help="Remote key in R2 (default: same as --save_path filename)")
    args = p.parse_args()

    from tools.r2 import check_connection
    check_connection()

    if Path(args.save_path).exists() and not args.force:
        print(f"Tokenizer already exists at {args.save_path} — skipping")
        print("Use --force to overwrite")
        return

    print(f"Training tokenizer: vocab_size={args.vocab_size}, "
          f"num_samples={args.num_samples:,}")
    print(f"Dataset: {args.dataset_name}/{args.dataset_subset}")

    ds = load_dataset(
        args.dataset_name, args.dataset_subset,
        split="train", streaming=True,
    )

    NetraTokenizer.train(
        ds,
        vocab_size=args.vocab_size,
        num_samples=args.num_samples,
        save_path=args.save_path,
    )
    print(f"Tokenizer saved → {args.save_path}")

    if args.r2_bucket:
        from tools.r2 import upload
        key = args.r2_key or Path(args.save_path).name
        upload(args.save_path, args.r2_bucket, key=key)
        print(f"Uploaded → r2://{args.r2_bucket}/{key}")


if __name__ == "__main__":
    main()

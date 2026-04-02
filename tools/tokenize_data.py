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

    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {args.tokenizer_path}. "
            "Train one first: python tools/train_tokenizer.py"
        )

    from netra import NetraTokenizer
    tokenizer = NetraTokenizer.from_file(args.tokenizer_path)

    use_u32 = tokenizer.vocab_size > 65536
    typecode = "I" if use_u32 else "H"
    bytes_per_token = 4 if use_u32 else 2
    dtype_name = "uint32" if use_u32 else "uint16"
    print(f"Vocab size: {tokenizer.vocab_size:,} → {dtype_name}")

    if os.path.exists(args.out) and not args.force:
        size_gb = os.path.getsize(args.out) / (1024**3)
        total_tokens = os.path.getsize(args.out) // bytes_per_token
        print(f"{args.out} already exists ({total_tokens:,} tokens, {size_gb:.1f} GB) — skipping")
        print("Use --force to overwrite")
        return

    max_tokens = int(args.max_tokens_b * 1e9) if args.max_tokens_b > 0 else 0
    limit_str = f" (capped at {args.max_tokens_b:.1f}B tokens)" if max_tokens else ""
    print(f"Tokenizing {args.dataset_name}/{args.dataset_subset} → {args.out}{limit_str}")

    if args.streaming:
        _tokenize_streaming(tokenizer, max_tokens, args, typecode)
    else:
        _tokenize_batch(tokenizer, max_tokens, args, typecode)

    size_gb = os.path.getsize(args.out) / (1024**3)
    total_tokens = os.path.getsize(args.out) // bytes_per_token
    print(f"Done: {total_tokens:,} tokens ({size_gb:.1f} GB) → {args.out}")

    if args.r2_bucket:
        from tools.r2 import upload
        key = args.r2_key or os.path.basename(args.out)
        print(f"Uploading to R2 ({size_gb:.1f} GB) …")
        upload(args.out, args.r2_bucket, key=key)
        print(f"Uploaded → r2://{args.r2_bucket}/{key}")


def _tokenize_worker(worker_args):
    """Worker: tokenize a subset of parquet files into a shard file."""
    import array
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    worker_id, parquet_files, tokenizer_path, eot_id, max_tok, out_path, typecode = worker_args
    raw_tok = Tokenizer.from_file(tokenizer_path)
    BATCH_SIZE = 5000
    eot = [eot_id] if eot_id is not None else []

    total_tokens = 0
    total_docs = 0
    done = False

    with open(out_path, "wb") as f:
        for pf in parquet_files:
            table = pq.read_table(pf, columns=["text"])
            texts = table.column("text").to_pylist()
            del table

            for start in range(0, len(texts), BATCH_SIZE):
                batch = texts[start : start + BATCH_SIZE]
                encoded = raw_tok.encode_batch(batch)
                total_docs += len(batch)

                buf = array.array(typecode)
                for enc in encoded:
                    buf.extend(enc.ids)
                    buf.extend(eot)

                f.write(buf.tobytes())
                total_tokens += len(buf)

                if max_tok and total_tokens >= max_tok:
                    done = True
                    break

            if done:
                break

    return worker_id, total_tokens, total_docs, out_path


def _tokenize_batch(tokenizer, max_tokens, args, typecode="H"):
    """Read parquet files directly, tokenize in parallel with num_proc workers."""
    import time
    import glob
    from multiprocessing import Pool
    from huggingface_hub import snapshot_download

    subset_path = args.dataset_subset.replace("-", "/")
    print(f"Locating parquet files (uses HF cache) …")
    local_dir = snapshot_download(
        repo_id=args.dataset_name,
        repo_type="dataset",
        allow_patterns=f"{subset_path}/*.parquet",
    )

    parquet_files = sorted(glob.glob(os.path.join(local_dir, subset_path, "*.parquet")))
    if not parquet_files:
        parquet_files = sorted(glob.glob(os.path.join(local_dir, "**", "*.parquet"), recursive=True))
    print(f"Found {len(parquet_files)} parquet files")

    num_workers = args.num_proc
    eot_id = tokenizer.eot_id
    max_per_worker = int(max_tokens * 1.2 / num_workers) if max_tokens else 0

    chunks = [[] for _ in range(num_workers)]
    for i, pf in enumerate(parquet_files):
        chunks[i % num_workers].append(pf)

    out_dir = os.path.dirname(args.out) or "."
    base = os.path.splitext(os.path.basename(args.out))[0]
    worker_args = []
    for wid in range(num_workers):
        shard_path = os.path.join(out_dir, f"{base}_{wid:05d}.bin")
        worker_args.append((
            wid, chunks[wid], args.tokenizer_path, eot_id,
            max_per_worker, shard_path, typecode,
        ))

    t0 = time.time()
    print(f"Tokenizing with {num_workers} workers …")

    with Pool(num_workers) as pool:
        results = pool.map(_tokenize_worker, worker_args)

    total_tokens = 0
    total_docs = 0
    shard_paths = []
    for wid, toks, docs, shard_path in sorted(results):
        total_tokens += toks
        total_docs += docs
        shard_paths.append(shard_path)
        print(f"  worker {wid}: {toks/1e9:.2f}B tokens, {docs:,} docs")

    print(f"Merging {len(shard_paths)} shards → {args.out} …")
    with open(args.out, "wb") as fout:
        for sp in shard_paths:
            with open(sp, "rb") as fin:
                while True:
                    chunk = fin.read(64 * 1024 * 1024)
                    if not chunk:
                        break
                    fout.write(chunk)
            os.remove(sp)

    bytes_per_token = 4 if typecode == "I" else 2
    if max_tokens and total_tokens > max_tokens:
        print(f"Trimming to {max_tokens/1e9:.1f}B tokens …")
        with open(args.out, "r+b") as f:
            f.truncate(max_tokens * bytes_per_token)
        total_tokens = max_tokens

    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(f"Tokenization done in {h}h{m:02d}m{s:02d}s ({total_docs:,} docs, {total_tokens/1e9:.2f}B tokens)")


def _prefetch_batches(ds, batch_size, queue):
    """Background thread: pull batches of texts from the stream into a queue."""
    from itertools import islice
    it = iter(ds)
    while True:
        batch = [row["text"] for row in islice(it, batch_size)]
        if not batch:
            queue.put(None)
            break
        queue.put(batch)


def _tokenize_streaming(tokenizer, max_tokens, args, typecode="H"):
    """Stream dataset with prefetch + encode_batch (low disk, high throughput)."""
    import array
    import time
    from queue import Queue
    from threading import Thread
    from datasets import load_dataset

    BATCH_SIZE = 2000
    PREFETCH = 4

    print("Streaming dataset …")
    ds = load_dataset(
        args.dataset_name, args.dataset_subset,
        split="train", streaming=True,
    )

    q = Queue(maxsize=PREFETCH)
    fetcher = Thread(target=_prefetch_batches, args=(ds, BATCH_SIZE, q), daemon=True)
    fetcher.start()

    raw_tok = tokenizer._tok
    eot_id = tokenizer.eot_id
    eot = [eot_id] if eot_id is not None else []
    total_tokens = 0
    total_docs = 0
    t0 = time.time()
    last_print = t0
    done = False

    with open(args.out, "wb") as f:
        while not done:
            texts = q.get()
            if texts is None:
                break

            encoded = raw_tok.encode_batch(texts)
            total_docs += len(texts)

            buf = array.array(typecode)
            for enc in encoded:
                buf.extend(enc.ids)
                buf.extend(eot)

            f.write(buf.tobytes())
            total_tokens += len(buf)

            if max_tokens and total_tokens >= max_tokens:
                done = True

            now = time.time()
            if now - last_print >= 10:
                elapsed = now - t0
                tok_per_sec = total_tokens / elapsed
                if max_tokens:
                    pct = total_tokens / max_tokens * 100
                    eta = (max_tokens - total_tokens) / tok_per_sec if tok_per_sec > 0 else 0
                    eta_m, eta_s = divmod(int(eta), 60)
                    eta_h, eta_m = divmod(eta_m, 60)
                    print(f"  {total_docs:>10,} docs │ {total_tokens/1e9:.2f}B/{max_tokens/1e9:.1f}B tokens "
                          f"({pct:.1f}%) │ {tok_per_sec/1e6:.1f}M tok/s │ ETA {eta_h}h{eta_m:02d}m{eta_s:02d}s")
                else:
                    print(f"  {total_docs:>10,} docs │ {total_tokens/1e9:.2f}B tokens │ {tok_per_sec/1e6:.1f}M tok/s")
                last_print = now

    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(f"Streaming done in {h}h{m:02d}m{s:02d}s")

    if max_tokens and total_tokens >= max_tokens:
        print(f"Reached {args.max_tokens_b:.1f}B token cap — stopping early")


if __name__ == "__main__":
    main()

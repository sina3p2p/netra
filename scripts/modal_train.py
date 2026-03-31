"""
Run Netra training on Modal with cloud GPU(s).

Usage:
    # 1. Train tokenizer first (only once):
    modal run scripts/modal_train.py::train_tokenizer

    # 2. Pre-tokenize dataset (only once, ~30-60 min):
    modal run scripts/modal_train.py::tokenize_dataset

    # 3. Train the model:
    modal run scripts/modal_train.py --config medium

    # 4. Multi-GPU training:
    NETRA_GPUS=4 modal run scripts/modal_train.py --config medium

    # 5. Run in background:
    modal run --detach scripts/modal_train.py --config medium
"""

import os
import modal

app = modal.App("netra")

NUM_GPUS = int(os.environ.get("NETRA_GPUS", "1"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("netra", remote_path="/root/netra/netra")
    .add_local_dir("configs", remote_path="/root/netra/configs")
    .add_local_dir("tools", remote_path="/root/netra/tools")
    .add_local_file("train.py", remote_path="/root/netra/train.py")
)

volume = modal.Volume.from_name("netra-data", create_if_missing=True)
VOLUME_PATH = "/data"

secrets = [
    modal.Secret.from_name("wandb-secret"),
    modal.Secret.from_name("huggingface-secret"),
]


@app.function(
    image=image,
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    secrets=secrets,
    timeout=3600,
)
def train_tokenizer(num_samples: int = 500_000):
    """Train BPE tokenizer and persist to Modal volume."""
    import sys
    sys.path.insert(0, "/root/netra")

    from datasets import load_dataset
    from netra import NetraTokenizer

    save_path = f"{VOLUME_PATH}/tokenizer.json"
    ds = load_dataset(
        "HuggingFaceFW/fineweb", "sample-10BT",
        split="train", streaming=True,
    )
    NetraTokenizer.train(ds, vocab_size=32_000, num_samples=num_samples,
                         save_path=save_path)
    volume.commit()
    print(f"Tokenizer saved → {save_path}")


@app.function(
    image=image,
    cpu=8,
    memory=131072,
    ephemeral_disk=1000_000,
    volumes={VOLUME_PATH: volume},
    secrets=secrets,
    timeout=12 * 3600,
)
def tokenize_dataset(
    dataset_name: str = "HuggingFaceFW/fineweb",
    dataset_subset: str = "sample-100BT",
    max_tokens_b: float = 22.5,
    num_proc: int = 8,
):
    """Pre-tokenize the dataset to a flat uint16 binary file on the Modal volume."""
    import sys
    import numpy as np
    sys.path.insert(0, "/root/netra")

    os.environ["HF_HOME"] = "/tmp/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache/datasets"

    from datasets import load_dataset
    from tokenizers import Tokenizer

    tok_path = f"{VOLUME_PATH}/tokenizer.json"
    if not os.path.exists(tok_path):
        raise FileNotFoundError(
            "No tokenizer found. Run first:\n"
            "  modal run scripts/modal_train.py::train_tokenizer"
        )
    tokenizer = Tokenizer.from_file(tok_path)
    eot_id = tokenizer.token_to_id("<|EOT|>")

    max_tokens = int(max_tokens_b * 1e9) if max_tokens_b > 0 else 0

    out_path = f"{VOLUME_PATH}/tokens.bin"
    limit_str = f" (capped at {max_tokens_b:.0f}B tokens)" if max_tokens else ""
    print(f"Tokenizing {dataset_name}/{dataset_subset} → {out_path}{limit_str}")
    print("Downloading dataset (non-streaming) …")

    ds = load_dataset(dataset_name, dataset_subset, split="train")
    print(f"Downloaded {len(ds):,} documents")

    def tokenize_batch(batch):
        all_ids = []
        encoded = tokenizer.encode_batch(batch["text"])
        for enc in encoded:
            ids = enc.ids
            if eot_id is not None:
                ids.append(eot_id)
            all_ids.extend(ids)
        return {"ids": [all_ids], "count": [len(all_ids)]}

    print(f"Tokenizing with {num_proc} workers …")
    tokenized = ds.map(
        tokenize_batch,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=ds.column_names,
    )

    print("Writing tokens.bin …")
    total_tokens = 0
    buf = np.empty(2**20, dtype=np.uint16)
    buf_pos = 0
    stopped_early = False

    with open(out_path, "wb") as f:
        for i, row in enumerate(tokenized):
            for tok in row["ids"]:
                buf[buf_pos] = tok
                buf_pos += 1
                if buf_pos == len(buf):
                    f.write(buf.tobytes())
                    total_tokens += buf_pos
                    buf_pos = 0

            if max_tokens and (total_tokens + buf_pos) >= max_tokens:
                stopped_early = True
                break

            if (i + 1) % 2_000_000 == 0:
                print(f"  {i+1:>12,} rows │ {(total_tokens + buf_pos) / 1e9:.2f}B tokens")

        if buf_pos > 0:
            f.write(buf[:buf_pos].tobytes())
            total_tokens += buf_pos

    if stopped_early:
        print(f"Reached {max_tokens_b:.0f}B token cap — stopping early")

    size_gb = os.path.getsize(out_path) / (1024**3)
    print(f"Done: {total_tokens:,} tokens ({size_gb:.1f} GB)")
    volume.commit()


@app.function(
    image=image,
    gpu=f"H100:{NUM_GPUS}" if NUM_GPUS > 1 else "H100",
    volumes={VOLUME_PATH: volume},
    secrets=secrets,
    timeout=24 * 3600,
)
def train(config: str = "nano", extra_args: str = ""):
    """Run Netra pretraining on cloud GPU(s). Uses torchrun for multi-GPU."""
    import subprocess, shutil, torch

    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
    os.chdir("/root/netra")

    tok_vol = f"{VOLUME_PATH}/tokenizer.json"
    tok_local = "/root/netra/tokenizer.json"
    if os.path.exists(tok_vol):
        shutil.copy(tok_vol, tok_local)
        print("Loaded tokenizer from volume")
    else:
        raise FileNotFoundError(
            "No tokenizer found. Run first:\n"
            "  modal run scripts/modal_train.py::train_tokenizer"
        )

    ckpt_dir = f"{VOLUME_PATH}/checkpoints/{config}"
    os.makedirs(ckpt_dir, exist_ok=True)

    data_path = f"{VOLUME_PATH}/tokens.bin"
    config_path = f"configs/{config}.yaml"

    train_args = [
        "--config", config_path,
        "--tokenizer_path", tok_local,
        "--checkpoint_dir", ckpt_dir,
        "--compile",
    ]
    if os.path.exists(data_path):
        train_args.extend(["--data_path", data_path])
        print(f"Using pre-tokenized data: {data_path}")
    else:
        print("No pre-tokenized data found, falling back to streaming")

    if extra_args:
        train_args.extend(extra_args.split())

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        cmd = [
            "torchrun",
            "--nproc_per_node", str(num_gpus),
            "--standalone",
            "train.py",
        ] + train_args
    else:
        cmd = ["python", "train.py"] + train_args

    print(f"GPUs available: {num_gpus}")
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    volume.commit()
    print(f"Checkpoints saved → {ckpt_dir}")


@app.local_entrypoint()
def main(config: str = "nano", extra: str = ""):
    print(f"Launching Netra training: {config} on {NUM_GPUS}x H100")
    print("To run in background: nohup modal run scripts/modal_train.py ... > training.log 2>&1 &")
    train.remote(config=config, extra_args=extra)
    print("Training complete!")

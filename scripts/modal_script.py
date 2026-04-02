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
    modal.Secret.from_name("custom-secret"),
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
    import subprocess
    os.chdir("/root/netra")

    save_path = f"{VOLUME_PATH}/tokenizer.json"
    cmd = [
        "python", "-u", "tools/train_tokenizer.py",
        "--save_path", save_path,
        "--num_samples", str(num_samples),
        "--force",
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    volume.commit()
    print(f"Tokenizer saved → {save_path}")


@app.function(
    image=image,
    cpu=8,
    memory=131072,
    ephemeral_disk=1_000_000,
    volumes={VOLUME_PATH: volume},
    secrets=secrets,
    timeout=12 * 3600,
)
def tokenize_dataset(
    dataset_name: str = "HuggingFaceFW/fineweb",
    dataset_subset: str = "sample-100BT",
    max_tokens_b: float = 22.5,
    num_proc: int = 8,
    streaming: bool = False,
):
    """Pre-tokenize the dataset following Modal best practices:

    1. Download dataset → /tmp/ (fast local SSD)
    2. Tokenize → /tmp/tokens.bin (local I/O)
    3. Upload tokens.bin → R2 (via env var, zero egress fees)
    4. Copy tokens.bin → volume (for Modal training)
    """
    import subprocess, shutil
    os.chdir("/root/netra")

    os.environ["HF_HOME"] = "/tmp/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache/datasets"

    tok_path = f"{VOLUME_PATH}/tokenizer.json"
    if not os.path.exists(tok_path):
        raise FileNotFoundError(
            "No tokenizer found. Run first:\n"
            "  modal run scripts/modal_script.py::train_tokenizer"
        )

    tmp_out = "/tmp/tokens.bin"
    cmd = [
        "python", "-u", "tools/tokenize_data.py",
        "--tokenizer_path", tok_path,
        "--dataset_name", dataset_name,
        "--dataset_subset", dataset_subset,
        "--max_tokens_b", str(max_tokens_b),
        "--num_proc", str(num_proc),
        "--out", tmp_out,
        "--r2_key", "tokens.bin",
        "--force",
    ]
    if streaming:
        cmd.append("--streaming")
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    vol_out = f"{VOLUME_PATH}/tokens.bin"
    size_gb = os.path.getsize(tmp_out) / (1024**3)
    print(f"Copying tokens.bin ({size_gb:.1f} GB) → volume …")
    shutil.copy2(tmp_out, vol_out)
    os.remove(tmp_out)
    volume.commit()
    print(f"Done → {vol_out}")


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
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "No pre-tokenized data found. Run first:\n"
            "  modal run scripts/modal_train.py::tokenize_dataset"
        )
    train_args.extend(["--data_path", data_path])
    print(f"Using pre-tokenized data: {data_path}")

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

"""
Run Netra training on Modal with cloud GPU(s).

Usage:
    # 1. Train tokenizer first (only once):
    modal run scripts/modal_train.py::train_tokenizer

    # 2. Pre-tokenize dataset (only once, ~30-60 min):
    modal run scripts/modal_train.py::tokenize_dataset

    # 3. Train the model:
    modal run scripts/modal_script.py -- --config medium

    # 4. Multi-GPU training:
    NETRA_GPUS=4 modal run scripts/modal_script.py -- --config medium

    # 5. Pass extra train.py args:
    modal run scripts/modal_script.py -- --config medium --lr 3e-4

    # 6. Run in background:
    modal run --detach scripts/modal_script.py -- --config medium
"""

import os
import modal

app = modal.App("netra")

NUM_GPUS = int(os.environ.get("NETRA_GPUS", "1"))
GPU_TYPE = "H100"

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

secrets = [modal.Secret.from_dotenv()]

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
    gpu=f"{GPU_TYPE}:{NUM_GPUS}" if NUM_GPUS > 1 else GPU_TYPE,
    volumes={VOLUME_PATH: volume},
    secrets=secrets,
    timeout=24 * 3600,
)
def train(*arglist):
    """Run Netra pretraining on cloud GPU(s). Uses torchrun for multi-GPU.

    Accepts any train.py CLI args directly, e.g.:
        modal run scripts/modal_script.py::train -- --config medium
        modal run scripts/modal_script.py::train -- --config medium --lr 3e-4
    """
    import argparse, subprocess, torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="nano")
    args, extra = parser.parse_known_args(arglist)

    config_name = args.config
    config_path = f"configs/{config_name}.yaml"
    num_gpus = torch.cuda.device_count()
    tok_path = f"{VOLUME_PATH}/tokenizer.json"
    data_path = f"{VOLUME_PATH}/tokens.bin"
    ckpt_dir = f"{VOLUME_PATH}/checkpoints/{config_name}"

    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
    os.chdir("/root/netra")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Launching Netra training: {config_name} on {num_gpus}x {GPU_TYPE}")

    # Resolve tokenizer and data: volume → R2 → error
    need_tok = not os.path.exists(tok_path)
    need_data = not os.path.exists(data_path)

    if need_tok or need_data:
        from tools.r2 import is_configured, download
        if not is_configured():
            raise RuntimeError("R2 is required to download missing files. Set R2 env vars.")
        bucket = os.environ["R2_BUCKET"]
        if need_tok:
            print(f"tokenizer.json not on volume — downloading from R2 ({bucket}) …")
            download("tokenizer.json", bucket, local_path=tok_path)
        if need_data:
            print(f"tokens.bin not on volume — downloading from R2 ({bucket}) …")
            download("tokens.bin", bucket, local_path=data_path)
        volume.commit()

    # Build command
    train_args = [
        "--config", config_path,
        "--tokenizer_path", tok_path,
        "--checkpoint_dir", ckpt_dir,
        "--compile",
        "--data_path", data_path,
    ] + list(extra)

    if num_gpus > 1:
        cmd = [
            "torchrun",
            "--nproc_per_node", str(num_gpus),
            "--standalone",
            "train.py",
        ] + train_args
    else:
        cmd = ["python", "train.py"] + train_args

    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    volume.commit()
    print(f"Checkpoints saved → {ckpt_dir}")


@app.local_entrypoint()
def main(*arglist):
    train.remote(*arglist)
    print("Training complete!")

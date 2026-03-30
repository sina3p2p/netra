"""
Run Netra training on Modal with a cloud GPU.

Usage:
    # 1. Train tokenizer first (only once):
    modal run train_modal.py::train_tokenizer

    # 2. Train the model:
    modal run train_modal.py                                     # nano (default)
    modal run train_modal.py --model-size mini
    modal run train_modal.py --model-size small
    modal run train_modal.py --model-size full

    # Pass extra flags to train.py:
    modal run train_modal.py --model-size nano --extra "--max_steps 500 --compile"
"""

import modal

app = modal.App("netra")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("netra", remote_path="/root/netra/netra")
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
    gpu="H100",
    volumes={VOLUME_PATH: volume},
    secrets=secrets,
    timeout=24 * 3600,
)
def train(model_size: str = "nano", extra_args: str = ""):
    """Run Netra pretraining on a cloud GPU."""
    import subprocess, shutil, os

    os.chdir("/root/netra")

    tok_vol = f"{VOLUME_PATH}/tokenizer.json"
    tok_local = "/root/netra/tokenizer.json"
    if os.path.exists(tok_vol):
        shutil.copy(tok_vol, tok_local)
        print("Loaded tokenizer from volume")
    else:
        raise FileNotFoundError(
            "No tokenizer found. Run first:\n"
            "  modal run train_modal.py::train_tokenizer"
        )

    ckpt_dir = f"{VOLUME_PATH}/checkpoints/{model_size}"
    os.makedirs(ckpt_dir, exist_ok=True)

    cmd = [
        "python", "train.py",
        "--model_size", model_size,
        "--tokenizer_path", tok_local,
        "--checkpoint_dir", ckpt_dir,
        "--compile"
    ]
    if extra_args:
        cmd.extend(extra_args.split())

    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    volume.commit()
    print(f"Checkpoints saved → {ckpt_dir}")


@app.local_entrypoint()
def main(model_size: str = "nano", extra: str = ""):
    print(f"Launching Netra training: {model_size}")
    train.remote(model_size=model_size, extra_args=extra)
    print("Training complete!")

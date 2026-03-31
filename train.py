#!/usr/bin/env python3
"""
Netra pretraining script with wandb monitoring.

Supports single-GPU and multi-GPU (DDP) training.

Usage:
    # Single GPU
    python train.py --config configs/nano.yaml

    # Multi-GPU (4 GPUs on one machine)
    torchrun --nproc_per_node=4 train.py --config configs/small.yaml

    # Multi-GPU with R2 backup
    torchrun --nproc_per_node=8 train.py --config configs/medium.yaml \\
        --data_path tokens.bin --r2_bucket netra-checkpoints
"""

import argparse
import inspect
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import wandb
from datasets import load_dataset

from netra import ModelConfig, Netra, NetraTokenizer
from netra.data import MemmapTokenDataset, StreamingTokenDataset

WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
BETA1 = 0.9
BETA2 = 0.95
NUM_WORKERS = 4
GENERATE_MAX_TOKENS = 100
DTYPE = "bfloat16"
DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_SUBSET = "sample-10BT"

# ── DDP helpers ───────────────────────────────────────────────────────


def setup_ddp():
    """Detect torchrun env vars and init process group. Returns (rank, local_rank, world_size)."""
    if "RANK" not in os.environ:
        return 0, 0, 1

    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ── Helpers ───────────────────────────────────────────────────────────


def resolve_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[name]


def cosine_lr(step: int, max_lr: float, min_lr: float,
              warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_raw_model(model):
    """Unwrap DDP / torch.compile to get the underlying Netra module."""
    m = model
    if isinstance(m, DDP):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def collect_moe_metrics(model) -> dict:
    raw = get_raw_model(model)
    metrics = {}
    for i, layer in enumerate(raw.layers):
        moe = layer.moe
        if not hasattr(moe, "_tokens_per_expert"):
            continue
        tpe = moe._tokens_per_expert
        total = tpe.sum().item()
        if total == 0:
            continue

        utilization = tpe / total
        metrics[f"moe/layer_{i}/utilization_std"] = utilization.std().item()
        metrics[f"moe/layer_{i}/max_load_ratio"] = (tpe.max() / tpe.mean()).item()

        for e in range(moe.n_experts):
            metrics[f"moe/layer_{i}/expert_{e}_frac"] = (tpe[e] / total).item()
            metrics[f"moe/layer_{i}/bias_{e}"] = moe.expert_bias[e].item()
    return metrics


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_tokens: int,
             temperature: float = 0.8, device: torch.device = torch.device("cpu")):
    raw = get_raw_model(model)
    raw.eval()
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_tokens):
        inp = ids[:, -raw.config.max_seq_len :]
        logits, _ = raw(inp)
        next_logits = logits[:, -1, :] / max(temperature, 1e-8)
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == tokenizer.eot_id:
            break
    raw.train()
    return tokenizer.decode(ids[0].tolist())


@torch.no_grad()
def evaluate(model, eval_batches, device, ctx):
    raw = get_raw_model(model)
    raw.eval()
    total_loss = 0.0
    for x, y in eval_batches:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with ctx:
            _, loss = raw(x, targets=y)
        total_loss += loss.item()
    raw.train()
    return total_loss / max(len(eval_batches), 1)


def save_checkpoint(model, optimizer, step, model_config, path,
                    r2_bucket=None, keep_local=True):
    raw = get_raw_model(model)
    torch.save({
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "model_config": asdict(model_config),
    }, path)

    if r2_bucket:
        try:
            import boto3
            s3 = boto3.client("s3",
                endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
            )
            key = f"checkpoints/{Path(path).name}"
            s3.upload_file(str(path), r2_bucket, key)
            print(f"  ↳ uploaded to r2://{r2_bucket}/{key}")
            if not keep_local:
                Path(path).unlink()
                print(f"  ↳ deleted local {path}")
        except Exception as e:
            print(f"  ⚠ R2 upload failed: {e} (local copy kept)")


# ── Config & argument parsing ─────────────────────────────────────────


def load_config(path: str) -> dict:
    """Load a YAML config file."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Netra pretraining")

    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config (e.g. configs/medium.yaml)")
    p.add_argument("--tokenizer_path", type=str, default="tokenizer.json")
    p.add_argument("--data_path", type=str, default=None,
                   help="Path to pre-tokenized tokens.bin (skips streaming)")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--r2_bucket", type=str, default=None,
                   help="Cloudflare R2 bucket for checkpoint backup")
    p.add_argument("--keep_local_ckpt", action="store_true", default=False,
                   help="Keep local checkpoint after R2 upload (default: delete)")

    args = p.parse_args()

    cfg = load_config(args.config)
    args._model_cfg = cfg.get("model", {})

    t = cfg.get("training", {})
    args.batch_size = t["batch_size"]
    args.grad_accum_steps = t["grad_accum_steps"]
    args.max_lr = t["max_lr"]
    args.min_lr_ratio = t.get("min_lr_ratio", 0.1)
    args.warmup_steps = t["warmup_steps"]
    args.max_steps = t["max_steps"]
    e = cfg.get("eval", {})
    args.eval_interval = e.get("eval_interval", 500)
    args.eval_steps = e.get("eval_steps", 20)
    args.generate_interval = e.get("generate_interval", 500)

    log = cfg.get("logging", {})
    args.log_interval = log.get("log_interval", 10)
    args.save_interval = log.get("save_interval", 1000)

    return args


# ── Main ──────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    ddp = world_size > 1

    if world_size > 1:
        args.max_steps = max(args.max_steps // world_size, 1)
        args.warmup_steps = max(args.warmup_steps // world_size, 1)
        if is_main:
            print(f"DDP: {world_size} GPUs → scaled max_steps to {args.max_steps}, "
                  f"warmup to {args.warmup_steps}")

    if ddp:
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype = resolve_dtype(DTYPE)
    min_lr = args.max_lr * args.min_lr_ratio

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    use_amp = device.type == "cuda"
    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler(enabled=(use_amp and dtype == torch.float16))

    # ── Tokenizer ─────────────────────────────────────────────────────

    tok_path = Path(args.tokenizer_path)
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tok_path}. "
            "Train one first: python tools/train_tokenizer.py"
        )
    tokenizer = NetraTokenizer.from_file(tok_path)
    if is_main:
        print(f"Tokenizer loaded — vocab size: {tokenizer.vocab_size:,}")

    # ── Model ─────────────────────────────────────────────────────────

    model_cfg_dict = dict(args._model_cfg, vocab_size=tokenizer.vocab_size)
    model_config = ModelConfig(**model_cfg_dict)
    model = Netra(model_config).to(device)
    n_params = count_parameters(model)
    config_name = Path(args.config).stem
    if is_main:
        print(f"Model: {config_name} | {n_params:,} params | "
              f"device: {device} | world_size: {world_size}")

    # ── Resume (before DDP wrap) ──────────────────────────────────────

    start_step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
        if is_main:
            print(f"Resumed model from step {start_step}")

    # ── Compile then DDP wrap ─────────────────────────────────────────

    if args.compile and hasattr(torch, "compile"):
        if is_main:
            print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ── Optimizer ─────────────────────────────────────────────────────

    raw_model = get_raw_model(model)
    decay, no_decay = [], []
    for name, param in raw_model.named_parameters():
        if param.dim() < 2 or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    use_fused = device.type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.max_lr, betas=(BETA1, BETA2), fused=use_fused)

    if args.resume_from and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        if is_main:
            print(f"Resumed optimizer from step {start_step}")

    # ── Data ──────────────────────────────────────────────────────────

    use_pinmem = device.type == "cuda"
    persist = NUM_WORKERS > 0

    if args.data_path and os.path.exists(args.data_path):
        if is_main:
            print(f"Using pre-tokenized data: {args.data_path}")
        import numpy as np
        total_tokens = os.path.getsize(args.data_path) // 2  # uint16
        eval_tokens = args.eval_steps * args.batch_size * (model_config.max_seq_len + 1)
        train_end = total_tokens - eval_tokens

        train_dataset = MemmapTokenDataset(
            args.data_path, seq_len=model_config.max_seq_len,
            rank=rank, world_size=world_size,
            start=0, end=train_end, shuffle=True, seed=0,
        )
        eval_dataset = MemmapTokenDataset(
            args.data_path, seq_len=model_config.max_seq_len,
            start=train_end, end=total_tokens,
        )
        if is_main:
            print(f"  train: {train_end:,} tokens │ eval: {eval_tokens:,} tokens")
    else:
        if is_main:
            print("Initialising data streams …")

        train_ds = load_dataset(
            DATASET_NAME, DATASET_SUBSET,
            split="train", streaming=True,
        ).shuffle(seed=0, buffer_size=10_000)

        eval_ds = load_dataset(
            DATASET_NAME, DATASET_SUBSET,
            split="train", streaming=True,
        ).shuffle(seed=42, buffer_size=10_000)

        train_dataset = StreamingTokenDataset(
            tokenizer, train_ds, seq_len=model_config.max_seq_len,
            rank=rank, world_size=world_size,
        )
        eval_dataset = StreamingTokenDataset(
            tokenizer, eval_ds, seq_len=model_config.max_seq_len,
        )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=NUM_WORKERS, pin_memory=use_pinmem,
        persistent_workers=persist,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size,
        num_workers=0, pin_memory=use_pinmem,
    )

    eval_batches = []
    if is_main:
        print("Buffering eval data …")
        for i, batch in enumerate(eval_loader):
            if i >= args.eval_steps:
                break
            eval_batches.append(batch)
        print(f"Buffered {len(eval_batches)} eval batches")

    train_iter = iter(train_loader)

    # ── Wandb (rank 0 only) ───────────────────────────────────────────

    eff_batch_tokens = args.batch_size * args.grad_accum_steps * world_size * model_config.max_seq_len

    if is_main:
        run_name = args.wandb_run_name or f"netra-{config_name}"
        wandb.init(
            project="netra",
            name=run_name,
            mode="disabled" if args.no_wandb else "online",
            config={
                "model": asdict(model_config),
                "n_params": n_params,
                "batch_size_per_gpu": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "world_size": world_size,
                "effective_batch_tokens": eff_batch_tokens,
                "max_lr": args.max_lr,
                "min_lr": min_lr,
                "warmup_steps": args.warmup_steps,
                "max_steps": args.max_steps,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
                "dtype": DTYPE,
                "device": str(device),
            },
        )

    # ── Training loop ─────────────────────────────────────────────────

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model.train()

    if is_main:
        print(f"\nEffective batch: {args.batch_size}×{args.grad_accum_steps}×{world_size} "
              f"= {args.batch_size * args.grad_accum_steps * world_size} sequences "
              f"({eff_batch_tokens:,} tokens)")
        print(f"Training for {args.max_steps:,} steps "
              f"(~{args.max_steps * eff_batch_tokens / 1e6:.0f}M tokens)\n")

    loss_accum = 0.0
    total_tokens = start_step * eff_batch_tokens
    t_start = time.time()

    for step in range(start_step, args.max_steps):
        t_step = time.time()

        lr = cosine_lr(step, args.max_lr, min_lr, args.warmup_steps, args.max_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for micro_step in range(args.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            is_last_micro = micro_step == args.grad_accum_steps - 1
            sync_ctx = nullcontext() if (not ddp or is_last_micro) else model.no_sync()

            with sync_ctx:
                with ctx:
                    _, loss = model(x, targets=y)
                    loss = loss / args.grad_accum_steps

                scaler.scale(loss).backward()
            step_loss += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        loss_accum += step_loss
        total_tokens += eff_batch_tokens
        step_ms = (time.time() - t_step) * 1000

        # ── Periodic logging (rank 0) ─────────────────────────────────

        if is_main and (step + 1) % args.log_interval == 0:
            avg_loss = loss_accum / args.log_interval
            elapsed = time.time() - t_start
            tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0

            log_dict = {
                "train/loss": avg_loss,
                "train/lr": lr,
                "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                "train/tokens_seen": total_tokens,
                "train/tok_per_sec": tok_per_sec,
                "train/step_ms": step_ms,
            }
            log_dict.update(collect_moe_metrics(model))
            wandb.log(log_dict, step=step + 1)

            print(f"step {step+1:>6d}/{args.max_steps} │ "
                  f"loss {avg_loss:.4f} │ lr {lr:.2e} │ "
                  f"grad_norm {log_dict['train/grad_norm']:.2f} │ "
                  f"tok/s {tok_per_sec:,.0f}")

            loss_accum = 0.0

        # ── Evaluation (rank 0) ───────────────────────────────────────

        if is_main and eval_batches and (step + 1) % args.eval_interval == 0:
            val_loss = evaluate(model, eval_batches, device, ctx)
            wandb.log({"eval/loss": val_loss}, step=step + 1)
            print(f"  ↳ eval loss: {val_loss:.4f}")

        # ── Text generation (rank 0) ──────────────────────────────────

        if is_main and (step + 1) % args.generate_interval == 0:
            prompts = ["The meaning of life is", "Once upon a time", "In 2025,"]
            table = wandb.Table(columns=["step", "prompt", "generation"])
            for p in prompts:
                text = generate(model, tokenizer, p, GENERATE_MAX_TOKENS, device=device)
                table.add_data(step + 1, p, text)
                print(f"  ↳ [{p}] → {text[:120]}")
            wandb.log({"samples": table}, step=step + 1)

        # ── Checkpoint (rank 0) ───────────────────────────────────────

        if is_main and (step + 1) % args.save_interval == 0:
            ckpt_path = Path(args.checkpoint_dir) / f"step_{step+1}.pt"
            save_checkpoint(model, optimizer, step + 1, model_config, ckpt_path,
                            r2_bucket=args.r2_bucket,
                            keep_local=args.keep_local_ckpt)
            print(f"  ↳ checkpoint saved → {ckpt_path}")

    # ── Final save ────────────────────────────────────────────────────

    if is_main:
        final_path = Path(args.checkpoint_dir) / "final.pt"
        save_checkpoint(model, optimizer, args.max_steps, model_config, final_path,
                        r2_bucket=args.r2_bucket, keep_local=True)
        print(f"\nTraining complete. Final checkpoint → {final_path}")
        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()

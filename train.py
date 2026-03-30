#!/usr/bin/env python3
"""
Netra pretraining script with wandb monitoring.

Usage:
    python train.py --model_size nano  --tokenizer_path tokenizer.json
    python train.py --model_size mini --max_steps 8000
    python train.py --model_size full  --batch_size 4 --grad_accum_steps 16
"""

import argparse
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from datasets import load_dataset

from netra import ModelConfig, Netra, NetraTokenizer
from netra.data import StreamingTokenDataset

# ── Training presets per model size ───────────────────────────────────

TRAIN_PRESETS = {
    "nano": dict(                           # ~300M tokens (20× ~16M params)
        batch_size=64, grad_accum_steps=1,  # eff batch = 64 × 512 = 32K tok
        max_lr=6e-4, warmup_steps=200, max_steps=9_200,
    ),
    "mini": dict(                           # ~1.3B tokens (20× ~65M params)
        batch_size=32, grad_accum_steps=2,  # eff batch = 64 × 1024 = 64K tok
        max_lr=4e-4, warmup_steps=650, max_steps=20_000,
    ),
    "small": dict(                          # ~5.2B tokens (20× ~260M params)
        batch_size=16, grad_accum_steps=4,  # eff batch = 64 × 2048 = 128K tok
        max_lr=3e-4, warmup_steps=1_400, max_steps=40_000,
    ),
    "full": dict(                           # ~15B tokens (20× ~750M params)
        batch_size=8, grad_accum_steps=8,   # eff batch = 64 × 2048 = 128K tok
        max_lr=2e-4, warmup_steps=2_800, max_steps=115_000,
    ),
}

# ── Helpers ───────────────────────────────────────────────────────────


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def collect_moe_metrics(model: Netra) -> dict:
    metrics = {}
    for i, layer in enumerate(model.layers):
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
    model.eval()
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_tokens):
        inp = ids[:, -model.config.max_seq_len :]
        logits, _ = model(inp)
        next_logits = logits[:, -1, :] / max(temperature, 1e-8)
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == tokenizer.eot_id:
            break
    model.train()
    return tokenizer.decode(ids[0].tolist())


@torch.no_grad()
def evaluate(model, eval_batches, device, ctx):
    model.eval()
    total_loss = 0.0
    for x, y in eval_batches:
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, targets=y)
        total_loss += loss.item()
    model.train()
    return total_loss / max(len(eval_batches), 1)


def save_checkpoint(model, optimizer, step, model_config, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "model_config": asdict(model_config),
    }, path)


# ── Argument parsing ──────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Netra pretraining")

    p.add_argument("--model_size", type=str, default="nano",
                    choices=["nano", "mini", "small", "full"])
    p.add_argument("--tokenizer_path", type=str, default="tokenizer.json")

    p.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset_subset", type=str, default="sample-10BT")

    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=None)
    p.add_argument("--max_lr", type=float, default=None)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    p.add_argument("--compile", action="store_true")

    p.add_argument("--wandb_project", type=str, default="netra")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=20)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--generate_interval", type=int, default=500)
    p.add_argument("--generate_max_tokens", type=int, default=100)

    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume_from", type=str, default=None)

    args = p.parse_args()

    preset = TRAIN_PRESETS[args.model_size]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    return args


# ── Main ──────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
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
            "Train one first with NetraTokenizer.train() (see notebooks/main.ipynb)."
        )
    tokenizer = NetraTokenizer.from_file(tok_path)
    print(f"Tokenizer loaded — vocab size: {tokenizer.vocab_size:,}")

    # ── Model ─────────────────────────────────────────────────────────

    model_config = getattr(ModelConfig, args.model_size)(vocab_size=tokenizer.vocab_size)
    model = Netra(model_config).to(device)
    n_params = count_parameters(model)
    print(f"Model: {args.model_size} | {n_params:,} params | device: {device}")

    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    # ── Optimizer ─────────────────────────────────────────────────────

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.dim() < 2 or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.max_lr, betas=(args.beta1, args.beta2))

    # ── Data ──────────────────────────────────────────────────────────

    print("Initialising data streams …")
    train_ds = load_dataset(
        args.dataset_name, args.dataset_subset,
        split="train", streaming=True,
    ).shuffle(seed=0, buffer_size=10_000)

    eval_ds = load_dataset(
        args.dataset_name, args.dataset_subset,
        split="train", streaming=True,
    ).shuffle(seed=42, buffer_size=10_000)

    train_dataset = StreamingTokenDataset(tokenizer, train_ds, seq_len=model_config.max_seq_len)
    eval_dataset = StreamingTokenDataset(tokenizer, eval_ds, seq_len=model_config.max_seq_len)

    use_pinmem = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, pin_memory=use_pinmem,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, pin_memory=use_pinmem,
    )

    # Buffer eval batches so we can re-use them without re-streaming
    print("Buffering eval data …")
    eval_batches = []
    for i, batch in enumerate(eval_loader):
        if i >= args.eval_steps:
            break
        eval_batches.append(batch)
    print(f"Buffered {len(eval_batches)} eval batches")

    train_iter = iter(train_loader)

    # ── Resume ────────────────────────────────────────────────────────

    start_step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"Resumed from checkpoint at step {start_step}")

    # ── Wandb ─────────────────────────────────────────────────────────

    run_name = args.wandb_run_name or f"netra-{args.model_size}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        mode="disabled" if args.no_wandb else "online",
        config={
            "model": asdict(model_config),
            "n_params": n_params,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_tokens": args.batch_size * args.grad_accum_steps * model_config.max_seq_len,
            "max_lr": args.max_lr,
            "min_lr": min_lr,
            "warmup_steps": args.warmup_steps,
            "max_steps": args.max_steps,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "dtype": args.dtype,
            "device": str(device),
        },
    )

    # ── Training loop ─────────────────────────────────────────────────

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model.train()

    eff_batch_tokens = args.batch_size * args.grad_accum_steps * model_config.max_seq_len
    print(f"\nEffective batch: {args.batch_size}×{args.grad_accum_steps} "
          f"= {args.batch_size * args.grad_accum_steps} sequences "
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

        for _micro in range(args.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            with ctx:
                _, loss = model(x, targets=y)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            step_loss += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_accum += step_loss
        total_tokens += eff_batch_tokens
        step_ms = (time.time() - t_step) * 1000

        # ── Periodic logging ──────────────────────────────────────────

        if (step + 1) % args.log_interval == 0:
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

        # ── Evaluation ────────────────────────────────────────────────

        if eval_batches and (step + 1) % args.eval_interval == 0:
            val_loss = evaluate(model, eval_batches, device, ctx)
            wandb.log({"eval/loss": val_loss}, step=step + 1)
            print(f"  ↳ eval loss: {val_loss:.4f}")

        # ── Text generation samples ──────────────────────────────────

        if (step + 1) % args.generate_interval == 0:
            prompts = ["The meaning of life is", "Once upon a time", "In 2025,"]
            table = wandb.Table(columns=["step", "prompt", "generation"])
            for p in prompts:
                text = generate(model, tokenizer, p, args.generate_max_tokens, device=device)
                table.add_data(step + 1, p, text)
                print(f"  ↳ [{p}] → {text[:120]}")
            wandb.log({"samples": table}, step=step + 1)

        # ── Checkpoint ────────────────────────────────────────────────

        if (step + 1) % args.save_interval == 0:
            ckpt_path = Path(args.checkpoint_dir) / f"step_{step+1}.pt"
            save_checkpoint(model, optimizer, step + 1, model_config, ckpt_path)
            print(f"  ↳ checkpoint saved → {ckpt_path}")

    # ── Final save ────────────────────────────────────────────────────

    final_path = Path(args.checkpoint_dir) / "final.pt"
    save_checkpoint(model, optimizer, args.max_steps, model_config, final_path)
    print(f"\nTraining complete. Final checkpoint → {final_path}")
    wandb.finish()


if __name__ == "__main__":
    main()

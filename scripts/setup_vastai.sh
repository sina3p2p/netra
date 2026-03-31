#!/bin/bash
# ── Netra Cloud GPU Setup ────────────────────────────────────────────
#
# End-to-end setup for Vast.ai / RunPod / any cloud GPU:
#   1. Installs dependencies
#   2. Trains tokenizer if missing
#   3. Downloads + pre-tokenizes data if tokens.bin missing
#   4. Launches multi-GPU training
#
# Usage:
#   1. Rent a multi-GPU pod (PyTorch 2.x template)
#   2. Upload your code:
#        rsync -avz --exclude '.git' --exclude 'notebooks' --exclude '__pycache__' \
#          . root@<pod-ip>:/workspace/netra/
#   3. SSH in and run:
#        cd /workspace/netra && bash scripts/setup_vastai.sh
#
# Environment variables (optional):
#   WANDB_API_KEY        - Weights & Biases API key for logging
#   NUM_GPUS             - Number of GPUs to use (default: auto-detect)
#   CONFIG               - YAML config file (default: configs/medium.yaml)
#   MAX_TOKENS_B         - Billions of tokens to tokenize (default: 22.5)
#   EXTRA_ARGS           - Extra arguments to pass to train.py
#   R2_BUCKET            - Cloudflare R2 bucket for checkpoint backup
#   R2_ENDPOINT_URL      - R2 S3-compatible endpoint
#   R2_ACCESS_KEY_ID     - R2 access key
#   R2_SECRET_ACCESS_KEY - R2 secret key

set -e

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

CONFIG="${CONFIG:-configs/medium.yaml}"
MAX_TOKENS_B="${MAX_TOKENS_B:-22.5}"
NUM_GPUS="${NUM_GPUS:-$(python3 -c 'import torch; print(torch.cuda.device_count())')}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CONFIG_NAME=$(basename "$CONFIG" .yaml)

echo "════════════════════════════════════════════════════════════════"
echo "  Netra Training Setup"
echo "  Config: $CONFIG_NAME | GPUs: $NUM_GPUS | Data: ${MAX_TOKENS_B}B tokens"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Install dependencies ──────────────────────────────────────
echo "▶ [1/4] Installing dependencies..."
pip install -q tokenizers datasets wandb pyyaml 2>/dev/null
echo "  ✓ Dependencies installed"
echo ""

# ── Step 2: Train tokenizer if missing ────────────────────────────────
if [ -f "tokenizer.json" ]; then
    echo "▶ [2/4] Tokenizer found — skipping"
else
    echo "▶ [2/4] Training tokenizer (500K docs from FineWeb)..."
    python3 tools/train_tokenizer.py --save_path tokenizer.json
    echo "  ✓ Tokenizer trained → tokenizer.json"
fi
echo ""

# ── Step 3: Pre-tokenize data if missing ──────────────────────────────
if [ -f "tokens.bin" ]; then
    SIZE_GB=$(python3 -c "import os; print(f'{os.path.getsize(\"tokens.bin\") / (1024**3):.1f}')")
    echo "▶ [3/4] tokens.bin found (${SIZE_GB} GB) — skipping"
else
    echo "▶ [3/4] Downloading & tokenizing ${MAX_TOKENS_B}B tokens from FineWeb..."
    echo "  This will take a while (1-3 hours on CPU)..."
    python3 tools/tokenize_data.py \
        --max_tokens_b "$MAX_TOKENS_B" \
        --out tokens.bin
    echo "  ✓ tokens.bin created"
fi
echo ""

# ── Step 4: Launch training ───────────────────────────────────────────
echo "▶ [4/4] Launching training: $CONFIG_NAME on $NUM_GPUS GPUs"
echo "════════════════════════════════════════════════════════════════"
echo ""

R2_ARGS=""
if [ -n "$R2_BUCKET" ]; then
    pip install -q boto3
    R2_ARGS="--r2_bucket $R2_BUCKET"
    echo "  R2 backup enabled → $R2_BUCKET (local copies deleted after upload)"
fi

torchrun --nproc_per_node="$NUM_GPUS" train.py \
    --config "$CONFIG" \
    --data_path tokens.bin \
    --checkpoint_dir checkpoints \
    --compile \
    $R2_ARGS \
    $EXTRA_ARGS

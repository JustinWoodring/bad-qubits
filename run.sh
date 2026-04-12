#!/usr/bin/env bash
# run.sh — Bad Qubits full pipeline runner for RunPod H200
#
# USAGE
#   ./run.sh --api-key <ZHIPUAI_KEY> [OPTIONS]
#
# OPTIONS
#   --api-key    KEY    ZhipuAI API key for explanation generation (required)
#   --fold       N      Run only fold N (1-5); default: all 5 folds
#   --sft-steps  N      SFT warm-up steps per fold (default: 60)
#   --grpo-steps N      GRPO fine-tuning steps per fold (default: 200)
#   --skip-explain      Skip explanation generation (reuse existing explanations.jsonl)
#   --limit      N      Limit explanation generation to N circuits (for testing)
#   --dataset-dir PATH  Path to dataset (default: dataset)
#   --data-dir    PATH  Path to processed data (default: data)
#   --results-dir PATH  Path to results output (default: results)
#   --models-dir  PATH  Path to model output (default: models)
#   --hf-repo    REPO   HuggingFace repo to push models to (e.g. username/bad-qubits)
#   --hf-token   TOKEN  HuggingFace token (default: $HF_TOKEN env var)
#   -h | --help         Show this message
#
# EXAMPLE
#   ./run.sh --api-key abc123
#   ./run.sh --api-key abc123 --fold 1 --skip-explain
#   ./run.sh --api-key abc123 --hf-repo username/bad-qubits

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
step() { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "[$(date '+%H:%M:%S')] $*"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
elapsed() {
    local secs=$(( $(date +%s) - _STEP_START ))
    printf "%dm%02ds" $(( secs/60 )) $(( secs%60 ))
}
_PIPELINE_START=$(date +%s)
_STEP_START=$(date +%s)

# ── RunPod path routing ───────────────────────────────────────────────────────
# On RunPod the volume disk is mounted at /workspace (20 GB persistent).
# Route the HF model cache and all large outputs there so they don't fill
# the 20 GB container disk.
if [[ -d "/workspace" ]]; then
    log "RunPod environment detected — routing outputs to /workspace"
    export HF_HOME="/workspace/.cache/huggingface"
    export TRANSFORMERS_CACHE="/workspace/.cache/huggingface"
    _DEFAULT_DATA_DIR="/workspace/data"
    _DEFAULT_RESULTS_DIR="/workspace/results"
    _DEFAULT_MODELS_DIR="/workspace/models"
else
    log "Local environment detected"
    _DEFAULT_DATA_DIR="data"
    _DEFAULT_RESULTS_DIR="results"
    _DEFAULT_MODELS_DIR="models"
fi

# ── dependency install ────────────────────────────────────────────────────────
# We use official unsloth pinned to a 2025.x release, which is compatible with
# PyTorch 2.4 (the RunPod base image). unsloth 2026.4+ pulls in unsloth-zoo
# 2026.4 → torchao>=0.13.0 → torch.int1 (only in PyTorch 2.5+), which breaks
# on RunPod pods whose driver cannot support PyTorch 2.5.
#
# The GRPO bfloat16/bnb-4bit dtype mismatch (upstream PR #4918) is handled
# entirely by the inspect.getsource patch in train_cv.py, so the anandn1 fork
# is no longer needed.
log "Checking dependencies..."
if ! python -c "import unsloth" &>/dev/null; then
    log "unsloth not found — installing 2025.x release (PyTorch 2.4 compatible)..."
    # Lock torch at its current version so pip doesn't upgrade it to a version
    # incompatible with the RunPod CUDA driver.
    _TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    log "  Locking torch==$_TORCH_VER, unsloth-zoo<2026, transformers<4.47, accelerate<1.2..."
    pip install \
        "unsloth<2026" \
        "unsloth-zoo<2026" \
        "torch==$_TORCH_VER" \
        "transformers>=4.47.0,<4.52.0" \
        "accelerate>=1.0.0,<1.2.0" \
        "trl>=0.13.0,<0.15.0" \
        "torchao<0.7.0" \
        2>&1 | while IFS= read -r line; do
        echo "  [pip] $line"
    done
    log "unsloth install complete."
    # Clear compiled cache so it regenerates for the newly installed version.
    rm -rf "$SCRIPT_DIR/unsloth_compiled_cache"
    log "Cleared unsloth_compiled_cache."
else
    log "unsloth already installed, skipping."
fi

log "Installing requirements.txt..."
pip install -r "$SCRIPT_DIR/requirements.txt" 2>&1 | while IFS= read -r line; do
    echo "  [pip] $line"
done
log "Dependencies ready."

# ── defaults ──────────────────────────────────────────────────────────────────
API_KEY=""
FOLD=""
SFT_STEPS=150
GRPO_STEPS=200
SKIP_EXPLAIN=0
LIMIT=""
DATASET_DIR="$SCRIPT_DIR/dataset"
DATA_DIR="$_DEFAULT_DATA_DIR"
RESULTS_DIR="$_DEFAULT_RESULTS_DIR"
MODELS_DIR="$_DEFAULT_MODELS_DIR"
HF_REPO=""
HF_TOKEN="${HF_TOKEN:-}"

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-key)      API_KEY="$2";       shift 2 ;;
        --fold)         FOLD="$2";          shift 2 ;;
        --sft-steps)    SFT_STEPS="$2";     shift 2 ;;
        --grpo-steps)   GRPO_STEPS="$2";    shift 2 ;;
        --skip-explain) SKIP_EXPLAIN=1;     shift   ;;
        --limit)        LIMIT="$2";         shift 2 ;;
        --dataset-dir)  DATASET_DIR="$2";   shift 2 ;;
        --data-dir)     DATA_DIR="$2";      shift 2 ;;
        --results-dir)  RESULTS_DIR="$2";   shift 2 ;;
        --models-dir)   MODELS_DIR="$2";    shift 2 ;;
        --hf-repo)      HF_REPO="$2";       shift 2 ;;
        --hf-token)     HF_TOKEN="$2";      shift 2 ;;
        -h|--help)
            sed -n '/^# USAGE/,/^[^#]/p' "$0" | head -n -1 | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$API_KEY" && "$SKIP_EXPLAIN" -eq 0 ]]; then
    echo "ERROR: --api-key is required (or pass --skip-explain to reuse explanations.jsonl)"
    exit 1
fi

# ── common flags ──────────────────────────────────────────────────────────────
COMMON="--dataset-dir $DATASET_DIR --data-dir $DATA_DIR --results-dir $RESULTS_DIR --models-dir $MODELS_DIR"
FOLD_FLAG=${FOLD:+"--fold $FOLD"}

echo ""
echo "============================================================"
echo "  Bad Qubits — Full Pipeline"
echo "============================================================"
echo "  Dataset dir : $DATASET_DIR"
echo "  Data dir    : $DATA_DIR"
echo "  Results dir : $RESULTS_DIR"
echo "  Models dir  : $MODELS_DIR"
[[ -n "$HF_REPO" ]] && echo "  HF repo     : $HF_REPO"
[[ -n "$FOLD" ]]    && echo "  Fold        : $FOLD"
echo "  SFT steps   : $SFT_STEPS"
echo "  GRPO steps  : $GRPO_STEPS"
[[ "$SKIP_EXPLAIN" -eq 1 ]] && echo "  Explain     : skipped"
echo "============================================================"

# ── STEP 1: prepare ───────────────────────────────────────────────────────────
step "[1/5] Prepare dataset"
_STEP_START=$(date +%s)
python main.py prepare $COMMON
log "Step 1 complete ($(elapsed))"

# ── STEP 2: explain ───────────────────────────────────────────────────────────
if [[ "$SKIP_EXPLAIN" -eq 0 ]]; then
    step "[2/5] Generate explanations"
    _STEP_START=$(date +%s)
    LIMIT_FLAG=${LIMIT:+"--limit $LIMIT"}
    [[ -n "$LIMIT" ]] && log "Limiting to $LIMIT circuits"
    python main.py explain $COMMON \
        --api-key "$API_KEY" \
        --manifest "$DATA_DIR/all_filenames.json" \
        --output explanations.jsonl \
        ${LIMIT_FLAG:-}
    log "Step 2 complete ($(elapsed))"
else
    step "[2/5] Skipping explanation generation (reusing explanations.jsonl)"
fi

# ── STEP 3: train ─────────────────────────────────────────────────────────────
step "[3/5] Train — SFT + GRPO${FOLD:+, fold $FOLD only}"
_STEP_START=$(date +%s)
log "SFT steps: $SFT_STEPS  |  GRPO steps: $GRPO_STEPS"
python main.py train $COMMON \
    --explanations explanations.jsonl \
    --sft-steps "$SFT_STEPS" \
    --grpo-steps "$GRPO_STEPS" \
    ${FOLD_FLAG:-}
log "Step 3 complete ($(elapsed))"

# ── STEP 4: aggregate ─────────────────────────────────────────────────────────
step "[4/5] Aggregate results"
_STEP_START=$(date +%s)
python main.py aggregate $COMMON
log "Step 4 complete ($(elapsed))"

# ── STEP 5: push to HuggingFace ───────────────────────────────────────────────
if [[ -n "$HF_REPO" ]]; then
    step "[5/5] Push models to HuggingFace — $HF_REPO"
    _STEP_START=$(date +%s)
    if [[ -z "$HF_TOKEN" ]]; then
        echo "ERROR: --hf-token or HF_TOKEN env var is required for HuggingFace push"
        exit 1
    fi
    python - <<PYEOF
import os, sys, time
from huggingface_hub import HfApi, upload_folder

def log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

api = HfApi(token="$HF_TOKEN")
repo_id = "$HF_REPO"

log(f"Creating/verifying repo: {repo_id}")
api.create_repo(repo_id=repo_id, exist_ok=True)
log("Repo ready.")

models_dir = "$MODELS_DIR"
fold_dirs = sorted(d for d in os.listdir(models_dir)
                   if os.path.isdir(os.path.join(models_dir, d)) and d.startswith("fold_"))

for i, fold_dir in enumerate(fold_dirs, 1):
    fold_path = os.path.join(models_dir, fold_dir)
    log(f"Uploading {fold_dir} ({i}/{len(fold_dirs)})...")
    t0 = time.time()
    upload_folder(
        repo_id=repo_id,
        folder_path=fold_path,
        path_in_repo=fold_dir,
        token="$HF_TOKEN",
    )
    log(f"  {fold_dir} done ({time.time()-t0:.0f}s)")

results_dir = "$RESULTS_DIR"
agg_dir = os.path.join(results_dir, "aggregate")
if os.path.isdir(agg_dir):
    log("Uploading aggregate results...")
    t0 = time.time()
    upload_folder(
        repo_id=repo_id,
        folder_path=agg_dir,
        path_in_repo="results/aggregate",
        token="$HF_TOKEN",
    )
    log(f"  aggregate done ({time.time()-t0:.0f}s)")

log(f"All uploads complete — https://huggingface.co/{repo_id}")
PYEOF
    log "Step 5 complete ($(elapsed))"
else
    log "Skipping HuggingFace push (no --hf-repo specified)"
fi

_TOTAL=$(( $(date +%s) - _PIPELINE_START ))
echo ""
echo "============================================================"
echo "  Pipeline complete in $(printf '%dm%02ds' $((_TOTAL/60)) $((_TOTAL%60)))"
echo "  Outputs:"
echo "    explanations.jsonl"
echo "    $RESULTS_DIR/aggregate/cv_summary.json"
echo "    $RESULTS_DIR/aggregate/avg_confusion_matrix.png"
echo "    $MODELS_DIR/fold_{1-5}/"
[[ -n "$HF_REPO" ]] && echo "    https://huggingface.co/$HF_REPO"
echo "============================================================"

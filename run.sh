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

# ── RunPod path routing ───────────────────────────────────────────────────────
# On RunPod the volume disk is mounted at /workspace (20 GB persistent).
# Route the HF model cache and all large outputs there so they don't fill
# the 20 GB container disk.
if [[ -d "/workspace" ]]; then
    export HF_HOME="/workspace/.cache/huggingface"
    export TRANSFORMERS_CACHE="/workspace/.cache/huggingface"
    _DEFAULT_DATA_DIR="/workspace/data"
    _DEFAULT_RESULTS_DIR="/workspace/results"
    _DEFAULT_MODELS_DIR="/workspace/models"
else
    _DEFAULT_DATA_DIR="data"
    _DEFAULT_RESULTS_DIR="results"
    _DEFAULT_MODELS_DIR="models"
fi

# ── dependency install ────────────────────────────────────────────────────────
# Auto-installs on a fresh RunPod pod. Safe to re-run (pip is idempotent).
if ! python -c "import unsloth" &>/dev/null; then
    echo "Installing unsloth..."
    pip install --quiet "unsloth[cu124-torch240]"
fi
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"

# ── defaults ──────────────────────────────────────────────────────────────────
API_KEY=""
FOLD=""
SFT_STEPS=60
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

echo "============================================================"
echo "  Bad Qubits — Full Pipeline"
echo "============================================================"
echo "  Dataset dir : $DATASET_DIR"
echo "  Data dir    : $DATA_DIR"
echo "  Results dir : $RESULTS_DIR"
echo "  Models dir  : $MODELS_DIR"
[[ -n "$HF_REPO" ]] && echo "  HF repo     : $HF_REPO"
[[ -n "$FOLD" ]]   && echo "  Fold        : $FOLD"
echo "  SFT steps   : $SFT_STEPS"
echo "  GRPO steps  : $GRPO_STEPS"
[[ "$SKIP_EXPLAIN" -eq 1 ]] && echo "  Explain     : skipped"
echo ""

# ── STEP 1: prepare ───────────────────────────────────────────────────────────
echo "--- [1/4] Prepare dataset ---"
python main.py prepare $COMMON

# ── STEP 2: explain ───────────────────────────────────────────────────────────
if [[ "$SKIP_EXPLAIN" -eq 0 ]]; then
    echo ""
    echo "--- [2/4] Generate explanations ---"
    LIMIT_FLAG=${LIMIT:+"--limit $LIMIT"}
    python main.py explain $COMMON \
        --api-key "$API_KEY" \
        --manifest "$DATA_DIR/all_filenames.json" \
        --output explanations.jsonl \
        ${LIMIT_FLAG:-}
else
    echo ""
    echo "--- [2/4] Skipping explanation generation ---"
fi

# ── STEP 3: train ─────────────────────────────────────────────────────────────
echo ""
echo "--- [3/4] Train (SFT + GRPO, 5-fold CV) ---"
python main.py train $COMMON \
    --explanations explanations.jsonl \
    --sft-steps "$SFT_STEPS" \
    --grpo-steps "$GRPO_STEPS" \
    ${FOLD_FLAG:-}

# ── STEP 4: aggregate ─────────────────────────────────────────────────────────
echo ""
echo "--- [4/4] Aggregate results ---"
python main.py aggregate $COMMON

# ── STEP 5: push to HuggingFace ───────────────────────────────────────────────
if [[ -n "$HF_REPO" ]]; then
    echo ""
    echo "--- [5/5] Push models to HuggingFace ($HF_REPO) ---"
    if [[ -z "$HF_TOKEN" ]]; then
        echo "ERROR: --hf-token or HF_TOKEN env var is required for HuggingFace push"
        exit 1
    fi
    python - <<PYEOF
import os
from huggingface_hub import HfApi, upload_folder

api = HfApi(token="$HF_TOKEN")
repo_id = "$HF_REPO"

# Create repo if it doesn't exist
api.create_repo(repo_id=repo_id, exist_ok=True)

models_dir = "$MODELS_DIR"
for fold_dir in sorted(os.listdir(models_dir)):
    fold_path = os.path.join(models_dir, fold_dir)
    if os.path.isdir(fold_path) and fold_dir.startswith("fold_"):
        print(f"  Uploading {fold_dir}...")
        upload_folder(
            repo_id=repo_id,
            folder_path=fold_path,
            path_in_repo=fold_dir,
            token="$HF_TOKEN",
        )

# Upload aggregate results
results_dir = "$RESULTS_DIR"
agg_dir = os.path.join(results_dir, "aggregate")
if os.path.isdir(agg_dir):
    print("  Uploading aggregate results...")
    upload_folder(
        repo_id=repo_id,
        folder_path=agg_dir,
        path_in_repo="results/aggregate",
        token="$HF_TOKEN",
    )

print(f"  Done — https://huggingface.co/{repo_id}")
PYEOF
fi

echo ""
echo "============================================================"
echo "  Done. Outputs:"
echo "    explanations.jsonl"
echo "    $RESULTS_DIR/aggregate/cv_summary.json"
echo "    $RESULTS_DIR/aggregate/avg_confusion_matrix.png"
echo "    $MODELS_DIR/fold_{1-5}/"
[[ -n "$HF_REPO" ]] && echo "    https://huggingface.co/$HF_REPO"
echo "============================================================"

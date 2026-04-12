#!/usr/bin/env python3
"""
train_cv.py - 5-fold cross-validation training for bad-qubits using Qwen 2.5 Coder.

Ported from malicious-qubits/train.py with these changes:
- Qwen 2.5 Coder only (all other models removed)
- Training targets are JSON strings from explanations.jsonl
- Label parsing reads the "safe" field from JSON output
- 5-fold CV with per-fold results saved to results/fold_N/
- VRAM freed between folds

Run via: python main.py train [--fold N]
Or directly: python train_cv.py [--fold N] [--data-dir data] [--results-dir results] [--models-dir models]
"""

import os
import gc
import re
import json
import time
import argparse
from datetime import datetime
from multiprocessing import cpu_count

import torch

# llm_blender (a trl transitive dep) references TRANSFORMERS_CACHE which was
# removed in transformers >=4.43. Patch it back before trl is imported.
import os as _os
import transformers.utils.hub as _hub
if not hasattr(_hub, "TRANSFORMERS_CACHE"):
    _hub.TRANSFORMERS_CACHE = _os.getenv(
        "TRANSFORMERS_CACHE",
        _os.path.join(_os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    )

from unsloth import FastLanguageModel, PatchFastRL
from datasets import Dataset
from trl import SFTTrainer, GRPOTrainer, GRPOConfig
from transformers import TrainingArguments

PatchFastRL("GRPO", FastLanguageModel)
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_recall_fscore_support,
    f1_score, precision_score, recall_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

from prepare_dataset import infer_category, infer_label
from generate_explanations import extract_circuit_properties


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-bnb-4bit"
MAX_SEQ_LENGTH = 8192
MAX_CIRCUIT_CHARS = 7168  # ~7k chars of circuit text; leaves ~1k tokens for prompt wrapper
MAX_NEW_TOKENS_INFERENCE = 256  # JSON with label + category + explanation; 256 gives full headroom


# ---------------------------------------------------------------------------
# CUDA setup (verbatim from source)
# ---------------------------------------------------------------------------

def setup_cuda():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ---------------------------------------------------------------------------
# Data formatting (Qwen chat format)
# ---------------------------------------------------------------------------

def format_data_qwen(examples):
    """Format chat messages into Qwen prompt strings for SFTTrainer."""
    texts = []
    for messages in examples["messages"]:
        formatted = ""
        for msg in messages:
            if msg["role"] == "user":
                formatted += f"<|user|>\n{msg['content']}<|endoftext|>\n"
            elif msg["role"] == "assistant":
                formatted += f"<|assistant|>\n{msg['content']}<|endoftext|>"
        texts.append(formatted)
    return {"text": texts}


def create_inference_prompt_with_props(circuit_code: str, props: dict) -> str:
    """Build Qwen inference prompt including structural circuit properties (no category hint)."""
    swap = props["gate_counts"].get("swap", 0)
    measure = props["gate_counts"].get("measure", 0)
    total = props["num_gates"]
    mfrac = f"{measure / total:.0%}" if total > 0 else "0%"
    top5 = list(props["gate_counts"].items())[:5]
    summary = (
        f"[Circuit: qubits={props['num_qubits']}, gates={total}, "
        f"swap={swap}, measure={measure}({mfrac}), top_gates={top5}]"
    )
    user_content = f"{summary}\n\nAnalyze this quantum circuit:\n{circuit_code}"
    return f"<|user|>\n{user_content}<|endoftext|>\n<|assistant|>\n"


# ---------------------------------------------------------------------------
# Batch inference (adapted from source ~line 154)
# ---------------------------------------------------------------------------

def get_optimal_batch_size() -> int:
    if not torch.cuda.is_available():
        return 1
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    available_memory = gpu_memory - torch.cuda.memory_allocated(0)
    estimated_memory_per_sample = 3 * 1024**3
    return max(1, int(available_memory * 0.6 / estimated_memory_per_sample))


def batch_classify_quantum_circuits(
    model, tokenizer, circuit_codes: list[str], circuit_props: list[dict] = None,
    batch_size: int = None,
) -> list[str]:
    """Run inference on a list of circuit code strings. Returns raw output strings."""
    if batch_size is None:
        batch_size = get_optimal_batch_size()
    if circuit_props is None:
        circuit_props = [{"num_qubits": 0, "num_gates": 0, "gate_counts": {}} for _ in circuit_codes]

    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Inferring {len(circuit_codes)} samples (batch_size={batch_size})...")

    for i in range(0, len(circuit_codes), batch_size):
        print(
            f"\r  Batch {i // batch_size + 1}/"
            f"{(len(circuit_codes) + batch_size - 1) // batch_size} "
            f"({i}/{len(circuit_codes)})",
            end="",
            flush=True,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_codes = circuit_codes[i : i + batch_size]
        batch_props = circuit_props[i : i + batch_size]
        prompts = [create_inference_prompt_with_props(code, props)
                   for code, props in zip(batch_codes, batch_props)]

        orig_padding = tokenizer.padding_side
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding=True,
            add_special_tokens=False,
        ).to(device)
        tokenizer.padding_side = orig_padding

        try:
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_bf16_supported()):
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=MAX_NEW_TOKENS_INFERENCE,
                        use_cache=True,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        num_beams=1,
                    )

            for j, output in enumerate(outputs):
                input_length = len(inputs.input_ids[j])
                response = tokenizer.decode(
                    output[input_length:], skip_special_tokens=True
                )
                results.append(response.strip())

            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM at batch {i // batch_size + 1}, halving batch size...")
            if batch_size > 1:
                remaining = batch_classify_quantum_circuits(
                    model, tokenizer, circuit_codes[i:], circuit_props[i:], batch_size // 2
                )
                results.extend(remaining)
                break
            else:
                raise

    print()
    return results


# ---------------------------------------------------------------------------
# Label parsing — JSON output
# ---------------------------------------------------------------------------

def parse_prediction(raw_output: str) -> tuple[str, str]:
    """
    Parse the model's JSON output into (label, category).
    label:    "safe" or "bad"
    category: "safe" | "immediate" | "shuttling" | "mixed" | "unknown"

    Falls back gracefully on malformed JSON.
    """
    # Try to find JSON object
    json_match = re.search(r"\{[^{}]*\}", raw_output, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            safe_val = str(data.get("safe", "")).lower().strip()
            label = "safe" if safe_val == "true" else "bad"
            category = data.get("category", "unknown")
            return label, category
        except json.JSONDecodeError:
            pass

    # Fallback: keyword search
    lower = raw_output.lower()
    label = "safe" if '"safe": "true"' in lower or "safe" in lower.split()[:5] else "bad"
    return label, "unknown"


# ---------------------------------------------------------------------------
# Loss curves (adapted from source ~line 222)
# ---------------------------------------------------------------------------

def plot_loss_curves(trainer, fold_num: int, results_dir: str) -> None:
    """Save training/validation loss curves and raw data to results_dir."""
    log_history = trainer.state.log_history

    train_steps, train_losses, learning_rates = [], [], []
    eval_steps, eval_losses = [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
            if "learning_rate" in entry:
                learning_rates.append(entry["learning_rate"])
        elif "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    if not train_losses:
        print("  No training loss data found.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(train_steps, train_losses, "b-", linewidth=2, label="Training Loss", alpha=0.8)
    if eval_losses:
        ax1.plot(eval_steps, eval_losses, "r-", linewidth=2, label="Validation Loss", alpha=0.8)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Fold {fold_num} — Training and Validation Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if learning_rates and len(learning_rates) == len(train_steps):
        ax2.plot(train_steps, learning_rates, "g-", linewidth=2, label="Learning Rate")
        ax2.set_xlabel("Training Steps")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    else:
        ax2.text(0.5, 0.5, "Learning Rate Data Not Available",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=14)
        ax2.set_title("Learning Rate Schedule")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "loss_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()

    loss_data_path = os.path.join(results_dir, "loss_data.txt")
    with open(loss_data_path, "w") as f:
        f.write(f"Fold {fold_num} Training and Validation Loss\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Step':<8} {'Train Loss':<12} {'Learning Rate':<15}\n")
        f.write("-" * 35 + "\n")
        for i, (step, loss) in enumerate(zip(train_steps, train_losses)):
            lr = learning_rates[i] if i < len(learning_rates) else "N/A"
            lr_str = f"{lr:.2e}" if isinstance(lr, float) else str(lr)
            f.write(f"{step:<8} {loss:<12.6f} {lr_str:<15}\n")
        if eval_losses:
            f.write(f"\nVALIDATION:\n{'Step':<8} {'Eval Loss':<12}\n")
            f.write("-" * 20 + "\n")
            for step, loss in zip(eval_steps, eval_losses):
                f.write(f"{step:<8} {loss:<12.6f}\n")
        f.write(f"\nFinal train loss: {train_losses[-1]:.6f}\n")
        if eval_losses:
            f.write(f"Final eval loss:  {eval_losses[-1]:.6f}\n")


# ---------------------------------------------------------------------------
# Evaluation (extended from source ~line 325)
# ---------------------------------------------------------------------------

def evaluate_fold_qualitative(
    model,
    tokenizer,
    test_dir: str,
    fold_num: int,
    results_dir: str,
) -> list[dict]:
    """
    Run inference on the held-out test set and save per-sample qualitative records.

    Each record contains: filename, true_label, true_category, pred_label,
    pred_category, explanation, correct (bool).

    Saves test_qualitative.json to results_dir and returns the list.
    """
    print(f"\n  Qualitative evaluation on test set ({test_dir})...")

    records = []
    for filename in sorted(os.listdir(test_dir)):
        if not filename.endswith(".qasm"):
            continue
        filepath = os.path.join(test_dir, filename)
        if os.path.islink(filepath):
            filepath = os.path.realpath(filepath)
        with open(filepath, "r") as f:
            content = f.read()
        escaped = content.replace("\n\n", "\n").replace("\n", "\\n")
        if len(escaped) > MAX_CIRCUIT_CHARS:
            escaped = escaped[:MAX_CIRCUIT_CHARS] + " [TRUNCATED]"
        props = extract_circuit_properties(content)

        records.append({
            "filename":      os.path.basename(filename),
            "true_label":    infer_label(os.path.basename(filename)),
            "true_category": infer_category(os.path.basename(filename)),
            "circuit_code":  escaped,
            "props":         props,
        })

    if not records:
        print("  No test samples found.")
        return []

    circuit_codes = [r["circuit_code"] for r in records]
    props_list    = [r["props"]        for r in records]
    predictions   = batch_classify_quantum_circuits(model, tokenizer, circuit_codes, props_list)

    qualitative = []
    for rec, prediction in zip(records, predictions):
        pred_label, pred_cat = parse_prediction(prediction)
        parsed = _parse_json_output(prediction)
        explanation = parsed.get("explanation", "") if parsed else ""
        correct = pred_label == rec["true_label"]
        qualitative.append({
            "filename":       rec["filename"],
            "true_label":     rec["true_label"],
            "true_category":  rec["true_category"],
            "pred_label":     pred_label,
            "pred_category":  pred_cat,
            "explanation":    explanation,
            "correct":        correct,
        })

    n_correct = sum(1 for q in qualitative if q["correct"])
    print(f"  Test set: {n_correct}/{len(qualitative)} correct "
          f"({n_correct / len(qualitative):.1%})")

    out_path = os.path.join(results_dir, "test_qualitative.json")
    with open(out_path, "w") as f:
        json.dump({"fold": fold_num, "samples": qualitative}, f, indent=2)
    print(f"  Qualitative results saved to {out_path}")

    return qualitative


def evaluate_fold(
    model,
    tokenizer,
    val_dir: str,
    fold_num: int,
    results_dir: str,
) -> dict:
    """
    Evaluate the model on all .qasm files in val_dir.
    Saves confusion_matrix.png, classification_report.txt, per_class_metrics.json.
    Returns metrics dict.
    """
    print(f"\n  Evaluating on validation set ({val_dir})...")

    y_true_binary = []   # "safe" or "bad"
    y_pred_binary = []
    y_true_category = []
    y_pred_category = []
    circuit_codes = []
    circuit_props_list = []
    filenames = []

    for filename in sorted(os.listdir(val_dir)):
        if not filename.endswith(".qasm"):
            continue
        filepath = os.path.join(val_dir, filename)
        # Resolve symlink to get real path if needed
        if os.path.islink(filepath):
            filepath = os.path.realpath(filepath)
        with open(filepath, "r") as f:
            content = f.read()
        escaped = content.replace("\n\n", "\n").replace("\n", "\\n")
        if len(escaped) > MAX_CIRCUIT_CHARS:
            escaped = escaped[:MAX_CIRCUIT_CHARS] + " [TRUNCATED]"

        true_label = infer_label(os.path.basename(filename))
        true_cat   = infer_category(os.path.basename(filename))

        y_true_binary.append(true_label)
        y_true_category.append(true_cat)
        circuit_codes.append(escaped)
        circuit_props_list.append(extract_circuit_properties(content))
        filenames.append(os.path.basename(filename))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start = time.time()
    predictions = batch_classify_quantum_circuits(model, tokenizer, circuit_codes, circuit_props_list)
    elapsed = time.time() - start
    print(f"  Inference: {elapsed:.1f}s ({len(circuit_codes) / elapsed:.2f} samples/sec)")

    for filename, prediction in zip(filenames, predictions):
        pred_label, pred_cat = parse_prediction(prediction)
        y_pred_binary.append(pred_label)
        y_pred_category.append(pred_cat)

    # --- Binary metrics ---
    labels_binary = ["safe", "bad"]
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=labels_binary)
    accuracy  = accuracy_score(y_true_binary, y_pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average="weighted", zero_division=0
    )
    class_report = classification_report(
        y_true_binary, y_pred_binary, labels=labels_binary, zero_division=0
    )

    print(f"\n  Fold {fold_num} Results:")
    print(f"  Accuracy={accuracy:.4f}  F1={f1:.4f}  Precision={precision:.4f}  Recall={recall:.4f}")
    print(f"\n  Classification Report:\n{class_report}")

    # --- Confusion matrix plot ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels_binary, yticklabels=labels_binary,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Fold {fold_num} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # --- Classification report text ---
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(f"Fold {fold_num} Classification Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n\n")
        f.write("Confusion Matrix (rows=actual, cols=predicted):\n")
        f.write(f"         {'safe':>8} {'bad':>8}\n")
        f.write(f"safe     {cm[0][0]:>8d} {cm[0][1]:>8d}\n")
        f.write(f"bad      {cm[1][0]:>8d} {cm[1][1]:>8d}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(class_report)
        f.write(f"\nTotal samples: {len(y_true_binary)}\n")
        f.write(f"Correct: {sum(a == b for a, b in zip(y_true_binary, y_pred_binary))}\n")

    # --- Per-category breakdown ---
    categories = ["safe", "immediate", "shuttling", "mixed"]
    by_category = {}
    for cat in categories:
        idx = [i for i, t in enumerate(y_true_category) if t == cat]
        if not idx:
            by_category[cat] = {"support": 0, "accuracy": None, "f1": None,
                                 "precision": None, "recall": None}
            continue
        # For "safe": true label is "safe", predicted "safe" = TP
        # For bad sub-cats: true label is "bad", predicted "bad" = TP
        true_b = [y_true_binary[i] for i in idx]
        pred_b = [y_pred_binary[i] for i in idx]
        cat_acc = accuracy_score(true_b, pred_b)
        # Treat the correct label for this category as positive
        pos_label = "safe" if cat == "safe" else "bad"
        cat_f1        = f1_score(true_b, pred_b, pos_label=pos_label, average="binary", zero_division=0)
        cat_precision = precision_score(true_b, pred_b, pos_label=pos_label, average="binary", zero_division=0)
        cat_recall    = recall_score(true_b, pred_b, pos_label=pos_label, average="binary", zero_division=0)
        by_category[cat] = {
            "support":   len(idx),
            "accuracy":  round(cat_acc, 4),
            "f1":        round(cat_f1, 4),
            "precision": round(cat_precision, 4),
            "recall":    round(cat_recall, 4),
        }

    metrics = {
        "fold": fold_num,
        "overall": {
            "accuracy":  round(accuracy, 4),
            "f1":        round(f1, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
        },
        "by_category": by_category,
        "confusion_matrix": cm.tolist(),
    }

    with open(os.path.join(results_dir, "per_class_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Results saved to {results_dir}/")
    return metrics


# ---------------------------------------------------------------------------
# GRPO reward functions
# ---------------------------------------------------------------------------

def _parse_json_output(completion: str) -> dict | None:
    """Parse a model completion as JSON. Returns None on failure."""
    m = re.search(r"\{[^{}]*\}", completion, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def reward_classification(prompts, completions, **kwargs) -> list[float]:
    """
    Primary reward: +1.0 for correct safe/bad label, -0.5/-0.8 for wrong.

    Key invariant: safe circuits predicted as safe ALWAYS get +1.0 regardless
    of explanation content. The model is never penalized for correctly identifying
    a circuit as benign.
    """
    labels = kwargs.get("label", [])
    rewards = []
    for completion, true_label in zip(completions, labels):
        parsed = _parse_json_output(completion)
        if parsed is None:
            rewards.append(-0.5)
            continue
        safe_val = str(parsed.get("safe", "")).lower().strip()
        pred = "safe" if safe_val == "true" else "bad"
        if pred == true_label:
            rewards.append(1.0)
        else:
            # Penalize bad-circuit misses more harshly (class imbalance compensation)
            rewards.append(-0.8 if true_label == "bad" else -0.5)
    return rewards


def reward_format(prompts, completions, **kwargs) -> list[float]:
    """Valid JSON with all required fields (safe, category, explanation): +0.3. Otherwise -0.1."""
    required = {"safe", "category", "explanation"}
    rewards = []
    for c in completions:
        parsed = _parse_json_output(c)
        if parsed is not None and required.issubset(parsed.keys()):
            rewards.append(0.3)
        else:
            rewards.append(-0.1)
    return rewards


def reward_brevity(prompts, completions, **kwargs) -> list[float]:
    """Explanation brevity: ≤20 words → +0.2, ≤30 words → +0.1, else 0.0."""
    rewards = []
    for c in completions:
        parsed = _parse_json_output(c)
        exp = str(parsed.get("explanation", "")) if parsed else ""
        n = len(exp.split())
        rewards.append(0.2 if n <= 20 else (0.1 if n <= 30 else 0.0))
    return rewards


def reward_category(prompts, completions, **kwargs) -> list[float]:
    """
    Correct sub-category for bad circuits: +0.2.
    Safe circuits: None (excluded from reward sum by GRPOTrainer nan handling).
    """
    labels = kwargs.get("label", [])
    categories = kwargs.get("category", [])
    rewards = []
    for c, tl, tc in zip(completions, labels, categories):
        if tl == "safe":
            rewards.append(float("nan"))  # excluded from weighted sum, completion still kept
            continue
        parsed = _parse_json_output(c)
        if parsed is None:
            rewards.append(0.0)
            continue
        pred_cat = str(parsed.get("category", "")).lower().strip()
        rewards.append(0.2 if pred_cat == tc else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_explanations(explanations_path: str) -> dict:
    """Load explanations.jsonl into a {filename: target_output} dict."""
    lookup = {}
    if not os.path.exists(explanations_path):
        return lookup
    with open(explanations_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                lookup[obj["filename"]] = obj.get("target_output", "")
            except (json.JSONDecodeError, KeyError):
                pass
    return lookup


def load_circuit_data(
    directory: str,
    explanations: dict,
) -> list[dict]:
    """
    Load .qasm files from directory and format as chat messages.
    Uses target_output from explanations if available; otherwise builds a
    minimal JSON target from filename-inferred label/category.
    """
    data = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".qasm"):
            continue

        # Resolve symlink
        filepath = os.path.join(directory, filename)
        if os.path.islink(filepath):
            filepath = os.path.realpath(filepath)

        with open(filepath, "r") as f:
            content = f.read()
        escaped = content.replace("\n\n", "\n").replace("\n", "\\n")
        if len(escaped) > MAX_CIRCUIT_CHARS:
            escaped = escaped[:MAX_CIRCUIT_CHARS] + " [TRUNCATED]"

        props = extract_circuit_properties(content)
        swap = props["gate_counts"].get("swap", 0)
        measure = props["gate_counts"].get("measure", 0)
        total = props["num_gates"]
        mfrac = f"{measure / total:.0%}" if total > 0 else "0%"
        top5 = list(props["gate_counts"].items())[:5]
        summary = (
            f"[Circuit: qubits={props['num_qubits']}, gates={total}, "
            f"swap={swap}, measure={measure}({mfrac}), top_gates={top5}]"
        )
        user_content = f"{summary}\n\nAnalyze this quantum circuit:\n{escaped}"

        base_name = os.path.basename(filename)
        if base_name in explanations and explanations[base_name]:
            target = explanations[base_name]
        else:
            # Fallback: minimal JSON target
            label = infer_label(base_name)
            cat   = infer_category(base_name)
            target = json.dumps({
                "safe": "true" if label == "safe" else "false",
                "category": cat,
                "explanation": "",
            })

        data.append({
            "messages": [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": target},
            ]
        })
    return data


def build_grpo_dataset(directory: str, oversample_ratio: int = 2) -> Dataset:
    """
    Build prompt-only dataset with label/category metadata for GRPO.
    Bad circuits are oversampled by oversample_ratio to address class imbalance.
    """
    safe_rows = []
    bad_rows = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".qasm"):
            continue
        filepath = os.path.join(directory, filename)
        if os.path.islink(filepath):
            filepath = os.path.realpath(filepath)
        with open(filepath, "r") as f:
            content = f.read()
        escaped = content.replace("\n\n", "\n").replace("\n", "\\n")
        if len(escaped) > MAX_CIRCUIT_CHARS:
            escaped = escaped[:MAX_CIRCUIT_CHARS] + " [TRUNCATED]"
        props = extract_circuit_properties(content)
        prompt = create_inference_prompt_with_props(escaped, props)
        label = infer_label(os.path.basename(filename))
        category = infer_category(os.path.basename(filename))
        row = {"prompt": prompt, "label": label, "category": category}
        if label == "safe":
            safe_rows.append(row)
        else:
            bad_rows.append(row)
    all_rows = safe_rows + bad_rows * oversample_ratio
    return Dataset.from_list(all_rows)


# ---------------------------------------------------------------------------
# SFT warm-up phase
# ---------------------------------------------------------------------------

def run_sft_warmup(
    model,
    tokenizer,
    train_dir: str,
    val_dir: str,
    explanations_path: str,
    output_dir: str,
    warmup_steps: int = 60,
) -> None:
    """
    Phase 1: Short SFT warm-up to teach JSON output format.
    Only trains on circuits with clean explanation targets; falls back to all
    circuits with fallback targets if too few clean targets exist.
    """
    explanations = load_explanations(explanations_path)

    def has_real_explanation(target: str) -> bool:
        try:
            obj = json.loads(target)
            exp = obj.get("explanation", "")
            return bool(exp) and "[generation failed" not in exp
        except Exception:
            return False

    clean = {k: v for k, v in explanations.items() if has_real_explanation(v)}
    print(f"  SFT warm-up: {len(clean)} circuits with clean explanations "
          f"(of {len(explanations)} total)")
    target_explanations = clean if len(clean) >= 50 else explanations

    train_raw = load_circuit_data(train_dir, target_explanations)
    val_raw   = load_circuit_data(val_dir,   target_explanations)
    print(f"  SFT train={len(train_raw)}, val={len(val_raw)}")

    train_dataset = Dataset.from_list(train_raw).map(format_data_qwen, batched=True,
                                                     num_proc=min(cpu_count(), 8))
    eval_dataset  = Dataset.from_list(val_raw).map(format_data_qwen, batched=True,
                                                   num_proc=min(cpu_count(), 8))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=min(cpu_count(), 8),
        packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            warmup_ratio=0.1,
            max_steps=warmup_steps,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            eval_strategy="no",
            save_strategy="no",
            optim="adamw_torch_fused",
            weight_decay=0.05,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=output_dir,
            dataloader_num_workers=min(cpu_count(), 8),
            dataloader_pin_memory=True,
            gradient_checkpointing=True,
        ),
    )
    print("  Starting SFT warm-up...")
    trainer.train()
    print("  SFT warm-up complete.")


# ---------------------------------------------------------------------------
# GRPO fine-tuning phase
# ---------------------------------------------------------------------------

def run_grpo_phase(
    model,
    tokenizer,
    train_dir: str,
    output_dir: str,
    grpo_steps: int = 200,
    oversample_ratio: int = 2,
) -> None:
    """
    Phase 2: GRPO fine-tuning with reward-based optimization.

    Correct safe classification always receives +1.0 reward regardless of
    explanation wording — the model is never penalized for correctly detecting
    a benign circuit.
    """
    # Re-cast LoRA weights: the SFT optimizer may have restored them to fp32.
    _compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(_compute_dtype)

    grpo_dataset = build_grpo_dataset(train_dir, oversample_ratio=oversample_ratio)
    print(f"  GRPO dataset: {len(grpo_dataset)} samples "
          f"(bad oversampled {oversample_ratio}x)")

    # H200: single GPU, large VRAM — use generous generation width
    num_generations = 8
    per_device = num_generations
    grad_accum = 2  # effective batch = 16

    grpo_args = GRPOConfig(
        output_dir=output_dir,
        max_prompt_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS_INFERENCE,
        max_completion_length=MAX_NEW_TOKENS_INFERENCE,  # 256 — room for full JSON + explanation
        num_generations=num_generations,
        temperature=0.7,
        beta=0.01,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        max_steps=grpo_steps,
        learning_rate=5e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_strategy="no",
        warmup_ratio=0.05,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        seed=3407,
        reward_weights=[1.0, 0.3, 0.2, 0.2],
        scale_rewards=True,
        dataloader_drop_last=True,
        auto_find_batch_size=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            reward_classification,
            reward_format,
            reward_brevity,
            reward_category,
        ],
        args=grpo_args,
        train_dataset=grpo_dataset,
        processing_class=tokenizer,
    )
    print("  Starting GRPO fine-tuning...")
    trainer.train()
    print("  GRPO fine-tuning complete.")


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------

def train_fold(
    fold_num: int,
    train_dir: str,
    val_dir: str,
    test_dir: str,
    results_dir: str,
    model_output_dir: str,
    explanations_path: str = "explanations.jsonl",
    sft_steps: int = 60,
    grpo_steps: int = 200,
) -> dict:
    """Train and evaluate one fold using SFT warm-up + GRPO fine-tuning."""
    os.makedirs(results_dir,     exist_ok=True)
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  FOLD {fold_num}/5")
    print(f"{'='*60}")

    # Load model
    print(f"  Loading {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        device_map={"": torch.cuda.current_device()},
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    print("  Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=128,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=True,
        loftq_config=None,
    )
    # Cast LoRA trainable weights to match the model's compute dtype so unsloth's
    # fast_lora kernel doesn't raise a dtype mismatch at runtime.
    _compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(_compute_dtype)

    # Phase 1: SFT warm-up (teaches JSON output format)
    sft_output = os.path.join(model_output_dir, "sft_warmup")
    os.makedirs(sft_output, exist_ok=True)
    run_sft_warmup(
        model, tokenizer, train_dir, val_dir, explanations_path,
        sft_output, warmup_steps=sft_steps,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Phase 2: GRPO fine-tuning (reward-based, never punishes correct benign detection)
    grpo_output = os.path.join(model_output_dir, "grpo")
    os.makedirs(grpo_output, exist_ok=True)
    run_grpo_phase(
        model, tokenizer, train_dir, grpo_output, grpo_steps=grpo_steps,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Save final adapter
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"  Model saved to {model_output_dir}/")

    # Switch to inference mode and evaluate
    FastLanguageModel.for_inference(model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    metrics = evaluate_fold(model, tokenizer, val_dir, fold_num, results_dir)

    # Qualitative analysis on the held-out test set
    if test_dir and os.path.isdir(test_dir):
        qualitative = evaluate_fold_qualitative(model, tokenizer, test_dir, fold_num, results_dir)
        metrics["test_qualitative_summary"] = {
            "n_samples": len(qualitative),
            "n_correct": sum(1 for q in qualitative if q["correct"]),
            "accuracy":  round(sum(1 for q in qualitative if q["correct"]) / len(qualitative), 4)
                         if qualitative else None,
        }

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="5-fold cross-validation training for bad-qubits (Qwen 2.5 Coder)"
    )
    parser.add_argument("--fold", type=int, default=None,
                        help="Run a single fold (1-5). If omitted, runs all 5 folds.")
    parser.add_argument("--data-dir",     default="data")
    parser.add_argument("--results-dir",  default="results")
    parser.add_argument("--models-dir",   default="models")
    parser.add_argument("--explanations", default="explanations.jsonl")
    parser.add_argument("--sft-steps",  type=int, default=60,
                        help="SFT warm-up steps per fold (default: 60)")
    parser.add_argument("--grpo-steps", type=int, default=200,
                        help="GRPO fine-tuning steps per fold (default: 200)")
    args = parser.parse_args()

    setup_cuda()

    folds_to_run = [args.fold] if args.fold else list(range(1, 6))

    all_metrics = []
    for fold_num in folds_to_run:
        train_dir     = os.path.join(args.data_dir,    f"fold_{fold_num}", "train")
        val_dir       = os.path.join(args.data_dir,    f"fold_{fold_num}", "val")
        test_dir      = os.path.join(args.data_dir,    f"fold_{fold_num}", "test")
        results_dir   = os.path.join(args.results_dir, f"fold_{fold_num}")
        model_out_dir = os.path.join(args.models_dir,  f"fold_{fold_num}")

        if not os.path.isdir(train_dir):
            print(f"ERROR: Train directory not found: {train_dir}")
            print("Run 'python main.py prepare' first.")
            return

        metrics = train_fold(
            fold_num=fold_num,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            results_dir=results_dir,
            model_output_dir=model_out_dir,
            explanations_path=args.explanations,
            sft_steps=args.sft_steps,
            grpo_steps=args.grpo_steps,
        )
        all_metrics.append(metrics)

        # Free VRAM before next fold
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print("  CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        for m in all_metrics:
            ov = m["overall"]
            print(f"  Fold {m['fold']}: accuracy={ov['accuracy']:.4f}  "
                  f"f1={ov['f1']:.4f}  precision={ov['precision']:.4f}  "
                  f"recall={ov['recall']:.4f}")
        import numpy as np
        accs = [m["overall"]["accuracy"]  for m in all_metrics]
        f1s  = [m["overall"]["f1"]        for m in all_metrics]
        print(f"\n  Mean accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  Mean F1:       {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"\nRun 'python main.py aggregate' to generate the full cv_summary.json")


if __name__ == "__main__":
    main()

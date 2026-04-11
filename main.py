#!/usr/bin/env python3
"""
main.py - Bad Qubits: Phase 2 pipeline entry point.

Classifies quantum circuits as "safe" or "bad" (exhibiting negative effects)
using Qwen 2.5 Coder 7B fine-tuned with LoRA and 5-fold cross-validation.

USAGE
  python main.py prepare              Rename dataset files + build 5-fold splits
  python main.py explain [--limit N]  Generate explanations.jsonl (LLM-based)
  python main.py train [--fold N]     Train all 5 folds, or a single fold
  python main.py aggregate            Aggregate fold results into cv_summary.json
  python main.py run-all              Run the full pipeline end-to-end
  python main.py classify <file>      Classify a single .qasm file
"""

import argparse
import json
import os
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def header(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}\n")


def check_prereq(path: str, step: str) -> None:
    """Exit with a friendly message if a required file/dir is missing."""
    if not os.path.exists(path):
        print(f"ERROR: Required path not found: {path}")
        print(f"       Run 'python main.py {step}' first.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_prepare(args) -> None:
    """Rename dataset files, sanitize register names, and build fold splits."""
    header("STEP 1 — Prepare Dataset")
    from prepare_dataset import (
        rename_dataset, write_master_manifest,
        build_fold_manifest, write_fold_directories, print_fold_summary,
    )

    dataset_dir = args.dataset_dir
    data_dir    = args.data_dir
    dry_run     = getattr(args, "dry_run", False)

    print(f"Dataset directory : {dataset_dir}/")
    print(f"Output directory  : {data_dir}/")
    print(f"Folds             : {args.n_folds}")
    print(f"Seed              : {args.seed}")
    if dry_run:
        print(f"Mode              : DRY RUN (no changes made)\n")

    # Rename + sanitize
    print(f"[1/3] Renaming malicious_* → bad_* and sanitizing register names...")
    n_renamed, n_sanitized = rename_dataset(dataset_dir, dry_run=dry_run)
    if not dry_run:
        print(f"      Renamed: {n_renamed}  |  Sanitized: {n_sanitized}")

    # Manifest
    print(f"\n[2/3] Building master manifest (data/all_filenames.json)...")
    manifest_path = os.path.join(data_dir, "all_filenames.json")
    if dry_run:
        from prepare_dataset import build_file_list
        entries = build_file_list(dataset_dir)
        print(f"      Would write {len(entries)} entries to {manifest_path}")
    else:
        entries = write_master_manifest(dataset_dir, manifest_path)

    from collections import Counter
    cat_counts = Counter(e["category"] for e in entries)
    print("      Categories: " + ", ".join(f"{k}={v}" for k, v in sorted(cat_counts.items())))

    # Folds
    print(f"\n[3/3] Building stratified {args.n_folds}-fold splits...")
    folds = build_fold_manifest(entries, n_folds=args.n_folds, seed=args.seed)
    if dry_run:
        print_fold_summary(folds, entries)
        print("Dry run complete. No files were modified.")
    else:
        write_fold_directories(folds, dataset_dir, data_dir)
        print_fold_summary(folds, entries)
        print(f"Done. Fold splits written to '{data_dir}/'")
        print("Next: python main.py explain")


def cmd_explain(args) -> None:
    """Generate explanations.jsonl using GLM-5 via ZhipuAI API."""
    header("STEP 2 — Generate Explanations (GLM-5)")
    check_prereq("data/all_filenames.json", "prepare")

    api_key = getattr(args, "api_key", None)
    if not api_key:
        print("ERROR: --api-key is required for the explain command.")
        sys.exit(1)

    from generate_explanations import generate_explanations

    limit  = getattr(args, "limit", None)
    resume = not getattr(args, "no_resume", False)
    delay  = getattr(args, "delay", 0.5)

    print(f"Model      : glm-5 (ZhipuAI paas/v4)")
    print(f"Dataset    : {args.dataset_dir}/")
    print(f"Manifest   : {args.manifest}")
    print(f"Output     : {args.output}")
    if limit:
        print(f"Limit      : {limit} circuits")
    print(f"Resume     : {resume}")
    print(f"Req delay  : {delay}s\n")

    generate_explanations(
        api_key=api_key,
        dataset_dir=args.dataset_dir,
        manifest_path=args.manifest,
        output_jsonl=args.output,
        limit=limit,
        resume=resume,
        request_delay=delay,
    )
    print("\nNext: python main.py train")


def cmd_train(args) -> None:
    """Run 5-fold CV training (or a single fold with --fold N)."""
    header("STEP 3 — Train (5-fold Cross-Validation)")
    check_prereq("data", "prepare")

    from train_cv import setup_cuda, train_fold
    import gc
    import torch
    import numpy as np

    setup_cuda()

    fold_arg = getattr(args, "fold", None)
    folds_to_run = [fold_arg] if fold_arg else list(range(1, 6))

    sft_steps  = getattr(args, "sft_steps",  60)
    grpo_steps = getattr(args, "grpo_steps", 200)

    print(f"Folds to run  : {folds_to_run}")
    print(f"Data dir      : {args.data_dir}/")
    print(f"Results dir   : {args.results_dir}/")
    print(f"Models dir    : {args.models_dir}/")
    print(f"Explanations  : {args.explanations}")
    print(f"SFT steps     : {sft_steps}")
    print(f"GRPO steps    : {grpo_steps}\n")

    all_metrics = []
    for fold_num in folds_to_run:
        train_dir     = os.path.join(args.data_dir,    f"fold_{fold_num}", "train")
        val_dir       = os.path.join(args.data_dir,    f"fold_{fold_num}", "val")
        test_dir      = os.path.join(args.data_dir,    f"fold_{fold_num}", "test")
        results_dir   = os.path.join(args.results_dir, f"fold_{fold_num}")
        model_out_dir = os.path.join(args.models_dir,  f"fold_{fold_num}")

        if not os.path.isdir(train_dir):
            print(f"ERROR: {train_dir} not found. Run 'python main.py prepare' first.")
            sys.exit(1)

        metrics = train_fold(
            fold_num=fold_num,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            results_dir=results_dir,
            model_output_dir=model_out_dir,
            explanations_path=args.explanations,
            sft_steps=sft_steps,
            grpo_steps=grpo_steps,
        )
        all_metrics.append(metrics)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_metrics) > 1:
        print(f"\n{'='*50}")
        print("  CROSS-VALIDATION SUMMARY")
        print(f"{'='*50}")
        for m in all_metrics:
            ov = m["overall"]
            print(f"  Fold {m['fold']}: accuracy={ov['accuracy']:.4f}  "
                  f"f1={ov['f1']:.4f}")
        accs = [m["overall"]["accuracy"] for m in all_metrics]
        f1s  = [m["overall"]["f1"]       for m in all_metrics]
        print(f"\n  Mean accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  Mean F1:       {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    print("\nNext: python main.py aggregate")


def cmd_aggregate(args) -> None:
    """Aggregate per-fold results into cv_summary.json."""
    header("STEP 4 — Aggregate Results")
    check_prereq(args.results_dir, "train")

    from aggregate_results import load_fold_metrics, compute_aggregate_stats, plot_avg_confusion_matrix
    import json

    fold_metrics = load_fold_metrics(args.results_dir, args.n_folds)
    if not fold_metrics:
        print("No fold metrics found. Run 'python main.py train' first.")
        sys.exit(1)

    summary = compute_aggregate_stats(fold_metrics)

    agg_dir = os.path.join(args.results_dir, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)

    summary_path = os.path.join(agg_dir, "cv_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")

    plot_avg_confusion_matrix(fold_metrics, os.path.join(agg_dir, "avg_confusion_matrix.png"))

    print(f"\n{'='*50}")
    print(f"  CROSS-VALIDATION RESULTS ({summary['n_folds']} folds)")
    print(f"{'='*50}")
    print(f"  Accuracy:  {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"  F1:        {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    print(f"  Precision: {summary['mean_precision']:.4f} ± {summary['std_precision']:.4f}")
    print(f"  Recall:    {summary['mean_recall']:.4f} ± {summary['std_recall']:.4f}")
    print(f"\n  Per-fold:")
    for pf in summary.get("per_fold", []):
        print(f"    Fold {pf['fold']}: acc={pf['accuracy']:.4f}  f1={pf['f1']:.4f}")
    print(f"\n  Per-category (mean F1):")
    for cat, stats in summary.get("per_category", {}).items():
        f1_mean = stats.get("mean_f1", "n/a")
        f1_std  = stats.get("std_f1",  "")
        f1_str  = f"{f1_mean:.4f} ± {f1_std:.4f}" if isinstance(f1_mean, float) else str(f1_mean)
        print(f"    {cat:<12}: f1={f1_str}")


def cmd_run_all(args) -> None:
    """Run the full pipeline: prepare → explain → train → aggregate."""
    header("FULL PIPELINE")
    print("Running: prepare → explain → train → aggregate\n")

    # Patch args for each step
    args.dry_run = False
    cmd_prepare(args)
    cmd_explain(args)
    cmd_train(args)
    cmd_aggregate(args)

    header("PIPELINE COMPLETE")
    print("Outputs:")
    print(f"  explanations.jsonl            — circuit explainability spec sheet")
    print(f"  results/aggregate/cv_summary.json  — cross-validation metrics")
    print(f"  results/aggregate/avg_confusion_matrix.png")
    print(f"  models/fold_{{1-5}}/            — per-fold LoRA adapters")


def cmd_classify(args) -> None:
    """Classify a single .qasm file using the trained model (fold 1 by default)."""
    header("CLASSIFY")

    circuit_file = args.circuit_file
    if not os.path.exists(circuit_file):
        print(f"ERROR: File not found: {circuit_file}")
        sys.exit(1)

    model_dir = getattr(args, "model_dir", None) or os.path.join(args.models_dir, "fold_1")
    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        print("Run 'python main.py train --fold 1' first.")
        sys.exit(1)

    import torch
    from unsloth import FastLanguageModel
    from train_cv import create_inference_prompt_with_props, parse_prediction, MAX_SEQ_LENGTH, MAX_NEW_TOKENS_INFERENCE
    from generate_explanations import extract_circuit_properties

    print(f"Circuit : {circuit_file}")
    print(f"Model   : {model_dir}\n")

    with open(circuit_file, "r") as f:
        content = f.read()
    escaped = content.replace("\n\n", "\n").replace("\n", "\\n")
    if len(escaped) > 4096:
        escaped = escaped[:4096] + " [TRUNCATED]"
    props = extract_circuit_properties(content)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)

    prompt = create_inference_prompt_with_props(escaped, props)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=10000, add_special_tokens=False).to(device)

    with torch.amp.autocast("cuda", enabled=torch.cuda.is_bf16_supported()):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=MAX_NEW_TOKENS_INFERENCE,
                use_cache=False,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=1,
            )

    input_length = inputs.input_ids.shape[1]
    raw = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

    label, category = parse_prediction(raw)
    print(f"Raw output:\n{raw}\n")

    # Pretty print the result
    try:
        result = json.loads(raw)
        print(json.dumps(result, indent=2))
    except json.JSONDecodeError:
        print(f"Label    : {label}")
        print(f"Category : {category}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared arguments added to each sub-parser
    def add_dirs(p):
        p.add_argument("--dataset-dir", default="dataset")
        p.add_argument("--data-dir",    default="data")
        p.add_argument("--results-dir", default="results")
        p.add_argument("--models-dir",  default="models")

    # --- prepare ---
    p_prep = subparsers.add_parser("prepare", help="Rename dataset and build fold splits")
    add_dirs(p_prep)
    p_prep.add_argument("--n-folds", type=int, default=5)
    p_prep.add_argument("--seed",    type=int, default=42)
    p_prep.add_argument("--dry-run", action="store_true",
                        help="Show what would change without modifying files")
    p_prep.set_defaults(func=cmd_prepare)

    # --- explain ---
    p_exp = subparsers.add_parser("explain", help="Generate explanations.jsonl")
    add_dirs(p_exp)
    p_exp.add_argument("--api-key",   required=True, help="ZhipuAI API key")
    p_exp.add_argument("--manifest",  default="data/all_filenames.json")
    p_exp.add_argument("--output",    default="explanations.jsonl")
    p_exp.add_argument("--limit",     type=int, default=None,
                       help="Process at most N circuits (for testing)")
    p_exp.add_argument("--no-resume", action="store_true",
                       help="Reprocess all circuits even if output already exists")
    p_exp.add_argument("--delay",     type=float, default=0.5,
                       help="Seconds between API requests (default: 0.5)")
    p_exp.set_defaults(func=cmd_explain)

    # --- train ---
    p_train = subparsers.add_parser("train", help="Run 5-fold CV training")
    add_dirs(p_train)
    p_train.add_argument("--fold",         type=int, default=None,
                         help="Run only this fold (1-5)")
    p_train.add_argument("--explanations", default="explanations.jsonl")
    p_train.add_argument("--sft-steps",    type=int, default=60,
                         help="SFT warm-up steps per fold (default: 60)")
    p_train.add_argument("--grpo-steps",   type=int, default=200,
                         help="GRPO fine-tuning steps per fold (default: 200)")
    p_train.set_defaults(func=cmd_train)

    # --- aggregate ---
    p_agg = subparsers.add_parser("aggregate", help="Aggregate fold results")
    add_dirs(p_agg)
    p_agg.add_argument("--n-folds", type=int, default=5)
    p_agg.set_defaults(func=cmd_aggregate)

    # --- run-all ---
    p_all = subparsers.add_parser("run-all", help="Run full pipeline end-to-end")
    add_dirs(p_all)
    p_all.add_argument("--n-folds",    type=int, default=5)
    p_all.add_argument("--seed",       type=int, default=42)
    p_all.add_argument("--fold",       type=int, default=None)
    p_all.add_argument("--manifest",   default="data/all_filenames.json")
    p_all.add_argument("--output",     default="explanations.jsonl")
    p_all.add_argument("--limit",      type=int, default=None)
    p_all.add_argument("--no-resume",  action="store_true")
    p_all.add_argument("--explanations", default="explanations.jsonl")
    p_all.add_argument("--sft-steps",  type=int, default=60)
    p_all.add_argument("--grpo-steps", type=int, default=200)
    p_all.set_defaults(func=cmd_run_all)

    # --- classify ---
    p_cls = subparsers.add_parser("classify", help="Classify a single .qasm file")
    add_dirs(p_cls)
    p_cls.add_argument("circuit_file", help="Path to .qasm file to classify")
    p_cls.add_argument("--model-dir",  default=None,
                       help="Path to LoRA adapter directory (default: models/fold_1)")
    p_cls.set_defaults(func=cmd_classify)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

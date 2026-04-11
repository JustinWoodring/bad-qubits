#!/usr/bin/env python3
"""
aggregate_results.py - Aggregate 5-fold cross-validation results into a summary.

Reads results/fold_{1-5}/per_class_metrics.json and produces:
  results/aggregate/cv_summary.json
  results/aggregate/avg_confusion_matrix.png

Run via: python main.py aggregate
Or directly: python aggregate_results.py [--results-dir results] [--n-folds 5]
"""

import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Load fold metrics
# ---------------------------------------------------------------------------

def load_fold_metrics(results_dir: str, n_folds: int = 5) -> list[dict]:
    """Read per_class_metrics.json from each fold_N/ subdirectory."""
    metrics = []
    for fold_num in range(1, n_folds + 1):
        path = os.path.join(results_dir, f"fold_{fold_num}", "per_class_metrics.json")
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping fold {fold_num}")
            continue
        with open(path, "r") as f:
            metrics.append(json.load(f))
    return metrics


def load_fold_qualitative(results_dir: str, n_folds: int = 5) -> list[dict]:
    """Read test_qualitative.json from each fold_N/ subdirectory."""
    qualitative = []
    for fold_num in range(1, n_folds + 1):
        path = os.path.join(results_dir, f"fold_{fold_num}", "test_qualitative.json")
        if not os.path.exists(path):
            print(f"  Note: {path} not found, skipping qualitative for fold {fold_num}")
            continue
        with open(path, "r") as f:
            qualitative.append(json.load(f))
    return qualitative


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------

def compute_aggregate_stats(fold_metrics: list[dict]) -> dict:
    """Compute mean ± std across folds for all metrics."""
    if not fold_metrics:
        return {}

    overall_keys = ["accuracy", "f1", "precision", "recall"]
    agg_overall = {}
    for key in overall_keys:
        vals = [m["overall"][key] for m in fold_metrics if m["overall"].get(key) is not None]
        agg_overall[f"mean_{key}"] = round(float(np.mean(vals)), 4)
        agg_overall[f"std_{key}"]  = round(float(np.std(vals)),  4)

    # Per-category aggregation
    categories = ["safe", "immediate", "shuttling", "mixed"]
    cat_keys = ["f1", "precision", "recall", "accuracy"]
    agg_by_category = {}
    for cat in categories:
        agg_by_category[cat] = {}
        for key in cat_keys:
            vals = [
                m["by_category"][cat][key]
                for m in fold_metrics
                if cat in m.get("by_category", {})
                and m["by_category"][cat].get(key) is not None
            ]
            if vals:
                agg_by_category[cat][f"mean_{key}"] = round(float(np.mean(vals)), 4)
                agg_by_category[cat][f"std_{key}"]  = round(float(np.std(vals)),  4)
            support_vals = [
                m["by_category"][cat]["support"]
                for m in fold_metrics
                if cat in m.get("by_category", {})
            ]
            if support_vals:
                agg_by_category[cat]["mean_support"] = round(float(np.mean(support_vals)), 1)

    per_fold = [
        {
            "fold":      m["fold"],
            "accuracy":  m["overall"]["accuracy"],
            "f1":        m["overall"]["f1"],
            "precision": m["overall"]["precision"],
            "recall":    m["overall"]["recall"],
        }
        for m in fold_metrics
    ]

    return {
        "n_folds":         len(fold_metrics),
        **agg_overall,
        "per_fold":        per_fold,
        "per_category":    agg_by_category,
    }


def compute_qualitative_summary(fold_qualitative: list[dict]) -> dict:
    """
    Aggregate test-set qualitative analysis across folds.

    Returns:
      - per-fold test accuracy
      - mean ± std test accuracy across folds
      - per-category test accuracy (mean ± std)
      - representative examples: one correct + one incorrect per category,
        drawn from the fold where each example appears
    """
    if not fold_qualitative:
        return {}

    categories = ["safe", "immediate", "shuttling", "mixed"]

    # Per-fold test accuracy
    per_fold_test = []
    for fold_data in fold_qualitative:
        samples = fold_data["samples"]
        n = len(samples)
        n_correct = sum(1 for s in samples if s["correct"])
        per_fold_test.append({
            "fold":     fold_data["fold"],
            "n":        n,
            "n_correct": n_correct,
            "accuracy": round(n_correct / n, 4) if n else None,
        })

    accs = [pf["accuracy"] for pf in per_fold_test if pf["accuracy"] is not None]
    mean_test_acc = round(float(np.mean(accs)), 4) if accs else None
    std_test_acc  = round(float(np.std(accs)),  4) if accs else None

    # Per-category accuracy across folds
    cat_accs: dict[str, list[float]] = {c: [] for c in categories}
    for fold_data in fold_qualitative:
        samples = fold_data["samples"]
        for cat in categories:
            cat_samples = [s for s in samples if s["true_category"] == cat]
            if cat_samples:
                cat_acc = sum(1 for s in cat_samples if s["correct"]) / len(cat_samples)
                cat_accs[cat].append(cat_acc)

    per_category_test = {}
    for cat in categories:
        vals = cat_accs[cat]
        per_category_test[cat] = {
            "mean_accuracy": round(float(np.mean(vals)), 4) if vals else None,
            "std_accuracy":  round(float(np.std(vals)),  4) if vals else None,
            "n_folds":       len(vals),
        }

    # Representative examples: 1 correct + 1 incorrect per category
    # Prefer examples where the explanation is non-empty
    examples: dict[str, dict] = {}
    for cat in categories:
        correct_ex   = None
        incorrect_ex = None
        for fold_data in fold_qualitative:
            for s in fold_data["samples"]:
                if s["true_category"] != cat:
                    continue
                has_exp = bool(s.get("explanation", "").strip())
                if s["correct"] and correct_ex is None:
                    if has_exp or correct_ex is None:
                        correct_ex = {**s, "fold": fold_data["fold"]}
                if not s["correct"] and incorrect_ex is None:
                    if has_exp or incorrect_ex is None:
                        incorrect_ex = {**s, "fold": fold_data["fold"]}
            if correct_ex and incorrect_ex:
                break
        examples[cat] = {
            "correct_example":   correct_ex,
            "incorrect_example": incorrect_ex,
        }

    return {
        "mean_test_accuracy": mean_test_acc,
        "std_test_accuracy":  std_test_acc,
        "per_fold_test":      per_fold_test,
        "per_category_test":  per_category_test,
        "representative_examples": examples,
    }


# ---------------------------------------------------------------------------
# Averaged confusion matrix plot
# ---------------------------------------------------------------------------

def plot_avg_confusion_matrix(fold_metrics: list[dict], output_path: str) -> None:
    """Plot row-normalized average confusion matrix across folds."""
    cms = [np.array(m["confusion_matrix"]) for m in fold_metrics if "confusion_matrix" in m]
    if not cms:
        print("  No confusion matrix data found.")
        return

    avg_cm = np.mean(cms, axis=0)

    # Row-normalize
    row_sums = avg_cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(avg_cm, row_sums, where=row_sums != 0)

    labels = ["safe", "bad"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        norm_cm,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Average Confusion Matrix ({len(cms)}-fold CV, row-normalized)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate cross-validation results into a summary."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    print(f"\nLoading fold metrics from '{args.results_dir}/'...")
    fold_metrics = load_fold_metrics(args.results_dir, args.n_folds)

    if not fold_metrics:
        print("No fold metrics found. Run 'python main.py train' first.")
        return

    print(f"  Loaded {len(fold_metrics)} fold(s)")

    fold_qualitative = load_fold_qualitative(args.results_dir, args.n_folds)

    summary = compute_aggregate_stats(fold_metrics)
    if fold_qualitative:
        summary["qualitative_analysis"] = compute_qualitative_summary(fold_qualitative)

    agg_dir = os.path.join(args.results_dir, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)

    # Write summary JSON
    summary_path = os.path.join(agg_dir, "cv_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved: {summary_path}")

    # Plot averaged confusion matrix
    plot_avg_confusion_matrix(fold_metrics, os.path.join(agg_dir, "avg_confusion_matrix.png"))

    # Print validation summary
    print(f"\n{'='*50}")
    print(f"  CROSS-VALIDATION RESULTS ({summary['n_folds']} folds, val set)")
    print(f"{'='*50}")
    print(f"  Accuracy:  {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"  F1:        {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    print(f"  Precision: {summary['mean_precision']:.4f} ± {summary['std_precision']:.4f}")
    print(f"  Recall:    {summary['mean_recall']:.4f} ± {summary['std_recall']:.4f}")
    print(f"\n  Per-fold breakdown (val):")
    for pf in summary.get("per_fold", []):
        print(f"    Fold {pf['fold']}: acc={pf['accuracy']:.4f}  f1={pf['f1']:.4f}")
    print(f"\n  Per-category val (mean F1):")
    for cat, stats in summary.get("per_category", {}).items():
        f1_mean = stats.get("mean_f1", "n/a")
        f1_std  = stats.get("std_f1",  "n/a")
        support = stats.get("mean_support", "?")
        f1_str  = f"{f1_mean:.4f} ± {f1_std:.4f}" if isinstance(f1_mean, float) else str(f1_mean)
        print(f"    {cat:<12}: f1={f1_str}  (avg support={support})")

    # Print qualitative test set summary
    qa = summary.get("qualitative_analysis")
    if qa:
        print(f"\n{'='*50}")
        print(f"  HELD-OUT TEST SET (qualitative analysis)")
        print(f"{'='*50}")
        mean_acc = qa.get("mean_test_accuracy")
        std_acc  = qa.get("std_test_accuracy")
        if mean_acc is not None:
            print(f"  Test accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"\n  Per-fold test accuracy:")
        for pf in qa.get("per_fold_test", []):
            acc_str = f"{pf['accuracy']:.4f}" if pf["accuracy"] is not None else "n/a"
            print(f"    Fold {pf['fold']}: {acc_str}  ({pf['n_correct']}/{pf['n']})")
        print(f"\n  Per-category test accuracy:")
        for cat, stats in qa.get("per_category_test", {}).items():
            m = stats.get("mean_accuracy")
            s = stats.get("std_accuracy")
            acc_str = f"{m:.4f} ± {s:.4f}" if m is not None else "n/a"
            print(f"    {cat:<12}: {acc_str}")
        print(f"\n  Representative examples (see cv_summary.json for full details):")
        for cat, exs in qa.get("representative_examples", {}).items():
            print(f"\n  [{cat}]")
            for kind in ("correct_example", "incorrect_example"):
                ex = exs.get(kind)
                if ex:
                    label = "CORRECT  " if kind == "correct_example" else "INCORRECT"
                    exp   = ex.get("explanation", "").strip() or "(no explanation)"
                    print(f"    {label} fold={ex['fold']} file={ex['filename']}")
                    print(f"             pred={ex['pred_label']}/{ex['pred_category']}  "
                          f"true={ex['true_label']}/{ex['true_category']}")
                    print(f"             explanation: \"{exp}\"")


if __name__ == "__main__":
    main()

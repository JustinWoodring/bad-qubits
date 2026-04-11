#!/usr/bin/env python3
"""
prepare_dataset.py - Rename dataset files and build stratified 5-fold splits.

Run via: python main.py prepare
Or directly: python prepare_dataset.py [--dry-run] [--n-folds 5] [--seed 42]
"""

import os
import json
import argparse
import random
from pathlib import Path
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Shared category utilities (imported by other scripts)
# ---------------------------------------------------------------------------

def infer_category(filename: str) -> str:
    """Infer circuit category from filename."""
    name = os.path.basename(filename)
    if "bad_immediate_measurement" in name:
        return "immediate"
    elif "bad_qubit_shuttling" in name:
        return "shuttling"
    elif "bad_mixed" in name:
        return "mixed"
    else:
        return "safe"


def infer_label(filename: str) -> str:
    """Infer binary label from filename."""
    name = os.path.basename(filename)
    return "bad" if name.startswith("bad_") else "safe"


# ---------------------------------------------------------------------------
# Rename logic
# ---------------------------------------------------------------------------

RENAME_MAP = [
    ("malicious_immediate_measurement_", "bad_immediate_measurement_"),
    ("malicious_qubit_shuttling_",       "bad_qubit_shuttling_"),
    ("malicious_trojan_attack_",         "bad_mixed_"),
]

# Register names that reveal circuit intent — replaced with neutral names.
# Format: (old_register_name, new_register_name)
# These appear in QASM as "creg early[N]" or "qreg control[1]" etc.
REGISTER_SANITIZE_MAP = [
    ("early",   "mout"),    # creg early[N] in immediate_measurement circuits
    ("control", "anc"),     # qreg control[1] in trojan/mixed circuits
]


def sanitize_registers(content: str) -> str:
    """
    Replace telling register names with neutral names.
    Handles both declaration lines and all usage references throughout the file.
    Uses whole-word replacement to avoid partial matches.
    """
    import re
    for old_name, new_name in REGISTER_SANITIZE_MAP:
        # Replace as whole-word identifier (e.g. "early[" or "early " but not "early_something")
        content = re.sub(
            rf'\b{re.escape(old_name)}\b',
            new_name,
            content,
        )
    return content


def rename_dataset(dataset_dir: str, dry_run: bool = False) -> tuple[int, int]:
    """
    Rename malicious_* files to bad_* in-place within dataset_dir,
    and sanitize telling register names within each file.
    Idempotent — already-renamed files are still checked for register sanitization.
    Returns (files_renamed, files_sanitized).
    """
    renamed = 0
    sanitized = 0
    for filename in sorted(os.listdir(dataset_dir)):
        if not filename.endswith(".qasm"):
            continue

        new_name = filename
        for old_prefix, new_prefix in RENAME_MAP:
            if filename.startswith(old_prefix):
                new_name = new_prefix + filename[len(old_prefix):]
                break

        src = os.path.join(dataset_dir, filename)
        dst = os.path.join(dataset_dir, new_name)

        # Read current content
        with open(src, "r") as f:
            original = f.read()
        sanitized_content = sanitize_registers(original)

        if dry_run:
            if new_name != filename:
                print(f"  [dry-run] rename: {filename} -> {new_name}")
            if sanitized_content != original:
                # Show which register names would be replaced
                for old_r, new_r in REGISTER_SANITIZE_MAP:
                    if f"creg {old_r}[" in original or f"qreg {old_r}[" in original:
                        print(f"  [dry-run] sanitize register '{old_r}' -> '{new_r}' in {filename}")
        else:
            # Write sanitized content (even if filename doesn't change)
            if sanitized_content != original:
                with open(src, "w") as f:
                    f.write(sanitized_content)
                sanitized += 1

            # Rename file if needed (after writing sanitized content to src)
            if new_name != filename:
                os.rename(src, dst)
                renamed += 1

    return renamed, sanitized


# ---------------------------------------------------------------------------
# Manifest and fold building
# ---------------------------------------------------------------------------

def build_file_list(dataset_dir: str) -> list[dict]:
    """Return sorted list of {filename, label, category} for all .qasm files."""
    entries = []
    for filename in sorted(os.listdir(dataset_dir)):
        if not filename.endswith(".qasm"):
            continue
        entries.append({
            "filename": filename,
            "label": infer_label(filename),
            "category": infer_category(filename),
        })
    return entries


def write_master_manifest(dataset_dir: str, output_path: str) -> list[dict]:
    """Write data/all_filenames.json and return the list."""
    entries = build_file_list(dataset_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Manifest written: {output_path} ({len(entries)} files)")
    return entries


def build_fold_manifest(
    entries: list[dict], n_folds: int = 5, seed: int = 42
) -> dict:
    """
    Build 70/20/10 train/val/test splits for each fold using stratified sampling.

    Each fold uses a different random seed so the splits are distinct.
    Stratification is on the 4-class category label (safe/immediate/shuttling/mixed).

    Returns {fold_num: {"train": [...], "val": [...], "test": [...]}}
    """
    filenames = [e["filename"] for e in entries]
    strat_labels = [e["category"] for e in entries]
    indices = list(range(len(filenames)))

    folds = {}
    for fold_num in range(1, n_folds + 1):
        fold_seed = seed + fold_num - 1

        # Step 1: hold out 10% as test (stratified)
        trainval_idx, test_idx = train_test_split(
            indices,
            test_size=0.10,
            stratify=strat_labels,
            random_state=fold_seed,
        )

        # Step 2: split remaining 90% into train (70% of total) and val (20% of total)
        # 20/90 = 2/9 ≈ 0.2222
        trainval_strat = [strat_labels[i] for i in trainval_idx]
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=2 / 9,
            stratify=trainval_strat,
            random_state=fold_seed,
        )

        folds[fold_num] = {
            "train": [filenames[i] for i in train_idx],
            "val":   [filenames[i] for i in val_idx],
            "test":  [filenames[i] for i in test_idx],
        }
    return folds


def write_fold_directories(
    folds: dict,
    dataset_dir: str,
    data_dir: str,
    dry_run: bool = False,
) -> None:
    """
    Create data/fold_N/train/, data/fold_N/val/, and data/fold_N/test/ with symlinks.
    Removes stale symlinks before recreating.
    """
    dataset_abs = os.path.abspath(dataset_dir)

    for fold_num, split in folds.items():
        for split_name, filenames in split.items():
            split_dir = os.path.join(data_dir, f"fold_{fold_num}", split_name)
            if not dry_run:
                os.makedirs(split_dir, exist_ok=True)
                # Remove stale symlinks
                for existing in os.listdir(split_dir):
                    link_path = os.path.join(split_dir, existing)
                    if os.path.islink(link_path):
                        os.unlink(link_path)

            for filename in filenames:
                src = os.path.join(dataset_abs, filename)
                dst = os.path.join(split_dir, filename)
                if dry_run:
                    pass  # skip verbose symlink output
                else:
                    if not os.path.exists(dst):
                        os.symlink(src, dst)

        if not dry_run:
            train_count = len(split["train"])
            val_count   = len(split["val"])
            test_count  = len(split["test"])
            bad_train   = sum(1 for f in split["train"] if f.startswith("bad_"))
            bad_val     = sum(1 for f in split["val"]   if f.startswith("bad_"))
            bad_test    = sum(1 for f in split["test"]  if f.startswith("bad_"))
            print(
                f"  Fold {fold_num}: train={train_count} ({bad_train} bad), "
                f"val={val_count} ({bad_val} bad), "
                f"test={test_count} ({bad_test} bad)"
            )


def print_fold_summary(folds: dict, entries: list[dict]) -> None:
    """Print per-fold category distribution for dry-run or verification."""
    cat_order = ["safe", "immediate", "shuttling", "mixed"]
    entry_map = {e["filename"]: e for e in entries}

    print(f"\n{'Fold':<6} {'Split':<6} {'Total':>6}  " +
          "  ".join(f"{c:>10}" for c in cat_order))
    print("-" * 70)

    for fold_num in sorted(folds.keys()):
        for split_name in ("train", "val", "test"):
            filenames = folds[fold_num][split_name]
            counts = {c: 0 for c in cat_order}
            for fn in filenames:
                cat = entry_map[fn]["category"]
                counts[cat] = counts.get(cat, 0) + 1
            print(
                f"{fold_num:<6} {split_name:<6} {len(filenames):>6}  " +
                "  ".join(f"{counts[c]:>10}" for c in cat_order)
            )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare bad-qubits dataset: rename files and build fold splits."
    )
    parser.add_argument("--dataset-dir", default="dataset",
                        help="Directory containing .qasm circuit files (default: dataset)")
    parser.add_argument("--data-dir", default="data",
                        help="Output directory for fold splits (default: data)")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for stratified split (default: 42)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be renamed/created without making changes")
    args = parser.parse_args()

    # Step 1: Rename + sanitize
    print(f"\n[1/3] Renaming malicious_* → bad_* and sanitizing register names in '{args.dataset_dir}/'...")
    n_renamed, n_sanitized = rename_dataset(args.dataset_dir, dry_run=args.dry_run)
    if args.dry_run:
        print(f"  (dry-run complete for this step)")
    else:
        if n_renamed > 0:
            print(f"  Renamed {n_renamed} files.")
        else:
            print("  No files to rename (already done).")
        if n_sanitized > 0:
            print(f"  Sanitized register names in {n_sanitized} files.")
        else:
            print("  No register names to sanitize (already done).")

    # Step 2: Build manifest
    print(f"\n[2/3] Building master manifest...")
    manifest_path = os.path.join(args.data_dir, "all_filenames.json")
    if args.dry_run:
        entries = build_file_list(args.dataset_dir)
        print(f"  Would write {len(entries)} entries to {manifest_path}")
    else:
        entries = write_master_manifest(args.dataset_dir, manifest_path)

    # Count categories
    from collections import Counter
    cat_counts = Counter(e["category"] for e in entries)
    print(f"  Categories: " + ", ".join(f"{k}={v}" for k, v in sorted(cat_counts.items())))

    # Step 3: Build and write folds
    print(f"\n[3/3] Building {args.n_folds}-fold stratified splits...")
    folds = build_fold_manifest(entries, n_folds=args.n_folds, seed=args.seed)

    if args.dry_run:
        print_fold_summary(folds, entries)
        print("  Dry run complete. No files were renamed or created.")
    else:
        write_fold_directories(folds, args.dataset_dir, args.data_dir)
        print_fold_summary(folds, entries)
        print(f"Fold directories written to '{args.data_dir}/'")
        print("Done. Run 'python main.py explain' next to generate explanations.jsonl")


if __name__ == "__main__":
    main()

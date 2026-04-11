# Bad Qubits: LLM-Based Detection and Explanation of Harmful Quantum Circuits

## Overview

This repository implements a pipeline for automatically classifying quantum circuits as **safe** or **bad** — circuits that exhibit negative operational effects on quantum hardware — using a fine-tuned large language model. The model produces structured output that simultaneously classifies a circuit and provides a natural-language explanation of the harmful pattern, making explainability a native property of the classifier rather than a post-hoc addition.

The project revises the framing of prior work (malicious-qubits) from adversarial intent ("malicious") to operational effect ("bad"), reflecting the observation that harmful circuit patterns may arise from misconfiguration, compiler errors, or hardware misuse as readily as from deliberate attack.

---

## Dataset

### Composition

The dataset consists of **1,500 OpenQASM 2.0 circuit files**:

| Class | Count | Description |
|---|---|---|
| Safe | 1,000 | Standard quantum algorithms: QFT, VQE, Deutsch-Jozsa, GHZ states, QAOA, etc. |
| Bad — immediate | ~167 | Circuits with premature or excessive measurement operations |
| Bad — shuttling | ~167 | Circuits with excessive SWAP gate chains |
| Bad — mixed | ~167 | Circuits combining both measurement and SWAP anomalies |

The binary label (`safe` / `bad`) is coarser than the 4-class category label. Classification metrics are reported at both levels.

### Harmful Pattern Definitions

**Immediate measurement (`bad/immediate`):** Measurement gates appear before the circuit has performed meaningful computation, or a disproportionate fraction of qubits are measured early. This causes premature wavefunction collapse and information leakage before the intended quantum computation completes.

**Qubit shuttling (`bad/shuttling`):** Long chains of SWAP gates move qubit state across the register without contributing to the algorithm. Each SWAP decomposes into three CNOT gates; excessive SWAPs amplify gate error rates and accelerate decoherence, degrading result fidelity.

**Mixed (`bad/mixed`):** Circuits exhibit both of the above patterns simultaneously, or other combinations of harmful behaviors.

### Preprocessing

Before training, all circuit files undergo two sanitization steps:

1. **File renaming:** `malicious_*` prefixes are replaced with `bad_*` throughout.
2. **Register name sanitization:** Register names that reveal circuit intent (`early`, `control`) are replaced with neutral names (`mout`, `anc`). This prevents the model from learning a trivial shortcut — classifying circuits based on register name strings rather than structural analysis.

The second step is critical for evaluation validity: without it, a model could achieve high accuracy by pattern-matching on register names rather than understanding circuit structure.

---

## Model

**Base model:** [Qwen 2.5 Coder 7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B) — a code-specialized language model chosen for its pretraining on structured text, including assembly-like languages similar to QASM syntax.

**Quantization:** 4-bit NF4 quantization via bitsandbytes, loaded through Unsloth's optimized inference path.

**Parameter-efficient fine-tuning:** LoRA (Low-Rank Adaptation) with the following configuration:

| Parameter | Value |
|---|---|
| Rank (`r`) | 32 |
| Alpha (`lora_alpha`) | 64 |
| Scaling (`use_rslora`) | True (RSLoRA: α / √r) |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Dropout | 0.0 |
| Bias | none |
| Gradient checkpointing | unsloth (memory-optimized) |

RSLoRA scaling (`α / √r` instead of `α / r`) stabilizes training at higher ranks by preventing gradient magnitudes from growing with rank.

---

## Output Format

The model is trained to produce a single JSON object as its complete output:

```json
{"safe": "true" | "false", "category": "safe" | "immediate" | "shuttling" | "mixed", "explanation": "<≤20 words>"}
```

The explanation field is constrained to 20 words maximum during both training data generation and reward optimization. This forces the model to identify and state the specific harmful pattern concisely rather than producing verbose, hedged descriptions. Explainability is therefore **intrinsic to the model output format**, not a secondary module.

At inference time, if the model fails to produce valid JSON, a fallback keyword-search parser is applied to recover a binary label.

---

## Context Window Handling

Qwen 2.5 Coder 7B has an 8,192-token native context window. The pipeline sets `MAX_SEQ_LENGTH = 5,000` tokens to leave headroom for the prompt structure and output tokens. Long circuits are handled by two complementary mechanisms:

**1. Circuit property summary (primary):** Before the QASM text, the prompt includes a structured summary extracted by lightweight regex parsing:

```
[Circuit: qubits=5, gates=142, swap=18, measure=6(4%), top_gates=[('cx', 44), ('h', 38), ...]]
```

This encodes the features most discriminative for classification — SWAP count, measurement count and fraction, total gate count, qubit count — in approximately 50 tokens, regardless of circuit length. The model sees the classification-relevant signals even when the full QASM is truncated.

**2. Hard character truncation (fallback):** Raw QASM text is truncated to `MAX_CIRCUIT_CHARS = 4,096` characters with a `[TRUNCATED]` marker. For the training and test datasets, most circuits fit within this limit; truncation is a safety net rather than a primary compression strategy.

This approach was chosen over chunking because chunking produces sub-circuits that lack the structural integrity needed for meaningful classification: a chunk boundary mid-circuit creates ambiguous examples with no valid label, producing noisy training signal.

---

## Training Pipeline

Training proceeds in four stages per fold:

### Stage 0: Explanation Generation (one-time, pre-training)

The untuned base Qwen model is used via ZhipuAI's GLM-5 API to generate `explanations.jsonl`. For each circuit, the API is called with the ground-truth category label provided as a hint (to ensure the explanation is grounded in the correct classification), producing:

- `thinking`: 3–5 sentences of chain-of-thought reasoning identifying the key structural pattern
- `target_output`: the final JSON string that will become the SFT training target

This file is generated once and reused across all folds. Because GLM-5 is given the ground-truth category, the explanations are anchored to correct labels rather than being model predictions that could propagate error into training targets.

### Stage 1: Supervised Fine-Tuning Warm-up (60 steps)

SFT trains the model on (prompt, target_output) pairs from `explanations.jsonl`. The training objective is standard next-token prediction on the `target_output` JSON string. The purpose is **format acquisition**: after SFT, the model reliably produces well-formed JSON with the correct fields before GRPO reward shaping begins.

Without SFT warm-up, GRPO's reward signal is mostly noise because the model's outputs are structurally invalid and none of the reward functions can score them meaningfully.

| Hyperparameter | Value |
|---|---|
| Steps | 60 |
| Learning rate | 2e-4 |
| LR schedule | cosine |
| Batch size | auto (GPU memory) |
| Gradient accumulation | auto |
| Optimizer | AdamW (fused) |
| Weight decay | 0.05 |
| Max grad norm | 1.0 |
| Sequence packing | enabled |

### Stage 2: GRPO Fine-Tuning (200 steps)

Group Relative Policy Optimization (GRPO; Shao et al. 2024) fine-tunes the model using reward-based optimization. For each prompt, the model generates `num_generations = 4` candidate completions. These are scored by a set of reward functions; each completion's advantage is computed relative to the mean reward within its group of 4, and the policy is updated to increase the probability of higher-advantage completions.

#### Reward Functions

| Function | Signal | Weight |
|---|---|---|
| `reward_classification` | Correct binary safe/bad label: +1.0. Wrong label on safe circuit: −0.5. Missed bad circuit: −0.8. Invalid JSON: −0.5. | 1.0 |
| `reward_format` | Valid JSON with all required fields (safe, category, explanation): +0.3. Otherwise: −0.1. | 0.3 |
| `reward_brevity` | Explanation ≤20 words: +0.2. ≤30 words: +0.1. Longer: 0.0. | 0.2 |
| `reward_category` | Correct sub-category for bad circuits: +0.2. Safe circuits: NaN (excluded from weighted sum). | 0.2 |

The asymmetric penalty in `reward_classification` (−0.8 for missed bad circuits vs. −0.5 for false alarms) compensates for the dataset's class imbalance (2:1 safe:bad ratio). The NaN handling in `reward_category` ensures that safe circuits are never penalized for the sub-category field — only bad circuits are evaluated on sub-category accuracy.

To further compensate for class imbalance, bad circuits are oversampled 2× in the GRPO training dataset. Final GRPO dataset composition: all safe circuits + (all bad circuits × 2).

| Hyperparameter | Value |
|---|---|
| Steps | 200 |
| Generations per prompt | 4 |
| Learning rate | 5e-5 |
| LR schedule | cosine |
| Temperature | 0.7 |
| KL penalty coefficient (β) | 0.01 |
| Batch size | 4 × num_gpus |
| Gradient accumulation | 1 |
| Optimizer | AdamW (fused) |
| Scale rewards | True |
| Bad circuit oversample ratio | 2× |

---

## Evaluation Methodology

### Dataset Splits

Each of the 5 folds uses a stratified 70/20/10 train/validation/test split:

1. **Test set (10%, 150 samples):** Stratified random sample held out before training begins. Not used for gradient updates or hyperparameter selection.
2. **Validation set (20%, 300 samples):** Used for post-training metric evaluation on the val split. Not used for gradient updates during training (`eval_strategy="no"`).
3. **Training set (70%, 1,050 samples):** Used for SFT and GRPO.

Stratification is performed on the 4-class category label to ensure all categories are proportionally represented in every split.

The 5 folds use seeds 42–46, producing distinct but not strictly disjoint test sets (approximately 10–15% overlap between any two folds' test sets). This is a consequence of the independent-random-split design; it is noted as a limitation.

### Metrics

**Validation set (primary):** Binary classification metrics (safe vs. bad) computed with weighted averaging: accuracy, F1, precision, recall, and confusion matrix. Per-category breakdowns (safe, immediate, shuttling, mixed) are computed by subsetting the validation set by true category.

**Test set (qualitative analysis):** Per-sample records capturing true label/category, predicted label/category, model-generated explanation, and correctness. These are aggregated into per-fold and per-category test accuracy, and representative correct/incorrect examples per category are selected for qualitative review in `cv_summary.json`.

**Cross-fold aggregation:** Mean ± standard deviation across all 5 folds, reported for accuracy, F1, precision, and recall on the validation set, and accuracy on the test set.

### Inference

At evaluation time, the model runs in greedy decoding mode (`do_sample=False`, `num_beams=1`) with `max_new_tokens=100`. The 2× headroom over the expected output length (~50 tokens for a JSON object with a 20-word explanation) is a practical guard against truncated outputs.

---

## Pipeline Execution

```
python main.py prepare              # Rename files, sanitize registers, build fold splits
python main.py explain --api-key X  # Generate explanations.jsonl via GLM-5 (one-time)
python main.py train                # Run all 5 folds (or --fold N for a single fold)
python main.py aggregate            # Aggregate fold results into cv_summary.json
python main.py classify circuit.qasm  # Classify a single circuit using trained model
```

The full pipeline can also be run end-to-end:
```
python main.py run-all --api-key X
```

---

## Output Files

After a complete run:

```
explanations.jsonl                        — per-circuit explanation spec (GLM-5 output)
data/all_filenames.json                   — master manifest (filename, label, category)
data/fold_{1-5}/train/                    — symlinks to training circuits
data/fold_{1-5}/val/                      — symlinks to validation circuits
data/fold_{1-5}/test/                     — symlinks to test circuits
models/fold_{1-5}/                        — LoRA adapter weights per fold
results/fold_{1-5}/confusion_matrix.png  — per-fold confusion matrix (val set)
results/fold_{1-5}/classification_report.txt
results/fold_{1-5}/per_class_metrics.json
results/fold_{1-5}/loss_curves.png
results/fold_{1-5}/test_qualitative.json  — per-sample test set predictions + explanations
results/aggregate/cv_summary.json         — cross-fold summary including qualitative analysis
results/aggregate/avg_confusion_matrix.png
```

---

## Dependencies

- Python 3.10+
- PyTorch (CUDA)
- [Unsloth](https://github.com/unslothai/unsloth) — optimized LoRA training and 4-bit inference
- `transformers`, `trl` (TRL ≥ 0.12 for GRPO support)
- `datasets`, `scikit-learn`, `matplotlib`, `seaborn`
- ZhipuAI API access (for the `explain` step only)

---

## Design Notes for Paper Inclusion

**Why an LLM over a classical classifier?** The structural features that distinguish bad circuits — measurement timing relative to the depth of meaningful computation, SWAP chain length and locality — are sequential and contextual properties. A bag-of-gates feature vector loses the ordering information that distinguishes, e.g., end-of-circuit measurements (safe) from mid-circuit measurements (bad/immediate). An LLM reading QASM as text preserves this sequential structure natively.

**Why GRPO over pure SFT?** SFT targets the explanation outputs from GLM-5, which are conditioned on the ground-truth label. The model trained by SFT alone learns to reproduce the form of correct explanations but does not receive a direct gradient signal on classification accuracy. GRPO directly optimizes the model's own classification decisions through the `reward_classification` signal, while the supporting rewards (`reward_format`, `reward_brevity`, `reward_category`) maintain output quality.

**Why are explanations trained rather than generated post-hoc?** Post-hoc explanation methods (e.g., LIME, SHAP) generate explanations by perturbing inputs and observing output changes. For a model classifying QASM text, such perturbations have no physical meaning — removing a random line of QASM may produce a syntactically invalid circuit. Explanations trained as part of the output format are grounded in the model's internal representation of the circuit and can be evaluated for correctness against the ground-truth category.

**Notation clarification:** Throughout this work, "bad" is used in place of "malicious" to reflect that the harmful patterns documented here can arise from hardware misconfiguration, compiler artifacts, or unintentional programming errors, and are not necessarily the product of adversarial intent. The classification task is therefore better framed as detecting operational harm rather than adversarial intent.

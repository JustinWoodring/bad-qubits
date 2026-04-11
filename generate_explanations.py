#!/usr/bin/env python3
"""
generate_explanations.py - Generate circuit explanations using GLM-5 via ZhipuAI API.

GLM-5 produces (thinking, JSON) pairs for each circuit. These become the training
targets for fine-tuning Qwen to understand and explain quantum circuit behavior.

Schema per entry in explanations.jsonl:
  {
    "filename":           "bad_qubit_shuttling_10_158.qasm",
    "label":              "bad",
    "category":           "shuttling",
    "thinking":           "<chain-of-thought from GLM-5>",
    "target_output":      '{"safe": "false", "category": "shuttling", "explanation": "..."}',
    "circuit_properties": { "num_qubits": ..., "num_gates": ..., ... }
  }

Run via: python main.py explain --api-key <key> [--limit N]
Or directly: python generate_explanations.py --api-key <key> [options]
"""

import os
import re
import json
import time
import argparse

import requests

from prepare_dataset import infer_category, infer_label


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZHIPUAI_BASE_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
MODEL_NAME = "glm-5"

SYSTEM_PROMPT = """You are an expert quantum computing engineer specializing in QASM circuit analysis.

You will be given a quantum circuit along with its ground-truth classification. Your job is to:
1. Analyze the circuit structure carefully to understand WHY it falls into that category.
2. Produce a clear, accurate explanation of the circuit's behavior.

Circuit categories:
- safe: A standard quantum algorithm (QFT, VQE, Deutsch-Jozsa, GHZ, etc.) with no anomalous patterns.
- immediate: Contains premature or excessive measurement operations (measure gates appearing before the circuit has performed meaningful computation, or measuring many qubits early). This causes information leakage and early wavefunction collapse.
- shuttling: Contains excessive SWAP gate chains that move qubit state across the register without contributing to the computation. Each SWAP adds gate errors and accelerates decoherence.
- mixed: Combines multiple bad patterns — both excessive measurements AND excessive SWAPs, or other compound harmful behaviors.

Output format — be brief. The fine-tuned model has a 5000-token context window shared between the circuit and your response, so every token counts.

Respond with strictly 3-5 sentences of reasoning inside <thinking> tags, then output ONLY a JSON object (no markdown, no code block, no extra text).

Rules for the explanation field:
- MAXIMUM 20 WORDS — count them before writing.
- For safe circuits: confirm the algorithm type and that measurements are end-of-circuit only.
- For bad circuits: name the specific harmful pattern and its hardware effect.

<thinking>
3-5 sentences ONLY. Identify the key pattern and why it matches the category.
</thinking>
{"safe": "true" or "false", "category": "safe" or "immediate" or "shuttling" or "mixed", "explanation": "One sentence, MAXIMUM 20 WORDS."}"""

CATEGORY_HINTS = {
    "safe":      "CLASSIFICATION: safe — This is a standard quantum algorithm with no anomalous patterns.",
    "immediate": "CLASSIFICATION: bad / immediate — This circuit contains premature or excessive measurements.",
    "shuttling": "CLASSIFICATION: bad / shuttling — This circuit contains excessive SWAP operations.",
    "mixed":     "CLASSIFICATION: bad / mixed — This circuit combines multiple bad patterns.",
}


# ---------------------------------------------------------------------------
# Circuit property extraction
# ---------------------------------------------------------------------------

def extract_circuit_properties(qasm_content: str) -> dict:
    """Extract circuit properties via lightweight regex parsing."""
    num_qubits = 0
    num_classical_bits = 0
    gate_counts = {}

    for m in re.finditer(r"qreg\s+\w+\[(\d+)\]", qasm_content):
        num_qubits += int(m.group(1))
    for m in re.finditer(r"creg\s+\w+\[(\d+)\]", qasm_content):
        num_classical_bits += int(m.group(1))

    has_measurements = bool(re.search(r"\bmeasure\b", qasm_content))

    for line in qasm_content.splitlines():
        line = line.strip()
        if not line or line.startswith(("//", "OPENQASM", "include", "qreg", "creg", "barrier", "gate ")):
            continue
        m = re.match(r"^([a-z_][a-z0-9_]*)\s*", line)
        if m:
            g = m.group(1)
            gate_counts[g] = gate_counts.get(g, 0) + 1

    return {
        "num_qubits":        num_qubits,
        "num_gates":         sum(gate_counts.values()),
        "gate_counts":       dict(sorted(gate_counts.items(), key=lambda x: -x[1])[:15]),
        "has_measurements":  has_measurements,
        "num_classical_bits": num_classical_bits,
    }


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_processed_filenames(output_jsonl: str) -> set:
    processed = set()
    if not os.path.exists(output_jsonl):
        return processed
    with open(output_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                processed.add(json.loads(line)["filename"])
            except (json.JSONDecodeError, KeyError):
                pass
    return processed


# ---------------------------------------------------------------------------
# GLM-5 API call
# ---------------------------------------------------------------------------

def call_glm(
    api_key: str,
    filename: str,
    category: str,
    circuit_content: str,
    properties: dict,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> tuple[str, str]:
    """
    Call GLM-5 to generate (thinking, target_output) for one circuit.
    Returns (thinking_text, json_string).
    """
    hint = CATEGORY_HINTS.get(category, "")
    max_circuit_chars = 3000
    truncated = circuit_content[:max_circuit_chars]
    if len(circuit_content) > max_circuit_chars:
        truncated += "\n... [TRUNCATED]"

    props_summary = (
        f"num_qubits={properties['num_qubits']}, "
        f"num_gates={properties['num_gates']}, "
        f"has_measurements={properties['has_measurements']}, "
        f"top_gates={list(properties['gate_counts'].items())[:5]}"
    )

    user_content = (
        f"File: {filename}\n"
        f"{hint}\n"
        f"Circuit properties: {props_summary}\n\n"
        f"Circuit QASM:\n{truncated}"
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "max_tokens": 400,
        "temperature": 0.1,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                ZHIPUAI_BASE_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            msg = data["choices"][0]["message"]
            content = msg.get("content", "")

            # GLM reasoning models may return reasoning_content separately
            reasoning = msg.get("reasoning_content", "")

            thinking, target_output = parse_response(content, reasoning)
            return thinking, target_output

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            last_error = f"HTTP {status}: {e}"
            if status == 429:
                # Rate limit — back off longer
                time.sleep(retry_delay * attempt * 2)
            else:
                time.sleep(retry_delay)
        except Exception as e:
            last_error = str(e)
            time.sleep(retry_delay)

    # All retries failed — return a minimal fallback
    label = "false" if category != "safe" else "true"
    fallback = json.dumps({
        "safe": "true" if category == "safe" else "false",
        "category": category,
        "explanation": f"[generation failed after {max_retries} attempts: {last_error}]",
    })
    return "", fallback


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _clean_explanation(text: str) -> str:
    """Remove leaked <thinking>/<reasoning>/<calculate> tag content from explanation."""
    cleaned = re.sub(
        r'<(thinking|reasoning|calculate)>.*?</(thinking|reasoning|calculate)>',
        '', text, flags=re.DOTALL,
    )
    # Remove unclosed opening tags (content truncated by max_tokens cutoff)
    cleaned = re.sub(r'<(thinking|reasoning|calculate)>.*', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


def parse_response(content: str, reasoning: str = "") -> tuple[str, str]:
    """
    Extract (thinking, target_json_string) from the model's response.

    Priority:
    1. Use reasoning_content (native reasoning field) if present.
    2. Otherwise extract <thinking>...</thinking> from content.
    3. JSON is whatever remains after the thinking block, or extracted via regex.
    """
    thinking = reasoning.strip() if reasoning else ""

    # Extract <thinking> block from content; always strip it from json_part
    json_part = content.strip()
    m = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
    if m:
        if not thinking:
            thinking = m.group(1).strip()
        json_part = content[m.end():].strip()

    # Strip markdown code fences if present
    json_part = re.sub(r"^```(?:json)?\s*", "", json_part, flags=re.MULTILINE)
    json_part = re.sub(r"```\s*$", "",       json_part, flags=re.MULTILINE)
    json_part = json_part.strip()

    # Extract JSON object
    m = re.search(r"\{.*\}", json_part, re.DOTALL)
    if m:
        json_part = m.group(0).strip()

    # Validate; fall back to wrapping explanation in minimal JSON
    try:
        obj = json.loads(json_part)
        # Clean and enforce ≤20-word limit on explanation
        explanation = _clean_explanation(str(obj.get("explanation", "")))
        words = explanation.split()
        if len(words) > 20:
            explanation = ' '.join(words[:20])
        # Normalise keys — model may vary capitalisation
        target = json.dumps({
            "safe":        str(obj.get("safe", "true")).lower(),
            "category":    str(obj.get("category", "safe")).lower(),
            "explanation": explanation,
        })
        return thinking, target
    except json.JSONDecodeError:
        # Treat the whole content as an explanation (clean and truncate)
        explanation = _clean_explanation(content.strip())
        words = explanation.split()
        if len(words) > 20:
            explanation = ' '.join(words[:20])
        target = json.dumps({
            "safe":        "true",
            "category":    "safe",
            "explanation": explanation,
        })
        return thinking, target


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_explanations(
    api_key: str,
    dataset_dir: str,
    manifest_path: str,
    output_jsonl: str,
    limit: int = None,
    resume: bool = True,
    request_delay: float = 0.5,
) -> None:
    with open(manifest_path, "r") as f:
        all_entries = json.load(f)

    processed = load_processed_filenames(output_jsonl) if resume else set()
    pending = [e for e in all_entries if e["filename"] not in processed]
    if limit is not None:
        pending = pending[:limit]

    total       = len(pending)
    already_done = len(processed)
    print(f"Circuits: {len(all_entries)} total, {already_done} done, {total} to generate")

    if total == 0:
        print("Nothing to do.")
        return

    ok_count   = 0
    fail_count = 0

    with open(output_jsonl, "a") as out_f:
        for idx, entry in enumerate(pending, start=1):
            filename = entry["filename"]
            category = entry["category"]
            label    = entry["label"]

            circuit_path = os.path.join(dataset_dir, filename)
            try:
                with open(circuit_path, "r") as f:
                    circuit_content = f.read()
            except FileNotFoundError:
                print(f"\n  [SKIP] Not found: {circuit_path}")
                continue

            properties = extract_circuit_properties(circuit_content)
            thinking, target_output = call_glm(
                api_key, filename, category, circuit_content, properties
            )

            failed = "[generation failed" in target_output
            if failed:
                fail_count += 1
            else:
                ok_count += 1

            record = {
                "filename":           filename,
                "label":              label,
                "category":           category,
                "thinking":           thinking,
                "target_output":      target_output,
                "circuit_properties": properties,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            # Rate limiting
            time.sleep(request_delay)

            status = "FAIL" if failed else "ok"
            print(
                f"\r  [{idx}/{total}] {idx/total*100:.1f}%  [{status}]  {filename[:50]:<50}",
                end="", flush=True,
            )

    print(f"\n\nDone. {ok_count} succeeded, {fail_count} failed.")
    if fail_count:
        print(f"  Re-run with --no-resume to regenerate failed entries, or they will be skipped in training.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate circuit explanations via GLM-5 API"
    )
    parser.add_argument("--api-key",     required=True,  help="ZhipuAI API key")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--manifest",    default="data/all_filenames.json")
    parser.add_argument("--output",      default="explanations.jsonl")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Process at most N circuits (for testing)")
    parser.add_argument("--no-resume",   action="store_true",
                        help="Reprocess all circuits even if output exists")
    parser.add_argument("--delay",       type=float, default=0.5,
                        help="Seconds between API requests (default: 0.5)")
    args = parser.parse_args()

    generate_explanations(
        api_key=args.api_key,
        dataset_dir=args.dataset_dir,
        manifest_path=args.manifest,
        output_jsonl=args.output,
        limit=args.limit,
        resume=not args.no_resume,
        request_delay=args.delay,
    )


if __name__ == "__main__":
    main()

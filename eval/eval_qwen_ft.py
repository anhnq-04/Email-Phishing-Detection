"""
eval_qwen_ft.py
===============
Evaluate the **fine-tuned** Qwen3-14B model (Phase 2 LoRA adapter) on the Phase 2 test set.

The model must be served via vLLM with LoRA beforehand:

    python -m vllm.entrypoints.openai.api_server \
      --model unsloth/qwen3-14b-unsloth-bnb-4bit \
      --enable-lora \
      --lora-modules phishing=models/qwen/model_final \
      --quantization bitsandbytes \
      --load-format bitsandbytes \
      --max-model-len 4096 \
      --port 8001 \
      --gpu-memory-utilization 0.9

Metrics computed:
  1. Label — accuracy, per-class precision / recall / F1 (phishing & legit)
  2. JSON parse success rate
  3. Evidence span verification — exact_span ↔ char_start / char_end correctness
  4. Signal set — micro precision / recall / F1

Usage:
    python -m eval.eval_qwen_ft                             # defaults
    python -m eval.eval_qwen_ft --max-samples 200           # quick smoke test
    python -m eval.eval_qwen_ft --base-url http://host:8001/v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI
from tqdm import tqdm

# ── Fix Windows console UTF-8 ──
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Project path ──
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))


# ═══════════════════════════════════════════════
# TEE OUTPUT — duplicate print to console + log file
# ═══════════════════════════════════════════════

class TeeOutput:
    """Write to both the original stdout and a log file simultaneously."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._original_stdout = sys.stdout
        self._log_file = open(log_path, "w", encoding="utf-8")

    def write(self, text: str):
        self._original_stdout.write(text)
        self._log_file.write(text)
        self._log_file.flush()

    def flush(self):
        self._original_stdout.flush()
        self._log_file.flush()

    def close(self):
        sys.stdout = self._original_stdout
        self._log_file.close()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *args):
        self.close()


# ═══════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════

# FT model served WITH --enable-lora --lora-modules phishing=...
# → model name in the API = "phishing"
DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_MODEL_NAME = "phishing"
DEFAULT_API_KEY = "dummy"

DEFAULT_DATA_PATH = str(
    _PROJECT_ROOT / "data" / "processed" / "qwen" / "phase2_eval.jsonl"
)

VALID_SIGNALS = {
    "urgency",
    "impersonation",
    "credential_request",
    "financial_lure",
    "threat",
    "call_to_action",
    "suspicious_attachment",
    "otp_or_code",
    "social_engineering",
    "extortion",
}

# Inference parameters — match training/inference config
MAX_TOKENS = 4096
# TEMPERATURE = 0.6
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = 1
MIN_P = 1.0
SEED = 3407
ENABLE_THINKING = False
TIMEOUT = 120


# ═══════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════

@dataclass
class TestSample:
    """A single test sample parsed from the JSONL."""
    system_prompt: str
    user_prompt: str
    gold_label: str                         # "phishing" | "legit"
    gold_signals: Set[str] = field(default_factory=set)
    gold_evidence: List[Dict[str, Any]] = field(default_factory=list)
    index: int = 0


def load_test_data(path: str, max_samples: int = 0) -> List[TestSample]:
    """Load phase2 eval JSONL → list of TestSample."""
    samples: List[TestSample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row["messages"]

            system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            asst_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")

            try:
                gold = json.loads(asst_msg)
            except json.JSONDecodeError:
                continue

            gold_label = gold.get("label", "unknown")
            gold_signals = set(gold.get("signals", []))
            gold_evidence = gold.get("evidence", [])

            samples.append(TestSample(
                system_prompt=system_msg,
                user_prompt=user_msg,
                gold_label=gold_label,
                gold_signals=gold_signals,
                gold_evidence=gold_evidence,
                index=idx,
            ))

            if 0 < max_samples <= len(samples):
                break

    return samples


# ═══════════════════════════════════════════════
# vLLM CLIENT
# ═══════════════════════════════════════════════

def create_client(base_url: str = DEFAULT_BASE_URL, api_key: str = DEFAULT_API_KEY) -> OpenAI:
    """Create an OpenAI-compatible client pointing to the vLLM server."""
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=TIMEOUT)
    # Quick health check
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        print(f"  vLLM server connected. Available models: {available}")
    except Exception as e:
        print(f"  WARNING: Could not reach vLLM at {base_url}: {e}")
    return client


def generate_response(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call the vLLM OpenAI-compatible chat endpoint."""
    extra_body: Dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
    }
    if TEMPERATURE > 0:
        extra_body["top_k"] = TOP_K
        extra_body["min_p"] = MIN_P

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        seed=SEED,
        extra_body=extra_body,
    )
    return response.choices[0].message.content or ""


# ═══════════════════════════════════════════════
# JSON PARSING
# ═══════════════════════════════════════════════

def remove_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def extract_first_balanced_json(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object start found")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("No balanced JSON object found")


def parse_model_output(raw_text: str) -> Optional[Dict[str, Any]]:
    """Try to parse the model output into a dict. Returns None on failure."""
    try:
        text = remove_think_blocks(raw_text)
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = json.loads(extract_first_balanced_json(text))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


# ═══════════════════════════════════════════════
# EVIDENCE SPAN VERIFICATION
# ═══════════════════════════════════════════════

def whitespace_flexible_pattern(span: str) -> str:
    pieces = re.split(r"\s+", span.strip())
    return r"\s+".join(re.escape(piece) for piece in pieces if piece)


def find_span_in_text(span: str, text: str) -> Optional[Tuple[int, int]]:
    """Find exact_span in the user prompt text. Returns (start, end) or None."""
    if not span or not text:
        return None

    pos = text.lower().find(span.lower())
    if pos >= 0:
        return pos, pos + len(span)

    try:
        pattern = whitespace_flexible_pattern(span)
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        return (match.start(), match.end()) if match else None
    except re.error:
        return None


def verify_evidence_spans(
    predicted_evidence: List[Dict[str, Any]],
    user_prompt: str,
) -> Dict[str, Any]:
    """
    Verify that each evidence item's exact_span can be found in the user prompt
    and that the char positions are correct.

    Returns a dict with:
        total:           total evidence items
        found:           spans found in user prompt
        not_found:       spans NOT found in user prompt
        char_correct:    spans where predicted char_start/char_end match found position
        char_incorrect:  spans where positions don't match (but span was found)
        details:         per-item details
    """
    total = len(predicted_evidence)
    found = 0
    not_found = 0
    char_correct = 0
    char_incorrect = 0
    details: List[Dict[str, Any]] = []

    for ev in predicted_evidence:
        exact_span = ev.get("exact_span", "")
        if not exact_span:
            not_found += 1
            details.append({"span": exact_span, "found": False})
            continue

        location = find_span_in_text(exact_span, user_prompt)
        if location is None:
            not_found += 1
            details.append({"span": exact_span[:80], "found": False})
        else:
            found += 1
            actual_start, actual_end = location
            pred_start = ev.get("char_start", -1)
            pred_end = ev.get("char_end", -1)
            if pred_start >= 0 and pred_end >= 0:
                if pred_start == actual_start and pred_end == actual_end:
                    char_correct += 1
                else:
                    char_incorrect += 1
            details.append({
                "span": exact_span[:80],
                "found": True,
                "actual_start": actual_start,
                "actual_end": actual_end,
                "pred_start": pred_start,
                "pred_end": pred_end,
            })

    return {
        "total": total,
        "found": found,
        "not_found": not_found,
        "char_correct": char_correct,
        "char_incorrect": char_incorrect,
        "details": details,
    }


# ═══════════════════════════════════════════════
# METRIC COMPUTATION
# ═══════════════════════════════════════════════

@dataclass
class ClassMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def compute_label_metrics(
    gold_labels: List[str],
    pred_labels: List[str],
) -> Dict[str, Any]:
    """Compute accuracy, per-class P/R/F1 for phishing and legit."""
    assert len(gold_labels) == len(pred_labels)
    n = len(gold_labels)
    correct = sum(g == p for g, p in zip(gold_labels, pred_labels))
    accuracy = correct / n if n > 0 else 0.0

    phishing = ClassMetrics()
    legit = ClassMetrics()

    for gold, pred in zip(gold_labels, pred_labels):
        # Phishing metrics
        if gold == "phishing" and pred == "phishing":
            phishing.tp += 1
        elif gold != "phishing" and pred == "phishing":
            phishing.fp += 1
        elif gold == "phishing" and pred != "phishing":
            phishing.fn += 1

        # Legit metrics
        if gold == "legit" and pred == "legit":
            legit.tp += 1
        elif gold != "legit" and pred == "legit":
            legit.fp += 1
        elif gold == "legit" and pred != "legit":
            legit.fn += 1

    # Phishing False Negative Rate = FN / (TP + FN) = 1 - Recall
    gold_phishing_total = phishing.tp + phishing.fn
    phishing_fnr = phishing.fn / gold_phishing_total if gold_phishing_total > 0 else 0.0

    return {
        "total_samples": n,
        "accuracy": accuracy,
        "phishing_precision": phishing.precision,
        "phishing_recall": phishing.recall,
        "phishing_f1": phishing.f1,
        "phishing_fnr": phishing_fnr,
        "phishing_fn": phishing.fn,
        "phishing_total_gold": gold_phishing_total,
        "legit_precision": legit.precision,
        "legit_recall": legit.recall,
        "legit_f1": legit.f1,
    }


def compute_signal_micro_metrics(
    gold_signal_sets: List[Set[str]],
    pred_signal_sets: List[Set[str]],
) -> Dict[str, Any]:
    """
    Compute micro precision / recall / F1 over signal sets.
    Each sample contributes its TP/FP/FN counts.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_signal_tp: Counter = Counter()
    per_signal_fp: Counter = Counter()
    per_signal_fn: Counter = Counter()

    for gold_set, pred_set in zip(gold_signal_sets, pred_signal_sets):
        # Filter to valid signals only
        gold_valid = gold_set & VALID_SIGNALS
        pred_valid = pred_set & VALID_SIGNALS

        tp = gold_valid & pred_valid
        fp = pred_valid - gold_valid
        fn = gold_valid - pred_valid

        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

        for s in tp:
            per_signal_tp[s] += 1
        for s in fp:
            per_signal_fp[s] += 1
        for s in fn:
            per_signal_fn[s] += 1

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    # Per-signal breakdown
    per_signal = {}
    all_signals = set(per_signal_tp) | set(per_signal_fp) | set(per_signal_fn)
    for s in sorted(all_signals):
        tp_s = per_signal_tp[s]
        fp_s = per_signal_fp[s]
        fn_s = per_signal_fn[s]
        p_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
        r_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
        f1_s = 2 * p_s * r_s / (p_s + r_s) if (p_s + r_s) > 0 else 0.0
        per_signal[s] = {
            "tp": tp_s, "fp": fp_s, "fn": fn_s,
            "precision": p_s, "recall": r_s, "f1": f1_s,
        }

    return {
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_signal": per_signal,
    }


# ═══════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════

def normalize_label(label: Any) -> str:
    value = str(label).strip().lower()
    if value in {"phishing", "phish", "malicious", "spam", "scam"}:
        return "phishing"
    if value in {"legit", "legitimate", "safe", "ham", "benign"}:
        return "legit"
    return "unknown"


def normalize_signal(signal: Any) -> str:
    value = str(signal).strip().lower()
    return value if value in VALID_SIGNALS else ""


def evaluate(
    client: OpenAI,
    model_name: str,
    tag: str = "FINE-TUNED",
    data_path: str = DEFAULT_DATA_PATH,
    max_samples: int = 0,
    output_dir: str = "",
):
    """Run full evaluation and print/save results."""

    # ── Set up tee logging ──
    log_dir = str(_PROJECT_ROOT / "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"eval_qwen_ft_{timestamp}.log")
    tee = TeeOutput(log_path)
    tee.__enter__()

    try:
        return _evaluate_inner(client, model_name, tag, data_path, max_samples, output_dir, log_path)
    finally:
        tee.__exit__(None, None, None)
        print(f"  Log saved → {log_path}")


def _evaluate_inner(
    client: OpenAI,
    model_name: str,
    tag: str,
    data_path: str,
    max_samples: int,
    output_dir: str,
    log_path: str,
):
    """Inner evaluation logic (runs under TeeOutput)."""
    print(f"  Log file: {log_path}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)
    print(f"  QWEN EVALUATION — {tag}")
    print(f"  Model: {model_name}")
    print(f"  Server: {client.base_url}")
    print(f"  Data:  {data_path}")
    print("=" * 80)

    # Resolve output directory and details file path early for checkpointing
    if not output_dir:
        output_dir = str(_PROJECT_ROOT / "eval")
    os.makedirs(output_dir, exist_ok=True)
    details_path = os.path.join(output_dir, "results_qwen_ft_details.jsonl")

    # Try loading existing checkpoint/details progress
    completed_results: Dict[int, Dict[str, Any]] = {}
    if os.path.exists(details_path):
        try:
            with open(details_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        idx = item.get("index")
                        if idx is not None:
                            completed_results[idx] = item
            print(f"Found existing progress at {details_path}. Loaded {len(completed_results)} completed samples.")
            print("  (To start fresh, delete or rename this details.jsonl file.)")
        except Exception as e:
            print(f"Could not load existing progress from {details_path}: {e}. Starting fresh.")

    # ── Load data ──
    samples = load_test_data(data_path, max_samples=max_samples)
    print(f"Loaded {len(samples)} test samples.")

    gold_labels: List[str] = []
    pred_labels: List[str] = []
    gold_signal_sets: List[Set[str]] = []
    pred_signal_sets: List[Set[str]] = []

    json_parse_success = 0
    json_parse_fail = 0

    evidence_stats = {
        "total_evidence_items": 0,
        "found_in_prompt": 0,
        "not_found_in_prompt": 0,
        "char_correct": 0,
        "char_incorrect": 0,
    }

    # Track gold evidence span verification too (sanity check)
    gold_evidence_stats = {
        "total_gold_evidence": 0,
        "gold_found_in_prompt": 0,
        "gold_not_found": 0,
    }

    raw_results: List[Dict[str, Any]] = []
    new_evaluations_count = 0

    start_time = time.time()

    for sample in tqdm(samples, desc="Evaluating"):
        # Check if already evaluated in previous run
        if sample.index in completed_results:
            item = completed_results[sample.index]
            pred_label = item["pred_label"]
            pred_signals = set(item.get("pred_signals", []))
            json_parsed = item.get("json_parsed", False)
            raw_response = item.get("raw_response", "")
            
            if json_parsed:
                json_parse_success += 1
            else:
                json_parse_fail += 1
                
            ev_verify = item.get("evidence_verification", {})
            evidence_stats["total_evidence_items"] += ev_verify.get("total", 0)
            evidence_stats["found_in_prompt"] += ev_verify.get("found", 0)
            evidence_stats["not_found_in_prompt"] += ev_verify.get("not_found", 0)
            evidence_stats["char_correct"] += ev_verify.get("char_correct", 0)
            evidence_stats["char_incorrect"] += ev_verify.get("char_incorrect", 0)
            
            # Reconstruct ev_result structure
            ev_result = {
                "total": ev_verify.get("total", 0),
                "found": ev_verify.get("found", 0),
                "not_found": ev_verify.get("not_found", 0),
                "char_correct": ev_verify.get("char_correct", 0),
                "char_incorrect": ev_verify.get("char_incorrect", 0),
            }
        else:
            # Generate response via vLLM API
            try:
                raw_response = generate_response(
                    client, model_name,
                    system_prompt=sample.system_prompt,
                    user_prompt=sample.user_prompt,
                )
            except Exception as e:
                tqdm.write(f"ERROR generating for sample {sample.index}: {e}")
                raw_response = ""

            # Parse JSON
            parsed = parse_model_output(raw_response)
            json_parsed = (parsed is not None)

            if json_parsed:
                json_parse_success += 1
                pred_label = normalize_label(parsed.get("label", "unknown"))
                pred_signals_raw = parsed.get("signals", [])
                pred_signals = {normalize_signal(s) for s in pred_signals_raw} - {""}
                pred_evidence = parsed.get("evidence", [])
                if not isinstance(pred_evidence, list):
                    pred_evidence = []
            else:
                json_parse_fail += 1
                pred_label = "unknown"
                pred_signals = set()
                pred_evidence = []

            # ── Predicted evidence span verification ──
            ev_result = verify_evidence_spans(pred_evidence, sample.user_prompt)
            evidence_stats["total_evidence_items"] += ev_result["total"]
            evidence_stats["found_in_prompt"] += ev_result["found"]
            evidence_stats["not_found_in_prompt"] += ev_result["not_found"]
            evidence_stats["char_correct"] += ev_result["char_correct"]
            evidence_stats["char_incorrect"] += ev_result["char_incorrect"]

        # Append variables for metric computation
        gold_labels.append(sample.gold_label)
        pred_labels.append(pred_label)
        gold_signal_sets.append(sample.gold_signals)
        pred_signal_sets.append(pred_signals)

        # ── Gold evidence verification (sanity check) ──
        for gev in sample.gold_evidence:
            gold_span = gev.get("exact_span", "")
            gold_evidence_stats["total_gold_evidence"] += 1
            if gold_span and find_span_in_text(gold_span, sample.user_prompt) is not None:
                gold_evidence_stats["gold_found_in_prompt"] += 1
            else:
                gold_evidence_stats["gold_not_found"] += 1

        raw_results.append({
            "index": sample.index,
            "gold_label": sample.gold_label,
            "pred_label": pred_label,
            "gold_signals": sorted(sample.gold_signals),
            "pred_signals": sorted(pred_signals),
            "json_parsed": json_parsed,
            "evidence_verification": {
                k: v for k, v in ev_result.items() if k != "details"
            },
            "raw_response": raw_response[:2000],
        })

        if sample.index not in completed_results:
            new_evaluations_count += 1
            if new_evaluations_count % 50 == 0:
                with open(details_path, "w", encoding="utf-8") as f:
                    for item in raw_results:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                tqdm.write(f"Saved checkpoint of {len(raw_results)} samples to {details_path}")

    elapsed = time.time() - start_time
    print(f"\n  Finished in {elapsed:.1f}s ({elapsed / len(samples):.2f}s/sample)\n")

    # ═══════════════════════════════════════════
    # COMPUTE ALL METRICS
    # ═══════════════════════════════════════════

    total_samples = len(samples)

    # 1. JSON Parse Rate
    parse_rate = json_parse_success / total_samples if total_samples > 0 else 0.0

    # 2. Label Metrics
    label_metrics = compute_label_metrics(gold_labels, pred_labels)

    # 3. Signal Micro Metrics
    signal_metrics = compute_signal_micro_metrics(gold_signal_sets, pred_signal_sets)

    # 4. Evidence Stats
    ev_total = evidence_stats["total_evidence_items"]
    ev_found_rate = (
        evidence_stats["found_in_prompt"] / ev_total if ev_total > 0 else 0.0
    )

    # 5. Hallucinated Evidence Rate = not_found / total predicted evidence
    hallucinated_evidence_rate = (
        evidence_stats["not_found_in_prompt"] / ev_total if ev_total > 0 else 0.0
    )

    # 6. Phishing False Negative Rate (from label_metrics)
    phishing_fnr = label_metrics["phishing_fnr"]

    gold_ev_total = gold_evidence_stats["total_gold_evidence"]
    gold_ev_found_rate = (
        gold_evidence_stats["gold_found_in_prompt"] / gold_ev_total
        if gold_ev_total > 0 else 0.0
    )

    # ═══════════════════════════════════════════
    # PRINT REPORT
    # ═══════════════════════════════════════════

    print("=" * 80)
    print(f"  RESULTS — {tag} ({model_name})")
    print("=" * 80)

    print(f"\n{'─' * 40}")
    print("  1. JSON PARSE RATE")
    print(f"{'─' * 40}")
    print(f"  Success: {json_parse_success} / {total_samples} ({parse_rate:.2%})")
    print(f"  Fail:    {json_parse_fail} / {total_samples} ({1 - parse_rate:.2%})")

    print(f"\n{'─' * 40}")
    print("  2. LABEL CLASSIFICATION")
    print(f"{'─' * 40}")
    print(f"  Accuracy:            {label_metrics['accuracy']:.4f}")
    print(f"  Phishing Precision:  {label_metrics['phishing_precision']:.4f}")
    print(f"  Phishing Recall:     {label_metrics['phishing_recall']:.4f}")
    print(f"  Phishing F1:         {label_metrics['phishing_f1']:.4f}")
    print(f"  Phishing FNR:        {phishing_fnr:.4f}  ({label_metrics['phishing_fn']}/{label_metrics['phishing_total_gold']} gold phishing missed)")
    print(f"  Legit Precision:     {label_metrics['legit_precision']:.4f}")
    print(f"  Legit Recall:        {label_metrics['legit_recall']:.4f}")
    print(f"  Legit F1:            {label_metrics['legit_f1']:.4f}")

    # Confusion matrix
    cm = defaultdict(int)
    for g, p in zip(gold_labels, pred_labels):
        cm[(g, p)] += 1
    print(f"\n  Confusion Matrix:")
    all_labels_seen = sorted(set(gold_labels) | set(pred_labels))
    header = "  {:>12s}".format("gold\\pred") + "".join(f"  {l:>10s}" for l in all_labels_seen)
    print(header)
    for g in all_labels_seen:
        row = f"  {g:>12s}" + "".join(f"  {cm[(g, p)]:>10d}" for p in all_labels_seen)
        print(row)

    print(f"\n{'─' * 40}")
    print("  3. EVIDENCE SPAN VERIFICATION")
    print(f"{'─' * 40}")
    print(f"  [Predicted evidence]")
    print(f"  Total evidence items:     {evidence_stats['total_evidence_items']}")
    print(f"  Found in user prompt:     {evidence_stats['found_in_prompt']} ({ev_found_rate:.2%})")
    print(f"  NOT found in prompt:      {evidence_stats['not_found_in_prompt']}")
    print(f"  Hallucinated evidence rate: {hallucinated_evidence_rate:.4f}  ({evidence_stats['not_found_in_prompt']}/{ev_total})")
    print(f"  char_start/end correct:   {evidence_stats['char_correct']}")
    print(f"  char_start/end incorrect: {evidence_stats['char_incorrect']}")
    print()
    print(f"  [Gold evidence sanity check]")
    print(f"  Total gold evidence:      {gold_evidence_stats['total_gold_evidence']}")
    print(f"  Found in user prompt:     {gold_evidence_stats['gold_found_in_prompt']} ({gold_ev_found_rate:.2%})")
    print(f"  NOT found in prompt:      {gold_evidence_stats['gold_not_found']}")

    print(f"\n{'─' * 40}")
    print("  4. SIGNAL SET — MICRO METRICS")
    print(f"{'─' * 40}")
    print(f"  Micro Precision: {signal_metrics['micro_precision']:.4f}")
    print(f"  Micro Recall:    {signal_metrics['micro_recall']:.4f}")
    print(f"  Micro F1:        {signal_metrics['micro_f1']:.4f}")
    print(f"  (TP={signal_metrics['total_tp']}, FP={signal_metrics['total_fp']}, FN={signal_metrics['total_fn']})")

    print(f"\n  Per-signal breakdown:")
    print(f"  {'Signal':<25s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s}")
    print(f"  {'─' * 62}")
    for sig, m in sorted(signal_metrics["per_signal"].items()):
        print(
            f"  {sig:<25s} {m['tp']:>5d} {m['fp']:>5d} {m['fn']:>5d} "
            f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}"
        )

    print("\n" + "=" * 80)

    # ═══════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════

    if not output_dir:
        output_dir = str(_PROJECT_ROOT / "eval")

    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "model": model_name,
        "tag": tag,
        "base_url": str(client.base_url),
        "data_path": data_path,
        "total_samples": total_samples,
        "elapsed_seconds": round(elapsed, 2),
        "json_parse_rate": round(parse_rate, 4),
        "label_metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in label_metrics.items()},
        "evidence_stats": {
            **evidence_stats,
            "span_found_rate": round(ev_found_rate, 4),
            "hallucinated_evidence_rate": round(hallucinated_evidence_rate, 4),
        },
        "gold_evidence_stats": {
            **gold_evidence_stats,
            "gold_span_found_rate": round(gold_ev_found_rate, 4),
        },
        "signal_micro_metrics": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in signal_metrics.items() if k != "per_signal"
        },
        "signal_per_signal": {
            sig: {k2: round(v2, 4) if isinstance(v2, float) else v2 for k2, v2 in m.items()}
            for sig, m in signal_metrics["per_signal"].items()
        },
    }

    summary_path = os.path.join(output_dir, "results_qwen_ft.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Summary saved → {summary_path}")

    details_path = os.path.join(output_dir, "results_qwen_ft_details.jsonl")
    with open(details_path, "w", encoding="utf-8") as f:
        for item in raw_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Details saved → {details_path}")

    return summary


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-14B FINE-TUNED model (LoRA) via vLLM server"
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL,
        help="vLLM OpenAI-compatible base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_NAME,
        help="Model name as registered in vLLM (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key", default=DEFAULT_API_KEY,
        help="API key for vLLM server (default: %(default)s)",
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to test JSONL")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples (0 = all)")
    parser.add_argument("--output-dir", default="", help="Output directory for results")
    args = parser.parse_args()

    client = create_client(base_url=args.base_url, api_key=args.api_key)
    evaluate(
        client,
        model_name=args.model,
        tag="FINE-TUNED",
        data_path=args.data_path,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

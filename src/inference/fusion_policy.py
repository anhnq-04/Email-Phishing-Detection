"""
fusion_policy.py
================
Layer 4 — Weighted fusion of multi-layer detection results.

Combines outputs from:
  - Layer 1: FastScan (rule-based)
  - Layer 2: CatBoost (URL ML)
  - Layer 3: Qwen LLM (semantic analysis via vLLM)

Produces a single FusionResult with aggregated risk score,
label, action recommendation, evidence, and per-layer metadata.
"""

from __future__ import annotations

from markdown_it.helpers import parse_link_label
from src.xai import evidence_builder

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("phishing_api.fusion")


# ════════════════════════════════════════════════════════
# FUSION WEIGHTS (tunable per message type)
# ════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "email": {"fastscan": 0.05, "catboost": 0.20, "qwen": 0.75},
    "sms":   {"fastscan": 0.05, "catboost": 0.20, "qwen": 0.75},
}


# ════════════════════════════════════════════════════════
# FUSION RESULT
# ════════════════════════════════════════════════════════

@dataclass
class FusionResult:
    """Aggregated result from all detection layers."""

    label: str
    risk_score: float
    risk_level: str
    action: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    layer_scores: Dict[str, float] = field(default_factory=dict)
    layer_labels: Dict[str, str] = field(default_factory=dict)
    disagreement: float = 0.0
    signals: List[str] = field(default_factory=list)
    reason: str = ""

    qwen_evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "action": self.action,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "weights": self.weights,
            "layer_scores": self.layer_scores,
            "layer_labels": self.layer_labels,
            "disagreement": self.disagreement,
            "signals": self.signals,
            "reason": self.reason,
            "qwen_evidence": self.qwen_evidence,
        }


# ════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════

def _risk_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    if score >= 0.0:
        return "low"
    return "uncertain"


def _action(label: str, risk_level: str) -> str:
    if label == "phishing" and risk_level == "high":
        return "block"
    if label in ("phishing", "suspicious"):
        return "warn"
    if label == "legit":
        return "allow"
    return "review"


def _fastscan_to_score(fastscan_result) -> Optional[float]:
    """Convert FastScan verdict → numeric score."""
    if fastscan_result is None:
        return None
    verdict = fastscan_result.verdict.upper()
    if verdict == "PHISHING":
        return 0.90   # contribute as a signal, not a hard override
    if verdict == "SAFE":
        return 0.05
    return None  # UNKNOWN — don't contribute to fusion


def _catboost_aggregate(url_results) -> Optional[float]:
    """Aggregate CatBoost batch results → single phishing score."""
    if url_results is None or len(url_results) == 0:
        return None
    scores = [r.phishing_score for r in url_results]
    # Use max score — per-URL threshold is already strict (0.9)
    return max(scores)


def _qwen_to_score(qwen_result) -> Optional[float]:
    """Extract Qwen risk score. Ignore failed/unknown Qwen outputs."""
    if qwen_result is None:
        return None

    if getattr(qwen_result, "error", None):
        logger.warning("Ignoring Qwen score because Qwen returned error: %s", qwen_result.error)
        return None

    label = str(getattr(qwen_result, "label", "unknown")).lower()
    if label == "unknown":
        logger.warning("Ignoring Qwen score because Qwen label is unknown.")
        return None

    return qwen_result.risk_score


# ════════════════════════════════════════════════════════
# MAIN FUSION FUNCTION
# ════════════════════════════════════════════════════════

def fuse_results(
    fastscan_result=None,
    url_results=None,
    qwen_result=None,
    message_type: str = "email",
) -> FusionResult:
    """
    Fuse multi-layer results into a single FusionResult.

    Parameters
    ----------
    fastscan_result : FastScanResult | None
        Output from Layer 1 (rule-based).
    url_results : list[CatBoostResult] | None
        Output from Layer 2 (URL ML).
    qwen_result : QwenResult | None
        Output from Layer 3 (LLM via vLLM).
    message_type : str
        "email" or "sms" — determines fusion weights.

    Returns
    -------
    FusionResult
    """
    weights_cfg = DEFAULT_WEIGHTS.get(message_type, DEFAULT_WEIGHTS["email"])

    # ── Convert each layer to a numeric score ───────────
    fastscan_score = _fastscan_to_score(fastscan_result)
    catboost_score = _catboost_aggregate(url_results)
    qwen_score = _qwen_to_score(qwen_result)

    layer_scores: Dict[str, float] = {}
    layer_labels: Dict[str, str] = {}
    active_weights: Dict[str, float] = {}
    evidence: List[str] = []
    signals: List[str] = []
    qwen_evidence: List[Dict[str, Any]] = []

    # ── FastScan ────────────────────────────────────────
    if fastscan_result is not None:
        layer_labels["fastscan"] = fastscan_result.verdict

        if fastscan_result.reason:
            evidence.append(f"[FastScan] {fastscan_result.reason}")

    if fastscan_score is not None:
        layer_scores["fastscan"] = round(fastscan_score, 4)
        active_weights["fastscan"] = weights_cfg["fastscan"]

    # ── CatBoost ────────────────────────────────────────
    if catboost_score is not None:
        layer_scores["catboost"] = round(catboost_score, 4)
        active_weights["catboost"] = weights_cfg["catboost"]

        if url_results:
            phishing_urls = [
                r for r in url_results
                if r.label == "Phishing"
            ]
            if phishing_urls:
                urls_str = ", ".join(r.url for r in phishing_urls[:3])
                evidence.append(f"[CatBoost] Phishing URL(s): {urls_str}")
                layer_labels["catboost"] = "phishing"
            else:
                layer_labels["catboost"] = "legitimate"

            # Gather signals from CatBoost results (like domain_spoofing)
            for r in url_results:
                if hasattr(r, "signals") and r.signals:
                    for sig in r.signals:
                        if sig not in signals:
                            signals.append(sig)

    # ── Qwen LLM ───────────────────────────────────────
    if qwen_score is not None:
        layer_scores["qwen"] = round(qwen_score, 4)
        active_weights["qwen"] = weights_cfg["qwen"]
        layer_labels["qwen"] = qwen_result.label

        if qwen_result.reason:
            evidence.append(f"[Qwen] {qwen_result.reason}")

        if qwen_result.signals:
            for sig in qwen_result.signals:
                if sig not in signals:
                    signals.append(sig)

        if hasattr(qwen_result, "evidence_dicts"):
            qwen_evidence = qwen_result.evidence_dicts()
        elif hasattr(qwen_result, "evidence"):
            qwen_evidence = [
                ev.to_dict() if hasattr(ev, "to_dict") else dict(ev)
                for ev in qwen_result.evidence
            ]

    # ── Weighted aggregation ────────────────────────────
    if not active_weights:
        return FusionResult(
            label="unknown",
            risk_score=0.50,
            risk_level="uncertain",
            action="review",
            confidence=0.0,
            evidence=evidence,
            weights=weights_cfg,
            layer_scores=layer_scores,
            layer_labels=layer_labels,
            disagreement=0.0,
            signals=signals,
            reason="No layers returned usable results.",
            qwen_evidence=qwen_evidence,
        )

    # Normalize weights to sum=1.0 for active layers only
    total_w = sum(active_weights.values())
    norm_weights = {k: v / total_w for k, v in active_weights.items()}

    risk_score = sum(
        norm_weights[layer] * layer_scores[layer]
        for layer in norm_weights
    )
    risk_score = round(min(max(risk_score, 0.0), 1.0), 4)

    # ── Disagreement ────────────────────────────────────
    scores = list(layer_scores.values())
    if len(scores) >= 2:
        disagreement = round(max(scores) - min(scores), 4)
    else:
        disagreement = 0.0

    # ── Confidence ──────────────────────────────────────
    n_layers = len(active_weights)
    base_confidence = 0.5 + 0.15 * n_layers  # more layers → more confident
    confidence = round(min(base_confidence * (1 - 0.5 * disagreement), 1.0), 4)

    # ── Label ───────────────────────────────────────────
    if risk_score >= 0.65:
        label = "phishing"
    elif risk_score >= 0.40:
        label = "suspicious"
    elif risk_score < 0.40:
        label = "legit"
    else:
        label = "unknown"

    # Bump to suspicious if there's high disagreement
    if label == "legit" and disagreement > 0.5:
        label = "suspicious"

    rl = _risk_level(risk_score)
    act = _action(label, rl)

    # ── Build reason ────────────────────────────────────
    layer_summary = ", ".join(
        f"{k}={v:.2f}" for k, v in layer_scores.items()
    )
    reason = (
        f"Fusion ({message_type}): risk={risk_score:.2f}, "
        f"layers=[{layer_summary}], disagreement={disagreement:.2f}"
    )

    return FusionResult(
        label=label,
        risk_score=risk_score,
        risk_level=rl,
        action=act,
        confidence=confidence,
        evidence=evidence,
        weights=norm_weights,
        layer_scores=layer_scores,
        layer_labels=layer_labels,
        disagreement=disagreement,
        signals=signals,
        reason=reason,
        qwen_evidence=qwen_evidence,
    )

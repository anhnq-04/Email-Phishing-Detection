"""
xai_engine.py
=============

Main XAI engine.

Public function:
    explain(fusion_result, body, subject, sender_email, message_type, url_results)
"""

from __future__ import annotations

from src.xai.evidence_builder import (
    build_catboost_evidence_items,
    build_disagreement_item,
    build_fastscan_evidence_items,
    build_qwen_evidence_items,
)
from src.xai.ranking import rank_and_deduplicate
from src.xai.recommendation import (
    build_layer_explanations,
    build_recommendation,
    build_summary,
)
from src.xai.xai_models import XAIReport


def explain(
    fusion_result,
    body: str = "",
    subject: str = "",
    sender_email: str = "",
    message_type: str = "email",
    url_results=None,
) -> XAIReport:
    """
    Build user-facing XAI explanation from verified evidence and fusion result.

    This function does NOT call LLM.
    It only uses verified evidence from detection layers.
    """

    evidence_items = []

    # 1. Qwen structured evidence
    evidence_items.extend(build_qwen_evidence_items(fusion_result))

    # 2. CatBoost URL evidence
    evidence_items.extend(build_catboost_evidence_items(url_results))

    # 3. FastScan hard-rule evidence
    evidence_items.extend(build_fastscan_evidence_items(fusion_result))

    # 4. Fusion disagreement evidence
    evidence_items.extend(build_disagreement_item(fusion_result))

    # 5. Deduplicate + rank
    evidence_items = rank_and_deduplicate(evidence_items, limit=8)

    top_signals = []
    seen = set()

    for item in evidence_items:
        if item.signal not in seen:
            top_signals.append(item.signal)
            seen.add(item.signal)

    if not top_signals:
        top_signals = list(getattr(fusion_result, "signals", []) or [])[:5]

    label = getattr(fusion_result, "label", "unknown")
    if label is None:
        label = "unknown"

    risk_score_raw = getattr(fusion_result, "risk_score", 0.5)
    if risk_score_raw is None:
        risk_score = 0.5
    else:
        try:
            risk_score = float(risk_score_raw)
        except (ValueError, TypeError):
            risk_score = 0.5

    risk_level = getattr(fusion_result, "risk_level", "uncertain")
    if risk_level is None:
        risk_level = "uncertain"

    report = XAIReport(
        label=label,
        risk_score=risk_score,
        risk_level=risk_level,
        summary=build_summary(fusion_result, evidence_items),
        evidence_items=evidence_items,
        top_signals=top_signals[:5],
        recommendation=build_recommendation(fusion_result),
        layer_explanations=build_layer_explanations(fusion_result),
    )

    return report
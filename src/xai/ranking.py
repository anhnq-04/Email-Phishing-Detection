"""
ranking.py
==========

Rank and deduplicate XAI evidence items.
"""

from __future__ import annotations

from typing import List, Tuple

from src.xai.xai_models import EvidenceItem
from src.xai.signal_catalog import get_signal_info


SEVERITY_SCORE = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}
DIRECT_HARM_SIGNALS = {
    # Content-level direct harm
    "credential_request": 5,
    "otp_or_code": 5,
    "suspicious_attachment": 4,
    "extortion": 4,
    "threat": 3,
    "financial_lure": 2,

    # URL-level direct harm
    "domain_spoofing": 4,
    "sensitive_url_parameters": 4,
    "suspicious_file_in_url": 4,
    "suspicious_url_keywords": 3,
    "suspicious_url_structure": 2,
    "url_shortener": 2,
    "suspicious_domain_pattern": 2,
    "risky_tld": 1,

    # Soft/contextual signals
    "urgency": 1,
    "call_to_action": 1,
}

def _item_key(item: EvidenceItem) -> Tuple[str, str, str]:
    return (
        item.source_layer,
        item.signal,
        (item.matched_text or "").strip().lower(),
    )


def rank_and_deduplicate(items: List[EvidenceItem], limit: int = 8) -> List[EvidenceItem]:
    seen = set()
    unique: List[EvidenceItem] = []

    for item in items:
        key = _item_key(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    def score(item: EvidenceItem):
        signal_info = get_signal_info(item.signal)
        priority = signal_info.get("priority", 10)
        severity_score = SEVERITY_SCORE.get(item.severity, 1)
        has_span = 1 if item.matched_text else 0
        direct_harm = DIRECT_HARM_SIGNALS.get(item.signal, 0)
        has_case_explanation = 1 if getattr(item, "why_suspicious", "") else 0

        return (
            direct_harm,
            severity_score,
            has_span,
            has_case_explanation,
            priority,
        )

    unique.sort(key=score, reverse=True)
    return unique[:limit]
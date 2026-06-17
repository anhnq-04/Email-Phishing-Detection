"""
xai_models.py
=============

Internal XAI data models.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

@dataclass
class EvidenceItem:
    category: str                     # url | content | entity | attachment | disagreement | layer
    signal: str                       # domain_spoofing, urgency, credential_request, ...
    severity: str                     # critical | high | medium | low
    title: str
    description: str
    source_layer: str = ""            # qwen | catboost | fastscan | fusion
    matched_text: Optional[str] = None

    why_suspicious: str = ""
    user_risk: str = ""
    safe_action: str = ""
    location: str = ""
    confidence: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class XAIReport:
    label: str
    risk_score: float
    risk_level: str
    summary: str
    evidence_items: List[EvidenceItem] = field(default_factory=list)
    top_signals: List[str] = field(default_factory=list)
    recommendation: str = ""
    layer_explanations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "summary": self.summary,
            "evidence_items": [item.to_dict() for item in self.evidence_items],
            "top_signals": self.top_signals,
            "recommendation": self.recommendation,
            "layer_explanations": self.layer_explanations,
        }
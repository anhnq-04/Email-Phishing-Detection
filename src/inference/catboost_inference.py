"""
catboost_inference.py
=====================
Inference 2-model:
  - Lexical CatBoost: always-on
  - HTML CatBoost: optional enrichment
  - Fusion: weighted average

Label:
  0 = Phishing
  1 = Legitimate
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

import math
import numpy as np
from catboost import CatBoostClassifier, Pool

from src.utils.feature_extraction import (
    extract_features_url,
    extract_features_html,
    FEATURES_URL,
    FEATURES_HTML,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]

LEXICAL_MODEL_PATH = str(
    _PROJECT_ROOT / "models" / "catboost" / "catboost_lexical_only.cbm"
)

HTML_MODEL_PATH = str(
    _PROJECT_ROOT / "models" / "catboost" / "catboost_html_only.cbm"
)

LABEL_MAP = {
    0: "Phishing",
    1: "Legitimate",
}

PHISHING_LABEL = 0
LEGIT_LABEL = 1


@dataclass
class CatBoostResult:
    url: str
    label: str
    confidence: float
    phishing_score: float
    lexical_score: float
    html_score: Optional[float] = None
    fetch_success: bool = False
    features: Dict = field(default_factory=dict)
    error: Optional[str] = None
    signals: List[str] = field(default_factory=list)

    def __str__(self):
        return (
            f"CatBoostResult("
            f"url={self.url}, "
            f"label={self.label}, "
            f"confidence={self.confidence}, "
            f"phishing_score={self.phishing_score}, "
            f"lexical_score={self.lexical_score}, "
            f"html_score={self.html_score}, "
            f"fetch_success={self.fetch_success}"
            f")"
        )


class DualCatBoostInference:
    """
    URL phishing inference using:
      1. Lexical CatBoost model
      2. HTML CatBoost model
      3. Weighted score fusion
    """

    def __init__(
        self,
        lexical_model_path: str = LEXICAL_MODEL_PATH,
        html_model_path: str = HTML_MODEL_PATH,
        html_weight: float = 0.25,
        phishing_threshold: float = 0.9,
    ):
        self.lexical_model = self._load_model(lexical_model_path)
        self.html_model = self._load_model(html_model_path)

        self.html_weight = html_weight
        self.lexical_weight = 1.0 - html_weight
        self.phishing_threshold = phishing_threshold

    @staticmethod
    def _load_model(model_path: str) -> CatBoostClassifier:
        model = CatBoostClassifier()
        model.load_model(model_path)
        return model

    @staticmethod
    def _is_valid_number(x) -> bool:
        try:
            return x is not None and not math.isnan(float(x))
        except Exception:
            return False

    @staticmethod
    def _has_complete_html_features(features_html: Dict) -> bool:
        for f in FEATURES_HTML:
            value = features_html.get(f)
            if value is None:
                return False
            try:
                if np.isnan(float(value)):
                    return False
            except Exception:
                return False
        return True

    @staticmethod
    def _get_phishing_proba(model: CatBoostClassifier, pool: Pool) -> float:
        classes = list(model.classes_)
        phish_idx = classes.index(PHISHING_LABEL)
        return float(model.predict_proba(pool)[0][phish_idx])

    @staticmethod
    def _build_pool(features: Dict, feature_names: list[str]) -> Pool:
        row = [[features.get(f, 0) for f in feature_names]]
        return Pool(row, feature_names=feature_names)

    def _predict_lexical(self, features_url: Dict) -> float:
        pool = self._build_pool(features_url, FEATURES_URL)
        return self._get_phishing_proba(self.lexical_model, pool)

    def _predict_html(self, features_html: Dict) -> Optional[float]:
        if not self._has_complete_html_features(features_html):
            return None

        pool = self._build_pool(features_html, FEATURES_HTML)
        return self._get_phishing_proba(self.html_model, pool)

    def _fuse_scores(
        self,
        lexical_score: float,
        html_score: Optional[float],
    ) -> float:
        if html_score is None:
            return lexical_score

        return (
            self.lexical_weight * lexical_score
            + self.html_weight * html_score
        )

    def predict_url(self, url: str) -> CatBoostResult:
        features_url = extract_features_url(url)
        features_html = extract_features_html(url)

        lexical_error = features_url.get("error")
        html_error = features_html.get("error")

        if lexical_error:
            return CatBoostResult(
                url=url,
                label="",
                confidence=0.0,
                phishing_score=0.0,
                lexical_score=0.0,
                html_score=None,
                fetch_success=False,
                features={},
                error=f"Lexical error: {lexical_error}",
            )

        lexical_score = self._predict_lexical(features_url)
        html_score = self._predict_html(features_html)

        fetch_success = html_score is not None

        final_phishing_score = self._fuse_scores(
            lexical_score=lexical_score,
            html_score=html_score,
        )

        if final_phishing_score > self.phishing_threshold:
            pred_class = PHISHING_LABEL
            confidence = final_phishing_score
        else:
            pred_class = LEGIT_LABEL
            confidence = 1.0 - final_phishing_score

        signals = ["domain_spoofing"] if pred_class == PHISHING_LABEL else []

        features = {
            **features_url,
            **features_html,
        }

        error = None
        if html_error and not fetch_success:
            error = f"HTML unavailable: {html_error}"

        return CatBoostResult(
            url=url,
            label=LABEL_MAP[pred_class],
            confidence=round(float(confidence), 4),
            phishing_score=round(float(final_phishing_score), 4),
            lexical_score=round(float(lexical_score), 4),
            html_score=round(float(html_score), 4) if html_score is not None else None,
            fetch_success=fetch_success,
            features=features,
            error=error,
            signals=signals,
        )

    def predict_batch(self, urls: list[str]) -> list[CatBoostResult]:
        results = []

        for url in urls:
            result = self.predict_url(url)
            results.append(result)

            if result.error:
                print(f"[WARN] {url}")
                print(f"       -> {result.error}")

            tag = "!!!" if result.label == "Phishing" else "OK "
            print(f"[{tag}] {url}")
            print(
                f"       -> {result.label} | "
                f"confidence={result.confidence} | "
                f"p_phish={result.phishing_score} | "
                f"lex={result.lexical_score} | "
                f"html={result.html_score} | "
                f"html_ok={result.fetch_success}"
            )

        return results
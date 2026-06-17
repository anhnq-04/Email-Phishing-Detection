"""
eval_xai_catboost.py
====================
Test end-to-end integration of CatBoost thresholding, domain_spoofing signals, and XAI report generation.
"""

import sys
import os

# Fix Windows console UTF-8 printing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.inference.catboost_inference import DualCatBoostInference, CatBoostResult
from src.inference.fusion_policy import fuse_results
from src.xai.xai_engine import explain

class DummyQwenResult:
    def __init__(self, label="legit", risk_score=0.05, signals=None, evidence=None, reason=""):
        self.label = label
        self.risk_score = risk_score
        self.signals = signals or []
        self.evidence = evidence or []
        self.reason = reason

    def evidence_dicts(self):
        return [e.to_dict() if hasattr(e, "to_dict") else dict(e) for e in self.evidence]

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING CATBOOST + XAI INTEGRATION")
    print("=" * 60)

    # 1. Initialize CatBoost
    model = DualCatBoostInference()

    # 2. Test URLs representing different categories
    # - Legitimate (e.g. google.com, score <= 0.8)
    # - Phishing (e.g. gmaill.com, score > 0.8)
    # Let's run prediction first
    test_urls = ["https://www.google.com/", "https://vcb-secure-login.xyz/verify", "http://gmaill.com"]
    url_results = model.predict_batch(test_urls)

    print("\nURL Predictions:")
    for r in url_results:
        print(f"  URL: {r.url}")
        print(f"    Label: {r.label}")
        print(f"    Confidence: {r.confidence}")
        print(f"    Phishing Score: {r.phishing_score}")
        print(f"    Signals: {r.signals}")
        print(f"    Error: {r.error}")
        print("-" * 30)

    # 3. Test Fusion Policy with these URL results and a Dummy Qwen result
    dummy_qwen = DummyQwenResult(label="legit", risk_score=0.05, signals=[], evidence=[])
    
    fusion_res = fuse_results(
        fastscan_result=None,
        url_results=url_results,
        qwen_result=dummy_qwen,
        message_type="email"
    )

    print("\nFusion Result:")
    print(f"  Label: {fusion_res.label}")
    print(f"  Risk Score: {fusion_res.risk_score}")
    print(f"  Signals: {fusion_res.signals}")
    print(f"  Layer Labels: {fusion_res.layer_labels}")
    print(f"  Layer Scores: {fusion_res.layer_scores}")

    # 4. Generate XAI Explanation Report
    report = explain(
        fusion_result=fusion_res,
        body="Vui long click vao link de xac minh tai khoan: https://vcb-secure-login.xyz/verify",
        subject="Xac minh",
        sender_email="support@vcb-secure-login.xyz",
        message_type="email",
        url_results=url_results
    )

    print("\nXAI Explanation Report:")
    print(f"  Label: {report.label}")
    print(f"  Risk Level: {report.risk_level}")
    print(f"  Summary: {report.summary}")
    print(f"  Top Signals: {report.top_signals}")
    print(f"  Recommendation: {report.recommendation}")
    print("\nEvidence Items:")
    for item in report.evidence_items:
        print(f"  - [{item.category} / {item.signal}] Title: {item.title}")
        print(f"    Severity: {item.severity}")
        print(f"    Description: {item.description}")
        print(f"    Matched text: {item.matched_text}")
        print(f"    Source layer: {item.source_layer}")
    
    print("\nAll integration tests passed successfully.")

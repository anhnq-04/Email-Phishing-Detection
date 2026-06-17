"""
test_catboost_inference.py
==========================
Test DualCatBoostInference trên một số URL mẫu.

Chạy từ thư mục gốc:
  python tests/test_catboost_inference.py
"""

import sys
import os
import warnings

from urllib3.exceptions import InsecureRequestWarning

warnings.simplefilter("ignore", InsecureRequestWarning)

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
)

from src.inference.catboost_inference import DualCatBoostInference


TEST_URLS = [
    "https://www.google.com/",
]


if __name__ == "__main__":
    print("=" * 60)
    print("TEST — DualCatBoostInference")
    print("=" * 60)

    model = DualCatBoostInference(
        html_weight=0.25,
        phishing_threshold=0.5,
    )

    results = model.predict_batch(TEST_URLS)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for r in results:
        if r.error:
            print(f"  [WARN] {r.url}")
            print(f"         {r.error}")

        print(f"  [{r.label:11}] {r.url}")
        print(f"    confidence    : {r.confidence}")
        print(f"    p_phishing    : {r.phishing_score}")
        print(f"    lexical_score : {r.lexical_score}")
        print(f"    html_score    : {r.html_score}")
        print(f"    html_success  : {r.fetch_success}")
        print(f"  {'-' * 56}")

        for feat, val in r.features.items():
            print(f"    {feat:<25}: {val}")

        print()
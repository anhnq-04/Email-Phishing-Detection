"""
test_fastscan_inference.py
==========================
Test FastScan trên một số email mẫu.

Chạy từ thư mục gốc:
  python scripts/test_fastscan_inference.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.inference.fastscan_inference import FastScan

# =======================
# TEST CASES
# =======================
TEST_CASES = [
    {
        "name": "Phishing — Blacklist domain",
        "sender": "noreply@vcb-secure.com",
        "body": "Tài khoản của bạn bị khóa. Truy cập ngay https://www.pcn-notictb.shop/com/",
        "headers": None,
    },
    {
        "name": "Phishing — Mạo danh Vietcombank",
        "sender": "security@vcb-verify.xyz",
        "body": "Kính gửi quý khách Vietcombank, vui lòng xác minh tại https://vcb-verify.xyz/login?reset=true",
        "headers": None,
    },
    {
        "name": "Phishing — SPF fail",
        "sender": "support@momo.vn",
        "body": "Tài khoản MoMo của bạn cần xác minh ngay.",
        "headers": "spf=fail dkim=pass",
    },
    {
        "name": "Legit — Whitelist match",
        "sender": "noreply@google.com",
        "body": "Someone signed in to your Google account from a new device.",
        "headers": "spf=pass dkim=pass",
    },
    {
        "name": "Unknown — Email thông thường",
        "sender": "info@company.com",
        "body": "Chào bạn, đây là thông báo từ phòng nhân sự về lịch họp tuần tới.",
        "headers": None,
    },
]

# =======================
# RUN
# =======================
if __name__ == "__main__":
    print("=" * 60)
    print("TEST — FastScan Inference")
    print("=" * 60)

    scanner = FastScan()

    for tc in TEST_CASES:
        result = scanner.fastscan_analyze(
            sender_email=tc["sender"],
            body_text=tc["body"],
            headers=tc["headers"],
        )

        print(f"\n{'─' * 60}")
        print(f"  📌 {tc['name']}")
        print(f"  📧 Sender  : {tc['sender']}")
        print(f"  🔍 Verdict : {result.verdict}")
        print(f"  📝 Reason  : {result.reason}")
        if result.details:
            print(f"  📋 Details : {result.details}")

    print(f"\n{'=' * 60}")

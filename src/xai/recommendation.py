"""
recommendation.py
=================

Build user-facing summary and recommendation.
"""

from __future__ import annotations

from typing import List

from src.xai.xai_models import EvidenceItem


def build_summary(fusion_result, evidence_items):
    label = getattr(fusion_result, "label", "unknown")
    risk_score = float(getattr(fusion_result, "risk_score", 0.5) or 0.5)

    if label in {"phishing", "suspicious"} and evidence_items:
        top = evidence_items[0]
        quote = f"“{top.matched_text}”" if top.matched_text else top.title

        return (
            f"Nội dung được đánh giá là {label} với điểm rủi ro {risk_score:.2f}. "
            f"Bằng chứng quan trọng nhất là {quote}. "
            f"Lý do: {top.why_suspicious or top.description}"
        )

    if label == "legit":
        return (
            f"Nội dung được đánh giá là an toàn hơn, điểm rủi ro {risk_score:.2f}. "
            "Không phát hiện bằng chứng mạnh như yêu cầu mật khẩu/OTP, đe dọa, link giả mạo hoặc tệp đáng ngờ."
        )

    return "Hệ thống chưa đủ bằng chứng cụ thể để kết luận chắc chắn."


def build_recommendation(fusion_result) -> str:
    label = getattr(fusion_result, "label", "unknown")
    risk_level = getattr(fusion_result, "risk_level", "uncertain")

    if label == "phishing" and risk_level == "high":
        return (
            "Không bấm vào liên kết, không tải tệp đính kèm và không nhập mật khẩu/OTP. "
            "Hãy kiểm tra trực tiếp qua website hoặc ứng dụng chính thức."
        )

    if label in {"phishing", "suspicious"}:
        return (
            "Hãy kiểm tra người gửi và đường link qua kênh chính thức trước khi thao tác. "
            "Không cung cấp thông tin nhạy cảm nếu chưa xác minh được nguồn gửi."
        )

    if label == "legit":
        return (
            "Có thể tiếp tục, nhưng vẫn nên cẩn thận với liên kết hoặc tệp đính kèm không mong đợi."
        )

    return "Nên kiểm tra thủ công hoặc chuyển cho người có chuyên môn xác minh."


def build_layer_explanations(fusion_result):
    layer_scores = getattr(fusion_result, "layer_scores", {}) or {}
    layer_labels = getattr(fusion_result, "layer_labels", {}) or {}
    weights = getattr(fusion_result, "weights", {}) or {}

    explanations = {}

    if "qwen" in layer_scores:
        explanations["qwen"] = (
            f"Nội dung văn bản đóng góp {weights.get('llm', 0):.0%} vào quyết định. "
            f"LLM đánh giá nội dung là {layer_labels.get('qwen')} "
            f"với điểm {layer_scores.get('qwen'):.2f}, chủ yếu dựa trên các câu đáng ngờ được highlight."
        )

    if "catboost" in layer_scores:
        explanations["catboost"] = (
            f"URL đóng góp {weights.get('url', 0):.0%} vào quyết định. "
            f"URL model đánh giá điểm rủi ro {layer_scores.get('catboost'):.2f}; "
            "các yếu tố như domain, từ khóa trong URL, HTTPS và cấu trúc link được dùng để đánh giá."
        )

    if "fastscan" in layer_labels:
        explanations["fastscan"] = (
            f"FastScan phát hiện kết quả {layer_labels.get('fastscan')}; "
            "đây là tầng luật nhanh để bắt các dấu hiệu rõ ràng như mạo danh thương hiệu, sender bất thường hoặc rule nguy hiểm."
        )

    return explanations
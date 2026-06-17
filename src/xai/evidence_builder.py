"""
evidence_builder.py
===================

Convert evidence from different layers into user-facing XAI evidence items.
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.xai.xai_models import EvidenceItem
from src.xai.signal_catalog import get_signal_info
from src.xai.url_explainer import explain_url_result


def build_qwen_evidence_items(fusion_result) -> List[EvidenceItem]:
    items: List[EvidenceItem] = []

    qwen_evidence = getattr(fusion_result, "qwen_evidence", []) or []

    for ev in qwen_evidence:
        if not isinstance(ev, dict):
            continue
        if not ev.get("verified", False):
            continue

        signal = ev.get("signal", "")
        exact_span = ev.get("exact_span", "")

        if not signal or not exact_span:
            continue

        # Filter weak one-word evidence for soft signals.
        if signal in {"urgency", "call_to_action"} and len(exact_span.split()) <= 1:
            continue

        info = get_signal_info(signal)

        severity = ev.get("severity") or info["severity"]
        if severity not in {"critical", "high", "medium", "low"}:
            severity = info["severity"]

        field = ev.get("field", "unknown")
        llm_explanation = ev.get("explanation", "").strip()
        why, user_risk, safe_action = _build_case_specific_description(signal, exact_span, field)

        item = EvidenceItem(
            category=info["category"],
            signal=signal,
            severity=severity,
            title=info["title"],
            description=llm_explanation or why,
            source_layer="qwen",
            matched_text=exact_span,
            why_suspicious=why,
            user_risk=user_risk,
            safe_action=safe_action,
            location=field,
            confidence=ev.get("confidence"),
            metadata={
                "field": field,
                "char_start": ev.get("char_start", -1),
                "char_end": ev.get("char_end", -1),
                "verified": ev.get("verified", False),
            },
        )

        items.append(item)

    return items


def build_catboost_evidence_items(url_results) -> List[EvidenceItem]:
    items: List[EvidenceItem] = []

    if not url_results:
        return items

    for r in url_results:
        label = str(getattr(r, "label", "") or "").strip().lower()

        try:
            score = float(getattr(r, "phishing_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0

        # Không phụ thuộc cứng vào label.
        # Nếu label là phishing/suspicious thì build evidence.
        # Nếu label thiếu nhưng score URL đủ cao thì vẫn build evidence.
        if label not in {"phishing", "suspicious"} and score < 0.55:
            continue

        url_items = explain_url_result(r)
        items.extend(url_items)

    return items


def build_fastscan_evidence_items(fusion_result) -> List[EvidenceItem]:
    items: List[EvidenceItem] = []

    layer_labels = getattr(fusion_result, "layer_labels", {}) or {}
    evidence_texts = getattr(fusion_result, "evidence", []) or []

    fastscan_label = str(layer_labels.get("fastscan") or "").upper()

    if fastscan_label == "PHISHING":
        for text in evidence_texts:
            if not str(text).startswith("[FastScan]"):
                continue

            items.append(
                EvidenceItem(
                    category="entity",
                    signal="fastscan_rule",
                    severity="high",
                    title="FastScan phát hiện dấu hiệu rủi ro rõ ràng",
                    description=str(text).replace("[FastScan]", "").strip(),
                    source_layer="fastscan",
                    matched_text=None,
                    metadata={},
                )
            )

    return items


def build_disagreement_item(fusion_result) -> List[EvidenceItem]:
    disagreement = float(getattr(fusion_result, "disagreement", 0.0) or 0.0)

    if disagreement <= 0.50:
        return []

    return [
        EvidenceItem(
            category="disagreement",
            signal="layer_disagreement",
            severity="medium",
            title="Các tầng phân tích chưa hoàn toàn đồng thuận",
            description=(
                "Một số tầng đánh giá rủi ro cao hơn các tầng còn lại. "
                "Hệ thống khuyến nghị người dùng kiểm tra kỹ trước khi thao tác."
            ),
            source_layer="fusion",
            matched_text=None,
            metadata={
                "disagreement": disagreement,
                "layer_scores": getattr(fusion_result, "layer_scores", {}),
                "layer_labels": getattr(fusion_result, "layer_labels", {}),
            },
        )
    ]


def _build_case_specific_description(signal: str, exact_span: str, field: str) -> tuple[str, str, str]:
    if signal == "credential_request":
        return (
            "Đoạn này yêu cầu người dùng cung cấp hoặc xác minh thông tin đăng nhập/thông tin nhạy cảm.",
            "Nếu làm theo, người dùng có thể bị đánh cắp tài khoản, mật khẩu hoặc thông tin cá nhân.",
            "Không nhập mật khẩu/OTP qua liên kết trong tin nhắn. Hãy mở ứng dụng hoặc website chính thức để kiểm tra."
        )

    if signal == "otp_or_code":
        return (
            "Đoạn này đề cập đến OTP/mã xác thực, đây là thông tin tuyệt đối không nên chia sẻ.",
            "Kẻ gian có thể dùng mã này để chiếm quyền đăng nhập hoặc thực hiện giao dịch.",
            "Không gửi OTP cho bất kỳ ai. Chỉ nhập OTP trong ứng dụng chính thức khi chính bạn đang thực hiện thao tác."
        )

    if signal == "urgency":
        return (
            "Đoạn này tạo áp lực thời gian để người dùng hành động nhanh mà không kiểm tra nguồn gửi.",
            "Người dùng dễ bấm link hoặc cung cấp thông tin trước khi kịp xác minh.",
            "Dừng lại và kiểm tra người gửi/link qua kênh chính thức."
        )

    if signal == "call_to_action":
        return (
            "Đoạn này kêu gọi người dùng thực hiện hành động rủi ro như bấm link, đăng nhập, xác minh hoặc tải tệp.",
            "Hành động này có thể dẫn đến trang giả mạo hoặc tải mã độc.",
            "Không thao tác trực tiếp từ tin nhắn nếu chưa xác minh nguồn gửi."
        )

    if signal == "threat":
        return (
            "Đoạn này đe dọa hậu quả tiêu cực để ép người dùng làm theo yêu cầu.",
            "Người dùng có thể bị thao túng tâm lý và làm theo hướng dẫn nguy hiểm.",
            "Không phản hồi theo hướng dẫn trong tin nhắn. Kiểm tra qua tổng đài/website chính thức."
        )

    if signal == "financial_lure":
        return (
            "Đoạn này dùng lợi ích tài chính/quà tặng/khoản tiền để dụ người dùng tương tác.",
            "Người dùng có thể bị dẫn tới trang lừa đảo hoặc bị yêu cầu đóng phí/cung cấp thông tin.",
            "Không cung cấp thông tin cá nhân hoặc thanh toán phí để nhận thưởng."
        )

    return (
        f"Đoạn này khớp với dấu hiệu rủi ro `{signal}` trong nội dung.",
        "Có khả năng người dùng bị dẫn dụ thực hiện thao tác không an toàn.",
        "Kiểm tra lại nguồn gửi trước khi tiếp tục."
    )
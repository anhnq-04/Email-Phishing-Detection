"""
schemas.py
==========
Pydantic models (request / response) for the Phishing Detection API.

Tách schemas ra file riêng để:
  - Tái sử dụng giữa các router
  - Dễ test, dễ maintain
  - Tự động sinh OpenAPI doc chuẩn
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =====================================================
# 1. SHARED ENUMS / CONSTANTS
# =====================================================

EXAMPLE_EMAIL_BODY = (
    "Kính gửi Quý khách, tài khoản của bạn tại Vietcombank "
    "đã bị khóa tạm thời. Vui lòng nhấp vào link sau để xác minh: "
    "https://vcb-secure-login.xyz/verify"
)


# =====================================================
# 2. REQUEST SCHEMAS
# =====================================================

class EmailAnalysisRequest(BaseModel):
    """Request body cho phân tích email phishing."""

    sender_email: str = Field(
        ...,
        description="Địa chỉ email người gửi.",
        examples=["support@vcb-verify.xyz"],
    )
    subject: str = Field(
        default="",
        description="Tiêu đề email.",
        examples=["[Khẩn cấp] Xác minh tài khoản ngay"],
    )
    body: str = Field(
        ...,
        description="Nội dung văn bản email (plain text).",
        examples=[EXAMPLE_EMAIL_BODY],
    )
    urls: Optional[List[str]] = Field(
        default=None,
        description=(
            "Danh sách URL trích xuất từ email (từ extension). "
            "Nếu có, server sẽ dùng list này thay vì tự extract từ body. "
            "URLs KHÔNG được append vào body gửi cho LLM."
        ),
    )
    headers: Optional[str] = Field(
        default=None,
        description="Chuỗi email header thô (SPF/DKIM). Nếu None thì bỏ qua.",
    )
    language: str = Field(
        default="VI",
        description="Ngôn ngữ của email (VI/EN).",
        examples=["VI"],
    )


class SmsAnalysisRequest(BaseModel):
    """Request body cho phân tích SMS phishing."""

    sender: str = Field(
        ...,
        description="Số điện thoại hoặc tên người gửi SMS.",
        examples=["+84901234567", "VIETCOMBANK"],
    )
    content: str = Field(
        ...,
        description="Nội dung tin nhắn SMS.",
        examples=["OTP cua ban la 123456. Truy cap https://vcb-verify.xyz de xac nhan."],
    )
    urls: Optional[List[str]] = Field(
        default=None,
        description="Danh sách URL trích xuất từ SMS (từ extension).",
    )
    language: str = Field(
        default="VI",
        description="Ngôn ngữ của SMS (VI/EN).",
        examples=["VI"],
    )


class UrlAnalysisRequest(BaseModel):
    """Request body cho phân tích URL phishing."""

    url: str = Field(
        ...,
        description="URL cần kiểm tra.",
        examples=["https://vcb-secure-login.xyz/verify"],
    )


class BatchUrlAnalysisRequest(BaseModel):
    """Request body cho phân tích batch URL."""

    urls: List[str] = Field(
        ...,
        description="Danh sách URL cần kiểm tra (tối đa 50).",
        min_length=1,
        max_length=50,
        examples=[["https://vcb-secure-login.xyz", "https://google.com"]],
    )


# =====================================================
# 3. RESPONSE SCHEMAS
# =====================================================

class UrlResult(BaseModel):
    """Kết quả phân tích một URL."""

    url: str
    label: str = Field(description="Phishing | Legitimate")
    confidence: float = Field(ge=0.0, le=1.0)
    phishing_score: float = Field(ge=0.0, le=1.0)
    lexical_score: float = Field(ge=0.0, le=1.0)
    html_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fetch_success: bool
    error: Optional[str] = None


class FastScanResponse(BaseModel):
    """Kết quả Layer 1 — FastScan."""

    verdict: str = Field(description="PHISHING | SAFE | UNKNOWN")
    reason: str
    details: Dict[str, Any] = Field(default_factory=dict)


class EvidenceItemResponse(BaseModel):
    """Một evidence item từ XAI."""

    category: str = Field(description="url | content | entity | disagreement")
    signal: str = Field(description="Tên signal, e.g. domain_spoofing, urgency")
    severity: str = Field(description="critical | high | medium | low")
    title: str
    description: str
    source_layer: str = ""
    matched_text: Optional[str] = Field(
        default=None,
        description="Exact text/URL from the message that grounds this evidence.",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QwenEvidenceResponse(BaseModel):
    """Structured Qwen evidence for frontend highlighting."""

    signal: str = Field(description="Signal name, e.g. domain_spoofing, urgency")
    exact_span: str = Field(description="Exact evidence span copied from input")
    severity: str = Field(description="low | medium | high")
    field: str = Field(
        default="unknown",
        description="Input field where evidence was found: subject | sender | body | message | url | attachment",
    )
    char_start: int = Field(default=-1)
    char_end: int = Field(default=-1)
    verified: bool = Field(default=False)

class XAIReportResponse(BaseModel):
    """XAI explanation report."""

    label: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: str
    summary: str
    evidence_items: List[EvidenceItemResponse] = Field(default_factory=list)
    top_signals: List[str] = Field(default_factory=list)
    recommendation: str = ""
    layer_explanations: Dict[str, str] = Field(default_factory=dict)


class FusionResponse(BaseModel):
    """Kết quả Fusion tổng hợp từ tất cả layers."""

    message_type: str = Field(description="email | sms")
    label: str = Field(description="phishing | suspicious | legit | unknown")
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: str = Field(description="high | medium | low | uncertain")
    action: str = Field(description="block | warn | allow | review")
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)

    qwen_evidence: List[QwenEvidenceResponse] = Field(
        default_factory=list,
        description="Structured Qwen evidence for UI highlighting.",
    )

    weights: Dict[str, float] = Field(default_factory=dict)
    layer_scores: Dict[str, float] = Field(default_factory=dict)
    layer_labels: Dict[str, str] = Field(default_factory=dict)
    disagreement: float = Field(ge=0.0, le=1.0)
    signals: List[str] = Field(default_factory=list)
    reason: str = ""
    xai_report: Optional[XAIReportResponse] = Field(
        default=None,
        description="XAI explanation report with detailed evidence.",
    )

class HealthResponse(BaseModel):
    """Response cho health check."""

    status: str = Field(description="ok | degraded | error")
    version: str
    layers: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Trạng thái từng layer. "
            "FastScan/CatBoost: loaded | unavailable. "
            "Qwen: connected | client_ready | unavailable."
        ),
    )

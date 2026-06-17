"""
url_explainer.py
================

Case-specific URL explanation for CatBoost URL results.

Goal
----
This module converts a URL model result into concrete, user-facing XAI
EvidenceItem objects. It avoids vague explanations such as "the URL looks
suspicious" and instead explains:

1. Which URL/domain was inspected.
2. Which concrete URL properties were risky.
3. Why those properties matter to the user.
4. What the user should do safely.

Compatibility
-------------
The code intentionally uses only the current EvidenceItem fields:
    category, signal, severity, title, description, source_layer,
    matched_text, metadata

Extra XAI details are placed in metadata so this file can replace the old
url_explainer.py without requiring xai_models.py to be changed first.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

from src.xai.signal_catalog import get_signal_info
from src.xai.xai_models import EvidenceItem


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_EVIDENCE_THRESHOLD = 0.55
HIGH_RISK_THRESHOLD = 0.80
VERY_HIGH_RISK_THRESHOLD = 0.90

SUSPICIOUS_KEYWORDS: Dict[str, str] = {
    "login": "đăng nhập",
    "signin": "đăng nhập",
    "verify": "xác minh",
    "verification": "xác minh",
    "confirm": "xác nhận",
    "secure": "bảo mật",
    "security": "bảo mật",
    "account": "tài khoản",
    "update": "cập nhật",
    "password": "mật khẩu",
    "reset": "đặt lại mật khẩu",
    "banking": "ngân hàng",
    "wallet": "ví điện tử",
    "payment": "thanh toán",
    "invoice": "hóa đơn",
    "unlock": "mở khóa",
    "limited": "giới hạn",
    "support": "hỗ trợ",
}

RISKY_TLDS = {
    "xyz",
    "top",
    "icu",
    "click",
    "work",
    "support",
    "live",
    "site",
    "shop",
    "online",
    "monster",
    "buzz",
    "cyou",
    "quest",
}

URL_SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "cutt.ly",
    "rebrand.ly",
    "s.id",
    "bom.so",
}

RISKY_PATH_EXTENSIONS = {
    ".exe",
    ".scr",
    ".bat",
    ".cmd",
    ".com",
    ".js",
    ".vbs",
    ".jar",
    ".apk",
    ".msi",
    ".zip",
    ".rar",
    ".7z",
    ".iso",
    ".html",
    ".htm",
}

SENSITIVE_QUERY_KEYS = {
    "password",
    "passwd",
    "pwd",
    "pass",
    "otp",
    "code",
    "token",
    "auth",
    "session",
    "login",
    "redirect",
    "continue",
    "return",
    "next",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_url(url: str) -> str:
    url = str(url or "").strip()
    if not url:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        return "http://" + url
    return url


def _parse_url(url: str):
    try:
        return urlparse(_normalize_url(url))
    except Exception:
        return urlparse("http://")


def _extract_domain(url: str) -> str:
    parsed = _parse_url(url)
    return (parsed.hostname or "").strip(".").lower()


def _extract_tld(domain: str) -> str:
    parts = [p for p in domain.lower().split(".") if p]
    if len(parts) >= 3 and ".".join(parts[-2:]) == "com.vn":
        return "com.vn"
    return parts[-1] if parts else ""


def _registered_like_domain(domain: str) -> str:
    """
    Lightweight approximation of registered domain.

    This is not a Public Suffix List implementation. It is good enough for
    user-facing explanation such as separating `a.b.example.com` into
    `example.com`, while still handling the common Vietnamese `com.vn` suffix.
    """

    parts = [p for p in domain.lower().split(".") if p]
    if not parts:
        return ""

    if len(parts) >= 3 and ".".join(parts[-2:]) == "com.vn":
        return ".".join(parts[-3:])

    if len(parts) >= 2:
        return ".".join(parts[-2:])

    return domain


def _subdomain_count(domain: str) -> int:
    parts = [p for p in domain.split(".") if p]
    tld = _extract_tld(domain)

    if tld == "com.vn":
        return max(0, len(parts) - 3)

    return max(0, len(parts) - 2)


def _is_ip_domain(domain: str) -> bool:
    try:
        ipaddress.ip_address(domain)
        return True
    except ValueError:
        return False


def _contains_punycode(domain: str) -> bool:
    return any(part.startswith("xn--") for part in domain.split("."))


def _contains_mixed_alpha_digit(label: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", label)) and bool(re.search(r"\d", label))


def _path_extension(path: str) -> str:
    lower_path = path.lower()

    for ext in sorted(RISKY_PATH_EXTENSIONS, key=len, reverse=True):
        if lower_path.endswith(ext) or f"{ext}?" in lower_path:
            return ext

    return ""


def _matched_keywords(url: str) -> List[Dict[str, str]]:
    lower_url = url.lower()
    matches = []

    for keyword, meaning in SUSPICIOUS_KEYWORDS.items():
        if keyword in lower_url:
            matches.append({"keyword": keyword, "meaning": meaning})

    return sorted(matches, key=lambda x: x["keyword"])


def _sensitive_query_keys(query: str) -> List[str]:
    try:
        parsed_query = parse_qs(query, keep_blank_values=True)
    except Exception:
        parsed_query = {}

    found = []

    for key in parsed_query.keys():
        key_lower = key.lower()
        if key_lower in SENSITIVE_QUERY_KEYS or any(k in key_lower for k in SENSITIVE_QUERY_KEYS):
            found.append(key)

    return sorted(set(found))


def _risk_level_from_score(score: float) -> str:
    if score >= VERY_HIGH_RISK_THRESHOLD:
        return "rất cao"
    if score >= HIGH_RISK_THRESHOLD:
        return "cao"
    if score >= MODEL_EVIDENCE_THRESHOLD:
        return "trung bình-cao"
    return "thấp"


def _severity_from_score_and_rules(score: float, reason_codes: List[str]) -> str:
    critical_codes = {
        "ip_domain",
        "punycode_domain",
        "userinfo_in_url",
        "risky_file_extension",
        "sensitive_query_keys",
    }

    if score >= 0.95 or any(code in reason_codes for code in critical_codes):
        return "critical" if score >= HIGH_RISK_THRESHOLD else "high"

    if score >= HIGH_RISK_THRESHOLD:
        return "high"

    return "medium"


def _safe_join(items: List[str], empty: str = "không có dấu hiệu cụ thể ngoài điểm mô hình") -> str:
    items = [str(x).strip() for x in items if str(x).strip()]

    if not items:
        return empty

    if len(items) == 1:
        return items[0]

    return "; ".join(items)


def _make_item(
    *,
    category: str,
    signal: str,
    severity: str,
    title: str,
    description: str,
    source_layer: str,
    matched_text: Optional[str],
    why_suspicious: str = "",
    user_risk: str = "",
    safe_action: str = "",
    location: str = "url",
    confidence: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> EvidenceItem:
    return EvidenceItem(
        category=category,
        signal=signal,
        severity=severity,
        title=title,
        description=description,
        source_layer=source_layer,
        matched_text=matched_text,
        why_suspicious=why_suspicious,
        user_risk=user_risk,
        safe_action=safe_action,
        location=location,
        confidence=confidence,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Reason extraction
# ---------------------------------------------------------------------------


def _analyze_url(url: str, score: float) -> Dict[str, Any]:
    parsed = _parse_url(url)
    domain = _extract_domain(url)
    registered_domain = _registered_like_domain(domain)
    tld = _extract_tld(domain)
    original_lower = str(url or "").lower().strip()
    normalized = _normalize_url(url)
    decoded_path = unquote(parsed.path or "")

    reasons: List[Dict[str, Any]] = []

    def add_reason(code: str, text: str, severity: str = "medium", **extra: Any) -> None:
        reasons.append({"code": code, "text": text, "severity": severity, **extra})

    # Missing scheme is treated as not explicitly safe because the raw message did
    # not show HTTPS to the user.
    if not original_lower.startswith("https://"):
        add_reason(
            "no_https",
            "liên kết không hiển thị HTTPS rõ ràng",
            "medium",
            scheme=parsed.scheme or "missing",
        )

    # Userinfo trick: https://real.com@evil.com
    if "@" in (parsed.netloc or ""):
        add_reason(
            "userinfo_in_url",
            "URL chứa ký tự '@', có thể che giấu domain thật phía sau",
            "high",
        )

    if domain and _is_ip_domain(domain):
        add_reason(
            "ip_domain",
            "URL dùng địa chỉ IP thay vì tên miền tổ chức rõ ràng",
            "high",
            domain=domain,
        )

    if domain and _contains_punycode(domain):
        add_reason(
            "punycode_domain",
            "domain chứa punycode dạng 'xn--', có thể liên quan đến giả mạo ký tự giống nhau",
            "high",
            domain=domain,
        )

    if tld in RISKY_TLDS:
        add_reason(
            "risky_tld",
            f"domain sử dụng đuôi .{tld}, thường cần kiểm tra kỹ hơn nguồn gửi",
            "medium",
            tld=tld,
        )

    if registered_domain in URL_SHORTENERS:
        add_reason(
            "url_shortener",
            f"domain {registered_domain} là dịch vụ rút gọn link, che khuất đích đến thật",
            "medium",
            registered_domain=registered_domain,
        )

    subdomain_count = _subdomain_count(domain)

    if subdomain_count >= 3:
        add_reason(
            "many_subdomains",
            f"domain có nhiều tầng subdomain ({subdomain_count}), dễ làm người dùng nhầm domain chính",
            "medium",
            subdomain_count=subdomain_count,
        )

    hyphen_count = domain.count("-")

    if hyphen_count >= 2:
        add_reason(
            "many_hyphens",
            f"domain chứa nhiều dấu gạch ngang ({hyphen_count}), thường gặp ở domain giả mạo hoặc domain tạo tự động",
            "medium",
            hyphen_count=hyphen_count,
        )

    suspicious_labels = [
        p for p in registered_domain.split(".")
        if _contains_mixed_alpha_digit(p)
    ]

    if suspicious_labels:
        add_reason(
            "mixed_alpha_digit",
            "domain trộn chữ và số trong cùng nhãn, có thể dùng để bắt chước tên thương hiệu",
            "medium",
            labels=suspicious_labels,
        )

    if len(domain) >= 55:
        add_reason(
            "long_domain",
            f"domain dài bất thường ({len(domain)} ký tự), khó kiểm tra nhanh bằng mắt",
            "medium",
            domain_length=len(domain),
        )

    if len(normalized) >= 140:
        add_reason(
            "long_url",
            f"URL rất dài ({len(normalized)} ký tự), có thể dùng để che giấu phần nguy hiểm",
            "medium",
            url_length=len(normalized),
        )

    if len(decoded_path) >= 90:
        add_reason(
            "long_path",
            f"đường dẫn phía sau domain dài bất thường ({len(decoded_path)} ký tự)",
            "medium",
            path_length=len(decoded_path),
        )

    keyword_matches = _matched_keywords(url)

    if keyword_matches:
        readable = [
            f"{m['keyword']} ({m['meaning']})"
            for m in keyword_matches[:6]
        ]

        add_reason(
            "suspicious_keywords",
            "URL chứa từ khóa nhạy cảm: " + ", ".join(readable),
            "medium",
            keywords=keyword_matches,
        )

    ext = _path_extension(decoded_path)

    if ext:
        add_reason(
            "risky_file_extension",
            f"đường dẫn URL trỏ tới tệp có đuôi {ext}, cần thận trọng trước khi mở/tải",
            "high" if ext not in {".html", ".htm", ".zip", ".rar", ".7z"} else "medium",
            extension=ext,
        )

    sensitive_keys = _sensitive_query_keys(parsed.query or "")

    if sensitive_keys:
        add_reason(
            "sensitive_query_keys",
            "tham số URL chứa tên gợi ý thông tin nhạy cảm: " + ", ".join(sensitive_keys[:6]),
            "high",
            query_keys=sensitive_keys,
        )

    reason_codes = [r["code"] for r in reasons]

    return {
        "url": str(url or ""),
        "normalized_url": normalized,
        "domain": domain,
        "registered_domain": registered_domain,
        "tld": tld,
        "scheme": parsed.scheme,
        "path": decoded_path,
        "query": parsed.query or "",
        "score": score,
        "score_label": _risk_level_from_score(score),
        "reasons": reasons,
        "reason_codes": reason_codes,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def explain_url_result(url_result) -> list[EvidenceItem]:
    """
    Convert one URL model result into concrete XAI evidence items.

    Expected url_result attributes, all optional:
        - url: str
        - phishing_score: float
        - lexical_score: float | None
        - html_score: float | None
        - fetch_success: bool | None
        - label: str | None

    Returns:
        list[EvidenceItem]
    """

    raw_url = str(getattr(url_result, "url", "") or "").strip()
    score = _safe_float(getattr(url_result, "phishing_score", 0.0), 0.0)

    if not raw_url:
        return []

    analysis = _analyze_url(raw_url, score)
    domain = analysis["domain"]
    reasons = analysis["reasons"]
    reason_texts = [r["text"] for r in reasons]
    reason_codes = analysis["reason_codes"]

    items: List[EvidenceItem] = []

    base_metadata: Dict[str, Any] = {
        "url": raw_url,
        "normalized_url": analysis["normalized_url"],
        "domain": domain,
        "registered_domain": analysis["registered_domain"],
        "tld": analysis["tld"],
        "phishing_score": score,
        "score_label": analysis["score_label"],
        "lexical_score": getattr(url_result, "lexical_score", None),
        "html_score": getattr(url_result, "html_score", None),
        "fetch_success": getattr(url_result, "fetch_success", None),
        "reason_codes": reason_codes,
        "reasons": reasons,
        "why_suspicious": (
            "URL có các đặc điểm thường xuất hiện trong liên kết phishing, "
            "đặc biệt khi được gửi kèm yêu cầu đăng nhập, xác minh, thanh toán hoặc cung cấp OTP."
        ),
        "user_risk": (
            "Nếu bấm vào, người dùng có thể bị chuyển tới trang giả mạo, "
            "bị đánh cắp mật khẩu/OTP hoặc bị dụ tải tệp độc hại."
        ),
        "safe_action": (
            "Không thao tác trực tiếp từ liên kết này. Hãy mở ứng dụng hoặc website chính thức "
            "bằng cách tự nhập địa chỉ, rồi kiểm tra lại thông báo trong tài khoản."
        ),
    }

    # Main model-backed URL evidence.
    if score >= MODEL_EVIDENCE_THRESHOLD:
        info = get_signal_info("domain_spoofing")
        severity = _severity_from_score_and_rules(score, reason_codes)
        risk_word = analysis["score_label"]
        reason_sentence = _safe_join(reason_texts)

        if domain:
            target = f"domain `{domain}`"
        else:
            target = "domain không đọc được rõ ràng"

        description = (
            f"Liên kết `{raw_url}` trỏ tới {target}. "
            f"URL model đánh giá điểm rủi ro {score:.2f} ({risk_word}). "
            f"Lý do cụ thể: {reason_sentence}. "
            "Rủi ro với người dùng: nếu làm theo liên kết này, người dùng có thể bị dẫn tới "
            "trang giả mạo để lấy mật khẩu, OTP hoặc thông tin tài khoản. "
            "Cách xử lý an toàn: không bấm link; hãy tự mở website/ứng dụng chính thức để kiểm tra."
        )

        items.append(
            _make_item(
                category="url",
                signal="domain_spoofing",
                severity=severity,
                title=info["title"],
                description=description,
                source_layer="catboost",
                matched_text=raw_url,
                why_suspicious=base_metadata["why_suspicious"],
                user_risk=base_metadata["user_risk"],
                safe_action=base_metadata["safe_action"],
                location="url",
                confidence=score,
                metadata=base_metadata,
            )
        )

    # Concrete rule evidence.
    for reason in reasons:
        code = reason["code"]
        text = reason["text"]
        rule_severity = reason.get("severity", "medium")

        if code == "suspicious_keywords":
            title = "URL chứa từ khóa nhạy cảm"
            signal = "suspicious_url_keywords"
            matched_text = raw_url
            risk = (
                "Các từ như login, verify, password, banking thường được dùng "
                "để dụ người dùng đăng nhập hoặc xác minh tài khoản giả."
            )
            action = "Không đăng nhập từ link trong tin nhắn; hãy tự mở trang chính thức."

        elif code == "risky_tld":
            title = "Tên miền dùng đuôi cần kiểm tra kỹ"
            signal = "risky_tld"
            matched_text = domain
            risk = (
                "Một số chiến dịch phishing dùng các đuôi tên miền rẻ/dễ đăng ký "
                "để tạo trang giả mạo nhanh."
            )
            action = "Kiểm tra domain chính thức của tổ chức trước khi thao tác."

        elif code == "many_hyphens":
            title = "Tên miền có cấu trúc bất thường"
            signal = "suspicious_domain_pattern"
            matched_text = domain
            risk = (
                "Domain nhiều dấu gạch ngang dễ được dùng để bắt chước thương hiệu "
                "hoặc làm domain trông có vẻ hợp lệ."
            )
            action = "So sánh domain này với domain chính thức của dịch vụ."

        elif code == "userinfo_in_url":
            title = "URL có thể che giấu domain thật"
            signal = "suspicious_url_structure"
            matched_text = raw_url
            risk = (
                "Phần trước ký tự '@' có thể khiến người dùng tưởng đang vào domain quen thuộc, "
                "trong khi domain thật nằm phía sau."
            )
            action = "Không bấm link dạng này; kiểm tra hostname thật trước."

        elif code == "ip_domain":
            title = "URL dùng địa chỉ IP thay vì tên miền rõ ràng"
            signal = "suspicious_url_structure"
            matched_text = domain
            risk = (
                "Dịch vụ hợp lệ hiếm khi yêu cầu người dùng đăng nhập qua một địa chỉ IP trực tiếp."
            )
            action = "Không nhập thông tin đăng nhập trên trang dùng IP lạ."

        elif code == "punycode_domain":
            title = "Domain có dấu hiệu homograph/punycode"
            signal = "domain_spoofing"
            matched_text = domain
            risk = (
                "Punycode có thể được dùng để tạo domain nhìn giống thương hiệu thật "
                "bằng ký tự tương đồng."
            )
            action = "Không bấm link; mở dịch vụ bằng bookmark hoặc nhập địa chỉ chính thức."

        elif code == "url_shortener":
            title = "Liên kết rút gọn che giấu đích đến"
            signal = "url_shortener"
            matched_text = domain
            risk = (
                "Người dùng không thấy domain thật trước khi bấm, nên khó kiểm tra nguồn gửi."
            )
            action = "Chỉ mở link rút gọn nếu chắc chắn người gửi đáng tin cậy."

        elif code == "risky_file_extension":
            title = "URL trỏ tới tệp cần thận trọng"
            signal = "suspicious_file_in_url"
            matched_text = raw_url
            risk = (
                "Tệp tải về có thể là mã độc, file giả hóa đơn hoặc trang HTML giả mạo đăng nhập."
            )
            action = "Không tải/mở tệp nếu không xác minh được người gửi."

        elif code == "sensitive_query_keys":
            title = "URL có tham số liên quan thông tin nhạy cảm"
            signal = "sensitive_url_parameters"
            matched_text = raw_url
            risk = (
                "Tham số như token, OTP, password hoặc session có thể liên quan đến "
                "chiếm quyền phiên đăng nhập hoặc xác thực."
            )
            action = "Không nhập/chia sẻ mã xác thực hoặc mật khẩu qua liên kết này."

        elif code in {"many_subdomains", "mixed_alpha_digit", "long_domain", "long_url", "long_path", "no_https"}:
            title = "URL có đặc điểm khó kiểm chứng bằng mắt"
            signal = "suspicious_url_structure"
            matched_text = raw_url if code in {"long_url", "long_path", "no_https"} else domain
            risk = (
                "Cấu trúc URL này làm người dùng khó nhận ra domain thật hoặc phần nguy hiểm của đường dẫn."
            )
            action = "Kiểm tra kỹ hostname chính trước khi thao tác."

        else:
            title = "URL có dấu hiệu đáng ngờ"
            signal = "suspicious_url_structure"
            matched_text = raw_url
            risk = (
                "Đặc điểm này có thể làm tăng rủi ro khi liên kết đi kèm yêu cầu thao tác tài khoản."
            )
            action = "Không thao tác nếu chưa xác minh được nguồn gửi."

        items.append(
            _make_item(
                category="url",
                signal=signal,
                severity=rule_severity,
                title=title,
                description=(
                    f"Bằng chứng cụ thể: {text}. "
                    f"Vì sao đáng ngờ: {risk} "
                    f"Cách xử lý an toàn: {action}"
                ),
                source_layer="url_rules",
                matched_text=matched_text,
                why_suspicious=risk,
                user_risk=base_metadata["user_risk"],
                safe_action=action,
                location="url",
                confidence=score,
                metadata={
                    **base_metadata,
                    "rule_code": code,
                    "rule_reason": reason,
                    "why_suspicious": risk,
                    "user_risk": base_metadata["user_risk"],
                    "safe_action": action,
                },
            )
        )

    return items
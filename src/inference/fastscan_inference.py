"""
fastscan_inference.py
=====================

Tầng 1 - FastScan rule-based evidence collector (Production-Ready Final).

Cập nhật:
    - SMS Brandname validation mềm dẻo (tránh false positive do exact match quá cứng).
    - Tín hiệu no_https_url chỉ sinh ra khi URL gốc thực sự dùng http://, không áp cho link không scheme.
    - Bổ sung hard‑rule: cybersquatting + entity_impersonation → PHISHING.
    - Giữ nguyên các ưu điểm: zero network I/O, chấm điểm Core‑Context, chống ReDoS, lazy NER.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
import email.utils
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse
import tldextract

try:
    from gliner import GLiNER
except Exception:
    GLiNER = None

try:
    from src.utils.file_helpers import load_json, load_txt
except Exception:
    load_json = None
    load_txt = None

logger = logging.getLogger(__name__)

# =============================================================================
# Paths + safe config loading
# =============================================================================

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LISTS_DIR = _PROJECT_ROOT / "data" / "processed" / "lists"


def _safe_load_txt(path: Path, default: Optional[Set[str]] = None) -> Set[str]:
    default = default or set()
    if load_txt is None:
        return default
    try:
        values = load_txt(str(path))
        if values is None:
            return default
        return {str(x).strip().lower() for x in values if str(x).strip()}
    except Exception as e:
        logger.warning("Cannot load txt list %s: %s", path, e)
        return default


def _safe_load_json(path: Path, default: Optional[dict] = None) -> dict:
    default = default or {}
    if load_json is None:
        return default
    try:
        data = load_json(str(path))
        if isinstance(data, dict):
            return data
        return default
    except Exception as e:
        logger.warning("Cannot load json list %s: %s", path, e)
        return default


DEFAULT_SHORTENERS: Set[str] = _safe_load_txt(
    _LISTS_DIR / "list_shorten_url.txt",
    default={
        "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
        "buff.ly", "cutt.ly", "rebrand.ly", "s.id", "bom.so",
    },
)

DEFAULT_TRUSTED_FREE_DOMAINS: Set[str] = _safe_load_txt(
    _LISTS_DIR / "free_domains.txt",
    default={
        "gmail.com", "googlemail.com", "outlook.com", "hotmail.com",
        "yahoo.com", "live.com", "icloud.com", "protonmail.com",
        "aol.com", "mail.com", "zoho.com", "yandex.com",
    },
)

DEFAULT_SENSITIVE_KEYWORDS: Set[str] = _safe_load_txt(
    _LISTS_DIR / "sensitive_keywords.txt",
    default={
        "login", "signin", "verify", "verification", "confirm", "secure",
        "security", "account", "update", "password", "reset", "banking",
        "wallet", "payment", "invoice", "unlock", "limited", "otp",
        "code", "token", "xác minh", "đăng nhập", "mật khẩu",
        "tài khoản", "ngân hàng", "thanh toán", "mã otp",
    },
)

DEFAULT_URGENCY_KEYWORDS: Set[str] = _safe_load_txt(
    _LISTS_DIR / "urgency_keywords.txt",
    default={
        "urgent", "immediately", "now", "alert", "action required",
        "account suspended", "limit exceeded", "last warning",
        "khẩn cấp", "ngay lập tức", "cảnh báo", "tạm khóa",
        "đăng nhập ngay", "xác minh gấp", "hết hạn",
    },
)

DEFAULT_WHITELIST: Dict[str, Any] = _safe_load_json(
    _LISTS_DIR / "whitelist.json",
    default={},
)

# =============================================================================
# NER config
# =============================================================================

DEFAULT_GLINER_MODEL = "urchade/gliner_medium-v2.1"
_NER_LABELS = ["organization", "person", "bank"]
_NER_THRESHOLD = 0.50
_NER_MAX_LENGTH = 1024

# =============================================================================
# Result models
# =============================================================================

@dataclass
class FastScanSignal:
    signal: str
    severity: str   # critical | high | medium | low
    title: str
    description: str
    matched_text: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FastScanResult:
    verdict: str       # PHISHING | SUSPICIOUS | UNKNOWN
    reason: str
    score: float = 0.0
    signals: List[FastScanSignal] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"FastScanResult(verdict={self.verdict}, score={self.score:.2f}, "
            f"reason={self.reason}, signals={len(self.signals)}, details={len(self.details)} keys)"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "score": self.score,
            "signals": [s.to_dict() for s in self.signals],
            "details": self.details,
        }


# =============================================================================
# FastScan Class
# =============================================================================

class FastScan:
    """FastScan Tầng 1 - Production-Ready Final."""

    _ner_model_cache: Optional[Any] = None

    _BLACKLIST_SKIP_DOMAINS = {
        "gmail.com", "googlemail.com", "outlook.com", "hotmail.com",
        "yahoo.com", "live.com", "icloud.com", "protonmail.com",
        "aol.com", "mail.com", "zoho.com", "yandex.com",
        "google.com", "facebook.com", "microsoft.com", "apple.com",
        "amazon.com", "twitter.com", "x.com", "linkedin.com", "instagram.com",
    }

    _FOOTER_SAFE_BRANDS = {
        "zalo", "whatsapp", "telegram", "viber", "messenger", "skype", "wechat", "line",
    }

    _FOOTER_PATTERNS = [
        re.compile(p, re.IGNORECASE) for p in [
            r"(liên\s*hệ|contact|hotline|hỗ\s*trợ|support)",
            r"(theo\s*dõi|follow\s*us|kết\s*nối)",
            r"(footer|chân\s*trang|unsubscribe|hủy\s*đăng\s*ký)",
            r"(©|copyright|\d{4}\s*all\s*rights)",
            r"(email\s*:|phone\s*:|điện\s*thoại\s*:)",
            r"(zalo\s*:|telegram\s*:|whatsapp\s*:)",
            r"(facebook\s*:|fanpage|social)",
        ]
    ]

    _RISKY_TLDS = {
        "xyz", "top", "icu", "click", "work", "support",
        "live", "site", "shop", "online", "monster", "buzz", "cyou", "quest",
    }

    _COMMON_SHORTNAMES = {
        "vcb": "vietcombank",
        "tcb": "techcombank",
        "vtb": "vietinbank",
        "mbb": "mbbank",
        "mb": "mbbank",
        "bid": "bidv",
        "vpb": "vpbank",
        "tpb": "tpbank",
        "vnpt": "vinaphone",
        "mbf": "mobifone",
    }

    # SMS sender thường dùng các từ chung chung, cần loại trừ để tránh false match
    _SMS_GENERIC_SENDERS = {"cskh", "hotro", "info", "noreply", "admin", "support", "contact"}

    _CORE_SIGNALS = {
        "blacklist_hit", "homograph", "punycode_domain", "ip_url",
        "header_from_mismatch", "reply_to_mismatch", "entity_impersonation",
        "brand_link_mismatch", "unknown_entity_mentioned", "cybersquatting"
    }

    _SIGNAL_WEIGHTS = {
        "blacklist_hit": 0.95,
        "homograph": 0.90,
        "punycode_domain": 0.85,
        "ip_url": 0.75,
        "header_from_mismatch": 0.80,
        "reply_to_mismatch": 0.75,
        "entity_impersonation": 0.70,
        "brand_link_mismatch": 0.70,
        "unknown_entity_mentioned": 0.50,
        "cybersquatting": 0.65,
        "dmarc_fail": 0.45,

        "message_id_mismatch": 0.25,
        "spf_fail": 0.20,
        "dkim_fail": 0.20,
        "auth_soft_fail": 0.05,
        "url_shortener": 0.15,
        "foreign_domain": 0.10,
        "sensitive_url_keyword": 0.15,
        "risky_tld": 0.10,
        "many_urls": 0.05,
        "no_https_url": 0.05,
        "sensitive_content_keyword": 0.05,
        "urgency_keyword": 0.10,
        "hidden_text_detected": 0.15,

        "trusted_sender": -0.25,
        "official_link": -0.15,
    }

    def __init__(
        self,
        whitelist: Optional[Dict[str, Any]] = None,
        shorteners: Optional[Set[str]] = None,
        sensitive_keywords: Optional[Set[str]] = None,
        urgency_keywords: Optional[Set[str]] = None,
        trusted_free_domains: Optional[Set[str]] = None,
        gliner_model: str = DEFAULT_GLINER_MODEL,
        enable_ner: bool = True,
        enable_local_blacklist: bool = True,
    ):
        self.whitelist = whitelist if whitelist is not None else DEFAULT_WHITELIST
        self.shorteners = {str(x).strip().lower() for x in (shorteners or DEFAULT_SHORTENERS) if str(x).strip()}
        self.sensitive_keywords = {str(x).strip().lower() for x in (sensitive_keywords or DEFAULT_SENSITIVE_KEYWORDS) if str(x).strip()}
        self.urgency_keywords = {str(x).strip().lower() for x in (urgency_keywords or DEFAULT_URGENCY_KEYWORDS) if str(x).strip()}
        self.trusted_free_domains = {str(x).strip().lower() for x in (trusted_free_domains or DEFAULT_TRUSTED_FREE_DOMAINS) if str(x).strip()}

        self.enable_local_blacklist = enable_local_blacklist
        self._local_blacklist_set: Set[str] = set()

        self._ner_model = None
        self.gliner_model = gliner_model
        self.enable_ner = enable_ner

        # Precompile regex
        self._re_email = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b")
        self._re_url_explicit = re.compile(r"(https?://[^\s<>'\"\)\]]+|(?<!@)www\.[^\s<>'\"\)\]]+)", re.IGNORECASE)
        self._re_bare_domain = re.compile(
            r"(?<![@\w.-])(?:[a-zA-Z0-9-]{1,63}\.)+(?:com\.vn|com|net|org|vn|edu|gov|info|biz|io|co|xyz|top|icu|click|work|support|live|site|shop|online|monster|buzz|cyou|quest)(?:/[^\s<>'\"\)\]]*)?",
            re.IGNORECASE,
        )
        self._re_non_alnum = re.compile(r"[^a-z0-9]")
        self._re_ip_hostname = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_unicode(text: str) -> str:
        return unicodedata.normalize("NFKC", str(text or ""))

    @staticmethod
    def _clean_text(text: str) -> str:
        return str(text or "").replace("\u200b", "").replace("\ufeff", "").strip()

    def _extract_email_address(self, sender: str) -> str:
        sender = str(sender or "").strip()
        _, email_address = email.utils.parseaddr(sender)
        if email_address and "@" in email_address:
            return email_address.lower()
        match = self._re_email.search(sender)
        return match.group(0).lower() if match else sender.lower()

    @staticmethod
    def extract_domain(url_or_email: str) -> str:
        value = str(url_or_email or "").strip().lower()
        if not value:
            return ""
        value = value.replace("[.]", ".").replace("(.)", ".")
        value = value.replace("hxxp://", "http://").replace("hxxps://", "https://")
        _, email_address = email.utils.parseaddr(value)
        if email_address and "@" in email_address:
            value = email_address.split("@")[-1]
        else:
            if "@" in value:
                value = value.split("@")[-1]
        ext = tldextract.extract(value)
        if not ext.domain or not ext.suffix:
            return ""
        return f"{ext.domain}.{ext.suffix}".lower()

    @staticmethod
    def _extract_tld(domain: str) -> str:
        ext = tldextract.extract(domain)
        return ext.suffix.lower() if ext.suffix else ""

    def _normalize_url_like(self, value: str) -> str:
        """Chuẩn hóa URL, nhưng giữ nguyên scheme gốc nếu không có."""
        value = str(value or "").strip()
        value = value.replace("[.]", ".").replace("(.)", ".")
        value = value.replace("hxxps://", "https://").replace("hxxp://", "http://")
        # Chỉ thêm http:// khi cần thiết cho urlparse (giữ nguyên scheme nếu đã có)
        if not value.startswith(("http://", "https://")):
            # Nếu là dạng www. hoặc domain thuần, thêm http:// tạm để parse, nhưng sẽ ghi nhận scheme rỗng
            value = "http://" + value
        return value

    def extract_urls(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Trích xuất URL và trả về (normalized_urls, raw_urls).
        raw_urls là các URL đã được làm sạch defang nhưng chưa được thêm scheme giả.
        """
        text = str(text or "")
        if not text:
            return [], []
        normalized = text.replace("[.]", ".").replace("(.)", ".")
        normalized = normalized.replace("hxxps://", "https://").replace("hxxp://", "http://")

        raw_urls = self._re_url_explicit.findall(normalized)
        text_no_email = self._re_email.sub(" ", normalized)
        raw_urls.extend(self._re_bare_domain.findall(text_no_email))

        cleaned_normalized = []
        cleaned_raw = []
        seen = set()
        for url in raw_urls:
            url = url.strip(".,;:!?)\"]}'")
            if not url:
                continue
            nurl = self._normalize_url_like(url)
            key = nurl.lower()
            if key not in seen:
                cleaned_normalized.append(nurl)
                cleaned_raw.append(url)   # giữ nguyên raw (chưa thêm scheme)
                seen.add(key)
        return cleaned_normalized, cleaned_raw

    # -------------------------------------------------------------------------
    # Whitelist & SMS Sender Matchers
    # -------------------------------------------------------------------------
    def _official_domain_set(self) -> Set[str]:
        domains = set()
        for official in self.whitelist.values():
            values = official if isinstance(official, list) else [official]
            for d in values:
                root = self.extract_domain(str(d))
                if root:
                    domains.add(root)
        return domains

    def _blacklist_should_skip(self, domain: str) -> bool:
        domain_root = self.extract_domain(domain)
        if not domain_root:
            return True
        return (domain_root in self._BLACKLIST_SKIP_DOMAINS or
                domain_root in self.trusted_free_domains or
                domain_root in self._official_domain_set())

    def is_official_domain(self, domain: str, official: Union[str, List[str]]) -> bool:
        domain_root = self.extract_domain(domain)
        if not domain_root:
            return False
        official_values = official if isinstance(official, list) else [official]
        for item in official_values:
            official_root = self.extract_domain(str(item))
            if official_root and domain_root == official_root:
                return True
        return False

    def _is_valid_sms_sender(self, sender: str, entity: str) -> bool:
        """
        SMS sender validation mềm dẻo:
        - Ưu tiên exact match sau khi làm sạch.
        - Cho phép sender là tập con của entity hoặc ngược lại nếu sender không phải từ khóa chung chung và đủ dài.
        - Hỗ trợ shortname mapping.
        """
        s_clean = self._re_non_alnum.sub("", str(sender or "").lower())
        e_clean = self._re_non_alnum.sub("", str(entity or "").lower())
        if not s_clean or not e_clean:
            return False

        # 1. Exact match
        if s_clean == e_clean:
            return True

        # 2. Shortname mapping
        for short, full in self._COMMON_SHORTNAMES.items():
            if s_clean == short and e_clean == full:
                return True
            if s_clean == full and e_clean == short:
                return True

        # 3. Subset match có kiểm soát
        # Điều kiện: sender không được là từ khóa chung chung, phải có ít nhất 3 ký tự
        if len(s_clean) >= 3 and s_clean not in self._SMS_GENERIC_SENDERS:
            # Nếu entity chứa sender (VD: "sacombank" chứa "scb" không đúng, nhưng "vietcombank" chứa "vcb")
            # hoặc sender chứa entity (hiếm, nhưng có thể)
            if s_clean in e_clean or e_clean in s_clean:
                # Tránh trường hợp sender quá ngắn (1-2 ký tự) vô tình khớp với entity lớn
                if len(s_clean) >= 3:
                    return True

        return False

    # -------------------------------------------------------------------------
    # Cybersquatting Levenshtein 1D
    # -------------------------------------------------------------------------
    @staticmethod
    def _is_cybersquatting(domain: str, official_domain: str) -> bool:
        d = domain.lower().strip()
        o = official_domain.lower().strip()
        if not d or not o:
            return False
        d_main = d.rsplit('.', 1)[0] if '.' in d else d
        o_main = o.rsplit('.', 1)[0] if '.' in o else o
        if d_main == o_main:
            return False
        suspicious_keywords = ["-secure", "-login", "-verify", "-support", ".com-", "vcb-", "shopee-"]
        if any(s in d_main for s in suspicious_keywords) and o_main in d_main:
            return True
        if abs(len(d_main) - len(o_main)) > 3 or len(d_main) <= 3 or len(o_main) <= 3:
            return False
        if len(set(d_main) & set(o_main)) < min(len(d_main), len(o_main)) / 2:
            return False
        prev = list(range(len(o_main) + 1))
        for ca in d_main:
            curr = [prev[0] + 1]
            for j, cb in enumerate(o_main):
                curr.append(min(prev[j + 1] + 1, curr[-1] + 1, prev[j] + (ca != cb)))
            prev = curr
        return prev[-1] <= 2

    # -------------------------------------------------------------------------
    # Signal management
    # -------------------------------------------------------------------------
    @staticmethod
    def _add_signal(
        signals: List[FastScanSignal],
        *,
        signal: str,
        severity: str,
        title: str,
        description: str,
        matched_text: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        metadata = metadata or {}
        key = (signal, str(matched_text or "").lower(),
               str(metadata.get("domain", "")).lower(),
               str(metadata.get("entity", "")).lower())
        for existing in signals:
            existing_key = (existing.signal, str(existing.matched_text or "").lower(),
                            str(existing.metadata.get("domain", "")).lower(),
                            str(existing.metadata.get("entity", "")).lower())
            if existing_key == key:
                return
        signals.append(FastScanSignal(
            signal=signal, severity=severity, title=title, description=description,
            matched_text=matched_text, confidence=max(0.0, min(1.0, float(confidence))),
            metadata=metadata,
        ))

    @staticmethod
    def _has_suspicious_unicode(domain: str) -> bool:
        domain = str(domain or "").strip().lower()
        if not domain:
            return False
        if "xn--" in domain:
            return True
        normalized = unicodedata.normalize("NFKC", domain)
        if normalized != domain:
            return True
        return any(ord(ch) > 127 for ch in domain)

    def _url_has_sensitive_keyword(self, value: str) -> bool:
        lower = str(value or "").lower()
        return any(kw in lower for kw in self.sensitive_keywords)

    # -------------------------------------------------------------------------
    # NER Lazy Loading
    # -------------------------------------------------------------------------
    def _get_ner_model(self):
        if self._ner_model is not None:
            return self._ner_model
        if not self.enable_ner or GLiNER is None:
            return None
        try:
            if FastScan._ner_model_cache is None:
                FastScan._ner_model_cache = GLiNER.from_pretrained(self.gliner_model)
                logger.info("GLiNER model initialized: %s", self.gliner_model)
            self._ner_model = FastScan._ner_model_cache
            return self._ner_model
        except Exception as e:
            logger.error("GLiNER load failed: %s", e)
            self._ner_model = None
            return None

    def _extract_ner_entities(self, text: str) -> List[str]:
        model = self._get_ner_model()
        if model is None:
            return []
        try:
            truncated = str(text or "")[:_NER_MAX_LENGTH]
            entities = model.predict_entities(truncated, _NER_LABELS, threshold=_NER_THRESHOLD)
            seen = set()
            result = []
            for ent in entities:
                name = str(ent.get("text", "")).strip()
                if not name:
                    continue
                key = name.lower()
                if key not in seen:
                    result.append(name)
                    seen.add(key)
            return result
        except Exception as e:
            logger.debug("NER extraction error: %s", e)
            return []

    def _entity_mentioned(self, whitelist_entity: str, ner_entities: List[str], text: str) -> bool:
        entity_lower = str(whitelist_entity or "").lower().strip()
        text_lower = str(text or "").lower()
        if not entity_lower:
            return False
        entity_words = entity_lower.split()
        for ner_entity in ner_entities:
            ner_lower = str(ner_entity or "").lower().strip()
            if ner_lower == entity_lower:
                return True
            ner_words = ner_lower.split()
            if entity_words and all(w in ner_words for w in entity_words):
                return True
        try:
            pattern = r"\b" + re.escape(entity_lower) + r"\b"
            if re.search(pattern, text_lower):
                return True
        except re.error:
            pass
        compact_text = self._re_non_alnum.sub("", text_lower)
        compact_entity = self._re_non_alnum.sub("", entity_lower)
        if compact_entity and len(compact_entity) >= 4 and compact_entity in compact_text:
            return True
        return False

    def _is_in_footer_context(self, brand: str, text: str) -> bool:
        brand_lower = str(brand or "").lower()
        text_lower = str(text or "").lower()
        if not brand_lower or not text_lower:
            return False
        try:
            match = re.search(r"\b" + re.escape(brand_lower) + r"\b", text_lower)
            if not match:
                return False
            start = max(0, match.start() - 300)
            end = min(len(text_lower), match.end() + 300)
            context = text_lower[start:end]
            return any(p.search(context) for p in self._FOOTER_PATTERNS)
        except re.error:
            return False

    # -------------------------------------------------------------------------
    # Collector Modules
    # -------------------------------------------------------------------------
    def _collect_auth_signals(self, headers: Optional[str], signals: List[FastScanSignal]) -> Dict[str, str]:
        if not headers:
            return {"spf": "none", "dkim": "none", "dmarc": "none"}
        spf_status = self._extract_auth_status(headers, "spf")
        dkim_status = self._extract_auth_status(headers, "dkim")
        dmarc_status = self._extract_auth_status(headers, "dmarc")

        if spf_status == "fail":
            self._add_signal(signals, signal="spf_fail", severity="medium", title="SPF Fail", description="SPF authentication failed.")
        elif spf_status in {"softfail", "neutral"}:
            self._add_signal(signals, signal="auth_soft_fail", severity="low", title="SPF Softfail", description="SPF softfailed.")
        if dkim_status == "fail":
            self._add_signal(signals, signal="dkim_fail", severity="medium", title="DKIM Fail", description="DKIM signature validation failed.")
        if dmarc_status == "fail":
            self._add_signal(signals, signal="dmarc_fail", severity="high", title="DMARC Fail", description="DMARC alignment rejected.")

        ret_match = re.search(r"^Return-Path:\s*<?([^>\s\n]+)>?", headers, re.IGNORECASE | re.MULTILINE)
        from_match = re.search(r"^From:\s*.*<?([^>\s\n]+)>?", headers, re.IGNORECASE | re.MULTILINE)
        if ret_match and from_match:
            ret_domain = self.extract_domain(ret_match.group(1))
            from_domain = self.extract_domain(from_match.group(1))
            if ret_domain and from_domain and ret_domain != from_domain:
                self._add_signal(signals, signal="header_from_mismatch", severity="high", title="Return-Path Mismatch",
                                 description=f"{ret_domain} != {from_domain}", metadata={"ret_domain": ret_domain, "from_domain": from_domain})

        reply_match = re.search(r"^Reply-To:\s*<?([^>\s\n]+)>?", headers, re.IGNORECASE | re.MULTILINE)
        if reply_match and from_match:
            reply_domain = self.extract_domain(reply_match.group(1))
            from_domain_r = self.extract_domain(from_match.group(1))
            if reply_domain and from_domain_r and reply_domain != from_domain_r:
                self._add_signal(signals, signal="reply_to_mismatch", severity="high", title="Reply-To Mismatch",
                                 description=f"{reply_domain} != {from_domain_r}")

        msgid_match = re.search(r"^Message-ID:\s*<[^@\s\n]+@([^>\s\n]+)>", headers, re.IGNORECASE | re.MULTILINE)
        if msgid_match and from_match:
            msgid_domain = self.extract_domain(msgid_match.group(1))
            from_domain_m = self.extract_domain(from_match.group(1))
            if msgid_domain and from_domain_m and msgid_domain != from_domain_m:
                self._add_signal(signals, signal="message_id_mismatch", severity="medium", title="Message-ID Mismatch",
                                 description="Message-ID domain differs.")

        return {"spf": spf_status, "dkim": dkim_status, "dmarc": dmarc_status}

    @staticmethod
    def _extract_auth_status(headers: str, key: str) -> str:
        match = re.search(rf"\b{re.escape(key)}\s*=\s*([a-zA-Z]+)", headers, re.IGNORECASE)
        return match.group(1).lower() if match else "none"

    def _collect_blacklist_signals(self, sender_domain: str, urls: List[str], signals: List[FastScanSignal]):
        if not self.enable_local_blacklist:
            return
        targets = []
        if sender_domain:
            targets.append(sender_domain)
        for url in urls[:5]:
            d = self.extract_domain(url)
            if d:
                targets.append(d)
        for target in sorted(set(targets)):
            if self._blacklist_should_skip(target):
                continue
            if target in self._local_blacklist_set:
                self._add_signal(signals, signal="blacklist_hit", severity="critical",
                                 title="Blacklisted Domain Spotted", description=f"Domain {target} found in local threat feed.",
                                 matched_text=target, metadata={"domain": target})

    def inject_local_blacklist_feed(self, domains_list: List[str]):
        for d in domains_list:
            clean_d = self.extract_domain(d)
            if clean_d:
                self._local_blacklist_set.add(clean_d)

    def _collect_url_signals(self, sender_domain: str, urls: List[str], raw_urls: List[str], signals: List[FastScanSignal]) -> Dict[str, Any]:
        url_domains = []
        foreign_domains = set()
        shortener_urls = []

        if len(urls) >= 5:
            self._add_signal(signals, signal="many_urls", severity="low", title="High Link Density", description="Too many URLs embedded inside text body.")

        # Tạo mapping từ normalized -> raw để kiểm tra scheme gốc
        raw_by_norm = {}
        for norm, raw in zip(urls, raw_urls):
            raw_by_norm[norm] = raw

        for url in urls:
            parsed = urlparse(url)
            hostname = parsed.hostname or ""
            ip_match = self._re_ip_hostname.match(hostname)
            if ip_match:
                if all(0 <= int(part) <= 255 for part in ip_match.groups()):
                    self._add_signal(signals, signal="ip_url", severity="high",
                                     title="IP-Address-Based URL", description=f"URL uses raw IP hostname: {hostname}",
                                     matched_text=url)
                    continue

            domain = self.extract_domain(url)
            if not domain:
                continue

            url_domains.append(domain)
            if domain != sender_domain and domain not in self.shorteners:
                foreign_domains.add(domain)

            if self._has_suspicious_unicode(domain):
                sig_name = "punycode_domain" if "xn--" in domain else "homograph"
                self._add_signal(signals, signal=sig_name, severity="critical", title="Unicode Homograph Attack", description=f"Suspicious Unicode/IDN domain detected: {domain}", matched_text=domain)

            if domain in self.shorteners:
                shortener_urls.append(url)
                self._add_signal(signals, signal="url_shortener", severity="medium", title="Obfuscated Short URL", matched_text=url)

            if self._extract_tld(domain) in self._RISKY_TLDS:
                self._add_signal(signals, signal="risky_tld", severity="medium", title="Untrusted TLD Extension", matched_text=domain)

            # Chỉ gắn no_https_url nếu URL gốc thực sự bắt đầu bằng http://
            raw_url = raw_by_norm.get(url, url)
            if raw_url.startswith("http://") and not raw_url.startswith("https://"):
                # Đảm bảo đây là http rõ ràng (không phải do ta thêm)
                self._add_signal(signals, signal="no_https_url", severity="low", title="Insecure Transport Protocol (HTTP)", matched_text=url)

            if self._url_has_sensitive_keyword(url):
                self._add_signal(signals, signal="sensitive_url_keyword", severity="medium", title="Credential-Harvesting Keywords in URL", matched_text=url)

        for fd in sorted(foreign_domains):
            self._add_signal(signals, signal="foreign_domain", severity="low", title="Cross-Domain Redirection Link", matched_text=fd)

        return {"url_count": len(urls), "url_domains": sorted(set(url_domains)), "shortener_urls": shortener_urls}

    def _collect_entity_signals(self, sender_email: str, sender_domain: str, subject: str,
                                body_text: str, urls: List[str], message_type: str, signals: List[FastScanSignal]) -> Dict[str, Any]:
        full_text = f"{subject}\n{body_text}".strip()
        sender_lower = sender_email.lower()
        early_text = f"{subject}\n{body_text[:600]}".lower()
        ner_entities = self._extract_ner_entities(full_text)
        detected_entities = []
        claimed_entities = []

        for entity_lower, official_domain in self.whitelist.items():
            if not entity_lower.strip():
                continue
            entity_in_body = self._entity_mentioned(entity_lower, ner_entities, full_text)
            entity_in_sender = self._entity_mentioned(entity_lower, [], sender_lower)
            if not entity_in_body and not entity_in_sender:
                continue

            if entity_lower in self._FOOTER_SAFE_BRANDS and entity_in_body:
                if self._is_in_footer_context(entity_lower, full_text):
                    continue

            detected_entities.append(entity_lower)
            brand_is_claimed = (entity_in_sender or self._entity_mentioned(entity_lower, [], early_text))

            if brand_is_claimed:
                claimed_entities.append(entity_lower)

                if message_type == "email":
                    if self.is_official_domain(sender_domain, official_domain):
                        self._add_signal(signals, signal="trusted_sender", severity="low", title="Verified Official Sender Domain", matched_text=sender_domain)
                    else:
                        self._add_signal(signals, signal="entity_impersonation", severity="high", title="Brand Impersonation Detected", matched_text=sender_email)
                        off_d = official_domain[0] if isinstance(official_domain, list) else official_domain
                        if self._is_cybersquatting(sender_domain, str(off_d)):
                            self._add_signal(signals, signal="cybersquatting", severity="high", title="Typosquatting/Cybersquatting Domain Layout", matched_text=sender_domain)

                elif message_type == "sms":
                    if self._is_valid_sms_sender(sender_email, entity_lower):
                        self._add_signal(signals, signal="trusted_sender", severity="low", title="Legitimate Brandname SMS Signature", matched_text=sender_email)
                    else:
                        self._add_signal(signals, signal="entity_impersonation", severity="high", title="Spoofed Brandname SMS Content", matched_text=sender_email)

            # URL mismatch
            for url in urls:
                url_domain = self.extract_domain(url)
                if not url_domain or url_domain in self.shorteners:
                    continue
                if self.is_official_domain(url_domain, official_domain):
                    self._add_signal(signals, signal="official_link", severity="low", title="Valid Identity Reference Anchor", matched_text=url)
                    continue
                if self._url_has_sensitive_keyword(url) and brand_is_claimed:
                    self._add_signal(signals, signal="brand_link_mismatch", severity="high", title="Brand Mentioned with Malicious Destination", matched_text=url)

        # Dynamic entity detection
        whitelist_names = set(self.whitelist.keys())
        for ent in ner_entities:
            ent_lower = ent.lower().strip()
            if ent_lower not in whitelist_names and len(ent_lower) > 3:
                if any(kw in ent_lower for kw in ["bank", "ngân hàng", "tập đoàn", "company", "corp", "group"]):
                    if self._entity_mentioned(ent, [], early_text) or self._entity_mentioned(ent, [], sender_lower):
                        self._add_signal(signals, signal="unknown_entity_mentioned", severity="high", title="Unregistered Critical Entity Target", matched_text=ent)

        return {"ner_entities": ner_entities, "detected_entities": sorted(set(detected_entities)), "claimed_entities": sorted(set(claimed_entities))}

    def _collect_content_signals(self, subject: str, body_text: str, signals: List[FastScanSignal]) -> Dict[str, Any]:
        text = f"{subject}\n{body_text}".lower()
        found_sensitive = [kw for kw in self.sensitive_keywords if kw and kw in text]
        if found_sensitive:
            shown = sorted(set(found_sensitive))[:12]
            self._add_signal(signals, signal="sensitive_content_keyword", severity="low", title="Contextual Sensitive Keywords Match", matched_text=", ".join(shown))

        found_urgent = [kw for kw in self.urgency_keywords if kw and kw in text]
        if found_urgent:
            shown_u = sorted(set(found_urgent))[:10]
            self._add_signal(signals, signal="urgency_keyword", severity="medium", title="Social Engineering Urgency Trigger", matched_text=", ".join(shown_u))

        hidden_hints = []
        if "\u200b" in body_text or "\ufeff" in body_text:
            hidden_hints.append("zero-width obfuscation layout")
        if hidden_hints:
            self._add_signal(signals, signal="hidden_text_detected", severity="medium", title="Obfuscated Text Masking Attempt", description="Body contains zero-width characters used for text obfuscation.")

        return {"sensitive_keywords_found": sorted(set(found_sensitive)), "urgency_keywords_found": sorted(set(found_urgent))}

    # -------------------------------------------------------------------------
    # Scoring & Verdict
    # -------------------------------------------------------------------------
    def _score_signals(self, signals: List[FastScanSignal]) -> float:
        core_score = 0.0
        context_modifier = 0.0
        trust_discount = 0.0

        for s in signals:
            weight = self._SIGNAL_WEIGHTS.get(s.signal, 0.05)
            val = weight * s.confidence

            if s.signal in self._CORE_SIGNALS or s.signal == "dmarc_fail":
                if val > core_score:
                    core_score = val
            elif val < 0:
                trust_discount += abs(val)
            else:
                context_modifier += val

        if core_score > 0:
            final_score = core_score + min(0.20, context_modifier) - trust_discount
        else:
            final_score = min(0.15, context_modifier) - trust_discount

        return max(0.0, min(1.0, final_score))

    def _decide_fastscan_verdict(self, signals: List[FastScanSignal]) -> Tuple[str, str, float]:
        score = self._score_signals(signals)
        names = {s.signal for s in signals}

        if "blacklist_hit" in names:
            return ("PHISHING", "FastScan: Domain match detected inside intelligence threat feed database.", score)
        if "homograph" in names or "punycode_domain" in names:
            return ("PHISHING", "FastScan: High confidence Internationalized Homograph deceptive domain target.", score)
        if "entity_impersonation" in names and "brand_link_mismatch" in names:
            return ("PHISHING", "FastScan: Brand impersonation claim bundled with dynamic link mismatch payload.", score)
        if "header_from_mismatch" in names and "entity_impersonation" in names:
            return ("PHISHING", "FastScan: Multi-layered header domain mismatch with target impersonation profile.", score)
        if "entity_impersonation" in names and "ip_url" in names:
            return ("PHISHING", "FastScan: Identity spoofing combined with unresolvable raw IP layout link.", score)
        # Bổ sung rule: cybersquatting + impersonation
        if "entity_impersonation" in names and "cybersquatting" in names:
            return ("PHISHING", "FastScan: Brand impersonation with cybersquatting domain.", score)

        if score >= 0.50:
            top_triggers = [s.title for s in signals[:3]]
            return ("SUSPICIOUS", "FastScan: Multiple anomalous vectors aggregated: " + ", ".join(top_triggers), score)

        return ("UNKNOWN", "FastScan: Baseline evidence collected. Deferred to downstream deep classifier.", score)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def fastscan_analyze(
        self,
        sender_email: str,
        body_text: str,
        headers: Optional[str] = None,
        urls: Optional[List[str]] = None,
        message_type: str = "email",
        subject: str = "",
    ) -> FastScanResult:
        sender_email = self._clean_text(sender_email)
        body_text = self._clean_text(body_text)
        subject = self._clean_text(subject)
        message_type = str(message_type or "email").lower().strip()
        signals: List[FastScanSignal] = []

        if not sender_email and not body_text:
            return FastScanResult(verdict="UNKNOWN", reason="Empty content payload context.", score=0.0)

        sender_domain = self.extract_domain(sender_email)

        if urls is None:
            normalized_urls, raw_urls = self.extract_urls(f"{subject}\n{body_text}")
        else:
            normalized_urls = [self._normalize_url_like(u) for u in urls if str(u).strip()]
            raw_urls = [u for u in urls if str(u).strip()]   # Giả định raw là chính nó (có thể đã có scheme)

        auth_info = {"spf": "none", "dkim": "none", "dmarc": "none"}
        if message_type == "email":
            try:
                auth_info = self._collect_auth_signals(headers=headers, signals=signals)
            except Exception as e:
                logger.error("Fail safely on auth parsing module: %s", e)

        url_info = {}
        try:
            url_info = self._collect_url_signals(sender_domain=sender_domain, urls=normalized_urls, raw_urls=raw_urls, signals=signals)
        except Exception as e:
            logger.error("Fail safely on URL analysis module: %s", e)

        try:
            self._collect_blacklist_signals(sender_domain=sender_domain, urls=normalized_urls, signals=signals)
        except Exception as e:
            logger.error("Fail safely on memory lookup module: %s", e)

        entity_info = {}
        try:
            entity_info = self._collect_entity_signals(
                sender_email=sender_email, sender_domain=sender_domain,
                subject=subject, body_text=body_text, urls=normalized_urls,
                message_type=message_type, signals=signals
            )
        except Exception as e:
            logger.error("Fail safely on Entity classification module: %s", e)

        content_info = {}
        try:
            content_info = self._collect_content_signals(subject=subject, body_text=body_text, signals=signals)
        except Exception as e:
            logger.error("Fail safely on Content parsing module: %s", e)

        severity_map = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        signals.sort(key=lambda s: (severity_map.get(s.severity, 1), s.confidence), reverse=True)

        verdict, reason, score = self._decide_fastscan_verdict(signals)

        details = {
            "message_type": message_type, "sender": sender_email, "sender_domain": sender_domain,
            "auth": auth_info, "urls": normalized_urls, "url_info": url_info, "entity_info": entity_info,
            "content_info": content_info, "signal_count": len(signals), "signal_names": [s.signal for s in signals],
            "note": "FastScan policy restrains from producing an absolute hard SAFE status verdict."
        }

        return FastScanResult(verdict=verdict, reason=reason, score=score, signals=signals, details=details)


# =============================================================================
# Quick Test
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scanner = FastScan(
        enable_ner=False,
        whitelist={
            "vietcombank": ["vietcombank.com.vn", "vcb.com.vn"],
            "sacombank": ["sacombank.com.vn"],
            "shopee": ["shopee.vn", "shopee.com"],
        },
    )
    scanner.inject_local_blacklist_feed(["vcb-update-banking.info", "malicious-phish-link.top"])

    # Test 1: SMS brandname linh hoạt (Sacombank với sender "SCB")
    sample_sms = scanner.fastscan_analyze(
        sender_email="SCB",
        subject="Thông báo",
        body_text="Sacombank thông báo cập nhật tài khoản. Vào link: http://sacombank.vn/verify",
        message_type="sms"
    )
    print(f"SMS Verdict: {sample_sms.verdict}, Score: {sample_sms.score:.2f}")
    for sig in sample_sms.signals:
        print(f" - {sig.signal}: {sig.title}")

    # Test 2: Email không có scheme rõ ràng, không bị no_https_url
    sample_email = scanner.fastscan_analyze(
        sender_email="notify@shopee.vn",
        subject="Đơn hàng",
        body_text="Truy cập shopee.vn/tracking để xem.",
        message_type="email"
    )
    print(f"\nEmail Verdict: {sample_email.verdict}, Score: {sample_email.score:.2f}")
    for sig in sample_email.signals:
        print(f" - {sig.signal}: {sig.title}")

    # Test 3: Cybersquatting + impersonation (sẽ là PHISHING)
    sample_cyber = scanner.fastscan_analyze(
        sender_email="Vietcombank Support <security@vietcombank-secure.com.vn>",
        subject="Cảnh báo",
        body_text="Vietcombank thông báo tài khoản sẽ bị khóa. Đăng nhập tại https://vietcombank-secure.com.vn/login",
        message_type="email"
    )
    print(f"\nCybersquatting Verdict: {sample_cyber.verdict}, Score: {sample_cyber.score:.2f}")
    for sig in sample_cyber.signals:
        print(f" - {sig.signal}: {sig.title}")
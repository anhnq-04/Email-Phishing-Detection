from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


# =======================
# CONFIG
# =======================

def _load_config_from_yaml() -> dict:
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "configs", "inference.yaml"
    )
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data.get("qwen_vllm", {}) or {}
        except Exception:
            pass
    return {}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


_yaml_cfg = _load_config_from_yaml()

DEFAULT_BASE_URL = os.getenv(
    "QWEN_VLLM_BASE_URL",
    _yaml_cfg.get("base_url") or "http://127.0.0.1:8001/v1",
)
DEFAULT_MODEL_NAME = os.getenv(
    "QWEN_VLLM_MODEL",
    _yaml_cfg.get("model_name") or "phishing",
)
DEFAULT_API_KEY = os.getenv(
    "QWEN_VLLM_API_KEY",
    _yaml_cfg.get("api_key") or "dummy",
)
DEFAULT_MAX_TOKENS = int(
    os.getenv("QWEN_MAX_TOKENS", str(_yaml_cfg.get("max_tokens") or 512))
)
DEFAULT_TIMEOUT = float(
    os.getenv("QWEN_TIMEOUT", str(_yaml_cfg.get("timeout") or 120))
)

# The final SFT dataset was supervised in non-thinking JSON format.
# Keep non-thinking as production default; enable thinking only for controlled A/B tests.
DEFAULT_ENABLE_THINKING = _env_bool(
    "QWEN_ENABLE_THINKING",
    bool(_yaml_cfg.get("enable_thinking", False)),
)
DEFAULT_TEMPERATURE = float(
    os.getenv(
        "QWEN_TEMPERATURE",
        str(_yaml_cfg.get("temperature", 0.6 if DEFAULT_ENABLE_THINKING else 0.0)),
    )
)
DEFAULT_TOP_P = float(
    os.getenv(
        "QWEN_TOP_P",
        str(_yaml_cfg.get("top_p", 0.95 if DEFAULT_ENABLE_THINKING else 1.0)),
    )
)
DEFAULT_TOP_K = int(
    os.getenv("QWEN_TOP_K", str(_yaml_cfg.get("top_k", 20)))
)
DEFAULT_MIN_P = float(
    os.getenv("QWEN_MIN_P", str(_yaml_cfg.get("min_p", 0.05)))
)
DEFAULT_SEED = int(
    os.getenv("QWEN_SEED", str(_yaml_cfg.get("seed", 42)))
)

# Character caps are retained as safe guards for a model trained with 4096-token context.
# Override through environment/config after measuring real request token lengths.
_MAX_BODY_CHARS = int(os.getenv("QWEN_MAX_BODY_CHARS", str(_yaml_cfg.get("max_body_chars") or 4500)))
_MAX_SUBJECT_CHARS = int(os.getenv("QWEN_MAX_SUBJECT_CHARS", str(_yaml_cfg.get("max_subject_chars") or 500)))
_MAX_ATTACHMENTS_CHARS = int(os.getenv("QWEN_MAX_ATTACHMENTS_CHARS", str(_yaml_cfg.get("max_attachments_chars") or 800)))


# =======================
# FINAL TRAIN-TIME TAXONOMY
# =======================

VALID_SIGNALS = {
    "urgency",
    "impersonation",
    "credential_request",
    "financial_lure",
    "threat",
    "call_to_action",
    "suspicious_attachment",
    "otp_or_code",
    "social_engineering",
    "extortion",
}

VALID_SEVERITY = {"low", "medium", "high"}

# Must match SYSTEM_PROMPT_V2_FINAL used to build the final Phase 1/Phase 2 JSONL.
SYSTEM_PROMPT = """You are a cybersecurity expert for defensive phishing detection.

Analyze an English, Vietnamese, or mixed email/SMS message and output ONLY valid JSON.

This is a multi-layer detection system:
- Qwen analyzes visible text, deceptive intent, and social-engineering behavior.
- Other layers handle URL reputation, domain spoofing, sender-domain mismatch, and raw URL risk.
- URLs may be replaced by neutral placeholders such as [LINK_1] and [LINK_2].
- A link placeholder only means that a link exists at that position.
- Do NOT infer URL risk, URL type, domain spoofing, hosting-platform risk, or phishing status from [LINK_N].

Required JSON schema:
{
  "label": "phishing" | "legit",
  "signals": ["..."],
  "evidence": [
    {
      "signal": "...",
      "exact_span": "...",
      "severity": "low" | "medium" | "high"
    }
  ]
}

Allowed signals:
urgency, impersonation, credential_request, financial_lure, threat, call_to_action, suspicious_attachment, otp_or_code, social_engineering, extortion.

The "signals" array and every evidence.signal value MUST use only values from the allowed signals list above. Never output domain_spoofing or any other signal.

Rules:
1. If the message is legitimate, output exactly:
   {"label":"legit","signals":[],"evidence":[]}

2. If the message is phishing, include only signals directly supported by exact visible-text evidence.

3. exact_span must be copied exactly from the input message. Do not translate, paraphrase, normalize, summarize, expand, or invent evidence.

4. Do NOT use a raw URL or a [LINK_N] placeholder as evidence. Cite the visible text that makes the interaction dangerous.

   Incorrect:
   {"signal":"call_to_action","exact_span":"[LINK_1]","severity":"high"}

   Incorrect:
   {"signal":"credential_request","exact_span":"https://example.com/login","severity":"high"}

   Correct:
   {"signal":"credential_request","exact_span":"enter your password to verify your account","severity":"high"}

   Correct:
   {"signal":"threat","exact_span":"tài khoản của bạn sẽ bị khóa nếu không xác minh","severity":"high"}

5. Weak cues alone are NOT enough to classify a message as phishing. The following may be legitimate when they appear alone:
   - a link or registration form
   - a company, school, bank, or service name
   - a deadline, scheduled maintenance notice, or temporary service interruption
   - a free offer, discount, voucher, prize announcement, newsletter, event invitation, or delivery update

   However, these cues MAY support a phishing classification when they occur together with an explicit dangerous request or coercive demand, such as:
   - asking for a password, login credential, OTP, PIN, bank/card data, identity information, recovery code, or private key;
   - asking the user to send or disclose sensitive information;
   - demanding a payment or transfer through deception or pressure;
   - threatening account suspension, loss of access, penalty, exposure, or harm unless the user performs such an action.

6. Classify as phishing only when visible text contains clear deceptive or harmful intent, such as:
   - obtaining credentials or sensitive personal/account/payment information;
   - requesting OTP/security-code disclosure or submission through a suspicious instruction;
   - forcing account verification or sensitive submission by threatening suspension or loss;
   - using fake authority or trusted identity to obtain a sensitive action;
   - using a refund, prize, payment, fee, or invoice lure to obtain money or sensitive data;
   - directing the user to open an unexpected or suspicious attachment;
   - extortion, blackmail, ransom, or threats of exposure/data leakage.

7. OTP distinction:
   - Legit: "Your OTP is 123456. Enter it to complete the transaction. Do not share it with anyone."
   - Legit: "Mã OTP của bạn là 586396. Không chia sẻ mã này cho bất kỳ ai."
   - Phishing: "Reply with your OTP to verify your account."
   - Phishing: "Gửi mã OTP vừa nhận để mở khóa tài khoản."
   - Phishing: "Your account will be locked unless you submit the OTP immediately."

8. If any part of a long message contains clear phishing evidence, classify it as phishing and cite only the dangerous part. Legitimate-looking greetings, signatures, disclaimers, or promotional content do not override explicit phishing intent.

9. Do not output markdown, comments, explanations, additional keys, or text outside the required JSON object.

Signal guide:
- urgency: artificial time pressure that pushes the user to provide sensitive information, make a payment, open an attachment, or perform account verification before checking legitimacy. Ordinary event deadlines or maintenance schedules are not urgency.

- impersonation: claiming or strongly implying trusted authority, such as a bank, platform, school, employer, government office, or support team, in order to obtain credentials, OTP, payment, sensitive information, or another harmful action. A brand or organization name alone is not impersonation evidence.

- credential_request: requesting a password, login credential, card/bank detail, sensitive identity information, recovery code, private key, or sensitive account verification information.

- financial_lure: offering a reward, refund, prize, bonus, unexpected payment, fake invoice, investment return, or fee reversal in order to make the user disclose information, contact a suspicious party, pay money, or follow a deceptive instruction. Ordinary discounts or legitimate free events are not financial_lure.

- threat: threatening account suspension, deactivation, deletion, loss of access, legal penalty, exposure, harm, embarrassment, or financial loss unless the user provides information, pays money, verifies an account, or performs another demanded action. A normal maintenance interruption is not threat.

- call_to_action: instructing the user to perform an unsafe action, such as submit sensitive information, verify or restore an account under pressure, send an OTP, make a payment or transfer, reply with private data, or open an unexpected attachment. Benign instructions such as saving work before maintenance or registering for a normal event are not call_to_action.

- suspicious_attachment: directing the user to open, download, enable, or review an unexpected or risky file, archive, HTML document, executable, macro-enabled file, or suspicious invoice/document.

- otp_or_code: requesting the user to disclose, send, reply with, forward, reveal, or submit an OTP, PIN, 2FA code, recovery code, or security code as part of an account-verification, unlock, payment, or deceptive instruction. A one-way OTP notification warning the user not to share the code is not otp_or_code.

- social_engineering: using fear, fake authority, secrecy, greed, intimidation, emotional pressure, or trust exploitation to obtain sensitive information, money, OTP, account access, or another unsafe action.

- extortion: demanding payment or another action to prevent exposure, harm, embarrassment, release of private material, or leakage of data.

Severity:
- low: a supporting cue that is suspicious only when combined with stronger evidence.
- medium: a clear deceptive instruction or manipulation cue, but not a direct high-impact demand by itself.
- high: a direct request for credentials, OTP/security codes, money transfer/payment, highly sensitive data, opening a dangerous attachment, extortion, or a severe threat used to force such action.

When no strong visible phishing evidence exists, classify the message as legit.
"""


# =======================
# TEXT / URL PREPROCESSING
# =======================

def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _collapse_whitespace(text: Any) -> str:
    text = _safe_str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate_text(text: str, max_chars: int, marker: str = "[TRUNCATED]") -> str:
    text = _safe_str(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0].strip() or text[:max_chars].strip()
    return f"{cut} {marker}"


def prepare_subject(subject: Any) -> str:
    return _truncate_text(_collapse_whitespace(subject), _MAX_SUBJECT_CHARS)


def prepare_body(body: Any) -> str:
    return _truncate_text(_collapse_whitespace(body), _MAX_BODY_CHARS)


def prepare_sender(sender: Any) -> str:
    return _collapse_whitespace(sender)


def prepare_attachments(attachments: Any) -> str:
    if attachments is None:
        return ""
    if isinstance(attachments, (list, tuple, set)):
        text = "\n".join(_safe_str(item) for item in attachments if _safe_str(item).strip())
    else:
        text = _safe_str(attachments)
    return _truncate_text(_collapse_whitespace(text), _MAX_ATTACHMENTS_CHARS)


# Incoming clients may supply raw URLs, bare domains, old typed placeholders,
# or already-neutral placeholders. All link-like inputs seen by Qwen are
# normalized to neutral [LINK_N], matching the final content-only model policy.
_DOMAIN_SUFFIX_PATTERN = (
    r"(?:com\.vn|gov\.vn|edu\.vn|org\.vn|net\.vn|"
    r"com|net|org|io|co|info|me|app|dev|xyz|site|online|"
    r"vn|top|club|biz|cc|vip|shop|work|live|click|support)"
)

RAW_URL_RE = re.compile(
    r"(?<![@\w.-])"
    r"(?:"
        r"(?:https?://|www\.)[^\s<>'\"\)\]]+"
        r"|"
        r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
        + _DOMAIN_SUFFIX_PATTERN +
        r"(?:/[^\s<>'\"\)\]]*)?"
    r")"
    r"(?![\w.-])",
    re.IGNORECASE,
)

OLD_TYPED_PLACEHOLDER_RE = re.compile(
    r"\[(?:FORM_REGISTRATION|SOCIAL_PROFILE|SOCIAL_GROUP|SOCIAL_MESSAGE|"
    r"OFFICIAL_PORTAL|THIRD_PARTY_HOSTED_PAGE|PAYMENT_PORTAL|"
    r"EVENT_REGISTRATION|DELIVERY_TRACKING)_LINK_\d+\]",
    re.IGNORECASE,
)
GENERIC_URL_TOKEN_RE = re.compile(
    r"(?:<\|\[?\[?URL\]?\]?\|>|\[\[?URL\]?\]|<URL>|\[URL\])",
    re.IGNORECASE,
)
NEUTRAL_LINK_RE = re.compile(r"\[LINK_\d+\]", re.IGNORECASE)
ANY_LINK_TOKEN_RE = re.compile(r"\[(?:[A-Z_]*LINK)_\d+\]", re.IGNORECASE)

# Signal-semantic guards: span verification proves only that copied text exists;
# it does not prove that the model assigned the correct signal.
ATTACHMENT_CUE_RE = re.compile(
    r"\b(?:attachment|attached|file|document)\b"
    r"|tệp\s*đính\s*kèm|file\s*đính\s*kèm|đính\s*kèm"
    r"|(?:tệp|file|tài\s*liệu)\s+(?:được\s+)?(?:đính\s*kèm|gửi\s+kèm)"
    r"|(?:open|download|mở|tải(?:\s+xuống)?)\s+(?:the\s+)?(?:attached\s+)?(?:file|document|attachment|tệp|tài\s*liệu)",
    re.IGNORECASE,
)

URGENCY_TIME_CUE_RE = re.compile(
    r"khẩn\s*cấp|khẩn|gấp|ngay\s+lập\s+tức|ngay\s+bây\s+giờ"
    r"|trong\s+vòng\s+\d+\s*(?:phút|giờ|ngày)"
    r"|chỉ\s+còn\s+\d+\s*(?:phút|giờ|ngày)"
    r"|hạn\s+chót|trước\s+ngày|kể\s+từ\s+ngày|hôm\s+nay"
    r"|urgent|immediately|right\s+now"
    r"|within\s+\d+\s*(?:minutes?|hours?|days?)"
    r"|deadline|expires?\s+today",
    re.IGNORECASE,
)


def mask_message_fields_for_llm(
    subject: str = "",
    sender: str = "",
    body: str = "",
    attachments: str = "",
) -> Tuple[Dict[str, str], List[str], List[Dict[str, str]]]:
    """
    Apply the exact final-data URL policy: every URL/old placeholder -> [LINK_N].
    One counter is shared in the same order as the final training schema:
    subject -> sender -> body -> attachment.
    """
    counter = 0
    extracted_urls: List[str] = []
    metadata: List[Dict[str, str]] = []

    combined = re.compile(
        OLD_TYPED_PLACEHOLDER_RE.pattern
        + r"|\[LINK_\d+\]"
        + r"|" + GENERIC_URL_TOKEN_RE.pattern
        + r"|" + RAW_URL_RE.pattern,
        re.IGNORECASE,
    )

    def mask_text(field_name: str, text: str) -> str:
        nonlocal counter
        if not text:
            return ""

        def replacement(match: re.Match) -> str:
            nonlocal counter
            raw_match = match.group(0)
            suffix = ""

            if RAW_URL_RE.fullmatch(raw_match):
                url = raw_match.rstrip(".,;:!?)>\"'")
                suffix = raw_match[len(url):]
                extracted_urls.append(url)
            else:
                url = ""

            counter += 1
            placeholder = f"[LINK_{counter}]"
            metadata.append({
                "field": field_name,
                "url": url,
                "kind": "LINK",
                "placeholder": placeholder,
            })
            return placeholder + suffix

        return combined.sub(replacement, text)

    fields = {
        "subject": mask_text("subject", subject),
        "sender": mask_text("sender", sender),
        "body": mask_text("body", body),
        "attachment": mask_text("attachment", attachments),
    }
    return fields, extracted_urls, metadata


def mask_urls_for_llm(text: str) -> Tuple[str, List[str], List[Dict[str, str]]]:
    """Compatibility wrapper: neutral-mask a standalone body string."""
    fields, urls, metadata = mask_message_fields_for_llm(body=text)
    return fields["body"], urls, metadata


# =======================
# DATA CLASSES
# =======================

@dataclass
class QwenEvidence:
    signal: str
    exact_span: str
    severity: str = "medium"
    field: str = "unknown"
    char_start: int = -1
    char_end: int = -1
    verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QwenResult:
    label: str
    risk_score: float
    signals: List[str] = field(default_factory=list)
    evidence: List[QwenEvidence] = field(default_factory=list)
    reason: str = ""
    raw_output: str = ""
    parsed_output: Dict[str, Any] = field(default_factory=dict)
    user_prompt: str = ""
    error: Optional[str] = None
    link_metadata: List[Dict[str, str]] = field(default_factory=list)

    @property
    def phishing_score(self) -> float:
        return self.risk_score

    def evidence_dicts(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.evidence]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "risk_score": self.risk_score,
            "signals": self.signals,
            "evidence": self.evidence_dicts(),
            "reason": self.reason,
            "raw_output": self.raw_output,
            "parsed_output": self.parsed_output,
            "user_prompt": self.user_prompt,
            "error": self.error,
            "link_metadata": self.link_metadata,
        }


# =======================
# CLIENT
# =======================

class QwenClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        model_name: str = DEFAULT_MODEL_NAME,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT,
        enable_thinking: bool = DEFAULT_ENABLE_THINKING,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        min_p: float = DEFAULT_MIN_P,
        seed: Optional[int] = DEFAULT_SEED,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.seed = seed

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    # ─────────────────────────────────────────
    # Prompt format — must match final SFT input
    # ─────────────────────────────────────────

    @staticmethod
    def build_user_prompt(
        subject: str = "",
        body: str = "",
        sender: str = "",
        attachments: Any = "",
        prefix: str = "EMAIL",
        language: str = "VI",
    ) -> str:
        """
        Final unified schema:
          EMAIL -> Subject (if present), Sender (if present), Body
          SMS   -> Sender (if present), Body

        The language argument is kept for API compatibility only.
        SMS content is never emitted as "Message:".
        """
        is_sms = _safe_str(prefix).upper().strip() == "SMS"

        clean_subject = prepare_subject(subject)
        clean_sender = prepare_sender(sender)
        clean_body = prepare_body(body)
        clean_attachments = prepare_attachments(attachments)

        parts: List[str] = []
        if clean_subject and not is_sms:
            parts.append(f"Subject: {clean_subject}")
        if clean_sender:
            parts.append(f"Sender: {clean_sender}")
        if clean_body:
            parts.append(f"Body: {clean_body}")
        if clean_attachments:
            parts.append(f"Attachment: {clean_attachments}")

        return "\n".join(parts).strip()

    @staticmethod
    def build_search_fields(
        subject: str = "",
        body: str = "",
        sender: str = "",
        attachments: Any = "",
        prefix: str = "EMAIL",
        language: str = "VI",
    ) -> Dict[str, str]:
        """Build fields exactly as Qwen sees them for evidence verification."""
        is_sms = _safe_str(prefix).upper().strip() == "SMS"

        fields: Dict[str, str] = {}
        clean_subject = prepare_subject(subject)
        clean_sender = prepare_sender(sender)
        clean_body = prepare_body(body)
        clean_attachments = prepare_attachments(attachments)

        if clean_subject and not is_sms:
            fields["subject"] = clean_subject
        if clean_sender:
            fields["sender"] = clean_sender
        if clean_body:
            fields["body"] = clean_body
        if clean_attachments:
            fields["attachment"] = clean_attachments

        return fields

    # ─────────────────────────────────────────
    # JSON parsing
    # ─────────────────────────────────────────

    @staticmethod
    def remove_think_blocks(text: str) -> str:
        text = _safe_str(text)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()

    @staticmethod
    def extract_first_balanced_json_object(text: str) -> str:
        text = _safe_str(text)
        start = text.find("{")
        if start < 0:
            raise ValueError("No JSON object start found")

        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        raise ValueError("No balanced JSON object found")

    @classmethod
    def extract_json(cls, raw_text: str) -> Dict[str, Any]:
        text = cls.remove_think_blocks(raw_text)
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = json.loads(cls.extract_first_balanced_json_object(text))
        return parsed if isinstance(parsed, dict) else {}

    # ─────────────────────────────────────────
    # Label / evidence normalization
    # ─────────────────────────────────────────

    @staticmethod
    def normalize_label(label: Any) -> str:
        value = _safe_str(label).strip().lower()
        if value in {"phishing", "phish", "malicious", "spam", "scam"}:
            return "phishing"
        if value in {"legit", "legitimate", "safe", "ham", "benign"}:
            return "legit"
        return "unknown"

    @staticmethod
    def normalize_signal(signal: Any) -> str:
        value = _safe_str(signal).strip().lower()
        return value if value in VALID_SIGNALS else ""

    @staticmethod
    def normalize_severity(severity: Any) -> str:
        value = _safe_str(severity).strip().lower()
        return value if value in VALID_SEVERITY else "medium"

    @staticmethod
    def whitespace_flexible_pattern(span: str) -> str:
        pieces = re.split(r"\s+", span.strip())
        return r"\s+".join(re.escape(piece) for piece in pieces if piece)

    @classmethod
    def find_span(cls, span: str, text: str) -> Optional[Tuple[int, int]]:
        span = _safe_str(span)
        text = _safe_str(text)
        if not span or not text:
            return None

        position = text.lower().find(span.lower())
        if position >= 0:
            return position, position + len(span)

        try:
            pattern = cls.whitespace_flexible_pattern(span)
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            return (match.start(), match.end()) if match else None
        except re.error:
            return None

    @classmethod
    def normalize_evidence(
        cls,
        raw_evidence: Any,
        search_fields: Dict[str, str],
        original_fields: Optional[Dict[str, str]] = None,
    ) -> List[QwenEvidence]:
        """
        Keep verified, content-level evidence only.
        Verification occurs against the neutral-masked text seen by Qwen.
        Offsets are mapped back to original request fields when possible.
        """
        if not isinstance(raw_evidence, list):
            return []

        original_fields = original_fields or search_fields
        verified_items: List[QwenEvidence] = []
        seen = set()

        for item in raw_evidence:
            if not isinstance(item, dict):
                continue

            signal = cls.normalize_signal(item.get("signal"))
            exact_span = _safe_str(item.get("exact_span")).strip()
            severity = cls.normalize_severity(item.get("severity", "medium"))

            if not signal or not exact_span:
                continue

            # The final model was not trained to expose link/domain evidence.
            if RAW_URL_RE.search(exact_span) or ANY_LINK_TOKEN_RE.search(exact_span):
                continue

            key = (signal, exact_span.lower())
            if key in seen:
                continue

            for field_name, masked_text in search_fields.items():
                masked_loc = cls.find_span(exact_span, masked_text)
                if not masked_loc:
                    continue

                raw_text = original_fields.get(field_name, "")
                raw_loc = cls.find_span(exact_span, raw_text)
                if raw_loc:
                    start, end = raw_loc
                    source_span = raw_text[start:end]
                else:
                    start, end = masked_loc
                    source_span = masked_text[start:end]

                verified_items.append(
                    QwenEvidence(
                        signal=signal,
                        exact_span=source_span,
                        severity=severity,
                        field=field_name,
                        char_start=start,
                        char_end=end,
                        verified=True,
                    )
                )
                seen.add(key)
                break

        return verified_items

    @staticmethod
    def derive_signals_from_verified_evidence(evidence: List[QwenEvidence]) -> List[str]:
        return sorted({item.signal for item in evidence if item.verified and item.signal in VALID_SIGNALS})

    @classmethod
    def sanitize_evidence_semantics(
        cls,
        evidence: List[QwenEvidence],
        user_prompt: str,
        original_fields: Dict[str, str],
    ) -> List[QwenEvidence]:
        """
        Remove impossible signal assignments without inventing replacement evidence.

        A verified exact_span only proves grounding. For example, a phrase
        containing "click vào link" can exist in the input but cannot support
        suspicious_attachment unless the message also mentions a file/document
        or an actual attachment field is present.
        """
        has_attachment_field = bool(_safe_str(original_fields.get("attachment", "")).strip())
        cleaned: List[QwenEvidence] = []

        for item in evidence:
            if item.signal == "suspicious_attachment":
                if not has_attachment_field and not ATTACHMENT_CUE_RE.search(user_prompt):
                    continue

            if item.signal == "urgency":
                if not URGENCY_TIME_CUE_RE.search(item.exact_span):
                    continue

            cleaned.append(item)

        return cleaned

    # ─────────────────────────────────────────
    # Risk / reason — confidence is heuristic, not an LLM probability
    # ─────────────────────────────────────────

    @staticmethod
    def estimate_risk_score(label: str, evidence: List[QwenEvidence]) -> float:
        weights = {
            "credential_request": 0.22,
            "extortion": 0.22,
            "otp_or_code": 0.20,
            "threat": 0.18,
            "social_engineering": 0.14,
            "suspicious_attachment": 0.15,
            "financial_lure": 0.12,
            "impersonation": 0.10,
            "urgency": 0.08,
            "call_to_action": 0.07,
        }
        severity_mult = {"low": 0.50, "medium": 0.75, "high": 1.00}
        signals = {item.signal for item in evidence if item.verified}

        if label == "legit":
            return 0.05
        if label != "phishing" or not evidence:
            return 0.50

        score = 0.50 + sum(
            weights.get(item.signal, 0.05) * severity_mult.get(item.severity, 0.75)
            for item in evidence
            if item.verified
        )
        for combination in [
            {"credential_request", "call_to_action"},
            {"credential_request", "impersonation"},
            {"threat", "call_to_action"},
            {"extortion", "threat"},
            {"otp_or_code", "call_to_action"},
        ]:
            if combination.issubset(signals):
                score += 0.08
        return round(min(score, 0.98), 4)

    @staticmethod
    def fallback_reason(label: str, evidence: List[QwenEvidence]) -> str:
        descriptions = {
            "urgency": "tạo áp lực thời gian",
            "impersonation": "có dấu hiệu mạo danh",
            "credential_request": "yêu cầu thông tin nhạy cảm hoặc thông tin tài khoản",
            "financial_lure": "dùng yếu tố tài chính làm mồi nhử",
            "threat": "đe dọa hậu quả tiêu cực",
            "call_to_action": "kêu gọi hành động rủi ro",
            "suspicious_attachment": "tệp đính kèm có ngữ cảnh rủi ro",
            "otp_or_code": "yêu cầu mã OTP hoặc mã xác thực",
            "social_engineering": "thao túng tâm lý",
            "extortion": "tống tiền hoặc đe dọa phát tán",
        }
        if label == "phishing" and evidence:
            parts = []
            for item in evidence[:3]:
                span = item.exact_span if len(item.exact_span) <= 120 else item.exact_span[:117] + "..."
                parts.append(f"{descriptions.get(item.signal, item.signal)}: “{span}”")
            return "Tin nhắn bị đánh giá là phishing vì " + "; ".join(parts) + "."
        if label == "legit":
            return "Không phát hiện evidence phishing rõ ràng."
        return "LLM trả về phishing nhưng không có evidence content-level xác minh được; cần review."

    # ─────────────────────────────────────────
    # API calls
    # ─────────────────────────────────────────

    def health(self) -> bool:
        try:
            return bool(self.client.models.list().data)
        except Exception:
            return False

    def raw_chat(
        self,
        user_prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        max_tokens: Optional[int] = None,
    ) -> str:
        extra_body: Dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
        }
        if self.temperature > 0:
            extra_body["top_k"] = self.top_k
            extra_body["min_p"] = self.min_p

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_tokens or self.max_tokens,
            seed=self.seed,
            extra_body=extra_body,
        )
        return response.choices[0].message.content or ""

    def predict(
        self,
        subject: str = "",
        body: str = "",
        language: str = "VI",
        prefix: str = "EMAIL",
        sender: str = "",
        urls: Any = None,
        attachments: Any = "",
    ) -> QwenResult:
        # Prepare fields once; neutral masking must be shared across all visible fields.
        prepared_original_fields = self.build_search_fields(
            subject=subject,
            body=body,
            sender=sender,
            attachments=attachments,
            prefix=prefix,
            language=language,
        )

        masked_fields, _extracted_urls, link_metadata = mask_message_fields_for_llm(
            subject=prepared_original_fields.get("subject", ""),
            sender=prepared_original_fields.get("sender", ""),
            body=prepared_original_fields.get("body", ""),
            attachments=prepared_original_fields.get("attachment", ""),
        )

        user_prompt = self.build_user_prompt(
            subject=masked_fields.get("subject", ""),
            body=masked_fields.get("body", ""),
            sender=masked_fields.get("sender", ""),
            attachments=masked_fields.get("attachment", ""),
            prefix=prefix,
            language=language,
        )

        search_fields = self.build_search_fields(
            subject=masked_fields.get("subject", ""),
            body=masked_fields.get("body", ""),
            sender=masked_fields.get("sender", ""),
            attachments=masked_fields.get("attachment", ""),
            prefix=prefix,
            language=language,
        )

        try:
            raw_output = self.raw_chat(user_prompt=user_prompt)
            parsed = self.extract_json(raw_output)
            label = self.normalize_label(parsed.get("label"))

            raw_evidence = parsed.get("evidence", [])
            print("=" * 80)
            print("[DEBUG] RAW QWEN OUTPUT:")
            print(raw_output)
            print("[DEBUG] RAW EVIDENCE FROM QWEN:")
            print(json.dumps(raw_evidence, ensure_ascii=False, indent=2))
            print("=" * 80)

            evidence = self.normalize_evidence(
                raw_evidence=raw_evidence,
                search_fields=search_fields,
                original_fields=prepared_original_fields,
            )
            evidence = self.sanitize_evidence_semantics(
                evidence=evidence,
                user_prompt=user_prompt,
                original_fields=prepared_original_fields,
            )
            print("[DEBUG] VERIFIED & SANITIZED EVIDENCE:")
            for idx, ev in enumerate(evidence, 1):
                print(f"  #{idx} Signal: {ev.signal} | Span: {ev.exact_span} | Verified: {ev.verified}")
            print("=" * 80)
            signals = self.derive_signals_from_verified_evidence(evidence)

            if label == "legit":
                evidence = []
                signals = []
            elif label == "phishing" and not evidence:
                # Enforce the final training contract: phishing requires grounded evidence.
                label = "unknown"
                signals = []

            risk_score = self.estimate_risk_score(label, evidence)
            reason = self.fallback_reason(label, evidence)

            return QwenResult(
                label=label,
                risk_score=risk_score,
                signals=signals,
                evidence=evidence,
                reason=reason,
                raw_output=raw_output,
                parsed_output=parsed,
                user_prompt=user_prompt,
                error=None,
                link_metadata=link_metadata,
            )
        except Exception as error:
            return QwenResult(
                label="unknown",
                risk_score=0.50,
                signals=[],
                evidence=[],
                reason="Qwen vLLM inference failed.",
                raw_output="",
                parsed_output={},
                user_prompt=user_prompt,
                error=str(error),
                link_metadata=link_metadata,
            )

    # Compatibility aliases
    def predict_email(
        self,
        subject: str = "",
        body: str = "",
        language: str = "VI",
        sender: str = "",
        urls: Any = None,
        attachments: Any = "",
    ) -> QwenResult:
        return self.predict(
            subject=subject,
            body=body,
            language=language,
            prefix="EMAIL",
            sender=sender,
            urls=urls,
            attachments=attachments,
        )

    def predict_sms(
        self,
        body: str = "",
        language: str = "VI",
        sender: str = "",
        urls: Any = None,
    ) -> QwenResult:
        return self.predict(
            subject="",
            body=body,
            language=language,
            prefix="SMS",
            sender=sender,
            urls=urls,
            attachments="",
        )

"""
main.py
=======
FastAPI application — Phishing Detection REST API.

Architecture:
  - Layer 1 (FastScan)  : Local — rule-based, loaded in-process
  - Layer 2 (CatBoost)  : Local — ML models, loaded in-process
  - Layer 3 (Qwen LLM)  : Remote — calls vLLM server via OpenAI-compatible API
  - Layer 4 (Fusion)    : Local — weighted aggregation
  - Layer 5 (XAI)       : Local — explainable AI report (optional)
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BatchUrlAnalysisRequest,
    EmailAnalysisRequest,
    FastScanResponse,
    FusionResponse,
    HealthResponse,
    SmsAnalysisRequest,
    UrlAnalysisRequest,
    UrlResult,
    XAIReportResponse,
)

# ── Logging ─────────────────────────────────────────────
logger = logging.getLogger("phishing_api")
logging.basicConfig(level=logging.INFO)

# ── Version ─────────────────────────────────────────────
API_VERSION = "2.1.0"

# ── Timeout (giây) ──────────────────────────────────────
TIMEOUT_FASTSCAN = 5
TIMEOUT_CATBOOST_PER_URL = 30
TIMEOUT_QWEN = 120          # vLLM server call — network latency included
TIMEOUT_FULL_PIPELINE = 180
TIMEOUT_XAI = 15

# ── Thread pool cho blocking inference ──────────────────
_executor = ThreadPoolExecutor(max_workers=4)


# ════════════════════════════════════════════════════════
# URL PREPROCESSING HELPER
# ════════════════════════════════════════════════════════

def _preprocess_urls(urls: list[str]) -> list[str]:
    """
    Preprocess extracted URLs before sending to CatBoost inference.

    Steps:
        1. Filter out malformed strings / non-URLs / placeholders
        2. Normalize (strip trailing punctuation artifacts)
        3. Deduplicate (preserving order)
    """
    import re

    # Minimum viable URL: must have at least a dot and 4+ chars
    _MIN_URL_LEN = 10
    _URL_LIKE = re.compile(
        r'^(https?://|www\.)',
        re.IGNORECASE,
    )
    # Placeholder / broken patterns to reject
    _REJECT_PATTERNS = re.compile(
        r'(\[URL\]|<URL>|example\.com|placeholder|localhost|127\.0\.0\.\d)',
        re.IGNORECASE,
    )

    seen: set[str] = set()
    clean: list[str] = []

    for url in urls:
        url = url.strip().rstrip(".,;:!?)>\"'")

        # Skip too short
        if len(url) < _MIN_URL_LEN:
            continue

        # Must look like a URL
        if not _URL_LIKE.match(url):
            continue

        # Skip placeholders / malformed
        if _REJECT_PATTERNS.search(url):
            continue

        # Deduplicate
        key = url.lower()
        if key not in seen:
            seen.add(key)
            clean.append(url)

    return clean


def _merge_extracted_urls(provided_urls: Optional[list[str]], text: str) -> list[str]:
    """
    Merge URLs supplied by the frontend with URLs extracted from message text.

    Important:
    - request.urls=None  -> extract from body/content
    - request.urls=[]    -> still extract from body/content
    - request.urls=[...] -> use both supplied URLs and extracted URLs, then dedupe

    This prevents losing URL analysis when the frontend sends an empty list.
    """
    provided = provided_urls or []

    # extract_urls is an instance method → use the loaded singleton
    extracted: list[str] = []
    if _fastscan_engine is not None:
        # extract_urls returns (normalized_urls, raw_urls); we use raw_urls
        _normalized, raw = _fastscan_engine.extract_urls(text or "")
        extracted = raw

    return _preprocess_urls([*provided, *extracted])


# ════════════════════════════════════════════════════════
# LAYER SINGLETONS (lazy load in lifespan)
# ════════════════════════════════════════════════════════

_fastscan_engine = None     # Local — rule-based
_catboost_engine = None     # Local — ML models
_qwen_client = None         # Remote — QwenClient → vLLM server

_layer_status = {
    "fastscan": "unavailable",
    "catboost": "unavailable",
    "qwen": "unavailable",
}


def _load_fastscan():
    global _fastscan_engine
    try:
        from src.inference.fastscan_inference import FastScan
        _fastscan_engine = FastScan()
        _layer_status["fastscan"] = "loaded"
        logger.info("Layer 1 (FastScan) loaded — local.")
    except Exception as e:
        logger.warning("Layer 1 (FastScan) unavailable: %s", e)


def _load_catboost():
    global _catboost_engine
    try:
        from src.inference.catboost_inference import DualCatBoostInference
        _catboost_engine = DualCatBoostInference()
        _layer_status["catboost"] = "loaded"
        logger.info("Layer 2 (CatBoost) loaded — local.")
    except Exception as e:
        logger.warning("Layer 2 (CatBoost) unavailable: %s", e)


def _load_qwen():
    """
    Load Qwen as a REMOTE client connecting to a vLLM server.

    Configuration via environment variables:
      QWEN_VLLM_BASE_URL  (default: http://127.0.0.1:8001/v1)
      QWEN_VLLM_MODEL     (default: phishing)
      QWEN_VLLM_API_KEY   (default: dummy)
      QWEN_MAX_TOKENS      (default: 512)
      QWEN_TIMEOUT          (default: 120)
    """
    global _qwen_client
    try:
        from src.inference.qwen_client import QwenClient
        _qwen_client = QwenClient()
        logger.info(
            "Qwen inference config: thinking=%s, temperature=%s, top_p=%s, top_k=%s, min_p=%s, seed=%s",
            _qwen_client.enable_thinking,
            _qwen_client.temperature,
            _qwen_client.top_p,
            _qwen_client.top_k,
            _qwen_client.min_p,
            _qwen_client.seed,
        )

        # Probe health of the vLLM server (optional, non-blocking)
        try:
            if _qwen_client.health():
                _layer_status["qwen"] = "connected"
                logger.info(
                    "Layer 3 (Qwen LLM) connected — vLLM server at %s",
                    _qwen_client.base_url,
                )
            else:
                _layer_status["qwen"] = "client_ready"
                logger.warning(
                    "Layer 3 (Qwen LLM) client ready but vLLM server "
                    "not responding at %s — will retry on first request.",
                    _qwen_client.base_url,
                )
        except Exception:
            _layer_status["qwen"] = "client_ready"
            logger.warning(
                "Layer 3 (Qwen LLM) client created but health check "
                "failed — vLLM server may not be running yet."
            )
    except Exception as e:
        logger.warning("Layer 3 (Qwen LLM) unavailable: %s", e)


# ════════════════════════════════════════════════════════
# TIMEOUT HELPER
# ════════════════════════════════════════════════════════

async def _run_with_timeout(func, *args, timeout: float, label: str):
    """
    Chạy một hàm blocking trong thread pool với timeout.

    Parameters
    ----------
    func : callable
        Hàm blocking cần chạy (inference layer).
    timeout : float
        Timeout tính bằng giây.
    label : str
        Tên layer (dùng để log).

    Returns
    -------
    result hoặc None nếu timeout / lỗi.
    """
    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, partial(func, *args)),
            timeout=timeout,
        )
        return result
    except asyncio.TimeoutError:
        logger.error("%s timed out after %.1fs", label, timeout)
        return None
    except Exception as e:
        logger.error("%s error: %s", label, e)
        return None


# ════════════════════════════════════════════════════════
# LIFESPAN — khởi tạo / giải phóng tài nguyên
# ════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load local models + init Qwen client  ──  Shutdown: cleanup."""
    logger.info("Starting Phishing Detection API v%s …", API_VERSION)

    # Local models
    _load_fastscan()
    _load_catboost()

    # Remote: Qwen vLLM client
    _load_qwen()

    yield

    logger.info("Shutting down Phishing Detection API.")
    _executor.shutdown(wait=False)


# ════════════════════════════════════════════════════════
# APP INSTANCE
# ════════════════════════════════════════════════════════

app = FastAPI(
    title="Phishing Detection API",
    description=(
        "Multi-layer phishing detection system.\n\n"
        "**Layers:**\n"
        "1. FastScan — Rule-based fast scan (local)\n"
        "2. CatBoost — URL phishing ML model (local)\n"
        "3. Qwen LLM — Semantic phishing analysis (remote vLLM server)\n"
        "4. Fusion Policy — Weighted aggregation\n"
        "5. XAI — Explainable AI evidence report\n"
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════
# HEALTH CHECK
# ════════════════════════════════════════════════════════

@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Kiểm tra trạng thái hệ thống",
)
async def get_health():
    """
    Trả về trạng thái hiện tại của API và các detection layers.

    Qwen status values:
      - "connected"    : vLLM server is up and responding
      - "client_ready" : client initialized but vLLM server not yet confirmed
      - "unavailable"  : client failed to initialize
    """
    # Re-check Qwen connectivity if client exists but was not connected
    if _qwen_client is not None and _layer_status["qwen"] == "client_ready":
        try:
            if _qwen_client.health():
                _layer_status["qwen"] = "connected"
        except Exception:
            pass

    all_operational = (
        _layer_status["fastscan"] == "loaded"
        and _layer_status["catboost"] == "loaded"
        and _layer_status["qwen"] == "connected"
    )

    return HealthResponse(
        status="ok" if all_operational else "degraded",
        version=API_VERSION,
        layers=_layer_status.copy(),
    )


# ════════════════════════════════════════════════════════
# QWEN vLLM INFERENCE HELPER
# ════════════════════════════════════════════════════════

def _qwen_predict(
    subject: str = "",
    body: str = "",
    language: str = "VI",
    prefix: str = "EMAIL",
    sender: str = "",
    urls: Any = "",
    attachments: Any = "",
):
    """
    Call Qwen LLM via QwenClient (HTTP → vLLM server).

    URLs are not appended as a separate prompt field. QwenClient performs
    final-model neutral URL masking inside predict(), so Qwen sees only
    placeholders such as [LINK_1] instead of raw URLs. URL/domain risk remains
    the responsibility of FastScan/CatBoost.
    """
    if _qwen_client is None:
        return None

    return _qwen_client.predict(
        subject=subject,
        body=body,
        language=language,
        prefix=prefix,
        sender=sender,
        urls=urls,
        attachments=attachments,
    )


# ════════════════════════════════════════════════════════
# XAI HELPER
# ════════════════════════════════════════════════════════

async def _build_xai_report(
    fusion_result,
    body: str = "",
    subject: str = "",
    sender_email: str = "",
    message_type: str = "email",
    url_results: Optional[List[Any]] = None,
):
    """
    Run XAI engine in thread pool and return XAIReportResponse (or None on error).
    """
    try:
        from src.xai.xai_engine import explain

        def _run():
            report = explain(
                fusion_result,
                body=body,
                subject=subject,
                sender_email=sender_email,
                message_type=message_type,
                url_results=url_results,
            )
            return report.to_dict()

        result = await _run_with_timeout(
            _run,
            timeout=TIMEOUT_XAI,
            label="XAI",
        )
        if result is not None:
            return XAIReportResponse(**result)
    except Exception as e:
        logger.warning("XAI report failed: %s", e)

    return None


# ════════════════════════════════════════════════════════
# EMAILS  — Phân tích email phishing
# ════════════════════════════════════════════════════════

@app.post(
    "/api/v1/emails/analyze",
    response_model=FusionResponse,
    tags=["Emails"],
    summary="Phân tích email phishing (full pipeline)",
    status_code=status.HTTP_200_OK,
)
async def analyze_email(request: EmailAnalysisRequest):
    """
    Chạy full pipeline phân tích email phishing qua tất cả layers.

    Design note:
    - Raw URLs are extracted/merged for FastScan + CatBoost.
    - QwenClient masks any visible URL into neutral [LINK_N] placeholders.
    - Qwen analyzes visible content only; it does not emit domain_spoofing.
    """
    from src.inference.fusion_policy import fuse_results

    # ── URL extraction first: shared by FastScan / CatBoost / XAI ─────
    # Include Subject as a URL-bearing field; QwenClient also masks Subject/Body
    # using the same final-training policy.
    email_url_text = "\n".join(part for part in [request.subject, request.body] if part)
    clean_urls = _merge_extracted_urls(request.urls, email_url_text)

    logger.info(
        "Email URL extraction/merge: provided=%d -> clean=%d after preprocessing: %s",
        len(request.urls or []),
        len(clean_urls),
        clean_urls[:10],
    )

    # ── Layer 1: FastScan (local) ──────────────────────
    fastscan_result = None
    if _fastscan_engine is not None:
        fastscan_result = await _run_with_timeout(
            _fastscan_engine.fastscan_analyze,
            request.sender_email,
            request.body,
            request.headers,
            clean_urls,
            "email",
            timeout=TIMEOUT_FASTSCAN,
            label="FastScan",
        )

    # ── Layer 2: CatBoost — URL detection (local) ──────
    url_results = None

    if _catboost_engine is not None and clean_urls:
        timeout_batch = TIMEOUT_CATBOOST_PER_URL * len(clean_urls)
        url_results = await _run_with_timeout(
            _catboost_engine.predict_batch,
            clean_urls,
            timeout=timeout_batch,
            label="CatBoost",
        )

    # ── Layer 3: Qwen LLM (remote — vLLM server) ──────
    # QwenClient applies the final SFT input contract:
    #   Subject/Sender/Body with every raw URL -> neutral [LINK_N].
    # Qwen no longer handles domain_spoofing; raw URL layers do.
    qwen_result = None
    if _qwen_client is not None:
        qwen_result = await _run_with_timeout(
            _qwen_predict,
            request.subject,        # subject
            request.body,           # raw body; neutral URL masking happens in QwenClient
            request.language,       # language
            "EMAIL",                # prefix
            request.sender_email,   # sender
            None,                   # do not append URLs as a separate prompt field
            timeout=TIMEOUT_QWEN,
            label="Qwen LLM (vLLM)",
        )

    # ── Layer 4: Fusion ─────────────────────────────────
    fusion = fuse_results(
        fastscan_result=fastscan_result,
        url_results=url_results,
        qwen_result=qwen_result,
        message_type="email",
    )

    # ── Layer 5: XAI Explanation ──────────────────────────
    xai_report = await _build_xai_report(
        fusion,
        body=request.body,
        subject=request.subject,
        sender_email=request.sender_email,
        message_type="email",
        url_results=url_results,
    )

    return FusionResponse(
        message_type="email",
        **fusion.to_dict(),
        xai_report=xai_report,
    )


@app.post(
    "/api/v1/emails/fastscan",
    response_model=FastScanResponse,
    tags=["Emails"],
    summary="Quét nhanh email (Layer 1 only)",
    status_code=status.HTTP_200_OK,
)
async def fastscan_email(request: EmailAnalysisRequest):
    """
    Chỉ chạy Layer 1 (FastScan) — quét nhanh bằng luật cứng.

    Kết quả:
    - **PHISHING**: Auth fail / Blacklist / Mạo danh / Homograph
    - **SAFE**: Khớp whitelist tuyệt đối
    - **UNKNOWN**: Chuyển xuống tầng phân tích sâu hơn
    """
    if _fastscan_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FastScan layer is not available.",
        )

    result = await _run_with_timeout(
        _fastscan_engine.fastscan_analyze,
        request.sender_email,
        request.body,
        request.headers,
        None,
        "email",
        timeout=TIMEOUT_FASTSCAN,
        label="FastScan",
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="FastScan analysis timed out or failed.",
        )

    return FastScanResponse(
        verdict=result.verdict,
        reason=result.reason,
        details=result.details,
    )


# ════════════════════════════════════════════════════════
# SMS  — Phân tích SMS phishing
# ════════════════════════════════════════════════════════

@app.post(
    "/api/v1/sms/analyze",
    response_model=FusionResponse,
    tags=["SMS"],
    summary="Phân tích SMS phishing (full pipeline)",
    status_code=status.HTTP_200_OK,
)
async def analyze_sms(request: SmsAnalysisRequest):
    """
    Chạy full pipeline phân tích SMS phishing.

    Design note:
    - Raw URLs are extracted/merged for CatBoost.
    - QwenClient formats SMS as Sender + Body, matching final training data.
    - QwenClient converts raw URLs to neutral [LINK_N] placeholders.
    """
    from src.inference.fusion_policy import fuse_results

    # ── URL extraction first: shared by CatBoost / XAI ─────
    clean_urls = _merge_extracted_urls(request.urls, request.content)

    logger.info(
        "SMS URL extraction/merge: provided=%d -> clean=%d after preprocessing: %s",
        len(request.urls or []),
        len(clean_urls),
        clean_urls[:10],
    )

    # ── Layer 1: FastScan (local) ──────────────────────
    fastscan_result = None
    if _fastscan_engine is not None:
        fastscan_result = await _run_with_timeout(
            _fastscan_engine.fastscan_analyze,
            request.sender,
            request.content,
            None,
            clean_urls,
            "sms",
            timeout=TIMEOUT_FASTSCAN,
            label="FastScan (SMS)",
        )

    # ── Layer 2: CatBoost — URL detection (local) ──────
    url_results = None

    if _catboost_engine is not None and clean_urls:
        timeout_batch = TIMEOUT_CATBOOST_PER_URL * len(clean_urls)
        url_results = await _run_with_timeout(
            _catboost_engine.predict_batch,
            clean_urls,
            timeout=timeout_batch,
            label="CatBoost (SMS)",
        )

    # ── Layer 3: Qwen LLM (remote — vLLM server) ──────
    qwen_result = None
    if _qwen_client is not None:
        qwen_result = await _run_with_timeout(
            _qwen_predict,
            "",                     # subject
            request.content,        # raw content; Body + neutral URL masking in QwenClient
            request.language,       # language
            "SMS",                  # prefix
            request.sender,         # sender
            None,                   # do not append URLs as a separate prompt field
            timeout=TIMEOUT_QWEN,
            label="Qwen LLM (vLLM)",
        )

    # ── Layer 4: Fusion ─────────────────────────────────
    fusion = fuse_results(
        fastscan_result=fastscan_result,
        url_results=url_results,
        qwen_result=qwen_result,
        message_type="sms",
    )

    # ── Layer 5: XAI Explanation ──────────────────────────
    xai_report = await _build_xai_report(
        fusion,
        body=request.content,
        subject="",
        sender_email="",
        message_type="sms",
        url_results=url_results,
    )

    return FusionResponse(
        message_type="sms",
        **fusion.to_dict(),
        xai_report=xai_report,
    )


@app.post(
    "/api/v1/urls/analyze",
    response_model=UrlResult,
    tags=["URLs"],
    summary="Phân tích một URL phishing",
    status_code=status.HTTP_200_OK,
)
async def analyze_url(request: UrlAnalysisRequest):
    """
    Phân tích URL đơn lẻ bằng CatBoost (lexical + HTML features).

    Trả về phishing score, label, và confidence.
    """
    if _catboost_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CatBoost URL model is not available.",
        )

    result = await _run_with_timeout(
        _catboost_engine.predict_url,
        request.url,
        timeout=TIMEOUT_CATBOOST_PER_URL,
        label="CatBoost URL",
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="URL analysis timed out or failed.",
        )

    return UrlResult(
        url=result.url,
        label=result.label,
        confidence=result.confidence,
        phishing_score=result.phishing_score,
        lexical_score=result.lexical_score,
        html_score=result.html_score,
        fetch_success=result.fetch_success,
        error=result.error,
    )


@app.post(
    "/api/v1/urls/batch-analyze",
    response_model=List[UrlResult],
    tags=["URLs"],
    summary="Phân tích batch URL phishing",
    status_code=status.HTTP_200_OK,
)
async def batch_analyze_urls(request: BatchUrlAnalysisRequest):
    """
    Phân tích nhiều URL cùng lúc (tối đa 50 URL).

    Trả về danh sách kết quả tương ứng với từng URL.
    """
    if _catboost_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CatBoost URL model is not available.",
        )

    timeout_batch = TIMEOUT_CATBOOST_PER_URL * len(request.urls)
    results = await _run_with_timeout(
        _catboost_engine.predict_batch,
        request.urls,
        timeout=timeout_batch,
        label="CatBoost Batch",
    )

    if results is None:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Batch URL analysis timed out or failed.",
        )

    return [
        UrlResult(
            url=r.url,
            label=r.label,
            confidence=r.confidence,
            phishing_score=r.phishing_score,
            lexical_score=r.lexical_score,
            html_score=r.html_score,
            fetch_success=r.fetch_success,
            error=r.error,
        )
        for r in results
    ]

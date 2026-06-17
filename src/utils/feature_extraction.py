"""
feature_extraction.py
=====================
Trích xuất đặc trưng từ URL và HTML cho phishing detection.

Thiết kế:
  - URL lexical features: không crawl, không requests, không phụ thuộc accessible.
  - HTML features: giữ bộ feature HTML cũ, chỉ dùng khi fetch HTML thành công.

Label bên ngoài:
  0 = phishing
  1 = legitimate
"""

import re
import math
import requests
import urllib3
import numpy as np

from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from collections import Counter


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


TIMEOUT = 8

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


FEATURES_URL = [
    # original lexical
    "IsHTTPS",
    "LetterRatioInURL",
    "NoOfSubDomain",
    "DegitRatioInURL",
    "SpacialCharRatioInURL",
    "DomainLength",
]


FEATURES_HTML = [
    "NoOfExternalRef",
    "NoOfSelfRef",
    "LineOfCode",
    "NoOfImage",
    "LargestLineLength",
    "HasDescription",
    "HasSocialNet",
    "NoOfJS",
    "URLTitleMatchScore",
]


SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "verification", "secure", "security",
    "account", "update", "confirm", "validate", "auth", "authentication",
    "wallet", "payment", "billing", "invoice", "recover", "unlock",
    "limited", "suspend", "suspended", "alert", "warning", "support",
]


SENSITIVE_KEYWORDS = [
    "password", "passwd", "credential", "credentials", "token", "otp", "2fa",
    "bank", "card", "credit", "debit", "ssn", "pin",
]


BRAND_KEYWORDS = [
    "paypal", "google", "microsoft", "apple", "facebook", "meta",
    "instagram", "amazon", "netflix", "steam", "binance", "chase",
    "wellsfargo", "dropbox", "onedrive", "office", "outlook",
    "icloud", "dhl", "fedex", "ups",
]


SOCIAL_DOMAINS = {
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "linkedin.com", "youtube.com", "tiktok.com", "pinterest.com",
    "reddit.com", "snapchat.com", "telegram.org", "t.me",
    "wa.me", "whatsapp.com"
}


def normalize_url(url: str) -> str:
    raw_url = str(url).strip()

    if not raw_url:
        return ""

    if not re.match(r"^https?://", raw_url, re.I):
        return "http://" + raw_url

    return raw_url


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0

    counts = Counter(s)
    length = len(s)

    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def has_ip_address(host: str) -> int:
    if not host:
        return 0

    ipv4_pattern = r"^\d{1,3}(\.\d{1,3}){3}$"

    if not re.match(ipv4_pattern, host):
        return 0

    parts = host.split(".")

    try:
        return int(all(0 <= int(part) <= 255 for part in parts))
    except ValueError:
        return 0


def count_keywords(text: str, keywords: list[str]) -> int:
    text = text.lower()
    return sum(1 for kw in keywords if kw in text)


def fetch_page(url):
    """
    Trả về (response, soup).
    Nếu lỗi trả về (None, None).

    Chỉ dùng cho HTML feature extraction.
    Không dùng trong URL lexical extraction.
    """
    try:
        normalized_url = normalize_url(url)

        if not normalized_url:
            return None, None

        resp = requests.get(
            normalized_url,
            headers=HEADERS,
            timeout=TIMEOUT,
            allow_redirects=True,
            verify=False,
        )

        if resp.status_code >= 400:
            return resp, None

        soup = BeautifulSoup(resp.text, "html.parser")
        return resp, soup

    except Exception:
        return None, None


def extract_features_url(url):
    """
    Trích xuất lexical URL features.
    Không request mạng.
    Không fetch HTML.
    Không check accessible.
    """
    features = {feature: 0 for feature in FEATURES_URL}

    if not url:
        return features

    try:
        normalized_url = normalize_url(url)

        if not normalized_url:
            return features

        parsed = urlparse(normalized_url)

        host = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""

        full_without_scheme = re.sub(
            r"^https?://",
            "",
            normalized_url,
            flags=re.I,
        )

        full_without_scheme = re.sub(
            r"^www\.",
            "",
            full_without_scheme,
            flags=re.I,
        )

        url_len = max(len(full_without_scheme), 1)

        # =====================
        # Original lexical features
        # =====================
        features["IsHTTPS"] = int(normalized_url.lower().startswith("https://"))
        features["DomainLength"] = len(host)

        domain_parts = host.split(".") if host else []
        features["NoOfSubDomain"] = max(0, len(domain_parts) - 2)

        num_letters = sum(c.isalpha() for c in full_without_scheme)
        num_digits = sum(c.isdigit() for c in full_without_scheme)

        num_equals = full_without_scheme.count("=")
        num_qmark = full_without_scheme.count("?")
        num_ampersand = full_without_scheme.count("&")
        total_special = sum(not c.isalnum() for c in full_without_scheme)

        num_other_special = total_special - (
            num_equals + num_qmark + num_ampersand
        )

        features["LetterRatioInURL"] = round(num_letters / url_len, 6)
        features["DegitRatioInURL"] = round(num_digits / url_len, 6)
        features["SpacialCharRatioInURL"] = round(num_other_special / url_len, 6)

        return features

    except Exception as e:
        return {**features, "error": str(e)}


def extract_features_html(url):
    """
    Trích xuất HTML features cũ.

    Missing mặc định = np.nan, không dùng -1.
    Khi train HTML-only, lọc rows HTML success bằng:
        df[FEATURES_HTML].notna().all(axis=1)
    """
    features = {feature: np.nan for feature in FEATURES_HTML}

    if not url:
        return features

    try:
        normalized_url = normalize_url(url)

        if not normalized_url:
            return features

        parsed = urlparse(normalized_url)
        base_host = parsed.hostname or ""

        resp, soup = fetch_page(normalized_url)

        if resp is None or soup is None:
            return features

        html_text = resp.text or ""
        lines = html_text.splitlines()

        features["LineOfCode"] = len(lines)
        features["LargestLineLength"] = max((len(line) for line in lines), default=0)
        features["NoOfImage"] = len(soup.find_all("img"))

        external_scripts = [s for s in soup.find_all("script") if s.get("src")]
        inline_scripts = [
            s for s in soup.find_all("script")
            if not s.get("src") and s.string
        ]

        features["NoOfJS"] = len(external_scripts) + len(inline_scripts)

        all_anchors = soup.find_all("a", href=True)

        # NoOfExternalRef
        cnt_external = 0

        for a in all_anchors:
            href = a["href"].strip()

            if href.startswith("http"):
                a_host = urlparse(href).hostname or ""

                if a_host and a_host != base_host:
                    cnt_external += 1

        features["NoOfExternalRef"] = cnt_external

        # NoOfSelfRef
        cnt_self = 0

        for a in all_anchors:
            href = a["href"].strip()

            if href.startswith("http"):
                a_host = urlparse(href).hostname or ""

                if a_host == base_host:
                    cnt_self += 1

            elif href.startswith("/") or not href.startswith(
                ("mailto:", "tel:", "#", "javascript:")
            ):
                cnt_self += 1

        features["NoOfSelfRef"] = cnt_self

        # HasDescription
        desc_tag = soup.find(
            "meta",
            attrs={"name": re.compile(r"^description$", re.I)},
        )

        features["HasDescription"] = int(
            bool(desc_tag and desc_tag.get("content", "").strip())
        )

        # HasSocialNet
        has_social = 0

        for a in all_anchors:
            href = a["href"].strip()

            if href.startswith("http"):
                a_host = urlparse(href).hostname or ""
                a_host = re.sub(r"^www\.", "", a_host)

                if a_host in SOCIAL_DOMAINS:
                    has_social = 1
                    break

        features["HasSocialNet"] = has_social

        # URLTitleMatchScore
        title_tag = soup.find("title")
        title_text = title_tag.get_text().strip().lower() if title_tag else ""

        host_clean = re.sub(r"^www\.", "", base_host)
        domain_no_tld = (
            host_clean.rsplit(".", 1)[0]
            if "." in host_clean
            else host_clean
        )

        if not domain_no_tld or not title_text:
            features["URLTitleMatchScore"] = 0.0
        else:
            lcs = lcs_length(domain_no_tld, title_text)
            features["URLTitleMatchScore"] = round(
                lcs / len(domain_no_tld) * 100,
                6,
            )

        return features

    except Exception as e:
        return {**features, "error": str(e)}


def lcs_length(s1, s2):
    """
    Longest Common Substring length.
    Dùng cho URLTitleMatchScore:
        LCS(domain_no_tld, title_lower) / len(domain_no_tld) * 100
    """
    m, n = len(s1), len(s2)

    if m == 0 or n == 0:
        return 0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    best = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                best = max(best, dp[i][j])
            else:
                dp[i][j] = 0

    return best
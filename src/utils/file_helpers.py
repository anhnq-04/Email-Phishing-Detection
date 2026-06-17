"""
file_helpers.py
===============
Tiện ích đọc file dữ liệu (JSON, TXT) cho toàn bộ hệ thống.
"""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def load_json(file_path: str) -> dict:
    """
    Đọc file JSON và trả về dict.
    Trả về dict rỗng nếu file không tồn tại hoặc không hợp lệ.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        log.warning(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        log.warning(f"Invalid JSON in {file_path}: {e}")
        return {}


def load_txt(file_path: str) -> set:
    """
    Đọc file text, mỗi dòng là một entry.
    Bỏ qua dòng trống và dòng bắt đầu bằng '#'.
    Trả về set rỗng nếu file không tồn tại.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return {
                line.strip() for line in file
                if line.strip() and not line.strip().startswith('#')
            }
    except FileNotFoundError:
        log.warning(f"File not found: {file_path}")
        return set()

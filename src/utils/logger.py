"""
logger.py
=========
Cấu hình logging tập trung cho toàn bộ hệ thống.
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "phishing_detection",
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """
    Tạo và cấu hình logger.

    Parameters
    ----------
    name : str
        Tên logger.
    level : int
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : str | None
        Đường dẫn file log. Nếu None, chỉ log ra console.

    Returns
    -------
    logging.Logger
        Logger đã được cấu hình.
    """
    logger = logging.getLogger(name)

    # Tránh tạo handler trùng lặp
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

"""
Utilities module — Shared helpers across the project.
  - feature_extraction: URL & HTML feature extraction
  - file_helpers: Load JSON/TXT data files
  - logger: Centralized logging configuration
"""

from src.utils.file_helpers import load_json, load_txt
from src.utils.feature_extraction import extract_features_url, extract_features_html

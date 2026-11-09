"""
Feature Builder for Marked Content Text Joining (Multilingual & Unicode-Aware)
Extracts 30 geometric and surface-lexical features from a boundary pair (Piece A, Piece B).
Optimized for tabular models (LightGBM/XGBoost) and hybrid cascades across Latin, Cyrillic,
and diverse multilingual scripts.
"""

import re
from typing import Dict, Any, List, Tuple
import numpy as np


FEATURE_NAMES = [
    # Geometry (16)
    "same_page",
    "dy",
    "dx",
    "width_a",
    "height_a",
    "width_b",
    "height_b",
    "font_size_a",
    "font_size_b",
    "font_size_ratio",
    "font_size_diff",
    "same_font",
    "same_bold",
    "same_italic",
    "is_same_line",
    "x_overlap",
    
    # Lexical / Surface cues (14)
    "len_a",
    "len_b",
    "ends_with_terminal_punct",
    "ends_with_soft_punct",
    "ends_with_hyphen",
    "ends_with_quote_or_paren",
    "starts_with_lowercase",
    "starts_with_uppercase",
    "starts_with_digit",
    "is_bullet_a",
    "is_bullet_b",
    "is_short_label_a",
    "starts_with_inverted_punct",
    "is_cyrillic"
]

TERMINAL_PUNCTUATION = {'.', '?', '!', '…', ';', '‽'}
SOFT_PUNCTUATION = {',', ';', ':', '·'}
HYPHENS = {'-', '–', '—', '\u00ad', '\u2010', '\u2011'}
QUOTES_AND_PARENS = {')', ']', '}', '"', "'", '”', '’', '»', '”', '„', '“', '›', '«', '）', '】'}
INVERTED_PUNCT = {'¿', '¡'}

UNIVERSAL_BULLET_REGEX = re.compile(
    r'^(?:[•\-\*–—·\u2022\u25E6\u2043\u25AA\u25AB]|(?:\d+|[a-zA-Zа-яА-ЯіїєґІЇЄҐ]|[ivxIVX]+)[\.\)\:\-\/])\s*$',
    re.UNICODE
)


def extract_boundary_features(boundary: Dict[str, Any]) -> np.ndarray:
    """
    Extracts a 30-element float32 feature vector from a boundary dictionary.
    """
    # Geometric features
    page_a = boundary.get('page_a', 0)
    page_b = boundary.get('page_b', 0)
    same_page = 1.0 if page_a == page_b else 0.0
    
    bbox_a = boundary.get('bbox_a', [0.0, 0.0, 0.0, 0.0])
    bbox_b = boundary.get('bbox_b', [0.0, 0.0, 0.0, 0.0])
    
    xa0, ya0, xa1, ya1 = bbox_a
    xb0, yb0, xb1, yb1 = bbox_b
    
    width_a = max(0.0, xa1 - xa0)
    height_a = max(0.0, ya1 - ya0)
    width_b = max(0.0, xb1 - xb0)
    height_b = max(0.0, yb1 - yb0)
    
    if same_page == 1.0:
        dy = yb0 - ya1
        dx = xb0 - xa0
        is_same_line = 1.0 if abs(ya0 - yb0) < 4.0 or abs(ya1 - yb1) < 4.0 else 0.0
        x_overlap = max(0.0, min(xa1, xb1) - max(xa0, xb0))
    else:
        # Cross page: use typical page-margin offsets
        dy = 50.0  # High vertical transition
        dx = xb0 - xa0
        is_same_line = 0.0
        x_overlap = 0.0
        
    fs_a = float(boundary.get('font_size_a', 10.0))
    fs_b = float(boundary.get('font_size_b', 10.0))
    fs_ratio = fs_b / max(fs_a, 0.1)
    fs_diff = abs(fs_a - fs_b)
    
    font_a = str(boundary.get('font_a', '')).lower()
    font_b = str(boundary.get('font_b', '')).lower()
    same_font = 1.0 if font_a == font_b else 0.0
    
    bold_a = 1.0 if boundary.get('is_bold_a') else 0.0
    bold_b = 1.0 if boundary.get('is_bold_b') else 0.0
    same_bold = 1.0 if bold_a == bold_b else 0.0
    
    ital_a = 1.0 if boundary.get('is_italic_a') else 0.0
    ital_b = 1.0 if boundary.get('is_italic_b') else 0.0
    same_italic = 1.0 if ital_a == ital_b else 0.0
    
    # Lexical features
    txt_a = str(boundary.get('text_a', '')).strip()
    txt_b = str(boundary.get('text_b', '')).strip()
    
    len_a = float(len(txt_a))
    len_b = float(len(txt_b))
    
    ends_with_terminal_punct = 1.0 if txt_a and txt_a[-1] in TERMINAL_PUNCTUATION else 0.0
    ends_with_soft_punct = 1.0 if txt_a and txt_a[-1] in SOFT_PUNCTUATION else 0.0
    ends_with_hyphen = 1.0 if txt_a and txt_a[-1] in HYPHENS else 0.0
    ends_with_quote_or_paren = 1.0 if txt_a and txt_a[-1] in QUOTES_AND_PARENS else 0.0
    
    first_char_b = txt_b[0] if txt_b else ''
    starts_with_inverted_punct = 1.0 if first_char_b in INVERTED_PUNCT else 0.0
    
    # Case inspection: if leading char is punctuation or quotes, look at first alphabetic char
    if first_char_b in INVERTED_PUNCT or first_char_b in QUOTES_AND_PARENS or first_char_b in '\'"`':
        first_alpha = next((c for c in txt_b if c.isalpha()), '')
        starts_with_lowercase = 1.0 if first_alpha.islower() else 0.0
        starts_with_uppercase = 1.0 if first_alpha.isupper() else 0.0
    else:
        starts_with_lowercase = 1.0 if first_char_b.islower() else 0.0
        starts_with_uppercase = 1.0 if first_char_b.isupper() else 0.0
        
    starts_with_digit = 1.0 if first_char_b.isdigit() else 0.0
    
    is_bullet_a = 1.0 if bool(UNIVERSAL_BULLET_REGEX.match(txt_a)) else 0.0
    is_bullet_b = 1.0 if bool(UNIVERSAL_BULLET_REGEX.match(txt_b)) else 0.0
    is_short_label_a = 1.0 if len(txt_a) <= 4 else 0.0
    
    # Cyrillic script indicator (Russian, Ukrainian)
    is_cyrillic = 1.0 if any('\u0400' <= c <= '\u04FF' for c in (txt_a + txt_b)) else 0.0
    
    vec = [
        same_page,
        dy,
        dx,
        width_a,
        height_a,
        width_b,
        height_b,
        fs_a,
        fs_b,
        fs_ratio,
        fs_diff,
        same_font,
        same_bold,
        same_italic,
        is_same_line,
        x_overlap,
        len_a,
        len_b,
        ends_with_terminal_punct,
        ends_with_soft_punct,
        ends_with_hyphen,
        ends_with_quote_or_paren,
        starts_with_lowercase,
        starts_with_uppercase,
        starts_with_digit,
        is_bullet_a,
        is_bullet_b,
        is_short_label_a,
        starts_with_inverted_punct,
        is_cyrillic
    ]
    
    return np.array(vec, dtype=np.float32)


def extract_batch_features(boundaries: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts feature matrix X and label vector y for a list of boundary dicts.
    """
    X = np.zeros((len(boundaries), len(FEATURE_NAMES)), dtype=np.float32)
    y = np.zeros(len(boundaries), dtype=np.int32)
    
    for i, b in enumerate(boundaries):
        X[i] = extract_boundary_features(b)
        y[i] = int(b.get('label', 0))
        
    return X, y

"""
Text cleaning / preprocessing utilities applied to raw scraped text before
chunking and embedding.
"""

from __future__ import annotations

import re
import unicodedata


_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_URL_RE = re.compile(r"https?://\S+")
_BOILERPLATE_PATTERNS = [
    re.compile(r"^(cookie|privacy) policy", re.IGNORECASE),
    re.compile(r"^subscribe to our newsletter", re.IGNORECASE),
    re.compile(r"^all rights reserved", re.IGNORECASE),
    re.compile(r"^share this", re.IGNORECASE),
]


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def strip_urls(text: str) -> str:
    return _URL_RE.sub("", text)


def collapse_whitespace(text: str) -> str:
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def remove_boilerplate_lines(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if any(pat.match(stripped) for pat in _BOILERPLATE_PATTERNS):
            continue
        if len(stripped) < 3:
            continue
        lines.append(stripped)
    return "\n".join(lines)


def clean_text(text: str) -> str:
    """Full cleaning pipeline applied to a single document's raw text."""
    text = normalize_unicode(text)
    text = strip_urls(text)
    text = remove_boilerplate_lines(text)
    text = collapse_whitespace(text)
    return text

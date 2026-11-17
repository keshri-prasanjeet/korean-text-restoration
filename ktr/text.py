"""Text normalisation and sentence filtering for Korean Wikipedia."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

_WS = re.compile(r"\s+")
_SENT_END = re.compile(r"(?<=[.!?])\s+")
_LATIN_END = re.compile(r"[A-Za-z]\.$")
_ALLOWED = re.compile(r"^[가-힣0-9A-Za-z .,!?·'\"()%~:/\-]+$")


def normalize(text: str) -> str:
    """NFC-normalise and collapse every run of whitespace to one space."""
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def is_hangul(ch: str) -> bool:
    return "가" <= ch <= "힣"


def hangul_ratio(text: str) -> float:
    chars = [c for c in text if c != " "]
    if not chars:
        return 0.0
    return sum(is_hangul(c) for c in chars) / len(chars)


def split_sentences(text: str) -> Iterator[str]:
    """Split article text into sentences on line breaks and sentence-final punctuation."""
    for para in text.split("\n"):
        para = normalize(para)
        if not para:
            continue
        for sent in _SENT_END.split(para):
            sent = sent.strip()
            if sent:
                yield sent


def is_good_sentence(sent: str, min_len: int = 20, max_len: int = 120) -> bool:
    """Keep complete, mostly-Hangul prose sentences with at least four words."""
    if not min_len <= len(sent) <= max_len:
        return False
    if sent.count(" ") < 3:
        return False
    if not sent.endswith((".", "!", "?")):
        return False
    if not _ALLOWED.match(sent):
        return False
    if hangul_ratio(sent) < 0.7:
        return False
    # A Latin letter before the final period is usually an initial or an
    # abbreviation ("로버트 F."), i.e. a wrong sentence split.
    if _LATIN_END.search(sent):
        return False
    # Unbalanced brackets usually mean a broken split or leftover markup.
    if sent.count("(") != sent.count(")"):
        return False
    return True

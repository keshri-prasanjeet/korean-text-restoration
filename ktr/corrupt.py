"""Spacing corruptions that turn a correctly spaced sentence into model input.

Conditions
----------
nospace : every space removed (e.g. text typed without spaces, OCR output).
mixed   : about half the spaces dropped and a few wrong spaces inserted
          between Hangul syllables (typical human spacing mistakes).
clean   : the sentence is left as it is. A good model must not "fix" it.
"""

from __future__ import annotations

import random

from .text import is_hangul, normalize

CONDITIONS = ("nospace", "mixed", "clean")


def remove_all(sent: str) -> str:
    return sent.replace(" ", "")


def drop_and_insert(sent: str, rng: random.Random, p_drop: float = 0.5, p_insert: float = 0.05) -> str:
    out: list[str] = []
    for i, ch in enumerate(sent):
        if ch == " ":
            if rng.random() >= p_drop:
                out.append(" ")
            continue
        out.append(ch)
        nxt = sent[i + 1] if i + 1 < len(sent) else ""
        if nxt and nxt != " " and is_hangul(ch) and is_hangul(nxt) and rng.random() < p_insert:
            out.append(" ")
    return normalize("".join(out))


def corrupt(sent: str, condition: str, rng: random.Random) -> str:
    if condition == "nospace":
        return remove_all(sent)
    if condition == "clean":
        return sent
    if condition == "mixed":
        # Retry so a "mixed" example almost never comes out identical to the target.
        for _ in range(10):
            noisy = drop_and_insert(sent, rng)
            if noisy != sent:
                return noisy
        return noisy
    raise ValueError(f"unknown condition: {condition}")

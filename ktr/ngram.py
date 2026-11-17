"""Character n-gram spacing baseline.

For every gap between two characters, the model looks at up to two characters
on each side and predicts a space when, in training data, a space followed
that context at least half the time. It backs off to shorter contexts when a
context was seen fewer than `min_count` times. It ignores any spaces already in
the input and re-spaces from scratch, so it runs in seconds on a CPU.
"""

from __future__ import annotations

from collections.abc import Iterable

from .metrics import boundaries, render

ORDERS = ((2, 2), (2, 1), (1, 2), (1, 1))
PAD_L, PAD_R = "\x02", "\x03"


def _contexts(chars: str, i: int) -> list[str]:
    padded = PAD_L * 2 + chars + PAD_R * 2
    j = i + 2  # index of chars[i] in padded
    return [f"{l}{r}|{padded[j - l + 1:j + 1]}|{padded[j + 1:j + 1 + r]}" for l, r in ORDERS]


class NgramSpacer:
    def __init__(self, min_count: int = 3, threshold: float = 0.5) -> None:
        self.min_count = min_count
        self.threshold = threshold
        self.total: dict[str, int] = {}
        self.spaced: dict[str, int] = {}

    def fit(self, sentences: Iterable[str]) -> NgramSpacer:
        total, spaced = self.total, self.spaced
        for sent in sentences:
            chars, bounds = boundaries(sent)
            for i in range(len(chars) - 1):
                is_space = i in bounds
                for key in _contexts(chars, i):
                    total[key] = total.get(key, 0) + 1
                    if is_space:
                        spaced[key] = spaced.get(key, 0) + 1
        return self

    def predict(self, sent: str) -> str:
        chars, _ = boundaries(sent)
        bounds: set[int] = set()
        for i in range(len(chars) - 1):
            for key in _contexts(chars, i):
                t = self.total.get(key, 0)
                if t >= self.min_count:
                    if self.spaced.get(key, 0) / t >= self.threshold:
                        bounds.add(i)
                    break
        return render(chars, bounds)

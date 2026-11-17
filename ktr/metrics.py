"""Spacing metrics.

A sentence is reduced to its non-space characters plus the set of positions
after which a space appears ("boundaries"). Prediction and target are compared
on those boundaries, on whole words, and on exact sentence match.

If a prediction changes the non-space characters (a model rewrote, dropped or
added text), it is counted as a content change: every gold boundary and word
in that sentence is counted as missed and the sentence is not an exact match.
`project_spacing` offers an optional, separately reported repair that copies
the predicted spaces back onto the original characters.
"""

from __future__ import annotations

from collections.abc import Iterable
from difflib import SequenceMatcher

from .text import normalize


def boundaries(sent: str) -> tuple[str, set[int]]:
    chars: list[str] = []
    bounds: set[int] = set()
    for ch in sent:
        if ch == " ":
            if chars:
                bounds.add(len(chars) - 1)
        else:
            chars.append(ch)
    bounds.discard(len(chars) - 1)
    return "".join(chars), bounds


def word_spans(n_chars: int, bounds: set[int]) -> set[tuple[int, int]]:
    spans: set[tuple[int, int]] = set()
    start = 0
    for i in range(n_chars):
        if i in bounds or i == n_chars - 1:
            spans.add((start, i))
            start = i + 1
    return spans


def render(chars: str, bounds: set[int]) -> str:
    return "".join(c + (" " if i in bounds else "") for i, c in enumerate(chars)).strip()


def project_spacing(source: str, prediction: str) -> str:
    """Apply the prediction's spaces to the source's characters.

    Characters are aligned with difflib; spaces that follow a predicted
    character matched to a source character are kept. The output always has
    exactly the source's non-space characters.
    """
    src_chars, _ = boundaries(normalize(source))
    pred_chars, pred_bounds = boundaries(normalize(prediction))
    mapping: dict[int, int] = {}
    matcher = SequenceMatcher(None, pred_chars, src_chars, autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            mapping[block.a + k] = block.b + k
    src_bounds = {mapping[i] for i in pred_bounds if i in mapping}
    src_bounds.discard(len(src_chars) - 1)
    return render(src_chars, src_bounds)


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(pairs: Iterable[tuple[str, str]]) -> dict[str, float]:
    """Score (prediction, target) pairs. Returns micro-averaged metrics."""
    tp = fp = fn = wtp = wfp = wfn = exact = n = changed = 0
    for pred, target in pairs:
        n += 1
        t_chars, t_bounds = boundaries(normalize(target))
        p_chars, p_bounds = boundaries(normalize(pred))
        t_words = word_spans(len(t_chars), t_bounds)
        if p_chars != t_chars:
            changed += 1
            fn += len(t_bounds)
            wfn += len(t_words)
            continue
        p_words = word_spans(len(p_chars), p_bounds)
        tp += len(p_bounds & t_bounds)
        fp += len(p_bounds - t_bounds)
        fn += len(t_bounds - p_bounds)
        wtp += len(p_words & t_words)
        wfp += len(p_words - t_words)
        wfn += len(t_words - p_words)
        exact += p_bounds == t_bounds
    bp, br, bf = _f1(tp, fp, fn)
    _, _, wf = _f1(wtp, wfp, wfn)
    return {
        "n": n,
        "boundary_precision": bp,
        "boundary_recall": br,
        "boundary_f1": bf,
        "word_f1": wf,
        "sentence_accuracy": exact / n if n else 0.0,
        "content_changed_rate": changed / n if n else 0.0,
    }


def score_by_condition(rows: list[dict], predictions: list[str]) -> dict[str, dict[str, float]]:
    """Score per test condition and overall. `rows` need `condition` and `target`."""
    out: dict[str, dict[str, float]] = {}
    for cond in sorted({r["condition"] for r in rows}):
        out[cond] = score((p, r["target"]) for r, p in zip(rows, predictions) if r["condition"] == cond)
    out["all"] = score((p, r["target"]) for r, p in zip(rows, predictions))
    return out

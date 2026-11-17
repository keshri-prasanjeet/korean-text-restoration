"""Reading and writing the processed JSONL splits."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from pathlib import Path

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict] = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    n = 0
    with opener(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def limit_per_condition(rows: list[dict], limit: int | None) -> list[dict]:
    """Keep the first `limit` rows of each condition (deterministic subset)."""
    if limit is None:
        return rows
    seen: dict[str, int] = {}
    out = []
    for r in rows:
        c = r["condition"]
        if seen.get(c, 0) < limit:
            out.append(r)
            seen[c] = seen.get(c, 0) + 1
    return out

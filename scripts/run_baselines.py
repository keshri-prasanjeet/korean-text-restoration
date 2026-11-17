"""CPU baselines on the test set: copy-input and a character n-gram model.

    python scripts/run_baselines.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ktr.data import PROCESSED, limit_per_condition, read_jsonl, write_jsonl  # noqa: E402
from ktr.metrics import score_by_condition  # noqa: E402
from ktr.ngram import NgramSpacer  # noqa: E402


def save(name: str, rows: list[dict], preds: list[str], extra: dict) -> None:
    metrics = score_by_condition(rows, preds)
    result = {"system": name, **extra, "metrics": metrics}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    write_jsonl(ROOT / "results" / "predictions" / f"{name}.jsonl.gz",
                ({"id": r["id"], "condition": r["condition"], "input": r["input"],
                  "prediction": p, "target": r["target"]} for r, p in zip(rows, preds)))
    print(name, {c: round(m["boundary_f1"], 4) for c, m in metrics.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=str(PROCESSED / "test.jsonl.gz"))
    ap.add_argument("--train", default=str(PROCESSED / "train.jsonl.gz"))
    ap.add_argument("--limit-per-condition", type=int, default=None)
    args = ap.parse_args()

    rows = limit_per_condition(read_jsonl(args.test), args.limit_per_condition)

    save("copy-input", rows, [r["input"] for r in rows], {"description": "returns the input unchanged"})

    t0 = time.time()
    train = [r["target"] for r in read_jsonl(args.train)]
    model = NgramSpacer().fit(train)
    fit_s = time.time() - t0
    save("char-ngram", rows, [model.predict(r["input"]) for r in rows],
         {"description": "character 2+2 context backoff model trained on the target side of train",
          "train_sentences": len(train), "contexts": len(model.total), "fit_seconds": round(fit_s, 1)})


if __name__ == "__main__":
    main()

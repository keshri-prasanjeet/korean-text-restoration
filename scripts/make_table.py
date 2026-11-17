"""Print a Markdown results table from results/*.json.

    python scripts/make_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["copy-input", "char-ngram"]
CONDS = ["nospace", "mixed", "clean", "all"]


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def main() -> None:
    results = [json.loads(p.read_text()) for p in sorted((ROOT / "results").glob("*.json"))]
    results.sort(key=lambda r: (ORDER.index(r["system"]) if r["system"] in ORDER else len(ORDER), r["system"]))

    print("Boundary F1 / sentence accuracy (%), by input condition\n")
    print("| System | " + " | ".join(CONDS) + " | Content changed |")
    print("|---|" + "---:|" * (len(CONDS) + 1))
    for r in results:
        m = r["metrics"]
        cells = [f"{pct(m[c]['boundary_f1'])} / {pct(m[c]['sentence_accuracy'])}" for c in CONDS]
        print(f"| `{r['system']}` | " + " | ".join(cells) + f" | {pct(m['all']['content_changed_rate'])} |")

    print("\nWord F1 (%), all conditions\n")
    print("| System | Word F1 | n |")
    print("|---|---:|---:|")
    for r in results:
        print(f"| `{r['system']}` | {pct(r['metrics']['all']['word_f1'])} | {r['metrics']['all']['n']} |")


if __name__ == "__main__":
    main()

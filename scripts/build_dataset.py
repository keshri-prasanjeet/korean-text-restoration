"""Build train/val/test spacing pairs from a Korean Wikipedia parquet shard.

Source: Hugging Face `wikimedia/wikipedia`, config `20231101.ko` (CC BY-SA 4.0).

    curl -L -o data/raw/kowiki-20231101-part2.parquet \
      https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.ko/train-00002-of-00003.parquet
    python scripts/build_dataset.py --parquet data/raw/kowiki-20231101-part2.parquet

Articles (not sentences) are assigned to splits by a hash of their id, so no
article contributes to more than one split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ktr.corrupt import CONDITIONS, corrupt  # noqa: E402
from ktr.data import PROCESSED, write_jsonl  # noqa: E402
from ktr.text import is_good_sentence, split_sentences  # noqa: E402

TRAIN_MIX = {"nospace": 0.40, "mixed": 0.45, "clean": 0.15}


def split_of(article_id: str) -> str:
    bucket = int(hashlib.md5(article_id.encode()).hexdigest(), 16) % 100
    if bucket < 2:
        return "test"
    if bucket < 4:
        return "val"
    return "train"


def seeded_rng(*parts: object) -> random.Random:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--train-size", type=int, default=60000)
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--test-size", type=int, default=1000)
    ap.add_argument("--per-article", type=int, default=4, help="max sentences taken from one article")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=str(PROCESSED))
    args = ap.parse_args()

    limits = {"train": args.train_size, "val": args.val_size, "test": args.test_size}
    sentences: dict[str, list[tuple[str, str]]] = {k: [] for k in limits}
    seen: set[str] = set()
    articles = 0

    for path in args.parquet:
        for batch in pq.ParquetFile(path).iter_batches(columns=["id", "text"], batch_size=2000):
            for art_id, text in zip(batch.column("id").to_pylist(), batch.column("text").to_pylist()):
                split = split_of(art_id)
                if len(sentences[split]) >= limits[split]:
                    continue
                articles += 1
                taken = 0
                for sent in split_sentences(text):
                    if taken >= args.per_article or len(sentences[split]) >= limits[split]:
                        break
                    if sent in seen or not is_good_sentence(sent):
                        continue
                    seen.add(sent)
                    sentences[split].append((art_id, sent))
                    taken += 1
            if all(len(sentences[k]) >= limits[k] for k in limits):
                break

    out = Path(args.out)
    stats: dict[str, object] = {"articles_read": articles, "source": "wikimedia/wikipedia 20231101.ko"}

    train_rows = []
    for idx, (art_id, sent) in enumerate(sentences["train"]):
        rng = seeded_rng(args.seed, "train", idx)
        cond = rng.choices(list(TRAIN_MIX), weights=list(TRAIN_MIX.values()))[0]
        train_rows.append({"id": f"train-{idx}", "article_id": art_id, "condition": cond,
                           "input": corrupt(sent, cond, rng), "target": sent})
    random.Random(args.seed).shuffle(train_rows)
    stats["train"] = write_jsonl(out / "train.jsonl.gz", train_rows)

    for split in ("val", "test"):
        rows = []
        for cond in CONDITIONS:  # grouped by condition so a per-condition prefix is a fixed subset
            for idx, (art_id, sent) in enumerate(sentences[split]):
                rng = seeded_rng(args.seed, split, idx, cond)
                rows.append({"id": f"{split}-{idx}-{cond}", "article_id": art_id, "condition": cond,
                             "input": corrupt(sent, cond, rng), "target": sent})
        stats[split] = write_jsonl(out / f"{split}.jsonl.gz", rows)

    stats["sentences"] = {k: len(v) for k, v in sentences.items()}
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    short = [k for k in limits if len(sentences[k]) < limits[k]]
    if short:
        print(f"warning: not enough sentences for {short}; add another parquet shard", file=sys.stderr)


if __name__ == "__main__":
    main()

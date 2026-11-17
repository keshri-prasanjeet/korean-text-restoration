"""Evaluate a chat model (optionally with a LoRA adapter) on the test set.

    # zero-shot base model
    python scripts/evaluate_llm.py --base-model Qwen/Qwen2.5-1.5B-Instruct --name qwen2.5-1.5b-zeroshot
    # fine-tuned
    python scripts/evaluate_llm.py --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter outputs/qwen2.5-1.5b-spacing/adapter --name qwen2.5-1.5b-lora

Writes results/<name>.json (strict metrics) and results/<name>+projected.json
(the same outputs after copying their spaces back onto the input characters).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ktr.data import PROCESSED, limit_per_condition, read_jsonl, write_jsonl  # noqa: E402
from ktr.metrics import project_spacing, score_by_condition  # noqa: E402
from ktr.prompt import build_messages, clean_output  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--test", default=str(PROCESSED / "test.jsonl.gz"))
    ap.add_argument("--limit-per-condition", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32

    tok = AutoTokenizer.from_pretrained(args.adapter or args.base_model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype,
                                                 device_map={"": 0} if cuda else None)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()

    rows = limit_per_condition(read_jsonl(args.test), args.limit_per_condition)
    prompts = [tok.apply_chat_template(build_messages(r["input"]), tokenize=False, add_generation_prompt=True)
               for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))
    preds: list[str] = [""] * len(rows)

    t0 = time.time()
    for start in range(0, len(order), args.batch_size):
        idx = order[start:start + args.batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(model.device)
        max_in = max(len(tok(rows[i]["input"], add_special_tokens=False)["input_ids"]) for i in idx)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=int(max_in * 1.6) + 16, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            preds[i] = clean_output(t)
        print(f"{min(start + args.batch_size, len(order))}/{len(order)}", flush=True)
    elapsed = time.time() - t0

    projected = [project_spacing(r["input"], p) for r, p in zip(rows, preds)]
    (ROOT / "results").mkdir(exist_ok=True)
    common = {"base_model": args.base_model, "adapter": args.adapter, "seconds": round(elapsed, 1),
              "decoding": "greedy"}
    for name, ps, note in ((args.name, preds, "strict: raw model output"),
                           (f"{args.name}+projected", projected,
                            "model spaces copied onto the input characters")):
        result = {"system": name, "description": note, **common, "metrics": score_by_condition(rows, ps)}
        (ROOT / "results" / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(name, {c: round(m["boundary_f1"], 4) for c, m in result["metrics"].items()})
    write_jsonl(ROOT / "results" / "predictions" / f"{args.name}.jsonl.gz",
                ({"id": r["id"], "condition": r["condition"], "input": r["input"], "prediction": p,
                  "projected": q, "target": r["target"]} for r, p, q in zip(rows, preds, projected)))


if __name__ == "__main__":
    main()

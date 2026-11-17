"""LoRA / QLoRA fine-tuning of a small chat model for Korean spacing correction.

Designed for one free 16 GB GPU (Kaggle or Colab T4):

    python scripts/train_lora.py --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --train-size 20000 --output-dir outputs/qwen2.5-1.5b-spacing --load-in-4bit

Loss is computed on the assistant answer only; the system prompt and the
corrupted input are masked out. Checkpoints are written every --save-steps, and
--resume continues from the latest one after a disconnected session.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ktr.data import PROCESSED, read_jsonl  # noqa: E402
from ktr.prompt import build_messages  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(rows: list[dict], tok, max_len: int) -> list[dict]:
    items = []
    for r in rows:
        prompt = tok.apply_chat_template(build_messages(r["input"]), tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(build_messages(r["input"], r["target"]), tokenize=False)
        if not full.startswith(prompt):
            raise ValueError("chat template does not extend the generation prompt; pick another model")
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
        ids = (p_ids + c_ids)[:max_len]
        labels = ([-100] * len(p_ids) + c_ids)[:max_len]
        if all(x == -100 for x in labels):
            continue
        items.append({"input_ids": ids, "labels": labels})
    return items


class Collator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            labels[i, :k] = torch.tensor(b["labels"])
            mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train", default=str(PROCESSED / "train.jsonl.gz"))
    ap.add_argument("--val", default=str(PROCESSED / "val.jsonl.gz"))
    ap.add_argument("--train-size", type=int, default=20000)
    ap.add_argument("--val-size", type=int, default=300)
    ap.add_argument("--output-dir", default="outputs/qwen2.5-1.5b-spacing")
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--load-in-4bit", action="store_true", help="QLoRA (needs CUDA + bitsandbytes)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    set_seed(args.seed)
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    fp16 = cuda and not bf16

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_items = encode(read_jsonl(args.train, args.train_size), tok, args.max_len)
    val_items = encode(read_jsonl(args.val, args.val_size), tok, args.max_len)
    print(f"train examples: {len(train_items)}  val examples: {len(val_items)}")

    model_kwargs: dict = {}
    if args.load_in_4bit:
        if not cuda:
            raise SystemExit("--load-in-4bit needs a CUDA GPU")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )
        model_kwargs["device_map"] = {"": 0}
    elif cuda:
        model_kwargs["dtype"] =torch.bfloat16 if bf16 else torch.float16
        model_kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
    ))
    if fp16:
        # Trainable LoRA weights must be fp32 for fp16 mixed precision.
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    model.print_trainable_parameters()

    out = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # a float below 1 is a ratio of total steps (transformers 5)
        weight_decay=0.0,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=cuda,
        gradient_checkpointing_kwargs={"use_reentrant": False} if cuda else None,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_items,
        eval_dataset=val_items,
        data_collator=Collator(tok.pad_token_id),
    )
    has_ckpt = any(out.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if (args.resume and has_ckpt) else None)

    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    meta = {"base_model": args.base_model, "train_examples": len(train_items),
            "args": vars(args), "log_history": trainer.state.log_history}
    (out / "training_log.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"saved adapter to {adapter_dir}")


if __name__ == "__main__":
    main()

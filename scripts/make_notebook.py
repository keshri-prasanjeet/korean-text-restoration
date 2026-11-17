"""Generate notebooks/train_and_evaluate.ipynb (Kaggle / Colab)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
RUN = "qwen2.5-1.5b"

cells = [
    ("markdown", """# Korean spacing correction: LoRA fine-tuning on a free GPU

Runs the whole experiment from this repository on **one free 16 GB GPU**.

**Kaggle:** Settings → Accelerator → *GPU T4 x2* (the notebook uses one), Internet → *On*.
**Colab:** Runtime → Change runtime type → *T4 GPU*.

Rough time on a T4: zero-shot evaluation about 15 min, training about 45–60 min, fine-tuned evaluation about 15 min.
If the session disconnects, run the cells again; training resumes from the last checkpoint (`--resume`).

Optional: add a Hugging Face write token as a secret named `HF_TOKEN` to upload the adapter at the end."""),
    ("code", """import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # use a single GPU even on "T4 x2"
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
    ("code", """os.chdir("/kaggle/working" if os.path.isdir("/kaggle/working") else "/content")
!git clone -q https://github.com/keshri-prasanjeet/korean-text-restoration.git || (cd korean-text-restoration && git pull -q)
%cd korean-text-restoration
!pip install -q -e ".[train]"
!python -m pytest -q"""),
    ("markdown", "## 1. Zero-shot baseline (base model, no fine-tuning)"),
    ("code", f"""!python scripts/evaluate_llm.py --base-model {MODEL} --name {RUN}-zeroshot"""),
    ("markdown", "## 2. LoRA fine-tuning (QLoRA, 4-bit base)"),
    ("code", f"""!python scripts/train_lora.py --base-model {MODEL} --train-size 20000 \\
    --output-dir outputs/{RUN}-spacing --load-in-4bit --resume"""),
    ("markdown", "## 3. Evaluate the fine-tuned model"),
    ("code", f"""!python scripts/evaluate_llm.py --base-model {MODEL} \\
    --adapter outputs/{RUN}-spacing/adapter --name {RUN}-lora"""),
    ("code", """!python scripts/make_table.py"""),
    ("markdown", """## 4. Save the results

Download `results.zip` (Kaggle: *Output* panel; Colab: the file browser), unzip it in the repository root, and commit `results/`.
The adapter is in `outputs/` and can be uploaded with the next cell."""),
    ("code", f"""!cp outputs/{RUN}-spacing/training_log.json results/{RUN}-lora-training_log.json
!zip -qr results.zip results
!zip -qr adapter.zip outputs/{RUN}-spacing/adapter
!ls -lh results.zip adapter.zip"""),
    ("code", f"""# Optional: upload the LoRA adapter to the Hugging Face Hub.
token = None
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
    except Exception:
        pass

if token:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    repo_id = f"{{api.whoami()['name']}}/{RUN}-ko-spacing-lora"
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(folder_path="outputs/{RUN}-spacing/adapter", repo_id=repo_id)
    print("uploaded to https://huggingface.co/" + repo_id)
else:
    print("No HF_TOKEN secret found; skipping upload.")"""),
]


def to_cell(kind: str, src: str) -> dict:
    lines = src.splitlines(keepends=True)
    cell = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


nb = {
    "cells": [to_cell(k, s) for k, s in cells],
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = ROOT / "notebooks" / "train_and_evaluate.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {out}")

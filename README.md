# Korean spacing correction with a small LLM

This project fine-tunes a 1.5B-parameter chat model (Qwen2.5-1.5B-Instruct) with LoRA to fix
Korean word spacing (띄어쓰기), and compares it with simple baselines on a test set built from
Korean Wikipedia. Everything runs on a laptop CPU except training and LLM evaluation, which
run on one free Kaggle or Colab T4 GPU.

```
input : 전구간국도제32호선에속하므로이노선에대한별도표기는생략한다.
target: 전 구간 국도 제32호선에 속하므로 이 노선에 대한 별도 표기는 생략한다.
```

## Why spacing

Korean is written in words (어절) separated by spaces, but particles and endings attach to the
word before them, and the rules for bound nouns (것, 수, 데), auxiliary verbs and compound
nouns are hard even for native writers. Text without spaces is common in chat, OCR output and
some web text, and missing spaces change how tokenizers and downstream models read a sentence.
It is a narrow task with an exact answer, so a small model can be measured cleanly against
rules and against larger models.

## Data

Sentences come from the Hugging Face dump `wikimedia/wikipedia`, config `20231101.ko`
(shard 3 of 3). `scripts/build_dataset.py` keeps sentences that:

- are 20 to 120 characters long and end with `.`, `!` or `?`
- have at least four words and at least 70% Hangul
- contain no markup or unusual symbols, and have balanced brackets
- do not end in a Latin letter plus a period (usually a wrong split at an initial)

At most four sentences are taken from one article, and duplicates are removed. Articles are
assigned to train, validation or test by a hash of their id, so no article appears in two
splits.

Each correct sentence is turned into an input by one of three corruptions:

| Condition | What the input looks like |
|---|---|
| `nospace` | all spaces removed |
| `mixed` | about half the spaces dropped and a few wrong spaces inserted between syllables |
| `clean` | unchanged; the model should return it as it is |

| Split | Sentences | Rows | Conditions |
|---|---:|---:|---|
| train | 60,000 | 60,000 | one per sentence: 40% nospace, 45% mixed, 15% clean |
| val | 500 | 1,500 | all three per sentence |
| test | 1,000 | 3,000 | all three per sentence |

The processed files are in `data/processed/` (gzipped JSONL with `id`, `article_id`,
`condition`, `input`, `target`).

## Metrics

Prediction and target are compared on their non-space characters and the positions where a
space follows a character.

- **Boundary F1:** precision and recall over space positions.
- **Word F1:** a word counts as correct only if both of its edges match.
- **Sentence accuracy:** the whole sentence is spaced exactly like the target.
- **Content changed:** the output does not have the same characters as the input. LLMs
  sometimes rewrite a word or drop a particle. Such outputs score zero on that sentence.

For LLMs, `evaluate_llm.py` also reports a `+projected` variant, which aligns the output
with the input and copies its spaces onto the original characters. It is reported
separately and never replaces the strict score.

## Results

Test set, 1,000 sentences per condition. Each cell is boundary F1 / sentence accuracy (%).

| System | nospace | mixed | clean | all | Content changed |
|---|---:|---:|---:|---:|---:|
| `copy-input` | 0.0 / 0.0 | 61.7 / 0.0 | 100.0 / 100.0 | 64.9 / 33.3 | 0.0 |
| `char-ngram` | 87.6 / 17.8 | 87.6 / 17.8 | 87.6 / 17.8 | 87.6 / 17.8 | 0.0 |
| `qwen2.5-1.5b-zeroshot` | pending GPU run | | | | |
| `qwen2.5-1.5b-lora` | pending GPU run | | | | |

`copy-input` returns the input unchanged. `char-ngram` re-spaces every input from scratch
using counts of two characters on each side of a gap, learned from the 60,000 training
targets (about 3 million contexts, 7 seconds to fit on a laptop). It ignores spaces already in
the input, so its score is the same in all three conditions. It gets 87.6% of space positions
right but only 17.8% of sentences fully right, because one mistake in a 50-character sentence
is enough to fail it.

The two Qwen rows are filled in by running the notebook below. Per-sentence predictions for
every system are kept in `results/predictions/`.

## Run it

CPU part (dataset and baselines), Python 3.10 or newer:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[data,dev]"
pytest
python scripts/run_baselines.py
python scripts/make_table.py
```

To rebuild the dataset from scratch, download the Wikipedia shard (about 180 MB) first:

```bash
curl -L -o data/raw/kowiki-20231101-part2.parquet https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.ko/train-00002-of-00003.parquet
python scripts/build_dataset.py --parquet data/raw/kowiki-20231101-part2.parquet
```

GPU part: open `notebooks/train_and_evaluate.ipynb` in Kaggle (Accelerator: GPU T4, Internet
on) or Colab (T4 runtime) and run all cells. It clones this repository, evaluates the base
model zero-shot, trains a QLoRA adapter on 20,000 training rows, evaluates the adapter and
prints the table. It saves `results.zip`; unzip it into the repository and commit `results/`.

Training settings: LoRA rank 16, alpha 32, dropout 0.05 on all attention and MLP projections;
4-bit NF4 base model; learning rate 2e-4 with cosine decay; effective batch size 16; one
epoch; loss on the answer only. Decoding is greedy.

## Layout

```
ktr/                 text filters, corruptions, metrics, n-gram baseline, prompt
scripts/
  build_dataset.py   Wikipedia parquet -> train/val/test JSONL
  run_baselines.py   copy-input and char-ngram on the test set
  train_lora.py      LoRA / QLoRA training
  evaluate_llm.py    batched generation and scoring for a model or adapter
  make_table.py      results/*.json -> Markdown table
  make_notebook.py   writes the Kaggle/Colab notebook
notebooks/           train_and_evaluate.ipynb
data/processed/      dataset splits
results/             metrics per system and per-sentence predictions
tests/               unit tests for metrics, corruptions and the baseline
```

## Limitations

- The targets are Wikipedia's spacing, not checked by a linguist. Wikipedia mostly follows
  standard orthography, but not always.
- Standard Korean orthography allows both spaced and joined forms for many compound nouns and
  proper names (for example 간양 교차로 / 간양교차로). Only Wikipedia's form counts as correct
  here, so every system is penalised for valid alternatives.
- The `mixed` errors are random. Real human spacing errors cluster around particular
  patterns (bound nouns, auxiliary verbs), so this condition is an approximation.
- All text is encyclopedic. Chat, news and learner writing would likely give different
  numbers.

## License

Code: MIT. Data in `data/processed/` and model outputs in `results/predictions/` are derived
from Korean Wikipedia and are released under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); see `data/README.md`.

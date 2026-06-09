# Evaluation — How to Run

All commands must be run from the **project root** (`Trump_GPT/`).

---

## 1. Setup

### Install dependencies

```bash
pip install torch transformers accelerate bitsandbytes huggingface_hub requests python-dotenv spacy pyyaml
python -m spacy download en_core_web_sm
```

> **Do NOT install `unbabel-comet`** — it requires `transformers<5.0` which breaks Gemma 4.
> COMET support is currently disabled for this reason (see Metric 2 below).

### Log in to HuggingFace

Required to download the Gemma and finetuned models (both are gated/private):

```bash
huggingface-cli login
```

Models are downloaded automatically on first run — no separate download step needed.

### Environment variables

Required only when using the API model or LLM judge. Create `.env` at the project root:

```
PCSS_API_KEY=your_token_here
PCSS_BASE_URL=https://llm.hpc.psnc.pl/v1/chat/completions
```

`PCSS_BASE_URL` is optional — the value above is the default.
If running with `--skip-api` and `--translate-only`, `.env` can be empty or omitted.

---

## 2. Prepare the evaluation set

The script always reads source examples from `eval_results/eval_set.jsonl` and translations
from `eval_results/translations/`. These files must exist before running evaluation.

Format of `eval_set.jsonl` — one JSON per line:
```json
{"idx": 1251, "problem": "...", "solution": "..."}
```

Format of each translation cache file (e.g. `translations/local_no_glos.jsonl`):
```json
{"idx": 1251, "problem_pl": "...", "solution_pl": "..."}
```

## 3. Run the evaluation

```bash
python evaluation/run_eval.py
```

### All options

| Argument | Default | Description |
|---|---|---|
| `--num-examples` | all | Number of examples to use from `eval_set.jsonl` |
| `--local-model` | `google/gemma-4-E4B-it` | Local path or HuggingFace Hub ID for base model, or `none` to skip |
| `--finetuned-model` | `Igor-S-666/gemma4-math-translation-2026-06-02_10.36.07` | Local path or HuggingFace Hub ID for finetuned model, or `none` to skip |
| `--eval-only` | off | **Skip all model loading and translation** — compute metrics from existing caches only |
| `--translate-only` | off | Only run translations + timing, skip all metric computation |
| `--skip-api` | off | Skip the API model (use when no PCSS API key is available) |
| `--run-comet` | off | Run Metric 2: COMET QE (currently broken — see note above) |
| `--run-llm-judge` | off | Run Metric 3: LLM-as-judge scoring |
| `--judge-model` | `gpt-oss_120b` | Model to use as LLM judge |

### Common examples

```bash
# Compute all metrics from already-cached translations (no GPU needed)
python evaluation/run_eval.py --eval-only

# Compute metrics + LLM judge from cache
python evaluation/run_eval.py --eval-only --run-llm-judge

# Translate with local + finetuned models only (no API), then compute metrics
python evaluation/run_eval.py --skip-api

# Translate only (no metrics), skip API
python evaluation/run_eval.py --translate-only --skip-api

# Resume interrupted translation run
python evaluation/run_eval.py --translate-only --skip-api
```

---

## 4. Output

Results are saved to `eval_results/` (gitignored):

```
eval_results/
├── translations/                   # cached translations (one JSONL per config)
│   ├── api_no_glos.jsonl
│   ├── api_glos.jsonl
│   ├── local_no_glos.jsonl
│   ├── local_glos.jsonl
│   ├── finetuned_no_glos.jsonl
│   └── finetuned_glos.jsonl
├── metric_0/
│   └── metric_0_avg_time.csv       # avg seconds per example (fresh runs only)
├── metric_1/
│   ├── metric_1_problems.csv
│   └── metric_1_solutions.csv
├── metric_2/
│   ├── scores/                     # per-example COMET scores (cached)
│   ├── metric_2_problems.csv
│   └── metric_2_solutions.csv
└── metric_3/
    ├── scores/                     # per-example judge scores (cached)
    ├── metric_3_problems.csv
    └── metric_3_solutions.csv
```

---

## 5. Metrics

### Metric 0 — Average translation time

Saved only when translations are computed fresh (not read from cache).
Re-running on already-translated examples skips timing output.

### Metric 1 — LaTeX preservation

Percentage of `$...$` and `$$...$$` math segments from the source that appear
identically in the translation (after normalizing `\_` → `_` and `a_11` → `a_{11}`).
Reported separately for problems and solutions.

Always computed (unless `--translate-only`).

### Metric 2 — COMET QE (`--run-comet`)

Reference-free translation quality estimation using `Unbabel/wmt22-cometkiwi-da`.

**Currently not usable** — `unbabel-comet` requires `transformers<5.0` which conflicts
with Gemma 4 support. Do not install `unbabel-comet` in the same environment.

### Metric 3 — LLM judge (`--run-llm-judge`)

Uses `gpt-oss_120b` via the PCSS API to rate each translation on 5 criteria (1–5 scale):
`mathematical_accuracy`, `terminology`, `grammar`, `naturalness`, `completeness`.
One API call per example per config. Requires `PCSS_API_KEY` in `.env`.

Scores are cached per example so interrupted runs are resumable.

---

## 6. Resuming interrupted runs

Translations and metric scores are written to disk immediately after each example.
Re-run the same command to continue — completed examples are skipped automatically.

---

## 6. Hardware requirements

- **GPU with ~8–10 GB VRAM** required to run local or finetuned models (4-bit NF4 quantization)
- API model (`--skip-api` off) requires no local GPU
- Models are loaded sequentially (local → finetuned) to avoid holding two in VRAM at once

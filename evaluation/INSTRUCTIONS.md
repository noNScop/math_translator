# Evaluation — How to Run

All commands must be run from the **project root** (`Trump_GPT/`).

---

## 1. Setup

### Environment variables

Create a `.env` file at the project root:

```
PCSS_API_KEY=your_token_here
PCSS_BASE_URL=https://llm.hpc.psnc.pl/v1/chat/completions
PCSS_MODEL=llama3.3:70b
```

`PCSS_BASE_URL` and `PCSS_MODEL` are optional — the values above are the defaults.

### Download the local model

```bash
python evaluation/download_model.py
```

This downloads `google/gemma-4-E4B-it` to `models/google/gemma-4-E4B-it`.
If the folder already exists and is non-empty, the download is skipped.

---

## 2. Run the evaluation

```bash
python evaluation/run_eval.py
```

### Options

| Argument | Default | Description |
|---|---|---|
| `--num-examples` | `100` | Number of examples to translate and evaluate |
| `--dataset` | `data/ready_dataset.jsonl` | Path to the source dataset |
| `--local-model` | `models/google/gemma-4-E4B-it` | Path to local model, or `none` to skip |
| `--finetuned-model` | `models/google/gemma-4-E4B-it_finetuned` | Path to finetuned model, or `none` to skip |

### Examples

```bash
# API model only (no GPU needed)
python evaluation/run_eval.py --local-model none --finetuned-model none

# Quick smoke test with 5 examples
python evaluation/run_eval.py --num-examples 5 --local-model none --finetuned-model none

# Full run with all 3 models
python evaluation/run_eval.py --num-examples 100

# Custom dataset
python evaluation/run_eval.py --dataset data/my_dataset.jsonl
```

---

## 3. Output

Results are saved to `eval_results/`:

```
eval_results/
├── translations/          # cached translations (one JSONL per config)
│   ├── api_no_glos.jsonl
│   ├── api_glos.jsonl
│   ├── local_no_glos.jsonl
│   ├── local_glos.jsonl
│   ├── finetuned_no_glos.jsonl
│   └── finetuned_glos.jsonl
├── metric_0/
│   └── metric_0_avg_time.csv    # avg seconds per example (only on fresh runs)
└── metric_1/
    ├── metric_1_problems.csv    # LaTeX preservation score — problems
    └── metric_1_solutions.csv   # LaTeX preservation score — solutions
```

### Metric 0 — Average translation time

Saved only when translations are computed fresh (not from cache).
If you re-run the evaluation on already-translated examples, timing is skipped.

### Metric 1 — LaTeX preservation

Percentage of `$...$` and `$$...$$` math segments from the source that appear
identically in the translation. Reported separately for problems and solutions.

Tables have 3 rows (models) × 2 columns (without / with glossary). Missing models
(skipped or not yet finetuned) show `N/A`.

---

## 4. Resuming interrupted runs

Translations are cached immediately after each example. If a run is interrupted,
simply re-run the same command — already-translated examples are skipped automatically.

---

## 5. Adding new metrics

Add a new file in `evaluation/metrics/` following the pattern of
`latex_preservation.py`. Then call it in `run_eval.py` alongside the existing
`compute_metric_1` call and add a corresponding `save_csv` output.

Stubs for the next two metrics are already in place:
- `evaluation/metrics/comet_score.py` — Metric 2 (COMET)
- `evaluation/metrics/llm_judge.py` — Metric 3 (LLM-as-judge, see also `src/translation_eval.ipynb`)

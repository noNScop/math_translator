# Math Translator

Finetuning a small multilingual model to translate math problems (with LaTeX) from **English to Polish** — cheaply, quickly, and at a quality close to much larger models.

## What & why

We needed a large Polish math-translation dataset for a separate project (finetuning the Polish LLM **Bielik** on math benchmarks), but none existed. So we translated an existing English dataset (AI-MO, ~900k problems) into Polish.

Large API models translated well but were too slow and expensive to run over the whole dataset; small local models were fast and cheap but low-quality. Our solution: **finetune `gemma-4-E4B-it` (LoRA, 3 epochs)** on 1000 examples pre-translated by `llama3.3:70b`, plus a **glossary system** that suggests correct Polish math terminology in the prompt.

- **Finetuned model:** [Igor-S-666/gemma4-math-translation](https://huggingface.co/Igor-S-666/gemma4-math-translation-2026-06-02_10.36.07)

## Key results

Evaluated on 300 held-out examples across 4 metrics (LaTeX preservation, COMET-Kiwi, LLM-as-judge, time):

- Finetuning raised **LaTeX preservation from 50.7% → 89.0%** and closed most of the quality gap to the 70B "teacher" model.
- The **glossary** gave a small but consistent boost on the LLM-judge metric, clearest for **terminology**.
- Self-hosted, our finetuned model reaches comparable quality to commercial translation APIs at roughly **12× faster** and **5× cheaper**.

### **Full details, tables, and plots are in [Report.md](Report.md)**.


## Project structure

```
├── data/          # datasets (DVC-tracked): raw, processed, glossary sources, translations
├── models/        # trained models (DVC-tracked)
├── src/           # data prep + training
│   ├── fine-tuning/   # LoRA/SFT training scripts
│   ├── glos_api/      # glossary-augmented translation pipeline
│   └── glossary/      # glossary building
├── evaluation/    # eval harness (run_eval.py), metrics, plotting notebooks
├── eval_results/  # evaluation outputs (DVC-tracked)
├── assets/        # images used in the report
├── Report.md      # full project report
└── INSTRUCTIONS.md # setup + how to run
```

## Getting started

Setup, DVC data pull, and how to run training and evaluation are documented in **[INSTRUCTIONS.md](INSTRUCTIONS.md)**.



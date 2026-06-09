#!/usr/bin/env python3
"""
run_eval.py — Evaluate math translation models.

Metrics:
  0 — per-example translation time (seconds)
  1 — per-example LaTeX preservation score (0.0–1.0)
  2 — per-example COMET QE score  (opt-in: --run-comet)
  3 — per-example LLM-as-judge scores  (opt-in: --run-llm-judge)

Source dataset:  eval_results/eval_set.jsonl  (English source examples)
Translations:    eval_results/translations/   (one JSONL per model/glossary config)
Output:          eval_results/metric_X.jsonl  (one line per example, all 6 configs)

Must be run from the project root:
    python evaluation/run_eval.py --eval-only
    python evaluation/run_eval.py --translate-only --skip-api
    python evaluation/run_eval.py --run-llm-judge
"""
import argparse
import json
import sys
import time
import gc
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results"
EVAL_SET_PATH    = EVAL_RESULTS_DIR / "eval_set.jsonl"
TRANSLATIONS_DIR = EVAL_RESULTS_DIR / "translations"
METRICS_DIR      = EVAL_RESULTS_DIR / "metrics"
METRIC2_DIR      = METRICS_DIR / "metric_2"
METRIC3_DIR      = METRICS_DIR / "metric_3"

MODEL_LABELS = {
    "api":       "llama3.3:70b (API)",
    "local":     "gemma-4-E4B-it (local)",
    "finetuned": "gemma-4-E4B-it (finetuned)",
}

CONFIGS = [f"{m}_{s}" for m in ["api", "local", "finetuned"] for s in ["no_glos", "glos"]]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate math translation models")
    p.add_argument(
        "--num-examples", type=int, default=None,
        help="Number of examples to use (default: all in eval_set.jsonl)",
    )
    p.add_argument(
        "--local-model", type=str,
        default="google/gemma-4-E4B-it",
        help="Local path or HuggingFace Hub ID for base local model, or 'none' to skip",
    )
    p.add_argument(
        "--finetuned-model", type=str,
        default="Igor-S-666/gemma4-math-translation-2026-06-02_10.36.07",
        help="Local path or HuggingFace Hub ID for finetuned model, or 'none' to skip",
    )
    p.add_argument(
        "--eval-only", action="store_true",
        help="Skip all translation/model loading; compute metrics from cached translations only",
    )
    p.add_argument(
        "--translate-only", action="store_true",
        help="Only run translations (with timing), skip all metric computation",
    )
    p.add_argument(
        "--skip-api", action="store_true",
        help="Skip the API model (useful when no PCSS API key is available)",
    )
    p.add_argument(
        "--run-latex", action="store_true",
        help="Run Metric 1: LaTeX preservation score",
    )
    p.add_argument(
        "--run-comet", action="store_true",
        help="Run Metric 2: COMET QE score (requires venv_comet)",
    )
    p.add_argument(
        "--run-llm-judge", action="store_true",
        help="Run Metric 3: LLM-as-judge score",
    )
    p.add_argument(
        "--judge-model", type=str, default="gpt-oss_120b",
        help="Model to use as LLM judge (default: gpt-oss_120b)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_eval_set(n: int | None = None) -> list[dict]:
    """Load English source examples from eval_results/eval_set.jsonl."""
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(
            f"{EVAL_SET_PATH} not found.\n"
            "Create it by extracting English source examples into eval_results/eval_set.jsonl\n"
            "Format: one JSON per line with keys: idx, problem, solution"
        )
    records = []
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if n is not None and len(records) >= n:
                break
    print(f"Loaded {len(records)} examples from {EVAL_SET_PATH}")
    return records


# ---------------------------------------------------------------------------
# Translation cache
# ---------------------------------------------------------------------------

def _cache_file(config_key: str) -> Path:
    TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)
    return TRANSLATIONS_DIR / f"{config_key}.jsonl"


def load_cache(config_key: str) -> dict[int, dict]:
    path = _cache_file(config_key)
    if not path.exists():
        return {}
    cache = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                cache[rec["idx"]] = rec
    return cache


def append_to_cache(config_key: str, idx: int, problem_pl: str, solution_pl: str) -> None:
    with open(_cache_file(config_key), "a", encoding="utf-8") as f:
        f.write(json.dumps({"idx": idx, "problem_pl": problem_pl, "solution_pl": solution_pl}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Translation runner
# ---------------------------------------------------------------------------

def run_translations(
    dataset: list[dict],
    config_key: str,
    use_glossary: bool,
    llm_fn,
) -> tuple[dict[int, dict], dict[int, float]]:
    """
    Returns (translations_cache, timing).
    timing maps idx -> seconds for freshly-translated examples only.
    Empty dict when cache was not empty at start (pre-cached timing is meaningless).
    """
    from translate import translate_item

    cache = load_cache(config_key)
    missing = [item for item in dataset if item["idx"] not in cache]
    was_empty = len(cache) == 0

    if missing:
        print(f"  [{config_key}] Translating {len(missing)} examples (glossary={use_glossary}) ...")
    else:
        print(f"  [{config_key}] All {len(dataset)} examples already cached.")

    timing: dict[int, float] = {}
    for item in missing:
        idx = item["idx"]
        try:
            t0 = time.perf_counter()
            problem_pl, solution_pl = translate_item(
                problem_en=item["problem"],
                solution_en=item["solution"],
                use_glossary=use_glossary,
                llm_fn=llm_fn,
            )
            elapsed = time.perf_counter() - t0
            if was_empty:
                timing[idx] = elapsed
            append_to_cache(config_key, idx, problem_pl, solution_pl)
            cache[idx] = {"idx": idx, "problem_pl": problem_pl, "solution_pl": solution_pl}
            print(f"    [{config_key}] {idx} ✓")
        except Exception as e:
            print(f"    [{config_key}] {idx} ✗ {e}")

    return cache, timing


# ---------------------------------------------------------------------------
# Metric 1
# ---------------------------------------------------------------------------

def compute_metric_1(
    dataset: list[dict],
    translations: dict[int, dict],
) -> dict[int, dict]:
    """Returns {idx: {"problem": score, "solution": score}} for each translated example."""
    from metrics.latex_preservation import latex_preservation_score

    result: dict[int, dict] = {}
    for item in dataset:
        t = translations.get(item["idx"])
        if t is None:
            continue
        result[item["idx"]] = {
            "problem":  latex_preservation_score(item["problem"],  t["problem_pl"]),
            "solution": latex_preservation_score(item["solution"], t["solution_pl"]),
        }
    return result


# ---------------------------------------------------------------------------
# Score cache (shared by metrics 2 and 3)
# ---------------------------------------------------------------------------

def _score_cache_file(metric_dir: Path, config_key: str) -> Path:
    scores_dir = metric_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    return scores_dir / f"{config_key}.jsonl"


def load_score_cache(metric_dir: Path, config_key: str) -> dict[int, dict]:
    path = _score_cache_file(metric_dir, config_key)
    if not path.exists():
        return {}
    cache = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                cache[rec["idx"]] = rec
    return cache


def append_to_score_cache(metric_dir: Path, config_key: str, record: dict) -> None:
    with open(_score_cache_file(metric_dir, config_key), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Metric 2 — COMET
# ---------------------------------------------------------------------------

def compute_metric_2(
    dataset: list[dict],
    translations: dict[int, dict],
    config_key: str,
    comet_model,
) -> dict[int, dict]:
    """Returns {idx: {"problem": score, "solution": score}}."""
    from metrics.comet_score import score_batch

    cache = load_score_cache(METRIC2_DIR, config_key)
    missing = [item for item in dataset if item["idx"] not in cache and item["idx"] in translations]

    if missing:
        print(f"  [comet/{config_key}] Scoring {len(missing)} examples ...")
        prob_scores = score_batch([item["problem"] for item in missing],
                                  [translations[item["idx"]]["problem_pl"] for item in missing],
                                  comet_model)
        sol_scores  = score_batch([item["solution"] for item in missing],
                                  [translations[item["idx"]]["solution_pl"] for item in missing],
                                  comet_model)
        for item, ps, ss in zip(missing, prob_scores, sol_scores):
            rec = {"idx": item["idx"], "problem_score": ps, "solution_score": ss}
            append_to_score_cache(METRIC2_DIR, config_key, rec)
            cache[item["idx"]] = rec

    return {
        item["idx"]: {"problem": cache[item["idx"]]["problem_score"],
                      "solution": cache[item["idx"]]["solution_score"]}
        for item in dataset if item["idx"] in cache
    }


# ---------------------------------------------------------------------------
# Metric 3 — LLM judge
# ---------------------------------------------------------------------------

def compute_metric_3(
    dataset: list[dict],
    translations: dict[int, dict],
    config_key: str,
    judge_model: str,
) -> dict[int, dict]:
    """Returns {idx: {"problem": {criteria...}, "solution": {criteria...}}}."""
    from metrics.llm_judge import judge_single

    cache = load_score_cache(METRIC3_DIR, config_key)
    missing = [item for item in dataset if item["idx"] not in cache and item["idx"] in translations]

    if missing:
        print(f"  [judge/{config_key}] Scoring {len(missing)} examples with {judge_model} ...")

    for item in missing:
        idx = item["idx"]
        t = translations[idx]
        prob_scores = judge_single(item["problem"],  t["problem_pl"],  judge_model)
        sol_scores  = judge_single(item["solution"], t["solution_pl"], judge_model)
        if prob_scores is None or sol_scores is None:
            print(f"    [{config_key}] idx={idx} ✗ parse failed, skipping")
            continue
        rec = {"idx": idx, "problem_scores": prob_scores, "solution_scores": sol_scores}
        append_to_score_cache(METRIC3_DIR, config_key, rec)
        cache[idx] = rec
        print(f"    [{config_key}] idx={idx} ✓")

    return {
        item["idx"]: {"problem": cache[item["idx"]]["problem_scores"],
                      "solution": cache[item["idx"]]["solution_scores"]}
        for item in dataset if item["idx"] in cache
    }


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------

def _write_metric_jsonl(path: Path, dataset: list[dict], scores_by_config: dict[str, dict]) -> None:
    """One line per example, all 6 config scores merged. null for missing configs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            idx = item["idx"]
            rec = {"idx": idx}
            for config_key in CONFIGS:
                rec[config_key] = scores_by_config.get(config_key, {}).get(idx)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Console summary tables (stdout only, no files)
# ---------------------------------------------------------------------------

def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _avg_ps(scores_by_config: dict[str, dict[int, dict]], key: str) -> dict:
    """Compute per-model/suffix averages of problem or solution scores for display."""
    result: dict[str, dict] = {}
    for model_key in MODEL_LABELS:
        result[model_key] = {}
        for suffix in ["no_glos", "glos"]:
            vals = [v[key] for v in scores_by_config.get(f"{model_key}_{suffix}", {}).values()
                    if v is not None and key in v]
            result[model_key][suffix] = _avg(vals)
    return result


def _avg_judge(scores_by_config: dict[str, dict[int, dict]], key: str) -> dict:
    result: dict[str, dict] = {}
    for model_key in MODEL_LABELS:
        result[model_key] = {}
        for suffix in ["no_glos", "glos"]:
            all_vals = [v for s in scores_by_config.get(f"{model_key}_{suffix}", {}).values()
                        if s is not None and key in s
                        for v in s[key].values()]
            result[model_key][suffix] = _avg(all_vals)
    return result


def _avg_timing(timing_by_config: dict[str, dict[int, float]]) -> dict:
    result: dict[str, dict] = {}
    for model_key in MODEL_LABELS:
        result[model_key] = {}
        for suffix in ["no_glos", "glos"]:
            vals = list(timing_by_config.get(f"{model_key}_{suffix}", {}).values())
            result[model_key][suffix] = _avg(vals)
    return result


def _fmt_pct(v: float | None) -> str:
    return "N/A" if v is None else f"{v * 100:.1f}%"


def _fmt_score(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.3f}"


def _fmt_time(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.1f}s"


def _print_table(avgs: dict, title: str, fmt_fn) -> None:
    header = f"\n{title}"
    print(header)
    print("-" * max(len(header), 67))
    print(f"{'Model':<35} | {'No Glossary':>12} | {'With Glossary':>13}")
    print("-" * 67)
    for key, label in MODEL_LABELS.items():
        row = avgs.get(key, {"no_glos": None, "glos": None})
        print(f"{label:<35} | {fmt_fn(row.get('no_glos')):>12} | {fmt_fn(row.get('glos')):>13}")


# ---------------------------------------------------------------------------
# Compute all metrics
# ---------------------------------------------------------------------------

def _compute_all_metrics(dataset: list[dict], args: argparse.Namespace) -> None:
    """Compute and save metric_1–3 JSONL files from cached translations."""
    all_translations: dict[str, dict[int, dict]] = {
        ck: load_cache(ck)
        for ck in CONFIGS
        if load_cache(ck)
    }

    # --- Metric 1: LaTeX preservation ---
    if args.run_latex:
        m1: dict[str, dict[int, dict]] = {
            ck: compute_metric_1(dataset, trans)
            for ck, trans in all_translations.items()
        }
        _write_metric_jsonl(METRICS_DIR / "metric_1.jsonl", dataset, m1)
        _print_table(_avg_ps(m1, "problem"),  "Metric 1 — LaTeX Preservation (problems, avg)",  _fmt_pct)
        _print_table(_avg_ps(m1, "solution"), "Metric 1 — LaTeX Preservation (solutions, avg)", _fmt_pct)

    # --- Metric 2: COMET ---
    if args.run_comet:
        print("\n=== Metric 2: COMET ===")
        from metrics.comet_score import load_comet_model
        comet_model = load_comet_model()
        m2: dict[str, dict[int, dict]] = {
            ck: compute_metric_2(dataset, trans, ck, comet_model)
            for ck, trans in all_translations.items()
        }
        _write_metric_jsonl(METRICS_DIR / "metric_2.jsonl", dataset, m2)
        _print_table(_avg_ps(m2, "problem"),  "Metric 2 — COMET (problems, avg)",  _fmt_score)
        _print_table(_avg_ps(m2, "solution"), "Metric 2 — COMET (solutions, avg)", _fmt_score)

    # --- Metric 3: LLM judge ---
    if args.run_llm_judge:
        print(f"\n=== Metric 3: LLM judge ({args.judge_model}) ===")
        m3: dict[str, dict[int, dict]] = {
            ck: compute_metric_3(dataset, trans, ck, args.judge_model)
            for ck, trans in all_translations.items()
        }
        _write_metric_jsonl(METRICS_DIR / "metric_3.jsonl", dataset, m3)
        _print_table(_avg_judge(m3, "problem"),  "Metric 3 — LLM Judge (problems, avg)",  _fmt_score)
        _print_table(_avg_judge(m3, "solution"), "Metric 3 — LLM Judge (solutions, avg)", _fmt_score)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    dataset = load_eval_set(args.num_examples)

    # --eval-only: skip all model loading and translation, just compute metrics
    if args.eval_only:
        print("[INFO] --eval-only: reading translations from cache, skipping model loading")
        _compute_all_metrics(dataset, args)
        print(f"\nDone. Results in {EVAL_RESULTS_DIR}/")
        return

    def _model_available(s: str) -> bool:
        if s.lower() == "none":
            return False
        if Path(s).exists():
            return True
        return "/" in s and not s.startswith("/")

    use_local     = _model_available(args.local_model)
    use_finetuned = _model_available(args.finetuned_model)

    if not use_local:
        print(f"[INFO] Skipping local model ({args.local_model})")
    if not use_finetuned:
        print(f"[INFO] Skipping finetuned model ({args.finetuned_model})")

    timing_by_config: dict[str, dict[int, float]] = {}

    # --- API model ---
    if args.skip_api:
        print("\n[INFO] Skipping API model (--skip-api)")
    else:
        print("\n=== API model ===")
        for use_glossary in [False, True]:
            suffix = "glos" if use_glossary else "no_glos"
            ck = f"api_{suffix}"
            _, t = run_translations(dataset, ck, use_glossary, llm_fn=None)
            timing_by_config[ck] = t

    # --- Local model ---
    if use_local:
        print("\n=== Local model ===")
        from translate import load_local_model
        local_wrapper = load_local_model(Path(args.local_model))
        for use_glossary in [False, True]:
            suffix = "glos" if use_glossary else "no_glos"
            ck = f"local_{suffix}"
            _, t = run_translations(dataset, ck, use_glossary, llm_fn=local_wrapper)
            timing_by_config[ck] = t
        del local_wrapper
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # --- Finetuned model ---
    if use_finetuned:
        print("\n=== Finetuned model ===")
        from translate import load_finetuned_model
        ft_wrapper = load_finetuned_model(Path(args.finetuned_model))
        for use_glossary in [False, True]:
            suffix = "glos" if use_glossary else "no_glos"
            ck = f"finetuned_{suffix}"
            _, t = run_translations(dataset, ck, use_glossary, llm_fn=ft_wrapper)
            timing_by_config[ck] = t
        del ft_wrapper
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # --- Save timing ---
    has_timing = any(t_dict for t_dict in timing_by_config.values())
    if has_timing:
        _write_metric_jsonl(METRICS_DIR / "metric_0.jsonl", dataset, timing_by_config)
        _print_table(_avg_timing(timing_by_config), "Metric 0 — Avg translation time (seconds)", _fmt_time)
    else:
        print("\n[Metric 0] All translations were already cached — timing not saved.")

    if args.translate_only:
        print(f"\nDone. Translations saved to {TRANSLATIONS_DIR}/")
        return

    # --- Compute metrics ---
    _compute_all_metrics(dataset, args)
    print(f"\nDone. Results in {EVAL_RESULTS_DIR}/")


if __name__ == "__main__":
    main()

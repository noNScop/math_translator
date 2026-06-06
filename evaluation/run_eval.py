#!/usr/bin/env python3
"""
run_eval.py — Evaluate math translation models.

Metrics:
  0 — avg translation time (auto, only on fresh runs)
  1 — LaTeX preservation score
  2 — COMET quality estimation  (opt-in: --run-comet)
  3 — LLM-as-judge score        (opt-in: --run-llm-judge)

Must be run from the project root:
    python evaluation/run_eval.py
    python evaluation/run_eval.py --run-comet --run-llm-judge
    python evaluation/run_eval.py --local-model none --finetuned-model none
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

TRANSLATIONS_DIR = PROJECT_ROOT / "eval_results" / "translations"
METRIC1_DIR = PROJECT_ROOT / "eval_results" / "metric_1"
METRIC2_DIR = PROJECT_ROOT / "eval_results" / "metric_2"
METRIC3_DIR = PROJECT_ROOT / "eval_results" / "metric_3"

MODEL_LABELS = {
    "api":       "llama3.3:70b (API)",
    "local":     "gemma-4-E4B-it (local)",
    "finetuned": "gemma-4-E4B-it (finetuned)",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate math translation models")
    p.add_argument(
        "--num-examples", type=int, default=100,
        help="Number of examples to translate and evaluate (default: 100)",
    )
    p.add_argument(
        "--dataset", type=Path,
        default=PROJECT_ROOT / "data" / "translations_output.jsonl",
        help="Path to the source dataset JSONL (supports both ready_dataset.jsonl and translations_output.jsonl formats)",
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
        "--translate-only", action="store_true",
        help="Only run translations (with timing), skip all metric computation",
    )
    p.add_argument(
        "--skip-api", action="store_true",
        help="Skip the API model (useful when no PCSS API key is available)",
    )
    p.add_argument(
        "--start-from", type=int, default=0,
        help="Start from this dataset index (default: 0)",
    )
    p.add_argument(
        "--run-comet", action="store_true",
        help="Run Metric 2: COMET QE score (requires pip install unbabel-comet)",
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

def _normalize_record(rec: dict) -> dict:
    """Normalize translations_output.jsonl format to internal format.
    translations_output.jsonl uses: id, problem_en, solution_en
    ready_dataset.jsonl uses:       idx, problem, solution
    """
    if "problem_en" in rec:
        rec["problem"] = rec.pop("problem_en")
    if "solution_en" in rec:
        rec["solution"] = rec.pop("solution_en")
    if "id" in rec and "idx" not in rec:
        rec["idx"] = rec.pop("id")
    return rec


def load_dataset(path: Path, n: int, start_from: int = 0) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = _normalize_record(json.loads(line))
            if rec["idx"] < start_from:
                continue
            records.append(rec)
            if len(records) >= n:
                break
    print(f"Loaded {len(records)} examples from {path} (starting from idx {start_from})")
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
) -> tuple[dict[int, dict], float | None]:
    """
    Returns (translations_cache, avg_seconds_per_example).
    avg_seconds is None when the cache was not empty at the start
    (timing pre-cached reads is meaningless).
    """
    from translate import translate_item

    cache = load_cache(config_key)
    missing = [item for item in dataset if item["idx"] not in cache]
    was_empty = len(cache) == 0

    if missing:
        print(f"  [{config_key}] Translating {len(missing)} examples (glossary={use_glossary}) ...")
    else:
        print(f"  [{config_key}] All {len(dataset)} examples already cached.")

    elapsed_times: list[float] = []
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
            elapsed_times.append(time.perf_counter() - t0)
            append_to_cache(config_key, idx, problem_pl, solution_pl)
            cache[idx] = {"idx": idx, "problem_pl": problem_pl, "solution_pl": solution_pl}
            print(f"    [{config_key}] {idx} ✓")
        except Exception as e:
            print(f"    [{config_key}] {idx} ✗ {e}")

    avg_time = (sum(elapsed_times) / len(elapsed_times)) if (was_empty and elapsed_times) else None
    return cache, avg_time


# ---------------------------------------------------------------------------
# Metric 1
# ---------------------------------------------------------------------------

def compute_metric_1(
    dataset: list[dict],
    translations: dict[int, dict],
) -> tuple[float | None, float | None]:
    from metrics.latex_preservation import latex_preservation_score

    prob_scores, sol_scores = [], []
    for item in dataset:
        t = translations.get(item["idx"])
        if t is None:
            continue
        prob_scores.append(latex_preservation_score(item["problem"], t["problem_pl"]))
        sol_scores.append(latex_preservation_score(item["solution"], t["solution_pl"]))

    prob_mean = sum(prob_scores) / len(prob_scores) if prob_scores else None
    sol_mean  = sum(sol_scores)  / len(sol_scores)  if sol_scores  else None
    return prob_mean, sol_mean


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
) -> tuple[float | None, float | None]:
    from metrics.comet_score import score_batch

    cache = load_score_cache(METRIC2_DIR, config_key)
    missing = [item for item in dataset if item["idx"] not in cache and item["idx"] in translations]

    if missing:
        print(f"  [comet/{config_key}] Scoring {len(missing)} examples ...")
        prob_sources = [item["problem"] for item in missing]
        prob_trans   = [translations[item["idx"]]["problem_pl"] for item in missing]
        sol_sources  = [item["solution"] for item in missing]
        sol_trans    = [translations[item["idx"]]["solution_pl"] for item in missing]

        prob_scores = score_batch(prob_sources, prob_trans, comet_model)
        sol_scores  = score_batch(sol_sources,  sol_trans,  comet_model)

        for item, ps, ss in zip(missing, prob_scores, sol_scores):
            rec = {"idx": item["idx"], "problem_score": ps, "solution_score": ss}
            append_to_score_cache(METRIC2_DIR, config_key, rec)
            cache[item["idx"]] = rec

    prob_vals = [cache[item["idx"]]["problem_score"]  for item in dataset if item["idx"] in cache]
    sol_vals  = [cache[item["idx"]]["solution_score"] for item in dataset if item["idx"] in cache]
    p = sum(prob_vals) / len(prob_vals) if prob_vals else None
    s = sum(sol_vals)  / len(sol_vals)  if sol_vals  else None
    return p, s


# ---------------------------------------------------------------------------
# Metric 3 — LLM judge
# ---------------------------------------------------------------------------

def compute_metric_3(
    dataset: list[dict],
    translations: dict[int, dict],
    config_key: str,
    judge_model: str,
) -> tuple[float | None, float | None]:
    from metrics.llm_judge import judge_single, mean_score

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

    prob_lists = [list(cache[item["idx"]]["problem_scores"].values())  for item in dataset if item["idx"] in cache]
    sol_lists  = [list(cache[item["idx"]]["solution_scores"].values()) for item in dataset if item["idx"] in cache]
    p = (sum(sum(r) / len(r) for r in prob_lists) / len(prob_lists)) if prob_lists else None
    s = (sum(sum(r) / len(r) for r in sol_lists)  / len(sol_lists))  if sol_lists  else None
    return p, s


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(score: float | None) -> str:
    return "N/A" if score is None else f"{score * 100:.1f}%"


def print_table(scores: dict, title: str, fmt_fn=_fmt) -> None:
    header = f"\n{title}"
    print(header)
    print("-" * len(header))
    print(f"{'Model':<35} | {'No Glossary':>12} | {'With Glossary':>13}")
    print("-" * 67)
    for key, label in MODEL_LABELS.items():
        row = scores.get(key, {"no_glos": None, "glos": None})
        print(f"{label:<35} | {fmt_fn(row['no_glos']):>12} | {fmt_fn(row['glos']):>13}")


def save_csv(scores: dict, filename: str, out_dir: Path, fmt_fn=_fmt) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Without Glossary", "With Glossary"])
        for key, label in MODEL_LABELS.items():
            row = scores.get(key, {"no_glos": None, "glos": None})
            writer.writerow([label, fmt_fn(row["no_glos"]), fmt_fn(row["glos"])])
    print(f"Saved {path}")


def _fmt_time(val: float | None) -> str:
    return "N/A" if val is None else f"{val:.1f}s"


def print_timing_table(timing: dict) -> None:
    header = "\nMetric 0 — Avg translation time per example"
    print(header)
    print("-" * len(header))
    print(f"{'Model':<35} | {'No Glossary':>12} | {'With Glossary':>13}")
    print("-" * 67)
    for key, label in MODEL_LABELS.items():
        row = timing.get(key, {"no_glos": None, "glos": None})
        print(f"{label:<35} | {_fmt_time(row['no_glos']):>12} | {_fmt_time(row['glos']):>13}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, args.num_examples, args.start_from)

    def _model_available(s: str) -> bool:
        if s.lower() == "none":
            return False
        if Path(s).exists():
            return True
        # HuggingFace Hub ID: "username/model-name" (not an absolute path)
        return "/" in s and not s.startswith("/")

    use_local     = _model_available(args.local_model)
    use_finetuned = _model_available(args.finetuned_model)

    if not use_local:
        print(f"[INFO] Skipping local model ({args.local_model})")
    if not use_finetuned:
        print(f"[INFO] Skipping finetuned model ({args.finetuned_model})")

    problem_scores: dict[str, dict] = {}
    solution_scores: dict[str, dict] = {}

    timing: dict[str, dict] = {}

    # --- API model ---
    if args.skip_api:
        print("\n[INFO] Skipping API model (--skip-api)")
        problem_scores["api"]  = {"no_glos": None, "glos": None}
        solution_scores["api"] = {"no_glos": None, "glos": None}
        timing["api"] = {"no_glos": None, "glos": None}
    else:
        print("\n=== API model ===")
        problem_scores["api"]  = {}
        solution_scores["api"] = {}
        timing["api"] = {}
        for use_glossary in [False, True]:
            suffix = "glos" if use_glossary else "no_glos"
            translations, avg_t = run_translations(dataset, f"api_{suffix}", use_glossary, llm_fn=None)
            if not args.translate_only:
                p, s = compute_metric_1(dataset, translations)
                problem_scores["api"][suffix]  = p
                solution_scores["api"][suffix] = s
            timing["api"][suffix] = avg_t

    # --- Local model ---
    if use_local:
        print("\n=== Local model ===")
        from translate import load_local_model
        local_wrapper = load_local_model(Path(args.local_model))
        problem_scores["local"]  = {}
        solution_scores["local"] = {}
        timing["local"] = {}
        for use_glossary in [False, True]:
            suffix = "glos" if use_glossary else "no_glos"
            translations, avg_t = run_translations(dataset, f"local_{suffix}", use_glossary, llm_fn=local_wrapper)
            if not args.translate_only:
                p, s = compute_metric_1(dataset, translations)
                problem_scores["local"][suffix]  = p
                solution_scores["local"][suffix] = s
            timing["local"][suffix] = avg_t
        del local_wrapper
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    else:
        problem_scores["local"]  = {"no_glos": None, "glos": None}
        solution_scores["local"] = {"no_glos": None, "glos": None}
        timing["local"] = {"no_glos": None, "glos": None}

    # --- Finetuned model ---
    if use_finetuned:
        print("\n=== Finetuned model ===")
        from translate import load_local_model
        ft_wrapper = load_local_model(Path(args.finetuned_model))
        problem_scores["finetuned"]  = {}
        solution_scores["finetuned"] = {}
        timing["finetuned"] = {}
        for use_glossary in [False, True]:
            suffix = "glos" if use_glossary else "no_glos"
            translations, avg_t = run_translations(dataset, f"finetuned_{suffix}", use_glossary, llm_fn=ft_wrapper)
            if not args.translate_only:
                p, s = compute_metric_1(dataset, translations)
                problem_scores["finetuned"][suffix]  = p
                solution_scores["finetuned"][suffix] = s
            timing["finetuned"][suffix] = avg_t
        del ft_wrapper
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    else:
        problem_scores["finetuned"]  = {"no_glos": None, "glos": None}
        solution_scores["finetuned"] = {"no_glos": None, "glos": None}
        timing["finetuned"] = {"no_glos": None, "glos": None}

    if args.translate_only:
        if has_timing := any(v is not None for row in timing.values() for v in row.values()):
            print_timing_table(timing)
            save_csv(timing, "metric_0_avg_time.csv", PROJECT_ROOT / "eval_results" / "metric_0", fmt_fn=_fmt_time)
        print(f"\nDone. Translations saved to {TRANSLATIONS_DIR}/")
        return

    # --- Collect all translations for metrics 2 & 3 ---
    all_translations: dict[str, dict[int, dict]] = {}
    for model_key in ["api", "local", "finetuned"]:
        for suffix in ["no_glos", "glos"]:
            config_key = f"{model_key}_{suffix}"
            cached = load_cache(config_key)
            if cached:
                all_translations[config_key] = cached

    # --- Metric 2: COMET ---
    comet_prob: dict[str, dict] = {k: {"no_glos": None, "glos": None} for k in MODEL_LABELS}
    comet_sol:  dict[str, dict] = {k: {"no_glos": None, "glos": None} for k in MODEL_LABELS}

    if args.run_comet:
        print("\n=== Metric 2: COMET ===")
        from metrics.comet_score import load_comet_model
        comet_model = load_comet_model()
        for model_key in MODEL_LABELS:
            comet_prob[model_key] = {}
            comet_sol[model_key]  = {}
            for suffix in ["no_glos", "glos"]:
                config_key = f"{model_key}_{suffix}"
                trans = all_translations.get(config_key, {})
                if not trans:
                    comet_prob[model_key][suffix] = None
                    comet_sol[model_key][suffix]  = None
                else:
                    p, s = compute_metric_2(dataset, trans, config_key, comet_model)
                    comet_prob[model_key][suffix] = p
                    comet_sol[model_key][suffix]  = s

    # --- Metric 3: LLM judge ---
    judge_prob: dict[str, dict] = {k: {"no_glos": None, "glos": None} for k in MODEL_LABELS}
    judge_sol:  dict[str, dict] = {k: {"no_glos": None, "glos": None} for k in MODEL_LABELS}

    if args.run_llm_judge:
        print(f"\n=== Metric 3: LLM judge ({args.judge_model}) ===")
        for model_key in MODEL_LABELS:
            judge_prob[model_key] = {}
            judge_sol[model_key]  = {}
            for suffix in ["no_glos", "glos"]:
                config_key = f"{model_key}_{suffix}"
                trans = all_translations.get(config_key, {})
                if not trans:
                    judge_prob[model_key][suffix] = None
                    judge_sol[model_key][suffix]  = None
                else:
                    p, s = compute_metric_3(dataset, trans, config_key, args.judge_model)
                    judge_prob[model_key][suffix] = p
                    judge_sol[model_key][suffix]  = s

    # --- Output ---
    METRIC0_DIR = PROJECT_ROOT / "eval_results" / "metric_0"

    def _fmt_score(v):
        return "N/A" if v is None else f"{v:.3f}"

    print_table(problem_scores,  "Metric 1 — LaTeX Preservation (problems)")
    print_table(solution_scores, "Metric 1 — LaTeX Preservation (solutions)")
    save_csv(problem_scores,  "metric_1_problems.csv",  METRIC1_DIR)
    save_csv(solution_scores, "metric_1_solutions.csv", METRIC1_DIR)

    if args.run_comet:
        print_table(comet_prob, "Metric 2 — COMET (problems)",  fmt_fn=_fmt_score)
        print_table(comet_sol,  "Metric 2 — COMET (solutions)", fmt_fn=_fmt_score)
        save_csv(comet_prob, "metric_2_problems.csv",  METRIC2_DIR, fmt_fn=_fmt_score)
        save_csv(comet_sol,  "metric_2_solutions.csv", METRIC2_DIR, fmt_fn=_fmt_score)

    if args.run_llm_judge:
        print_table(judge_prob, "Metric 3 — LLM Judge (problems)",  fmt_fn=_fmt_score)
        print_table(judge_sol,  "Metric 3 — LLM Judge (solutions)", fmt_fn=_fmt_score)
        save_csv(judge_prob, "metric_3_problems.csv",  METRIC3_DIR, fmt_fn=_fmt_score)
        save_csv(judge_sol,  "metric_3_solutions.csv", METRIC3_DIR, fmt_fn=_fmt_score)

    # Timing: only save rows where at least one value was freshly measured
    has_timing = any(
        v is not None
        for row in timing.values()
        for v in row.values()
    )
    if has_timing:
        print_timing_table(timing)
        save_csv(timing, "metric_0_avg_time.csv", METRIC0_DIR, fmt_fn=_fmt_time)
    else:
        print("\n[Metric 0] All translations were already cached — timing not saved.")

    print(f"\nDone. Results in {PROJECT_ROOT / 'eval_results'}/")


if __name__ == "__main__":
    main()

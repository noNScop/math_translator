"""
Metric 3: LLM-as-judge score.

Uses gpt-oss_120b via the PCSS API to rate each translation on 5 criteria
from 1 (very poor) to 5 (excellent). One API call per example per config.

Criteria (same as src/translation_eval.ipynb):
  mathematical_accuracy, terminology, grammar, naturalness, completeness
"""
import json
import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()

_BASE_URL = os.getenv("PCSS_BASE_URL", "https://llm.hpc.psnc.pl/v1/chat/completions")
_API_KEY  = os.getenv("PCSS_API_KEY", "")

CRITERIA = ["mathematical_accuracy", "terminology", "grammar", "naturalness", "completeness"]

JUDGE_SYSTEM_PROMPT = """\
You are an expert bilingual evaluator specializing in English-to-Polish mathematical translation.

Rate the Polish translation of the given English mathematical text on each of the following criteria \
using a scale from 1 (very poor) to 5 (excellent):

1. mathematical_accuracy — all mathematical steps, formulas, and logical structure are correctly preserved
2. terminology — correct Polish mathematical terminology is used
3. grammar — the Polish text is grammatically correct
4. naturalness — the Polish reads naturally for a native speaker
5. completeness — all content from the source is present in the translation

Respond with ONLY a valid JSON object, no explanation or extra text:
{"mathematical_accuracy": X, "terminology": X, "grammar": X, "naturalness": X, "completeness": X}\
"""

JUDGE_USER_PROMPT = """\
English source:
{source}

Polish translation:
{translation}\
"""


def _call_api(source: str, translation: str, model: str) -> str:
    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": JUDGE_USER_PROMPT.format(source=source, translation=translation)},
        ],
        "temperature": 0.0,
    }
    resp = requests.post(_BASE_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _parse_scores(text: str) -> dict | None:
    try:
        scores = json.loads(text)
        if all(k in scores for k in CRITERIA):
            return {k: int(scores[k]) for k in CRITERIA}
    except Exception:
        pass
    # Regex fallback: extract first {...} block
    match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if match:
        try:
            scores = json.loads(match.group())
            if all(k in scores for k in CRITERIA):
                return {k: int(scores[k]) for k in CRITERIA}
        except Exception:
            pass
    return None


def judge_single(
    source: str,
    translation: str,
    model: str = "gpt-oss_120b",
) -> dict | None:
    """Score one translation. Returns dict with 5 integer scores, or None on failure."""
    try:
        response = _call_api(source, translation, model)
        return _parse_scores(response)
    except Exception as e:
        print(f"    Judge API error: {e}")
        return None


def mean_score(scores_list: list[dict]) -> float:
    """Average all criteria across all examples."""
    all_vals = [v for s in scores_list for v in s.values()]
    return sum(all_vals) / len(all_vals) if all_vals else 0.0

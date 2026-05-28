from prompts import (
    TRANSLATE_SYSTEM_PROMPT,
    TRANSLATE_PROBLEM_PROMPT,
    TRANSLATE_SOLUTION_PROMPT,
)

from api import call_llm

import json
from pathlib import Path




DATA_FILE = Path("./data/ready_dataset.jsonl")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

print(f"Loaded {len(data)} examples")
print(data[0].keys())


def save_translation(
    output_file: Path,
    idx: int,
    task: dict,
    problem_pl: str,
    solution_pl: str,
):
    record = {
        "id": idx,
        "problem_en": task["problem"],
        "problem_pl": problem_pl,
        "solution_en": task["solution"],
        "solution_pl": solution_pl,
    }

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")




OUTPUT_FILE = Path("./data/translations_output_working.jsonl")


def get_processed_ids(output_file: Path) -> set:
    if not output_file.exists():
        return set()

    with open(output_file, "r", encoding="utf-8") as f:
        return {
            json.loads(line)["id"]
            for line in f
            if line.strip()
        }


processed = get_processed_ids(OUTPUT_FILE)

print(f"Already processed: {len(processed)} records")



NUM = 1

for i in range(min(NUM, len(data))):

    if i in processed:
        print(f"Skipping {i} (already done)")
        continue

    task = data[i]

    try:

        problem_pl = call_llm(
            TRANSLATE_SYSTEM_PROMPT,
            TRANSLATE_PROBLEM_PROMPT.format(
                text=task["problem"]
            )
        )

        solution_pl = call_llm(
            TRANSLATE_SYSTEM_PROMPT,
            TRANSLATE_SOLUTION_PROMPT.format(
                text=task["solution"]
            )
        )

        save_translation(
            OUTPUT_FILE,
            i,
            task,
            problem_pl,
            solution_pl,
        )

        print(f"[{i+1}/{len(data)}] ✓ saved")

    except Exception as e:
        print(f"[{i}] ✗ Error: {e} — skipping")
        continue
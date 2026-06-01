from prompts import (
    TRANSLATE_SYSTEM_PROMPT,
    TRANSLATE_PROBLEM_PROMPT,
    TRANSLATE_SOLUTION_PROMPT,
)

import json
from pathlib import Path

from config import CONFIG
from query_glos import translate_with_glossary
import time



DATA_FILE = Path(
    CONFIG["paths"]["dataset"]
)

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

print(f"Loaded {len(data)} examples")




OUTPUT_FILE = Path(
    CONFIG["paths"]["output"]
)




NUM = CONFIG["translation"]["num_examples"]

USE_GLOSSARY = CONFIG["glossary"]["enabled"]




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
        f.write(
            json.dumps(record, ensure_ascii=False)
            + "\n"
        )


def get_processed_ids(
    output_file: Path,
) -> set:

    if not output_file.exists():
        return set()

    with open(
        output_file,
        "r",
        encoding="utf-8"
    ) as f:

        return {
            json.loads(line)["id"]
            for line in f
            if line.strip()
        }




processed = get_processed_ids(
    OUTPUT_FILE
)

print(
    f"Already processed: "
    f"{len(processed)} records"
)



for i in range(
    min(NUM, len(data))
):

    if i in processed:
        print(
            f"Skipping {i} "
            f"(already done)"
        )
        continue

    task = data[i]

    try:
        start = time.perf_counter()
        problem_pl = (
            translate_with_glossary(
                text=task["problem"],
                system_prompt=TRANSLATE_SYSTEM_PROMPT,
                user_prompt_template=TRANSLATE_PROBLEM_PROMPT,
                use_glossary=USE_GLOSSARY,
            )
        )
        end = time.perf_counter()
        print(f"LLM call took {end - start:.4f} seconds")

        start = time.perf_counter()
        solution_pl = (
            translate_with_glossary(
                text=task["solution"],
                system_prompt=TRANSLATE_SYSTEM_PROMPT,
                user_prompt_template=TRANSLATE_SOLUTION_PROMPT,
                use_glossary=USE_GLOSSARY,
            )
        )
        end = time.perf_counter()
        print(f"LLM call took {end - start:.4f} seconds")

        save_translation(
            OUTPUT_FILE,
            i,
            task,
            problem_pl,
            solution_pl,
        )

        print(
            f"[{i+1}/{len(data)}] ✓ saved"
        )

    except Exception as e:

        print(
            f"[{i}] ✗ Error: "
            f"{e} — skipping"
        )

        continue
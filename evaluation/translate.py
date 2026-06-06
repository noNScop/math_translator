"""
Translation backends for the evaluation framework.

Three model types are supported:
  - API (llama3.3:70b via PCSS) — uses existing call_llm from src/glos_api/api.py
  - local (google/gemma-4-E4B-it, 4-bit quantized)
  - finetuned (same architecture, different weights)

All backends go through translate_with_glossary so glossary injection is uniform.
"""
import sys
from pathlib import Path

# Patch sys.path so bare imports inside src/glos_api resolve correctly.
# Note: config.py inside glos_api opens "src/glos_api/config.yaml" relative to CWD,
# so run_eval.py must be invoked from the project root.
_GLOS_API_DIR = Path(__file__).resolve().parents[1] / "src" / "glos_api"
if str(_GLOS_API_DIR) not in sys.path:
    sys.path.insert(0, str(_GLOS_API_DIR))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from query_glos import translate_with_glossary
from prompts import (
    TRANSLATE_SYSTEM_PROMPT,
    TRANSLATE_PROBLEM_PROMPT,
    TRANSLATE_SOLUTION_PROMPT,
)


class LocalModelWrapper:
    """Wraps a local HF causal LM to match the call_llm(system, user) -> str interface."""

    def __init__(self, model_path: str | Path, max_new_tokens: int = 2048):
        model_path = str(model_path)
        print(f"Loading local model from {model_path} ...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        print("Model ready.")

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)


def load_local_model(model_path: Path) -> LocalModelWrapper:
    return LocalModelWrapper(model_path)


def translate_item(
    problem_en: str,
    solution_en: str,
    use_glossary: bool,
    llm_fn=None,
) -> tuple[str, str]:
    """
    Translate a single problem+solution pair.
    llm_fn=None uses the API (call_llm); pass a LocalModelWrapper for local inference.
    Returns (problem_pl, solution_pl).
    """
    problem_pl = translate_with_glossary(
        text=problem_en,
        system_prompt=TRANSLATE_SYSTEM_PROMPT,
        user_prompt_template=TRANSLATE_PROBLEM_PROMPT,
        use_glossary=use_glossary,
        llm_fn=llm_fn,
    )
    solution_pl = translate_with_glossary(
        text=solution_en,
        system_prompt=TRANSLATE_SYSTEM_PROMPT,
        user_prompt_template=TRANSLATE_SOLUTION_PROMPT,
        use_glossary=use_glossary,
        llm_fn=llm_fn,
    )
    return problem_pl, solution_pl

import json
import os
import re
import sys
import glob
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

load_dotenv()

API_KEY = os.getenv("PCSS_API_KEY", "")
BASE_URL = os.getenv("PCSS_BASE_URL", "https://llm.hpc.psnc.pl/v1/chat/completions")
MODEL = os.getenv("PCSS_MODEL", "llama3.3:70b")


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call the LLM API (OpenAI-compatible).
    Swap BASE_URL / auth headers here when moving to PCSS.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    response = requests.post(
        BASE_URL,
        headers=headers,
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

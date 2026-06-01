import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import torch
from dotenv import load_dotenv

ROOT_PATH = Path(__file__).resolve().parents[0]
load_dotenv(ROOT_PATH / ".env", override=True)


@dataclass
class Config:
    # --- Secrets (from .env) ---
    PUSH_TO_HUB: bool = field(default_factory=lambda: os.getenv('PUSH_TO_HUB', 'false').lower() == 'true')
    LOG_TO_WANDB: bool = field(default_factory=lambda: os.getenv('LOG_TO_WANDB', 'false').lower() == 'true')
    HF_TOKEN: str = field(default_factory=lambda: os.getenv('HF_TOKEN', ''))
    WANDB_API_KEY: str = field(default_factory=lambda: os.getenv('WANDB_API_KEY', ''))
    # Valid values: "none" | "4b" | "8b". On H100 "none" is preferred (native bf16 is fast enough).
    QUANTIZATION: str = field(default_factory=lambda: os.getenv('QUANTIZATION', 'none').lower())

    # --- Model / Dataset ---
    BASE_MODEL: str = "google/gemma-4-E4B-it"  # replace with "meta/gemma-4-E4B-it" for actual training
    PROJECT_NAME: str = "gemma4-math-translation"
    HF_USER: str = "Igor-S-666"
    # Path to the local JSONL file with EN→PL translation pairs
    # default_factory=lambda: str(Path(__file__).resolve().parents[2] / "data" / "translations_output.jsonl")
    DATASET_PATH: str = field(
        default_factory=lambda: str(Path(__file__).resolve().parents[2] / "data" / "translations_output.jsonl")
    )

    # System prompt for the translation task
    SYSTEM_PROMPT: str = (
        """You are an expert mathematical translator specializing in English to Polish translation for olympiad and academic mathematics.

        STRICT RULES:
        1. Translate ALL natural language text from English to Polish
        2. Keep ALL LaTeX commands, formulas, and math expressions EXACTLY as they are — do not modify anything inside $...$ or $$...$$
        3. Keep variable names, labels, and point names unchanged (e.g. $A$, $B$, $ABC$, $f(x)$)
        4. Output ONLY the translated text — no explanations, no comments, no preamble
        5. Preserve all formatting, newlines, and structure from the original
        6. Use correct Polish grammatical forms — pay special attention to:
        - noun declension (e.g. "trójkąt" → "trójkąta", "trójkątowi", "trójkącie")
        - adjective agreement (e.g. "prostokątny" must agree in gender/case with its noun)
        - verb conjugation (e.g. "udowodnij", "wyznacz", "oblicz", "pokaż, że")
        - preposition + case agreement (e.g. "dla trójkąta", "w okręgu", "na prostej")
        - correct mathematical terminology translation (e.g. calculus -> analiza matematyczna)
        """
    )

    SEED: int = 42

    # Training hyperparameters:
    EPOCHS: int = 3
    BATCH_SIZE: int = 4               # safe default on H100 without quantization
    EVAL_BATCH_SIZE: int = 4
    SAVE_TOTAL_LIMIT: int = 3
    MAX_SEQUENCE_LENGTH: int = 4096
    GRADIENT_ACCUMULATION_STEPS: int = 4 
    DATALOADER_NUM_WORKERS: int = 4

    # LORA hyperparameters:
    LORA_R: int = 32
    LORA_DROPOUT: float = 0.05
    TARGET_MODULES: str = "all-linear"

    # Optimizer & LR scheduler:
    LEARNING_RATE: float = 2e-4
    WARMUP_RATIO: float = 0.05
    LR_SCHEDULER_TYPE: str = "cosine"
    WEIGHT_DECAY: float = 0.001
    MAX_GRAD_NORM: float = 0.3


    VAL_SPLIT: float = 0.1 
    SAVE_STEPS: int = 200
    LOG_STEPS: int = 10

    # --- Derived fields (computed in __post_init__) ---
    LORA_ALPHA: int = field(init=False, default=0)
    RUN_NAME: str = field(init=False, default="")
    PROJECT_RUN_NAME: str = field(init=False, default="")
    SAVE_DIR: str = field(init=False, default="")
    HUB_MODEL_NAME: str = field(init=False, default="")
    OPTIMIZER: str = field(init=False, default="")
    USE_BF16: bool = field(init=False, default=False)

    def __post_init__(self):
        self.LORA_ALPHA = self.LORA_R * 2

        ts = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        self.RUN_NAME = ts
        self.PROJECT_RUN_NAME = f"{self.PROJECT_NAME}-{self.RUN_NAME}"
        self.SAVE_DIR = f"./{self.PROJECT_RUN_NAME}"
        self.HUB_MODEL_NAME = f"{self.HF_USER}/{self.PROJECT_RUN_NAME}"

        # Use appropriate optimizer based on quantization and device
        if torch.cuda.is_available():
            # GPU: use bitsandbytes paged optimizers (memory-efficient)
            self.OPTIMIZER = "paged_adamw_8bit" if self.QUANTIZATION in ("4b", "8b") else "paged_adamw_32bit"
        else:
            # CPU: use standard PyTorch optimizer
            print("Warning: CUDA not available. Using CPU-compatible optimizer (adamw_torch)")
            self.OPTIMIZER = "adamw_torch"

        # H100 is sm_90 (capability >= 8) and supports bfloat16 natively
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        self.USE_BF16 = cap[0] >= 8

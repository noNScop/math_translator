import random

import numpy as np
import torch
import wandb
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (  # type: ignore[attr-defined]
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from config import Config


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def to_translation_examples(examples: dict, system_prompt: str) -> dict:
    """
    Batched map function: converts one JSONL record into two training examples —
    one for the problem translation and one for the solution translation.
    Each record in translations_output.jsonl has:
        problem_en, problem_pl, solution_en, solution_pl
    Returning lists twice as long as the input lets HuggingFace datasets flatten
    each record into two rows automatically.
    """
    prompts, completions = [], []
    for p_en, p_pl, s_en, s_pl in zip(
        examples["problem_en"], examples["problem_pl"],
        examples["solution_en"], examples["solution_pl"],
    ):
        for label, source_text, translated_text in (("problem", p_en, p_pl), ("solution", s_en, s_pl)):
            prompts.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""Translate the following mathematical {label} 
                                              from English to Polish:\n\n{source_text.strip()}"""},
            ])
            completions.append([
                {"role": "assistant", "content": translated_text.strip()},
            ])
    return {"prompt": prompts, "completion": completions}


if __name__ == "__main__":
    cfg = Config()

    assert cfg.QUANTIZATION in ("none", "4b", "8b"), (
        f"Unknown QUANTIZATION: {cfg.QUANTIZATION!r}. Expected 'none', '4b', or '8b'."
    )

    set_global_seed(cfg.SEED)

    # Enable TF32 for matrix multiplications on Ampere+ GPUs (H100 = sm_90)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 1. Auth and logging
    if cfg.HF_TOKEN:
        login(cfg.HF_TOKEN, add_to_git_credential=True)
    else:
        print("Warning: HF_TOKEN not set. Pushing to Hub will fail.")

    if cfg.LOG_TO_WANDB and cfg.WANDB_API_KEY:
        wandb.login(key=cfg.WANDB_API_KEY)
        wandb.init(project=cfg.PROJECT_NAME, name=cfg.RUN_NAME)

    # Dataset
    print(f"Loading dataset from: {cfg.DATASET_PATH}...")
    raw_dataset = load_dataset("json", data_files=cfg.DATASET_PATH, split="train")
    splits = raw_dataset.train_test_split(test_size=cfg.VAL_SPLIT, seed=cfg.SEED)

    train_ds = splits["train"]
    val_ds = splits["test"]

    # Each record yields 2 examples (problem + solution), so we use batched=True
    map_fn = lambda ex: to_translation_examples(ex, cfg.SYSTEM_PROMPT)
    train_ds = train_ds.map(map_fn, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(map_fn, batched=True, remove_columns=val_ds.column_names)

    print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    # 3. Quantization config
    quant_config = None
    if cfg.QUANTIZATION == "4b":
        print("Applying 4-bit NF4 quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.USE_BF16 else torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    elif cfg.QUANTIZATION == "8b":
        print("Applying 8-bit quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16 if cfg.USE_BF16 else torch.float16,
        )
    else:
        print("No quantization. Loading in bf16." if cfg.USE_BF16 else "No quantization. Loading in fp16.")

    # 4. Tokenizer
    processor = AutoTokenizer.from_pretrained(cfg.BASE_MODEL, use_fast=True, trust_remote_code=True)
    processor.padding_side = "right"
    # Avoid conflating <eos> (strong "end of text" signal) with the neutral padding token
    if processor.pad_token is None:
        processor.add_special_tokens({"pad_token": "[PAD]"})

    # 5. Base model
    # Pass torch_dtype explicitly when not quantizing to avoid fp32 default
    model_kwargs: dict = {"device_map": "auto", "trust_remote_code": True}
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if cfg.USE_BF16 else torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(cfg.BASE_MODEL, **model_kwargs)

    # Disable KV-cache during training; sync all token IDs between model configs
    base_model.config.use_cache = False
    base_model.config.pad_token_id = processor.pad_token_id
    base_model.config.bos_token_id = processor.bos_token_id
    base_model.config.eos_token_id = processor.eos_token_id
    base_model.generation_config.pad_token_id = processor.pad_token_id
    base_model.generation_config.bos_token_id = processor.bos_token_id
    base_model.generation_config.eos_token_id = processor.eos_token_id

    # Sync embedding table size if we added a new pad token
    if processor.pad_token == "[PAD]":
        base_model.resize_token_embeddings(len(processor))

    # 6. LoRA — guard against re-wrapping if the model is already a PeftModel
    if isinstance(base_model, PeftModel):
        base_model = base_model.unload()
        base_model.config.use_cache = False

    if quant_config is not None:
        base_model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.TARGET_MODULES,  # "all-linear" covers every projection automatically
    )

    # 7. Training config
    sft_config = SFTConfig(
        output_dir=cfg.SAVE_DIR,
        num_train_epochs=cfg.EPOCHS,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
        optim=cfg.OPTIMIZER,
        learning_rate=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
        max_grad_norm=cfg.MAX_GRAD_NORM,
        warmup_ratio=cfg.WARMUP_RATIO,
        lr_scheduler_type=cfg.LR_SCHEDULER_TYPE,
        fp16=not cfg.USE_BF16,
        bf16=cfg.USE_BF16,
        tf32=cfg.USE_BF16,           # TF32 is only meaningful on Ampere+ (same guard as bf16)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=cfg.MAX_SEQUENCE_LENGTH,
        completion_only_loss=True,   # compute loss only on assistant tokens, not on the prompt
        packing=False,
        remove_unused_columns=False,
        dataloader_num_workers=cfg.DATALOADER_NUM_WORKERS,
        save_strategy="steps",
        save_steps=cfg.SAVE_STEPS,
        save_total_limit=cfg.SAVE_TOTAL_LIMIT,
        eval_strategy="steps",
        eval_steps=cfg.SAVE_STEPS,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        logging_steps=cfg.LOG_STEPS,
        report_to="wandb" if cfg.LOG_TO_WANDB else "none",
        run_name=cfg.RUN_NAME,
        hub_strategy="every_save",
        push_to_hub=cfg.PUSH_TO_HUB,
        hub_model_id=cfg.HUB_MODEL_NAME,
        hub_private_repo=True,
        seed=cfg.SEED,
    )

    # 8. Trainer
    trainer = SFTTrainer(
        model=base_model,
        processing_class=processor,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        args=sft_config,
    )

    # Guard against misconfigured LoRA that attaches no trainable parameters
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    if trainable_params == 0:
        raise RuntimeError(
            "No trainable LoRA parameters were attached. Check target_modules before training."
        )
    print(f"Trainable LoRA parameters: {trainable_params:,}")

    # 9. Train
    print("Starting SFT training...")
    trainer.train()

    # Re-enable KV-cache and switch to eval mode after training
    trainer.model.eval()
    trainer.model.config.use_cache = True

    # 10. Save locally, then optionally push adapters and tokenizer to Hub
    print(f"Saving adapter and tokenizer to {cfg.SAVE_DIR}...")
    trainer.model.save_pretrained(cfg.SAVE_DIR)
    processor.save_pretrained(cfg.SAVE_DIR)

    if cfg.PUSH_TO_HUB:
        print(f"Pushing to HF Hub: {cfg.HUB_MODEL_NAME}...")
        trainer.model.push_to_hub(cfg.HUB_MODEL_NAME, private=True)
        processor.push_to_hub(cfg.HUB_MODEL_NAME, private=True)

    if cfg.LOG_TO_WANDB:
        wandb.finish()

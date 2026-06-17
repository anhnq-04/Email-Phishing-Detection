"""
train_qwen_phase2.py
====================
Phase 2: Vietnamese-heavy adaptation + English replay + hard negatives
Final dataset policy: content-only Qwen V2, neutral [LINK_N], no domain_spoofing.
Current final train distribution is intentionally mildly legit-heavy due to hard negatives.

Model: models/qwen/model_phase1 (adapter learned in Phase 1)
Data:  data/processed/qwen/phase2_train.jsonl
Eval:  data/processed/qwen/phase2_eval.jsonl

Hardware target: 1x NVIDIA A100 40GB
Run:
  python -m src.training.train_qwen_phase2

Important:
- Paths are intentionally unchanged.
- Put the final token-safe JSONL into the existing paths above.
- This script requires every formatted sequence to be <= 4096 tokens and fails fast otherwise.
"""

import inspect
from pathlib import Path

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# =======================
# PATHS — KEPT UNCHANGED
# =======================
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _PROJECT_ROOT / "data" / "processed" / "qwen"

MODEL_PATH = str(_PROJECT_ROOT / "models" / "qwen" / "model_phase1")

DATA_PATH = str(DATA_DIR / "phase2_train.jsonl")
VAL_PATH  = str(DATA_DIR / "phase2_eval.jsonl")

OUTPUT_DIR = str(_PROJECT_ROOT / "models" / "qwen" / "checkpoint_phase2")

# =======================
# TRAINING CONFIG — A100 40GB, QLoRA, 4096 context
# =======================
SEED = 3407
MAX_SEQ_LENGTH = 4096

PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8   # Effective train batch size = 32 on 1 GPU.

NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 8e-6
WARMUP_RATIO = 0.05

LOGGING_STEPS = 10
EVAL_STEPS = 100
SAVE_STEPS = 100

FLAG_ENABLE_THINKING = True

# =======================
# GPU CHECK / A100 SPEED OPTIONS
# =======================
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not found. This script is configured for one NVIDIA A100 40GB.")

gpu_name = torch.cuda.get_device_name(0)
print(f"GPU: {gpu_name}")
print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")

if not torch.cuda.is_bf16_supported():
    raise RuntimeError("BF16 is not supported on the detected GPU. Use an A100 or update precision settings.")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =======================
# LOAD PHASE 1 ADAPTER FOR CONTINUED TRAINING
# =======================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)


def print_and_validate_trainable_parameters(model) -> None:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    pct = 100.0 * trainable / total if total else 0.0
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    if trainable == 0:
        raise RuntimeError(
            "No trainable parameters found after loading model_phase1. "
            "Phase 2 would not update the LoRA adapter."
        )


# Unsloth loads saved PEFT adapters in trainable mode; this assertion prevents a silent no-op run.
print_and_validate_trainable_parameters(model)

# =======================
# LOAD DATA
# =======================
dataset = load_dataset(
    "json",
    data_files={
        "train": DATA_PATH,
        "validation": VAL_PATH,
    },
)

ds_train = dataset["train"]
ds_val = dataset["validation"]

print(f"Train rows: {len(ds_train):,}")
print(f"Eval rows : {len(ds_val):,}")

# =======================
# FORMAT CHAT
# =======================
def format_chat(example):
    messages = example["messages"]
    if messages and isinstance(messages[0], list):
        messages = messages[0]

    # Must match Phase 1 and later inference: strict JSON, no Qwen thinking output.
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=FLAG_ENABLE_THINKING,
    )
    if not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token
    return {"text": text}


original_train_columns = ds_train.column_names
original_val_columns = ds_val.column_names

ds_train = ds_train.map(format_chat, remove_columns=original_train_columns)
ds_val = ds_val.map(format_chat, remove_columns=original_val_columns)


def add_token_length(example):
    length = len(
        tokenizer(
            example["text"],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    )
    return {"num_tokens": length}


def validate_token_lengths(split, split_name: str):
    checked = split.map(add_token_length)
    token_lengths = checked["num_tokens"]
    max_tokens = max(token_lengths)
    over_limit = sum(length > MAX_SEQ_LENGTH for length in token_lengths)
    print(f"{split_name}: rows={len(checked):,}, max_tokens={max_tokens:,}, >{MAX_SEQ_LENGTH}={over_limit}")
    if over_limit:
        raise ValueError(
            f"{split_name} has {over_limit} rows above MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}. "
            "Use the token-safe final JSONL; do not silently truncate evidence examples."
        )
    return checked.remove_columns(["num_tokens"])


ds_train = validate_token_lengths(ds_train, "train")
ds_val = validate_token_lengths(ds_val, "validation")

# =======================
# VERSION-TOLERANT SFT CONFIG
# =======================
def build_sft_config() -> SFTConfig:
    config_params = inspect.signature(SFTConfig.__init__).parameters

    kwargs = dict(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        log_level="info",
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="constant_with_warmup",
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        tf32=True,
        packing=False,
        seed=SEED,
        report_to="none",
        dataloader_pin_memory=True,
    )

    if "max_length" in config_params:
        kwargs["max_length"] = MAX_SEQ_LENGTH
    elif "max_seq_length" in config_params:
        kwargs["max_seq_length"] = MAX_SEQ_LENGTH

    if "eval_strategy" in config_params:
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = EVAL_STEPS
    else:
        kwargs["evaluation_strategy"] = "steps"
        kwargs["eval_steps"] = EVAL_STEPS

    if "padding_free" in config_params:
        kwargs["padding_free"] = True

    return SFTConfig(**kwargs)


def build_trainer() -> SFTTrainer:
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    kwargs = dict(
        model=model,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        args=build_sft_config(),
    )

    if "processing_class" in trainer_params:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer

    if "max_seq_length" in trainer_params:
        kwargs["max_seq_length"] = MAX_SEQ_LENGTH

    return SFTTrainer(**kwargs)


trainer = build_trainer()

# =======================
# TRAIN
# =======================
if __name__ == "__main__":
    print("Training Phase 2 — VN + EN replay + hard negatives, A100 40GB, MAX_SEQ_LENGTH=4096")
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # Final model output path intentionally unchanged.
    model_output = str(_PROJECT_ROOT / "models" / "qwen" / "model_final")
    trainer.save_model(model_output)
    tokenizer.save_pretrained(model_output)

    print(f"Saved final Phase 2 adapter/tokenizer -> {model_output}")

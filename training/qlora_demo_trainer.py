#!/usr/bin/env python3
"""
ROCm-safe, version-agnostic QLoRA demo trainer.

This script:
- Uses a tiny in-memory dataset (no HF datasets)
- Avoids all features that break across Transformers versions
- Uses BF16 (ROCm-friendly)
- Uses LoRA via PEFT (no bitsandbytes)
- Produces a working adapter folder for your GitHub demo
"""

import os
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, field
import argparse
import typing

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model


# -----------------------------
# Tiny in-memory dataset
# -----------------------------
CORPUS = [
    "This is a tiny demo corpus for QLoRA training.",
    "The goal is to produce a working adapter for the GitHub repo.",
    "Everything here runs on an AMD RX 7700 XT using ROCm.",
    "This demo avoids all external datasets and HF Hub issues.",
    "Training should complete quickly and save LoRA adapters.",
]


class TinyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        return {k: torch.tensor(v) for k, v in enc.items()}


# -----------------------------
# Args
# -----------------------------
@dataclass
class Args:
    model_name_or_path: str = field(default="EleutherAI/gpt-neo-125M")
    output_dir: str = field(default="./demo-output")
    max_seq_length: int = field(default=128)
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=5e-5)
    seed: int = field(default=42)
    force_bf16: bool = field(default=True)
    trust_remote_code: bool = field(default=True)


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="ROCm-safe QLoRA demo trainer")
    resolved = typing.get_type_hints(Args)

    for name, field_def in Args.__dataclass_fields__.items():
        arg_name = f"--{name}"
        ann_type = resolved.get(name, str)

        if ann_type is bool:
            parser.add_argument(arg_name, type=str, default=str(field_def.default))
        else:
            parser.add_argument(arg_name, type=ann_type, default=field_def.default)

    ns = parser.parse_args()

    parsed = {}
    for k, v in vars(ns).items():
        ann_type = resolved.get(k, str)
        if ann_type is bool:
            parsed[k] = str(v).lower() in ("1", "true", "yes", "y")
        else:
            parsed[k] = v

    return Args(**parsed)


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # Load model + tokenizer
    dtype = torch.bfloat16 if args.force_bf16 else None
    print(f"[INFO] Loading model {args.model_name_or_path} (dtype={dtype})")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Dataset
    train_dataset = TinyDataset(CORPUS, tokenizer, max_length=args.max_seq_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args (no evaluation_strategy)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_steps=999999,  # effectively disable mid-training saves
        bf16=args.force_bf16,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    print(f"[INFO] Saving adapter + tokenizer to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[DONE] Training complete.")


if __name__ == "__main__":
    main()

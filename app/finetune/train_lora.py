from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from app.config import get_settings


@dataclass
class FinetuneConfig:
    model_name: str
    data_path: Path
    output_dir: Path
    lr: float = 2e-4
    epochs: int = 2
    batch_size: int = 2
    grad_accum: int = 8
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


def build_trainer(cfg: FinetuneConfig, train_ds, eval_ds) -> SFTTrainer:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    lora_model = get_peft_model(base_model, peft_config)

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=lora_model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="prompt",
        max_seq_length=2048,
        packing=False,
        args=training_args,
        formatting_func=None,
        compute_metrics=None,
    )
    return trainer


def finetune(cfg: FinetuneConfig, train_ds, eval_ds) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(cfg, train_ds, eval_ds)
    trainer.train()
    trainer.model.save_pretrained(str(cfg.output_dir))


def main_cli() -> None:
    import argparse
    from app.finetune.dataset import load_threat_dataset

    settings = get_settings()
    parser = argparse.ArgumentParser(description="LoRA Finetune LLaMA 3 for classification via SFT")
    parser.add_argument("data", type=Path, help="Path to JSON/JSONL/CSV with text/label fields")
    parser.add_argument("--model", type=str, default=settings.llama_base_model)
    parser.add_argument("--output", type=Path, default=settings.lora_output_dir)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    ds: Dict[str, object] = load_threat_dataset(args.data)
    cfg = FinetuneConfig(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
    )
    finetune(cfg, ds["train"], ds["eval"])  # type: ignore[arg-type]


if __name__ == "__main__":
    main_cli()



"""Cloud training entrypoint for E10 / PEFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.training.training_utils import (
    ALLOWED_E10_TASK_TYPES,
    build_output_paths,
    build_sft_dataset,
    ensure_output_dirs,
    load_manifest_records,
    load_train_config,
    write_training_metadata,
)


def check_training_dependencies() -> None:
    missing: list[str] = []
    for module_name in ["accelerate", "bitsandbytes", "peft", "trl", "datasets"]:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise ImportError(
            "云端训练依赖缺失，请先安装: " + ", ".join(missing)
        )


def run_training(config_path: str | Path, dry_run: bool = False) -> dict[str, Any]:
    config = load_train_config(config_path)
    train_rows = load_manifest_records(config.train_manifest_path, set(config.task_types))
    dev_rows = load_manifest_records(config.dev_manifest_path, set(config.task_types))
    train_dataset = build_sft_dataset(train_rows)
    dev_dataset = build_sft_dataset(dev_rows)
    output_paths = build_output_paths(config) if dry_run else ensure_output_dirs(config)
    adapter_metadata_path = None
    summary_path = None
    if not dry_run:
        adapter_metadata_path, summary_path = write_training_metadata(
            config=config,
            output_paths=output_paths,
            train_rows=train_rows,
            dev_rows=dev_rows,
        )

    result = {
        "config": config.model_dump(),
        "train_sample_count": len(train_dataset),
        "dev_sample_count": len(dev_dataset),
        "output_paths": {key: str(value) for key, value in output_paths.items()},
        "adapter_metadata_path": str(adapter_metadata_path) if adapter_metadata_path else "",
        "train_summary_path": str(summary_path) if summary_path else "",
        "dry_run": dry_run,
    }

    if dry_run:
        return result

    check_training_dependencies()

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    bnb_config = None
    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16" if config.bf16 else "float16",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.target_modules,
    )

    train_dataset_hf = Dataset.from_list(train_dataset)
    eval_dataset_hf = Dataset.from_list(dev_dataset)
    training_args = TrainingArguments(
        output_dir=str(output_paths["checkpoint_dir"]),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        logging_dir=str(output_paths["log_dir"]),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=config.bf16,
        fp16=not config.bf16,
        report_to=[],
        save_total_limit=2,
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset_hf,
        eval_dataset=eval_dataset_hf,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.model.save_pretrained(str(output_paths["adapter_dir"]))
    tokenizer.save_pretrained(str(output_paths["adapter_dir"]))
    adapter_metadata_path, summary_path = write_training_metadata(
        config=config,
        output_paths=output_paths,
        train_rows=train_rows,
        dev_rows=dev_rows,
    )
    result["adapter_metadata_path"] = str(adapter_metadata_path)
    result["train_summary_path"] = str(summary_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run_training(args.config, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

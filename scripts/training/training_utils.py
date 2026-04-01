"""Shared helpers for E10 / PEFT training scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from scripts.shared.experiment_utils import ROOT_DIR, load_jsonl, stable_hash, utc_now_iso, write_json


ALLOWED_E10_TASK_TYPES = {
    "preference_parse",
    "clarification",
    "constraint_honesty",
    "feedback_update",
}


class E10TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model_id: str
    adapter_type: str
    train_manifest_path: str
    dev_manifest_path: str
    task_types: list[str]
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 2
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=list)
    load_in_4bit: bool = True
    bf16: bool = True
    output_adapter_dir: str
    notes: str = ""


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def load_train_config(path: str | Path) -> E10TrainConfig:
    config_path = resolve_repo_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing E10 train config: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = E10TrainConfig.model_validate(payload)
    invalid_task_types = sorted(set(config.task_types) - ALLOWED_E10_TASK_TYPES)
    if invalid_task_types:
        raise ValueError(
            f"Unsupported task_types in E10 config: {', '.join(invalid_task_types)}"
        )
    if config.adapter_type != "qlora":
        raise ValueError(
            f"E10 当前只支持 qlora，收到 adapter_type={config.adapter_type}"
        )
    return config


def load_manifest_records(path: str | Path, allowed_task_types: set[str]) -> list[dict[str, Any]]:
    manifest_path = resolve_repo_path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")
    rows = load_jsonl(manifest_path)
    filtered = [row for row in rows if row["task_type"] in allowed_task_types]
    if not filtered:
        raise ValueError(
            f"Manifest {manifest_path} 中没有可用于训练的 task_types={sorted(allowed_task_types)}"
        )
    return filtered


def build_sft_text_sample(row: dict[str, Any]) -> dict[str, str]:
    instruction = (
        f"Task: {row['task_type']}\n"
        "Return only JSON that satisfies the required target schema."
    )
    user_payload = json.dumps(row["input_payload"], ensure_ascii=False, sort_keys=True)
    target_payload = json.dumps(row["target_payload"], ensure_ascii=False, sort_keys=True)
    text = (
        "<|system|>\n"
        "You are a hotel recommendation workflow training assistant.\n"
        "<|user|>\n"
        f"{instruction}\nInput JSON:\n{user_payload}\n"
        "<|assistant|>\n"
        f"{target_payload}"
    )
    return {
        "record_id": row["record_id"],
        "query_id": row["query_id"],
        "task_type": row["task_type"],
        "text": text,
    }


def build_sft_dataset(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [build_sft_text_sample(row) for row in rows]


def build_output_paths(config: E10TrainConfig) -> dict[str, Path]:
    adapter_dir = resolve_repo_path(config.output_adapter_dir)
    if adapter_dir.suffix:
        raise ValueError("output_adapter_dir 必须是目录路径，不能是文件路径。")
    checkpoint_dir = Path(str(adapter_dir).replace("/models/adapters/", "/training/checkpoints/"))
    log_dir = Path(str(adapter_dir).replace("/models/adapters/", "/training/logs/"))
    report_dir = Path(str(adapter_dir).replace("/models/adapters/", "/training/reports/"))
    return {
        "adapter_dir": adapter_dir,
        "checkpoint_dir": checkpoint_dir,
        "log_dir": log_dir,
        "report_dir": report_dir,
    }


def ensure_output_dirs(config: E10TrainConfig) -> dict[str, Path]:
    paths = build_output_paths(config)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_adapter_metadata_payload(
    config: E10TrainConfig,
    output_paths: dict[str, Path],
    train_count: int,
    dev_count: int,
) -> dict[str, Any]:
    adapter_name = output_paths["adapter_dir"].name
    return {
        "adapter_name": adapter_name,
        "base_model_id": config.base_model_id,
        "served_model_id": f"{Path(config.base_model_id).name}-PEFT-{adapter_name}",
        "adapter_path": str(output_paths["adapter_dir"]),
        "backend": "api",
        "adapter_type": config.adapter_type,
        "train_manifest_path": config.train_manifest_path,
        "dev_manifest_path": config.dev_manifest_path,
        "task_types": config.task_types,
        "train_sample_count": train_count,
        "dev_sample_count": dev_count,
        "generated_at": utc_now_iso(),
        "config_hash": stable_hash(config.model_dump()),
    }


def write_training_metadata(
    config: E10TrainConfig,
    output_paths: dict[str, Path],
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    adapter_metadata = build_adapter_metadata_payload(
        config=config,
        output_paths=output_paths,
        train_count=len(train_rows),
        dev_count=len(dev_rows),
    )
    summary_payload = {
        "generated_at": utc_now_iso(),
        "base_model_id": config.base_model_id,
        "adapter_type": config.adapter_type,
        "task_types": config.task_types,
        "train_sample_count": len(train_rows),
        "dev_sample_count": len(dev_rows),
        "config_hash": stable_hash(config.model_dump()),
    }
    adapter_metadata_path = output_paths["report_dir"] / "adapter_metadata.json"
    summary_path = output_paths["report_dir"] / "train_summary.json"
    write_json(adapter_metadata_path, adapter_metadata)
    write_json(summary_path, summary_payload)
    return adapter_metadata_path, summary_path

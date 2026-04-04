"""Shared helpers for E10 / PEFT training scripts."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from pathlib import PurePath
from pathlib import PurePosixPath
from pathlib import PureWindowsPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from scripts.shared.experiment_utils import ROOT_DIR, load_jsonl, stable_hash, utc_now_iso, write_json


ALLOWED_E10_TASK_TYPES = {
    "preference_parse",
    "clarification",
    "constraint_honesty",
    "feedback_update",
    "grounded_recommendation",
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
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
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


def normalize_path_reference(path_value: str | Path) -> Path | PurePath:
    path_text = str(path_value).strip()
    if not path_text:
        return ROOT_DIR
    local_path = Path(path_text)
    if local_path.is_absolute():
        return local_path
    posix_path = PurePosixPath(path_text)
    if posix_path.is_absolute():
        return posix_path
    windows_path = PureWindowsPath(path_text)
    if windows_path.is_absolute():
        return windows_path
    return ROOT_DIR / Path(path_text)


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


def compact_grounded_input_payload_for_training(
    input_payload: dict[str, Any],
    *,
    max_hotels: int = 2,
    max_sentences_per_aspect: int = 2,
) -> dict[str, Any]:
    compact_payload = dict(input_payload)
    compact_payload["candidate_hotels"] = [
        {
            "hotel_id": hotel["hotel_id"],
            "hotel_name": hotel["hotel_name"],
        }
        for hotel in input_payload.get("candidate_hotels", [])[:max_hotels]
    ]

    user_preference = input_payload.get("user_preference_gold", {})
    relevant_aspects = set(user_preference.get("focus_aspects", [])) | set(
        user_preference.get("avoid_aspects", [])
    )

    compact_packs: list[dict[str, Any]] = []
    for pack in input_payload.get("evidence_packs", [])[:max_hotels]:
        compact_aspects: dict[str, list[dict[str, str]]] = {}
        allowed_sentence_ids: list[str] = []
        for aspect in sorted(pack.get("evidence_by_aspect", {})):
            if relevant_aspects and aspect not in relevant_aspects:
                continue
            compact_rows = []
            for sentence in pack["evidence_by_aspect"][aspect][:max_sentences_per_aspect]:
                compact_rows.append(
                    {
                        "sentence_id": sentence["sentence_id"],
                        "sentence_text": sentence["sentence_text"],
                    }
                )
                allowed_sentence_ids.append(sentence["sentence_id"])
            if compact_rows:
                compact_aspects[aspect] = compact_rows
        compact_packs.append(
            {
                "hotel_id": pack["hotel_id"],
                "evidence_by_aspect": compact_aspects,
                "allowed_sentence_ids": allowed_sentence_ids,
            }
        )
    compact_payload["evidence_packs"] = compact_packs
    return compact_payload


def build_sft_text_sample(row: dict[str, Any]) -> dict[str, str]:
    instruction = (
        f"Task: {row['task_type']}\n"
        "Return only JSON that satisfies the required target schema."
    )
    input_payload = row["input_payload"]
    if row["task_type"] == "grounded_recommendation":
        input_payload = compact_grounded_input_payload_for_training(input_payload)
    user_payload = json.dumps(input_payload, ensure_ascii=False, sort_keys=True)
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


def assert_sft_samples_within_max_seq_length(
    samples: list[dict[str, str]],
    tokenizer: Any,
    max_seq_length: int,
    *,
    dataset_name: str,
) -> None:
    overflow_rows: list[tuple[str, str, int]] = []
    for row in samples:
        token_count = len(
            tokenizer(
                row["text"],
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        if token_count > max_seq_length:
            overflow_rows.append((row["record_id"], row["task_type"], token_count))

    if overflow_rows:
        preview = ", ".join(
            f"{record_id}:{task_type}:{token_count}"
            for record_id, task_type, token_count in overflow_rows[:5]
        )
        raise ValueError(
            f"{dataset_name} 中存在超过 max_seq_length={max_seq_length} 的样本，"
            f"请先缩减 grounded payload 或调整样本构造。样本预览：{preview}"
        )


def build_sft_trainer_kwargs(
    trainer_cls: type,
    *,
    model: Any,
    tokenizer: Any,
    args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    peft_config: Any,
    max_seq_length: int,
) -> dict[str, Any]:
    sig = inspect.signature(trainer_cls.__init__)
    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
    }
    if "tokenizer" in sig.parameters:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in sig.parameters:
        kwargs["processing_class"] = tokenizer

    if "dataset_text_field" in sig.parameters:
        kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in sig.parameters:
        kwargs["max_seq_length"] = max_seq_length
    elif "max_length" in sig.parameters:
        kwargs["max_length"] = max_seq_length
    return kwargs


def build_output_paths(config: E10TrainConfig) -> dict[str, Path | PurePath]:
    adapter_dir = normalize_path_reference(config.output_adapter_dir)
    if adapter_dir.suffix:
        raise ValueError("output_adapter_dir 必须是目录路径，不能是文件路径。")
    path_cls = type(adapter_dir)
    checkpoint_dir = path_cls(str(adapter_dir).replace("/models/adapters/", "/training/checkpoints/"))
    log_dir = path_cls(str(adapter_dir).replace("/models/adapters/", "/training/logs/"))
    report_dir = path_cls(str(adapter_dir).replace("/models/adapters/", "/training/reports/"))
    return {
        "adapter_dir": adapter_dir,
        "checkpoint_dir": checkpoint_dir,
        "log_dir": log_dir,
        "report_dir": report_dir,
    }


def ensure_output_dirs(config: E10TrainConfig) -> dict[str, Path]:
    paths = build_output_paths(config)
    materialized_paths: dict[str, Path] = {}
    for key, path in paths.items():
        if not isinstance(path, Path):
            raise ValueError(
                "当前系统无法直接创建该 output_adapter_dir 对应的目标路径。"
                "请在目标训练环境中执行正式训练，或仅在本机执行 dry-run。"
            )
        path.mkdir(parents=True, exist_ok=True)
        materialized_paths[key] = path
    return materialized_paths


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

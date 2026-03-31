"""Shared runtime helpers for behavior-model backends."""

from __future__ import annotations

import os
from typing import Any

from scripts.shared.experiment_schemas import BehaviorRuntimeConfig


def parse_env_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def parse_env_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def parse_env_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def flatten_openai_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts).strip()
    return str(content).strip()


def resolve_behavior_runtime_config(
    cfg: dict[str, Any],
    frozen_config: dict[str, Any] | None = None,
) -> tuple[BehaviorRuntimeConfig, str | None]:
    merged_behavior = dict((frozen_config or {}).get("behavior", {}))
    merged_behavior.update(cfg.get("behavior", {}))

    llm_backend = os.environ.get("BEHAVIOR_LLM_BACKEND", merged_behavior.get("llm_backend", "local"))
    model_id = os.environ.get(
        "BEHAVIOR_MODEL_ID",
        merged_behavior.get("model_id") or merged_behavior.get("base_model"),
    )
    if not model_id:
        raise ValueError("behavior.model_id/base_model 未配置，无法初始化行为实验模型。")

    api_base_url = (
        os.environ.get("BEHAVIOR_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or merged_behavior.get("api_base_url")
    )
    api_key_env = str(merged_behavior.get("api_key_env", "OPENAI_API_KEY"))
    api_key = (
        os.environ.get("BEHAVIOR_OPENAI_API_KEY")
        or os.environ.get(api_key_env)
        or merged_behavior.get("api_key")
    )
    enable_thinking = parse_env_bool(
        os.environ.get("BEHAVIOR_ENABLE_THINKING"),
        bool(merged_behavior.get("enable_thinking", False)),
    )
    temperature = parse_env_float(
        os.environ.get("BEHAVIOR_TEMPERATURE"),
        float(merged_behavior.get("temperature", 0.0)),
    )
    max_new_tokens = parse_env_int(
        os.environ.get("BEHAVIOR_MAX_NEW_TOKENS"),
        int(merged_behavior.get("max_new_tokens", 256)),
    )
    api_timeout_seconds = parse_env_int(
        os.environ.get("BEHAVIOR_API_TIMEOUT_SECONDS"),
        int(merged_behavior.get("api_timeout_seconds", 120)),
    )

    if llm_backend not in {"local", "api"}:
        raise ValueError(f"不支持的 behavior.llm_backend: {llm_backend}")
    if llm_backend == "api" and not api_base_url:
        raise ValueError("behavior.llm_backend=api 时必须配置 OPENAI_BASE_URL 或 behavior.api_base_url。")
    if llm_backend == "api" and not api_key:
        api_key = "EMPTY"

    runtime_config = BehaviorRuntimeConfig(
        llm_backend=llm_backend,
        model_id=str(model_id),
        api_base_url=str(api_base_url) if api_base_url else None,
        api_key_env=api_key_env,
        api_key_present=bool(api_key),
        enable_thinking=enable_thinking,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        api_timeout_seconds=api_timeout_seconds,
    )
    return runtime_config, api_key

"""DeepSeek Reasoner batch helper for E10 v4 query/target draft generation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

from scripts.evaluation.evaluate_e3_e5_behavior import parse_json_with_repair
from scripts.shared.behavior_runtime import flatten_openai_content
from scripts.shared.experiment_utils import ROOT_DIR, load_jsonl, utc_now_iso, write_jsonl


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-reasoner"


def load_env_for_deepseek() -> None:
    candidate_paths = [
        Path.cwd() / ".env",
        ROOT_DIR / ".env",
        ROOT_DIR.parent / ".env",
    ]
    seen: set[Path] = set()
    for path in candidate_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            load_dotenv(path, override=False)
    discovered = find_dotenv(usecwd=True)
    if discovered:
        load_dotenv(discovered, override=False)


def resolve_deepseek_settings() -> dict[str, Any]:
    load_env_for_deepseek()
    api_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("BEHAVIOR_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "未检测到 DeepSeek API Key。请在 .env 或当前 shell 中设置 "
            "DEEPSEEK_API_KEY。"
        )
    base_url = (
        os.environ.get("DEEPSEEK_BASE_URL")
        or os.environ.get("BEHAVIOR_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_DEEPSEEK_BASE_URL
    )
    model_name = (
        os.environ.get("DEEPSEEK_REASONER_MODEL")
        or os.environ.get("DEEPSEEK_MODEL")
        or DEFAULT_DEEPSEEK_MODEL
    )
    timeout = int(os.environ.get("DEEPSEEK_API_TIMEOUT_SECONDS", "180"))
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model_name": model_name,
        "timeout": timeout,
    }


def build_client(settings: dict[str, Any]):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "缺少 openai 依赖，请先执行 `pip install -r requirements.txt`。"
        ) from exc
    return OpenAI(
        api_key=settings["api_key"],
        base_url=settings["base_url"],
        timeout=settings["timeout"],
    )


def default_request_path(stage: str) -> Path:
    assets = ROOT_DIR / "experiments" / "assets"
    if stage == "query":
        return assets / "e10_v4_deepseek_query_requests.jsonl"
    if stage == "target":
        return assets / "e10_v4_deepseek_target_requests.jsonl"
    raise ValueError(f"Unsupported stage: {stage}")


def default_output_path() -> Path:
    return ROOT_DIR / "experiments" / "assets" / "e10_v4_deepseek_drafts.jsonl"


def load_existing_drafts(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    return {
        (str(row.get("stage", "")), str(row.get("seed_id", ""))): row
        for row in rows
    }


def build_provenance(
    *,
    stage: str,
    request_id: str,
    settings: dict[str, Any],
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    return {
        "generator_provider": "deepseek",
        "generator_model_name": settings["model_name"],
        "generation_stage": stage,
        "request_id": request_id,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_version": f"v4_{stage}",
        "generated_at": utc_now_iso(),
        "api_base_url": settings["base_url"],
    }


def build_query_draft_row(request_row: dict[str, Any], response_text: str, settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": request_row["request_id"],
        "seed_id": request_row["seed_id"],
        "stage": "query_draft",
        "split": request_row["split"],
        "source_mode": request_row["source_mode"],
        "primary_slice": request_row["primary_slice"],
        "secondary_tags": request_row.get("secondary_tags", []),
        "city": request_row.get("city"),
        "query_text_zh": response_text.strip(),
        "review_status": "query_generated",
        "raw_response": response_text,
        "provenance": build_provenance(
            stage="query_draft",
            request_id=request_row["request_id"],
            settings=settings,
            temperature=float(request_row["temperature"]),
            top_p=float(request_row["top_p"]),
        ),
    }


def build_target_draft_row(request_row: dict[str, Any], response_text: str, settings: dict[str, Any]) -> dict[str, Any]:
    payload, _repaired = parse_json_with_repair(response_text)
    return {
        "sample_id": request_row["request_id"],
        "seed_id": request_row["seed_id"],
        "stage": "target_draft",
        "split": request_row["split"],
        "source_mode": request_row["source_mode"],
        "primary_slice": request_row["primary_slice"],
        "secondary_tags": request_row.get("secondary_tags", []),
        "query_id": request_row.get("query_id"),
        "query_text_zh": request_row.get("query_text_zh", ""),
        "target_payload": payload if isinstance(payload, dict) else None,
        "response_parse_ok": isinstance(payload, dict),
        "review_status": "target_generated",
        "raw_response": response_text,
        "provenance": build_provenance(
            stage="target_draft",
            request_id=request_row["request_id"],
            settings=settings,
            temperature=float(request_row["temperature"]),
            top_p=float(request_row["top_p"]),
        ),
    }


def run_deepseek_generation(
    *,
    stage: str,
    input_path: Path,
    output_path: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    settings = resolve_deepseek_settings()
    client = build_client(settings)
    request_rows = load_jsonl(input_path)
    if limit is not None:
        request_rows = request_rows[:limit]
    existing_rows = load_existing_drafts(output_path)

    for request_row in request_rows:
        response = client.chat.completions.create(
            model=settings["model_name"],
            temperature=float(request_row["temperature"]),
            top_p=float(request_row["top_p"]),
            messages=request_row["messages"],
        )
        content = flatten_openai_content(response.choices[0].message.content)
        key = (str(request_row["stage"]), str(request_row["seed_id"]))
        if stage == "query":
            existing_rows[key] = build_query_draft_row(request_row, content, settings)
        else:
            existing_rows[key] = build_target_draft_row(request_row, content, settings)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = [
        existing_rows[key]
        for key in sorted(existing_rows)
    ]
    write_jsonl(output_path, ordered_rows)
    return {
        "stage": stage,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "request_count": len(request_rows),
        "model_name": settings["model_name"],
        "base_url": settings["base_url"],
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["query", "target"], required=True)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input_path) if args.input_path else default_request_path(args.stage)
    output_path = Path(args.output_path) if args.output_path else default_output_path()
    result = run_deepseek_generation(
        stage=args.stage,
        input_path=input_path,
        output_path=output_path,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

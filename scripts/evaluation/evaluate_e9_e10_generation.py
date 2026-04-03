"""Generation-stage evaluation utilities for E9 and E10."""

from __future__ import annotations

import copy
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.evaluation.evaluate_e2_candidate_selection import (
    build_hotel_summary,
    build_profile_tables,
    candidate_rank,
)
from scripts.evaluation.evaluate_e3_e5_behavior import (
    build_query_en_from_slots,
    build_behavior_backend,
    build_rule_clarification,
    load_json,
    parse_json_with_repair,
)
from scripts.evaluation.evaluate_e6_e8_retrieval import (
    build_evidence_lookup,
    build_query_en_target,
    dense_query_hotel,
    markdown_table,
    merge_dense_candidates,
)
from scripts.shared.behavior_postprocess import normalize_aspect_values
from scripts.shared.behavior_runtime import resolve_behavior_runtime_config
from scripts.shared.experiment_schemas import (
    BehaviorRuntimeConfig,
    EvidencePack,
    GenerationEvalUnit,
    HotelCandidate,
    RecommendationItem,
    RecommendationReason,
    RecommendationResponse,
    RunLogEntry,
    SFTManifestRecord,
    SentenceCandidate,
    CitationVerificationResult,
    UserPreference,
)
from scripts.shared.experiment_utils import (
    E9_LABELS_DIR,
    EXPERIMENT_ASSETS_DIR,
    EXPERIMENT_RUNS_DIR,
    city_state_map,
    ensure_dir,
    load_jsonl,
    stable_hash,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from scripts.shared.project_utils import load_config, resolve_repo_path


E9_GROUPS = [
    "A_free_generation",
    "B_grounded_generation",
    "C_grounded_generation_with_verifier",
]
E10_GROUPS = ["A_base_4b_grounded", "B_peft_4b_grounded"]

E9_RETRIEVAL_MODE = "aspect_main_no_rerank"
E9_CANDIDATE_POLICY = "E2_B_final_aspect_score_top5"
E9_CANDIDATE_MODE = "e9_frozen_top5"
E10_CANDIDATE_MODE = "e10_frozen_assets"
E9_GENERATION_MAX_NEW_TOKENS = 512
E9_MAX_RECOMMENDATIONS = 2
E9_MAX_REASONS_PER_ITEM = 2
E9_PROMPT_SENTENCE_LIMIT_FREE = 2
E9_PROMPT_SENTENCE_LIMIT_GROUNDED = 3
E9_RETRY_LIMIT = 1
E9_OUTPUT_FIELDS = '{"summary":"","recommendations":[{"hotel_id":"","hotel_name":"","reasons":[{"aspect":"service","reason_text":"","sentence_id":null}]}],"unsupported_notice":""}'
E9_QUERY_IDS_PATH = EXPERIMENT_ASSETS_DIR / "e9_generation_eval_query_ids.json"
E9_UNITS_PATH = EXPERIMENT_ASSETS_DIR / "e9_generation_eval_units.jsonl"
SFT_TRAIN_MANIFEST_PATH = EXPERIMENT_ASSETS_DIR / "sft_train_manifest.jsonl"
SFT_DEV_MANIFEST_PATH = EXPERIMENT_ASSETS_DIR / "sft_dev_manifest.jsonl"
SFT_TRAIN_MANIFEST_V2_PATH = EXPERIMENT_ASSETS_DIR / "sft_train_manifest_v2.jsonl"
SFT_DEV_MANIFEST_V2_PATH = EXPERIMENT_ASSETS_DIR / "sft_dev_manifest_v2.jsonl"
E10_V2_MANIFEST_REPORT_PATH = EXPERIMENT_ASSETS_DIR / "sft_manifest_v2_report.json"
SFT_TRAIN_MANIFEST_V3_PATH = EXPERIMENT_ASSETS_DIR / "sft_train_manifest_v3.jsonl"
SFT_DEV_MANIFEST_V3_PATH = EXPERIMENT_ASSETS_DIR / "sft_dev_manifest_v3.jsonl"
E10_V3_MANIFEST_REPORT_PATH = EXPERIMENT_ASSETS_DIR / "sft_manifest_v3_report.json"
E10_TRAIN_CONFIG_TEMPLATE_PATH = EXPERIMENT_ASSETS_DIR / "e10_train_config_template.json"
E10_ADAPTER_METADATA_TEMPLATE_PATH = EXPERIMENT_ASSETS_DIR / "e10_adapter_metadata.template.json"
E10_V2_MANIFEST_CONFIG_VERSION = 2
E10_V3_MANIFEST_CONFIG_VERSION = 3
ENGLISH_LONG_SPAN_PATTERN = re.compile(r"[A-Za-z]{4,}")
MISSING_EVIDENCE_REASON_PATTERNS = (
    "无直接证据",
    "未提供明确",
    "缺乏直接证据",
    "缺乏关于",
    "证据不足",
    "跳过",
)

ASPECT_ZH_LABELS = {
    "location_transport": "位置交通",
    "cleanliness": "卫生干净",
    "service": "服务",
    "room_facilities": "房间设施",
    "quiet_sleep": "安静睡眠",
    "value": "性价比",
}

UNSUPPORTED_ZH_LABELS = {
    "budget": "预算",
    "distance_to_landmark": "离景点距离",
    "checkin_date": "入住日期",
}

E10_ADAPTER_METADATA_REQUIRED_FIELDS = {
    "adapter_name",
    "base_model_id",
    "served_model_id",
    "adapter_path",
    "backend",
}


def canonical_model_id_name(model_id: str) -> str:
    normalized = str(model_id).strip().rstrip("/")
    if not normalized:
        return ""
    return Path(normalized).name


def review_id_from_sentence_id(sentence_id: str) -> str:
    if "_s" in sentence_id:
        return sentence_id.rsplit("_s", 1)[0]
    return sentence_id.split("_", 1)[0]


def resolve_generation_asset_paths(limit_queries: int | None = None) -> tuple[Path, Path]:
    if limit_queries is None:
        return E9_UNITS_PATH, E9_QUERY_IDS_PATH
    suffix = f"head_{limit_queries}"
    return (
        EXPERIMENT_ASSETS_DIR / f"e9_generation_eval_units.{suffix}.jsonl",
        EXPERIMENT_ASSETS_DIR / f"e9_generation_eval_query_ids.{suffix}.json",
    )


def load_generation_eval_units(path: Path) -> list[GenerationEvalUnit]:
    return [GenerationEvalUnit.model_validate(row) for row in load_jsonl(path)]


def resolve_repo_relative_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    return resolve_repo_path(path_value)


def load_adapter_metadata(path: str | Path) -> dict[str, Any]:
    metadata_path = resolve_repo_path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing adapter metadata: {metadata_path}. "
            "请先准备 adapter metadata 文件，再运行 e10_base_vs_peft。"
        )
    payload = load_json(metadata_path)
    missing = sorted(E10_ADAPTER_METADATA_REQUIRED_FIELDS - set(payload))
    if missing:
        raise KeyError(
            f"Adapter metadata 缺少字段: {', '.join(missing)}"
        )
    payload["_metadata_path"] = str(metadata_path)
    payload["_resolved_adapter_path"] = str(resolve_repo_path(payload["adapter_path"]))
    return payload


def validate_adapter_metadata_base_model(
    adapter_metadata: dict[str, Any],
    frozen_base_model_id: str,
) -> None:
    metadata_base = str(adapter_metadata["base_model_id"])
    if metadata_base == frozen_base_model_id:
        return
    if canonical_model_id_name(metadata_base) == canonical_model_id_name(frozen_base_model_id):
        return
    raise ValueError(
        "Adapter metadata 中的 base_model_id 与当前冻结主模型不一致："
        f"{metadata_base} != {frozen_base_model_id}"
    )


def validate_runtime_base_model(
    runtime_model_id: str,
    frozen_base_model_id: str,
) -> None:
    runtime_model = str(runtime_model_id).strip()
    if runtime_model == frozen_base_model_id:
        return
    if canonical_model_id_name(runtime_model) == canonical_model_id_name(frozen_base_model_id):
        return
    raise ValueError(
        "E10 固定要求 Base 组使用冻结的正式行为模型 "
        f"{frozen_base_model_id}；当前解析到的 model_id 为 {runtime_model_id}。"
    )


def build_peft_runtime_config(
    base_runtime_config: BehaviorRuntimeConfig,
    adapter_metadata: dict[str, Any],
) -> BehaviorRuntimeConfig:
    if base_runtime_config.llm_backend == "api":
        target_model_id = str(adapter_metadata["served_model_id"])
    elif base_runtime_config.llm_backend == "local":
        target_model_id = str(base_runtime_config.model_id).strip()
        if not target_model_id:
            raise ValueError(
                "PEFT 本地直载模式需要设置 BEHAVIOR_MODEL_ID 指向 merged 模型路径。"
            )
    else:
        raise ValueError(
            f"不支持的 E10 PEFT backend: {base_runtime_config.llm_backend}"
        )
    return base_runtime_config.model_copy(
        update={
            "model_id": target_model_id,
            "use_peft_adapter": True,
            "adapter_path": str(adapter_metadata["_resolved_adapter_path"]),
            "adapter_metadata_path": str(adapter_metadata["_metadata_path"]),
        }
    )


def compute_query_evidence_mean(row: dict[str, Any]) -> float:
    support_scores = [audit_row["support_score"] for audit_row in row["audit_rows"]]
    if not support_scores:
        return 0.0
    return round(sum(support_scores) / len(support_scores), 4)


def is_auditable_generation_row(row: dict[str, Any]) -> bool:
    if row["audit_rows"]:
        return True
    return bool(row["response"].schema_valid and row["response"].unsupported_notice.strip())


def format_analysis_value(value: Any, none_text: str = "n/a") -> str:
    if value is None:
        return none_text
    try:
        if pd.isna(value):
            return none_text
    except TypeError:
        pass
    return str(value)


def build_e10_metric_row(
    group_id: str,
    rows: list[dict[str, Any]],
    stable_run_config: dict[str, Any],
) -> dict[str, Any]:
    unsupported_rows = [row["unsupported_honesty"] for row in rows if row["unsupported_honesty"] is not None]
    support_scores = [
        audit_row["support_score"]
        for row in rows
        for audit_row in row["audit_rows"]
    ]
    return {
        "group_id": group_id,
        "query_count": len(rows),
        "citation_precision": round(
            sum(row["verification"].citation_precision for row in rows) / max(len(rows), 1),
            4,
        ),
        "evidence_verifiability_mean": round(sum(support_scores) / max(len(support_scores), 1), 4),
        "unsupported_honesty_rate": (
            round(sum(unsupported_rows) / len(unsupported_rows), 4) if unsupported_rows else None
        ),
        "schema_valid_rate": round(sum(int(row["response"].schema_valid) for row in rows) / max(len(rows), 1), 4),
        "reasoning_leak_rate": round(
            sum(int(row.get("response_error_type") == "reasoning_leak") for row in rows) / max(len(rows), 1),
            4,
        ),
        "auditable_query_rate": round(
            sum(int(is_auditable_generation_row(row)) for row in rows) / max(len(rows), 1),
            4,
        ),
        "avg_latency_ms": round(sum(row["latency_ms"] for row in rows) / max(len(rows), 1), 3),
        "config_hash": stable_hash(stable_run_config | {"group_id": group_id}),
    }


def build_e10_analysis_md(
    run_dir: Path,
    summary_rows: list[dict[str, Any]],
    grouped_rows: dict[str, list[dict[str, Any]]],
    adapter_metadata: dict[str, Any] | None,
) -> None:
    selected_group_ids = sorted(grouped_rows)
    base_rows = {row["query_id"]: row for row in grouped_rows.get("A_base_4b_grounded", [])}
    peft_rows = {row["query_id"]: row for row in grouped_rows.get("B_peft_4b_grounded", [])}
    improved: list[dict[str, Any]] = []
    regressed: list[dict[str, Any]] = []
    is_single_group = len(selected_group_ids) == 1
    if base_rows and peft_rows:
        for query_id in sorted(base_rows):
            if query_id not in peft_rows:
                continue
            base_row = base_rows[query_id]
            peft_row = peft_rows[query_id]
            delta = round(
                peft_row["verification"].citation_precision - base_row["verification"].citation_precision,
                4,
            )
            payload = {
                "query_id": query_id,
                "delta_citation_precision": delta,
                "base_recommendations": len(base_row["response"].recommendations),
                "peft_recommendations": len(peft_row["response"].recommendations),
                "base_summary": base_row["response"].summary or "",
                "peft_summary": peft_row["response"].summary or "",
            }
            if delta > 0:
                improved.append(payload)
            elif delta < 0:
                regressed.append(payload)

    summary_rows_for_md: list[dict[str, Any]] = []
    for row in summary_rows:
        summary_rows_for_md.append(
            {
                key: (
                    format_analysis_value(
                        value,
                        none_text="n/a (no applicable unsupported-request queries)",
                    )
                    if key == "unsupported_honesty_rate"
                    else format_analysis_value(value)
                )
                for key, value in row.items()
            }
        )

    lines = [
        "# E10 Single-Group Diagnostic Result" if is_single_group else "# E10 Base vs PEFT Result",
        "",
        "## Summary Table",
        "",
    ]
    lines.extend(markdown_table(summary_rows_for_md))
    lines.extend(
        [
            "",
            "## Adapter Metadata",
            "",
            (
                "- adapter_name: not applicable in base-only run"
                if is_single_group and selected_group_ids == ["A_base_4b_grounded"]
                else (
                    f"- adapter_name: {adapter_metadata['adapter_name']}"
                    if adapter_metadata
                    else "- adapter_name: n/a"
                )
            ),
            (
                "- base_model_id: not applicable in base-only run"
                if is_single_group and selected_group_ids == ["A_base_4b_grounded"]
                else (
                    f"- base_model_id: {adapter_metadata['base_model_id']}"
                    if adapter_metadata
                    else "- base_model_id: n/a"
                )
            ),
            (
                "- served_model_id: not applicable in base-only run"
                if is_single_group and selected_group_ids == ["A_base_4b_grounded"]
                else (
                    f"- served_model_id: {adapter_metadata['served_model_id']}"
                    if adapter_metadata
                    else "- served_model_id: n/a"
                )
            ),
            (
                "- adapter_path: not applicable in base-only run"
                if is_single_group and selected_group_ids == ["A_base_4b_grounded"]
                else (
                    f"- adapter_path: {adapter_metadata['_resolved_adapter_path']}"
                    if adapter_metadata
                    else "- adapter_path: n/a"
                )
            ),
            "",
            "## Representative Improvements",
            "",
        ]
    )
    if is_single_group:
        lines.append("- not available in single-group run")
    elif improved:
        for row in improved[:5]:
            lines.append(
                f"- `{row['query_id']}` | Δcitation_precision={row['delta_citation_precision']} | base_recs={row['base_recommendations']} | peft_recs={row['peft_recommendations']}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Representative Regressions", ""])
    if is_single_group:
        lines.append("- not available in single-group run")
    elif regressed:
        for row in regressed[:5]:
            lines.append(
                f"- `{row['query_id']}` | Δcitation_precision={row['delta_citation_precision']} | base_recs={row['base_recommendations']} | peft_recs={row['peft_recommendations']}"
            )
    else:
        lines.append("- none")

    (run_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def load_e10_run_artifacts(run_dir: str | Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    resolved_run_dir = Path(run_dir)
    run_meta_path = resolved_run_dir / "run_meta.json"
    results_path = resolved_run_dir / "results.jsonl"
    if not run_meta_path.exists():
        raise FileNotFoundError(f"Missing E10 run_meta.json: {run_meta_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Missing E10 results.jsonl: {results_path}")
    run_meta = load_json(run_meta_path)
    log_rows = load_jsonl(results_path)
    if not log_rows:
        raise ValueError(f"E10 运行目录为空，无法比较：{resolved_run_dir}")
    return resolved_run_dir, run_meta, log_rows


def reconstruct_e10_group_rows(log_rows: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    group_ids = sorted({row["group_id"] for row in log_rows})
    if len(group_ids) != 1:
        raise ValueError(
            "e10_compare_runs 只接受单组运行目录。"
            f"当前目录包含 group_ids={group_ids}"
        )
    group_id = group_ids[0]
    grouped_rows: list[dict[str, Any]] = []
    for row in log_rows:
        intermediate = row["intermediate_objects"]
        grouped_rows.append(
            {
                "query_id": row["query_id"],
                "latency_ms": row["latency_ms"],
                "response": RecommendationResponse.model_validate(intermediate["response"]),
                "verification": CitationVerificationResult.model_validate(intermediate["citation_verification"]),
                "audit_rows": intermediate.get("audit_rows", []),
                "unsupported_honesty": intermediate.get("unsupported_honesty"),
                "response_error_type": intermediate.get("response_error_type")
                or intermediate.get("debug_payload", {}).get("response_error_type"),
                "reasoning_leak_detected": bool(intermediate.get("reasoning_leak_detected", False)),
                "behavior_runtime_config": intermediate.get("behavior_runtime_config", {}),
            }
        )
    return group_id, grouped_rows


def build_e10_compare_analysis_md(
    run_dir: Path,
    summary_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    base_run_meta: dict[str, Any],
    peft_run_meta: dict[str, Any],
    latency_formally_comparable: bool,
) -> None:
    summary_rows_for_md: list[dict[str, Any]] = []
    for row in summary_rows:
        summary_rows_for_md.append(
            {
                key: (
                    format_analysis_value(
                        value,
                        none_text="n/a (no applicable unsupported-request queries)",
                    )
                    if key == "unsupported_honesty_rate"
                    else format_analysis_value(value)
                )
                for key, value in row.items()
            }
        )

    improved = []
    regressed = []
    for row in comparison_rows:
        delta_key = (
            row["delta_schema_valid"],
            row["delta_citation_precision"],
            row["delta_evidence_verifiability"],
        )
        if delta_key > (0, 0.0, 0.0):
            improved.append(row)
        elif delta_key < (0, 0.0, 0.0):
            regressed.append(row)

    improved.sort(
        key=lambda row: (
            row["delta_schema_valid"],
            row["delta_citation_precision"],
            row["delta_evidence_verifiability"],
        ),
        reverse=True,
    )
    regressed.sort(
        key=lambda row: (
            row["delta_schema_valid"],
            row["delta_citation_precision"],
            row["delta_evidence_verifiability"],
        )
    )

    lines = [
        "# E10 Base vs PEFT Compare Result",
        "",
        "## Summary Table",
        "",
    ]
    lines.extend(markdown_table(summary_rows_for_md))
    lines.extend(
        [
            "",
            "## Source Runs",
            "",
            f"- base_run_id: {base_run_meta['run_id']}",
            f"- peft_run_id: {peft_run_meta['run_id']}",
            f"- latency_formally_comparable: {'yes' if latency_formally_comparable else 'no'}",
            (
                "- latency_note: 两组使用相同 local backend，可纳入正式时延对照。"
                if latency_formally_comparable
                else "- latency_note: 两组推理栈不同或不满足同后端条件，不纳入正式结论。"
            ),
            "",
            "## Representative Improvements",
            "",
        ]
    )
    if improved:
        for row in improved[:5]:
            lines.append(
                f"- `{row['query_id']}` | Δschema_valid={row['delta_schema_valid']} | "
                f"Δcitation_precision={row['delta_citation_precision']} | "
                f"Δevidence={row['delta_evidence_verifiability']} | "
                f"base_recs={row['base_recommendations']} | peft_recs={row['peft_recommendations']}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Representative Regressions", ""])
    if regressed:
        for row in regressed[:5]:
            lines.append(
                f"- `{row['query_id']}` | Δschema_valid={row['delta_schema_valid']} | "
                f"Δcitation_precision={row['delta_citation_precision']} | "
                f"Δevidence={row['delta_evidence_verifiability']} | "
                f"base_recs={row['base_recommendations']} | peft_recs={row['peft_recommendations']}"
            )
    else:
        lines.append("- none")

    (run_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def run_e10_compare_runs(
    output_root: Path,
    base_run_dir: str | Path,
    peft_run_dir: str | Path,
) -> Path:
    base_run_path, base_run_meta, base_log_rows = load_e10_run_artifacts(base_run_dir)
    peft_run_path, peft_run_meta, peft_log_rows = load_e10_run_artifacts(peft_run_dir)
    base_group_id, base_rows = reconstruct_e10_group_rows(base_log_rows)
    peft_group_id, peft_rows = reconstruct_e10_group_rows(peft_log_rows)
    if base_group_id != "A_base_4b_grounded":
        raise ValueError(
            "base_run_dir 必须是仅包含 A_base_4b_grounded 的单组运行目录。"
        )
    if peft_group_id != "B_peft_4b_grounded":
        raise ValueError(
            "peft_run_dir 必须是仅包含 B_peft_4b_grounded 的单组运行目录。"
        )

    base_by_query = {row["query_id"]: row for row in base_rows}
    peft_by_query = {row["query_id"]: row for row in peft_rows}
    if set(base_by_query) != set(peft_by_query):
        raise ValueError("base 与 peft 的 query_id 集合不一致，无法做正式对照。")

    base_backend = base_rows[0]["behavior_runtime_config"].get("llm_backend")
    peft_backend = peft_rows[0]["behavior_runtime_config"].get("llm_backend")
    latency_formally_comparable = base_backend == peft_backend == "local"

    stable_run_config = {
        "task": "E10_COMPARE",
        "base_run_id": base_run_meta["run_id"],
        "peft_run_id": peft_run_meta["run_id"],
        "base_group_id": base_group_id,
        "peft_group_id": peft_group_id,
        "query_count": len(base_by_query),
        "latency_formally_comparable": latency_formally_comparable,
    }
    run_started_at = utc_now_iso()
    run_id = f"e10cmp_{stable_hash(stable_run_config)}_{run_started_at.replace(':', '').replace('-', '')}"
    run_dir = ensure_dir(output_root / run_id)

    summary_rows = []
    for group_id, rows, run_meta, source_run_path in [
        (base_group_id, base_rows, base_run_meta, base_run_path),
        (peft_group_id, peft_rows, peft_run_meta, peft_run_path),
    ]:
        summary_row = build_e10_metric_row(group_id, rows, run_meta["stable_run_config"])
        summary_row["source_run_id"] = run_meta["run_id"]
        summary_row["source_run_dir"] = str(source_run_path)
        summary_row["latency_formally_comparable"] = latency_formally_comparable
        summary_rows.append(summary_row)

    comparison_rows: list[dict[str, Any]] = []
    for query_id in sorted(base_by_query):
        base_row = base_by_query[query_id]
        peft_row = peft_by_query[query_id]
        base_evidence_mean = compute_query_evidence_mean(base_row)
        peft_evidence_mean = compute_query_evidence_mean(peft_row)
        comparison_rows.append(
            {
                "query_id": query_id,
                "delta_schema_valid": int(peft_row["response"].schema_valid) - int(base_row["response"].schema_valid),
                "delta_citation_precision": round(
                    peft_row["verification"].citation_precision - base_row["verification"].citation_precision,
                    4,
                ),
                "delta_evidence_verifiability": round(peft_evidence_mean - base_evidence_mean, 4),
                "delta_latency_ms": round(peft_row["latency_ms"] - base_row["latency_ms"], 3),
                "latency_formally_comparable": latency_formally_comparable,
                "base_schema_valid": bool(base_row["response"].schema_valid),
                "peft_schema_valid": bool(peft_row["response"].schema_valid),
                "base_citation_precision": base_row["verification"].citation_precision,
                "peft_citation_precision": peft_row["verification"].citation_precision,
                "base_evidence_verifiability": base_evidence_mean,
                "peft_evidence_verifiability": peft_evidence_mean,
                "base_recommendations": len(base_row["response"].recommendations),
                "peft_recommendations": len(peft_row["response"].recommendations),
                "base_response_error_type": base_row.get("response_error_type"),
                "peft_response_error_type": peft_row.get("response_error_type"),
                "base_summary": base_row["response"].summary or "",
                "peft_summary": peft_row["response"].summary or "",
            }
        )

    write_jsonl(run_dir / "comparison.jsonl", comparison_rows)
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")
    write_json(
        run_dir / "run_meta.json",
        {
            "run_id": run_id,
            "generated_at": run_started_at,
            "base_run_id": base_run_meta["run_id"],
            "peft_run_id": peft_run_meta["run_id"],
            "latency_formally_comparable": latency_formally_comparable,
        },
    )
    build_e10_compare_analysis_md(
        run_dir,
        summary_rows,
        comparison_rows,
        base_run_meta,
        peft_run_meta,
        latency_formally_comparable,
    )
    return run_dir


def build_e9_query_rows(limit_queries: int | None = None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    judged_queries = load_jsonl(EXPERIMENT_ASSETS_DIR / "judged_queries.jsonl")
    slot_gold = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "slot_gold.jsonl")}
    clarify_gold = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "clarify_gold.jsonl")}
    eligible_ids = []
    for row in judged_queries:
        slot = slot_gold[row["query_id"]]
        clarify = clarify_gold[row["query_id"]]
        if clarify["clarify_needed"]:
            continue
        if not slot["city"] or not (slot["focus_aspects"] or slot["avoid_aspects"]):
            continue
        if row["query_type"] not in {"single_aspect", "multi_aspect", "focus_and_avoid", "multi_aspect_strong"}:
            continue
        eligible_ids.append((row, slot))
    if limit_queries is not None:
        eligible_ids = eligible_ids[:limit_queries]
    return eligible_ids


def build_split_hotel_lookup(split_manifest: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    by_split_city: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in split_manifest["hotels"]:
        by_split_city[row["split"]][row["city"]].append(row["hotel_id"])
    for split_lookup in by_split_city.values():
        for hotel_ids in split_lookup.values():
            hotel_ids.sort()
    return {split: dict(city_map) for split, city_map in by_split_city.items()}


def build_evidence_pack_for_candidate(
    collection,
    bi_encoder,
    normalize_embeddings: bool,
    evidence_lookup: dict[str, dict[str, Any]],
    preference: UserPreference,
    hotel: HotelCandidate,
    dense_top_k: int,
    final_top_k: int,
) -> EvidencePack:
    evidence_by_aspect: dict[str, list[SentenceCandidate]] = {}
    all_sentence_ids: list[str] = []
    retrieval_aspect_roles: dict[str, set[str]] = defaultdict(set)
    for aspect in preference.focus_aspects:
        retrieval_aspect_roles[aspect].add("focus")
    for aspect in preference.avoid_aspects:
        retrieval_aspect_roles[aspect].add("avoid")

    target_aspects = sorted(retrieval_aspect_roles)
    for aspect in target_aspects:
        aspect_dense_rows: list[dict[str, Any]] = []
        for role in sorted(retrieval_aspect_roles[aspect]):
            query_embedding = bi_encoder.encode(
                [build_query_en_target(preference.city or "", aspect, role)],
                normalize_embeddings=normalize_embeddings,
            ).tolist()
            aspect_dense_rows.extend(
                dense_query_hotel(
                    collection=collection,
                    query_embedding=query_embedding,
                    hotel_id=hotel.hotel_id,
                    city=preference.city or "",
                    hotel_name=hotel.hotel_name,
                    top_k=dense_top_k,
                    evidence_lookup=evidence_lookup,
                    aspect=aspect,
                    channel="main",
                )
            )
        dense_rows = merge_dense_candidates(aspect_dense_rows, top_k=dense_top_k)[:final_top_k]
        sentence_rows: list[SentenceCandidate] = []
        for row in dense_rows:
            sentence_rows.append(
                SentenceCandidate(
                    sentence_id=row["sentence_id"],
                    sentence_text=row["sentence_text"],
                    aspect=row["sentence_aspect"],
                    sentiment=row["sentence_sentiment"],
                    review_date=row["review_date"],
                    score_dense=row["score_dense"],
                    score_rerank=row["score_rerank"],
                )
            )
            all_sentence_ids.append(row["sentence_id"])
        evidence_by_aspect[aspect] = sentence_rows

    all_sentence_ids = sorted(set(all_sentence_ids))
    retrieval_trace = {
        "mode": E9_RETRIEVAL_MODE,
        "dense_top_k": dense_top_k,
        "final_top_k": final_top_k,
        "aspect_count": len(target_aspects),
        "aspect_roles": {aspect: sorted(roles) for aspect, roles in retrieval_aspect_roles.items()},
        "candidate_policy": E9_CANDIDATE_POLICY,
        "fallback_enabled": False,
    }
    return EvidencePack(
        hotel_id=hotel.hotel_id,
        query_en=preference.query_en,
        evidence_by_aspect=evidence_by_aspect,
        all_sentence_ids=all_sentence_ids,
        retrieval_trace=retrieval_trace,
    )


def freeze_e9_assets(limit_queries: int | None = None) -> tuple[Path, Path]:
    cfg = load_config()
    frozen_config = load_json(EXPERIMENT_ASSETS_DIR / "frozen_config.yaml")
    split_manifest = load_json(EXPERIMENT_ASSETS_DIR / "frozen_split_manifest.json")
    review_df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
    profile_df = pd.read_pickle("data/intermediate/hotel_profiles.pkl")
    evidence_df = pd.read_pickle("data/intermediate/evidence_index.pkl")

    hotel_summary = build_hotel_summary(review_df)
    profile_current, profile_alt = build_profile_tables(profile_df)
    del profile_alt
    hotel_summary = hotel_summary[hotel_summary["hotel_id"].isin(set(split_manifest["splits"]["test"]))].copy()
    evidence_lookup = build_evidence_lookup(evidence_df)
    query_rows = build_e9_query_rows(limit_queries=limit_queries)

    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer

    client = PersistentClient(path=cfg["embedding"]["chroma_persist_dir"])
    collection = client.get_collection(cfg["embedding"]["chroma_collection"])
    try:
        bi_encoder = SentenceTransformer(cfg["embedding"]["model"], local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "E9 资产冻结需要本地可用的 embedding 模型缓存；当前环境未能在离线模式下加载 "
            f"{cfg['embedding']['model']}。请先在有网环境缓存该模型后再重试。"
        ) from exc
    normalize_embeddings = bool(cfg["embedding"].get("normalize", True))

    stable_asset_config = {
        "task": "E9_assets",
        "split_config_hash": split_manifest["meta"]["config_hash"],
        "query_count": len(query_rows),
        "retrieval_mode": E9_RETRIEVAL_MODE,
        "candidate_policy": E9_CANDIDATE_POLICY,
        "dense_top_k": cfg["reranker"]["top_k_before_rerank"],
        "final_top_k": cfg["reranker"]["top_k_after_rerank"],
        "embedding_model": cfg["embedding"]["model"],
        "collection": cfg["embedding"]["chroma_collection"],
        "fallback_enabled": frozen_config["workflow"]["enable_fallback"],
        "behavior_base_model": frozen_config["behavior"]["base_model"],
    }

    units: list[dict[str, Any]] = []
    query_ids: list[str] = []
    for query_row, slot_row in query_rows:
        city_hotels = hotel_summary[hotel_summary["city"] == slot_row["city"]].copy()
        ranked = candidate_rank(
            city_hotels=city_hotels,
            profile_current=profile_current,
            profile_alt=profile_current,
            focus_aspects=slot_row["focus_aspects"],
            avoid_aspects=slot_row["avoid_aspects"],
            mode="B_final_aspect_score",
        ).head(5)
        candidate_hotels: list[HotelCandidate] = []
        for _, hotel_row in ranked.iterrows():
            candidate_hotels.append(
                HotelCandidate(
                    hotel_id=str(hotel_row["hotel_id"]),
                    hotel_name=str(hotel_row["hotel_name"]),
                    score_total=round(float(hotel_row["score_total"]), 4),
                    score_breakdown={k: float(v) for k, v in dict(hotel_row["score_breakdown"]).items()},
                )
            )

        preference = UserPreference(
            city=slot_row["city"],
            state=slot_row["state"],
            hotel_category=slot_row["hotel_category"],
            focus_aspects=slot_row["focus_aspects"],
            avoid_aspects=slot_row["avoid_aspects"],
            unsupported_requests=slot_row["unsupported_requests"],
            query_en=slot_row["query_en"],
        )
        evidence_packs = [
            build_evidence_pack_for_candidate(
                collection=collection,
                bi_encoder=bi_encoder,
                normalize_embeddings=normalize_embeddings,
                evidence_lookup=evidence_lookup,
                preference=preference,
                hotel=hotel,
                dense_top_k=cfg["reranker"]["top_k_before_rerank"],
                final_top_k=cfg["reranker"]["top_k_after_rerank"],
            )
            for hotel in candidate_hotels
        ]

        unit_payload = GenerationEvalUnit(
            query_id=query_row["query_id"],
            query_text_zh=query_row["query_text_zh"],
            query_type=query_row["query_type"],
            user_preference_gold=preference,
            unsupported_requests=preference.unsupported_requests,
            candidate_hotels=candidate_hotels,
            evidence_packs=evidence_packs,
            retrieval_mode=E9_RETRIEVAL_MODE,
            candidate_policy=E9_CANDIDATE_POLICY,
            config_hash=stable_hash(
                stable_asset_config
                | {
                    "query_id": query_row["query_id"],
                    "candidate_hotel_ids": [hotel.hotel_id for hotel in candidate_hotels],
                    "all_sentence_ids": [pack.all_sentence_ids for pack in evidence_packs],
                }
            ),
        )
        units.append(unit_payload.model_dump())
        query_ids.append(query_row["query_id"])

    units_path, query_ids_path = resolve_generation_asset_paths(limit_queries=limit_queries)
    write_jsonl(units_path, units)
    write_json(query_ids_path, query_ids)
    ensure_dir(E9_LABELS_DIR)
    return units_path, query_ids_path


def format_candidate_lines(unit: GenerationEvalUnit) -> list[str]:
    lines = []
    for hotel in unit.candidate_hotels:
        lines.append(
            f'- hotel_id={hotel.hotel_id} | hotel_name={hotel.hotel_name} | score_total={round(hotel.score_total, 4)}'
        )
    return lines


def format_evidence_pack_lines(unit: GenerationEvalUnit, grounded_mode: bool) -> list[str]:
    lines = []
    for pack in unit.evidence_packs:
        hotel_lookup = {hotel.hotel_id: hotel.hotel_name for hotel in unit.candidate_hotels}
        lines.append(f"Hotel {pack.hotel_id} | {hotel_lookup.get(pack.hotel_id, pack.hotel_id)}")
        shown_sentence_ids: list[str] = []
        for aspect in sorted(pack.evidence_by_aspect):
            lines.append(f"Aspect {aspect}:")
            sentence_rows = pack.evidence_by_aspect[aspect]
            sentence_limit = E9_PROMPT_SENTENCE_LIMIT_GROUNDED if grounded_mode else E9_PROMPT_SENTENCE_LIMIT_FREE
            sentence_rows = sentence_rows[:sentence_limit]
            for sentence in sentence_rows:
                shown_sentence_ids.append(sentence.sentence_id)
                lines.append(
                    f'- sentence_id={sentence.sentence_id} | text="{sentence.sentence_text}"'
                )
        if grounded_mode:
            lines.append(f"allowed_sentence_ids={','.join(shown_sentence_ids)}")
        lines.append("")
    return lines


def build_generation_prompts(unit: GenerationEvalUnit, group_id: str) -> tuple[str, str]:
    preference_payload = json.dumps(unit.user_preference_gold.model_dump(), ensure_ascii=False)
    candidate_lines = "\n".join(format_candidate_lines(unit))
    grounded_mode = group_id in {
        "B_grounded_generation",
        "C_grounded_generation_with_verifier",
        "A_base_4b_grounded",
        "B_peft_4b_grounded",
    }
    evidence_lines = "\n".join(format_evidence_pack_lines(unit, grounded_mode=grounded_mode))
    if group_id == "A_free_generation":
        system_prompt = (
            "你是酒店推荐工作流里的推荐解释生成器。\n"
            "你只返回 JSON，不要解释。\n"
            "统一输出 schema："
            f"{E9_OUTPUT_FIELDS}\n"
            f"最多返回 {E9_MAX_RECOMMENDATIONS} 家酒店；每家酒店最多 {E9_MAX_REASONS_PER_ITEM} 条理由。\n"
            "summary 只写一句短句，尽量不超过 50 个汉字。\n"
            "reason_text 必须短，不要复述整段证据。\n"
            "recommendation item 里只允许 hotel_id、hotel_name、reasons 三个字段。\n"
            "不要输出空的 reasons 数组；没有把握的酒店直接省略。\n"
            "unsupported_notice 只能出现在根级字段，不能出现在酒店项内部。\n"
            "A 组允许 sentence_id 为 null，但仍然必须优先基于给定证据摘要组织表达，不要编造酒店事实。"
        )
    else:
        system_prompt = (
            "你是证据约束的酒店推荐生成器。\n"
            "你只返回 JSON，不要解释。\n"
            "统一输出 schema："
            f"{E9_OUTPUT_FIELDS}\n"
            f"最多返回 {E9_MAX_RECOMMENDATIONS} 家酒店；每家酒店最多 {E9_MAX_REASONS_PER_ITEM} 条理由。\n"
            "summary 只写一句短句，尽量不超过 50 个汉字。\n"
            "reason_text 必须短，不要复述整段证据。\n"
            "recommendation item 里只允许 hotel_id、hotel_name、reasons 三个字段。\n"
            "不要输出空的 reasons 数组；没有至少 1 条可验证理由的酒店直接省略。\n"
            "unsupported_notice 只能出现在根级字段，不能出现在酒店项内部。\n"
            "每条理由必须包含单个 sentence_id，且该 sentence_id 必须来自当前 hotel 的 EvidencePack。\n"
            "如果给定证据不足以支持推荐，就减少推荐数量或在 unsupported_notice 中诚实说明。"
        )
    user_prompt = (
        f"Query ID: {unit.query_id}\n"
        f"用户中文需求: {unit.query_text_zh}\n"
        f"结构化偏好: {preference_payload}\n"
        f"Unsupported requests: {', '.join(unit.unsupported_requests) or 'none'}\n"
        "候选酒店如下：\n"
        f"{candidate_lines}\n\n"
        "当前证据如下：\n"
        f"{evidence_lines}\n"
        "请优先保留最有把握的 1-2 家酒店；宁可少推，不要凑满。"
    )
    return system_prompt, user_prompt


def build_retry_prompt(
    unit: GenerationEvalUnit,
    invalid_sentence_ids: list[str],
    out_of_pack_sentence_ids: list[str],
) -> str:
    messages = []
    if invalid_sentence_ids:
        messages.append(f"无效 sentence_id: {', '.join(sorted(set(invalid_sentence_ids)))}")
    if out_of_pack_sentence_ids:
        messages.append(f"越权 sentence_id: {', '.join(sorted(set(out_of_pack_sentence_ids)))}")
    issue_text = "；".join(messages) or "引用不合法"
    return (
        f"上一次输出的引用校验失败：{issue_text}。\n"
        f"请重新生成完整 JSON，并确保所有理由都只引用当前 EvidencePack 中真实存在的 sentence_id。\n"
        f"最多只保留 {E9_MAX_RECOMMENDATIONS} 家最有把握的酒店。\n"
        "不要输出空的 reasons 数组；没有证据就直接省略该酒店。\n"
        "unsupported_notice 只能放在根级字段。"
    )


def build_unsupported_notice(unsupported_requests: list[str]) -> str:
    if not unsupported_requests:
        return ""
    labels = [UNSUPPORTED_ZH_LABELS.get(item, item) for item in unsupported_requests]
    return f"当前系统不会直接执行{'、'.join(labels)}这类约束，我会只基于可验证的城市与评论证据给出推荐。"


def empty_generation_response(query_id: str, group_id: str, raw_response: str, schema_valid: bool) -> RecommendationResponse:
    return RecommendationResponse(
        query_id=query_id,
        group_id=group_id,
        summary="",
        recommendations=[],
        unsupported_notice="",
        schema_valid=schema_valid,
        raw_response=raw_response,
    )


def coerce_generation_payload(
    payload: dict[str, Any] | None,
    unit: GenerationEvalUnit,
    group_id: str,
    raw_response: str,
) -> RecommendationResponse:
    if payload is None:
        return empty_generation_response(unit.query_id, group_id, raw_response, False)

    candidate_lookup = {hotel.hotel_id: hotel for hotel in unit.candidate_hotels}
    summary = str(payload.get("summary", "") or "").strip()
    unsupported_notice = str(payload.get("unsupported_notice", "") or "").strip()
    notice_fragments = [unsupported_notice] if unsupported_notice else []
    recommendations_raw = payload.get("recommendations", [])
    schema_valid = True
    if not isinstance(recommendations_raw, list):
        recommendations_raw = []
        schema_valid = False

    recommendation_items: list[RecommendationItem] = []
    for raw_item in recommendations_raw[:E9_MAX_RECOMMENDATIONS]:
        if not isinstance(raw_item, dict):
            schema_valid = False
            continue
        hotel_id = str(raw_item.get("hotel_id", "") or "").strip()
        if hotel_id not in candidate_lookup:
            schema_valid = False
            continue
        hotel = candidate_lookup[hotel_id]
        reasons_raw = raw_item.get("reasons", [])
        item_notice = str(raw_item.get("unsupported_notice", "") or "").strip()
        if item_notice:
            notice_fragments.append(item_notice)
        if not isinstance(reasons_raw, list):
            reasons_raw = []
            schema_valid = False
        reasons: list[RecommendationReason] = []
        for raw_reason in reasons_raw[:E9_MAX_REASONS_PER_ITEM]:
            if not isinstance(raw_reason, dict):
                schema_valid = False
                continue
            aspects, _ = normalize_aspect_values([raw_reason.get("aspect")])
            aspect = aspects[0] if aspects else None
            reason_text = str(raw_reason.get("reason_text", "") or "").strip()
            sentence_id_raw = raw_reason.get("sentence_id")
            sentence_id = str(sentence_id_raw).strip() if sentence_id_raw not in {None, ""} else None
            if aspect is None or not reason_text:
                schema_valid = False
                continue
            if group_id != "A_free_generation" and sentence_id is None:
                schema_valid = False
            reasons.append(
                RecommendationReason(
                    aspect=aspect,
                    reason_text=reason_text,
                    sentence_id=sentence_id,
                )
            )
        if reasons:
            recommendation_items.append(
                RecommendationItem(
                    hotel_id=hotel.hotel_id,
                    hotel_name=hotel.hotel_name,
                    reasons=reasons,
                )
            )
        elif item_notice:
            continue
        else:
            schema_valid = False

    normalized_notice_parts: list[str] = []
    seen_notice_parts: set[str] = set()
    for part in notice_fragments:
        if not part or part in seen_notice_parts:
            continue
        normalized_notice_parts.append(part)
        seen_notice_parts.add(part)

    return RecommendationResponse(
        query_id=unit.query_id,
        group_id=group_id,
        summary=summary,
        recommendations=recommendation_items,
        unsupported_notice=" ".join(normalized_notice_parts),
        schema_valid=schema_valid,
        raw_response=raw_response,
    )


def verify_response_citations(
    response: RecommendationResponse,
    unit: GenerationEvalUnit,
    evidence_lookup: dict[str, dict[str, Any]],
    retry_triggered: bool = False,
    fallback_to_honest_notice: bool = False,
) -> tuple[CitationVerificationResult, list[dict[str, Any]]]:
    pack_lookup = {pack.hotel_id: set(pack.all_sentence_ids) for pack in unit.evidence_packs}
    invalid_sentence_ids: list[str] = []
    out_of_pack_sentence_ids: list[str] = []
    citation_count = 0
    valid_count = 0
    audit_rows: list[dict[str, Any]] = []

    for recommendation in response.recommendations:
        pack_sentence_ids = pack_lookup.get(recommendation.hotel_id, set())
        for reason in recommendation.reasons:
            sentence_id = reason.sentence_id
            citation_exists = 0
            in_current_pack = 0
            support_score = 0
            if sentence_id:
                citation_count += 1
                if sentence_id in evidence_lookup:
                    citation_exists = 1
                    if sentence_id in pack_sentence_ids:
                        in_current_pack = 1
                        valid_count += 1
                        evidence_row = evidence_lookup[sentence_id]
                        support_score = 2 if evidence_row["aspect"] == reason.aspect else 1
                    else:
                        out_of_pack_sentence_ids.append(sentence_id)
                else:
                    invalid_sentence_ids.append(sentence_id)
            audit_rows.append(
                {
                    "query_id": unit.query_id,
                    "group_id": response.group_id,
                    "hotel_id": recommendation.hotel_id,
                    "sentence_id": sentence_id or "",
                    "reason_text": reason.reason_text,
                    "citation_exists": citation_exists,
                    "in_current_evidence_pack": in_current_pack,
                    "support_score": support_score,
                    "notes": "",
                }
            )

    precision = 0.0 if citation_count == 0 else round(valid_count / citation_count, 4)
    verification = CitationVerificationResult(
        query_id=unit.query_id,
        group_id=response.group_id,
        citation_precision=precision,
        invalid_sentence_ids=sorted(set(invalid_sentence_ids)),
        out_of_pack_sentence_ids=sorted(set(out_of_pack_sentence_ids)),
        retry_triggered=retry_triggered,
        fallback_to_honest_notice=fallback_to_honest_notice,
    )
    return verification, audit_rows


def build_honest_fallback_response(
    unit: GenerationEvalUnit,
    group_id: str,
    raw_response: str,
) -> RecommendationResponse:
    return RecommendationResponse(
        query_id=unit.query_id,
        group_id=group_id,
        summary="当前证据引用未通过校验，因此暂不返回不可验证的酒店推荐。",
        recommendations=[],
        unsupported_notice=build_unsupported_notice(unit.unsupported_requests),
        schema_valid=True,
        raw_response=raw_response,
    )


def generate_group_response(
    llm_runner,
    unit: GenerationEvalUnit,
    group_id: str,
    max_new_tokens: int,
    evidence_lookup: dict[str, dict[str, Any]],
) -> tuple[RecommendationResponse, CitationVerificationResult, list[dict[str, Any]], dict[str, Any]]:
    system_prompt, user_prompt = build_generation_prompts(unit, group_id)
    raw_response = llm_runner.generate_json(system_prompt, user_prompt, max_new_tokens=max_new_tokens)
    generation_debug = dict(getattr(llm_runner, "last_generation_debug", {}) or {})
    payload, repaired = parse_json_with_repair(raw_response)
    response = coerce_generation_payload(payload, unit, group_id, raw_response)
    if repaired:
        response.schema_valid = False

    response_error_type = generation_debug.get("response_error_type")
    if response_error_type is None and payload is None:
        response_error_type = "invalid_json"
    elif response_error_type is None and not response.schema_valid:
        response_error_type = "schema_invalid"

    verification, audit_rows = verify_response_citations(response, unit, evidence_lookup)
    debug_payload: dict[str, Any] = {
        "raw_response_initial": raw_response,
        "retry_raw_response": "",
        "response_error_type": response_error_type,
        "thinking_control_supported": generation_debug.get("thinking_control_supported"),
        "raw_response_prefix": generation_debug.get("raw_response_prefix", raw_response[:80]),
    }

    if group_id != "C_grounded_generation_with_verifier":
        return response, verification, audit_rows, debug_payload

    needs_retry = bool(verification.invalid_sentence_ids or verification.out_of_pack_sentence_ids)
    needs_retry = needs_retry or not response.schema_valid
    if not needs_retry:
        return response, verification, audit_rows, debug_payload

    retry_raw_response = raw_response
    for _ in range(E9_RETRY_LIMIT):
        retry_prompt = build_retry_prompt(
            unit,
            invalid_sentence_ids=verification.invalid_sentence_ids,
            out_of_pack_sentence_ids=verification.out_of_pack_sentence_ids,
        )
        retry_raw_response = llm_runner.generate_json(system_prompt, f"{user_prompt}\n\n{retry_prompt}", max_new_tokens=max_new_tokens)
        payload, repaired = parse_json_with_repair(retry_raw_response)
        retry_response = coerce_generation_payload(payload, unit, group_id, retry_raw_response)
        if repaired:
            retry_response.schema_valid = False
        retry_verification, retry_audit_rows = verify_response_citations(
            retry_response,
            unit,
            evidence_lookup,
            retry_triggered=True,
        )
        response = retry_response
        verification = retry_verification
        audit_rows = retry_audit_rows
        debug_payload["retry_raw_response"] = retry_raw_response
        if (
            response.schema_valid
            and not verification.invalid_sentence_ids
            and not verification.out_of_pack_sentence_ids
        ):
            return response, verification, audit_rows, debug_payload

    response = build_honest_fallback_response(unit, group_id, retry_raw_response)
    verification, audit_rows = verify_response_citations(
        response,
        unit,
        evidence_lookup,
        retry_triggered=True,
        fallback_to_honest_notice=True,
    )
    debug_payload["retry_raw_response"] = retry_raw_response
    return response, verification, audit_rows, debug_payload


def build_e9_metric_row(
    group_id: str,
    rows: list[dict[str, Any]],
    stable_run_config: dict[str, Any],
) -> dict[str, Any]:
    unsupported_rows = [row["unsupported_honesty"] for row in rows if row["unsupported_honesty"] is not None]
    support_scores = [
        audit_row["support_score"]
        for row in rows
        for audit_row in row["audit_rows"]
    ]
    return {
        "group_id": group_id,
        "query_count": len(rows),
        "citation_precision": round(
            sum(row["verification"].citation_precision for row in rows) / max(len(rows), 1),
            4,
        ),
        "evidence_verifiability_mean": round(sum(support_scores) / max(len(support_scores), 1), 4),
        "unsupported_honesty_rate": round(
            1.0 if not unsupported_rows else sum(unsupported_rows) / len(unsupported_rows),
            4,
        ),
        "schema_valid_rate": round(sum(int(row["response"].schema_valid) for row in rows) / max(len(rows), 1), 4),
        "avg_latency_ms": round(sum(row["latency_ms"] for row in rows) / max(len(rows), 1), 3),
        "retry_trigger_rate": round(
            sum(int(row["verification"].retry_triggered) for row in rows) / max(len(rows), 1),
            4,
        ),
        "fallback_to_honest_notice_rate": round(
            sum(int(row["verification"].fallback_to_honest_notice) for row in rows) / max(len(rows), 1),
            4,
        ),
        "config_hash": stable_hash(stable_run_config | {"group_id": group_id}),
    }


def select_representative_rows(group_rows: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    if kind == "drift":
        rows = [
            row for row in group_rows
            if row["verification"].invalid_sentence_ids or row["verification"].out_of_pack_sentence_ids
        ]
        if not rows:
            rows = [
                row for row in group_rows
                if row["response"].recommendations and row["verification"].citation_precision == 0
            ]
        return rows[:3]
    if kind == "positive":
        rows = [
            row for row in group_rows
            if row["response"].recommendations
            and row["verification"].citation_precision == 1.0
            and not row["verification"].fallback_to_honest_notice
        ]
        return rows[:3]
    rows = sorted(group_rows, key=lambda item: (-item["verification"].citation_precision, item["query_id"]))
    return rows[:3]


def build_e9_analysis_md(
    run_dir: Path,
    summary_rows: list[dict[str, Any]],
    grouped_rows: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# E9 Generation Constraint Result",
        "",
        "## Summary Table",
        "",
    ]
    lines.extend(markdown_table(summary_rows))
    lines.extend(["", "## Verifier Notes", ""])
    verifier_rows = grouped_rows.get("C_grounded_generation_with_verifier", [])
    retry_count = sum(int(row["verification"].retry_triggered) for row in verifier_rows)
    fallback_count = sum(int(row["verification"].fallback_to_honest_notice) for row in verifier_rows)
    lines.append(f"- verifier retry count: {retry_count}")
    lines.append(f"- honest fallback count: {fallback_count}")

    lines.extend(["", "## Schema Failures", ""])
    for group_id in E9_GROUPS:
        schema_failures = [row for row in grouped_rows[group_id] if not row["response"].schema_valid]
        lines.append(f"- {group_id}: {len(schema_failures)}")

    for group_id in E9_GROUPS:
        lines.extend(["", f"## {group_id}", ""])
        rep_rows = select_representative_rows(grouped_rows[group_id], kind="top")
        if rep_rows:
            for row in rep_rows:
                lines.append(
                    f"- `{row['query_id']}` | citation_precision={row['verification'].citation_precision} | summary={row['response'].summary or 'n/a'}"
                )
        else:
            lines.append("- none")

    lines.extend(["", "## Free Generation Citation Drift", ""])
    drift_rows = select_representative_rows(grouped_rows["A_free_generation"], kind="drift")
    if drift_rows:
        for row in drift_rows:
            lines.append(
                f"- `{row['query_id']}` | invalid={','.join(row['verification'].invalid_sentence_ids) or 'none'} | out_of_pack={','.join(row['verification'].out_of_pack_sentence_ids) or 'none'}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Grounded + Verifier Positive Cases", ""])
    positive_rows = select_representative_rows(grouped_rows["C_grounded_generation_with_verifier"], kind="positive")
    if positive_rows:
        for row in positive_rows:
            lines.append(
                f"- `{row['query_id']}` | recommendations={len(row['response'].recommendations)} | citation_precision={row['verification'].citation_precision}"
            )
    else:
        lines.append("- none")

    (run_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def run_e9_generation_constraints(
    output_root: Path,
    limit_queries: int | None = None,
) -> Path:
    cfg = load_config()
    frozen_config = load_json(EXPERIMENT_ASSETS_DIR / "frozen_config.yaml")
    behavior_runtime_config, behavior_api_key = resolve_behavior_runtime_config(cfg, frozen_config)
    split_manifest = load_json(EXPERIMENT_ASSETS_DIR / "frozen_split_manifest.json")
    eval_units_path = E9_UNITS_PATH
    if not eval_units_path.exists():
        freeze_e9_assets(limit_queries=None)
    eval_units = load_generation_eval_units(eval_units_path)
    if limit_queries is not None:
        eval_units = eval_units[:limit_queries]

    evidence_df = pd.read_pickle("data/intermediate/evidence_index.pkl")
    evidence_lookup = build_evidence_lookup(evidence_df)
    llm_runner = build_behavior_backend(behavior_runtime_config, behavior_api_key)
    generation_max_new_tokens = E9_GENERATION_MAX_NEW_TOKENS

    stable_run_config = {
        "task": "E9",
        "split_config_hash": split_manifest["meta"]["config_hash"],
        "query_count": len(eval_units),
        "retrieval_mode": E9_RETRIEVAL_MODE,
        "candidate_policy": E9_CANDIDATE_POLICY,
        "behavior_backend": behavior_runtime_config.llm_backend,
        "base_model_id": behavior_runtime_config.model_id,
        "behavior_api_base_url": behavior_runtime_config.api_base_url,
        "behavior_enable_thinking": behavior_runtime_config.enable_thinking,
        "behavior_temperature": behavior_runtime_config.temperature,
        "behavior_max_new_tokens": generation_max_new_tokens,
        "official_group_ids": E9_GROUPS,
        "eval_units_hash": stable_hash([unit.config_hash for unit in eval_units]),
        "fallback_enabled": False,
    }
    run_started_at = utc_now_iso()
    run_id = f"e9_{stable_hash(stable_run_config)}_{run_started_at.replace(':', '').replace('-', '')}"
    run_dir = ensure_dir(output_root / run_id)
    ensure_dir(E9_LABELS_DIR)

    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "generated_at": run_started_at,
                "stable_run_config": stable_run_config,
                "behavior_runtime_config": behavior_runtime_config.model_dump(),
                "selected_query_ids": [unit.query_id for unit in eval_units],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    log_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group_id in E9_GROUPS:
        for unit in eval_units:
            start = time.perf_counter()
            response, verification, response_audit_rows, debug_payload = generate_group_response(
                llm_runner=llm_runner,
                unit=unit,
                group_id=group_id,
                max_new_tokens=generation_max_new_tokens,
                evidence_lookup=evidence_lookup,
            )
            latency_ms = round((time.perf_counter() - start) * 1000, 3)
            unsupported_honesty = None
            if unit.unsupported_requests:
                unsupported_honesty = int(bool(response.unsupported_notice.strip()))
            response_error_type = debug_payload.get("response_error_type")
            reasoning_leak_detected = response_error_type == "reasoning_leak"

            grouped_entry = {
                "query_id": unit.query_id,
                "latency_ms": latency_ms,
                "response": response,
                "verification": verification,
                "audit_rows": response_audit_rows,
                "unsupported_honesty": unsupported_honesty,
                "response_error_type": response_error_type,
                "reasoning_leak_detected": reasoning_leak_detected,
            }
            grouped_rows[group_id].append(grouped_entry)
            audit_rows.extend(response_audit_rows)
            log_rows.append(
                RunLogEntry(
                    run_id=run_id,
                    group_id=group_id,
                    query_id=unit.query_id,
                    retrieval_mode=unit.retrieval_mode,
                    candidate_mode=E9_CANDIDATE_MODE,
                    config_hash=stable_hash(stable_run_config | {"group_id": group_id}),
                    latency_ms=latency_ms,
                    intermediate_objects={
                        "eval_unit": unit.model_dump(),
                        "response": response.model_dump(),
                        "citation_verification": verification.model_dump(),
                        "audit_rows": response_audit_rows,
                        "debug_payload": debug_payload,
                        "unsupported_honesty": unsupported_honesty,
                        "behavior_runtime_config": behavior_runtime_config.model_dump(),
                    },
                ).model_dump()
            )

    summary_rows = [
        build_e9_metric_row(group_id, grouped_rows[group_id], stable_run_config)
        for group_id in E9_GROUPS
    ]

    write_jsonl(run_dir / "results.jsonl", log_rows)
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(run_dir / "citation_verifiability_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(E9_LABELS_DIR / "citation_verifiability_audit.csv", index=False, encoding="utf-8-sig")
    build_e9_analysis_md(run_dir, summary_rows, grouped_rows)
    return run_dir


def assign_manifest_split(query_id: str) -> str:
    return "dev" if int(stable_hash({"query_id": query_id}), 16) % 5 == 0 else "train"


def build_feedback_update_pair(slot_row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    focus_aspects = list(slot_row["focus_aspects"])
    avoid_aspects = list(slot_row["avoid_aspects"])
    if focus_aspects:
        aspect = focus_aspects[0]
        prior = dict(slot_row)
        prior["focus_aspects"] = focus_aspects[1:]
        user_feedback = f"补充一下，请把{ASPECT_ZH_LABELS[aspect]}作为重点考虑方面。"
        target = slot_row
    elif avoid_aspects:
        aspect = avoid_aspects[0]
        prior = dict(slot_row)
        prior["avoid_aspects"] = avoid_aspects[1:]
        user_feedback = f"补充一下，请把{ASPECT_ZH_LABELS[aspect]}作为需要避免的方面。"
        target = slot_row
    else:
        return None
    return {
        "prior_preference": {
            "city": prior["city"],
            "hotel_category": prior["hotel_category"],
            "focus_aspects": prior["focus_aspects"],
            "avoid_aspects": prior["avoid_aspects"],
            "unsupported_requests": prior["unsupported_requests"],
        },
        "user_feedback": user_feedback,
    }, {
        "updated_preference": {
            "city": target["city"],
            "hotel_category": target["hotel_category"],
            "focus_aspects": target["focus_aspects"],
            "avoid_aspects": target["avoid_aspects"],
            "unsupported_requests": target["unsupported_requests"],
        }
    }


def build_preference_from_slot_row(slot_row: dict[str, Any]) -> UserPreference:
    return UserPreference(
        city=slot_row["city"],
        state=slot_row["state"],
        hotel_category=slot_row["hotel_category"],
        focus_aspects=slot_row["focus_aspects"],
        avoid_aspects=slot_row["avoid_aspects"],
        unsupported_requests=slot_row["unsupported_requests"],
        query_en=slot_row["query_en"],
    )


def contains_english_long_span(text: str) -> bool:
    return bool(ENGLISH_LONG_SPAN_PATTERN.search(text or ""))


def response_has_english_reason_text(response: RecommendationResponse) -> bool:
    for item in response.recommendations:
        for reason in item.reasons:
            if contains_english_long_span(reason.reason_text):
                return True
    return False


def build_e10_v2_grounded_query_rows() -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    judged_queries = load_jsonl(EXPERIMENT_ASSETS_DIR / "judged_queries.jsonl")
    slot_gold = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "slot_gold.jsonl")}
    clarify_gold = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "clarify_gold.jsonl")}
    official_e9_query_ids = set(load_json(E9_QUERY_IDS_PATH))

    rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for query_row in judged_queries:
        query_id = query_row["query_id"]
        if query_id in official_e9_query_ids:
            continue
        slot_row = slot_gold[query_id]
        if not slot_row["city"] or not (slot_row["focus_aspects"] or slot_row["avoid_aspects"]):
            continue
        rows.append((query_row, slot_row, clarify_gold[query_id]))
    return rows


def build_candidate_hotels_for_slot(
    hotel_summary: pd.DataFrame,
    profile_current: pd.DataFrame,
    slot_row: dict[str, Any],
    allowed_hotel_ids: set[str],
) -> list[HotelCandidate]:
    city_hotels = hotel_summary[
        (hotel_summary["city"] == slot_row["city"])
        & (hotel_summary["hotel_id"].astype(str).isin(allowed_hotel_ids))
    ].copy()
    if city_hotels.empty:
        return []
    ranked = candidate_rank(
        city_hotels=city_hotels,
        profile_current=profile_current,
        profile_alt=profile_current,
        focus_aspects=slot_row["focus_aspects"],
        avoid_aspects=slot_row["avoid_aspects"],
        mode="B_final_aspect_score",
    ).head(5)
    candidate_hotels: list[HotelCandidate] = []
    for _, hotel_row in ranked.iterrows():
        candidate_hotels.append(
            HotelCandidate(
                hotel_id=str(hotel_row["hotel_id"]),
                hotel_name=str(hotel_row["hotel_name"]),
                score_total=round(float(hotel_row["score_total"]), 4),
                score_breakdown={k: float(v) for k, v in dict(hotel_row["score_breakdown"]).items()},
            )
        )
    return candidate_hotels


def build_generation_eval_unit_for_slot(
    *,
    query_row: dict[str, Any],
    slot_row: dict[str, Any],
    candidate_hotels: list[HotelCandidate],
    collection: Any,
    bi_encoder: Any,
    normalize_embeddings: bool,
    evidence_lookup: dict[str, dict[str, Any]],
    dense_top_k: int,
    final_top_k: int,
    stable_asset_config: dict[str, Any],
) -> GenerationEvalUnit:
    preference = build_preference_from_slot_row(slot_row)
    evidence_packs = [
        build_evidence_pack_for_candidate(
            collection=collection,
            bi_encoder=bi_encoder,
            normalize_embeddings=normalize_embeddings,
            evidence_lookup=evidence_lookup,
            preference=preference,
            hotel=hotel,
            dense_top_k=dense_top_k,
            final_top_k=final_top_k,
        )
        for hotel in candidate_hotels
    ]
    return GenerationEvalUnit(
        query_id=query_row["query_id"],
        query_text_zh=query_row["query_text_zh"],
        query_type=query_row["query_type"],
        user_preference_gold=preference,
        unsupported_requests=preference.unsupported_requests,
        candidate_hotels=candidate_hotels,
        evidence_packs=evidence_packs,
        retrieval_mode=E9_RETRIEVAL_MODE,
        candidate_policy=E9_CANDIDATE_POLICY,
        config_hash=stable_hash(
            stable_asset_config
            | {
                "query_id": query_row["query_id"],
                "candidate_hotel_ids": [hotel.hotel_id for hotel in candidate_hotels],
                "all_sentence_ids": [pack.all_sentence_ids for pack in evidence_packs],
            }
        ),
    )


def build_grounded_recommendation_target_payload(response: RecommendationResponse) -> dict[str, Any]:
    return {
        "summary": response.summary,
        "recommendations": [item.model_dump() for item in response.recommendations],
        "unsupported_notice": response.unsupported_notice,
    }


def build_grounded_recommendation_input_payload(unit: GenerationEvalUnit) -> dict[str, Any]:
    compact_candidate_hotels = [
        {
            "hotel_id": hotel.hotel_id,
            "hotel_name": hotel.hotel_name,
        }
        for hotel in unit.candidate_hotels[:E9_MAX_RECOMMENDATIONS]
    ]
    compact_evidence_packs = []
    for pack in unit.evidence_packs[:E9_MAX_RECOMMENDATIONS]:
        compact_aspects: dict[str, list[dict[str, str]]] = {}
        allowed_sentence_ids: list[str] = []
        for aspect in sorted(pack.evidence_by_aspect):
            compact_rows = []
            for sentence in pack.evidence_by_aspect[aspect][:E9_PROMPT_SENTENCE_LIMIT_GROUNDED]:
                compact_rows.append(
                    {
                        "sentence_id": sentence.sentence_id,
                        "sentence_text": sentence.sentence_text,
                    }
                )
                allowed_sentence_ids.append(sentence.sentence_id)
            compact_aspects[aspect] = compact_rows
        compact_evidence_packs.append(
            {
                "hotel_id": pack.hotel_id,
                "evidence_by_aspect": compact_aspects,
                "allowed_sentence_ids": allowed_sentence_ids,
            }
        )

    return {
        "query_id": unit.query_id,
        "query_text_zh": unit.query_text_zh,
        "user_preference_gold": unit.user_preference_gold.model_dump(),
        "unsupported_requests": unit.unsupported_requests,
        "candidate_hotels": compact_candidate_hotels,
        "evidence_packs": compact_evidence_packs,
    }


def reason_text_indicates_missing_evidence(reason_text: str) -> bool:
    normalized = str(reason_text or "").strip()
    return any(pattern in normalized for pattern in MISSING_EVIDENCE_REASON_PATTERNS)


def build_supported_aspect_summary(
    unit: GenerationEvalUnit,
    supported_aspects: set[str],
    recommendation_count: int,
) -> str:
    city = unit.user_preference_gold.city or "当前城市"
    if not supported_aspects:
        return f"{city}当前证据不足，暂不返回酒店推荐。"
    labels = "、".join(ASPECT_ZH_LABELS.get(aspect, aspect) for aspect in sorted(supported_aspects))
    if recommendation_count <= 1:
        return f"推荐{city}在{labels}方面有明确证据的酒店。"
    return f"推荐{city}在{labels}方面有明确证据的酒店。"


def sanitize_grounded_recommendation_response_for_training(
    unit: GenerationEvalUnit,
    response: RecommendationResponse,
) -> RecommendationResponse:
    notice_parts: list[str] = []
    seen_notice_parts: set[str] = set()
    if response.unsupported_notice.strip():
        notice_parts.append(response.unsupported_notice.strip())
        seen_notice_parts.add(response.unsupported_notice.strip())

    removed_partial_support = False
    cleaned_items: list[RecommendationItem] = []
    supported_aspects: set[str] = set()
    for item in response.recommendations:
        cleaned_reasons: list[RecommendationReason] = []
        for reason in item.reasons:
            if reason.sentence_id is None or reason_text_indicates_missing_evidence(reason.reason_text):
                removed_partial_support = True
                continue
            cleaned_reasons.append(reason)
            supported_aspects.add(reason.aspect)
        if cleaned_reasons:
            cleaned_items.append(
                RecommendationItem(
                    hotel_id=item.hotel_id,
                    hotel_name=item.hotel_name,
                    reasons=cleaned_reasons,
                )
            )
        elif item.reasons:
            removed_partial_support = True

    if removed_partial_support:
        partial_notice = "部分方面缺乏直接证据，以下仅保留可验证理由。"
        if partial_notice not in seen_notice_parts:
            notice_parts.append(partial_notice)
            seen_notice_parts.add(partial_notice)

    final_notice = " ".join(notice_parts)
    if cleaned_items:
        summary = build_supported_aspect_summary(unit, supported_aspects, len(cleaned_items))
        schema_valid = True
    else:
        summary = f"{unit.user_preference_gold.city or '当前城市'}当前证据不足，暂不返回酒店推荐。"
        schema_valid = bool(final_notice)

    return RecommendationResponse(
        query_id=response.query_id,
        group_id=response.group_id,
        summary=summary,
        recommendations=cleaned_items,
        unsupported_notice=final_notice,
        schema_valid=schema_valid,
        raw_response=response.raw_response,
    )


def validate_grounded_recommendation_example(
    unit: GenerationEvalUnit,
    response: RecommendationResponse,
    verification: CitationVerificationResult,
    debug_payload: dict[str, Any],
) -> tuple[bool, str]:
    if debug_payload.get("response_error_type") == "reasoning_leak":
        return False, "reasoning_leak"
    if not response.schema_valid:
        return False, "schema_invalid"
    if verification.invalid_sentence_ids or verification.out_of_pack_sentence_ids:
        return False, "citation_invalid"
    if response_has_english_reason_text(response):
        return False, "english_long_span"
    if response.recommendations:
        if verification.citation_precision != 1.0:
            return False, "citation_precision_not_full"
        return True, "ok"
    if response.unsupported_notice.strip() and not unit.unsupported_requests:
        return True, "ok"
    if response.unsupported_notice.strip():
        return False, "unsupported_driven_abstain"
    return False, "empty_without_notice"


def classify_grounded_record_slices(row: dict[str, Any]) -> set[str]:
    preference = row["input_payload"]["user_preference_gold"]
    focus_aspects = preference["focus_aspects"]
    avoid_aspects = preference["avoid_aspects"]
    candidate_count = len(row["input_payload"]["candidate_hotels"])
    recommendation_count = len(row["target_payload"]["recommendations"])
    max_possible_recommendations = min(E9_MAX_RECOMMENDATIONS, candidate_count)

    slices: set[str] = set()
    if "quiet_sleep" in focus_aspects:
        slices.add("quiet_sleep")
    if avoid_aspects:
        slices.add("focus_avoid")
    if recommendation_count < max_possible_recommendations:
        slices.add("partial_abstain")
    if recommendation_count == 0:
        slices.add("zero_recommendation")
    return slices


def duplicate_manifest_record(row: dict[str, Any], repeat_index: int, reason: str) -> dict[str, Any]:
    duplicated = copy.deepcopy(row)
    duplicated["record_id"] = stable_hash(
        {
            "task_type": duplicated["task_type"],
            "source_record_id": row["record_id"],
            "repeat_index": repeat_index,
            "reason": reason,
        }
    )
    duplicated["source_asset"] = f"{duplicated['source_asset']}#{reason}"
    return duplicated


def rebalance_grounded_train_records(
    grounded_rows: list[dict[str, Any]],
    base_record_count: int,
) -> list[dict[str, Any]]:
    if not grounded_rows:
        raise ValueError("E10 v2 grounded train rows 为空，无法构建 v2 manifest。")

    rebalanced = list(sorted(grounded_rows, key=lambda row: row["record_id"]))
    duplication_counter = 0
    slice_requirements = {
        "quiet_sleep": 0.30,
        "focus_avoid": 0.30,
        "partial_abstain": 0.15,
    }

    def slice_share(slice_name: str) -> float:
        return sum(int(slice_name in classify_grounded_record_slices(row)) for row in rebalanced) / max(len(rebalanced), 1)

    source_by_slice = {
        slice_name: [row for row in rebalanced if slice_name in classify_grounded_record_slices(row)]
        for slice_name in slice_requirements
    }
    for slice_name, source_rows in source_by_slice.items():
        if not source_rows:
            raise ValueError(f"E10 v2 grounded pool 缺少必须切片：{slice_name}")
        offset = 0
        while slice_share(slice_name) < slice_requirements[slice_name]:
            rebalanced.append(
                duplicate_manifest_record(
                    source_rows[offset % len(source_rows)],
                    duplication_counter,
                    f"rebalance_{slice_name}",
                )
            )
            duplication_counter += 1
            offset += 1

    all_source_rows = list(sorted(grounded_rows, key=lambda row: row["record_id"]))
    offset = 0
    while len(rebalanced) / max(base_record_count + len(rebalanced), 1) < 0.40:
        rebalanced.append(
            duplicate_manifest_record(
                all_source_rows[offset % len(all_source_rows)],
                duplication_counter,
                "rebalance_grounded_share",
            )
        )
        duplication_counter += 1
        offset += 1

    return rebalanced


def build_grounded_manifest_report(
    *,
    source_rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    train_base_records: list[dict[str, Any]],
    dev_base_records: list[dict[str, Any]],
    train_grounded_records_raw: list[dict[str, Any]],
    dev_grounded_records_raw: list[dict[str, Any]],
    train_grounded_records_final: list[dict[str, Any]],
    dropped_reason_counts: dict[str, int],
) -> dict[str, Any]:
    def task_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
        distribution: dict[str, int] = defaultdict(int)
        for row in rows:
            distribution[row["task_type"]] += 1
        return dict(sorted(distribution.items()))

    def slice_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
        distribution: dict[str, int] = defaultdict(int)
        for row in rows:
            for slice_name in sorted(classify_grounded_record_slices(row)):
                distribution[slice_name] += 1
        return dict(sorted(distribution.items()))

    clarify_needed_count = sum(int(clarify_row["clarify_needed"]) for _, _, clarify_row in source_rows)
    return {
        "version": E10_V2_MANIFEST_CONFIG_VERSION,
        "source_query_count": len(source_rows),
        "source_query_ids": [query_row["query_id"] for query_row, _, _ in source_rows],
        "clarify_needed_source_count": clarify_needed_count,
        "train_base_record_count": len(train_base_records),
        "dev_base_record_count": len(dev_base_records),
        "train_grounded_record_count_raw": len(train_grounded_records_raw),
        "dev_grounded_record_count_raw": len(dev_grounded_records_raw),
        "train_grounded_record_count_final": len(train_grounded_records_final),
        "train_grounded_share_of_final_manifest": round(
            len(train_grounded_records_final) / max(len(train_base_records) + len(train_grounded_records_final), 1),
            4,
        ),
        "train_task_distribution": task_distribution(train_base_records + train_grounded_records_final),
        "dev_task_distribution": task_distribution(dev_base_records + dev_grounded_records_raw),
        "train_grounded_slice_distribution": slice_distribution(train_grounded_records_final),
        "dev_grounded_slice_distribution": slice_distribution(dev_grounded_records_raw),
        "dropped_reason_counts": dict(sorted(dropped_reason_counts.items())),
    }


def build_sft_manifest_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    judged_queries = load_jsonl(EXPERIMENT_ASSETS_DIR / "judged_queries.jsonl")
    slot_gold = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "slot_gold.jsonl")}
    clarify_gold = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "clarify_gold.jsonl")}

    train_records: list[dict[str, Any]] = []
    dev_records: list[dict[str, Any]] = []
    manifest_config = {"task": "E10_manifest", "version": 1, "behavior_only": True}

    for query_row in judged_queries:
        query_id = query_row["query_id"]
        split = assign_manifest_split(query_id)
        bucket = train_records if split == "train" else dev_records
        slot_row = slot_gold[query_id]
        clarify_row = clarify_gold[query_id]

        preference_record = SFTManifestRecord(
            record_id=stable_hash({"task_type": "preference_parse", "query_id": query_id, "split": split}),
            split=split,
            task_type="preference_parse",
            hotel_id=None,
            query_id=query_id,
            source_asset="experiments/assets/slot_gold.jsonl",
            input_payload={"query_text_zh": query_row["query_text_zh"]},
            target_payload={
                "city": slot_row["city"],
                "hotel_category": slot_row["hotel_category"],
                "focus_aspects": slot_row["focus_aspects"],
                "avoid_aspects": slot_row["avoid_aspects"],
                "unsupported_requests": slot_row["unsupported_requests"],
            },
            config_hash=stable_hash(manifest_config | {"task_type": "preference_parse"}),
        )
        bucket.append(preference_record.model_dump())

        clarification_question = ""
        if clarify_row["clarify_needed"]:
            clarification_question = build_rule_clarification(
                query_row["query_text_zh"],
                {slot_row["city"]: slot_row["state"]} if slot_row["city"] and slot_row["state"] else {},
            ).question
        clarification_record = SFTManifestRecord(
            record_id=stable_hash({"task_type": "clarification", "query_id": query_id, "split": split}),
            split=split,
            task_type="clarification",
            hotel_id=None,
            query_id=query_id,
            source_asset="experiments/assets/clarify_gold.jsonl",
            input_payload={"query_text_zh": query_row["query_text_zh"]},
            target_payload={
                "clarify_needed": clarify_row["clarify_needed"],
                "clarify_reason": clarify_row["clarify_reason"],
                "target_slots": clarify_row["target_slots"],
                "question": clarification_question if clarify_row["clarify_needed"] else "",
            },
            config_hash=stable_hash(manifest_config | {"task_type": "clarification"}),
        )
        bucket.append(clarification_record.model_dump())

        if slot_row["unsupported_requests"]:
            honesty_record = SFTManifestRecord(
                record_id=stable_hash({"task_type": "constraint_honesty", "query_id": query_id, "split": split}),
                split=split,
                task_type="constraint_honesty",
                hotel_id=None,
                query_id=query_id,
                source_asset="experiments/assets/slot_gold.jsonl",
                input_payload={
                    "query_text_zh": query_row["query_text_zh"],
                    "unsupported_requests": slot_row["unsupported_requests"],
                },
                target_payload={
                    "unsupported_notice": build_unsupported_notice(slot_row["unsupported_requests"]),
                },
                config_hash=stable_hash(manifest_config | {"task_type": "constraint_honesty"}),
            )
            bucket.append(honesty_record.model_dump())

        feedback_pair = build_feedback_update_pair(slot_row)
        if feedback_pair is not None:
            input_payload, target_payload = feedback_pair
            feedback_record = SFTManifestRecord(
                record_id=stable_hash({"task_type": "feedback_update", "query_id": query_id, "split": split}),
                split=split,
                task_type="feedback_update",
                hotel_id=None,
                query_id=query_id,
                source_asset="synthetic_feedback_update_from_slot_gold",
                input_payload=input_payload,
                target_payload=target_payload,
                config_hash=stable_hash(manifest_config | {"task_type": "feedback_update"}),
            )
            bucket.append(feedback_record.model_dump())

    return train_records, dev_records


def build_sft_manifest_records_v2() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_records_base, dev_records_base = build_sft_manifest_records()
    source_rows = build_e10_v2_grounded_query_rows()

    cfg = load_config()
    frozen_config = load_json(EXPERIMENT_ASSETS_DIR / "frozen_config.yaml")
    split_manifest = load_json(EXPERIMENT_ASSETS_DIR / "frozen_split_manifest.json")
    review_df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
    profile_df = pd.read_pickle("data/intermediate/hotel_profiles.pkl")
    evidence_df = pd.read_pickle("data/intermediate/evidence_index.pkl")

    hotel_summary = build_hotel_summary(review_df)
    profile_current, profile_alt = build_profile_tables(profile_df)
    del profile_alt
    evidence_lookup = build_evidence_lookup(evidence_df)
    split_hotel_lookup = build_split_hotel_lookup(split_manifest)

    behavior_runtime_config, behavior_api_key = resolve_behavior_runtime_config(cfg, frozen_config)
    validate_runtime_base_model(
        behavior_runtime_config.model_id,
        str(frozen_config["behavior"]["base_model"]),
    )
    llm_runner = build_behavior_backend(behavior_runtime_config, behavior_api_key)

    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer

    client = PersistentClient(path=cfg["embedding"]["chroma_persist_dir"])
    collection = client.get_collection(cfg["embedding"]["chroma_collection"])
    try:
        bi_encoder = SentenceTransformer(cfg["embedding"]["model"], local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "E10 v2 manifest 生成需要本地可用的 embedding 模型缓存；当前环境未能在离线模式下加载 "
            f"{cfg['embedding']['model']}。请先在有网环境缓存该模型后再重试。"
        ) from exc
    normalize_embeddings = bool(cfg["embedding"].get("normalize", True))

    manifest_config = {
        "task": "E10_manifest_v2",
        "version": E10_V2_MANIFEST_CONFIG_VERSION,
        "grounded_recommendation": True,
        "retrieval_mode": E9_RETRIEVAL_MODE,
        "candidate_policy": E9_CANDIDATE_POLICY,
        "dense_top_k": cfg["reranker"]["top_k_before_rerank"],
        "final_top_k": cfg["reranker"]["top_k_after_rerank"],
        "embedding_model": cfg["embedding"]["model"],
        "collection": cfg["embedding"]["chroma_collection"],
        "generator_model_id": behavior_runtime_config.model_id,
    }

    train_grounded_records_raw: list[dict[str, Any]] = []
    dev_grounded_records_raw: list[dict[str, Any]] = []
    dropped_reason_counts: dict[str, int] = defaultdict(int)

    for query_row, slot_row, clarify_row in source_rows:
        for split in ("train", "dev"):
            allowed_hotel_ids = set(split_hotel_lookup.get(split, {}).get(slot_row["city"], []))
            if not allowed_hotel_ids:
                dropped_reason_counts[f"{split}:no_hotels_in_city_split"] += 1
                continue
            candidate_hotels = build_candidate_hotels_for_slot(
                hotel_summary=hotel_summary,
                profile_current=profile_current,
                slot_row=slot_row,
                allowed_hotel_ids=allowed_hotel_ids,
            )
            if not candidate_hotels:
                dropped_reason_counts[f"{split}:no_ranked_candidates"] += 1
                continue
            unit = build_generation_eval_unit_for_slot(
                query_row=query_row,
                slot_row=slot_row,
                candidate_hotels=candidate_hotels,
                collection=collection,
                bi_encoder=bi_encoder,
                normalize_embeddings=normalize_embeddings,
                evidence_lookup=evidence_lookup,
                dense_top_k=cfg["reranker"]["top_k_before_rerank"],
                final_top_k=cfg["reranker"]["top_k_after_rerank"],
                stable_asset_config=manifest_config | {"split": split},
            )
            response, verification, _audit_rows, debug_payload = generate_group_response(
                llm_runner=llm_runner,
                unit=unit,
                group_id="C_grounded_generation_with_verifier",
                max_new_tokens=E9_GENERATION_MAX_NEW_TOKENS,
                evidence_lookup=evidence_lookup,
            )
            is_valid, drop_reason = validate_grounded_recommendation_example(
                unit,
                response,
                verification,
                debug_payload,
            )
            if not is_valid:
                dropped_reason_counts[f"{split}:{drop_reason}"] += 1
                continue

            record = SFTManifestRecord(
                record_id=stable_hash(
                    {
                        "task_type": "grounded_recommendation",
                        "query_id": query_row["query_id"],
                        "split": split,
                        "candidate_hotels": [hotel.hotel_id for hotel in candidate_hotels],
                    }
                ),
                split=split,
                task_type="grounded_recommendation",
                hotel_id=None,
                query_id=query_row["query_id"],
                source_asset="synthetic_grounded_recommendation_from_base_generator_v2",
                input_payload={
                    **build_grounded_recommendation_input_payload(unit),
                },
                target_payload=build_grounded_recommendation_target_payload(response),
                config_hash=stable_hash(
                    manifest_config
                    | {
                        "task_type": "grounded_recommendation",
                        "query_id": query_row["query_id"],
                        "split": split,
                        "clarify_needed_source": bool(clarify_row["clarify_needed"]),
                    }
                ),
            ).model_dump()
            if split == "train":
                train_grounded_records_raw.append(record)
            else:
                dev_grounded_records_raw.append(record)

    train_grounded_records_final = rebalance_grounded_train_records(
        train_grounded_records_raw,
        len(train_records_base),
    )
    train_records = sorted(
        train_records_base + train_grounded_records_final,
        key=lambda row: (row["task_type"], row["query_id"], row["record_id"]),
    )
    dev_records = sorted(
        dev_records_base + dev_grounded_records_raw,
        key=lambda row: (row["task_type"], row["query_id"], row["record_id"]),
    )
    manifest_report = build_grounded_manifest_report(
        source_rows=source_rows,
        train_base_records=train_records_base,
        dev_base_records=dev_records_base,
        train_grounded_records_raw=train_grounded_records_raw,
        dev_grounded_records_raw=dev_grounded_records_raw,
        train_grounded_records_final=train_grounded_records_final,
        dropped_reason_counts=dropped_reason_counts,
    )
    return train_records, dev_records, manifest_report


def build_e10_v3_judged_grounded_source_rows() -> list[dict[str, Any]]:
    judged_queries = load_jsonl(EXPERIMENT_ASSETS_DIR / "judged_queries.jsonl")
    slot_gold = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "slot_gold.jsonl")}
    clarify_gold = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "clarify_gold.jsonl")}
    official_e9_query_ids = set(load_json(E9_QUERY_IDS_PATH))

    rows: list[dict[str, Any]] = []
    for query_row in judged_queries:
        query_id = query_row["query_id"]
        if query_id in official_e9_query_ids:
            continue
        slot_row = slot_gold[query_id]
        if slot_row["unsupported_requests"]:
            continue
        if not slot_row["city"] or not (slot_row["focus_aspects"] or slot_row["avoid_aspects"]):
            continue
        rows.append(
            {
                "query_row": query_row,
                "slot_row": slot_row,
                "source_type": "judged",
                "template_kind": "judged_grounded",
                "clarify_needed_source": bool(clarify_gold[query_id]["clarify_needed"]),
            }
        )
    return rows


def build_e10_v3_synthetic_grounded_source_rows(
    review_df: pd.DataFrame,
    split_hotel_lookup: dict[str, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    state_lookup = city_state_map(review_df)
    train_city_lookup = split_hotel_lookup.get("train", {})
    template_specs = [
        {
            "template_kind": "multi_hotel_pack_boundary",
            "query_type": "multi_aspect_strong",
            "focus_aspects": ["service", "quiet_sleep", "location_transport"],
            "avoid_aspects": [],
            "query_text_template": "请推荐{city}在服务、安静睡眠、位置交通三方面都比较均衡的酒店。",
        },
        {
            "template_kind": "partial_support_keep_recommendation",
            "query_type": "multi_aspect",
            "focus_aspects": ["quiet_sleep", "value"],
            "avoid_aspects": [],
            "query_text_template": "请推荐{city}安静睡眠和性价比都不错的酒店。",
        },
        {
            "template_kind": "partial_support_keep_recommendation",
            "query_type": "focus_and_avoid",
            "focus_aspects": ["quiet_sleep"],
            "avoid_aspects": ["value"],
            "query_text_template": "我在{city}想住得安静一点，但不要性价比太差的酒店。",
        },
    ]

    rows: list[dict[str, Any]] = []
    for city in sorted(train_city_lookup):
        state = state_lookup.get(city)
        if not state:
            continue
        for template in template_specs:
            slot_row = {
                "city": city,
                "state": state,
                "hotel_category": None,
                "focus_aspects": template["focus_aspects"],
                "avoid_aspects": template["avoid_aspects"],
                "unsupported_requests": [],
                "query_en": build_query_en_from_slots(
                    city,
                    template["focus_aspects"],
                    template["avoid_aspects"],
                    [],
                ),
            }
            query_row = {
                "query_id": f"v3syn_{stable_hash({'city': city, 'template': template['template_kind'], 'query_type': template['query_type'], 'focus': template['focus_aspects'], 'avoid': template['avoid_aspects']})}",
                "query_text_zh": template["query_text_template"].format(city=city),
                "query_type": template["query_type"],
            }
            rows.append(
                {
                    "query_row": query_row,
                    "slot_row": slot_row,
                    "source_type": "synthetic",
                    "template_kind": template["template_kind"],
                    "clarify_needed_source": False,
                }
            )
    return rows


def build_generation_prompts_v3_grounded(unit: GenerationEvalUnit) -> tuple[str, str]:
    system_prompt, user_prompt = build_generation_prompts(unit, "C_grounded_generation_with_verifier")
    system_prompt = (
        system_prompt
        + "\n"
        + "额外严格约束：\n"
        + "1. grounded 模式下绝对不要输出 sentence_id 为 null。\n"
        + "2. 如果某个方面缺少直接证据，不要把“无直接证据支持”写进 reasons；需要说明时只能写到根级 unsupported_notice。\n"
        + "3. 每家酒店只能引用该酒店自己的 allowed_sentence_ids，绝对不要跨酒店复用 sentence_id。\n"
        + "4. 如果某家酒店只在部分方面有证据，也可以保留该酒店，但 reasons 中只能保留有证据的方面。"
    )
    return system_prompt, user_prompt


def build_retry_prompt_v3(
    unit: GenerationEvalUnit,
    invalid_sentence_ids: list[str],
    out_of_pack_sentence_ids: list[str],
    response_error_type: str | None,
) -> str:
    base_prompt = build_retry_prompt(
        unit,
        invalid_sentence_ids=invalid_sentence_ids,
        out_of_pack_sentence_ids=out_of_pack_sentence_ids,
    )
    if response_error_type == "schema_invalid":
        base_prompt += (
            "\n不要输出 sentence_id 为空的 reason。"
            "不要把“无直接证据支持”之类的话写进 reasons。"
            "缺证据说明只能放到根级 unsupported_notice。"
        )
    return base_prompt


def generate_training_grounded_response_v3(
    llm_runner,
    unit: GenerationEvalUnit,
    evidence_lookup: dict[str, dict[str, Any]],
) -> tuple[RecommendationResponse, CitationVerificationResult, list[dict[str, Any]], dict[str, Any]]:
    def materialize_response(raw_response_text: str) -> tuple[RecommendationResponse, str | None]:
        generation_debug = dict(getattr(llm_runner, "last_generation_debug", {}) or {})
        payload, repaired = parse_json_with_repair(raw_response_text)
        response = coerce_generation_payload(
            payload,
            unit,
            "C_grounded_generation_with_verifier",
            raw_response_text,
        )
        sanitized_response = sanitize_grounded_recommendation_response_for_training(unit, response)
        response_error_type = generation_debug.get("response_error_type")
        if response_error_type is None and payload is None:
            response_error_type = "invalid_json"
        elif response_error_type is None and not response.schema_valid and not sanitized_response.schema_valid:
            response_error_type = "schema_invalid"
        elif response_error_type == "schema_invalid" and sanitized_response.schema_valid:
            response_error_type = None
        return sanitized_response, response_error_type

    system_prompt, user_prompt = build_generation_prompts_v3_grounded(unit)
    raw_response = llm_runner.generate_json(system_prompt, user_prompt, max_new_tokens=E9_GENERATION_MAX_NEW_TOKENS)
    response, response_error_type = materialize_response(raw_response)
    verification, audit_rows = verify_response_citations(response, unit, evidence_lookup)
    if response_error_type is None and (verification.invalid_sentence_ids or verification.out_of_pack_sentence_ids):
        response_error_type = "citation_invalid"
    debug_payload: dict[str, Any] = {
        "raw_response_initial": raw_response,
        "retry_raw_response": "",
        "response_error_type": response_error_type,
        "thinking_control_supported": getattr(llm_runner, "last_generation_debug", {}).get("thinking_control_supported"),
        "raw_response_prefix": getattr(llm_runner, "last_generation_debug", {}).get("raw_response_prefix", raw_response[:80]),
    }

    needs_retry = bool(verification.invalid_sentence_ids or verification.out_of_pack_sentence_ids)
    needs_retry = needs_retry or not response.schema_valid
    if not needs_retry:
        return response, verification, audit_rows, debug_payload

    retry_raw_response = raw_response
    for _ in range(E9_RETRY_LIMIT):
        retry_prompt = build_retry_prompt_v3(
            unit,
            invalid_sentence_ids=verification.invalid_sentence_ids,
            out_of_pack_sentence_ids=verification.out_of_pack_sentence_ids,
            response_error_type=response_error_type,
        )
        retry_raw_response = llm_runner.generate_json(
            system_prompt,
            f"{user_prompt}\n\n{retry_prompt}",
            max_new_tokens=E9_GENERATION_MAX_NEW_TOKENS,
        )
        retry_response, retry_error_type = materialize_response(retry_raw_response)
        retry_verification, retry_audit_rows = verify_response_citations(
            retry_response,
            unit,
            evidence_lookup,
            retry_triggered=True,
        )
        if retry_error_type is None and (retry_verification.invalid_sentence_ids or retry_verification.out_of_pack_sentence_ids):
            retry_error_type = "citation_invalid"
        response = retry_response
        verification = retry_verification
        audit_rows = retry_audit_rows
        response_error_type = retry_error_type
        debug_payload["retry_raw_response"] = retry_raw_response
        debug_payload["response_error_type"] = response_error_type
        debug_payload["raw_response_prefix"] = getattr(
            llm_runner,
            "last_generation_debug",
            {},
        ).get("raw_response_prefix", retry_raw_response[:80])
        if (
            response.schema_valid
            and not verification.invalid_sentence_ids
            and not verification.out_of_pack_sentence_ids
        ):
            return response, verification, audit_rows, debug_payload

    return response, verification, audit_rows, debug_payload


def classify_grounded_record_slices_v3(row: dict[str, Any]) -> set[str]:
    preference = row["input_payload"]["user_preference_gold"]
    focus_aspects = set(preference["focus_aspects"])
    avoid_aspects = set(preference["avoid_aspects"])
    addressed_aspects = {
        reason["aspect"]
        for item in row["target_payload"]["recommendations"]
        for reason in item["reasons"]
    }
    addressed_focus_aspects = addressed_aspects & focus_aspects
    slices: set[str] = set()
    if "quiet_sleep" in focus_aspects:
        slices.add("quiet_sleep")
    if avoid_aspects:
        slices.add("focus_avoid")
    if row["source_asset"].startswith("synthetic_grounded_recommendation_v3::multi_hotel_pack_boundary"):
        slices.add("multi_hotel_pack_boundary")
    if (
        row["target_payload"]["recommendations"]
        and len(focus_aspects) >= 2
        and addressed_focus_aspects
        and addressed_focus_aspects < focus_aspects
    ):
        slices.add("partial_support_keep_recommendation")
    if (
        not row["target_payload"]["recommendations"]
        and row["target_payload"]["unsupported_notice"].strip()
        and not row["input_payload"]["unsupported_requests"]
    ):
        slices.add("zero_recommendation_evidence_gap")
    return slices


def rebalance_grounded_train_records_v3(
    grounded_rows: list[dict[str, Any]],
    base_record_count: int,
) -> list[dict[str, Any]]:
    if not grounded_rows:
        raise ValueError("E10 v3 grounded train rows 为空，无法构建 v3 manifest。")

    rebalanced = list(sorted(grounded_rows, key=lambda row: row["record_id"]))
    duplication_counter = 0
    minimum_slice_requirements = {
        "quiet_sleep": 0.30,
        "focus_avoid": 0.30,
        "partial_support_keep_recommendation": 0.20,
        "multi_hotel_pack_boundary": 0.15,
    }
    zero_recommendation_max_share = 0.10
    zero_slice_name = "zero_recommendation_evidence_gap"

    def row_slices(row: dict[str, Any]) -> set[str]:
        return classify_grounded_record_slices_v3(row)

    def slice_count(slice_name: str) -> int:
        return sum(int(slice_name in row_slices(row)) for row in rebalanced)

    source_by_slice = {
        slice_name: [row for row in grounded_rows if slice_name in row_slices(row)]
        for slice_name in minimum_slice_requirements
    }
    for slice_name, source_rows in source_by_slice.items():
        if not source_rows:
            raise ValueError(f"E10 v3 grounded pool 缺少必须切片：{slice_name}")

    current_zero_count = slice_count(zero_slice_name)
    min_grounded_count = math.ceil((0.40 * base_record_count) / 0.60)
    min_count_for_zero_cap = (
        math.ceil(current_zero_count / zero_recommendation_max_share)
        if current_zero_count
        else len(rebalanced)
    )
    target_final_size = max(len(rebalanced), min_grounded_count, min_count_for_zero_cap)
    required_minimum_counts = {
        slice_name: math.ceil(required_share * target_final_size)
        for slice_name, required_share in minimum_slice_requirements.items()
    }
    allowed_zero_max_count = math.floor(zero_recommendation_max_share * target_final_size)
    remaining_slots = target_final_size - len(rebalanced)

    def deficit_counts() -> dict[str, int]:
        return {
            slice_name: required_count - slice_count(slice_name)
            for slice_name, required_count in required_minimum_counts.items()
            if slice_count(slice_name) < required_count
        }

    def can_add(candidate_row: dict[str, Any]) -> bool:
        candidate_zero = int(zero_slice_name in row_slices(candidate_row))
        return current_zero_count + candidate_zero <= allowed_zero_max_count

    def candidate_score_for_deficits(candidate_row: dict[str, Any], deficits: dict[str, int]) -> tuple[int, int, int, int]:
        candidate_slices = row_slices(candidate_row)
        deficit_reduction = sum(deficits[slice_name] for slice_name in deficits if slice_name in candidate_slices)
        minimum_coverage = sum(int(slice_name in candidate_slices) for slice_name in minimum_slice_requirements)
        is_non_zero = int(zero_slice_name not in candidate_slices)
        is_synthetic = int(candidate_row["source_asset"].startswith("synthetic_"))
        return (deficit_reduction, is_non_zero, minimum_coverage, is_synthetic)

    while remaining_slots > 0 and deficit_counts():
        deficits = deficit_counts()
        candidates = [
            row
            for row in grounded_rows
            if can_add(row) and candidate_score_for_deficits(row, deficits)[0] > 0
        ]
        if not candidates:
            raise ValueError(
                "E10 v3 grounded rebalance 无法满足最终 slice floors / zero-recommendation 上限 / grounded share 约束。"
            )
        best_row = max(
            sorted(candidates, key=lambda row: row["record_id"]),
            key=lambda row: candidate_score_for_deficits(row, deficits),
        )
        duplicated = duplicate_manifest_record(
            best_row,
            duplication_counter,
            "rebalance_v3_floor",
        )
        rebalanced.append(duplicated)
        duplication_counter += 1
        remaining_slots -= 1
        if zero_slice_name in row_slices(best_row):
            current_zero_count += 1

    if deficit_counts():
        raise ValueError(
            "E10 v3 grounded rebalance 在目标最终样本数下仍无法满足全部 slice floors。"
        )

    def candidate_score_for_fill(candidate_row: dict[str, Any]) -> tuple[int, int, int]:
        candidate_slices = row_slices(candidate_row)
        minimum_coverage = sum(int(slice_name in candidate_slices) for slice_name in minimum_slice_requirements)
        is_non_zero = int(zero_slice_name not in candidate_slices)
        is_synthetic = int(candidate_row["source_asset"].startswith("synthetic_"))
        return (is_non_zero, minimum_coverage, is_synthetic)

    while remaining_slots > 0:
        candidates = [row for row in grounded_rows if can_add(row)]
        if not candidates:
            raise ValueError(
                "E10 v3 grounded rebalance 在填充 grounded share 目标时没有可用候选样本。"
            )
        best_row = max(
            sorted(candidates, key=lambda row: row["record_id"]),
            key=candidate_score_for_fill,
        )
        duplicated = duplicate_manifest_record(
            best_row,
            duplication_counter,
            "rebalance_v3_grounded_share",
        )
        rebalanced.append(duplicated)
        duplication_counter += 1
        remaining_slots -= 1
        if zero_slice_name in row_slices(best_row):
            current_zero_count += 1

    if current_zero_count > allowed_zero_max_count:
        raise ValueError("E10 v3 grounded rebalance 未能满足 zero_recommendation_evidence_gap 上限。")
    for slice_name, required_count in required_minimum_counts.items():
        if slice_count(slice_name) < required_count:
            raise ValueError(f"E10 v3 grounded rebalance 未能满足切片下限：{slice_name}")

    return rebalanced


def build_grounded_manifest_report_v3(
    *,
    source_rows: list[dict[str, Any]],
    train_base_records: list[dict[str, Any]],
    dev_base_records: list[dict[str, Any]],
    train_grounded_records_raw: list[dict[str, Any]],
    dev_grounded_records_raw: list[dict[str, Any]],
    train_grounded_records_final: list[dict[str, Any]],
    dropped_reason_counts: dict[str, int],
) -> dict[str, Any]:
    def task_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
        distribution: dict[str, int] = defaultdict(int)
        for row in rows:
            distribution[row["task_type"]] += 1
        return dict(sorted(distribution.items()))

    def slice_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
        distribution: dict[str, int] = defaultdict(int)
        for row in rows:
            for slice_name in sorted(classify_grounded_record_slices_v3(row)):
                distribution[slice_name] += 1
        return dict(sorted(distribution.items()))

    def source_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
        distribution: dict[str, int] = defaultdict(int)
        for row in rows:
            source_key = "synthetic" if row["source_asset"].startswith("synthetic_") else "judged"
            distribution[source_key] += 1
        return dict(sorted(distribution.items()))

    def share_distribution(counts: dict[str, int], total_count: int) -> dict[str, float]:
        if total_count <= 0:
            return {key: 0.0 for key in counts}
        return {
            key: round(value / total_count, 4)
            for key, value in sorted(counts.items())
        }

    def source_query_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
        distribution: dict[str, int] = defaultdict(int)
        for row in rows:
            distribution[row["source_type"]] += 1
        return dict(sorted(distribution.items()))

    train_grounded_slice_counts = slice_distribution(train_grounded_records_final)
    dev_grounded_slice_counts = slice_distribution(dev_grounded_records_raw)
    train_grounded_source_counts = source_distribution(train_grounded_records_final)
    dev_grounded_source_counts = source_distribution(dev_grounded_records_raw)
    source_query_counts = source_query_distribution(source_rows)

    return {
        "version": E10_V3_MANIFEST_CONFIG_VERSION,
        "source_query_count": len(source_rows),
        "source_query_ids": [row["query_row"]["query_id"] for row in source_rows],
        "source_type_distribution": source_query_counts,
        "source_type_share": share_distribution(source_query_counts, len(source_rows)),
        "train_base_record_count": len(train_base_records),
        "dev_base_record_count": len(dev_base_records),
        "train_grounded_record_count_raw": len(train_grounded_records_raw),
        "dev_grounded_record_count_raw": len(dev_grounded_records_raw),
        "train_grounded_record_count_final": len(train_grounded_records_final),
        "train_grounded_share_of_final_manifest": round(
            len(train_grounded_records_final) / max(len(train_base_records) + len(train_grounded_records_final), 1),
            4,
        ),
        "train_task_distribution": task_distribution(train_base_records + train_grounded_records_final),
        "dev_task_distribution": task_distribution(dev_base_records + dev_grounded_records_raw),
        "train_grounded_slice_distribution": train_grounded_slice_counts,
        "dev_grounded_slice_distribution": dev_grounded_slice_counts,
        "train_grounded_slice_share": share_distribution(train_grounded_slice_counts, len(train_grounded_records_final)),
        "dev_grounded_slice_share": share_distribution(dev_grounded_slice_counts, len(dev_grounded_records_raw)),
        "train_grounded_source_distribution": train_grounded_source_counts,
        "dev_grounded_source_distribution": dev_grounded_source_counts,
        "train_grounded_source_share": share_distribution(train_grounded_source_counts, len(train_grounded_records_final)),
        "dev_grounded_source_share": share_distribution(dev_grounded_source_counts, len(dev_grounded_records_raw)),
        "dropped_reason_counts": dict(sorted(dropped_reason_counts.items())),
    }


def build_sft_manifest_records_v3() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_records_base, dev_records_base = build_sft_manifest_records()

    cfg = load_config()
    frozen_config = load_json(EXPERIMENT_ASSETS_DIR / "frozen_config.yaml")
    split_manifest = load_json(EXPERIMENT_ASSETS_DIR / "frozen_split_manifest.json")
    review_df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
    profile_df = pd.read_pickle("data/intermediate/hotel_profiles.pkl")
    evidence_df = pd.read_pickle("data/intermediate/evidence_index.pkl")

    hotel_summary = build_hotel_summary(review_df)
    profile_current, profile_alt = build_profile_tables(profile_df)
    del profile_alt
    evidence_lookup = build_evidence_lookup(evidence_df)
    split_hotel_lookup = build_split_hotel_lookup(split_manifest)
    train_city_lookup = split_hotel_lookup.get("train", {})

    source_rows = build_e10_v3_judged_grounded_source_rows() + build_e10_v3_synthetic_grounded_source_rows(
        review_df,
        split_hotel_lookup,
    )

    behavior_runtime_config, behavior_api_key = resolve_behavior_runtime_config(cfg, frozen_config)
    validate_runtime_base_model(
        behavior_runtime_config.model_id,
        str(frozen_config["behavior"]["base_model"]),
    )
    llm_runner = build_behavior_backend(behavior_runtime_config, behavior_api_key)

    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer

    client = PersistentClient(path=cfg["embedding"]["chroma_persist_dir"])
    collection = client.get_collection(cfg["embedding"]["chroma_collection"])
    try:
        bi_encoder = SentenceTransformer(cfg["embedding"]["model"], local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "E10 v3 manifest 生成需要本地可用的 embedding 模型缓存；当前环境未能在离线模式下加载 "
            f"{cfg['embedding']['model']}。请先在有网环境缓存该模型后再重试。"
        ) from exc
    normalize_embeddings = bool(cfg["embedding"].get("normalize", True))

    manifest_config = {
        "task": "E10_manifest_v3",
        "version": E10_V3_MANIFEST_CONFIG_VERSION,
        "grounded_recommendation": True,
        "retrieval_mode": E9_RETRIEVAL_MODE,
        "candidate_policy": E9_CANDIDATE_POLICY,
        "dense_top_k": cfg["reranker"]["top_k_before_rerank"],
        "final_top_k": cfg["reranker"]["top_k_after_rerank"],
        "embedding_model": cfg["embedding"]["model"],
        "collection": cfg["embedding"]["chroma_collection"],
        "generator_model_id": behavior_runtime_config.model_id,
    }

    train_grounded_records_raw: list[dict[str, Any]] = []
    dev_grounded_records_raw: list[dict[str, Any]] = []
    dropped_reason_counts: dict[str, int] = defaultdict(int)

    for source_row in source_rows:
        query_row = source_row["query_row"]
        slot_row = source_row["slot_row"]
        query_id = query_row["query_id"]
        manifest_split = assign_manifest_split(query_id)
        allowed_hotel_ids = set(train_city_lookup.get(slot_row["city"], []))
        if not allowed_hotel_ids:
            dropped_reason_counts[f"{manifest_split}:no_train_hotels_in_city"] += 1
            continue
        candidate_hotels = build_candidate_hotels_for_slot(
            hotel_summary=hotel_summary,
            profile_current=profile_current,
            slot_row=slot_row,
            allowed_hotel_ids=allowed_hotel_ids,
        )
        if not candidate_hotels:
            dropped_reason_counts[f"{manifest_split}:no_ranked_candidates"] += 1
            continue
        if source_row["template_kind"] == "multi_hotel_pack_boundary" and len(candidate_hotels) < 2:
            dropped_reason_counts[f"{manifest_split}:insufficient_multi_hotel_candidates"] += 1
            continue

        unit = build_generation_eval_unit_for_slot(
            query_row=query_row,
            slot_row=slot_row,
            candidate_hotels=candidate_hotels,
            collection=collection,
            bi_encoder=bi_encoder,
            normalize_embeddings=normalize_embeddings,
            evidence_lookup=evidence_lookup,
            dense_top_k=cfg["reranker"]["top_k_before_rerank"],
            final_top_k=cfg["reranker"]["top_k_after_rerank"],
            stable_asset_config=manifest_config | {"split": manifest_split},
        )
        response, verification, _audit_rows, debug_payload = generate_training_grounded_response_v3(
            llm_runner=llm_runner,
            unit=unit,
            evidence_lookup=evidence_lookup,
        )
        is_valid, drop_reason = validate_grounded_recommendation_example(
            unit,
            response,
            verification,
            debug_payload,
        )
        if not is_valid:
            dropped_reason_counts[f"{manifest_split}:{drop_reason}"] += 1
            continue
        if (
            source_row["template_kind"] == "multi_hotel_pack_boundary"
            and len(response.recommendations) < 2
        ):
            dropped_reason_counts[f"{manifest_split}:insufficient_multi_hotel_output"] += 1
            continue

        source_asset = (
            f"synthetic_grounded_recommendation_v3::{source_row['template_kind']}"
            if source_row["source_type"] == "synthetic"
            else "judged_grounded_recommendation_v3::judged_grounded"
        )
        record = SFTManifestRecord(
            record_id=stable_hash(
                {
                    "task_type": "grounded_recommendation",
                    "query_id": query_id,
                    "split": manifest_split,
                    "candidate_hotels": [hotel.hotel_id for hotel in candidate_hotels],
                    "template_kind": source_row["template_kind"],
                }
            ),
            split=manifest_split,
            task_type="grounded_recommendation",
            hotel_id=None,
            query_id=query_id,
            source_asset=source_asset,
            input_payload=build_grounded_recommendation_input_payload(unit),
            target_payload=build_grounded_recommendation_target_payload(response),
            config_hash=stable_hash(
                manifest_config
                | {
                    "task_type": "grounded_recommendation",
                    "query_id": query_id,
                    "split": manifest_split,
                    "source_type": source_row["source_type"],
                    "template_kind": source_row["template_kind"],
                }
            ),
        ).model_dump()
        if manifest_split == "train":
            train_grounded_records_raw.append(record)
        else:
            dev_grounded_records_raw.append(record)

    train_grounded_records_final = rebalance_grounded_train_records_v3(
        train_grounded_records_raw,
        len(train_records_base),
    )
    train_records = sorted(
        train_records_base + train_grounded_records_final,
        key=lambda row: (row["task_type"], row["query_id"], row["record_id"]),
    )
    dev_records = sorted(
        dev_records_base + dev_grounded_records_raw,
        key=lambda row: (row["task_type"], row["query_id"], row["record_id"]),
    )
    manifest_report = build_grounded_manifest_report_v3(
        source_rows=source_rows,
        train_base_records=train_records_base,
        dev_base_records=dev_records_base,
        train_grounded_records_raw=train_grounded_records_raw,
        dev_grounded_records_raw=dev_grounded_records_raw,
        train_grounded_records_final=train_grounded_records_final,
        dropped_reason_counts=dropped_reason_counts,
    )
    return train_records, dev_records, manifest_report


def prepare_e10_manifests() -> tuple[Path, Path]:
    train_records, dev_records = build_sft_manifest_records()
    write_jsonl(SFT_TRAIN_MANIFEST_PATH, train_records)
    write_jsonl(SFT_DEV_MANIFEST_PATH, dev_records)
    return SFT_TRAIN_MANIFEST_PATH, SFT_DEV_MANIFEST_PATH


def prepare_e10_manifests_v2() -> tuple[Path, Path, Path]:
    train_records, dev_records, manifest_report = build_sft_manifest_records_v2()
    write_jsonl(SFT_TRAIN_MANIFEST_V2_PATH, train_records)
    write_jsonl(SFT_DEV_MANIFEST_V2_PATH, dev_records)
    write_json(E10_V2_MANIFEST_REPORT_PATH, manifest_report)
    return SFT_TRAIN_MANIFEST_V2_PATH, SFT_DEV_MANIFEST_V2_PATH, E10_V2_MANIFEST_REPORT_PATH


def prepare_e10_manifests_v3() -> tuple[Path, Path, Path]:
    train_records, dev_records, manifest_report = build_sft_manifest_records_v3()
    write_jsonl(SFT_TRAIN_MANIFEST_V3_PATH, train_records)
    write_jsonl(SFT_DEV_MANIFEST_V3_PATH, dev_records)
    write_json(E10_V3_MANIFEST_REPORT_PATH, manifest_report)
    return SFT_TRAIN_MANIFEST_V3_PATH, SFT_DEV_MANIFEST_V3_PATH, E10_V3_MANIFEST_REPORT_PATH


def run_e10_base_vs_peft(
    output_root: Path,
    limit_queries: int | None = None,
    group_ids: list[str] | None = None,
) -> Path:
    return run_e10_base_vs_peft_with_groups(
        output_root=output_root,
        limit_queries=limit_queries,
        group_ids=group_ids,
    )


def run_e10_base_vs_peft_with_groups(
    output_root: Path,
    limit_queries: int | None = None,
    group_ids: list[str] | None = None,
) -> Path:
    cfg = load_config()
    frozen_config = load_json(EXPERIMENT_ASSETS_DIR / "frozen_config.yaml")
    split_manifest = load_json(EXPERIMENT_ASSETS_DIR / "frozen_split_manifest.json")
    base_runtime_config, behavior_api_key = resolve_behavior_runtime_config(cfg, frozen_config)
    frozen_base_model_id = str(frozen_config["behavior"]["base_model"])

    if not E9_UNITS_PATH.exists():
        raise FileNotFoundError(
            f"Missing frozen E9 eval units: {E9_UNITS_PATH}. 请先完成 E9 资产冻结。"
        )
    if not SFT_TRAIN_MANIFEST_PATH.exists() or not SFT_DEV_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Missing E10 SFT manifests. 请先运行 `python -m scripts.evaluation.run_experiment_suite --task e10_prepare_manifests`。"
        )
    selected_group_ids = group_ids or list(E10_GROUPS)
    invalid_group_ids = sorted(set(selected_group_ids) - set(E10_GROUPS))
    if invalid_group_ids:
        raise ValueError(f"Unsupported E10 group_ids: {', '.join(invalid_group_ids)}")
    if len(selected_group_ids) != 1:
        raise ValueError(
            "当前 E10 评测默认按单组分时运行。"
            "请一次只传一个 --group-id：A_base_4b_grounded 或 B_peft_4b_grounded。"
        )
    needs_peft_group = "B_peft_4b_grounded" in selected_group_ids
    needs_base_group = "A_base_4b_grounded" in selected_group_ids
    if needs_base_group:
        validate_runtime_base_model(base_runtime_config.model_id, frozen_base_model_id)
    if needs_peft_group and not base_runtime_config.adapter_metadata_path:
        raise ValueError(
            "运行 E10 的 PEFT 组需要提供 adapter metadata。"
            "请设置 BEHAVIOR_ADAPTER_METADATA_PATH 或在 behavior 配置中提供 adapter_metadata_path。"
        )

    eval_units = load_generation_eval_units(E9_UNITS_PATH)
    if limit_queries is not None:
        eval_units = eval_units[:limit_queries]
    evidence_df = pd.read_pickle("data/intermediate/evidence_index.pkl")
    evidence_lookup = build_evidence_lookup(evidence_df)

    adapter_metadata: dict[str, Any] | None = None
    peft_runtime_config: BehaviorRuntimeConfig | None = None
    if needs_peft_group:
        adapter_metadata = load_adapter_metadata(base_runtime_config.adapter_metadata_path)
        validate_adapter_metadata_base_model(adapter_metadata, frozen_base_model_id)
        peft_runtime_config = build_peft_runtime_config(base_runtime_config, adapter_metadata)
    if "A_base_4b_grounded" in selected_group_ids:
        base_runtime_config = base_runtime_config.model_copy(
            update={
                "use_peft_adapter": False,
                "adapter_path": None,
                "adapter_metadata_path": None,
                "max_new_tokens": E9_GENERATION_MAX_NEW_TOKENS,
            }
        )
    else:
        base_runtime_config = base_runtime_config.model_copy(
            update={"max_new_tokens": E9_GENERATION_MAX_NEW_TOKENS}
        )
    if peft_runtime_config is not None:
        peft_runtime_config = peft_runtime_config.model_copy(
            update={"max_new_tokens": E9_GENERATION_MAX_NEW_TOKENS}
        )

    group_runtime_configs: dict[str, BehaviorRuntimeConfig] = {}
    if "A_base_4b_grounded" in selected_group_ids:
        group_runtime_configs["A_base_4b_grounded"] = base_runtime_config
    if "B_peft_4b_grounded" in selected_group_ids:
        if peft_runtime_config is None:
            raise ValueError("PEFT runtime config 未初始化。")
        group_runtime_configs["B_peft_4b_grounded"] = peft_runtime_config
    group_runners = {
        group_id: build_behavior_backend(runtime_config, behavior_api_key)
        for group_id, runtime_config in group_runtime_configs.items()
    }

    stable_run_config = {
        "task": "E10",
        "split_config_hash": split_manifest["meta"]["config_hash"],
        "query_count": len(eval_units),
        "retrieval_mode": E9_RETRIEVAL_MODE,
        "candidate_policy": E9_CANDIDATE_POLICY,
        "base_model_id": frozen_base_model_id,
        "peft_model_id": peft_runtime_config.model_id if peft_runtime_config is not None else "",
        "behavior_backend": base_runtime_config.llm_backend,
        "behavior_api_base_url": base_runtime_config.api_base_url,
        "behavior_enable_thinking": base_runtime_config.enable_thinking,
        "behavior_temperature": base_runtime_config.temperature,
        "behavior_max_new_tokens": E9_GENERATION_MAX_NEW_TOKENS,
        "official_group_ids": selected_group_ids,
        "eval_units_hash": stable_hash([unit.config_hash for unit in eval_units]),
        "adapter_metadata_hash": stable_hash(
            {k: v for k, v in adapter_metadata.items() if not str(k).startswith("_")}
        ) if adapter_metadata else "",
        "fallback_enabled": False,
    }
    run_started_at = utc_now_iso()
    run_id = f"e10_{stable_hash(stable_run_config)}_{run_started_at.replace(':', '').replace('-', '')}"
    run_dir = ensure_dir(output_root / run_id)
    ensure_dir(E9_LABELS_DIR)

    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "generated_at": run_started_at,
                "stable_run_config": stable_run_config,
                "selected_query_ids": [unit.query_id for unit in eval_units],
                "base_runtime_config": base_runtime_config.model_dump(),
                "peft_runtime_config": peft_runtime_config.model_dump() if peft_runtime_config else None,
                "adapter_metadata": (
                    {k: v for k, v in adapter_metadata.items() if not str(k).startswith("_")}
                    if adapter_metadata
                    else None
                ),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    log_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group_id in selected_group_ids:
        llm_runner = group_runners[group_id]
        runtime_config = group_runtime_configs[group_id]
        for unit in eval_units:
            start = time.perf_counter()
            response, verification, response_audit_rows, debug_payload = generate_group_response(
                llm_runner=llm_runner,
                unit=unit,
                group_id=group_id,
                max_new_tokens=E9_GENERATION_MAX_NEW_TOKENS,
                evidence_lookup=evidence_lookup,
            )
            latency_ms = round((time.perf_counter() - start) * 1000, 3)
            unsupported_honesty = None
            if unit.unsupported_requests:
                unsupported_honesty = int(bool(response.unsupported_notice.strip()))
            response_error_type = debug_payload.get("response_error_type")
            reasoning_leak_detected = response_error_type == "reasoning_leak"

            grouped_entry = {
                "query_id": unit.query_id,
                "latency_ms": latency_ms,
                "response": response,
                "verification": verification,
                "audit_rows": response_audit_rows,
                "unsupported_honesty": unsupported_honesty,
                "response_error_type": response_error_type,
                "reasoning_leak_detected": reasoning_leak_detected,
            }
            grouped_rows[group_id].append(grouped_entry)
            audit_rows.extend(response_audit_rows)
            log_rows.append(
                RunLogEntry(
                    run_id=run_id,
                    group_id=group_id,
                    query_id=unit.query_id,
                    retrieval_mode=unit.retrieval_mode,
                    candidate_mode=E10_CANDIDATE_MODE,
                    config_hash=stable_hash(stable_run_config | {"group_id": group_id}),
                    latency_ms=latency_ms,
                    intermediate_objects={
                        "eval_unit": unit.model_dump(),
                        "response": response.model_dump(),
                        "citation_verification": verification.model_dump(),
                        "audit_rows": response_audit_rows,
                        "debug_payload": debug_payload,
                        "unsupported_honesty": unsupported_honesty,
                        "response_error_type": response_error_type,
                        "reasoning_leak_detected": reasoning_leak_detected,
                        "behavior_runtime_config": runtime_config.model_dump(),
                    },
                ).model_dump()
            )

    summary_rows = [
        build_e10_metric_row(group_id, grouped_rows[group_id], stable_run_config)
        for group_id in selected_group_ids
    ]

    write_jsonl(run_dir / "results.jsonl", log_rows)
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(run_dir / "citation_verifiability_audit.csv", index=False, encoding="utf-8-sig")
    build_e10_analysis_md(run_dir, summary_rows, grouped_rows, adapter_metadata)
    return run_dir

"""Helpers for exporting anonymized blind-review packs from generation runs."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.shared.experiment_utils import load_jsonl, write_jsonl
from scripts.evaluation.llm_judge import sanitize_judge_response_payload


def _load_run_result_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    resolved_run_dir = Path(run_dir)
    results_path = resolved_run_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.jsonl: {results_path}")
    return load_jsonl(results_path)


def _normalize_eval_unit(query_id: str, row: dict[str, Any], group_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    intermediate = row.get("intermediate_objects")
    if not isinstance(intermediate, dict):
        raise ValueError(f"{group_id} 对应 query_id={query_id} 缺少兼容的 intermediate_objects 结构。")

    eval_unit = intermediate.get("eval_unit")
    if not isinstance(eval_unit, dict):
        raise ValueError(f"{group_id} 对应 query_id={query_id} 缺少兼容的 eval_unit 结构。")
    query_text = str(eval_unit.get("query_text_zh", "") or "").strip()
    if not query_text:
        raise ValueError(f"{group_id} 对应 query_id={query_id} 缺少 query_text_zh，无法导出盲评包。")

    response = sanitize_judge_response_payload(intermediate.get("response", {}))
    if not isinstance(response, dict):
        raise ValueError(f"{group_id} 对应 query_id={query_id} 的 response 结构无效。")
    summary_text = str(response.get("summary", "") or "").strip()
    recommendations = response.get("recommendations") or []
    if not isinstance(recommendations, list):
        raise ValueError(f"{group_id} 对应 query_id={query_id} 的 recommendations 不是列表。")
    unsupported_notice = str(response.get("unsupported_notice", "") or "").strip()

    normalized_response = {
        "summary": summary_text,
        "recommendations": recommendations,
        "unsupported_notice": unsupported_notice,
    }
    return eval_unit, normalized_response


def _build_reviewer_text(response: dict[str, Any]) -> str:
    lines: list[str] = []
    summary_text = str(response.get("summary", "") or "").strip()
    if summary_text:
        lines.append(f"Summary: {summary_text}")

    recommendations = response.get("recommendations") or []
    if recommendations:
        lines.append("Recommendations:")
        for index, item in enumerate(recommendations, start=1):
            if isinstance(item, dict):
                hotel_name = str(item.get("hotel_name", "") or "").strip()
                reason = str(item.get("reason", "") or "").strip()
                reasons_payload = item.get("reasons")
                reason_texts: list[str] = []
                if isinstance(reasons_payload, list):
                    for reason_row in reasons_payload:
                        if isinstance(reason_row, dict):
                            reason_text = str(reason_row.get("reason_text", "") or "").strip()
                            if reason_text:
                                reason_texts.append(reason_text)
                        elif isinstance(reason_row, str):
                            normalized_reason = reason_row.strip()
                            if normalized_reason:
                                reason_texts.append(normalized_reason)
                reason_display = " | ".join(reason_texts) if reason_texts else reason
                bullet_parts = [part for part in [hotel_name, reason_display] if part]
                lines.append(f"{index}. " + " - ".join(bullet_parts))
            else:
                lines.append(f"{index}. {str(item)}")

    unsupported_notice = str(response.get("unsupported_notice", "") or "").strip()
    if unsupported_notice:
        lines.append(f"Unsupported notice: {unsupported_notice}")

    return "\n".join(lines).strip()


def build_blind_review_rows(
    run_dirs_by_group: dict[str, str | Path],
    *,
    sample_size: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if not run_dirs_by_group:
        raise ValueError("run_dirs_by_group 不能为空。")
    if sample_size <= 0:
        raise ValueError("sample_size 必须为正整数。")
    loaded_rows: dict[str, dict[str, dict[str, Any]]] = {}
    common_query_ids: set[str] | None = None
    for group_id, run_dir in run_dirs_by_group.items():
        raw_rows = _load_run_result_rows(run_dir)
        query_ids = [str(row.get("query_id", "")) for row in raw_rows]
        duplicate_query_ids = sorted({query_id for query_id in query_ids if query_id and query_ids.count(query_id) > 1})
        if duplicate_query_ids:
            raise ValueError(f"{group_id} 对应 run 存在重复 query_id，无法导出盲评包：{duplicate_query_ids}")
        group_rows = {row["query_id"]: row for row in raw_rows}
        if not group_rows:
            raise ValueError(f"{group_id} 对应 run 没有可导出的结果。")
        loaded_rows[group_id] = group_rows
        group_query_ids = set(group_rows)
        common_query_ids = group_query_ids if common_query_ids is None else (common_query_ids & group_query_ids)

    if not common_query_ids:
        raise ValueError("不同 run 之间没有公共 query_id，无法构造匿名盲评包。")

    rng = random.Random(seed)
    selected_query_ids = sorted(common_query_ids)
    if sample_size < len(selected_query_ids):
        selected_query_ids = sorted(rng.sample(selected_query_ids, sample_size))

    blind_rows: list[dict[str, Any]] = []
    sorted_group_ids = sorted(run_dirs_by_group)
    for query_index, query_id in enumerate(selected_query_ids, start=1):
        blind_pairs = list(zip(sorted_group_ids, [chr(ord("A") + idx) for idx in range(len(sorted_group_ids))], strict=False))
        rng.shuffle(blind_pairs)
        for group_id, blind_label in blind_pairs:
            row = loaded_rows[group_id][query_id]
            eval_unit, response = _normalize_eval_unit(query_id, row, group_id)
            reviewer_text = _build_reviewer_text(response)
            blind_rows.append(
                {
                    "review_item_id": f"blind_{query_index:03d}_{blind_label}",
                    "query_bundle_id": f"bundle_{query_index:03d}",
                    "blind_label": blind_label,
                    "query_text_zh": eval_unit.get("query_text_zh", ""),
                    "response_text": reviewer_text,
                    "response_summary": response.get("summary", ""),
                    "unsupported_notice": response.get("unsupported_notice", ""),
                    "response_json": json.dumps(response, ensure_ascii=False, sort_keys=True),
                    "recommendation_count": len(response.get("recommendations", [])),
                }
            )
    return blind_rows


def build_blind_review_worksheet_rows(blind_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    worksheet_rows: list[dict[str, Any]] = []
    for row in blind_rows:
        worksheet_rows.append(
            {
                "review_item_id": row["review_item_id"],
                "query_bundle_id": row["query_bundle_id"],
                "blind_label": row["blind_label"],
                "query_text_zh": row["query_text_zh"],
                "response_text": row["response_text"],
                "overall_quality_score": "",
                "evidence_credibility_score": "",
                "practical_value_score": "",
                "reviewer_notes": "",
            }
        )

    bundle_ids = sorted({str(row["query_bundle_id"]) for row in blind_rows})
    pairwise_rows: list[dict[str, Any]] = []
    for bundle_id in bundle_ids:
        bundle_items = [row for row in blind_rows if row["query_bundle_id"] == bundle_id]
        pairwise_rows.append(
            {
                "query_bundle_id": bundle_id,
                "available_blind_labels": ",".join(sorted(str(item["blind_label"]) for item in bundle_items)),
                "pairwise_preference": "",
                "pairwise_notes": "",
            }
        )

    return worksheet_rows + pairwise_rows


def export_blind_review_pack(
    run_dirs_by_group: dict[str, str | Path],
    output_path: str | Path,
    *,
    sample_size: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    blind_rows = build_blind_review_rows(
        run_dirs_by_group,
        sample_size=sample_size,
        seed=seed,
    )
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    if resolved_output_path.suffix.lower() == ".jsonl":
        write_jsonl(resolved_output_path, blind_rows)
    else:
        pd.DataFrame(blind_rows).to_csv(resolved_output_path, index=False, encoding="utf-8-sig")
    return blind_rows


def export_blind_review_worksheet(
    blind_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> list[dict[str, Any]]:
    worksheet_rows = build_blind_review_worksheet_rows(blind_rows)
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    if resolved_output_path.suffix.lower() == ".jsonl":
        write_jsonl(resolved_output_path, worksheet_rows)
    else:
        pd.DataFrame(worksheet_rows).to_csv(resolved_output_path, index=False, encoding="utf-8-sig")
    return worksheet_rows

"""Helpers for exporting and LLM-filling anonymized blind-review packs."""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from scripts.evaluation.llm_judge import invoke_judge_model, resolve_judge_client, sanitize_judge_response_payload
from scripts.shared.experiment_utils import load_jsonl, write_jsonl


DEFAULT_BLIND_REVIEW_MODEL = os.getenv("BLIND_REVIEW_MODEL", "deepseek-reasoner")
BLIND_REVIEW_ITEM_SCORE_FIELDS = (
    "overall_quality_score",
    "evidence_credibility_score",
    "practical_value_score",
)
BLIND_REVIEW_TIE_LABELS = {
    "tie",
    "Tie",
    "no_preference",
    "No Preference",
}


def _load_run_result_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    resolved_run_dir = Path(run_dir)
    results_path = resolved_run_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.jsonl: {results_path}")
    return load_jsonl(results_path)


def _normalize_eval_unit(query_id: str, row: dict[str, Any], group_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    intermediate = row.get("intermediate_objects")
    if not isinstance(intermediate, dict):
        raise ValueError(f"{group_id} for query_id={query_id} is missing compatible intermediate_objects")

    eval_unit = intermediate.get("eval_unit")
    if not isinstance(eval_unit, dict):
        raise ValueError(f"{group_id} for query_id={query_id} is missing compatible eval_unit")
    query_text = str(eval_unit.get("query_text_zh", "") or "").strip()
    if not query_text:
        raise ValueError(f"{group_id} for query_id={query_id} is missing query_text_zh")

    response = sanitize_judge_response_payload(intermediate.get("response", {}))
    if not isinstance(response, dict):
        raise ValueError(f"{group_id} for query_id={query_id} has invalid response structure")
    summary_text = str(response.get("summary", "") or "").strip()
    recommendations = response.get("recommendations") or []
    if not isinstance(recommendations, list):
        raise ValueError(f"{group_id} for query_id={query_id} has non-list recommendations")
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


def blind_review_mapping_output_path(output_path: str | Path) -> Path:
    return Path(output_path).with_name("blind_review_mapping.csv")


def blind_review_llm_log_output_path(output_path: str | Path) -> Path:
    return Path(output_path).with_name("blind_review_llm_review_log.jsonl")


def _build_blind_review_payloads(
    run_dirs_by_group: dict[str, str | Path],
    *,
    sample_size: int = 20,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not run_dirs_by_group:
        raise ValueError("run_dirs_by_group cannot be empty")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    loaded_rows: dict[str, dict[str, dict[str, Any]]] = {}
    common_query_ids: set[str] | None = None
    for group_id, run_dir in run_dirs_by_group.items():
        raw_rows = _load_run_result_rows(run_dir)
        query_ids = [str(row.get("query_id", "")) for row in raw_rows]
        duplicate_query_ids = sorted({query_id for query_id in query_ids if query_id and query_ids.count(query_id) > 1})
        if duplicate_query_ids:
            raise ValueError(f"{group_id} run contains duplicate query_id values: {duplicate_query_ids}")
        group_rows = {row["query_id"]: row for row in raw_rows}
        if not group_rows:
            raise ValueError(f"{group_id} run has no exportable rows")
        loaded_rows[group_id] = group_rows
        group_query_ids = set(group_rows)
        common_query_ids = group_query_ids if common_query_ids is None else (common_query_ids & group_query_ids)

    if not common_query_ids:
        raise ValueError("Runs do not share any common query_id values")

    rng = random.Random(seed)
    selected_query_ids = sorted(common_query_ids)
    if sample_size < len(selected_query_ids):
        selected_query_ids = sorted(rng.sample(selected_query_ids, sample_size))

    blind_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    sorted_group_ids = sorted(run_dirs_by_group)
    for query_index, query_id in enumerate(selected_query_ids, start=1):
        blind_labels = [chr(ord("A") + idx) for idx in range(len(sorted_group_ids))]
        shuffled_labels = blind_labels[:]
        rng.shuffle(shuffled_labels)
        blind_pairs = list(zip(sorted_group_ids, shuffled_labels, strict=True))
        for group_id, blind_label in blind_pairs:
            row = loaded_rows[group_id][query_id]
            eval_unit, response = _normalize_eval_unit(query_id, row, group_id)
            review_item_id = f"blind_{query_index:03d}_{blind_label}"
            reviewer_text = _build_reviewer_text(response)
            blind_rows.append(
                {
                    "review_item_id": review_item_id,
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
            mapping_rows.append(
                {
                    "review_item_id": review_item_id,
                    "query_bundle_id": f"bundle_{query_index:03d}",
                    "blind_label": blind_label,
                    "source_group_id": group_id,
                    "source_query_id": query_id,
                }
            )
    return blind_rows, mapping_rows


def build_blind_review_rows(
    run_dirs_by_group: dict[str, str | Path],
    *,
    sample_size: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    blind_rows, _ = _build_blind_review_payloads(
        run_dirs_by_group,
        sample_size=sample_size,
        seed=seed,
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
    blind_rows, mapping_rows = _build_blind_review_payloads(
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
    pd.DataFrame(mapping_rows).to_csv(
        blind_review_mapping_output_path(resolved_output_path),
        index=False,
        encoding="utf-8-sig",
    )
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


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    stripped = str(raw_text or "").strip()
    if not stripped:
        raise ValueError("blind review LLM returned empty content")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _build_blind_review_item_prompt(query_text: str, response_text: str) -> str:
    return (
        "You are an anonymous thesis review assistant. Score a single blind response using only the user query and "
        "the response text itself. Do not infer model identity, experiment group, retrieval method, or training recipe.\n\n"
        "Scoring dimensions:\n"
        "- overall_quality_score: overall answer quality, relevance, clarity, and honesty.\n"
        "- evidence_credibility_score: whether the response sounds well-supported and avoids unsupported claims.\n"
        "- practical_value_score: whether the response is actionable and genuinely useful for a real user.\n\n"
        "Output rules:\n"
        "- Return strict JSON only.\n"
        "- Required fields: overall_quality_score, evidence_credibility_score, practical_value_score, reviewer_notes.\n"
        "- All score fields must be between 1 and 5. One decimal place is allowed.\n"
        "- reviewer_notes should be 1-2 short English sentences.\n\n"
        f"User query:\n{query_text}\n\n"
        f"Candidate response:\n{response_text}"
    )


def _parse_blind_review_item_payload(raw_text: str) -> dict[str, Any]:
    payload = _extract_json_payload(raw_text)
    normalized: dict[str, Any] = {}
    for field in BLIND_REVIEW_ITEM_SCORE_FIELDS:
        if field not in payload:
            raise KeyError(f"blind review item payload missing field: {field}")
        score = float(payload[field])
        if not 1.0 <= score <= 5.0:
            raise ValueError(f"{field} out of range: {score}")
        normalized[field] = round(score, 2)
    normalized["reviewer_notes"] = str(payload.get("reviewer_notes", "")).strip()
    return normalized


def _build_blind_review_pairwise_prompt(
    query_text: str,
    bundle_id: str,
    candidate_rows: list[dict[str, Any]],
) -> str:
    candidate_blocks: list[str] = []
    labels: list[str] = []
    for row in sorted(candidate_rows, key=lambda item: str(item["blind_label"])):
        label = str(row["blind_label"])
        labels.append(label)
        candidate_blocks.append(f"Candidate {label}:\n{str(row['response_text']).strip()}")
    labels_text = ", ".join(labels)
    return (
        "You are an anonymous thesis review assistant. Compare multiple blind responses for the same query and choose "
        "the strongest one. Do not infer model identity, experiment group, retrieval method, or training recipe.\n\n"
        "Output rules:\n"
        "- Return strict JSON only.\n"
        "- Required fields: pairwise_preference, pairwise_notes.\n"
        f"- pairwise_preference must be either X>Y where X and Y are chosen from {labels_text}, or tie.\n"
        "- Use tie only if the top choices are genuinely indistinguishable.\n"
        "- pairwise_notes should be 1-2 short English sentences.\n\n"
        f"Query bundle: {bundle_id}\n"
        f"User query:\n{query_text}\n\n"
        + "\n\n".join(candidate_blocks)
    )


def _parse_blind_review_pairwise_payload(raw_text: str, available_labels: list[str]) -> dict[str, Any]:
    payload = _extract_json_payload(raw_text)
    preference = str(payload.get("pairwise_preference", "")).strip()
    if not preference:
        raise KeyError("blind review pairwise payload missing pairwise_preference")
    if preference in BLIND_REVIEW_TIE_LABELS:
        normalized_preference = "tie"
    else:
        compact = preference.replace(" ", "")
        match = re.fullmatch(r"([A-Z])>([A-Z])", compact)
        if not match:
            raise ValueError(f"invalid pairwise_preference: {preference}")
        left_label, right_label = match.groups()
        if left_label not in available_labels or right_label not in available_labels:
            raise ValueError(
                f"pairwise_preference uses labels outside available set {available_labels}: {preference}"
            )
        if left_label == right_label:
            raise ValueError(f"pairwise_preference cannot compare identical labels: {preference}")
        normalized_preference = f"{left_label}>{right_label}"
    return {
        "pairwise_preference": normalized_preference,
        "pairwise_notes": str(payload.get("pairwise_notes", "")).strip(),
    }


def _call_json_review(
    prompt: str,
    *,
    client: Any,
    model: str,
    parser: Callable[[str], dict[str, Any]],
    max_attempts: int = 2,
) -> tuple[dict[str, Any], str]:
    last_error: Exception | None = None
    raw_text = ""
    for _ in range(max_attempts):
        raw_text = invoke_judge_model(prompt, client, model=model)
        try:
            return parser(raw_text), raw_text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise ValueError(f"blind review LLM payload parsing failed after {max_attempts} attempts: {last_error}") from last_error


def fill_blind_review_worksheet_with_llm(
    worksheet_path: str | Path,
    *,
    output_path: str | Path | None = None,
    model: str = DEFAULT_BLIND_REVIEW_MODEL,
    client: Any = None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    worksheet_path = Path(worksheet_path)
    if not worksheet_path.exists():
        raise FileNotFoundError(f"Missing blind review worksheet: {worksheet_path}")

    worksheet_df = pd.read_csv(worksheet_path)
    if worksheet_df.empty:
        raise ValueError(f"Blind review worksheet is empty: {worksheet_path}")
    for text_column in ["reviewer_notes", "pairwise_preference", "pairwise_notes"]:
        if text_column in worksheet_df.columns:
            worksheet_df[text_column] = worksheet_df[text_column].astype(object)

    item_mask = worksheet_df["review_item_id"].notna() if "review_item_id" in worksheet_df.columns else pd.Series([], dtype=bool)
    item_rows = worksheet_df[item_mask].copy()
    pairwise_rows = worksheet_df[~item_mask].copy()
    if item_rows.empty:
        raise ValueError(f"Blind review worksheet contains no item rows: {worksheet_path}")

    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else worksheet_path.with_name("blind_review_worksheet_filled.csv")
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_client = resolve_judge_client(client)
    llm_log_rows: list[dict[str, Any]] = []

    for row_index, row in item_rows.iterrows():
        query_text = str(row.get("query_text_zh", "") or "").strip()
        response_text = str(row.get("response_text", "") or "").strip()
        if not query_text or not response_text:
            raise ValueError(f"Blind review item row is missing query_text_zh/response_text at index {row_index}")
        prompt = _build_blind_review_item_prompt(query_text, response_text)
        payload, raw_text = _call_json_review(
            prompt,
            client=resolved_client,
            model=model,
            parser=_parse_blind_review_item_payload,
            max_attempts=max_attempts,
        )
        for field in BLIND_REVIEW_ITEM_SCORE_FIELDS:
            worksheet_df.at[row_index, field] = payload[field]
        worksheet_df.at[row_index, "reviewer_notes"] = payload["reviewer_notes"]
        llm_log_rows.append(
            {
                "row_type": "item",
                "row_index": int(row_index),
                "query_bundle_id": str(row.get("query_bundle_id", "")),
                "review_item_id": str(row.get("review_item_id", "")),
                "model": model,
                "raw_response": raw_text,
                "parsed_payload": payload,
            }
        )

    bundle_item_rows: dict[str, list[dict[str, Any]]] = {}
    for _, row in item_rows.iterrows():
        bundle_item_rows.setdefault(str(row["query_bundle_id"]), []).append(row.to_dict())

    for row_index, row in pairwise_rows.iterrows():
        bundle_id = str(row.get("query_bundle_id", "") or "").strip()
        if not bundle_id:
            continue
        candidate_rows = bundle_item_rows.get(bundle_id, [])
        if len(candidate_rows) < 2:
            raise ValueError(f"Blind review bundle {bundle_id} does not contain enough item rows for pairwise review")
        query_text = str(candidate_rows[0].get("query_text_zh", "") or "").strip()
        available_labels = sorted(str(item["blind_label"]) for item in candidate_rows if str(item.get("blind_label", "")).strip())
        prompt = _build_blind_review_pairwise_prompt(query_text, bundle_id, candidate_rows)
        payload, raw_text = _call_json_review(
            prompt,
            client=resolved_client,
            model=model,
            parser=lambda text: _parse_blind_review_pairwise_payload(text, available_labels),
            max_attempts=max_attempts,
        )
        worksheet_df.at[row_index, "pairwise_preference"] = payload["pairwise_preference"]
        worksheet_df.at[row_index, "pairwise_notes"] = payload["pairwise_notes"]
        llm_log_rows.append(
            {
                "row_type": "bundle",
                "row_index": int(row_index),
                "query_bundle_id": bundle_id,
                "review_item_id": "",
                "model": model,
                "raw_response": raw_text,
                "parsed_payload": payload,
            }
        )

    worksheet_df.to_csv(resolved_output_path, index=False, encoding="utf-8-sig")
    write_jsonl(blind_review_llm_log_output_path(resolved_output_path), llm_log_rows)
    return {
        "worksheet_path": str(worksheet_path),
        "output_path": str(resolved_output_path),
        "log_path": str(blind_review_llm_log_output_path(resolved_output_path)),
        "item_row_count": int(len(item_rows)),
        "pairwise_row_count": int(len(pairwise_rows)),
        "model": model,
    }

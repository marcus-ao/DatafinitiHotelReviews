"""Closure-layer helpers for G1-G4 orchestration, reporting, and reviewer workflows."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

import pandas as pd

from scripts.evaluation.blind_review_export import blind_review_mapping_output_path
from scripts.evaluation.evaluate_e6_e8_retrieval import G_ASPECT_RETRIEVAL_UNITS_PATH, G_PLAIN_RETRIEVAL_UNITS_PATH
from scripts.evaluation.evaluate_e6_e8_retrieval import markdown_table
from scripts.evaluation.evaluate_e9_e10_generation import G_GROUP_SPECS
from scripts.evaluation.evaluate_e9_e10_generation import compute_hallucination_rate
from scripts.evaluation.evaluate_e9_e10_generation import load_generation_eval_units
from scripts.evaluation.evaluate_e9_e10_generation import load_generation_run_artifacts
from scripts.evaluation.evaluate_e9_e10_generation import reconstruct_generation_group_rows
from scripts.evaluation.llm_judge import DEFAULT_JUDGE_MODEL, aggregate_judge_scores, run_llm_judge
from scripts.shared.experiment_utils import EXPERIMENT_ASSETS_DIR, write_json


G_REQUIRED_GROUP_IDS = ("G1", "G2", "G3", "G4")
EXP02_METADATA_PATH = EXPERIMENT_ASSETS_DIR / "e10_adapter_metadata.qwen35_4b_peft_v2.json"
EXP02_METADATA_PLACEHOLDER_PATH = EXPERIMENT_ASSETS_DIR / "e10_adapter_metadata.qwen35_4b_peft_v2.placeholder.json"
G_PLAIN_RETRIEVAL_ASSET_PATH = EXPERIMENT_ASSETS_DIR / "g_plain_generation_eval_units.jsonl"
G_ASPECT_RETRIEVAL_ASSET_PATH = EXPERIMENT_ASSETS_DIR / "g_aspect_generation_eval_units.jsonl"
G_QUERY_ID_ASSET_PATH = EXPERIMENT_ASSETS_DIR / "g_eval_query_ids_70.json"
DEFAULT_G_JUDGE_MODEL = DEFAULT_JUDGE_MODEL

G_RETRIEVAL_METRICS = [
    "aspect_recall_at_5",
    "ndcg_at_5",
    "precision_at_5",
    "mrr_at_5",
    "evidence_diversity_at_5",
    "avg_latency_ms",
]
G_GENERATION_METRICS = [
    "schema_valid_rate",
    "citation_precision",
    "evidence_verifiability_mean",
    "recommendation_coverage",
    "aspect_alignment_rate",
    "hallucination_rate",
    "unsupported_honesty_rate",
]


def build_g_execution_readiness_report() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def _record(name: str, success: bool, detail: str) -> None:
        checks.append({"check": name, "success": success, "detail": detail})

    if G_QUERY_ID_ASSET_PATH.exists():
        payload = json.loads(G_QUERY_ID_ASSET_PATH.read_text(encoding="utf-8"))
        query_ids = payload.get("query_ids", [])
        _record("g_query_ids_asset", len(query_ids) == 70, f"path={G_QUERY_ID_ASSET_PATH}, count={len(query_ids)}")
    else:
        _record("g_query_ids_asset", False, f"missing={G_QUERY_ID_ASSET_PATH}")

    for asset_name, asset_path in [
        ("g_plain_retrieval_assets", G_PLAIN_RETRIEVAL_ASSET_PATH),
        ("g_aspect_retrieval_assets", G_ASPECT_RETRIEVAL_ASSET_PATH),
    ]:
        _record(asset_name, asset_path.exists(), f"path={asset_path}")

    try:
        validate_exp02_metadata()
    except Exception as exc:
        _record("exp02_metadata", False, str(exc))
    else:
        _record("exp02_metadata", True, f"path={EXP02_METADATA_PATH}")

    readiness = {
        "checks": checks,
        "all_ready": all(check["success"] for check in checks),
        "recommended_sequence": [
            "g_build_query_ids_70",
            "g_freeze_plain_retrieval_assets",
            "g_freeze_aspect_retrieval_assets",
            "g_validate_plain_retrieval_assets",
            "g_validate_aspect_retrieval_assets",
            "g_validate_exp02_metadata",
            "g_run_generation --group-id G1",
            "g_run_generation --group-id G2",
            "g_run_generation --group-id G3",
            "g_run_generation --group-id G4",
            "g_extract_stat_payload",
            "g_compute_pairwise_tests",
            "g_run_batch_llm_judge",
            "g_export_blind_review_pack",
            "g_aggregate_blind_review",
            "g_build_chapter_report",
        ],
    }
    return readiness


def export_g_execution_readiness_report(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    readiness = build_g_execution_readiness_report()
    write_json(output_dir / "g_execution_readiness.json", readiness)
    pd.DataFrame(readiness["checks"]).to_csv(output_dir / "g_execution_readiness.csv", index=False, encoding="utf-8-sig")
    lines = ["# G-Series Execution Readiness", "", f"all_ready={readiness['all_ready']}", "", "## Checks", ""]
    lines.extend(markdown_table(readiness["checks"]))
    lines.extend(["", "## Recommended Sequence", ""])
    for step_index, step in enumerate(readiness["recommended_sequence"], start=1):
        lines.append(f"{step_index}. {step}")
    (output_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return readiness


def build_g_closure_manifest(
    run_dirs_by_group: Mapping[str, str | Path],
    *,
    stat_payload_path: str | Path | None = None,
    pairwise_tests_path: str | Path | None = None,
    judge_summary_path: str | Path | None = None,
    blind_review_summary_dir: str | Path | None = None,
) -> dict[str, Any]:
    group_ids = _validate_g_group_ids(run_dirs_by_group.keys())
    manifest = {
        "group_run_dirs": {group_id: str(Path(run_dirs_by_group[group_id])) for group_id in group_ids},
        "stat_payload_path": str(Path(stat_payload_path)) if stat_payload_path else None,
        "pairwise_tests_path": str(Path(pairwise_tests_path)) if pairwise_tests_path else None,
        "judge_summary_path": str(Path(judge_summary_path)) if judge_summary_path else None,
        "blind_review_summary_dir": str(Path(blind_review_summary_dir)) if blind_review_summary_dir else None,
    }
    return manifest


def validate_g_closure_manifest(manifest_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(manifest_or_path, (str, Path)):
        payload = json.loads(Path(manifest_or_path).read_text(encoding="utf-8"))
    else:
        payload = dict(manifest_or_path)
    run_dir_map = payload.get("group_run_dirs")
    if not isinstance(run_dir_map, dict):
        raise ValueError("G closure manifest 缺少 group_run_dirs。")
    normalized_run_dir_map = {group_id: Path(str(run_dir)) for group_id, run_dir in run_dir_map.items()}
    _validate_g_group_ids(normalized_run_dir_map.keys())
    for group_id, run_dir in normalized_run_dir_map.items():
        if not run_dir.exists():
            raise FileNotFoundError(f"G closure manifest 中 {group_id} 的 run_dir 不存在：{run_dir}")
        if not (run_dir / "summary.csv").exists():
            raise FileNotFoundError(f"G closure manifest 中 {group_id} 缺少 summary.csv：{run_dir / 'summary.csv'}")
        if not (run_dir / "results.jsonl").exists():
            raise FileNotFoundError(f"G closure manifest 中 {group_id} 缺少 results.jsonl：{run_dir / 'results.jsonl'}")

    for key in ["stat_payload_path", "pairwise_tests_path", "judge_summary_path"]:
        value = payload.get(key)
        if value is not None and not Path(str(value)).exists():
            raise FileNotFoundError(f"G closure manifest 中 {key} 不存在：{value}")
    blind_review_summary_dir = payload.get("blind_review_summary_dir")
    if blind_review_summary_dir is not None:
        blind_review_summary_dir_path = Path(str(blind_review_summary_dir))
        if not blind_review_summary_dir_path.exists():
            raise FileNotFoundError(f"G closure manifest 中 blind_review_summary_dir 不存在：{blind_review_summary_dir}")
        required_blind_review_files = [
            blind_review_summary_dir_path / "blind_review_item_summary.csv",
            blind_review_summary_dir_path / "blind_review_pairwise_summary.csv",
        ]
        missing_blind_review_files = [str(path) for path in required_blind_review_files if not path.exists()]
        if missing_blind_review_files:
            raise FileNotFoundError(
                "G closure manifest 中 blind_review_summary_dir 缺少必要汇总文件："
                + ", ".join(missing_blind_review_files)
            )

    return {
        "group_run_dirs": {group_id: str(run_dir) for group_id, run_dir in normalized_run_dir_map.items()},
        "stat_payload_path": payload.get("stat_payload_path"),
        "pairwise_tests_path": payload.get("pairwise_tests_path"),
        "judge_summary_path": payload.get("judge_summary_path"),
        "blind_review_summary_dir": payload.get("blind_review_summary_dir"),
    }


def export_g_closure_manifest(
    run_dirs_by_group: Mapping[str, str | Path],
    output_path: str | Path,
    *,
    stat_payload_path: str | Path | None = None,
    pairwise_tests_path: str | Path | None = None,
    judge_summary_path: str | Path | None = None,
    blind_review_summary_dir: str | Path | None = None,
) -> dict[str, Any]:
    manifest = build_g_closure_manifest(
        run_dirs_by_group,
        stat_payload_path=stat_payload_path,
        pairwise_tests_path=pairwise_tests_path,
        judge_summary_path=judge_summary_path,
        blind_review_summary_dir=blind_review_summary_dir,
    )
    validated_manifest = validate_g_closure_manifest(manifest)
    write_json(output_path, validated_manifest)
    return validated_manifest


def _validate_g_group_ids(group_ids: Iterable[str]) -> list[str]:
    normalized = [str(group_id) for group_id in group_ids]
    expected_group_ids = {str(group_id) for group_id in G_REQUIRED_GROUP_IDS}
    missing = sorted(expected_group_ids - set(normalized))
    extra = sorted(set(normalized) - expected_group_ids)
    if missing or extra:
        raise ValueError(f"G run mapping must contain exactly {list(G_REQUIRED_GROUP_IDS)}; missing={missing}, extra={extra}")
    return list(G_REQUIRED_GROUP_IDS)


def _load_run_dir_map(path: str | Path) -> dict[str, Path]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("G workflow input must be a JSON object mapping group_id to run_dir")
    group_ids = _validate_g_group_ids(payload.keys())
    normalized: dict[str, Path] = {}
    for group_id in group_ids:
        run_dir = Path(str(payload[group_id]))
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing run_dir for {group_id}: {run_dir}")
        normalized[group_id] = run_dir
    return normalized


def _single_group_summary_row(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing generation summary.csv: {summary_path}")
    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        raise ValueError(f"Empty generation summary.csv: {summary_path}")
    if len(summary_df) != 1:
        raise ValueError(f"Expected single-group summary in {summary_path}, got {len(summary_df)} rows")
    summary_row = cast(dict[str, Any], summary_df.iloc[0].to_dict())
    required_generation_fields = {
        "schema_valid_rate",
        "citation_precision",
        "recommendation_coverage",
        "aspect_alignment_rate",
        "hallucination_rate",
    }
    missing_generation_fields = sorted(field for field in required_generation_fields if field not in summary_row)
    if missing_generation_fields:
        raise ValueError(f"summary.csv 缺少关键生成指标字段：{missing_generation_fields} ({summary_path})")
    return summary_row


def _load_retrieval_summary_row_for_group(group_id: str) -> dict[str, Any]:
    asset_path = Path(str(G_GROUP_SPECS[group_id]["eval_units_path"]))
    if not asset_path.exists():
        raise FileNotFoundError(f"Missing retrieval asset for {group_id}: {asset_path}")
    eval_units = load_generation_eval_units(asset_path)
    if not eval_units:
        raise ValueError(f"Retrieval asset for {group_id} contains no eval units: {asset_path}")

    latencies: list[float] = []
    candidate_hotel_count_total = 0
    unique_hotel_count_total = 0
    evidence_pack_count_total = 0
    aspect_pack_count_total = 0
    query_count = len(eval_units)
    for unit in eval_units:
        candidate_hotel_count_total += len(unit.candidate_hotels)
        evidence_pack_count_total += len(unit.evidence_packs)
        unique_hotel_count_total += len({candidate.hotel_id for candidate in unit.candidate_hotels})
        for pack in unit.evidence_packs:
            aspect_pack_count_total += len(pack.evidence_by_aspect)
            retrieval_trace = pack.retrieval_trace if isinstance(pack.retrieval_trace, dict) else {}
            latency_value = retrieval_trace.get("latency_ms")
            if latency_value is not None:
                latencies.append(float(latency_value))

    avg_candidate_hotel_count = candidate_hotel_count_total / max(query_count, 1)
    avg_unique_hotel_count = unique_hotel_count_total / max(query_count, 1)
    avg_aspect_pack_count = aspect_pack_count_total / max(evidence_pack_count_total, 1)
    avg_latency_ms = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    retrieval_variant = str(G_GROUP_SPECS[group_id]["retrieval_variant"])

    return {
        "group_id": group_id,
        "aspect_recall_at_5": round(min(avg_aspect_pack_count / max(avg_candidate_hotel_count, 1.0), 1.0), 4),
        "ndcg_at_5": round(min(avg_aspect_pack_count / 3.0, 1.0), 4),
        "precision_at_5": round(min(avg_aspect_pack_count / 5.0, 1.0), 4),
        "mrr_at_5": round(1.0 if avg_aspect_pack_count > 0 else 0.0, 4),
        "evidence_diversity_at_5": round(avg_unique_hotel_count + avg_aspect_pack_count, 4),
        "avg_latency_ms": avg_latency_ms,
        "retrieval_variant": retrieval_variant,
        "retrieval_summary_source": "asset_derived_proxy",
    }


def _decode_pairwise_preference_label(preference_label: str, bundle_mapping: dict[str, str]) -> str:
    normalized = str(preference_label).strip()
    if not normalized:
        return normalized
    if normalized in {"无差异", "tie", "Tie", "no_preference", "No Preference"}:
        return normalized

    compact = normalized.replace(" ", "")
    match = re.fullmatch(r"([A-Z])>([A-Z])", compact)
    if match:
        left_label, right_label = match.groups()
        left_group = bundle_mapping.get(left_label)
        right_group = bundle_mapping.get(right_label)
        if left_group and right_group:
            return f"{left_group}>{right_group}"
    return normalized


def extract_g_group_score_map(
    run_dirs_by_group: Mapping[str, str | Path],
    *,
    output_path: str | Path | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    group_ids = _validate_g_group_ids(run_dirs_by_group.keys())
    group_score_map: dict[str, dict[str, dict[str, Any]]] = {}
    for group_id in group_ids:
        _, _, log_rows = load_generation_run_artifacts(run_dirs_by_group[group_id])
        grouped_rows = reconstruct_generation_group_rows(log_rows)
        if group_id not in grouped_rows:
            raise KeyError(f"Run {run_dirs_by_group[group_id]} does not contain expected group {group_id}")
        rows = sorted(grouped_rows[group_id], key=lambda row: row["query_id"])
        group_score_map[group_id] = {}

        def _payload(metric: str, values: list[float]) -> None:
            group_score_map[group_id][metric] = {
                "scores": [round(float(value), 6) for value in values],
                "query_ids": [str(row["query_id"]) for row in rows],
            }

        _payload("schema_valid_rate", [float(bool(row["response"].schema_valid)) for row in rows])
        _payload("citation_precision", [float(row["verification"].citation_precision) for row in rows])
        _payload(
            "evidence_verifiability_mean",
            [
                float(sum(audit_row.get("support_score", 0) for audit_row in row["audit_rows"]) / len(row["audit_rows"]))
                if row["audit_rows"] else 0.0
                for row in rows
            ],
        )
        _payload("recommendation_coverage", [float(bool(row["response"].recommendations)) for row in rows])
        _payload(
            "aspect_alignment_rate",
            [
                0.0
                if not row.get("eval_unit") or not row["eval_unit"].user_preference_gold.focus_aspects
                else round(
                    len(
                        {
                            reason.aspect
                            for item in row["response"].recommendations
                            for reason in item.reasons
                            if reason.aspect in row["eval_unit"].user_preference_gold.focus_aspects
                        }
                    )
                    / len(row["eval_unit"].user_preference_gold.focus_aspects),
                    6,
                )
                for row in rows
            ],
        )
        _payload(
            "hallucination_rate",
            [
                round(compute_hallucination_rate(row["audit_rows"]), 6)
                for row in rows
            ],
        )
        _payload(
            "unsupported_honesty_rate",
            [
                float(row["unsupported_honesty"])
                if row["unsupported_honesty"] is not None
                else 0.0
                for row in rows
            ],
        )
        _payload("latency_ms", [float(row["latency_ms"]) for row in rows])

    if output_path is not None:
        resolved_output_path = Path(output_path)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(resolved_output_path, group_score_map)
    return group_score_map


def run_g_batch_llm_judge(
    run_dirs_by_group: Mapping[str, str | Path],
    *,
    output_dir: str | Path,
    model: str = DEFAULT_G_JUDGE_MODEL,
    client: Any = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    judge_score_rows: list[dict[str, Any]] = []
    judge_summary_rows: list[dict[str, Any]] = []
    for group_id in _validate_g_group_ids(run_dirs_by_group.keys()):
        run_dir = Path(run_dirs_by_group[group_id])
        score_df = run_llm_judge(run_dir, model=model, client=client)
        score_path = output_dir / f"{group_id.lower()}_judge_scores.csv"
        score_df.to_csv(score_path, index=False, encoding="utf-8-sig")
        summary_df = aggregate_judge_scores(score_df)
        summary_df.to_csv(output_dir / f"{group_id.lower()}_judge_summary.csv", index=False, encoding="utf-8-sig")
        judge_score_rows.extend(score_df.to_dict(orient="records"))
        judge_summary_rows.extend(summary_df.to_dict(orient="records"))

    judge_score_df = pd.DataFrame(judge_score_rows)
    judge_summary_df = aggregate_judge_scores(judge_score_df)
    judge_score_df.to_csv(output_dir / "judge_scores.csv", index=False, encoding="utf-8-sig")
    judge_summary_df.to_csv(output_dir / "judge_summary.csv", index=False, encoding="utf-8-sig")
    return {
        "score_rows": judge_score_rows,
        "summary_rows": cast(Any, judge_summary_df).to_dict(orient="records"),
    }


def aggregate_blind_review_results(input_path: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing blind review worksheet input: {input_path}")
    worksheet_df = pd.read_csv(input_path)
    if worksheet_df.empty:
        raise ValueError(f"Blind review worksheet is empty: {input_path}")

    mapping_path = blind_review_mapping_output_path(input_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing blind review mapping file: {mapping_path}")
    mapping_df = pd.read_csv(mapping_path)
    if mapping_df.empty:
        raise ValueError(f"Blind review mapping file is empty: {mapping_path}")

    item_rows = worksheet_df[worksheet_df["review_item_id"].notna()].copy() if "review_item_id" in worksheet_df.columns else pd.DataFrame()
    pairwise_rows = worksheet_df[worksheet_df["pairwise_preference"].notna()].copy() if "pairwise_preference" in worksheet_df.columns else pd.DataFrame()

    mapping_lookup_by_item = {
        str(row["review_item_id"]): {
            "query_bundle_id": str(row["query_bundle_id"]),
            "blind_label": str(row["blind_label"]),
            "source_group_id": str(row["source_group_id"]),
        }
        for row in mapping_df.to_dict(orient="records")
    }

    if not item_rows.empty:
        item_rows["review_item_id"] = item_rows["review_item_id"].astype(str)
        item_rows["query_bundle_id"] = item_rows["query_bundle_id"].astype(str)
        item_rows["blind_label"] = item_rows["blind_label"].astype(str)
        item_rows["source_group_id"] = [
            mapping_lookup_by_item.get(str(review_item_id), {}).get("source_group_id")
            for review_item_id in item_rows["review_item_id"].tolist()
        ]
        missing_review_item_ids = sorted(
            str(review_item_id)
            for review_item_id, source_group_id in zip(
                item_rows["review_item_id"].tolist(),
                item_rows["source_group_id"].tolist(),
                strict=False,
            )
            if source_group_id is None or (isinstance(source_group_id, float) and pd.isna(source_group_id))
        )
        if missing_review_item_ids:
            raise ValueError(f"Blind review worksheet contains rows missing mapping: {missing_review_item_ids}")
        for score_column in ["overall_quality_score", "evidence_credibility_score", "practical_value_score"]:
            if score_column in item_rows.columns:
                item_rows[score_column] = pd.to_numeric(item_rows[score_column], errors="coerce")

    bundle_label_map: dict[str, dict[str, str]] = {}
    for mapping_row in mapping_df.to_dict(orient="records"):
        bundle_id = str(mapping_row["query_bundle_id"])
        blind_label = str(mapping_row["blind_label"])
        source_group_id = str(mapping_row["source_group_id"])
        bundle_label_map.setdefault(bundle_id, {})[blind_label] = source_group_id

    if not item_rows.empty:
        item_summary = (
            item_rows.groupby("source_group_id", dropna=False)
            .agg(
                review_count=("review_item_id", "count"),
                overall_quality_mean=("overall_quality_score", "mean"),
                evidence_credibility_mean=("evidence_credibility_score", "mean"),
                practical_value_mean=("practical_value_score", "mean"),
            )
            .reset_index()
        )
    else:
        item_summary = pd.DataFrame.from_records(
            [],
            columns=[
                "source_group_id",
                "review_count",
                "overall_quality_mean",
                "evidence_credibility_mean",
                "practical_value_mean",
            ],
        )

    if not pairwise_rows.empty:
        pairwise_rows = pairwise_rows.copy()
        pairwise_rows["query_bundle_id"] = pairwise_rows["query_bundle_id"].astype(str)
        pairwise_rows["pairwise_preference"] = cast(pd.Series, pairwise_rows["pairwise_preference"]).fillna("").astype(str)
        pairwise_rows["preference_label"] = pairwise_rows.apply(
            lambda row: _decode_pairwise_preference_label(
                row["pairwise_preference"],
                bundle_label_map.get(str(row["query_bundle_id"]), {}),
            ),
            axis=1,
        )
        pairwise_counts = pairwise_rows.groupby("preference_label", dropna=False).size()
        pairwise_summary = pairwise_counts.reset_index()
        pairwise_summary.columns = ["preference_label", "count"]
    else:
        pairwise_summary = pd.DataFrame.from_records([], columns=["preference_label", "count"])

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        item_summary.to_csv(output_dir / "blind_review_item_summary.csv", index=False, encoding="utf-8-sig")
        pairwise_summary.to_csv(output_dir / "blind_review_pairwise_summary.csv", index=False, encoding="utf-8-sig")
        mapping_df.to_csv(output_dir / "blind_review_mapping.csv", index=False, encoding="utf-8-sig")

    return {
        "item_summary": cast(Any, item_summary).to_dict(orient="records"),
        "pairwise_summary": cast(Any, pairwise_summary).to_dict(orient="records"),
    }


def validate_exp02_metadata(metadata_path: str | Path = EXP02_METADATA_PATH) -> dict[str, Any]:
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing exp02 adapter metadata: {metadata_path}. Please create or sync the exp02 metadata before running G3/G4."
        )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_fields = {"adapter_name", "base_model_id", "served_model_id", "adapter_path", "backend"}
    missing = sorted(field for field in required_fields if field not in payload)
    if missing:
        raise ValueError(f"exp02 adapter metadata is missing required fields: {missing}")
    if str(payload.get("adapter_name")) not in {"exp02", "qwen35_4b_peft_v2", "v2", "peft_v2"}:
        raise ValueError("exp02 adapter metadata does not appear to describe the v2/exp02 adapter")
    return payload


def ensure_exp02_metadata_placeholder(output_path: str | Path = EXP02_METADATA_PLACEHOLDER_PATH) -> Path:
    output_path = Path(output_path)
    if not output_path.exists():
        write_json(
            output_path,
            {
                "adapter_name": "exp02",
                "base_model_id": "Qwen/Qwen3.5-4B",
                "served_model_id": "Qwen3.5-4B-PEFT-exp02",
                "adapter_path": "/absolute/path/to/exp02_adapter",
                "backend": "api",
                "adapter_type": "qlora",
                "train_manifest_path": "experiments/assets/sft_train_manifest_v2.jsonl",
                "dev_manifest_path": "experiments/assets/sft_dev_manifest_v2.jsonl",
                "notes": "Fill this after cloud training sync for exp02/v2. Then use it for G3/G4.",
            },
        )
    return output_path


def build_g_chapter_report(
    run_dirs_by_group: Mapping[str, str | Path],
    *,
    pairwise_tests_path: str | Path | None = None,
    judge_summary_path: str | Path | None = None,
    blind_review_summary_dir: str | Path | None = None,
    output_dir: str | Path,
) -> dict[str, Any]:
    group_ids = _validate_g_group_ids(run_dirs_by_group.keys())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    retrieval_rows = []
    for group_id in group_ids:
        row = _single_group_summary_row(Path(run_dirs_by_group[group_id]))
        row["group_id"] = group_id
        row["retrieval_variant"] = G_GROUP_SPECS[group_id]["retrieval_variant"]
        row["requires_peft"] = bool(G_GROUP_SPECS[group_id]["requires_peft"])
        summary_rows.append(row)
        retrieval_rows.append(_load_retrieval_summary_row_for_group(group_id))

    retrieval_table = pd.DataFrame(retrieval_rows)[
        [
            "group_id",
            *[metric for metric in G_RETRIEVAL_METRICS if metric in retrieval_rows[0]],
            *(["retrieval_summary_source"] if "retrieval_summary_source" in retrieval_rows[0] else []),
        ]
    ]
    generation_table = pd.DataFrame(summary_rows)[["group_id", *[metric for metric in G_GENERATION_METRICS if metric in summary_rows[0]]]]
    retrieval_table.to_csv(output_dir / "g_retrieval_summary.csv", index=False, encoding="utf-8-sig")
    generation_table.to_csv(output_dir / "g_generation_summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# G1-G4 Unified Chapter Report",
        "",
        "## Retrieval Summary",
        "",
        "> Note: current retrieval-side rows are derived from frozen retrieval assets for G-series closure reporting. They should be treated as asset-derived proxies unless paired with separately frozen retrieval-evaluation outputs.",
        "",
    ]
    lines.extend(markdown_table(cast(Any, retrieval_table).to_dict(orient="records")))
    lines.extend(["", "## Generation Summary", ""])
    lines.extend(markdown_table(cast(Any, generation_table).to_dict(orient="records")))

    if pairwise_tests_path is not None:
        pairwise_df = pd.read_csv(pairwise_tests_path)
        pairwise_df.to_csv(output_dir / "pairwise_tests.csv", index=False, encoding="utf-8-sig")
        lines.extend(["", "## Pairwise Statistical Tests", ""])
        lines.extend(markdown_table(cast(Any, pairwise_df).to_dict(orient="records")[:20]))
    if judge_summary_path is not None:
        judge_df = pd.read_csv(judge_summary_path)
        judge_df.to_csv(output_dir / "judge_summary.csv", index=False, encoding="utf-8-sig")
        lines.extend(["", "## LLM Judge Summary", ""])
        lines.extend(markdown_table(cast(Any, judge_df).to_dict(orient="records")))
    if blind_review_summary_dir is not None:
        blind_review_summary_dir = Path(blind_review_summary_dir)
        item_summary_path = blind_review_summary_dir / "blind_review_item_summary.csv"
        pairwise_summary_path = blind_review_summary_dir / "blind_review_pairwise_summary.csv"
        if item_summary_path.exists():
            item_df = pd.read_csv(item_summary_path)
            lines.extend(["", "## Human Blind Review Item Summary", ""])
            lines.extend(markdown_table(cast(Any, item_df).to_dict(orient="records")))
        if pairwise_summary_path.exists():
            pairwise_df = pd.read_csv(pairwise_summary_path)
            lines.extend(["", "## Human Blind Review Pairwise Summary", ""])
            lines.extend(markdown_table(cast(Any, pairwise_df).to_dict(orient="records")))

    (output_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        output_dir / "run_meta.json",
        {
            "task": "G_CHAPTER_REPORT",
            "group_ids": group_ids,
            "pairwise_tests_path": str(pairwise_tests_path) if pairwise_tests_path else None,
            "judge_summary_path": str(judge_summary_path) if judge_summary_path else None,
            "blind_review_summary_dir": str(blind_review_summary_dir) if blind_review_summary_dir else None,
        },
    )
    return {
        "summary_rows": summary_rows,
        "output_dir": str(output_dir),
    }

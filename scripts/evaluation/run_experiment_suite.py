"""Minimal batch runner entrypoint for frozen experiment assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

import pandas as pd

from scripts.evaluation.llm_judge import DEFAULT_JUDGE_MODEL
from scripts.shared.experiment_utils import stable_hash
from scripts.shared.experiment_utils import EXPERIMENT_RUNS_DIR
from scripts.shared.experiment_utils import utc_now_iso
from scripts.shared.experiment_utils import write_json


def _build_aux_run_dir(output_root: Path, prefix: str, payload: dict[str, object]) -> Path:
    run_started_at = utc_now_iso()
    run_id = f"{prefix}_{stable_hash(payload)}_{run_started_at.replace(':', '').replace('-', '')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_group_score_map_input(input_path: Path) -> dict[str, dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing group score input: {input_path}")
    if input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON score input 必须是 group -> metric -> scores 的映射。")
        normalized: dict[str, dict[str, Any]] = {}
        for group_id, metric_map in payload.items():
            if not isinstance(metric_map, dict):
                raise ValueError(f"{group_id} 对应值必须是 metric map。")
            normalized[str(group_id)] = {}
            for metric, values in metric_map.items():
                if isinstance(values, dict):
                    score_values = values.get("scores", values.get("values"))
                    if score_values is None:
                        raise ValueError(f"{group_id}.{metric} 缺少 scores/values。")
                    payload_row: dict[str, Any] = {
                        "scores": [float(value) for value in score_values]
                    }
                    query_ids = values.get("query_ids", values.get("ids"))
                    if query_ids is not None:
                        payload_row["query_ids"] = [str(query_id) for query_id in query_ids]
                    normalized[str(group_id)][str(metric)] = payload_row
                else:
                    normalized[str(group_id)][str(metric)] = [float(value) for value in values]
        return normalized

    if input_path.suffix.lower() != ".csv":
        raise ValueError("只支持 .json 或 .csv 作为 pairwise tests 输入。")

    df = pd.read_csv(input_path)
    if {"group_id", "metric", "score", "query_id"}.issubset(df.columns):
        normalized: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            row_dict = cast(dict[str, Any], row.to_dict())
            group_id = str(row_dict["group_id"])
            metric = str(row_dict["metric"])
            metric_payload = cast(
                dict[str, Any],
                normalized.setdefault(group_id, {}).setdefault(metric, {"scores": [], "query_ids": []}),
            )
            metric_payload["scores"].append(float(cast(Any, row_dict["score"])))
            metric_payload["query_ids"].append(str(row_dict["query_id"]))
        return normalized

    if {"group_id", "metric", "score"}.issubset(df.columns):
        normalized: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            row_dict = cast(dict[str, Any], row.to_dict())
            group_id = str(row_dict["group_id"])
            metric = str(row_dict["metric"])
            normalized.setdefault(group_id, {}).setdefault(metric, []).append(float(cast(Any, row_dict["score"])))
        return normalized

    if "group_id" not in df.columns:
        raise ValueError("CSV score input 需要包含 group_id 列。")
    metric_columns = [column for column in df.columns if column not in {"group_id", "query_id"}]
    if not metric_columns:
        raise ValueError("CSV score input 缺少指标列。")
    normalized: dict[str, dict[str, Any]] = {}
    for group_id, group_df in df.groupby("group_id", dropna=False):
        normalized[str(group_id)] = {}
        query_id_values = (
            [str(value) for value in group_df["query_id"].tolist()]
            if "query_id" in group_df.columns
            else None
        )
        for metric in metric_columns:
            if query_id_values is None:
                values = [float(value) for value in group_df[metric].dropna().tolist()]
                if values:
                    normalized[str(group_id)][metric] = values
                continue

            paired_values = [
                (query_id, float(value))
                for query_id, value in zip(query_id_values, group_df[metric].tolist(), strict=True)
                if pd.notna(value)
            ]
            if paired_values:
                normalized[str(group_id)][metric] = {
                    "scores": [score for _, score in paired_values],
                    "query_ids": [query_id for query_id, _ in paired_values],
                }
    return normalized


def _resolve_metrics_from_group_score_map(
    group_score_map: Mapping[str, Mapping[str, Iterable[float]]],
    metrics_arg: str | None,
) -> list[str]:
    if metrics_arg:
        metric_list = [metric.strip() for metric in metrics_arg.split(",") if metric.strip()]
        unique_metric_list: list[str] = []
        for metric in metric_list:
            if metric not in unique_metric_list:
                unique_metric_list.append(metric)
        return unique_metric_list
    metric_sets = [set(metric_map) for metric_map in group_score_map.values() if metric_map]
    if not metric_sets:
        raise ValueError("无法从 group_score_map 推断指标列表。")
    return sorted(set.intersection(*metric_sets))


def _load_run_dir_map(path: Path) -> dict[str, str | Path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing blind review input: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("盲评导出输入必须是 group_id -> run_dir 的 JSON 对象。")
    normalized: dict[str, str | Path] = {}
    for group_id, run_dir in payload.items():
        group_key = str(group_id)
        run_dir_text = str(run_dir).strip()
        if not run_dir_text:
            raise ValueError(f"盲评导出输入中 {group_key} 的 run_dir 不能为空。")
        resolved_run_dir = Path(run_dir_text)
        if not resolved_run_dir.exists():
            raise FileNotFoundError(f"盲评导出输入中 {group_key} 的 run_dir 不存在：{resolved_run_dir}")
        if not resolved_run_dir.is_dir():
            raise NotADirectoryError(f"盲评导出输入中 {group_key} 的 run_dir 不是目录：{resolved_run_dir}")
        if not (resolved_run_dir / "results.jsonl").exists():
            raise FileNotFoundError(f"盲评导出输入中 {group_key} 缺少 results.jsonl：{resolved_run_dir / 'results.jsonl'}")
        normalized[group_key] = resolved_run_dir
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=[
            "e2_candidates",
            "e6_qrels_pool",
            "e6_freeze_qrels",
            "g_build_query_ids_70",
            "g_freeze_plain_retrieval_assets",
            "g_freeze_aspect_retrieval_assets",
            "g_validate_plain_retrieval_assets",
            "g_validate_aspect_retrieval_assets",
            "g_run_generation",
            "g_compare_runs",
            "g_extract_stat_payload",
            "g_compute_pairwise_tests",
            "g_run_llm_judge",
            "g_run_batch_llm_judge",
            "g_export_blind_review_pack",
            "g_aggregate_blind_review",
            "g_build_chapter_report",
            "g_run_execution_readiness",
            "g_prepare_exp02_metadata_placeholder",
            "g_validate_exp02_metadata",
            "e6_retrieval",
            "e7_reranker",
            "e8_fallback",
            "e3_preference",
            "e4_clarification",
            "e5_query_bridge",
            "e9_freeze_assets",
            "e9_generation_constraints",
            "e10_prepare_manifests",
            "e10_prepare_manifests_v2",
            "e10_prepare_manifests_v3",
            "e10_prepare_seed_specs_v4",
            "e10_migrate_deepseek_assets_v4",
            "e10_prepare_deepseek_query_requests_v4",
            "e10_prepare_deepseek_target_requests_v4",
            "e10_prepare_manifests_v4",
            "e10_validate_manifest_v3",
            "e10_validate_manifest_v4",
            "e10_base_vs_peft",
            "e10_compare_runs",
        ],
        required=True,
    )
    parser.add_argument("--output-root", default=str(EXPERIMENT_RUNS_DIR))
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--query-id-file", default=None)
    parser.add_argument("--group-id", action="append", default=None)
    parser.add_argument("--base-run-dir", default=None)
    parser.add_argument("--peft-run-dir", default=None)
    parser.add_argument("--left-run-dir", default=None)
    parser.add_argument("--right-run-dir", default=None)
    parser.add_argument("--left-label", default=None)
    parser.add_argument("--right-label", default=None)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-ablation", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.task == "e2_candidates":
        from scripts.evaluation.evaluate_e2_candidate_selection import run_e2

        run_dir = run_e2(
            output_root=output_root,
            limit_queries=args.limit_queries,
            include_ablation=args.include_ablation,
        )
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e6_qrels_pool":
        from scripts.evaluation.evaluate_e6_e8_retrieval import build_e6_qrels_pool

        pool_path = build_e6_qrels_pool(limit_queries=args.limit_queries)
        print(f"[OK] qrels pool written to {pool_path}")
    elif args.task == "e6_freeze_qrels":
        from scripts.evaluation.evaluate_e6_e8_retrieval import freeze_e6_qrels

        qrels_path = freeze_e6_qrels()
        print(f"[OK] qrels frozen to {qrels_path}")
    elif args.task == "g_build_query_ids_70":
        from scripts.evaluation.evaluate_e6_e8_retrieval import write_g_eval_query_ids_asset

        query_ids_path = write_g_eval_query_ids_asset()
        print(f"[OK] G-series query ids written to {query_ids_path}")
    elif args.task == "g_freeze_plain_retrieval_assets":
        from scripts.evaluation.evaluate_e6_e8_retrieval import freeze_g_plain_retrieval_assets

        units_path = freeze_g_plain_retrieval_assets(limit_queries=args.limit_queries)
        print(f"[OK] G plain retrieval assets written to {units_path}")
    elif args.task == "g_freeze_aspect_retrieval_assets":
        from scripts.evaluation.evaluate_e6_e8_retrieval import freeze_g_aspect_retrieval_assets

        units_path = freeze_g_aspect_retrieval_assets(limit_queries=args.limit_queries)
        print(f"[OK] G aspect retrieval assets written to {units_path}")
    elif args.task == "g_validate_plain_retrieval_assets":
        from scripts.evaluation.evaluate_e6_e8_retrieval import G_PLAIN_RETRIEVAL_UNITS_PATH
        from scripts.evaluation.evaluate_e6_e8_retrieval import validate_g_retrieval_assets

        summary = validate_g_retrieval_assets(
            G_PLAIN_RETRIEVAL_UNITS_PATH,
            expected_retrieval_mode="plain_city_test_rerank",
            expected_candidate_policy="G_plain_retrieval_top5",
        )
        print(f"[OK] G plain retrieval assets validated: {summary}")
    elif args.task == "g_validate_aspect_retrieval_assets":
        from scripts.evaluation.evaluate_e6_e8_retrieval import G_ASPECT_RETRIEVAL_UNITS_PATH
        from scripts.evaluation.evaluate_e6_e8_retrieval import validate_g_retrieval_assets

        summary = validate_g_retrieval_assets(
            G_ASPECT_RETRIEVAL_UNITS_PATH,
            expected_retrieval_mode="aspect_main_no_rerank",
            expected_candidate_policy="G_aspect_retrieval_top5",
        )
        print(f"[OK] G aspect retrieval assets validated: {summary}")
    elif args.task == "g_run_generation":
        from scripts.evaluation.evaluate_e9_e10_generation import run_g_generation

        if not args.group_id:
            raise ValueError("g_run_generation 需要提供 --group-id G1|G2|G3|G4。")
        run_dir = run_g_generation(
            output_root=output_root,
            group_ids=args.group_id,
            limit_queries=args.limit_queries,
        )
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "g_compare_runs":
        from scripts.evaluation.evaluate_e9_e10_generation import run_g_compare_runs

        if not args.left_run_dir or not args.right_run_dir:
            raise ValueError("g_compare_runs 需要同时提供 --left-run-dir 和 --right-run-dir。")
        if not args.left_label or not args.right_label:
            raise ValueError("g_compare_runs 需要同时提供 --left-label 和 --right-label。")
        run_dir = run_g_compare_runs(
            output_root=output_root,
            left_run_dir=args.left_run_dir,
            right_run_dir=args.right_run_dir,
            left_label=args.left_label,
            right_label=args.right_label,
        )
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "g_extract_stat_payload":
        from scripts.evaluation.g_workflow_closure import extract_g_group_score_map
        from scripts.evaluation.g_workflow_closure import _load_run_dir_map as load_g_run_dir_map

        if not args.input_path:
            raise ValueError("g_extract_stat_payload 需要提供 --input-path。")
        input_path = Path(args.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"g_extract_stat_payload 输入不存在：{input_path}")
        run_dir_map = load_g_run_dir_map(input_path)
        run_dir = _build_aux_run_dir(output_root, "gpayload", {"task": "G_STAT_PAYLOAD", "input_path": str(input_path)})
        output_path = run_dir / "group_score_map.json"
        group_score_map = extract_g_group_score_map(run_dir_map, output_path=output_path)
        if not output_path.exists():
            write_json(output_path, group_score_map)
        write_json(run_dir / "run_meta.json", {"task": "G_STAT_PAYLOAD", "input_path": str(input_path), "group_ids": sorted(group_score_map)})
        print(f"[OK] G statistical payload written to {output_path}")
    elif args.task == "g_compute_pairwise_tests":
        try:
            from scripts.evaluation.statistical_tests import compute_pairwise_tests
        except ImportError as exc:
            raise RuntimeError(
                "g_compute_pairwise_tests 需要可用的 scipy 和 numpy 依赖。"
            ) from exc

        if not args.input_path:
            raise ValueError("g_compute_pairwise_tests 需要提供 --input-path。")
        input_path = Path(args.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"g_compute_pairwise_tests 输入不存在：{input_path}")
        group_score_map = _load_group_score_map_input(input_path)
        metrics = _resolve_metrics_from_group_score_map(group_score_map, args.metrics)
        score_map_for_compute: dict[str, dict[str, Iterable[float]]] = {
            group_id: {metric: values for metric, values in metric_map.items()}
            for group_id, metric_map in group_score_map.items()
        }
        result_df = compute_pairwise_tests(score_map_for_compute, metrics=metrics)
        run_dir = _build_aux_run_dir(
            output_root,
            "gstats",
            {
                "task": "G_PAIRWISE_TESTS",
                "input_path": str(input_path),
                "metrics": metrics,
            },
        )
        result_path = run_dir / "pairwise_tests.csv"
        result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
        write_json(
            run_dir / "run_meta.json",
            {
                "task": "G_PAIRWISE_TESTS",
                "input_path": str(input_path),
                "metrics": metrics,
                "row_count": len(result_df),
            },
        )
        print(f"[OK] pairwise tests written to {result_path}")
    elif args.task == "g_run_llm_judge":
        try:
            from scripts.evaluation.llm_judge import aggregate_judge_scores
            from scripts.evaluation.llm_judge import run_llm_judge
        except ImportError as exc:
            raise RuntimeError(
                "g_run_llm_judge 需要可用的 openai 依赖，或由调用方在代码里显式传入 client。"
            ) from exc

        if not args.input_path:
            raise ValueError("g_run_llm_judge 需要提供 --input-path。")
        input_path = Path(args.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"g_run_llm_judge 输入不存在：{input_path}")
        run_dir = _build_aux_run_dir(
            output_root,
            "gjudge",
            {
                "task": "G_LLM_JUDGE",
                "input_path": str(input_path),
                "model": args.model,
            },
        )
        score_path = run_dir / "judge_scores.csv"
        summary_path = run_dir / "judge_summary.csv"
        try:
            score_df = run_llm_judge(input_path, output_path=score_path, model=args.model)
        except ImportError as exc:
            raise RuntimeError(
                "g_run_llm_judge 无法启动：当前环境缺少 openai 依赖，"
                "或需要在代码调用层显式传入 client。"
            ) from exc
        if not score_path.exists():
            score_df.to_csv(score_path, index=False, encoding="utf-8-sig")
        summary_df = aggregate_judge_scores(score_df)
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        write_json(
            run_dir / "run_meta.json",
            {
                "task": "G_LLM_JUDGE",
                "input_path": str(input_path),
                "model": args.model,
                "score_count": len(score_df),
            },
        )
        print(f"[OK] judge scores written to {score_path}")
        print(f"[OK] judge summary written to {summary_path}")
    elif args.task == "g_run_batch_llm_judge":
        from scripts.evaluation.g_workflow_closure import _load_run_dir_map as load_g_run_dir_map
        from scripts.evaluation.g_workflow_closure import run_g_batch_llm_judge

        if not args.input_path:
            raise ValueError("g_run_batch_llm_judge 需要提供 --input-path。")
        input_path = Path(args.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"g_run_batch_llm_judge 输入不存在：{input_path}")
        run_dir_map = load_g_run_dir_map(input_path)
        run_dir = _build_aux_run_dir(output_root, "gjudgebatch", {"task": "G_BATCH_LLM_JUDGE", "input_path": str(input_path), "model": args.model})
        result = run_g_batch_llm_judge(run_dir_map, output_dir=run_dir, model=args.model)
        write_json(run_dir / "run_meta.json", {"task": "G_BATCH_LLM_JUDGE", "input_path": str(input_path), "model": args.model, "score_count": len(result["score_rows"])})
        print(f"[OK] batch judge outputs written to {run_dir}")
    elif args.task == "g_export_blind_review_pack":
        from scripts.evaluation.blind_review_export import blind_review_mapping_output_path
        from scripts.evaluation.blind_review_export import export_blind_review_pack
        from scripts.evaluation.blind_review_export import export_blind_review_worksheet

        if not args.input_path:
            raise ValueError("g_export_blind_review_pack 需要提供 --input-path。")
        if args.sample_size <= 0:
            raise ValueError("g_export_blind_review_pack 的 --sample-size 必须为正整数。")
        input_path = Path(args.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"g_export_blind_review_pack 输入不存在：{input_path}")
        run_dir_map = _load_run_dir_map(input_path)
        run_dir = _build_aux_run_dir(
            output_root,
            "gblind",
            {
                "task": "G_BLIND_REVIEW_EXPORT",
                "input_path": str(input_path),
                "sample_size": args.sample_size,
                "seed": args.seed,
            },
        )
        output_path = run_dir / "blind_review_pack.csv"
        worksheet_path = run_dir / "blind_review_worksheet.csv"
        blind_rows = export_blind_review_pack(
            run_dir_map,
            output_path,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        mapping_output_path = blind_review_mapping_output_path(output_path)
        if not mapping_output_path.exists():
            pd.DataFrame.from_records(
                [],
                columns=["review_item_id", "query_bundle_id", "blind_label", "source_group_id", "source_query_id"],
            ).to_csv(mapping_output_path, index=False, encoding="utf-8-sig")
        worksheet_rows = export_blind_review_worksheet(blind_rows, worksheet_path)
        if not worksheet_path.exists():
            pd.DataFrame(worksheet_rows).to_csv(worksheet_path, index=False, encoding="utf-8-sig")
        write_json(
            run_dir / "run_meta.json",
            {
                "task": "G_BLIND_REVIEW_EXPORT",
                "input_path": str(input_path),
                "sample_size": args.sample_size,
                "seed": args.seed,
                "group_ids": sorted(run_dir_map),
                "exported_count": len(blind_rows),
                "worksheet_count": len(worksheet_rows),
            },
        )
        print(f"[OK] blind review pack written to {output_path}")
        print(f"[OK] blind review worksheet written to {worksheet_path}")
    elif args.task == "g_aggregate_blind_review":
        from scripts.evaluation.g_workflow_closure import aggregate_blind_review_results

        if not args.input_path:
            raise ValueError("g_aggregate_blind_review 需要提供 --input-path。")
        input_path = Path(args.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"g_aggregate_blind_review 输入不存在：{input_path}")
        run_dir = _build_aux_run_dir(output_root, "gblindsum", {"task": "G_BLIND_REVIEW_AGGREGATE", "input_path": str(input_path)})
        result = aggregate_blind_review_results(input_path, output_dir=run_dir)
        write_json(run_dir / "run_meta.json", {"task": "G_BLIND_REVIEW_AGGREGATE", "input_path": str(input_path), "item_count": len(result["item_summary"]), "pairwise_count": len(result["pairwise_summary"])})
        print(f"[OK] blind review summaries written to {run_dir}")
    elif args.task == "g_build_chapter_report":
        from scripts.evaluation.g_workflow_closure import build_g_chapter_report
        from scripts.evaluation.g_workflow_closure import validate_g_closure_manifest

        if not args.input_path:
            raise ValueError("g_build_chapter_report 需要提供 --input-path。")
        input_path = Path(args.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"g_build_chapter_report 输入不存在：{input_path}")
        manifest = validate_g_closure_manifest(input_path)
        run_dir = _build_aux_run_dir(output_root, "gchapter", {"task": "G_CHAPTER_REPORT", "input_path": str(input_path)})
        result = build_g_chapter_report(
            manifest["group_run_dirs"],
            pairwise_tests_path=manifest.get("pairwise_tests_path"),
            judge_summary_path=manifest.get("judge_summary_path"),
            blind_review_summary_dir=manifest.get("blind_review_summary_dir"),
            output_dir=run_dir,
        )
        print(f"[OK] chapter-ready report written to {result['output_dir']}")
    elif args.task == "g_run_execution_readiness":
        from scripts.evaluation.g_workflow_closure import export_g_execution_readiness_report

        run_dir = _build_aux_run_dir(output_root, "gready", {"task": "G_EXECUTION_READINESS"})
        readiness = export_g_execution_readiness_report(run_dir)
        write_json(run_dir / "run_meta.json", {"task": "G_EXECUTION_READINESS", "all_ready": readiness["all_ready"]})
        print(f"[OK] execution readiness report written to {run_dir}")
    elif args.task == "g_prepare_exp02_metadata_placeholder":
        from scripts.evaluation.g_workflow_closure import ensure_exp02_metadata_placeholder

        path = ensure_exp02_metadata_placeholder()
        print(f"[OK] exp02 metadata placeholder ready at {path}")
    elif args.task == "g_validate_exp02_metadata":
        from scripts.evaluation.g_workflow_closure import validate_exp02_metadata

        payload = validate_exp02_metadata()
        print(f"[OK] exp02 metadata validated: {payload['adapter_name']}")
    elif args.task == "e6_retrieval":
        from scripts.evaluation.evaluate_e6_e8_retrieval import run_retrieval_eval

        run_dir = run_retrieval_eval("E6", output_root=output_root, limit_queries=args.limit_queries)
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e7_reranker":
        from scripts.evaluation.evaluate_e6_e8_retrieval import run_retrieval_eval

        run_dir = run_retrieval_eval("E7", output_root=output_root, limit_queries=args.limit_queries)
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e8_fallback":
        from scripts.evaluation.evaluate_e6_e8_retrieval import run_retrieval_eval

        run_dir = run_retrieval_eval("E8", output_root=output_root, limit_queries=args.limit_queries)
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e3_preference":
        from scripts.evaluation.evaluate_e3_e5_behavior import run_e3_preference_eval

        run_dir = run_e3_preference_eval(
            output_root=output_root,
            limit_queries=args.limit_queries,
            query_id_file=args.query_id_file,
        )
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e4_clarification":
        from scripts.evaluation.evaluate_e3_e5_behavior import run_e4_clarification_eval

        run_dir = run_e4_clarification_eval(
            output_root=output_root,
            limit_queries=args.limit_queries,
            query_id_file=args.query_id_file,
        )
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e5_query_bridge":
        from scripts.evaluation.evaluate_e3_e5_behavior import run_e5_query_bridge_eval

        run_dir = run_e5_query_bridge_eval(output_root=output_root, limit_queries=args.limit_queries)
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e9_freeze_assets":
        from scripts.evaluation.evaluate_e9_e10_generation import freeze_e9_assets

        units_path, query_ids_path = freeze_e9_assets(limit_queries=args.limit_queries)
        print(f"[OK] eval units written to {units_path}")
        print(f"[OK] query ids written to {query_ids_path}")
    elif args.task == "e9_generation_constraints":
        from scripts.evaluation.evaluate_e9_e10_generation import run_e9_generation_constraints

        run_dir = run_e9_generation_constraints(output_root=output_root, limit_queries=args.limit_queries)
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e10_prepare_manifests":
        from scripts.evaluation.evaluate_e9_e10_generation import prepare_e10_manifests

        train_path, dev_path = prepare_e10_manifests()
        print(f"[OK] train manifest written to {train_path}")
        print(f"[OK] dev manifest written to {dev_path}")
    elif args.task == "e10_prepare_manifests_v2":
        from scripts.evaluation.evaluate_e9_e10_generation import prepare_e10_manifests_v2

        train_path, dev_path, report_path = prepare_e10_manifests_v2()
        print(f"[OK] train manifest written to {train_path}")
        print(f"[OK] dev manifest written to {dev_path}")
        print(f"[OK] manifest report written to {report_path}")
    elif args.task == "e10_prepare_manifests_v3":
        from scripts.evaluation.evaluate_e9_e10_generation import prepare_e10_manifests_v3

        train_path, dev_path, report_path = prepare_e10_manifests_v3()
        print(f"[OK] train manifest written to {train_path}")
        print(f"[OK] dev manifest written to {dev_path}")
        print(f"[OK] manifest report written to {report_path}")
    elif args.task == "e10_prepare_seed_specs_v4":
        from scripts.evaluation.evaluate_e9_e10_generation import prepare_e10_seed_specs_v4

        seed_path, gold_path, deepseek_path, review_log_path, accepted_path, prompt_path = prepare_e10_seed_specs_v4()
        print(f"[OK] seed specs written to {seed_path}")
        print(f"[OK] gold patch bootstrap written to {gold_path}")
        print(f"[OK] deepseek drafts bootstrap written to {deepseek_path}")
        print(f"[OK] review log bootstrap written to {review_log_path}")
        print(f"[OK] accepted grounded bootstrap written to {accepted_path}")
        print(f"[OK] DeepSeek prompt templates written to {prompt_path}")
    elif args.task == "e10_migrate_deepseek_assets_v4":
        from scripts.evaluation.evaluate_e9_e10_generation import migrate_e10_deepseek_assets_v4

        result = migrate_e10_deepseek_assets_v4()
        print("[OK] E10 v4 DeepSeek assets migrated")
        print(f"[OK] moved_legacy_paths={result['moved_legacy_paths']}")
        print(f"[OK] updated_paths={result['updated_paths']}")
    elif args.task == "e10_prepare_deepseek_query_requests_v4":
        from scripts.evaluation.evaluate_e9_e10_generation import prepare_e10_deepseek_query_requests_v4

        request_path = prepare_e10_deepseek_query_requests_v4()
        print(f"[OK] DeepSeek query request package written to {request_path}")
    elif args.task == "e10_prepare_deepseek_target_requests_v4":
        from scripts.evaluation.evaluate_e9_e10_generation import prepare_e10_deepseek_target_requests_v4

        request_path = prepare_e10_deepseek_target_requests_v4()
        print(f"[OK] DeepSeek target request package written to {request_path}")
    elif args.task == "e10_prepare_manifests_v4":
        from scripts.evaluation.evaluate_e9_e10_generation import prepare_e10_manifests_v4

        train_path, dev_path, report_path = prepare_e10_manifests_v4()
        print(f"[OK] train manifest written to {train_path}")
        print(f"[OK] dev manifest written to {dev_path}")
        print(f"[OK] manifest report written to {report_path}")
    elif args.task == "e10_validate_manifest_v3":
        from scripts.evaluation.evaluate_e9_e10_generation import validate_e10_manifest_report_v3

        report = validate_e10_manifest_report_v3()
        print("[OK] E10 v3 manifest assets validated")
        print(f"[OK] grounded share={report['train_grounded_share_of_final_manifest']}")
        print(f"[OK] slice share={report['train_grounded_slice_share']}")
    elif args.task == "e10_validate_manifest_v4":
        from scripts.evaluation.evaluate_e9_e10_generation import validate_e10_manifest_report_v4

        report = validate_e10_manifest_report_v4()
        print("[OK] E10 v4 manifest assets validated")
        print(f"[OK] dataset_profile={report['dataset_profile']}")
        print(f"[OK] accepted_count={report['accepted_count']}")
        print(f"[OK] source_mode_distribution={report['source_mode_distribution']}")
    elif args.task == "e10_base_vs_peft":
        from scripts.evaluation.evaluate_e9_e10_generation import run_e10_base_vs_peft

        run_dir = run_e10_base_vs_peft(
            output_root=output_root,
            limit_queries=args.limit_queries,
            group_ids=args.group_id,
        )
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e10_compare_runs":
        from scripts.evaluation.evaluate_e9_e10_generation import run_e10_compare_runs

        if not args.base_run_dir or not args.peft_run_dir:
            raise ValueError("e10_compare_runs 需要同时提供 --base-run-dir 和 --peft-run-dir。")
        run_dir = run_e10_compare_runs(
            output_root=output_root,
            base_run_dir=args.base_run_dir,
            peft_run_dir=args.peft_run_dir,
        )
        print(f"[OK] run saved to {run_dir}")


if __name__ == "__main__":
    main()

"""Minimal batch runner entrypoint for frozen experiment assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.evaluation.evaluate_e3_e5_behavior import (
    run_e3_preference_eval,
    run_e4_clarification_eval,
    run_e5_query_bridge_eval,
)
from scripts.evaluation.evaluate_e9_e10_generation import (
    freeze_e9_assets,
    migrate_e10_deepseek_assets_v4,
    prepare_e10_deepseek_query_requests_v4,
    prepare_e10_deepseek_target_requests_v4,
    prepare_e10_manifests,
    prepare_e10_manifests_v4,
    prepare_e10_seed_specs_v4,
    prepare_e10_manifests_v2,
    prepare_e10_manifests_v3,
    validate_e10_manifest_report_v4,
    validate_e10_manifest_report_v3,
    run_e10_compare_runs,
    run_e10_base_vs_peft,
    run_e9_generation_constraints,
)
from scripts.evaluation.evaluate_e2_candidate_selection import run_e2
from scripts.evaluation.evaluate_e6_e8_retrieval import (
    build_e6_qrels_pool,
    freeze_e6_qrels,
    run_retrieval_eval,
)
from scripts.shared.experiment_utils import EXPERIMENT_RUNS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=[
            "e2_candidates",
            "e6_qrels_pool",
            "e6_freeze_qrels",
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
    parser.add_argument("--include-ablation", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.task == "e2_candidates":
        run_dir = run_e2(
            output_root=output_root,
            limit_queries=args.limit_queries,
            include_ablation=args.include_ablation,
        )
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e6_qrels_pool":
        pool_path = build_e6_qrels_pool(limit_queries=args.limit_queries)
        print(f"[OK] qrels pool written to {pool_path}")
    elif args.task == "e6_freeze_qrels":
        qrels_path = freeze_e6_qrels()
        print(f"[OK] qrels frozen to {qrels_path}")
    elif args.task == "e6_retrieval":
        run_dir = run_retrieval_eval("E6", output_root=output_root, limit_queries=args.limit_queries)
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e7_reranker":
        run_dir = run_retrieval_eval("E7", output_root=output_root, limit_queries=args.limit_queries)
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e8_fallback":
        run_dir = run_retrieval_eval("E8", output_root=output_root, limit_queries=args.limit_queries)
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e3_preference":
        run_dir = run_e3_preference_eval(
            output_root=output_root,
            limit_queries=args.limit_queries,
            query_id_file=args.query_id_file,
        )
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e4_clarification":
        run_dir = run_e4_clarification_eval(
            output_root=output_root,
            limit_queries=args.limit_queries,
            query_id_file=args.query_id_file,
        )
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e5_query_bridge":
        run_dir = run_e5_query_bridge_eval(output_root=output_root, limit_queries=args.limit_queries)
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e9_freeze_assets":
        units_path, query_ids_path = freeze_e9_assets(limit_queries=args.limit_queries)
        print(f"[OK] eval units written to {units_path}")
        print(f"[OK] query ids written to {query_ids_path}")
    elif args.task == "e9_generation_constraints":
        run_dir = run_e9_generation_constraints(output_root=output_root, limit_queries=args.limit_queries)
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e10_prepare_manifests":
        train_path, dev_path = prepare_e10_manifests()
        print(f"[OK] train manifest written to {train_path}")
        print(f"[OK] dev manifest written to {dev_path}")
    elif args.task == "e10_prepare_manifests_v2":
        train_path, dev_path, report_path = prepare_e10_manifests_v2()
        print(f"[OK] train manifest written to {train_path}")
        print(f"[OK] dev manifest written to {dev_path}")
        print(f"[OK] manifest report written to {report_path}")
    elif args.task == "e10_prepare_manifests_v3":
        train_path, dev_path, report_path = prepare_e10_manifests_v3()
        print(f"[OK] train manifest written to {train_path}")
        print(f"[OK] dev manifest written to {dev_path}")
        print(f"[OK] manifest report written to {report_path}")
    elif args.task == "e10_prepare_seed_specs_v4":
        seed_path, gold_path, deepseek_path, review_log_path, accepted_path, prompt_path = prepare_e10_seed_specs_v4()
        print(f"[OK] seed specs written to {seed_path}")
        print(f"[OK] gold patch bootstrap written to {gold_path}")
        print(f"[OK] deepseek drafts bootstrap written to {deepseek_path}")
        print(f"[OK] review log bootstrap written to {review_log_path}")
        print(f"[OK] accepted grounded bootstrap written to {accepted_path}")
        print(f"[OK] DeepSeek prompt templates written to {prompt_path}")
    elif args.task == "e10_migrate_deepseek_assets_v4":
        result = migrate_e10_deepseek_assets_v4()
        print("[OK] E10 v4 DeepSeek assets migrated")
        print(f"[OK] moved_legacy_paths={result['moved_legacy_paths']}")
        print(f"[OK] updated_paths={result['updated_paths']}")
    elif args.task == "e10_prepare_deepseek_query_requests_v4":
        request_path = prepare_e10_deepseek_query_requests_v4()
        print(f"[OK] DeepSeek query request package written to {request_path}")
    elif args.task == "e10_prepare_deepseek_target_requests_v4":
        request_path = prepare_e10_deepseek_target_requests_v4()
        print(f"[OK] DeepSeek target request package written to {request_path}")
    elif args.task == "e10_prepare_manifests_v4":
        train_path, dev_path, report_path = prepare_e10_manifests_v4()
        print(f"[OK] train manifest written to {train_path}")
        print(f"[OK] dev manifest written to {dev_path}")
        print(f"[OK] manifest report written to {report_path}")
    elif args.task == "e10_validate_manifest_v3":
        report = validate_e10_manifest_report_v3()
        print("[OK] E10 v3 manifest assets validated")
        print(f"[OK] grounded share={report['train_grounded_share_of_final_manifest']}")
        print(f"[OK] slice share={report['train_grounded_slice_share']}")
    elif args.task == "e10_validate_manifest_v4":
        report = validate_e10_manifest_report_v4()
        print("[OK] E10 v4 manifest assets validated")
        print(f"[OK] dataset_profile={report['dataset_profile']}")
        print(f"[OK] accepted_count={report['accepted_count']}")
        print(f"[OK] source_mode_distribution={report['source_mode_distribution']}")
    elif args.task == "e10_base_vs_peft":
        run_dir = run_e10_base_vs_peft(
            output_root=output_root,
            limit_queries=args.limit_queries,
            group_ids=args.group_id,
        )
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e10_compare_runs":
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

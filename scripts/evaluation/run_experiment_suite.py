"""Minimal batch runner entrypoint for frozen experiment assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.evaluation.evaluate_e3_e5_behavior import (
    run_e3_preference_eval,
    run_e4_clarification_eval,
    run_e5_query_bridge_eval,
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
        ],
        required=True,
    )
    parser.add_argument("--output-root", default=str(EXPERIMENT_RUNS_DIR))
    parser.add_argument("--limit-queries", type=int, default=None)
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
        run_dir = run_e3_preference_eval(output_root=output_root, limit_queries=args.limit_queries)
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e4_clarification":
        run_dir = run_e4_clarification_eval(output_root=output_root, limit_queries=args.limit_queries)
        print(f"[OK] run saved to {run_dir}")
    elif args.task == "e5_query_bridge":
        run_dir = run_e5_query_bridge_eval(output_root=output_root, limit_queries=args.limit_queries)
        print(f"[OK] run saved to {run_dir}")


if __name__ == "__main__":
    main()

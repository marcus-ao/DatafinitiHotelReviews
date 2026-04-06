from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.evaluation import evaluate_e3_e5_behavior as behavior_mod
from scripts.shared.experiment_schemas import BehaviorRuntimeConfig, ClarificationDecision


class E3E5BehaviorAuditContractsTestCase(unittest.TestCase):
    def test_select_audited_behavior_query_rows_filters_non_behavior_query_types(self) -> None:
        judged_queries = [
            {"query_id": "q001", "query_type": "single_aspect"},
            {"query_id": "q002", "query_type": "unsupported_budget"},
            {"query_id": "q003", "query_type": "conflict"},
            {"query_id": "q004", "query_type": "missing_city"},
        ]
        rows, scope, query_ids = behavior_mod.select_audited_behavior_query_rows(judged_queries)
        self.assertEqual(scope, "audited_behavior_queries")
        self.assertEqual([row["query_id"] for row in rows], ["q001", "q002"])
        self.assertEqual(query_ids, ["q001", "q002"])

    def test_e5_zh_direct_group_uses_same_target_structure_with_language_bridge(self) -> None:
        unit = {
            "query_id": "q001",
            "city": "Anaheim",
            "target_aspect": "service",
            "target_role": "focus",
            "query_text_zh": "帮我找服务好的酒店",
            "query_en_target": "hotel in Anaheim with helpful and reliable service",
        }
        zh_target = behavior_mod.build_query_target_zh(unit["city"], unit["target_aspect"], unit["target_role"])
        en_target = behavior_mod.build_query_en_target(unit["city"], unit["target_aspect"], unit["target_role"])
        self.assertNotEqual(zh_target, unit["query_text_zh"])
        self.assertEqual(en_target, unit["query_en_target"])
        self.assertIn("service", en_target)
        self.assertIn("服务", zh_target)

    def test_e5_e8_core_query_ids_loader_enforces_core40_contract(self) -> None:
        payload = {
            "query_ids": [f"q{i:03d}" for i in range(1, 41)],
            "query_type_order": ["single_aspect", "multi_aspect", "focus_and_avoid", "multi_aspect_strong"],
            "query_type_counts": {
                "single_aspect": 10,
                "multi_aspect": 10,
                "focus_and_avoid": 10,
                "multi_aspect_strong": 10,
            },
        }
        with mock.patch.object(behavior_mod, "load_e5_e8_core_query_ids", return_value=payload["query_ids"]):
            self.assertEqual(len(behavior_mod.load_e5_e8_core_query_ids()), 40)

    def test_run_e4_clarification_eval_persists_latency_for_summary(self) -> None:
        judged_queries = [{"query_id": "q001", "query_type": "single_aspect", "query_text_zh": "我想找位置好的酒店"}]
        clarify_gold = [{"query_id": "q001", "clarify_needed": True}]
        slot_gold = [{"query_id": "q001", "city": "Anaheim", "state": "CA"}]

        def fake_load_json(path: str | Path) -> dict:
            path = Path(path)
            if path.name == "frozen_config.yaml":
                return {
                    "behavior": {"prompt_versions": {"e4_clarification": "e4_v2_cn_decision_label_fewshot"}},
                    "workflow": {"default_retrieval_mode": "aspect_main_no_rerank", "enable_fallback": False},
                }
            if path.name == "frozen_split_manifest.json":
                return {"meta": {"config_hash": "split_cfg"}}
            raise AssertionError(f"unexpected json path: {path}")

        def fake_load_jsonl(path: str | Path) -> list[dict]:
            path = Path(path)
            if path.name == "judged_queries.jsonl":
                return judged_queries
            if path.name == "clarify_gold.jsonl":
                return clarify_gold
            if path.name == "slot_gold.jsonl":
                return slot_gold
            raise AssertionError(f"unexpected jsonl path: {path}")

        runtime_config = BehaviorRuntimeConfig(
            llm_backend="local",
            model_id="Qwen/Qwen3.5-4B",
            enable_thinking=False,
            temperature=0.0,
            max_new_tokens=256,
            api_timeout_seconds=120,
        )
        decision = ClarificationDecision(
            query_id="",
            group_id="A_rule_clarify",
            clarify_needed=True,
            clarify_reason="needs clarification",
            target_slots=["city"],
            question="请问你想去哪个城市？",
            schema_valid=True,
            raw_response="",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "runs"
            labels_dir = Path(tmp_dir) / "labels"
            with (
                mock.patch.object(behavior_mod, "E4_GROUPS", ["A_rule_clarify"]),
                mock.patch.object(behavior_mod, "E4_LABELS_DIR", labels_dir),
                mock.patch.object(behavior_mod, "load_config", return_value={"behavior": {}}),
                mock.patch.object(behavior_mod, "load_json", side_effect=fake_load_json),
                mock.patch.object(behavior_mod, "load_jsonl", side_effect=fake_load_jsonl),
                mock.patch.object(
                    behavior_mod,
                    "resolve_behavior_runtime_config",
                    return_value=(runtime_config, None),
                ),
                mock.patch.object(
                    behavior_mod,
                    "select_audited_behavior_query_rows",
                    return_value=(judged_queries, "audited_behavior_queries", ["q001"]),
                ),
                mock.patch.object(behavior_mod, "build_balanced_e4_subset", return_value=["q001"]),
                mock.patch.object(behavior_mod, "build_rule_clarification", return_value=decision),
            ):
                run_dir = behavior_mod.run_e4_clarification_eval(output_root=output_root)

            summary_path = run_dir / "summary.csv"
            self.assertTrue(summary_path.exists())
            with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertIn("avg_latency_ms", rows[0])
            self.assertNotEqual(rows[0]["avg_latency_ms"], "")
            audit_path = labels_dir / "clarification_question_audit.csv"
            self.assertTrue(audit_path.exists())


if __name__ == "__main__":
    unittest.main()

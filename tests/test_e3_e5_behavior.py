from __future__ import annotations

import unittest
from unittest import mock

from scripts.evaluation import evaluate_e3_e5_behavior as behavior_mod


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


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.evaluation import blind_review_export as blind_mod


class _FakeResponsesAPI:
    def __init__(self, payloads):
        self.payloads = list(payloads)

    def create(self, model, input):
        if not self.payloads:
            raise AssertionError("No fake payload left")
        return {"output_text": self.payloads.pop(0)}


class _FakeClient:
    def __init__(self, payloads):
        self.responses = _FakeResponsesAPI(payloads)


class BlindReviewExportTestCase(unittest.TestCase):
    def test_build_blind_review_rows_is_reproducible_and_hides_group_labels(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_dirs = {}
            for group_id in ["G1", "G2"]:
                run_dir = tmp_path / group_id
                run_dir.mkdir()
                run_dirs[group_id] = run_dir
                rows = []
                for query_id in ["q001", "q002"]:
                    rows.append(
                        {
                            "query_id": query_id,
                            "group_id": group_id,
                            "intermediate_objects": {
                                "eval_unit": {"query_id": query_id, "query_text_zh": f"{query_id} 的查询"},
                                "response": {
                                    "query_id": query_id,
                                    "group_id": group_id,
                                    "summary": f"{query_id} 的推荐摘要",
                                    "recommendations": [],
                                    "unsupported_notice": "",
                                },
                            },
                        }
                    )
                (run_dir / "results.jsonl").write_text(
                    "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                    encoding="utf-8",
                )

            rows_a = blind_mod.build_blind_review_rows(run_dirs, sample_size=2, seed=7)
            rows_b = blind_mod.build_blind_review_rows(run_dirs, sample_size=2, seed=7)

        self.assertEqual(rows_a, rows_b)
        serialized = json.dumps(rows_a, ensure_ascii=False)
        self.assertNotIn("G1", serialized)
        self.assertNotIn("G2", serialized)
        self.assertNotIn("query_id", serialized)
        self.assertNotIn("group_id", serialized)
        self.assertTrue(all("blind_label" in row for row in rows_a))
        self.assertTrue(all("query_bundle_id" in row for row in rows_a))
        self.assertTrue(all("response_text" in row for row in rows_a))
        self.assertTrue(all("unsupported_notice" in row for row in rows_a))
        self.assertTrue(all(row["response_text"] for row in rows_a))
        label_by_group = {
            str(row["review_item_id"]).split("_")[-1]: row["response_summary"]
            for row in rows_a
            if row["query_bundle_id"] == "bundle_001"
        }
        self.assertEqual(sorted(label_by_group), ["A", "B"])

    def test_build_blind_review_rows_rejects_non_positive_sample_size(self):
        with self.assertRaises(ValueError):
            blind_mod.build_blind_review_rows({"G1": "unused"}, sample_size=0, seed=7)

    def test_build_blind_review_rows_rejects_duplicate_query_ids(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "G1"
            run_dir.mkdir()
            rows = [
                {
                    "query_id": "q001",
                    "group_id": "G1",
                    "intermediate_objects": {
                        "eval_unit": {"query_id": "q001", "query_text_zh": "q001 的查询"},
                        "response": {"summary": "A", "recommendations": [], "unsupported_notice": ""},
                    },
                },
                {
                    "query_id": "q001",
                    "group_id": "G1",
                    "intermediate_objects": {
                        "eval_unit": {"query_id": "q001", "query_text_zh": "q001 的查询"},
                        "response": {"summary": "B", "recommendations": [], "unsupported_notice": ""},
                    },
                },
            ]
            (run_dir / "results.jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                blind_mod.build_blind_review_rows({"G1": run_dir}, sample_size=1, seed=7)

    def test_build_blind_review_rows_rejects_missing_query_text(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "G1"
            run_dir.mkdir()
            row = {
                "query_id": "q001",
                "group_id": "G1",
                "intermediate_objects": {
                    "eval_unit": {"query_id": "q001", "query_text_zh": ""},
                    "response": {"summary": "A", "recommendations": [], "unsupported_notice": ""},
                },
            }
            (run_dir / "results.jsonl").write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                blind_mod.build_blind_review_rows({"G1": run_dir}, sample_size=1, seed=7)

    def test_build_blind_review_worksheet_rows_contains_score_and_pairwise_fields(self):
        blind_rows = [
            {
                "review_item_id": "blind_001_A",
                "query_bundle_id": "bundle_001",
                "blind_label": "A",
                "query_text_zh": "q001 的查询",
                "response_text": "Summary: 推荐酒店",
                "response_summary": "推荐酒店",
                "unsupported_notice": "",
                "response_json": "{}",
                "recommendation_count": 1,
            },
            {
                "review_item_id": "blind_001_B",
                "query_bundle_id": "bundle_001",
                "blind_label": "B",
                "query_text_zh": "q001 的查询",
                "response_text": "Summary: 另一条推荐",
                "response_summary": "另一条推荐",
                "unsupported_notice": "",
                "response_json": "{}",
                "recommendation_count": 1,
            },
        ]
        worksheet_rows = blind_mod.build_blind_review_worksheet_rows(blind_rows)
        self.assertEqual(len(worksheet_rows), 3)
        self.assertEqual(worksheet_rows[0]["overall_quality_score"], "")
        self.assertEqual(worksheet_rows[0]["evidence_credibility_score"], "")
        self.assertEqual(worksheet_rows[0]["practical_value_score"], "")
        self.assertEqual(worksheet_rows[0]["reviewer_notes"], "")
        self.assertEqual(worksheet_rows[-1]["query_bundle_id"], "bundle_001")
        self.assertEqual(worksheet_rows[-1]["pairwise_preference"], "")
        self.assertEqual(worksheet_rows[-1]["available_blind_labels"], "A,B")

    def test_export_blind_review_pack_writes_hidden_mapping_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_dirs = {}
            for group_id in ["G1", "G2"]:
                run_dir = tmp_path / group_id
                run_dir.mkdir()
                run_dirs[group_id] = run_dir
                rows = [
                    {
                        "query_id": "q001",
                        "group_id": group_id,
                        "intermediate_objects": {
                            "eval_unit": {"query_id": "q001", "query_text_zh": "q001 的查询"},
                            "response": {
                                "query_id": "q001",
                                "group_id": group_id,
                                "summary": f"{group_id} 推荐摘要",
                                "recommendations": [],
                                "unsupported_notice": "",
                            },
                        },
                    }
                ]
                (run_dir / "results.jsonl").write_text(
                    "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                    encoding="utf-8",
                )

            output_path = tmp_path / "blind_review_pack.csv"
            blind_rows = blind_mod.export_blind_review_pack(run_dirs, output_path, sample_size=1, seed=7)
            mapping_path = blind_mod.blind_review_mapping_output_path(output_path)
            self.assertEqual(len(blind_rows), 2)
            self.assertTrue(mapping_path.exists())
            mapping_text = mapping_path.read_text(encoding="utf-8")
            self.assertIn("source_group_id", mapping_text)
            self.assertIn("G1", mapping_text)
            self.assertIn("G2", mapping_text)

    def test_fill_blind_review_worksheet_with_llm_scores_items_and_pairwise_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            worksheet_path = tmp_path / "blind_review_worksheet.csv"
            pd.DataFrame(
                [
                    {
                        "review_item_id": "blind_001_A",
                        "query_bundle_id": "bundle_001",
                        "blind_label": "A",
                        "query_text_zh": "请推荐安静的酒店",
                        "response_text": "Summary: 推荐酒店A",
                        "overall_quality_score": "",
                        "evidence_credibility_score": "",
                        "practical_value_score": "",
                        "reviewer_notes": "",
                        "available_blind_labels": "",
                        "pairwise_preference": "",
                        "pairwise_notes": "",
                    },
                    {
                        "review_item_id": "blind_001_B",
                        "query_bundle_id": "bundle_001",
                        "blind_label": "B",
                        "query_text_zh": "请推荐安静的酒店",
                        "response_text": "Summary: 推荐酒店B",
                        "overall_quality_score": "",
                        "evidence_credibility_score": "",
                        "practical_value_score": "",
                        "reviewer_notes": "",
                        "available_blind_labels": "",
                        "pairwise_preference": "",
                        "pairwise_notes": "",
                    },
                    {
                        "review_item_id": None,
                        "query_bundle_id": "bundle_001",
                        "blind_label": None,
                        "query_text_zh": None,
                        "response_text": None,
                        "overall_quality_score": "",
                        "evidence_credibility_score": "",
                        "practical_value_score": "",
                        "reviewer_notes": "",
                        "available_blind_labels": "A,B",
                        "pairwise_preference": "",
                        "pairwise_notes": "",
                    },
                ]
            ).to_csv(worksheet_path, index=False, encoding="utf-8-sig")

            client = _FakeClient(
                [
                    json.dumps(
                        {
                            "overall_quality_score": 4.6,
                            "evidence_credibility_score": 4.7,
                            "practical_value_score": 4.5,
                            "reviewer_notes": "Strong and useful.",
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "overall_quality_score": 4.2,
                            "evidence_credibility_score": 4.1,
                            "practical_value_score": 4.0,
                            "reviewer_notes": "Good but slightly weaker.",
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "pairwise_preference": "A>B",
                            "pairwise_notes": "A is slightly more complete.",
                        },
                        ensure_ascii=False,
                    ),
                ]
            )

            result = blind_mod.fill_blind_review_worksheet_with_llm(
                worksheet_path,
                client=client,
                model="deepseek-reasoner",
            )

            filled_df = pd.read_csv(result["output_path"])
            self.assertEqual(result["item_row_count"], 2)
            self.assertEqual(result["pairwise_row_count"], 1)
            self.assertAlmostEqual(float(filled_df.loc[0, "overall_quality_score"]), 4.6, places=2)
            self.assertEqual(str(filled_df.loc[0, "reviewer_notes"]), "Strong and useful.")
            self.assertEqual(str(filled_df.loc[2, "pairwise_preference"]), "A>B")
            self.assertTrue(Path(result["log_path"]).exists())


if __name__ == "__main__":
    unittest.main()

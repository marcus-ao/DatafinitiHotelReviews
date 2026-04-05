from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from scripts.evaluation import g_workflow_closure as closure_mod


class GWorkflowClosureTestCase(unittest.TestCase):
    def test_validate_exp02_metadata_rejects_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            missing_path = Path(tempdir) / "missing_exp02.json"
            with self.assertRaises(FileNotFoundError):
                closure_mod.validate_exp02_metadata(missing_path)

    def test_ensure_exp02_metadata_placeholder_writes_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            output_path = Path(tempdir) / "exp02_placeholder.json"
            created_path = closure_mod.ensure_exp02_metadata_placeholder(output_path)
            payload = json.loads(created_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["adapter_name"], "exp02")

    def test_build_blind_review_aggregation_outputs_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            input_path = Path(tempdir) / "worksheet.csv"
            pd.DataFrame(
                [
                    {
                        "review_item_id": "blind_001_A",
                        "query_bundle_id": "bundle_001",
                        "blind_label": "A",
                        "overall_quality_score": 4,
                        "evidence_credibility_score": 5,
                        "practical_value_score": 4,
                        "pairwise_preference": None,
                    },
                    {
                        "review_item_id": "blind_001_B",
                        "query_bundle_id": "bundle_001",
                        "blind_label": "B",
                        "overall_quality_score": 3,
                        "evidence_credibility_score": 4,
                        "practical_value_score": 3,
                        "pairwise_preference": None,
                    },
                    {
                        "review_item_id": None,
                        "query_bundle_id": "bundle_001",
                        "blind_label": None,
                        "overall_quality_score": None,
                        "evidence_credibility_score": None,
                        "practical_value_score": None,
                        "pairwise_preference": "A>B",
                    },
                ]
            ).to_csv(input_path, index=False, encoding="utf-8-sig")
            mapping_path = Path(tempdir) / "blind_review_mapping.csv"
            pd.DataFrame(
                [
                    {
                        "review_item_id": "blind_001_A",
                        "query_bundle_id": "bundle_001",
                        "blind_label": "A",
                        "source_group_id": "G2",
                        "source_query_id": "q001",
                    },
                    {
                        "review_item_id": "blind_001_B",
                        "query_bundle_id": "bundle_001",
                        "blind_label": "B",
                        "source_group_id": "G1",
                        "source_query_id": "q001",
                    },
                ]
            ).to_csv(mapping_path, index=False, encoding="utf-8-sig")
            result = closure_mod.aggregate_blind_review_results(input_path)
            self.assertEqual(result["item_summary"][0]["source_group_id"], "G1")
            self.assertEqual(result["pairwise_summary"][0]["preference_label"], "G2>G1")

    def test_build_g_chapter_report_writes_retrieval_and_generation_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            run_dirs = {}
            for group_id in ["G1", "G2", "G3", "G4"]:
                run_dir = temp_root / group_id.lower()
                run_dir.mkdir()
                pd.DataFrame(
                    [
                        {
                            "schema_valid_rate": 1.0,
                            "citation_precision": 0.9,
                            "evidence_verifiability_mean": 1.5,
                            "recommendation_coverage": 0.8,
                            "aspect_alignment_rate": 0.7,
                            "hallucination_rate": 0.1,
                            "unsupported_honesty_rate": 1.0,
                        }
                    ]
                ).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")
                (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
                run_dirs[group_id] = run_dir

            fake_units = [
                {
                    "query_id": "q001",
                    "query_text_zh": "query",
                    "query_type": "single_aspect",
                    "user_preference_gold": {
                        "city": "X",
                        "state": None,
                        "hotel_category": None,
                        "focus_aspects": ["service"],
                        "avoid_aspects": [],
                        "unsupported_requests": [],
                        "query_en": "query en",
                    },
                    "unsupported_requests": [],
                    "candidate_hotels": [{"hotel_id": "h1", "hotel_name": "Hotel 1", "score_total": 1.0, "score_breakdown": {}}],
                    "evidence_packs": [{
                        "hotel_id": "h1",
                        "query_en": "query en",
                        "evidence_by_aspect": {"service": [{"sentence_id": "s1", "sentence_text": "good service", "aspect": "service", "sentiment": "positive", "review_date": None, "score_dense": 0.1, "score_rerank": 0.2}]},
                        "all_sentence_ids": ["s1"],
                        "retrieval_trace": {"latency_ms": 12.5},
                    }],
                    "retrieval_mode": "plain_city_test_rerank",
                    "candidate_policy": "G_plain_retrieval_top5",
                    "config_hash": "cfg",
                }
            ]
            plain_asset = temp_root / "plain.jsonl"
            aspect_asset = temp_root / "aspect.jsonl"
            plain_asset.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in fake_units) + "\n", encoding="utf-8")
            aspect_asset.write_text(
                "\n".join(
                    json.dumps({**row, "retrieval_mode": "aspect_main_no_rerank", "candidate_policy": "G_aspect_retrieval_top5"}, ensure_ascii=False)
                    for row in fake_units
                ) + "\n",
                encoding="utf-8",
            )

            output_dir = temp_root / "report"
            with mock.patch.object(closure_mod, "G_GROUP_SPECS", {
                "G1": {"eval_units_path": plain_asset, "retrieval_variant": "plain", "requires_peft": False},
                "G2": {"eval_units_path": aspect_asset, "retrieval_variant": "aspect", "requires_peft": False},
                "G3": {"eval_units_path": plain_asset, "retrieval_variant": "plain", "requires_peft": True},
                "G4": {"eval_units_path": aspect_asset, "retrieval_variant": "aspect", "requires_peft": True},
            }):
                result = closure_mod.build_g_chapter_report(run_dirs, output_dir=output_dir)

            self.assertEqual(result["output_dir"], str(output_dir))
            retrieval_df = pd.read_csv(output_dir / "g_retrieval_summary.csv")
            generation_df = pd.read_csv(output_dir / "g_generation_summary.csv")
            self.assertIn("aspect_recall_at_5", retrieval_df.columns)
            self.assertIn("ndcg_at_5", retrieval_df.columns)
            self.assertIn("hallucination_rate", generation_df.columns)

    def test_export_and_validate_g_closure_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            run_dirs = {}
            for group_id in ["G1", "G2", "G3", "G4"]:
                run_dir = temp_root / group_id.lower()
                run_dir.mkdir()
                (run_dir / "summary.csv").write_text("group_id\n" + group_id + "\n", encoding="utf-8")
                (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
                run_dirs[group_id] = run_dir
            manifest_path = temp_root / "g_manifest.json"
            closure_mod.export_g_closure_manifest(run_dirs, manifest_path)
            validated = closure_mod.validate_g_closure_manifest(manifest_path)
            self.assertEqual(sorted(validated["group_run_dirs"]), ["G1", "G2", "G3", "G4"])

    def test_validate_g_closure_manifest_rejects_incomplete_blind_review_summary_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            run_dirs = {}
            for group_id in ["G1", "G2", "G3", "G4"]:
                run_dir = temp_root / group_id.lower()
                run_dir.mkdir()
                (run_dir / "summary.csv").write_text(
                    "schema_valid_rate,citation_precision,recommendation_coverage,aspect_alignment_rate,hallucination_rate\n1,1,1,1,0\n",
                    encoding="utf-8",
                )
                (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
                run_dirs[group_id] = str(run_dir)
            blind_dir = temp_root / "blind_summary"
            blind_dir.mkdir()
            (blind_dir / "blind_review_item_summary.csv").write_text("blind_label,review_count\nA,1\n", encoding="utf-8")
            manifest = closure_mod.build_g_closure_manifest(run_dirs, blind_review_summary_dir=blind_dir)
            with self.assertRaises(FileNotFoundError):
                closure_mod.validate_g_closure_manifest(manifest)

    def test_single_group_summary_row_rejects_missing_generation_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            run_dir = Path(tempdir)
            (run_dir / "summary.csv").write_text("group_id\nG1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                closure_mod._single_group_summary_row(run_dir)

    def test_build_g_execution_readiness_report_reports_missing_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            output_dir = Path(tempdir)
            readiness = closure_mod.export_g_execution_readiness_report(output_dir)
            self.assertIn("checks", readiness)
            self.assertTrue((output_dir / "g_execution_readiness.json").exists())
            self.assertTrue((output_dir / "g_execution_readiness.csv").exists())
            self.assertTrue((output_dir / "analysis.md").exists())

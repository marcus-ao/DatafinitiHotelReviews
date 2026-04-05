from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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
            result = closure_mod.aggregate_blind_review_results(input_path)
            self.assertEqual(result["item_summary"][0]["blind_label"], "A")
            self.assertEqual(result["pairwise_summary"][0]["preference_label"], "A>B")

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

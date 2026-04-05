from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.evaluation import run_experiment_suite


class RunExperimentSuiteDispatchTests(unittest.TestCase):
    def test_load_group_score_map_input_preserves_query_ids_in_wide_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            input_path = Path(tempdir) / "scores.csv"
            input_path.write_text(
                "\n".join(
                    [
                        "group_id,query_id,citation_precision,hallucination_rate",
                        "G1,q001,0.8,0.1",
                        "G1,q002,0.9,0.2",
                        "G2,q002,0.95,0.3",
                        "G2,q001,0.85,0.4",
                    ]
                ),
                encoding="utf-8",
            )
            payload = run_experiment_suite._load_group_score_map_input(input_path)
        self.assertEqual(payload["G1"]["citation_precision"]["query_ids"], ["q001", "q002"])
        self.assertEqual(payload["G2"]["citation_precision"]["query_ids"], ["q002", "q001"])
        self.assertEqual(payload["G1"]["hallucination_rate"]["scores"], [0.1, 0.2])

    def test_load_group_score_map_input_preserves_query_ids_in_structured_json(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            input_path = Path(tempdir) / "scores.json"
            input_path.write_text(
                json.dumps(
                    {
                        "G1": {
                            "citation_precision": {
                                "scores": [0.8, 0.9],
                                "query_ids": ["q001", "q002"],
                            }
                        },
                        "G2": {
                            "citation_precision": {
                                "scores": [0.85, 0.95],
                                "query_ids": ["q001", "q002"],
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            payload = run_experiment_suite._load_group_score_map_input(input_path)
        self.assertEqual(payload["G1"]["citation_precision"]["scores"], [0.8, 0.9])
        self.assertEqual(payload["G2"]["citation_precision"]["query_ids"], ["q001", "q002"])

    def _run_main(self, *args: str) -> None:
        with patch.object(sys, "argv", ["run_experiment_suite.py", *args]):
            run_experiment_suite.main()

    def test_g_run_generation_dispatches_to_generation_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            output_root = Path(tempdir)
            expected_run_dir = output_root / "ggen_fake"
            with patch(
                "scripts.evaluation.evaluate_e9_e10_generation.run_g_generation",
                return_value=expected_run_dir,
            ) as mocked_run:
                self._run_main(
                    "--task",
                    "g_run_generation",
                    "--output-root",
                    str(output_root),
                    "--group-id",
                    "G1",
                    "--limit-queries",
                    "2",
                )
            mocked_run.assert_called_once_with(
                output_root=output_root,
                group_ids=["G1"],
                limit_queries=2,
            )

    def test_g_compare_runs_dispatches_to_compare_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            output_root = Path(tempdir)
            expected_run_dir = output_root / "gcmp_fake"
            with patch(
                "scripts.evaluation.evaluate_e9_e10_generation.run_g_compare_runs",
                return_value=expected_run_dir,
            ) as mocked_run:
                self._run_main(
                    "--task",
                    "g_compare_runs",
                    "--output-root",
                    str(output_root),
                    "--left-run-dir",
                    "experiments/runs/ggen_left",
                    "--right-run-dir",
                    "experiments/runs/ggen_right",
                    "--left-label",
                    "G1",
                    "--right-label",
                    "G2",
                )
            mocked_run.assert_called_once_with(
                output_root=output_root,
                left_run_dir="experiments/runs/ggen_left",
                right_run_dir="experiments/runs/ggen_right",
                left_label="G1",
                right_label="G2",
            )

    def test_g_compute_pairwise_tests_writes_output_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            output_root = temp_root / "runs"
            input_path = temp_root / "group_scores.json"
            input_path.write_text(
                json.dumps(
                    {
                        "G1": {"citation_precision": [0.9, 1.0]},
                        "G2": {"citation_precision": [0.8, 0.95]},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            result_df = pd.DataFrame(
                [
                    {
                        "group_a": "G1",
                        "group_b": "G2",
                        "metric": "citation_precision",
                        "n": 2,
                        "mean_a": 0.95,
                        "mean_b": 0.875,
                        "mean_delta": -0.075,
                        "median_delta": -0.075,
                        "p_value": 0.5,
                        "effect_size": -0.6,
                        "ci_low": -0.2,
                        "ci_high": 0.02,
                    }
                ]
            )
            with patch(
                "scripts.evaluation.statistical_tests.compute_pairwise_tests",
                return_value=result_df,
            ) as mocked_compute:
                self._run_main(
                    "--task",
                    "g_compute_pairwise_tests",
                    "--output-root",
                    str(output_root),
                    "--input-path",
                    str(input_path),
                )

            mocked_compute.assert_called_once()
            self.assertEqual(mocked_compute.call_args.kwargs["metrics"], ["citation_precision"])
            run_dirs = list(output_root.glob("gstats_*"))
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue((run_dirs[0] / "pairwise_tests.csv").exists())
            run_meta = json.loads((run_dirs[0] / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(run_meta["task"], "G_PAIRWISE_TESTS")
            self.assertEqual(run_meta["metrics"], ["citation_precision"])

    def test_g_run_llm_judge_writes_score_and_summary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            output_root = temp_root / "runs"
            input_path = temp_root / "results.jsonl"
            input_path.write_text("", encoding="utf-8")
            score_df = pd.DataFrame(
                [
                    {
                        "query_id": "q001",
                        "group_id": "G1",
                        "relevance": 4.0,
                        "traceability": 4.2,
                        "fluency": 4.1,
                        "completeness": 3.9,
                        "honesty": 4.3,
                        "overall_mean": 4.1,
                    }
                ]
            )
            summary_df = pd.DataFrame(
                [
                    {
                        "group_id": "G1",
                        "judge_count": 1,
                        "relevance": 4.0,
                        "traceability": 4.2,
                        "fluency": 4.1,
                        "completeness": 3.9,
                        "honesty": 4.3,
                        "overall_mean": 4.1,
                    }
                ]
            )
            with patch(
                "scripts.evaluation.llm_judge.run_llm_judge",
                return_value=score_df,
            ) as mocked_run, patch(
                "scripts.evaluation.llm_judge.aggregate_judge_scores",
                return_value=summary_df,
            ) as mocked_aggregate:
                self._run_main(
                    "--task",
                    "g_run_llm_judge",
                    "--output-root",
                    str(output_root),
                    "--input-path",
                    str(input_path),
                    "--model",
                    "gpt-4o-mini",
                )

            mocked_run.assert_called_once()
            self.assertEqual(mocked_run.call_args.args[0], input_path)
            self.assertEqual(mocked_run.call_args.kwargs["model"], "gpt-4o-mini")
            mocked_aggregate.assert_called_once()
            run_dirs = list(output_root.glob("gjudge_*"))
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue((run_dirs[0] / "judge_scores.csv").exists())
            self.assertTrue((run_dirs[0] / "judge_summary.csv").exists())
            run_meta = json.loads((run_dirs[0] / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(run_meta["task"], "G_LLM_JUDGE")
            self.assertEqual(run_meta["model"], "gpt-4o-mini")

    def test_g_run_llm_judge_rejects_missing_input_path(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            output_root = Path(tempdir) / "runs"
            missing_path = Path(tempdir) / "missing_results.jsonl"
            with self.assertRaises(FileNotFoundError):
                self._run_main(
                    "--task",
                    "g_run_llm_judge",
                    "--output-root",
                    str(output_root),
                    "--input-path",
                    str(missing_path),
                )

    def test_g_compute_pairwise_tests_rejects_missing_input_path(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            output_root = Path(tempdir) / "runs"
            missing_path = Path(tempdir) / "missing_scores.json"
            with self.assertRaises(FileNotFoundError):
                self._run_main(
                    "--task",
                    "g_compute_pairwise_tests",
                    "--output-root",
                    str(output_root),
                    "--input-path",
                    str(missing_path),
                )

    def test_g_extract_stat_payload_writes_output_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            output_root = temp_root / "runs"
            run_dir_map_path = temp_root / "g_runs.json"
            run_dirs = {}
            for group_id in ["G1", "G2", "G3", "G4"]:
                run_dir = temp_root / group_id.lower()
                run_dir.mkdir()
                run_dirs[group_id] = str(run_dir)
            run_dir_map_path.write_text(json.dumps(run_dirs, ensure_ascii=False), encoding="utf-8")
            with patch(
                "scripts.evaluation.g_workflow_closure.extract_g_group_score_map",
                return_value={group_id: {"citation_precision": {"scores": [1.0], "query_ids": ["q001"]}} for group_id in ["G1", "G2", "G3", "G4"]},
            ) as mocked_extract:
                self._run_main(
                    "--task",
                    "g_extract_stat_payload",
                    "--output-root",
                    str(output_root),
                    "--input-path",
                    str(run_dir_map_path),
                )
            mocked_extract.assert_called_once()
            run_dirs = list(output_root.glob("gpayload_*"))
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue((run_dirs[0] / "group_score_map.json").exists())

    def test_g_run_batch_llm_judge_writes_output_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            output_root = temp_root / "runs"
            run_dir_map_path = temp_root / "g_runs.json"
            run_dirs = {}
            for group_id in ["G1", "G2", "G3", "G4"]:
                run_dir = temp_root / group_id.lower()
                run_dir.mkdir()
                run_dirs[group_id] = str(run_dir)
            run_dir_map_path.write_text(json.dumps(run_dirs, ensure_ascii=False), encoding="utf-8")
            with patch(
                "scripts.evaluation.g_workflow_closure.run_g_batch_llm_judge",
                return_value={"score_rows": [{"query_id": "q001"}], "summary_rows": [{"group_id": "G1"}]},
            ) as mocked_run:
                self._run_main(
                    "--task",
                    "g_run_batch_llm_judge",
                    "--output-root",
                    str(output_root),
                    "--input-path",
                    str(run_dir_map_path),
                )
            mocked_run.assert_called_once()
            run_dirs = list(output_root.glob("gjudgebatch_*"))
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue((run_dirs[0] / "run_meta.json").exists())

    def test_g_aggregate_blind_review_writes_output_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            output_root = temp_root / "runs"
            worksheet_path = temp_root / "worksheet.csv"
            worksheet_path.write_text("review_item_id\nblind_001_A\n", encoding="utf-8")
            with patch(
                "scripts.evaluation.g_workflow_closure.aggregate_blind_review_results",
                return_value={"item_summary": [{"blind_label": "A"}], "pairwise_summary": [{"preference_label": "A>B"}]},
            ) as mocked_aggregate:
                self._run_main(
                    "--task",
                    "g_aggregate_blind_review",
                    "--output-root",
                    str(output_root),
                    "--input-path",
                    str(worksheet_path),
                )
            mocked_aggregate.assert_called_once()
            run_dirs = list(output_root.glob("gblindsum_*"))
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue((run_dirs[0] / "run_meta.json").exists())

    def test_g_build_chapter_report_dispatches(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            output_root = temp_root / "runs"
            manifest_path = temp_root / "g_manifest.json"
            run_dirs = {}
            for group_id in ["G1", "G2", "G3", "G4"]:
                run_dir = temp_root / group_id.lower()
                run_dir.mkdir()
                (run_dir / "summary.csv").write_text("group_id\n" + group_id + "\n", encoding="utf-8")
                (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
                run_dirs[group_id] = str(run_dir)
            stats_path = temp_root / "pairwise_tests.csv"
            stats_path.write_text("group_a,group_b,metric\nG1,G2,citation_precision\n", encoding="utf-8")
            judge_path = temp_root / "judge_summary.csv"
            judge_path.write_text("group_id,judge_count\nG1,1\n", encoding="utf-8")
            blind_dir = temp_root / "blind_summary"
            blind_dir.mkdir()
            (blind_dir / "blind_review_item_summary.csv").write_text("blind_label,review_count\nA,1\n", encoding="utf-8")
            (blind_dir / "blind_review_pairwise_summary.csv").write_text("preference_label,count\nA>B,1\n", encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "group_run_dirs": run_dirs,
                        "pairwise_tests_path": str(stats_path),
                        "judge_summary_path": str(judge_path),
                        "blind_review_summary_dir": str(blind_dir),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch(
                "scripts.evaluation.g_workflow_closure.build_g_chapter_report",
                return_value={"output_dir": str(output_root / "gchapter_fake")},
            ) as mocked_build:
                self._run_main(
                    "--task",
                    "g_build_chapter_report",
                    "--output-root",
                    str(output_root),
                    "--input-path",
                    str(manifest_path),
                )
            mocked_build.assert_called_once()

    def test_g_prepare_and_validate_exp02_metadata_tasks(self) -> None:
        with patch(
            "scripts.evaluation.g_workflow_closure.ensure_exp02_metadata_placeholder",
            return_value=Path("experiments/assets/e10_adapter_metadata.qwen35_4b_peft_v2.placeholder.json"),
        ) as mocked_prepare, patch(
            "scripts.evaluation.g_workflow_closure.validate_exp02_metadata",
            return_value={"adapter_name": "exp02"},
        ) as mocked_validate:
            self._run_main("--task", "g_prepare_exp02_metadata_placeholder")
            self._run_main("--task", "g_validate_exp02_metadata")
        mocked_prepare.assert_called_once()
        mocked_validate.assert_called_once()

    def test_g_run_execution_readiness_writes_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            output_root = Path(tempdir) / "runs"
            with patch(
                "scripts.evaluation.g_workflow_closure.export_g_execution_readiness_report",
                return_value={"all_ready": False, "checks": [{"check": "exp02_metadata", "success": False, "detail": "missing"}]},
            ) as mocked_ready:
                self._run_main(
                    "--task",
                    "g_run_execution_readiness",
                    "--output-root",
                    str(output_root),
                )
            mocked_ready.assert_called_once()
            run_dirs = list(output_root.glob("gready_*"))
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue((run_dirs[0] / "run_meta.json").exists())

    def test_g_export_blind_review_pack_writes_meta_file(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            output_root = temp_root / "runs"
            g1_run_dir = temp_root / "g1"
            g2_run_dir = temp_root / "g2"
            g1_run_dir.mkdir()
            g2_run_dir.mkdir()
            (g1_run_dir / "results.jsonl").write_text("", encoding="utf-8")
            (g2_run_dir / "results.jsonl").write_text("", encoding="utf-8")
            input_path = temp_root / "blind_input.json"
            input_path.write_text(
                json.dumps({"G1": str(g1_run_dir), "G2": str(g2_run_dir)}, ensure_ascii=False),
                encoding="utf-8",
            )
            with patch(
                "scripts.evaluation.blind_review_export.export_blind_review_pack",
                return_value=[{"review_item_id": "blind_001_A"}],
            ) as mocked_export, patch(
                "scripts.evaluation.blind_review_export.export_blind_review_worksheet",
                return_value=[{"review_item_id": "blind_001_A", "overall_quality_score": ""}],
            ) as mocked_worksheet:
                self._run_main(
                    "--task",
                    "g_export_blind_review_pack",
                    "--output-root",
                    str(output_root),
                    "--input-path",
                    str(input_path),
                    "--sample-size",
                    "8",
                    "--seed",
                    "7",
                )

            mocked_export.assert_called_once()
            mocked_worksheet.assert_called_once()
            self.assertEqual(mocked_export.call_args.args[0], {"G1": g1_run_dir, "G2": g2_run_dir})
            self.assertEqual(mocked_export.call_args.kwargs["sample_size"], 8)
            self.assertEqual(mocked_export.call_args.kwargs["seed"], 7)
            run_dirs = list(output_root.glob("gblind_*"))
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue((run_dirs[0] / "blind_review_worksheet.csv").exists())
            self.assertTrue((run_dirs[0] / "blind_review_mapping.csv").exists())
            run_meta = json.loads((run_dirs[0] / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(run_meta["task"], "G_BLIND_REVIEW_EXPORT")
            self.assertEqual(run_meta["group_ids"], ["G1", "G2"])
            self.assertEqual(run_meta["exported_count"], 1)
            self.assertEqual(run_meta["worksheet_count"], 1)

    def test_g_export_blind_review_pack_rejects_missing_input_path(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            output_root = Path(tempdir) / "runs"
            missing_path = Path(tempdir) / "missing_blind_input.json"
            with self.assertRaises(FileNotFoundError):
                self._run_main(
                    "--task",
                    "g_export_blind_review_pack",
                    "--output-root",
                    str(output_root),
                    "--input-path",
                    str(missing_path),
                )

    def test_g_export_blind_review_pack_rejects_non_positive_sample_size(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            output_root = temp_root / "runs"
            input_path = temp_root / "blind_input.json"
            input_path.write_text(json.dumps({"G1": str(temp_root / 'g1')}, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                self._run_main(
                    "--task",
                    "g_export_blind_review_pack",
                    "--output-root",
                    str(output_root),
                    "--input-path",
                    str(input_path),
                    "--sample-size",
                    "0",
                )


if __name__ == "__main__":
    unittest.main()

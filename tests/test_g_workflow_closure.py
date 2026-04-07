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

    def test_build_g_execution_readiness_report_runs_semantic_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            fake_root = Path(tempdir)
            query_asset = fake_root / "g_query_ids.json"
            query_asset.write_text(json.dumps({"query_ids": [f"q{i:03d}" for i in range(1, 69)], "query_count": 68}), encoding="utf-8")
            plain_asset = fake_root / "plain.jsonl"
            aspect_asset = fake_root / "aspect.jsonl"
            qrels_asset = fake_root / "g_qrels.jsonl"
            plain_asset.write_text("[]", encoding="utf-8")
            aspect_asset.write_text("[]", encoding="utf-8")
            qrels_asset.write_text("[]", encoding="utf-8")
            with mock.patch.object(closure_mod, "G_QUERY_ID_ASSET_PATH", query_asset), \
                mock.patch.object(closure_mod, "G_PLAIN_RETRIEVAL_ASSET_PATH", plain_asset), \
                mock.patch.object(closure_mod, "G_ASPECT_RETRIEVAL_ASSET_PATH", aspect_asset), \
                mock.patch.object(closure_mod, "G_QRELS_EVIDENCE_PATH", qrels_asset), \
                mock.patch.object(closure_mod, "validate_g_retrieval_assets", side_effect=[{"query_count": 68}, ValueError("bad aspect asset")]), \
                mock.patch.object(closure_mod, "validate_g_qrels", return_value={"query_count": 68}), \
                mock.patch.object(closure_mod, "validate_exp02_metadata", return_value={"adapter_name": "exp02"}):
                readiness = closure_mod.build_g_execution_readiness_report()

        checks_by_name = {row["check"]: row for row in readiness["checks"]}
        self.assertTrue(checks_by_name["g_plain_retrieval_assets"]["success"])
        self.assertFalse(checks_by_name["g_aspect_retrieval_assets"]["success"])
        self.assertIn("bad aspect asset", checks_by_name["g_aspect_retrieval_assets"]["detail"])
        self.assertTrue(checks_by_name["g_qrels_asset"]["success"])

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

    def _write_generation_run(self, root: Path, group_id: str) -> Path:
        run_dir = root / group_id.lower()
        run_dir.mkdir()
        (run_dir / "run_meta.json").write_text(
            json.dumps({"run_id": f"{group_id}_run", "stable_run_config": {"task": "G"}}),
            encoding="utf-8",
        )
        (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
        # ???? summary.csv?????????????
        pd.DataFrame(
            [{
                "schema_valid_rate": 0.0,
                "citation_precision": 0.0,
                "evidence_verifiability_mean": 0.0,
                "recommendation_coverage": 0.0,
                "aspect_alignment_rate": 0.0,
                "hallucination_rate": 1.0,
                "unsupported_honesty_rate": 0.0,
            }]
        ).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")
        return run_dir

    def _write_retrieval_run(self, root: Path, name: str, variant: str, ndcg: float) -> Path:
        run_dir = root / name
        run_dir.mkdir()
        pd.DataFrame(
            [
                {
                    "group_id": f"{variant}_retrieval",
                    "aspect_recall_at_5": 0.8 if variant == "plain" else 0.9,
                    "ndcg_at_5": ndcg,
                    "precision_at_5": 0.4 if variant == "plain" else 0.5,
                    "mrr_at_5": 0.6 if variant == "plain" else 0.7,
                    "evidence_diversity_at_5": 0.7 if variant == "plain" else 0.8,
                    "avg_latency_ms": 12.5 if variant == "plain" else 15.0,
                    "retrieval_variant": variant,
                    "retrieval_mode": "plain_city_test_rerank" if variant == "plain" else "aspect_main_no_rerank",
                    "candidate_policy": "G_plain_retrieval_top5" if variant == "plain" else "G_aspect_retrieval_top5",
                    "retrieval_summary_source": "formal_retrieval_eval",
                }
            ]
        ).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")
        return run_dir

    def test_build_g_chapter_report_rebuilds_generation_summary_and_skips_pending_blind_review(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            run_dirs = {group_id: self._write_generation_run(temp_root, group_id) for group_id in ["G1", "G2", "G3", "G4"]}
            plain_run = self._write_retrieval_run(temp_root, "plain_ret", "plain", 0.55)
            aspect_run = self._write_retrieval_run(temp_root, "aspect_ret", "aspect", 0.77)
            retrieval_run_dirs = {
                "G1": plain_run,
                "G2": aspect_run,
                "G3": plain_run,
                "G4": aspect_run,
            }
            blind_dir = temp_root / "blind_summary"
            blind_dir.mkdir()
            pd.DataFrame([{"source_group_id": "G1", "review_count": 20}]).to_csv(
                blind_dir / "blind_review_item_summary.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame([{"preference_label": "G2>G1", "count": 3}]).to_csv(
                blind_dir / "blind_review_pairwise_summary.csv", index=False, encoding="utf-8-sig"
            )

            summary_by_group = {
                "g1": {"group_id": "G1", "schema_valid_rate": 0.9, "citation_precision": 0.91, "evidence_verifiability_mean": 1.0, "recommendation_coverage": 0.95, "aspect_alignment_rate": 0.8, "hallucination_rate": 0.2, "unsupported_honesty_rate": 0.9},
                "g2": {"group_id": "G2", "schema_valid_rate": 1.0, "citation_precision": 0.93, "evidence_verifiability_mean": 1.5, "recommendation_coverage": 0.97, "aspect_alignment_rate": 0.85, "hallucination_rate": 0.1, "unsupported_honesty_rate": 1.0},
                "g3": {"group_id": "G3", "schema_valid_rate": 0.88, "citation_precision": 0.89, "evidence_verifiability_mean": 1.1, "recommendation_coverage": 0.95, "aspect_alignment_rate": 0.82, "hallucination_rate": 0.25, "unsupported_honesty_rate": 1.0},
                "g4": {"group_id": "G4", "schema_valid_rate": 0.89, "citation_precision": 0.92, "evidence_verifiability_mean": 1.4, "recommendation_coverage": 0.95, "aspect_alignment_rate": 0.83, "hallucination_rate": 0.22, "unsupported_honesty_rate": 1.0},
            }

            def _fake_summary(run_dir: Path | str):
                run_dir = Path(run_dir)
                return {
                    "run_dir": run_dir,
                    "run_meta": {"run_id": f"{run_dir.name}_run"},
                    "summary_rows": [summary_by_group[run_dir.name]],
                }

            output_dir = temp_root / "report"
            with mock.patch.object(closure_mod, "summarize_generation_run", side_effect=_fake_summary):
                result = closure_mod.build_g_chapter_report(
                    run_dirs,
                    retrieval_run_dirs=retrieval_run_dirs,
                    blind_review_summary_dir=blind_dir,
                    blind_review_status=closure_mod.BLIND_REVIEW_STATUS_PENDING,
                    output_dir=output_dir,
                )

            self.assertEqual(result["output_dir"], str(output_dir))
            retrieval_df = pd.read_csv(output_dir / "g_retrieval_summary.csv")
            generation_df = pd.read_csv(output_dir / "g_generation_summary.csv")
            analysis_text = (output_dir / "analysis.md").read_text(encoding="utf-8")
            self.assertEqual(float(generation_df.loc[generation_df["group_id"] == "G1", "schema_valid_rate"].iloc[0]), 0.9)
            self.assertEqual(float(generation_df.loc[generation_df["group_id"] == "G1", "hallucination_rate"].iloc[0]), 0.2)
            self.assertEqual(retrieval_df.loc[0, "retrieval_summary_source"], "formal_retrieval_eval")
            self.assertIn("independent human review round is required", analysis_text)
            self.assertNotIn("## Human Blind Review Item Summary", analysis_text)

    def test_export_and_validate_g_closure_manifest_defaults_blind_review_status_to_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            run_dirs = {group_id: self._write_generation_run(temp_root, group_id) for group_id in ["G1", "G2", "G3", "G4"]}
            plain_run = self._write_retrieval_run(temp_root, "plain_ret", "plain", 0.55)
            aspect_run = self._write_retrieval_run(temp_root, "aspect_ret", "aspect", 0.77)
            retrieval_run_dirs = {
                "G1": plain_run,
                "G2": aspect_run,
                "G3": plain_run,
                "G4": aspect_run,
            }
            manifest_path = temp_root / "g_manifest.json"
            closure_mod.export_g_closure_manifest(run_dirs, manifest_path, retrieval_run_dirs=retrieval_run_dirs)
            validated = closure_mod.validate_g_closure_manifest(manifest_path)
            self.assertEqual(sorted(validated["group_run_dirs"]), ["G1", "G2", "G3", "G4"])
            self.assertEqual(sorted(validated["retrieval_run_dirs"]), ["G1", "G2", "G3", "G4"])
            self.assertEqual(validated["blind_review_status"], closure_mod.BLIND_REVIEW_STATUS_PENDING)

    def test_update_final_rerun_registry_updates_known_key(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            registry_path = Path(tempdir) / "final_rerun_registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "registry_version": 1,
                        "experiments": {"G1": None},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            run_dir = Path(tempdir) / "g1_run"
            run_dir.mkdir()
            (run_dir / "summary.csv").write_text("group_id\nG1\n", encoding="utf-8")
            (run_dir / "analysis.md").write_text("# analysis\n", encoding="utf-8")
            payload = closure_mod.build_registry_payload_for_run(
                "G1",
                run_dir=run_dir,
                query_scope="68",
                thesis_role="decisive evidence",
            )
            updated = closure_mod.update_final_rerun_registry({"G1": payload}, path=registry_path)
            self.assertEqual(updated["experiments"]["G1"]["run_dir"], str(run_dir))

    def test_validate_registry_matches_g_closure_manifest_detects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            run_dirs = {group_id: str(self._write_generation_run(temp_root, group_id)) for group_id in ["G1", "G2", "G3", "G4"]}
            plain_run = self._write_retrieval_run(temp_root, "plain_ret", "plain", 0.55)
            aspect_run = self._write_retrieval_run(temp_root, "aspect_ret", "aspect", 0.77)
            manifest_path = temp_root / "g_manifest.json"
            closure_mod.export_g_closure_manifest(
                run_dirs,
                manifest_path,
                retrieval_run_dirs={"G1": plain_run, "G2": aspect_run, "G3": plain_run, "G4": aspect_run},
            )
            registry_path = temp_root / "final_rerun_registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "registry_version": 1,
                        "experiments": {
                            "G1": {"run_dir": str(temp_root / "wrong_g1")},
                            "G2": {"run_dir": run_dirs["G2"]},
                            "G3": {"run_dir": run_dirs["G3"]},
                            "G4": {"run_dir": run_dirs["G4"]},
                            "G_retrieval_plain": {"run_dir": str(plain_run)},
                            "G_retrieval_aspect": {"run_dir": str(aspect_run)},
                            "G_pairwise_stats": {"summary_path": None},
                            "G_llm_judge": {"summary_path": None},
                            "G_blind_review": {"run_dir": None},
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                closure_mod.validate_registry_matches_g_closure_manifest(
                    manifest_or_path=manifest_path,
                    registry_path=registry_path,
                )

    def test_export_latest_g_closure_manifest_from_workspace_raises_clear_error_when_retrieval_runs_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            run_dirs = {
                group_id: str(self._write_generation_run(temp_root, group_id))
                for group_id in ["G1", "G2", "G3", "G4"]
            }
            run_dirs_json = temp_root / "g_run_dirs.json"
            run_dirs_json.write_text(json.dumps(run_dirs), encoding="utf-8")

            with self.assertRaises(FileNotFoundError) as ctx:
                closure_mod.export_latest_g_closure_manifest_from_workspace(
                    run_dirs_json_path=run_dirs_json,
                    output_path=temp_root / "g_closure_manifest.json",
                    run_root=temp_root,
                )

            self.assertIn("G plain retrieval run", str(ctx.exception))

    def test_run_g_batch_llm_judge_seeds_from_previous_batch_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            run_dirs = {group_id: self._write_generation_run(temp_root, group_id) for group_id in ["G1", "G2", "G3", "G4"]}
            prior_batch_dir = temp_root / "gjudgebatch_prior"
            prior_batch_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "query_id": "q001",
                        "group_id": "G1",
                        "judge_model": "deepseek-chat",
                        "relevance": 4.0,
                        "traceability": 4.0,
                        "fluency": 4.0,
                        "completeness": 4.0,
                        "honesty": 4.0,
                        "overall_mean": 4.0,
                        "brief_rationale": "seeded",
                    }
                ]
            ).to_csv(prior_batch_dir / "g1_judge_scores.csv", index=False, encoding="utf-8-sig")

            observed_seed_contents: list[str] = []

            def _fake_run_llm_judge(run_dir: Path, *, output_path: Path, model: str, client=None):
                if Path(output_path).name == "g1_judge_scores.csv":
                    observed_seed_contents.append(Path(output_path).read_text(encoding="utf-8-sig"))
                return pd.DataFrame(
                    [
                        {
                            "query_id": "q001",
                            "group_id": run_dir.name.upper(),
                            "judge_model": model,
                            "relevance": 4.0,
                            "traceability": 4.0,
                            "fluency": 4.0,
                            "completeness": 4.0,
                            "honesty": 4.0,
                            "overall_mean": 4.0,
                            "brief_rationale": "ok",
                        }
                    ]
                )

            current_batch_dir = temp_root / "gjudgebatch_current"
            with mock.patch.object(closure_mod, "run_llm_judge", side_effect=_fake_run_llm_judge):
                closure_mod.run_g_batch_llm_judge(run_dirs, output_dir=current_batch_dir, model="deepseek-chat")

            self.assertEqual(len(observed_seed_contents), 1)
            self.assertIn("seeded", observed_seed_contents[0])
            self.assertTrue((current_batch_dir / "g1_judge_scores.csv").exists())

    def test_validate_g_closure_manifest_requires_blind_review_dir_when_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            run_dirs = {group_id: str(self._write_generation_run(temp_root, group_id)) for group_id in ["G1", "G2", "G3", "G4"]}
            plain_run = self._write_retrieval_run(temp_root, "plain_ret", "plain", 0.55)
            aspect_run = self._write_retrieval_run(temp_root, "aspect_ret", "aspect", 0.77)
            retrieval_run_dirs = {
                "G1": str(plain_run),
                "G2": str(aspect_run),
                "G3": str(plain_run),
                "G4": str(aspect_run),
            }
            manifest = closure_mod.build_g_closure_manifest(
                run_dirs,
                retrieval_run_dirs=retrieval_run_dirs,
                blind_review_status=closure_mod.BLIND_REVIEW_STATUS_READY,
            )
            with self.assertRaises(FileNotFoundError):
                closure_mod.validate_g_closure_manifest(manifest)

    def test_build_g_chapter_report_renders_llm_blind_review_section(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            run_dirs = {group_id: self._write_generation_run(temp_root, group_id) for group_id in ["G1", "G2", "G3", "G4"]}
            plain_run = self._write_retrieval_run(temp_root, "plain_ret", "plain", 0.55)
            aspect_run = self._write_retrieval_run(temp_root, "aspect_ret", "aspect", 0.77)
            retrieval_run_dirs = {
                "G1": plain_run,
                "G2": aspect_run,
                "G3": plain_run,
                "G4": aspect_run,
            }
            blind_dir = temp_root / "blind_summary"
            blind_dir.mkdir()
            pd.DataFrame([{"source_group_id": "G1", "review_count": 20}]).to_csv(
                blind_dir / "blind_review_item_summary.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame([{"preference_label": "G2>G1", "count": 3}]).to_csv(
                blind_dir / "blind_review_pairwise_summary.csv", index=False, encoding="utf-8-sig"
            )

            def _fake_summary(run_dir: Path | str):
                run_dir = Path(run_dir)
                group_id = run_dir.name.upper()
                return {
                    "run_dir": run_dir,
                    "run_meta": {"run_id": f"{run_dir.name}_run"},
                    "summary_rows": [{
                        "group_id": group_id,
                        "schema_valid_rate": 1.0,
                        "citation_precision": 0.95,
                        "evidence_verifiability_mean": 1.5,
                        "recommendation_coverage": 0.95,
                        "aspect_alignment_rate": 0.9,
                        "hallucination_rate": 0.02,
                        "unsupported_honesty_rate": 1.0,
                    }],
                }

            output_dir = temp_root / "report_llm"
            with mock.patch.object(closure_mod, "summarize_generation_run", side_effect=_fake_summary):
                closure_mod.build_g_chapter_report(
                    run_dirs,
                    retrieval_run_dirs=retrieval_run_dirs,
                    blind_review_summary_dir=blind_dir,
                    blind_review_status=closure_mod.BLIND_REVIEW_STATUS_LLM_READY,
                    output_dir=output_dir,
                )

            analysis_text = (output_dir / "analysis.md").read_text(encoding="utf-8")
            self.assertIn("## LLM Blind Re-Review", analysis_text)
            self.assertIn("must not be presented as independent human evaluation", analysis_text)
            self.assertIn("## LLM Blind Re-Review Item Summary", analysis_text)

    def test_build_g_chapter_report_renders_human_verified_blind_review_section(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            run_dirs = {group_id: self._write_generation_run(temp_root, group_id) for group_id in ["G1", "G2", "G3", "G4"]}
            plain_run = self._write_retrieval_run(temp_root, "plain_ret", "plain", 0.55)
            aspect_run = self._write_retrieval_run(temp_root, "aspect_ret", "aspect", 0.77)
            retrieval_run_dirs = {
                "G1": plain_run,
                "G2": aspect_run,
                "G3": plain_run,
                "G4": aspect_run,
            }
            blind_dir = temp_root / "blind_summary"
            blind_dir.mkdir()
            pd.DataFrame([{"source_group_id": "G1", "review_count": 20}]).to_csv(
                blind_dir / "blind_review_item_summary.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame([{"preference_label": "G2>G1", "count": 3}]).to_csv(
                blind_dir / "blind_review_pairwise_summary.csv", index=False, encoding="utf-8-sig"
            )

            def _fake_summary(run_dir: Path | str):
                run_dir = Path(run_dir)
                group_id = run_dir.name.upper()
                return {
                    "run_dir": run_dir,
                    "run_meta": {"run_id": f"{run_dir.name}_run"},
                    "summary_rows": [{
                        "group_id": group_id,
                        "schema_valid_rate": 1.0,
                        "citation_precision": 0.95,
                        "evidence_verifiability_mean": 1.5,
                        "recommendation_coverage": 0.95,
                        "aspect_alignment_rate": 0.9,
                        "hallucination_rate": 0.02,
                        "unsupported_honesty_rate": 1.0,
                    }],
                }

            output_dir = temp_root / "report_human_verified"
            with mock.patch.object(closure_mod, "summarize_generation_run", side_effect=_fake_summary):
                closure_mod.build_g_chapter_report(
                    run_dirs,
                    retrieval_run_dirs=retrieval_run_dirs,
                    blind_review_summary_dir=blind_dir,
                    blind_review_status=closure_mod.BLIND_REVIEW_STATUS_HUMAN_VERIFIED,
                    output_dir=output_dir,
                )

            analysis_text = (output_dir / "analysis.md").read_text(encoding="utf-8")
            self.assertIn("## Human-Verified Blind Review", analysis_text)
            self.assertIn("manually checked and accepted by the researcher", analysis_text)
            self.assertIn("## Human-Verified Blind Review Item Summary", analysis_text)

    def test_extract_g_group_score_map_uses_query_level_metrics_and_filters_unsupported(self) -> None:
        run_dirs = {group_id: f"/fake/{group_id}" for group_id in ["G1", "G2", "G3", "G4"]}
        grouped_rows = {
            group_id: [
                {
                    "query_id": "q001",
                    "response": mock.Mock(schema_valid=True, recommendations=[mock.Mock(reasons=[])]),
                    "verification": mock.Mock(citation_precision=1.0),
                    "audit_rows": [
                        {"support_score": 2, "citation_exists": 1},
                        {"support_score": 2, "citation_exists": 1},
                        {"support_score": 2, "citation_exists": 1},
                        {"support_score": 2, "citation_exists": 1},
                    ],
                    "eval_unit": None,
                    "unsupported_honesty": 1,
                    "latency_ms": 10.0,
                },
                {
                    "query_id": "q002",
                    "response": mock.Mock(schema_valid=True, recommendations=[]),
                    "verification": mock.Mock(citation_precision=0.9),
                    "audit_rows": [{"support_score": 0, "citation_exists": 0}],
                    "eval_unit": None,
                    "unsupported_honesty": None,
                    "latency_ms": 11.0,
                },
            ]
            for group_id in ["G1", "G2", "G3", "G4"]
        }
        with mock.patch.object(closure_mod, "load_generation_run_artifacts", return_value=({}, [], [])), mock.patch.object(
            closure_mod,
            "reconstruct_generation_group_rows",
            return_value=grouped_rows,
        ):
            payload = closure_mod.extract_g_group_score_map(run_dirs)
        self.assertEqual(payload["G1"]["evidence_verifiability_mean"]["scores"], [2.0, 0.0])
        self.assertEqual(payload["G1"]["hallucination_rate"]["scores"], [0.0, 1.0])
        self.assertEqual(payload["G1"]["unsupported_honesty_rate"]["scores"], [1.0])
        self.assertEqual(payload["G1"]["unsupported_honesty_rate"]["query_ids"], ["q001"])

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


if __name__ == "__main__":
    unittest.main()

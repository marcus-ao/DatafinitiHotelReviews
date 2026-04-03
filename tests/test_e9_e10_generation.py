import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import pandas as pd

from scripts.evaluation import evaluate_e9_e10_generation as generation_mod
from scripts.shared.experiment_schemas import (
    CitationVerificationResult,
    EvidencePack,
    GenerationEvalUnit,
    HotelCandidate,
    RecommendationReason,
    RecommendationItem,
    RecommendationResponse,
    SentenceCandidate,
    UserPreference,
)


class _FakeLLMRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.runtime_config = None

    def generate_json(self, system_prompt: str, user_prompt: str, max_new_tokens: int | None = None) -> str:
        if not self.responses:
            raise AssertionError("No fake response left for generate_json")
        return self.responses.pop(0)


def _build_eval_unit() -> GenerationEvalUnit:
    return GenerationEvalUnit(
        query_id="q001",
        query_text_zh="我想在Anaheim找一家位置交通比较好的酒店。",
        query_type="single_aspect",
        user_preference_gold=UserPreference(
            city="Anaheim",
            state="CA",
            hotel_category=None,
            focus_aspects=["location_transport"],
            avoid_aspects=[],
            unsupported_requests=[],
            query_en="hotel in Anaheim with convenient location and transportation",
        ),
        unsupported_requests=[],
        candidate_hotels=[
            HotelCandidate(
                hotel_id="hotel_1",
                hotel_name="Hotel One",
                score_total=1.5,
                score_breakdown={"focus_location_transport": 1.5},
            )
        ],
        evidence_packs=[
            EvidencePack(
                hotel_id="hotel_1",
                query_en="hotel in Anaheim with convenient location and transportation",
                evidence_by_aspect={
                    "location_transport": [
                        SentenceCandidate(
                            sentence_id="s_valid",
                            sentence_text="Very convenient to the stadium.",
                            aspect="location_transport",
                            sentiment="positive",
                            review_date="2018-01-01",
                            score_dense=0.12,
                            score_rerank=None,
                        )
                    ]
                },
                all_sentence_ids=["s_valid"],
                retrieval_trace={"mode": "aspect_main_no_rerank"},
            )
        ],
        retrieval_mode="aspect_main_no_rerank",
        candidate_policy="E2_B_final_aspect_score_top5",
        config_hash="cfg001",
    )


def _build_generation_log_row(
    run_id: str,
    group_id: str,
    query_id: str = "q001",
    *,
    schema_valid: bool = True,
    citation_precision: float = 1.0,
    support_scores: list[int] | None = None,
    unsupported_notice: str = "",
    unsupported_honesty: int | None = None,
    response_error_type: str | None = None,
    latency_ms: float = 100.0,
    llm_backend: str = "local",
    model_id: str = "/tmp/model",
) -> dict[str, object]:
    support_scores = support_scores or []
    reasons = []
    audit_rows = []
    if support_scores:
        reasons.append(
            {
                "aspect": "location_transport",
                "reason_text": "位置方便。",
                "sentence_id": "s_valid",
            }
        )
        audit_rows = [
            {
                "query_id": query_id,
                "group_id": group_id,
                "hotel_id": "hotel_1",
                "sentence_id": f"s_valid_{idx}",
                "reason_text": "位置方便。",
                "citation_exists": 1,
                "in_current_evidence_pack": 1,
                "support_score": score,
                "notes": "",
            }
            for idx, score in enumerate(support_scores, start=1)
        ]
    response = RecommendationResponse(
        query_id=query_id,
        group_id=group_id,
        summary="测试摘要" if schema_valid and (reasons or unsupported_notice) else "",
        recommendations=(
            [RecommendationItem(hotel_id="hotel_1", hotel_name="Hotel One", reasons=[
                RecommendationReason(**reason) for reason in reasons
            ])]
            if reasons
            else []
        ),
        unsupported_notice=unsupported_notice,
        schema_valid=schema_valid,
        raw_response='{"summary":"测试摘要"}' if schema_valid else "Thinking Process:\n1. test",
    )
    verification = CitationVerificationResult(
        query_id=query_id,
        group_id=group_id,
        citation_precision=citation_precision,
        invalid_sentence_ids=[],
        out_of_pack_sentence_ids=[],
        retry_triggered=False,
        fallback_to_honest_notice=False,
    )
    return {
        "run_id": run_id,
        "group_id": group_id,
        "query_id": query_id,
        "retrieval_mode": "aspect_main_no_rerank",
        "candidate_mode": "e10_frozen_assets",
        "config_hash": "cfg",
        "latency_ms": latency_ms,
        "intermediate_objects": {
            "response": response.model_dump(),
            "citation_verification": verification.model_dump(),
            "audit_rows": audit_rows,
            "unsupported_honesty": unsupported_honesty,
            "response_error_type": response_error_type,
            "reasoning_leak_detected": response_error_type == "reasoning_leak",
            "behavior_runtime_config": {
                "llm_backend": llm_backend,
                "model_id": model_id,
            },
        },
    }


class E9E10GenerationTestCase(unittest.TestCase):
    def test_build_e9_query_rows_matches_frozen_query_count(self):
        rows = generation_mod.build_e9_query_rows(limit_queries=None)
        self.assertEqual(len(rows), 40)
        self.assertEqual(rows[0][0]["query_id"], "q001")

    def test_coerce_generation_payload_enforces_citation_for_grounded_groups(self):
        unit = _build_eval_unit()
        payload = {
            "summary": "推荐结果",
            "recommendations": [
                {
                    "hotel_id": "hotel_1",
                    "hotel_name": "Hotel One",
                    "reasons": [
                        {
                            "aspect": "location_transport",
                            "reason_text": "位置交通方便。",
                            "sentence_id": None,
                        }
                    ],
                }
            ],
            "unsupported_notice": "",
        }
        response = generation_mod.coerce_generation_payload(
            payload=payload,
            unit=unit,
            group_id="B_grounded_generation",
            raw_response=json.dumps(payload, ensure_ascii=False),
        )
        self.assertFalse(response.schema_valid)
        self.assertEqual(len(response.recommendations), 1)

    def test_verify_response_citations_distinguishes_valid_invalid_and_out_of_pack(self):
        unit = _build_eval_unit()
        response = RecommendationResponse(
            query_id="q001",
            group_id="B_grounded_generation",
            summary="推荐结果",
            recommendations=[
                RecommendationItem(
                    hotel_id="hotel_1",
                    hotel_name="Hotel One",
                    reasons=[
                        RecommendationReason(
                            aspect="location_transport",
                            reason_text="位置很好。",
                            sentence_id="s_valid",
                        ),
                        RecommendationReason(
                            aspect="location_transport",
                            reason_text="位置也不错。",
                            sentence_id="s_other_pack",
                        ),
                        RecommendationReason(
                            aspect="location_transport",
                            reason_text="靠近景点。",
                            sentence_id="s_missing",
                        ),
                    ],
                )
            ],
            unsupported_notice="",
            schema_valid=True,
            raw_response="{}",
        )
        verification, audit_rows = generation_mod.verify_response_citations(
            response=response,
            unit=unit,
            evidence_lookup={
                "s_valid": {"aspect": "location_transport"},
                "s_other_pack": {"aspect": "location_transport"},
            },
        )
        self.assertIsInstance(verification, CitationVerificationResult)
        self.assertEqual(verification.citation_precision, 0.3333)
        self.assertEqual(verification.invalid_sentence_ids, ["s_missing"])
        self.assertEqual(verification.out_of_pack_sentence_ids, ["s_other_pack"])
        self.assertEqual([row["support_score"] for row in audit_rows], [2, 0, 0])

    def test_coerce_generation_payload_lifts_item_level_unsupported_notice(self):
        unit = _build_eval_unit()
        payload = {
            "summary": "",
            "recommendations": [
                {
                    "hotel_id": "hotel_1",
                    "hotel_name": "Hotel One",
                    "reasons": [],
                    "unsupported_notice": "该酒店缺少足够证据。",
                }
            ],
            "unsupported_notice": "",
        }
        response = generation_mod.coerce_generation_payload(
            payload=payload,
            unit=unit,
            group_id="B_grounded_generation",
            raw_response=json.dumps(payload, ensure_ascii=False),
        )
        self.assertTrue(response.schema_valid)
        self.assertEqual(response.recommendations, [])
        self.assertEqual(response.unsupported_notice, "该酒店缺少足够证据。")

    def test_generate_group_response_falls_back_after_repeat_invalid_citations(self):
        unit = _build_eval_unit()
        invalid_payload = json.dumps(
            {
                "summary": "推荐结果",
                "recommendations": [
                    {
                        "hotel_id": "hotel_1",
                        "hotel_name": "Hotel One",
                        "reasons": [
                            {
                                "aspect": "location_transport",
                                "reason_text": "位置很好。",
                                "sentence_id": "s_missing",
                            }
                        ],
                    }
                ],
                "unsupported_notice": "",
            },
            ensure_ascii=False,
        )
        runner = _FakeLLMRunner([invalid_payload, invalid_payload])
        response, verification, audit_rows, debug_payload = generation_mod.generate_group_response(
            llm_runner=runner,
            unit=unit,
            group_id="C_grounded_generation_with_verifier",
            max_new_tokens=256,
            evidence_lookup={"s_valid": {"aspect": "location_transport"}},
        )
        self.assertTrue(verification.retry_triggered)
        self.assertTrue(verification.fallback_to_honest_notice)
        self.assertEqual(response.recommendations, [])
        self.assertIn("retry_raw_response", debug_payload)
        self.assertGreaterEqual(len(audit_rows), 0)

    def test_build_sft_manifest_records_contains_only_allowed_task_types(self):
        train_records, dev_records = generation_mod.build_sft_manifest_records()
        allowed = {"preference_parse", "clarification", "constraint_honesty", "feedback_update"}
        self.assertTrue(train_records)
        self.assertTrue(dev_records)
        task_types = {row["task_type"] for row in train_records + dev_records}
        self.assertTrue(task_types.issubset(allowed))
        self.assertTrue(all(row["hotel_id"] is None for row in train_records + dev_records))

    def test_load_adapter_metadata_validates_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path = Path(tmp_dir) / "adapter_metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "adapter_name": "peft_v1",
                        "base_model_id": "Qwen/Qwen3.5-4B",
                        "served_model_id": "Qwen/Qwen3.5-4B-PEFT",
                        "adapter_path": "/tmp/adapter",
                        "backend": "api",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            payload = generation_mod.load_adapter_metadata(metadata_path)
        self.assertEqual(payload["adapter_name"], "peft_v1")
        self.assertEqual(payload["_resolved_adapter_path"], "/tmp/adapter")

    def test_build_peft_runtime_config_requires_matching_backend(self):
        runtime_config = generation_mod.BehaviorRuntimeConfig(
            llm_backend="api",
            model_id="Qwen/Qwen3.5-4B",
            api_base_url="http://127.0.0.1:8000/v1",
            api_key_env="OPENAI_API_KEY",
            api_key_present=True,
            enable_thinking=False,
            temperature=0.0,
            max_new_tokens=512,
            api_timeout_seconds=120,
        )
        peft_runtime = generation_mod.build_peft_runtime_config(
            runtime_config,
            {
                "adapter_name": "peft_v1",
                "base_model_id": "Qwen/Qwen3.5-4B",
                "served_model_id": "Qwen/Qwen3.5-4B-PEFT",
                "adapter_path": "/tmp/adapter",
                "backend": "api",
                "_metadata_path": "/tmp/adapter_metadata.json",
                "_resolved_adapter_path": "/tmp/adapter",
            },
        )
        self.assertTrue(peft_runtime.use_peft_adapter)
        self.assertEqual(peft_runtime.model_id, "Qwen/Qwen3.5-4B-PEFT")
        self.assertEqual(peft_runtime.adapter_path, "/tmp/adapter")
        self.assertEqual(peft_runtime.adapter_metadata_path, "/tmp/adapter_metadata.json")

    def test_build_peft_runtime_config_supports_local_merged_model_path(self):
        runtime_config = generation_mod.BehaviorRuntimeConfig(
            llm_backend="local",
            model_id="/root/autodl-tmp/models/merged/qwen35_4b_merged_exp01",
            api_base_url=None,
            api_key_env="OPENAI_API_KEY",
            api_key_present=False,
            enable_thinking=False,
            temperature=0.0,
            max_new_tokens=512,
            api_timeout_seconds=120,
        )
        peft_runtime = generation_mod.build_peft_runtime_config(
            runtime_config,
            {
                "adapter_name": "peft_v1",
                "base_model_id": "Qwen/Qwen3.5-4B",
                "served_model_id": "Qwen3.5-4B-PEFT-exp01",
                "adapter_path": "/tmp/adapter",
                "backend": "api",
                "_metadata_path": "/tmp/adapter_metadata.json",
                "_resolved_adapter_path": "/tmp/adapter",
            },
        )
        self.assertEqual(peft_runtime.llm_backend, "local")
        self.assertEqual(peft_runtime.model_id, "/root/autodl-tmp/models/merged/qwen35_4b_merged_exp01")
        self.assertTrue(peft_runtime.use_peft_adapter)

    def test_validate_adapter_metadata_base_model_accepts_path_and_hf_name(self):
        generation_mod.validate_adapter_metadata_base_model(
            {
                "base_model_id": "/root/autodl-tmp/models/base/Qwen3.5-4B",
            },
            "Qwen/Qwen3.5-4B",
        )

    def test_validate_runtime_base_model_accepts_path_and_hf_name(self):
        generation_mod.validate_runtime_base_model(
            "/root/autodl-tmp/models/base/Qwen3.5-4B",
            "Qwen/Qwen3.5-4B",
        )

    def test_run_e10_base_vs_peft_requires_adapter_metadata_path(self):
        runtime_config = generation_mod.BehaviorRuntimeConfig(
            llm_backend="api",
            model_id="Qwen/Qwen3.5-4B",
            api_base_url="http://127.0.0.1:8000/v1",
            api_key_env="OPENAI_API_KEY",
            api_key_present=True,
            enable_thinking=False,
            temperature=0.0,
            max_new_tokens=256,
            api_timeout_seconds=120,
        )
        with mock.patch.object(
            generation_mod,
            "resolve_behavior_runtime_config",
            return_value=(runtime_config, "EMPTY"),
        ):
            with self.assertRaises(ValueError):
                generation_mod.run_e10_base_vs_peft(Path("/tmp/e10_missing_adapter"), limit_queries=1)

    def test_run_e10_base_group_only_does_not_require_adapter_metadata(self):
        runtime_config = generation_mod.BehaviorRuntimeConfig(
            llm_backend="api",
            model_id="Qwen/Qwen3.5-4B",
            api_base_url="http://127.0.0.1:8000/v1",
            api_key_env="OPENAI_API_KEY",
            api_key_present=True,
            enable_thinking=False,
            temperature=0.0,
            max_new_tokens=256,
            api_timeout_seconds=120,
        )
        with mock.patch.object(
            generation_mod,
            "resolve_behavior_runtime_config",
            return_value=(runtime_config, "EMPTY"),
        ), mock.patch.object(
            generation_mod,
            "build_behavior_backend",
            side_effect=RuntimeError("stop_after_runtime_resolution"),
        ):
            with self.assertRaises(RuntimeError):
                generation_mod.run_e10_base_vs_peft(
                    Path("/tmp/e10_base_only"),
                    limit_queries=1,
                    group_ids=["A_base_4b_grounded"],
                )

    def test_run_e10_base_group_writes_reasoning_diagnostics_without_name_error(self):
        runtime_config = generation_mod.BehaviorRuntimeConfig(
            llm_backend="local",
            model_id="Qwen/Qwen3.5-4B",
            api_base_url=None,
            api_key_env="OPENAI_API_KEY",
            api_key_present=False,
            enable_thinking=False,
            temperature=0.0,
            max_new_tokens=256,
            api_timeout_seconds=120,
        )
        unit = _build_eval_unit()
        response = RecommendationResponse(
            query_id="q001",
            group_id="A_base_4b_grounded",
            summary="ok",
            recommendations=[],
            unsupported_notice="",
            schema_valid=True,
            raw_response="{}",
        )
        verification = CitationVerificationResult(
            query_id="q001",
            group_id="A_base_4b_grounded",
            citation_precision=1.0,
            invalid_sentence_ids=[],
            out_of_pack_sentence_ids=[],
            retry_triggered=False,
            fallback_to_honest_notice=False,
        )
        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch.object(
            generation_mod,
            "resolve_behavior_runtime_config",
            return_value=(runtime_config, None),
        ), mock.patch.object(
            generation_mod,
            "load_generation_eval_units",
            return_value=[unit],
        ), mock.patch.object(
            generation_mod.pd,
            "read_pickle",
            return_value=object(),
        ), mock.patch.object(
            generation_mod,
            "build_evidence_lookup",
            return_value={},
        ), mock.patch.object(
            generation_mod,
            "build_behavior_backend",
            return_value=object(),
        ), mock.patch.object(
            generation_mod,
            "generate_group_response",
            return_value=(
                response,
                verification,
                [],
                {
                    "raw_response_initial": "{}",
                    "retry_raw_response": "",
                    "response_error_type": None,
                    "thinking_control_supported": True,
                    "raw_response_prefix": "{}",
                },
            ),
        ):
            run_dir = generation_mod.run_e10_base_vs_peft(
                Path(tmp_dir),
                limit_queries=1,
                group_ids=["A_base_4b_grounded"],
            )
            self.assertTrue((run_dir / "results.jsonl").exists())
            with (run_dir / "results.jsonl").open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(rows[0]["intermediate_objects"]["response_error_type"], None)
            self.assertFalse(rows[0]["intermediate_objects"]["reasoning_leak_detected"])

    def test_run_e10_multiple_groups_now_rejected(self):
        runtime_config = generation_mod.BehaviorRuntimeConfig(
            llm_backend="api",
            model_id="Qwen/Qwen3.5-4B",
            api_base_url="http://127.0.0.1:8000/v1",
            api_key_env="OPENAI_API_KEY",
            api_key_present=True,
            enable_thinking=False,
            temperature=0.0,
            max_new_tokens=256,
            api_timeout_seconds=120,
        )
        with mock.patch.object(
            generation_mod,
            "resolve_behavior_runtime_config",
            return_value=(runtime_config, "EMPTY"),
        ):
            with self.assertRaises(ValueError):
                generation_mod.run_e10_base_vs_peft(
                    Path("/tmp/e10_bad_groups"),
                    limit_queries=1,
                    group_ids=["A_base_4b_grounded", "B_peft_4b_grounded"],
                )

    def test_build_e10_metric_row_uses_na_for_unsupported_when_not_applicable(self):
        row = {
            "query_id": "q001",
            "latency_ms": 100.0,
            "response": RecommendationResponse(
                query_id="q001",
                group_id="A_base_4b_grounded",
                summary="ok",
                recommendations=[],
                unsupported_notice="",
                schema_valid=True,
                raw_response="{}",
            ),
            "verification": CitationVerificationResult(
                query_id="q001",
                group_id="A_base_4b_grounded",
                citation_precision=1.0,
                invalid_sentence_ids=[],
                out_of_pack_sentence_ids=[],
                retry_triggered=False,
                fallback_to_honest_notice=False,
            ),
            "audit_rows": [],
            "unsupported_honesty": None,
            "response_error_type": "reasoning_leak",
        }
        metric_row = generation_mod.build_e10_metric_row(
            "A_base_4b_grounded",
            [row],
            {"task": "E10"},
        )
        self.assertIsNone(metric_row["unsupported_honesty_rate"])
        self.assertEqual(metric_row["reasoning_leak_rate"], 1.0)
        self.assertEqual(metric_row["auditable_query_rate"], 0.0)

    def test_build_e10_analysis_md_marks_single_group_sections_as_not_applicable(self):
        grouped_rows = {
            "A_base_4b_grounded": [
                {
                    "query_id": "q001",
                    "latency_ms": 10.0,
                    "response": RecommendationResponse(
                        query_id="q001",
                        group_id="A_base_4b_grounded",
                        summary="ok",
                        recommendations=[],
                        unsupported_notice="",
                        schema_valid=True,
                        raw_response="{}",
                    ),
                    "verification": CitationVerificationResult(
                        query_id="q001",
                        group_id="A_base_4b_grounded",
                        citation_precision=1.0,
                        invalid_sentence_ids=[],
                        out_of_pack_sentence_ids=[],
                        retry_triggered=False,
                        fallback_to_honest_notice=False,
                    ),
                    "audit_rows": [],
                    "unsupported_honesty": None,
                    "response_error_type": None,
                }
            ]
        }
        summary_rows = [
            {
                "group_id": "A_base_4b_grounded",
                "query_count": 1,
                "citation_precision": 1.0,
                "evidence_verifiability_mean": 0.0,
                "unsupported_honesty_rate": None,
                "schema_valid_rate": 1.0,
                "reasoning_leak_rate": 0.0,
                "auditable_query_rate": 0.0,
                "avg_latency_ms": 10.0,
                "config_hash": "cfg",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            generation_mod.build_e10_analysis_md(run_dir, summary_rows, grouped_rows, adapter_metadata=None)
            analysis_text = (run_dir / "analysis.md").read_text(encoding="utf-8")
        self.assertIn("E10 Single-Group Diagnostic Result", analysis_text)
        self.assertIn("not applicable in base-only run", analysis_text)
        self.assertIn("not available in single-group run", analysis_text)
        self.assertIn("n/a (no applicable unsupported-request queries)", analysis_text)

    def test_run_e10_compare_runs_generates_comparison_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            base_run_dir = tmp_path / "base_run"
            peft_run_dir = tmp_path / "peft_run"
            base_run_dir.mkdir()
            peft_run_dir.mkdir()

            base_log_rows = [
                _build_generation_log_row(
                    "base_run",
                    "A_base_4b_grounded",
                    query_id="q001",
                    citation_precision=0.5,
                    support_scores=[1],
                    latency_ms=100.0,
                )
            ]
            peft_log_rows = [
                _build_generation_log_row(
                    "peft_run",
                    "B_peft_4b_grounded",
                    query_id="q001",
                    citation_precision=1.0,
                    support_scores=[2],
                    latency_ms=120.0,
                )
            ]

            (base_run_dir / "run_meta.json").write_text(
                json.dumps({"run_id": "base_run", "stable_run_config": {"task": "E10"}}),
                encoding="utf-8",
            )
            (peft_run_dir / "run_meta.json").write_text(
                json.dumps({"run_id": "peft_run", "stable_run_config": {"task": "E10"}}),
                encoding="utf-8",
            )
            (base_run_dir / "results.jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in base_log_rows),
                encoding="utf-8",
            )
            (peft_run_dir / "results.jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in peft_log_rows),
                encoding="utf-8",
            )

            compare_dir = generation_mod.run_e10_compare_runs(
                output_root=tmp_path,
                base_run_dir=base_run_dir,
                peft_run_dir=peft_run_dir,
            )

            self.assertTrue((compare_dir / "summary.csv").exists())
            self.assertTrue((compare_dir / "analysis.md").exists())
            self.assertTrue((compare_dir / "comparison.jsonl").exists())
            analysis_text = (compare_dir / "analysis.md").read_text(encoding="utf-8")
            self.assertIn("E10 Base vs PEFT Compare Result", analysis_text)
            self.assertIn("base_run", analysis_text)
            self.assertIn("peft_run", analysis_text)

    def test_build_e10_v2_grounded_query_rows_excludes_official_e9_ids(self):
        rows = generation_mod.build_e10_v2_grounded_query_rows()
        official_ids = set(generation_mod.load_json(generation_mod.E9_QUERY_IDS_PATH))
        self.assertTrue(rows)
        self.assertTrue(all(query_row["query_id"] not in official_ids for query_row, _, _ in rows))
        self.assertTrue(all(slot_row["city"] for _, slot_row, _ in rows))
        self.assertTrue(all(slot_row["focus_aspects"] or slot_row["avoid_aspects"] for _, slot_row, _ in rows))

    def test_validate_grounded_recommendation_example_rejects_english_reason(self):
        unit = _build_eval_unit()
        response = RecommendationResponse(
            query_id="q001",
            group_id="C_grounded_generation_with_verifier",
            summary="推荐 Anaheim 酒店。",
            recommendations=[
                RecommendationItem(
                    hotel_id="hotel_1",
                    hotel_name="Hotel One",
                    reasons=[
                        RecommendationReason(
                            aspect="location_transport",
                            reason_text="Great location for stadium access.",
                            sentence_id="s_valid",
                        )
                    ],
                )
            ],
            unsupported_notice="",
            schema_valid=True,
            raw_response="{}",
        )
        verification = CitationVerificationResult(
            query_id="q001",
            group_id="C_grounded_generation_with_verifier",
            citation_precision=1.0,
            invalid_sentence_ids=[],
            out_of_pack_sentence_ids=[],
            retry_triggered=False,
            fallback_to_honest_notice=False,
        )
        is_valid, reason = generation_mod.validate_grounded_recommendation_example(
            unit,
            response,
            verification,
            {"response_error_type": None},
        )
        self.assertFalse(is_valid)
        self.assertEqual(reason, "english_long_span")

    def test_validate_grounded_recommendation_example_allows_english_city_in_summary(self):
        unit = _build_eval_unit()
        response = RecommendationResponse(
            query_id="q001",
            group_id="C_grounded_generation_with_verifier",
            summary="推荐 Anaheim 交通便利的酒店。",
            recommendations=[
                RecommendationItem(
                    hotel_id="hotel_1",
                    hotel_name="Hotel One",
                    reasons=[
                        RecommendationReason(
                            aspect="location_transport",
                            reason_text="位置交通方便。",
                            sentence_id="s_valid",
                        )
                    ],
                )
            ],
            unsupported_notice="",
            schema_valid=True,
            raw_response="{}",
        )
        verification = CitationVerificationResult(
            query_id="q001",
            group_id="C_grounded_generation_with_verifier",
            citation_precision=1.0,
            invalid_sentence_ids=[],
            out_of_pack_sentence_ids=[],
            retry_triggered=False,
            fallback_to_honest_notice=False,
        )
        is_valid, reason = generation_mod.validate_grounded_recommendation_example(
            unit,
            response,
            verification,
            {"response_error_type": None},
        )
        self.assertTrue(is_valid)
        self.assertEqual(reason, "ok")

    def test_validate_grounded_recommendation_example_rejects_zero_rec_from_unsupported(self):
        unit = _build_eval_unit().model_copy(update={"unsupported_requests": ["budget"]})
        response = RecommendationResponse(
            query_id="q001",
            group_id="C_grounded_generation_with_verifier",
            summary="预算条件无法满足。",
            recommendations=[],
            unsupported_notice="预算条件当前不支持，无法给出 grounded 推荐。",
            schema_valid=True,
            raw_response="{}",
        )
        verification = CitationVerificationResult(
            query_id="q001",
            group_id="C_grounded_generation_with_verifier",
            citation_precision=0.0,
            invalid_sentence_ids=[],
            out_of_pack_sentence_ids=[],
            retry_triggered=False,
            fallback_to_honest_notice=False,
        )
        is_valid, reason = generation_mod.validate_grounded_recommendation_example(
            unit,
            response,
            verification,
            {"response_error_type": None},
        )
        self.assertFalse(is_valid)
        self.assertEqual(reason, "unsupported_driven_abstain")

    def test_rebalance_grounded_train_records_enforces_minimum_grounded_share(self):
        def make_row(record_id, focus_aspects, avoid_aspects, recommendations):
            return {
                "record_id": record_id,
                "task_type": "grounded_recommendation",
                "query_id": record_id,
                "source_asset": "synthetic",
                "input_payload": {
                    "user_preference_gold": {
                        "focus_aspects": focus_aspects,
                        "avoid_aspects": avoid_aspects,
                    },
                    "candidate_hotels": [{"hotel_id": "hotel_1"}, {"hotel_id": "hotel_2"}],
                },
                "target_payload": {
                    "summary": "测试",
                    "recommendations": recommendations,
                    "unsupported_notice": "证据不足" if not recommendations else "",
                },
            }

        rows = [
            make_row("r1", ["quiet_sleep"], ["service"], []),
            make_row("r2", ["quiet_sleep"], [], [{"hotel_id": "hotel_1"}]),
            make_row("r3", ["location_transport"], ["cleanliness"], [{"hotel_id": "hotel_1"}]),
            make_row("r4", ["service"], [], [{"hotel_id": "hotel_1"}]),
        ]
        rebalanced = generation_mod.rebalance_grounded_train_records(rows, base_record_count=12)
        grounded_share = len(rebalanced) / (12 + len(rebalanced))
        self.assertGreaterEqual(grounded_share, 0.4)
        quiet_sleep_share = sum(
            int("quiet_sleep" in generation_mod.classify_grounded_record_slices(row))
            for row in rebalanced
        ) / len(rebalanced)
        self.assertGreaterEqual(quiet_sleep_share, 0.3)

    def test_build_grounded_recommendation_input_payload_is_compact(self):
        unit = _build_eval_unit()
        payload = generation_mod.build_grounded_recommendation_input_payload(unit)
        self.assertEqual(sorted(payload.keys()), [
            "candidate_hotels",
            "evidence_packs",
            "query_id",
            "query_text_zh",
            "unsupported_requests",
            "user_preference_gold",
        ])
        self.assertEqual(sorted(payload["candidate_hotels"][0].keys()), ["hotel_id", "hotel_name"])
        self.assertEqual(
            sorted(payload["evidence_packs"][0].keys()),
            ["allowed_sentence_ids", "evidence_by_aspect", "hotel_id"],
        )

    def test_sanitize_grounded_recommendation_response_for_training_removes_null_and_missing_evidence_reasons(self):
        unit = _build_eval_unit().model_copy(
            update={
                "user_preference_gold": _build_eval_unit().user_preference_gold.model_copy(
                    update={"focus_aspects": ["location_transport", "quiet_sleep"]}
                )
            }
        )
        response = RecommendationResponse(
            query_id="q001",
            group_id="C_grounded_generation_with_verifier",
            summary="原始摘要",
            recommendations=[
                RecommendationItem(
                    hotel_id="hotel_1",
                    hotel_name="Hotel One",
                    reasons=[
                        RecommendationReason(
                            aspect="location_transport",
                            reason_text="位置交通方便。",
                            sentence_id="s_valid",
                        ),
                        RecommendationReason(
                            aspect="quiet_sleep",
                            reason_text="无直接证据支持安静睡眠。",
                            sentence_id=None,
                        ),
                    ],
                )
            ],
            unsupported_notice="",
            schema_valid=True,
            raw_response="{}",
        )

        sanitized = generation_mod.sanitize_grounded_recommendation_response_for_training(unit, response)

        self.assertTrue(sanitized.schema_valid)
        self.assertEqual(len(sanitized.recommendations), 1)
        self.assertEqual(len(sanitized.recommendations[0].reasons), 1)
        self.assertEqual(sanitized.recommendations[0].reasons[0].aspect, "location_transport")
        self.assertIn("仅保留可验证理由", sanitized.unsupported_notice)
        self.assertNotIn("无直接证据支持", json.dumps(sanitized.model_dump(), ensure_ascii=False))

    def test_build_e10_v3_judged_grounded_source_rows_excludes_official_and_unsupported_queries(self):
        rows = generation_mod.build_e10_v3_judged_grounded_source_rows()
        official_ids = set(generation_mod.load_json(generation_mod.E9_QUERY_IDS_PATH))
        self.assertTrue(rows)
        self.assertTrue(all(row["query_row"]["query_id"] not in official_ids for row in rows))
        self.assertTrue(all(not row["slot_row"]["unsupported_requests"] for row in rows))
        self.assertTrue(all(row["slot_row"]["city"] for row in rows))

    def test_build_e10_v3_synthetic_grounded_source_rows_only_uses_train_split_cities(self):
        review_df = pd.DataFrame(
            {
                "city": ["Anaheim", "Seattle", "Boston"],
                "state": ["CA", "WA", "MA"],
            }
        )
        split_hotel_lookup = {
            "train": {
                "Anaheim": ["h1", "h2"],
                "Seattle": ["h3"],
            },
            "dev": {
                "Boston": ["h4"],
            },
        }
        rows = generation_mod.build_e10_v3_synthetic_grounded_source_rows(review_df, split_hotel_lookup)
        self.assertTrue(rows)
        self.assertTrue(all(row["source_type"] == "synthetic" for row in rows))
        self.assertTrue(all(row["slot_row"]["city"] in {"Anaheim", "Seattle"} for row in rows))
        self.assertFalse(any(row["slot_row"]["city"] == "Boston" for row in rows))

    def test_classify_grounded_record_slices_v3_marks_partial_support_and_boundary(self):
        row = {
            "record_id": "r1",
            "task_type": "grounded_recommendation",
            "query_id": "v3syn_test",
            "source_asset": "synthetic_grounded_recommendation_v3::multi_hotel_pack_boundary",
            "input_payload": {
                "unsupported_requests": [],
                "user_preference_gold": {
                    "focus_aspects": ["quiet_sleep", "value"],
                    "avoid_aspects": ["service"],
                },
            },
            "target_payload": {
                "summary": "测试",
                "unsupported_notice": "部分方面缺乏直接证据，以下仅保留可验证理由。",
                "recommendations": [
                    {
                        "hotel_id": "hotel_1",
                        "hotel_name": "Hotel One",
                        "reasons": [
                            {
                                "aspect": "quiet_sleep",
                                "reason_text": "安静。",
                                "sentence_id": "s1",
                            }
                        ],
                    }
                ],
            },
        }
        slices = generation_mod.classify_grounded_record_slices_v3(row)
        self.assertIn("quiet_sleep", slices)
        self.assertIn("focus_avoid", slices)
        self.assertIn("multi_hotel_pack_boundary", slices)
        self.assertIn("partial_support_keep_recommendation", slices)

    def test_classify_grounded_record_slices_v3_does_not_treat_focus_avoid_as_partial_support_by_default(self):
        row = {
            "record_id": "r2",
            "task_type": "grounded_recommendation",
            "query_id": "v3syn_focus_avoid",
            "source_asset": "synthetic_grounded_recommendation_v3::partial_support_keep_recommendation",
            "input_payload": {
                "unsupported_requests": [],
                "user_preference_gold": {
                    "focus_aspects": ["quiet_sleep"],
                    "avoid_aspects": ["value"],
                },
            },
            "target_payload": {
                "summary": "测试",
                "unsupported_notice": "",
                "recommendations": [
                    {
                        "hotel_id": "hotel_1",
                        "hotel_name": "Hotel One",
                        "reasons": [
                            {
                                "aspect": "quiet_sleep",
                                "reason_text": "安静。",
                                "sentence_id": "s1",
                            }
                        ],
                    }
                ],
            },
        }
        slices = generation_mod.classify_grounded_record_slices_v3(row)
        self.assertIn("focus_avoid", slices)
        self.assertNotIn("partial_support_keep_recommendation", slices)

    def test_rebalance_grounded_train_records_v3_preserves_final_floor_and_zero_cap(self):
        def make_row(
            record_id: str,
            *,
            focus_aspects: list[str],
            avoid_aspects: list[str],
            reasons: list[dict[str, str | None]],
            source_asset: str,
            unsupported_notice: str = "",
        ) -> dict[str, Any]:
            return {
                "record_id": record_id,
                "task_type": "grounded_recommendation",
                "query_id": record_id,
                "source_asset": source_asset,
                "input_payload": {
                    "unsupported_requests": [],
                    "user_preference_gold": {
                        "focus_aspects": focus_aspects,
                        "avoid_aspects": avoid_aspects,
                    },
                },
                "target_payload": {
                    "summary": "测试",
                    "unsupported_notice": unsupported_notice,
                    "recommendations": (
                        []
                        if not reasons
                        else [
                            {
                                "hotel_id": "hotel_1",
                                "hotel_name": "Hotel One",
                                "reasons": reasons,
                            }
                        ]
                    ),
                },
            }

        rows = [
            make_row(
                "quiet",
                focus_aspects=["quiet_sleep"],
                avoid_aspects=[],
                reasons=[{"aspect": "quiet_sleep", "reason_text": "安静。", "sentence_id": "s1"}],
                source_asset="judged_grounded_recommendation_v3::judged_grounded",
            ),
            make_row(
                "focus_avoid",
                focus_aspects=["service"],
                avoid_aspects=["value"],
                reasons=[{"aspect": "service", "reason_text": "服务好。", "sentence_id": "s2"}],
                source_asset="judged_grounded_recommendation_v3::judged_grounded",
            ),
            make_row(
                "partial",
                focus_aspects=["quiet_sleep", "value"],
                avoid_aspects=[],
                reasons=[{"aspect": "quiet_sleep", "reason_text": "安静。", "sentence_id": "s3"}],
                source_asset="synthetic_grounded_recommendation_v3::partial_support_keep_recommendation",
            ),
            make_row(
                "boundary",
                focus_aspects=["service", "quiet_sleep", "location_transport"],
                avoid_aspects=[],
                reasons=[
                    {"aspect": "service", "reason_text": "服务好。", "sentence_id": "s4"},
                    {"aspect": "quiet_sleep", "reason_text": "安静。", "sentence_id": "s5"},
                ],
                source_asset="synthetic_grounded_recommendation_v3::multi_hotel_pack_boundary",
            ),
            make_row(
                "generic",
                focus_aspects=["cleanliness"],
                avoid_aspects=[],
                reasons=[{"aspect": "cleanliness", "reason_text": "干净。", "sentence_id": "s6"}],
                source_asset="judged_grounded_recommendation_v3::judged_grounded",
            ),
            make_row(
                "zero_gap",
                focus_aspects=["quiet_sleep"],
                avoid_aspects=[],
                reasons=[],
                source_asset="judged_grounded_recommendation_v3::judged_grounded",
                unsupported_notice="当前证据不足，暂不返回酒店推荐。",
            ),
        ]

        rebalanced = generation_mod.rebalance_grounded_train_records_v3(rows, base_record_count=30)
        grounded_share = len(rebalanced) / (30 + len(rebalanced))
        self.assertGreaterEqual(grounded_share, 0.4)

        def share(slice_name: str) -> float:
            return sum(
                int(slice_name in generation_mod.classify_grounded_record_slices_v3(row))
                for row in rebalanced
            ) / len(rebalanced)

        self.assertGreaterEqual(share("quiet_sleep"), 0.30)
        self.assertGreaterEqual(share("focus_avoid"), 0.30)
        self.assertGreaterEqual(share("partial_support_keep_recommendation"), 0.20)
        self.assertGreaterEqual(share("multi_hotel_pack_boundary"), 0.15)
        self.assertLessEqual(share("zero_recommendation_evidence_gap"), 0.10)

    def test_build_grounded_manifest_report_v3_includes_share_fields(self):
        row = {
            "record_id": "r1",
            "task_type": "grounded_recommendation",
            "query_id": "v3syn_test",
            "source_asset": "synthetic_grounded_recommendation_v3::multi_hotel_pack_boundary",
            "input_payload": {
                "unsupported_requests": [],
                "user_preference_gold": {
                    "focus_aspects": ["quiet_sleep"],
                    "avoid_aspects": [],
                },
            },
            "target_payload": {
                "summary": "测试",
                "unsupported_notice": "",
                "recommendations": [
                    {
                        "hotel_id": "hotel_1",
                        "hotel_name": "Hotel One",
                        "reasons": [
                            {
                                "aspect": "quiet_sleep",
                                "reason_text": "安静。",
                                "sentence_id": "s1",
                            }
                        ],
                    }
                ],
            },
        }
        report = generation_mod.build_grounded_manifest_report_v3(
            source_rows=[
                {
                    "query_row": {"query_id": "v3syn_test"},
                    "slot_row": {"city": "Anaheim"},
                    "source_type": "synthetic",
                    "template_kind": "multi_hotel_pack_boundary",
                }
            ],
            train_base_records=[],
            dev_base_records=[],
            train_grounded_records_raw=[row],
            dev_grounded_records_raw=[row],
            train_grounded_records_final=[row],
            dropped_reason_counts={"train:none": 1},
        )
        self.assertEqual(report["source_type_distribution"]["synthetic"], 1)
        self.assertIn("source_type_share", report)
        self.assertIn("train_grounded_slice_share", report)
        self.assertIn("train_grounded_source_share", report)


if __name__ == "__main__":
    unittest.main()

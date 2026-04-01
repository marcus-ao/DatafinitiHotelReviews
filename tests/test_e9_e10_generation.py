import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()

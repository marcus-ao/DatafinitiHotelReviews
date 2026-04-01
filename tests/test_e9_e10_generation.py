import json
import unittest

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


if __name__ == "__main__":
    unittest.main()

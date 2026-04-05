import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from scripts.evaluation import evaluate_e3_e5_behavior as behavior_mod
from scripts.evaluation import evaluate_e6_e8_retrieval as retrieval_mod
from scripts.shared.experiment_schemas import GenerationEvalUnit


class RetrievalGAssetsTestCase(unittest.TestCase):
    def test_build_g_eval_query_id_payload_has_expected_composition(self):
        judged_queries = retrieval_mod.load_jsonl(
            Path(__file__).resolve().parents[1] / "experiments/assets/judged_queries.jsonl"
        )
        payload = retrieval_mod.build_g_eval_query_id_payload(judged_queries)

        self.assertEqual(len(payload["query_ids"]), 70)
        self.assertEqual(len(payload["core_query_ids"]), 40)
        self.assertEqual(len(payload["robustness_query_ids"]), 30)
        self.assertEqual(
            payload["query_type_counts"],
            {
                "single_aspect": 10,
                "multi_aspect": 10,
                "focus_and_avoid": 10,
                "multi_aspect_strong": 10,
                "unsupported_budget": 10,
                "unsupported_distance": 10,
                "unsupported_heavy": 10,
            },
        )
        self.assertEqual(payload["excluded_query_types"], ["conflict", "missing_city"])
        excluded_ids = {f"q{n:03d}" for n in range(51, 57)} | {f"q{n:03d}" for n in range(57, 67)}
        self.assertTrue(excluded_ids.isdisjoint(set(payload["query_ids"])))

    def test_load_g_eval_query_ids_matches_checked_in_asset(self):
        query_ids = retrieval_mod.load_g_eval_query_ids(
            Path(__file__).resolve().parents[1] / "experiments/assets/g_eval_query_ids_70.json"
        )
        self.assertEqual(len(query_ids), 70)
        self.assertEqual(query_ids[:4], ["q001", "q006", "q011", "q016"])
        self.assertEqual(query_ids[-3:], ["q074", "q075", "q076"])

    def test_load_g_eval_query_ids_rejects_invalid_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            invalid_path = Path(tmp_dir) / "bad_query_ids.json"
            invalid_path.write_text(
                json.dumps(
                    {
                        "query_ids": ["q001", "q001"],
                        "query_type_counts": {"single_aspect": 2},
                        "excluded_query_types": ["conflict"],
                        "query_type_order": ["single_aspect"],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                retrieval_mod.load_g_eval_query_ids(invalid_path)

    def test_build_retrieval_summary_row_contains_unified_six_metrics(self):
        summary = retrieval_mod.build_retrieval_summary_row(
            group_id="aspect_main_no_rerank",
            query_count=40,
            target_unit_count=80,
            latencies=[10.0, 20.0],
            metric_rows=[
                {
                    "aspect_recall_at_5": 1.0,
                    "ndcg_at_5": 0.5,
                    "precision_at_5": 0.4,
                    "mrr_at_5": 0.5,
                    "evidence_diversity_at_5": 0.6,
                },
                {
                    "aspect_recall_at_5": 0.0,
                    "ndcg_at_5": 0.7,
                    "precision_at_5": 0.2,
                    "mrr_at_5": 0.25,
                    "evidence_diversity_at_5": 0.8,
                },
            ],
            config_hash="cfg123",
        )
        self.assertEqual(summary["group_id"], "aspect_main_no_rerank")
        self.assertEqual(summary["query_count"], 40)
        self.assertEqual(summary["target_unit_count"], 80)
        self.assertEqual(summary["avg_latency_ms"], 15.0)
        self.assertEqual(summary["aspect_recall_at_5"], 0.5)
        self.assertEqual(summary["ndcg_at_5"], 0.6)
        self.assertEqual(summary["precision_at_5"], 0.3)
        self.assertEqual(summary["mrr_at_5"], 0.375)
        self.assertEqual(summary["evidence_diversity_at_5"], 0.7)

    def test_generation_unit_from_retrieval_assets_matches_schema(self):
        candidate_hotels = [
            retrieval_mod.HotelCandidate(
                hotel_id="hotel_1",
                hotel_name="Hotel One",
                score_total=1.2,
                score_breakdown={"focus_service": 1.2},
            )
        ]
        evidence_packs = [
            retrieval_mod.rows_to_evidence_pack(
                hotel_id="hotel_1",
                query_en="hotel in Anaheim with helpful and reliable service",
                rows=[
                    {
                        "sentence_id": "s001",
                        "sentence_text": "Staff were very helpful.",
                        "sentence_aspect": "service",
                        "sentence_sentiment": "positive",
                        "review_date": "2024-01-01",
                        "score_dense": 0.1,
                        "score_rerank": None,
                    }
                ],
                retrieval_trace={"mode": "plain_city_test_rerank"},
            )
        ]
        unit = retrieval_mod.generation_unit_from_retrieval_assets(
            query_row={
                "query_id": "q001",
                "query_text_zh": "请推荐服务好的酒店。",
                "query_type": "single_aspect",
            },
            slot_row={
                "city": "Anaheim",
                "state": "CA",
                "hotel_category": None,
                "focus_aspects": ["service"],
                "avoid_aspects": [],
                "unsupported_requests": [],
                "query_en": "hotel in Anaheim with helpful and reliable service",
            },
            candidate_hotels=candidate_hotels,
            evidence_packs=evidence_packs,
            retrieval_mode="plain_city_test_rerank",
            candidate_policy="G_plain_retrieval_top5",
            config_hash="cfg123",
        )
        self.assertIsInstance(unit, GenerationEvalUnit)
        self.assertEqual(unit.retrieval_mode, "plain_city_test_rerank")
        self.assertEqual(unit.candidate_policy, "G_plain_retrieval_top5")
        self.assertEqual(unit.evidence_packs[0].evidence_by_aspect["service"][0].sentence_id, "s001")

    def test_freeze_g_retrieval_assets_outputs_generation_eval_units(self):
        fake_review_df = pd.DataFrame(
            [
                {"hotel_id": "hotel_1", "city": "Anaheim", "hotel_name": "Hotel One", "review_id": "r1", "rating": 5},
                {"hotel_id": "hotel_2", "city": "Anaheim", "hotel_name": "Hotel Two", "review_id": "r2", "rating": 4},
            ]
        )
        fake_profile_df = pd.DataFrame(
            [
                {
                    "hotel_id": "hotel_1",
                    "aspect": "service",
                    "final_aspect_score": 1.0,
                    "recency_weighted_pos": 1.0,
                    "recency_weighted_neg": 0.0,
                },
                {
                    "hotel_id": "hotel_2",
                    "aspect": "service",
                    "final_aspect_score": 0.7,
                    "recency_weighted_pos": 0.7,
                    "recency_weighted_neg": 0.0,
                },
            ]
        )
        fake_evidence_df = pd.DataFrame(
            [
                {
                    "sentence_id": "r1_s001",
                    "sentence_text": "Staff were excellent.",
                    "aspect": "service",
                    "sentiment": "positive",
                    "review_date": "2024-01-01",
                }
            ]
        )
        fake_units = ["q001"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "units.jsonl"
            fake_chromadb_module = SimpleNamespace(PersistentClient=mock.Mock())
            fake_e2_module = SimpleNamespace(
                build_hotel_summary=mock.Mock(return_value=fake_review_df[["hotel_id", "city", "hotel_name", "review_id", "rating"]].assign(avg_rating=[5.0, 4.0], review_count=[1, 1])),
                build_profile_tables=mock.Mock(return_value=(pd.DataFrame({"service": [1.0, 0.7]}, index=["hotel_1", "hotel_2"]), pd.DataFrame({"service": [1.0, 0.7]}, index=["hotel_1", "hotel_2"]))),
                candidate_rank=mock.Mock(return_value=pd.DataFrame([
                    {"hotel_id": "hotel_1", "hotel_name": "Hotel One", "score_total": 1.0, "score_breakdown": {"focus_service": 1.0}},
                    {"hotel_id": "hotel_2", "hotel_name": "Hotel Two", "score_total": 0.7, "score_breakdown": {"focus_service": 0.7}},
                ])),
            )
            with mock.patch.object(retrieval_mod, "load_config", return_value={
                "embedding": {
                    "chroma_persist_dir": "unused",
                    "chroma_collection": "unused",
                    "model": "fake-model",
                    "normalize": True,
                },
                "reranker": {
                    "model": "fake-reranker",
                    "top_k_before_rerank": 3,
                    "top_k_after_rerank": 2,
                },
            }), \
                mock.patch.object(retrieval_mod, "load_json", return_value={"meta": {"config_hash": "splitcfg"}, "splits": {"test": ["hotel_1", "hotel_2"]}}), \
                mock.patch.object(retrieval_mod, "load_jsonl", return_value=[{"query_id": "q001", "query_text_zh": "请推荐服务好的酒店。", "query_type": "single_aspect"}]), \
                mock.patch.object(retrieval_mod, "load_slot_gold_lookup", return_value={
                    "q001": {
                        "city": "Anaheim",
                        "state": "CA",
                        "hotel_category": None,
                        "focus_aspects": ["service", "cleanliness"],
                        "avoid_aspects": ["value"],
                        "unsupported_requests": [],
                        "query_en": "hotel in Anaheim with helpful and reliable service and clean rooms avoiding poor value",
                    }
                }), \
                mock.patch.object(retrieval_mod, "load_clarify_gold_lookup", return_value={"q001": {"clarify_needed": False}}), \
                mock.patch.object(retrieval_mod, "load_g_eval_query_ids", return_value=fake_units), \
                mock.patch.object(retrieval_mod.pd, "read_pickle", side_effect=[fake_review_df, fake_profile_df, fake_evidence_df]), \
                mock.patch.object(retrieval_mod, "build_city_test_hotels", return_value={"Anaheim": [{"hotel_id": "hotel_1", "hotel_name": "Hotel One"}, {"hotel_id": "hotel_2", "hotel_name": "Hotel Two"}]}), \
                mock.patch.object(retrieval_mod, "build_evidence_lookup", return_value={
                    "r1_s001": {
                        "sentence_text": "Staff were excellent.",
                        "aspect": "service",
                        "sentiment": "positive",
                        "review_date": "2024-01-01",
                        "review_id": "r1",
                    }
                }), \
                mock.patch.object(retrieval_mod, "warm_up_models", return_value=None), \
                mock.patch.object(retrieval_mod, "retrieve_official_mode", side_effect=[
                    {
                        "rows": [
                            {
                                "sentence_id": "r1_s001",
                                "sentence_text": "Staff were excellent.",
                                "sentence_aspect": "service",
                                "sentence_sentiment": "positive",
                                "review_date": "2024-01-01",
                                "score_dense": 0.1,
                                "score_rerank": 0.2,
                            }
                        ],
                        "retrieval_trace": {"mode": "plain_city_test_rerank", "aspect": "service"},
                    },
                    {
                        "rows": [
                            {
                                "sentence_id": "r1_s002",
                                "sentence_text": "Room was spotless.",
                                "sentence_aspect": "cleanliness",
                                "sentence_sentiment": "positive",
                                "review_date": "2024-01-02",
                                "score_dense": 0.11,
                                "score_rerank": 0.21,
                            }
                        ],
                        "retrieval_trace": {"mode": "plain_city_test_rerank", "aspect": "cleanliness"},
                    },
                    {
                        "rows": [
                            {
                                "sentence_id": "r1_s003",
                                "sentence_text": "Price felt high.",
                                "sentence_aspect": "value",
                                "sentence_sentiment": "negative",
                                "review_date": "2024-01-03",
                                "score_dense": 0.12,
                                "score_rerank": 0.22,
                            }
                        ],
                        "retrieval_trace": {"mode": "plain_city_test_rerank", "aspect": "value"},
                    },
                    {
                        "rows": [
                            {
                                "sentence_id": "r2_s001",
                                "sentence_text": "Staff were polite.",
                                "sentence_aspect": "service",
                                "sentence_sentiment": "positive",
                                "review_date": "2024-01-04",
                                "score_dense": 0.13,
                                "score_rerank": 0.23,
                            }
                        ],
                        "retrieval_trace": {"mode": "plain_city_test_rerank", "aspect": "service"},
                    },
                    {
                        "rows": [
                            {
                                "sentence_id": "r2_s002",
                                "sentence_text": "Clean enough.",
                                "sentence_aspect": "cleanliness",
                                "sentence_sentiment": "neutral",
                                "review_date": "2024-01-05",
                                "score_dense": 0.14,
                                "score_rerank": 0.24,
                            }
                        ],
                        "retrieval_trace": {"mode": "plain_city_test_rerank", "aspect": "cleanliness"},
                    },
                    {
                        "rows": [
                            {
                                "sentence_id": "r2_s003",
                                "sentence_text": "A bit expensive.",
                                "sentence_aspect": "value",
                                "sentence_sentiment": "negative",
                                "review_date": "2024-01-06",
                                "score_dense": 0.15,
                                "score_rerank": 0.25,
                            }
                        ],
                        "retrieval_trace": {"mode": "plain_city_test_rerank", "aspect": "value"},
                    },
                ]), \
                mock.patch("scripts.evaluation.evaluate_e6_e8_retrieval.SentenceTransformer"), \
                mock.patch("scripts.evaluation.evaluate_e6_e8_retrieval.CrossEncoder"), \
                mock.patch.dict(sys.modules, {
                    "chromadb": fake_chromadb_module,
                    "scripts.evaluation.evaluate_e2_candidate_selection": fake_e2_module,
                }):
                path = retrieval_mod.freeze_g_plain_retrieval_assets(output_path=output_path, query_ids=fake_units)

            self.assertEqual(path, output_path)
            rows = [GenerationEvalUnit.model_validate(json.loads(line)) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].retrieval_mode, "plain_city_test_rerank")
            self.assertEqual(rows[0].candidate_policy, "G_plain_retrieval_top5")
            self.assertEqual(rows[0].candidate_hotels[0].hotel_id, "hotel_1")
            self.assertIn("service", rows[0].evidence_packs[0].evidence_by_aspect)
            self.assertIn("cleanliness", rows[0].evidence_packs[0].evidence_by_aspect)
            self.assertIn("value", rows[0].evidence_packs[0].evidence_by_aspect)
            self.assertEqual(rows[0].evidence_packs[0].retrieval_trace["aspect_roles"], {
                "service": ["focus"],
                "cleanliness": ["focus"],
                "value": ["avoid"],
            })

    def test_freeze_g_retrieval_assets_rejects_empty_candidate_set(self):
        fake_review_df = pd.DataFrame(
            [{"hotel_id": "hotel_1", "city": "Anaheim", "hotel_name": "Hotel One", "review_id": "r1", "rating": 5}]
        )
        fake_profile_df = pd.DataFrame(
            [{"hotel_id": "hotel_1", "aspect": "service", "final_aspect_score": 1.0, "recency_weighted_pos": 1.0, "recency_weighted_neg": 0.0}]
        )
        fake_evidence_df = pd.DataFrame([{"sentence_id": "r1_s001", "sentence_text": "Staff were excellent.", "aspect": "service", "sentiment": "positive", "review_date": "2024-01-01"}])

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "units.jsonl"
            fake_chromadb_module = SimpleNamespace(PersistentClient=mock.Mock())
            fake_e2_module = SimpleNamespace(
                build_hotel_summary=mock.Mock(return_value=fake_review_df[["hotel_id", "city", "hotel_name", "review_id", "rating"]].assign(avg_rating=[5.0], review_count=[1])),
                build_profile_tables=mock.Mock(return_value=(pd.DataFrame({"service": [1.0]}, index=["hotel_1"]), pd.DataFrame({"service": [1.0]}, index=["hotel_1"]))),
                candidate_rank=mock.Mock(return_value=pd.DataFrame(columns=["hotel_id", "hotel_name", "score_total", "score_breakdown"])),
            )
            with mock.patch.object(retrieval_mod, "load_config", return_value={
                "embedding": {"chroma_persist_dir": "unused", "chroma_collection": "unused", "model": "fake-model", "normalize": True},
                "reranker": {"model": "fake-reranker", "top_k_before_rerank": 3, "top_k_after_rerank": 2},
            }), \
                mock.patch.object(retrieval_mod, "load_json", return_value={"meta": {"config_hash": "splitcfg"}, "splits": {"test": ["hotel_1"]}}), \
                mock.patch.object(retrieval_mod, "load_jsonl", return_value=[{"query_id": "q001", "query_text_zh": "请推荐服务好的酒店。", "query_type": "single_aspect"}]), \
                mock.patch.object(retrieval_mod, "load_slot_gold_lookup", return_value={"q001": {"city": "Anaheim", "state": "CA", "hotel_category": None, "focus_aspects": ["service"], "avoid_aspects": [], "unsupported_requests": [], "query_en": "hotel in Anaheim with helpful and reliable service"}}), \
                mock.patch.object(retrieval_mod, "load_clarify_gold_lookup", return_value={"q001": {"clarify_needed": False}}), \
                mock.patch.object(retrieval_mod, "load_g_eval_query_ids", return_value=["q001"]), \
                mock.patch.object(retrieval_mod.pd, "read_pickle", side_effect=[fake_review_df, fake_profile_df, fake_evidence_df]), \
                mock.patch.object(retrieval_mod, "build_city_test_hotels", return_value={"Anaheim": [{"hotel_id": "hotel_1", "hotel_name": "Hotel One"}]}), \
                mock.patch.object(retrieval_mod, "build_evidence_lookup", return_value={}), \
                mock.patch.object(retrieval_mod, "warm_up_models", return_value=None), \
                mock.patch("scripts.evaluation.evaluate_e6_e8_retrieval.SentenceTransformer"), \
                mock.patch("scripts.evaluation.evaluate_e6_e8_retrieval.CrossEncoder"), \
                mock.patch.dict(sys.modules, {"chromadb": fake_chromadb_module, "scripts.evaluation.evaluate_e2_candidate_selection": fake_e2_module}):
                with self.assertRaises(ValueError):
                    retrieval_mod.freeze_g_plain_retrieval_assets(output_path=output_path, query_ids=["q001"])

    def test_validate_g_retrieval_assets_accepts_valid_units(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            asset_path = Path(tmp_dir) / "g_plain_units.jsonl"
            unit = retrieval_mod.generation_unit_from_retrieval_assets(
                query_row={"query_id": "q001", "query_text_zh": "请推荐服务好的酒店。", "query_type": "single_aspect"},
                slot_row={
                    "city": "Anaheim",
                    "state": "CA",
                    "hotel_category": None,
                    "focus_aspects": ["service"],
                    "avoid_aspects": [],
                    "unsupported_requests": [],
                    "query_en": "hotel in Anaheim with helpful and reliable service",
                },
                candidate_hotels=[retrieval_mod.HotelCandidate(hotel_id="hotel_1", hotel_name="Hotel One", score_total=1.0, score_breakdown={"focus_service": 1.0})],
                evidence_packs=[
                    retrieval_mod.EvidencePack(
                        hotel_id="hotel_1",
                        query_en="hotel in Anaheim with helpful and reliable service",
                        evidence_by_aspect={
                            "service": [
                                retrieval_mod.SentenceCandidate(
                                    sentence_id="s001",
                                    sentence_text="Great service.",
                                    aspect="service",
                                    sentiment="positive",
                                    review_date="2024-01-01",
                                    score_dense=0.1,
                                    score_rerank=None,
                                )
                            ]
                        },
                        all_sentence_ids=["s001"],
                        retrieval_trace={"mode": "plain_city_test_rerank"},
                    )
                ],
                retrieval_mode="plain_city_test_rerank",
                candidate_policy="G_plain_retrieval_top5",
                config_hash="cfg123",
            )
            asset_path.write_text(json.dumps(unit.model_dump(), ensure_ascii=False) + "\n", encoding="utf-8")

            summary = retrieval_mod.validate_g_retrieval_assets(
                asset_path,
                expected_retrieval_mode="plain_city_test_rerank",
                expected_candidate_policy="G_plain_retrieval_top5",
                expected_query_ids=["q001"],
            )
            self.assertEqual(summary["query_count"], 1)
            self.assertEqual(summary["retrieval_mode"], "plain_city_test_rerank")

    def test_validate_g_retrieval_assets_rejects_candidate_evidence_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            asset_path = Path(tmp_dir) / "bad_units.jsonl"
            bad_payload = retrieval_mod.generation_unit_from_retrieval_assets(
                query_row={"query_id": "q001", "query_text_zh": "请推荐服务好的酒店。", "query_type": "single_aspect"},
                slot_row={
                    "city": "Anaheim",
                    "state": "CA",
                    "hotel_category": None,
                    "focus_aspects": ["service"],
                    "avoid_aspects": [],
                    "unsupported_requests": [],
                    "query_en": "hotel in Anaheim with helpful and reliable service",
                },
                candidate_hotels=[retrieval_mod.HotelCandidate(hotel_id="hotel_1", hotel_name="Hotel One", score_total=1.0, score_breakdown={"focus_service": 1.0})],
                evidence_packs=[
                    retrieval_mod.EvidencePack(
                        hotel_id="hotel_2",
                        query_en="hotel in Anaheim with helpful and reliable service",
                        evidence_by_aspect={
                            "service": [
                                retrieval_mod.SentenceCandidate(
                                    sentence_id="s001",
                                    sentence_text="Great service.",
                                    aspect="service",
                                    sentiment="positive",
                                    review_date="2024-01-01",
                                    score_dense=0.1,
                                    score_rerank=None,
                                )
                            ]
                        },
                        all_sentence_ids=["s001"],
                        retrieval_trace={"mode": "plain_city_test_rerank"},
                    )
                ],
                retrieval_mode="plain_city_test_rerank",
                candidate_policy="G_plain_retrieval_top5",
                config_hash="cfg123",
            ).model_dump()
            asset_path.write_text(json.dumps(bad_payload, ensure_ascii=False) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                retrieval_mod.validate_g_retrieval_assets(
                    asset_path,
                    expected_retrieval_mode="plain_city_test_rerank",
                    expected_candidate_policy="G_plain_retrieval_top5",
                    expected_query_ids=["q001"],
                )

    def test_e5_summary_builder_exposes_six_metrics(self):
        summary = retrieval_mod.build_retrieval_summary_row(
            group_id=behavior_mod.E5_GROUPS[0],
            query_count=10,
            target_unit_count=20,
            latencies=[8.0, 12.0],
            metric_rows=[
                {
                    "aspect_recall_at_5": 0.8,
                    "ndcg_at_5": 0.7,
                    "precision_at_5": 0.6,
                    "mrr_at_5": 0.5,
                    "evidence_diversity_at_5": 0.4,
                }
            ],
            config_hash="cfg",
        )
        self.assertEqual(
            list(summary.keys()),
            [
                "group_id",
                "query_count",
                "target_unit_count",
                "avg_latency_ms",
                "config_hash",
                "aspect_recall_at_5",
                "ndcg_at_5",
                "precision_at_5",
                "mrr_at_5",
                "evidence_diversity_at_5",
            ],
        )

    def test_run_retrieval_eval_raises_clear_error_for_missing_city(self):
        fake_review_df = pd.DataFrame(
            [{"hotel_id": "hotel_1", "city": "Orlando", "hotel_name": "Hotel One", "review_id": "r1", "rating": 5}]
        )
        fake_evidence_df = pd.DataFrame(
            [{"sentence_id": "r1_s001", "sentence_text": "Great staff.", "aspect": "service", "sentiment": "positive", "review_date": "2024-01-01", "review_id": "r1"}]
        )
        target_units = [
            {
                "query_id": "q001",
                "city": "Anaheim",
                "query_type": "single_aspect",
                "target_aspect": "service",
                "target_role": "focus",
                "query_text_zh": "请推荐服务好的酒店。",
                "query_en_full": "hotel in Anaheim with helpful and reliable service",
                "query_en_target": "hotel in Anaheim with helpful and reliable service",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.object(retrieval_mod, "load_config", return_value={
                "embedding": {"chroma_persist_dir": "unused", "chroma_collection": "unused", "model": "fake-model", "normalize": True},
                "reranker": {"model": "fake-reranker", "top_k_before_rerank": 3, "top_k_after_rerank": 2},
            }), \
                mock.patch.object(retrieval_mod, "load_json", return_value={"meta": {"config_hash": "splitcfg"}, "splits": {"test": ["hotel_1"]}}), \
                mock.patch.object(retrieval_mod.pd, "read_pickle", side_effect=[fake_review_df, fake_evidence_df]), \
                mock.patch.object(retrieval_mod, "build_city_test_hotels", return_value={"Orlando": [{"hotel_id": "hotel_1", "hotel_name": "Hotel One"}]}), \
                mock.patch.object(retrieval_mod, "build_evidence_lookup", return_value={}), \
                mock.patch.object(retrieval_mod, "build_target_units", return_value=target_units), \
                mock.patch.object(retrieval_mod, "load_qrels_lookup", return_value={("q001", "service", "focus"): {}}), \
                mock.patch.object(retrieval_mod, "load_jsonl", return_value=[{"query_id": "q001"}]), \
                mock.patch.object(retrieval_mod, "warm_up_models", return_value=None), \
                mock.patch("scripts.evaluation.evaluate_e6_e8_retrieval.require_chromadb_client", return_value=mock.Mock(return_value=mock.Mock(get_collection=mock.Mock(return_value=mock.Mock())))), \
                mock.patch("scripts.evaluation.evaluate_e6_e8_retrieval.require_retrieval_backends", return_value=(mock.Mock(), mock.Mock())):
                with self.assertRaises(KeyError) as ctx:
                    retrieval_mod.run_retrieval_eval("E6", output_root=Path(tmp_dir), limit_queries=1)
        self.assertIn("City 'Anaheim'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

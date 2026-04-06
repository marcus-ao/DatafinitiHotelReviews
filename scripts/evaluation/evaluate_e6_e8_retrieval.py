"""Shared retrieval evaluation engine for E6, E7, E8, and G-series retrieval assets."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder as CrossEncoderType
    from sentence_transformers import SentenceTransformer as SentenceTransformerType

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ImportError:  # pragma: no cover - optional runtime dependency for retrieval execution
    CrossEncoder = None  # type: ignore[assignment]
    SentenceTransformer = None  # type: ignore[assignment]
from scripts.shared.experiment_schemas import ASPECT_NAME, RunLogEntry
from scripts.shared.experiment_schemas import EvidencePack, GenerationEvalUnit, HotelCandidate, SentenceCandidate, UserPreference
from scripts.shared.experiment_utils import (
    E6_LABELS_DIR,
    EXPERIMENT_ASSETS_DIR,
    EXPERIMENT_RUNS_DIR,
    G_QRELS_LABELS_DIR,
    ensure_dir,
    load_jsonl,
    stable_hash,
    utc_now_iso,
    write_jsonl,
)
from scripts.shared.project_utils import load_config


ALLOWED_QUERY_TYPES = {
    "single_aspect",
    "multi_aspect",
    "focus_and_avoid",
    "multi_aspect_strong",
}
CORE_QUERY_TYPES = (
    "single_aspect",
    "multi_aspect",
    "focus_and_avoid",
    "multi_aspect_strong",
)
ROBUSTNESS_QUERY_TYPES = (
    "unsupported_budget",
    "unsupported_distance",
    "unsupported_heavy",
)
G_EVAL_QUERY_TYPE_ORDER = CORE_QUERY_TYPES + ROBUSTNESS_QUERY_TYPES
G_EVAL_QUERY_IDS_PATH = EXPERIMENT_ASSETS_DIR / "g_eval_query_ids_68.json"
E5_E8_QUERY_IDS_PATH = EXPERIMENT_ASSETS_DIR / "e5_e8_core_query_ids.json"
G_DECISIVE_EXCLUDED_QUERY_IDS = {"q021", "q024"}
G_PLAIN_RETRIEVAL_UNITS_PATH = EXPERIMENT_ASSETS_DIR / "g_plain_generation_eval_units.jsonl"
G_ASPECT_RETRIEVAL_UNITS_PATH = EXPERIMENT_ASSETS_DIR / "g_aspect_generation_eval_units.jsonl"
G_QRELS_POOL_PATH = G_QRELS_LABELS_DIR / "g_qrels_pool.csv"
G_QRELS_EVIDENCE_PATH = G_QRELS_LABELS_DIR / "g_qrels_evidence.jsonl"
G_QRELS_LOG_PATH = G_QRELS_LABELS_DIR / "g_labeling_log.md"
G_RETRIEVAL_VARIANT_SPECS = {
    "plain": {
        "retrieval_mode": "plain_city_test_rerank",
        "summary_group_id": "plain_retrieval",
        "candidate_policy": "G_plain_retrieval_top5",
    },
    "aspect": {
        "retrieval_mode": "aspect_main_no_rerank",
        "summary_group_id": "aspect_retrieval",
        "candidate_policy": "G_aspect_retrieval_top5",
    },
}

ASPECT_EN = {
    "location_transport": "convenient location and transportation",
    "cleanliness": "clean and hygienic rooms",
    "service": "helpful and reliable service",
    "room_facilities": "comfortable rooms and facilities",
    "quiet_sleep": "quiet rooms for good sleep",
    "value": "good value for money",
}

ASPECT_EN_AVOID = {
    "location_transport": "location or transportation issues",
    "cleanliness": "cleanliness problems",
    "service": "service problems",
    "room_facilities": "room or facility problems",
    "quiet_sleep": "noise or sleep issues",
    "value": "poor value for money",
}

OFFICIAL_MODES = [
    "plain_city_test_rerank",
    "aspect_main_rerank",
    "aspect_main_no_rerank",
    "aspect_main_fallback_rerank",
]

TASK_TO_MODES = {
    "E6": ["plain_city_test_rerank", "aspect_main_rerank"],
    "E7": ["aspect_main_no_rerank", "aspect_main_rerank"],
    "E8": ["aspect_main_rerank", "aspect_main_fallback_rerank"],
}

CANDIDATE_MODE = "city_test_all"
QUERY_SCOPE = sorted(ALLOWED_QUERY_TYPES)
TARGET_SCOPE = "focus_and_avoid"
POOL_TOP_K = 5
FALLBACK_MIN_SENTENCES = 2
FALLBACK_MIN_UNIQUE_REVIEWS = 2


def require_retrieval_backends() -> tuple[Any, Any]:
    if SentenceTransformer is None or CrossEncoder is None:
        raise ImportError(
            "Missing retrieval dependencies. Please install sentence-transformers to run retrieval evaluation/freezing tasks."
        )
    return SentenceTransformer, CrossEncoder


def require_chromadb_client() -> Any:
    try:
        from chromadb import PersistentClient
    except ImportError as exc:  # pragma: no cover - optional runtime dependency for retrieval execution
        raise ImportError(
            "Missing retrieval dependency 'chromadb'. Please install ChromaDB to run retrieval evaluation/freezing tasks."
        ) from exc
    return PersistentClient
def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def review_id_from_sentence_id(sentence_id: str) -> str:
    if "_s" in sentence_id:
        return sentence_id.rsplit("_s", 1)[0]
    return sentence_id.split("_", 1)[0]


def build_query_en_target(city: str, aspect: str, target_role: str) -> str:
    if target_role == "focus":
        return f"hotel in {city} with {ASPECT_EN[aspect]}"
    return f"hotel in {city} avoiding {ASPECT_EN_AVOID[aspect]}"


def load_slot_gold_lookup() -> dict[str, dict[str, Any]]:
    return {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "slot_gold.jsonl")}


def load_clarify_gold_lookup() -> dict[str, dict[str, Any]]:
    return {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "clarify_gold.jsonl")}


def _query_id_sort_key(query_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in str(query_id) if ch.isdigit())
    return (int(digits) if digits else 10**9, str(query_id))


def build_g_eval_query_id_payload(judged_queries: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[str]] = {query_type: [] for query_type in G_EVAL_QUERY_TYPE_ORDER}
    excluded_query_types = {"conflict", "missing_city"}

    for row in judged_queries:
        query_type = row.get("query_type")
        query_id = row.get("query_id")
        if query_type in grouped and query_id and str(query_id) not in G_DECISIVE_EXCLUDED_QUERY_IDS:
            grouped[query_type].append(str(query_id))

    for query_type in grouped:
        grouped[query_type] = sorted(grouped[query_type], key=_query_id_sort_key)

    core_query_ids = [query_id for query_type in CORE_QUERY_TYPES for query_id in grouped[query_type]]
    robustness_query_ids = [query_id for query_type in ROBUSTNESS_QUERY_TYPES for query_id in grouped[query_type]]
    ordered_query_ids = core_query_ids + robustness_query_ids

    expected_counts = {
        "core": 39,
        "robustness": 29,
        "total": 68,
    }
    actual_counts = {
        "core": len(core_query_ids),
        "robustness": len(robustness_query_ids),
        "total": len(ordered_query_ids),
    }
    if actual_counts != expected_counts:
        raise AssertionError(
            f"G-series query set count mismatch: expected {expected_counts}, got {actual_counts}"
        )
    if len(set(ordered_query_ids)) != len(ordered_query_ids):
        raise AssertionError("G-series query ids contain duplicates")

    return {
        "query_ids": ordered_query_ids,
        "query_type_counts": {query_type: len(grouped[query_type]) for query_type in G_EVAL_QUERY_TYPE_ORDER},
        "core_query_ids": core_query_ids,
        "robustness_query_ids": robustness_query_ids,
        "excluded_query_types": sorted(excluded_query_types),
        "excluded_query_ids": sorted(G_DECISIVE_EXCLUDED_QUERY_IDS),
        "query_type_order": list(G_EVAL_QUERY_TYPE_ORDER),
    }


def write_g_eval_query_ids_asset(output_path: Path = G_EVAL_QUERY_IDS_PATH) -> Path:
    payload = build_g_eval_query_id_payload(load_jsonl(EXPERIMENT_ASSETS_DIR / "judged_queries.jsonl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output_path


def build_e5_e8_core_query_id_payload(judged_queries: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[str]] = {query_type: [] for query_type in CORE_QUERY_TYPES}
    excluded_query_types = set(ROBUSTNESS_QUERY_TYPES) | {"conflict", "missing_city"}

    for row in judged_queries:
        query_type = row.get("query_type")
        query_id = row.get("query_id")
        if query_type in grouped and query_id and str(query_id) not in G_DECISIVE_EXCLUDED_QUERY_IDS:
            grouped[query_type].append(str(query_id))

    for query_type in grouped:
        grouped[query_type] = sorted(grouped[query_type], key=_query_id_sort_key)

    ordered_query_ids = [query_id for query_type in CORE_QUERY_TYPES for query_id in grouped[query_type]]
    expected_counts = {query_type: 10 for query_type in CORE_QUERY_TYPES}
    actual_counts = {query_type: len(grouped[query_type]) for query_type in CORE_QUERY_TYPES}
    if actual_counts != expected_counts:
        raise AssertionError(f"E5-E8 core query count mismatch: expected {expected_counts}, got {actual_counts}")
    if len(ordered_query_ids) != 40:
        raise AssertionError(f"E5-E8 core query payload size mismatch: expected 40, got {len(ordered_query_ids)}")
    if len(set(ordered_query_ids)) != len(ordered_query_ids):
        raise AssertionError("E5-E8 core query ids contain duplicates")

    return {
        "query_ids": ordered_query_ids,
        "query_count": 40,
        "query_type_counts": actual_counts,
        "query_type_order": list(CORE_QUERY_TYPES),
        "excluded_query_types": sorted(excluded_query_types),
    }


def write_e5_e8_core_query_ids_asset(output_path: Path = E5_E8_QUERY_IDS_PATH) -> Path:
    payload = build_e5_e8_core_query_id_payload(load_jsonl(EXPERIMENT_ASSETS_DIR / "judged_queries.jsonl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output_path


def load_g_eval_query_ids(path: Path = G_EVAL_QUERY_IDS_PATH) -> list[str]:
    payload = load_json(path)
    query_ids = payload.get("query_ids")
    if not isinstance(query_ids, list) or not all(isinstance(item, str) for item in query_ids):
        raise ValueError(f"Invalid g_eval query id asset: {path}")
    if len(query_ids) != 68:
        raise ValueError(f"Invalid g_eval query id asset size in {path}: expected 68, got {len(query_ids)}")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError(f"Invalid g_eval query id asset with duplicate query ids: {path}")
    expected_order = list(G_EVAL_QUERY_TYPE_ORDER)
    if payload.get("query_type_order") != expected_order:
        raise ValueError(f"Invalid query_type_order in {path}: expected {expected_order}")
    expected_excluded = ["conflict", "missing_city"]
    if payload.get("excluded_query_types") != expected_excluded:
        raise ValueError(f"Invalid excluded_query_types in {path}: expected {expected_excluded}")
    expected_counts = {query_type: 10 for query_type in G_EVAL_QUERY_TYPE_ORDER}
    expected_counts["single_aspect"] = 9
    expected_counts["unsupported_budget"] = 9
    if payload.get("query_type_counts") != expected_counts:
        raise ValueError(f"Invalid query_type_counts in {path}: expected {expected_counts}")
    expected_excluded_query_ids = sorted(G_DECISIVE_EXCLUDED_QUERY_IDS)
    if payload.get("excluded_query_ids") != expected_excluded_query_ids:
        raise ValueError(f"Invalid excluded_query_ids in {path}: expected {expected_excluded_query_ids}")
    return query_ids


def load_e5_e8_core_query_ids(path: Path = E5_E8_QUERY_IDS_PATH) -> list[str]:
    payload = load_json(path)
    query_ids = payload.get("query_ids")
    if not isinstance(query_ids, list) or not all(isinstance(item, str) for item in query_ids):
        raise ValueError(f"Invalid E5-E8 core query id asset: {path}")
    if len(query_ids) != 40:
        raise ValueError(f"Invalid E5-E8 core query id asset size in {path}: expected 40, got {len(query_ids)}")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError(f"Invalid E5-E8 core query id asset with duplicate query ids: {path}")
    if payload.get("query_type_order") != list(CORE_QUERY_TYPES):
        raise ValueError(f"Invalid query_type_order in {path}: expected {list(CORE_QUERY_TYPES)}")
    expected_counts = {query_type: 10 for query_type in CORE_QUERY_TYPES}
    if payload.get("query_type_counts") != expected_counts:
        raise ValueError(f"Invalid query_type_counts in {path}: expected {expected_counts}")
    return query_ids


def build_target_units(limit_queries: int | None = None) -> list[dict[str, Any]]:
    return build_target_units_filtered(limit_queries=limit_queries, query_ids=load_e5_e8_core_query_ids())


def build_g_target_units(limit_queries: int | None = None) -> list[dict[str, Any]]:
    return build_target_units_filtered(
        limit_queries=limit_queries,
        allowed_query_types=set(G_EVAL_QUERY_TYPE_ORDER),
        query_ids=load_g_eval_query_ids(),
    )


def build_target_units_filtered(
    limit_queries: int | None = None,
    *,
    allowed_query_types: set[str] | None = None,
    query_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    judged_queries = load_jsonl(EXPERIMENT_ASSETS_DIR / "judged_queries.jsonl")
    slot_gold = load_slot_gold_lookup()
    clarify_gold = load_clarify_gold_lookup()
    allowed_types = allowed_query_types or ALLOWED_QUERY_TYPES
    allowed_query_id_set = set(query_ids) if query_ids is not None else None
    query_id_order = {query_id: idx for idx, query_id in enumerate(query_ids or [])}

    eligible_queries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in judged_queries:
        if allowed_query_id_set is not None and row["query_id"] not in allowed_query_id_set:
            continue
        slot = slot_gold[row["query_id"]]
        clarify = clarify_gold[row["query_id"]]
        if clarify["clarify_needed"]:
            continue
        if not slot["city"] or not (slot["focus_aspects"] or slot["avoid_aspects"]):
            continue
        if row["query_type"] not in allowed_types:
            continue
        eligible_queries.append((row, slot))

    if query_ids is not None:
        eligible_queries.sort(key=lambda item: query_id_order[item[0]["query_id"]])

    if limit_queries is not None:
        eligible_queries = eligible_queries[:limit_queries]

    units: list[dict[str, Any]] = []
    for query_row, slot in eligible_queries:
        for aspect in slot["focus_aspects"]:
            units.append(
                {
                    "unit_id": f"{query_row['query_id']}__focus__{aspect}",
                    "query_id": query_row["query_id"],
                    "city": slot["city"],
                    "query_type": query_row["query_type"],
                    "target_aspect": aspect,
                    "target_role": "focus",
                    "query_text_zh": query_row["query_text_zh"],
                    "query_en_full": slot["query_en"],
                    "query_en_target": build_query_en_target(slot["city"], aspect, "focus"),
                }
            )
        for aspect in slot["avoid_aspects"]:
            units.append(
                {
                    "unit_id": f"{query_row['query_id']}__avoid__{aspect}",
                    "query_id": query_row["query_id"],
                    "city": slot["city"],
                    "query_type": query_row["query_type"],
                    "target_aspect": aspect,
                    "target_role": "avoid",
                    "query_text_zh": query_row["query_text_zh"],
                    "query_en_full": slot["query_en"],
                    "query_en_target": build_query_en_target(slot["city"], aspect, "avoid"),
                }
            )
    return units


def build_retrieval_metric_summary(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {
            "aspect_recall_at_5": 0.0,
            "ndcg_at_5": 0.0,
            "precision_at_5": 0.0,
            "mrr_at_5": 0.0,
            "evidence_diversity_at_5": 0.0,
        }
    summary = {}
    for metric_name in [
        "aspect_recall_at_5",
        "ndcg_at_5",
        "precision_at_5",
        "mrr_at_5",
        "evidence_diversity_at_5",
    ]:
        summary[metric_name] = round(
            sum(float(row.get(metric_name, 0.0)) for row in metric_rows) / len(metric_rows),
            4,
        )
    return summary


def build_retrieval_summary_row(
    *,
    group_id: str,
    query_count: int,
    target_unit_count: int,
    latencies: list[float],
    metric_rows: list[dict[str, float]],
    config_hash: str,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "group_id": group_id,
        "query_count": query_count,
        "target_unit_count": target_unit_count,
        "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 3),
        "config_hash": config_hash,
    }
    summary.update(build_retrieval_metric_summary(metric_rows))
    if extra_fields:
        summary.update(extra_fields)
    return summary


def rows_to_evidence_pack(
    *,
    hotel_id: str,
    query_en: str,
    rows: list[dict[str, Any]],
    retrieval_trace: dict[str, Any],
) -> EvidencePack:
    grouped: dict[str, list[SentenceCandidate]] = {}
    all_sentence_ids: list[str] = []
    for row in rows:
        aspect_value: ASPECT_NAME = cast(ASPECT_NAME, str(row.get("sentence_aspect") or "general"))
        candidate = SentenceCandidate(
            sentence_id=row["sentence_id"],
            sentence_text=row["sentence_text"],
            aspect=aspect_value,
            sentiment=row.get("sentence_sentiment", "neutral"),
            review_date=row.get("review_date"),
            score_dense=row.get("score_dense"),
            score_rerank=row.get("score_rerank"),
        )
        grouped.setdefault(aspect_value, []).append(candidate)
        all_sentence_ids.append(candidate.sentence_id)
    return EvidencePack(
        hotel_id=hotel_id,
        query_en=query_en,
        evidence_by_aspect=grouped,
        all_sentence_ids=all_sentence_ids,
        retrieval_trace=dict(retrieval_trace),
    )


def generation_unit_from_retrieval_assets(
    *,
    query_row: dict[str, Any],
    slot_row: dict[str, Any],
    candidate_hotels: list[HotelCandidate],
    evidence_packs: list[EvidencePack],
    retrieval_mode: str,
    candidate_policy: str,
    config_hash: str,
) -> GenerationEvalUnit:
    preference = UserPreference.model_validate(
        {
            "city": slot_row.get("city"),
            "state": slot_row.get("state"),
            "hotel_category": slot_row.get("hotel_category"),
            "focus_aspects": slot_row.get("focus_aspects", []),
            "avoid_aspects": slot_row.get("avoid_aspects", []),
            "unsupported_requests": slot_row.get("unsupported_requests", []),
            "query_en": slot_row["query_en"],
        }
    )
    return GenerationEvalUnit(
        query_id=query_row["query_id"],
        query_text_zh=query_row["query_text_zh"],
        query_type=query_row["query_type"],
        user_preference_gold=preference,
        unsupported_requests=slot_row.get("unsupported_requests", []),
        candidate_hotels=candidate_hotels,
        evidence_packs=evidence_packs,
        retrieval_mode=retrieval_mode,
        candidate_policy=candidate_policy,
        config_hash=config_hash,
    )


def build_city_test_hotels(split_manifest: dict, review_df: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    hotel_meta = cast(pd.DataFrame, review_df.loc[:, ["hotel_id", "city", "hotel_name"]].copy())
    hotel_meta = cast(pd.DataFrame, hotel_meta.loc[~hotel_meta["hotel_id"].duplicated()].copy())
    hotel_meta = cast(pd.DataFrame, hotel_meta.sort_values(["city", "hotel_name", "hotel_id"]))
    test_ids = list(split_manifest["splits"]["test"])
    hotel_meta = cast(pd.DataFrame, hotel_meta[hotel_meta["hotel_id"].isin(test_ids)].copy())

    city_map: dict[str, list[dict[str, str]]] = {}
    for city, group in hotel_meta.groupby("city", sort=True):
        city_key = str(city)
        city_map[city_key] = [
            {
                "hotel_id": str(row["hotel_id"]),
                "hotel_name": str(row["hotel_name"]),
            }
            for _, row in group.iterrows()
        ]
    return city_map


def build_evidence_lookup(evidence_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup = evidence_df.set_index("sentence_id").to_dict("index")
    for sentence_id, row in lookup.items():
        row["review_id"] = review_id_from_sentence_id(sentence_id)
    return lookup


def dense_query_hotel(
    collection,
    query_embedding: list[list[float]],
    hotel_id: str,
    city: str,
    hotel_name: str,
    top_k: int,
    evidence_lookup: dict[str, dict[str, Any]],
    aspect: str | None = None,
    channel: str = "main",
) -> list[dict[str, Any]]:
    where_terms: list[dict[str, str]] = [{"hotel_id": str(hotel_id)}, {"city": city}]
    if aspect is not None:
        where_terms.append({"aspect": aspect})

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where={"$and": where_terms},
        include=["distances"],
    )

    sentence_ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    rows: list[dict[str, Any]] = []
    for idx, sentence_id in enumerate(sentence_ids):
        if sentence_id not in evidence_lookup:
            continue
        meta = evidence_lookup[sentence_id]
        review_date_value = meta.get("review_date")
        review_date_iso = None
        review_date_text = "" if review_date_value is None else str(review_date_value)
        if review_date_text and review_date_text != "NaT":
            review_date_iso = str(pd.Timestamp(review_date_text).date())
        rows.append(
            {
                "hotel_id": hotel_id,
                "hotel_name": hotel_name,
                "sentence_id": sentence_id,
                "sentence_text": meta["sentence_text"],
                "sentence_aspect": meta["aspect"],
                "sentence_sentiment": meta["sentiment"],
                "review_id": meta["review_id"],
                "review_date": review_date_iso,
                "score_dense": round(float(distances[idx]), 6),
                "score_rerank": None,
                "channel": channel,
            }
        )
    return rows


def merge_dense_candidates(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    best_by_sentence: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (item["score_dense"], item["sentence_id"], item["channel"] != "main")):
        sentence_id = row["sentence_id"]
        existing = best_by_sentence.get(sentence_id)
        if existing is None:
            best_by_sentence[sentence_id] = row
            continue
        should_replace = False
        if row["score_dense"] < existing["score_dense"]:
            should_replace = True
        elif row["score_dense"] == existing["score_dense"] and existing["channel"] == "fallback" and row["channel"] == "main":
            should_replace = True
        if should_replace:
            best_by_sentence[sentence_id] = row

    merged = list(best_by_sentence.values())
    merged.sort(key=lambda item: (item["score_dense"], item["sentence_id"]))
    return merged[:top_k]


def apply_rerank(
    query_text: str,
    rows: list[dict[str, Any]],
    reranker: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    ranked_rows = [dict(row) for row in rows]
    pairs = [(query_text, row["sentence_text"]) for row in ranked_rows]
    scores = reranker.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
    for row, score in zip(ranked_rows, scores):
        row["score_rerank"] = round(float(score), 6)
    ranked_rows.sort(
        key=lambda item: (
            -float(item["score_rerank"]),
            float(item["score_dense"]),
            item["sentence_id"],
        )
    )
    return ranked_rows[:top_k]


def evaluate_evidence_insufficiency(rows: list[dict[str, Any]]) -> tuple[bool, str]:
    reasons = []
    if len(rows) < FALLBACK_MIN_SENTENCES:
        reasons.append(f"sentence_count<{FALLBACK_MIN_SENTENCES}")
    unique_reviews = len({row["review_id"] for row in rows})
    if unique_reviews < FALLBACK_MIN_UNIQUE_REVIEWS:
        reasons.append(f"unique_reviews<{FALLBACK_MIN_UNIQUE_REVIEWS}")
    if reasons:
        return True, "; ".join(reasons)
    return False, "sufficient_main_evidence"


def warm_up_models(collection, bi_encoder: Any, reranker: Any, normalize_embeddings: bool) -> None:
    warmup_embedding = bi_encoder.encode(
        ["hotel retrieval warmup query"],
        normalize_embeddings=normalize_embeddings,
    ).tolist()
    collection.query(query_embeddings=warmup_embedding, n_results=1, include=["distances"])
    reranker.predict(
        [("hotel retrieval warmup query", "warmup sentence")],
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def retrieve_official_mode(
    unit: dict[str, Any],
    mode: str,
    city_hotels: list[dict[str, str]],
    collection,
    bi_encoder: Any,
    reranker: Any,
    normalize_embeddings: bool,
    dense_top_k: int,
    final_top_k: int,
    evidence_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    start = time.perf_counter()
    query_embedding = bi_encoder.encode(
        [unit["query_en_target"]],
        normalize_embeddings=normalize_embeddings,
    ).tolist()

    def run_dense(aspect: str | None, channel: str) -> tuple[list[dict[str, Any]], int]:
        all_rows: list[dict[str, Any]] = []
        call_count = 0
        for hotel in city_hotels:
            all_rows.extend(
                dense_query_hotel(
                    collection=collection,
                    query_embedding=query_embedding,
                    hotel_id=hotel["hotel_id"],
                    city=unit["city"],
                    hotel_name=hotel["hotel_name"],
                    top_k=dense_top_k,
                    evidence_lookup=evidence_lookup,
                    aspect=aspect,
                    channel=channel,
                )
            )
            call_count += 1
        return merge_dense_candidates(all_rows, top_k=dense_top_k), call_count

    main_dense: list[dict[str, Any]] = []
    fallback_dense: list[dict[str, Any]] = []
    main_top5: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    call_count = 0
    fallback_activated = False
    insufficiency_reason = ""

    if mode == "plain_city_test_rerank":
        dense_rows, calls = run_dense(aspect=None, channel="plain")
        call_count += calls
        final_rows = apply_rerank(unit["query_en_target"], dense_rows, reranker, top_k=final_top_k)
        main_top5 = final_rows
        insufficiency_flag, insufficiency_reason = evaluate_evidence_insufficiency(final_rows)
        retrieval_trace = {
            "mode": mode,
            "dense_pool_size": len(dense_rows),
            "rerank_input_count": len(dense_rows),
            "fallback_activated": False,
            "main_insufficiency_flag": insufficiency_flag,
            "main_insufficiency_reason": insufficiency_reason,
        }
    else:
        main_dense, calls = run_dense(aspect=unit["target_aspect"], channel="main")
        call_count += calls
        if mode == "aspect_main_no_rerank":
            final_rows = [dict(row) for row in main_dense[:final_top_k]]
            main_top5 = final_rows
            insufficiency_flag, insufficiency_reason = evaluate_evidence_insufficiency(main_top5)
            retrieval_trace = {
                "mode": mode,
                "dense_pool_size": len(main_dense),
                "rerank_input_count": 0,
                "fallback_activated": False,
                "main_insufficiency_flag": insufficiency_flag,
                "main_insufficiency_reason": insufficiency_reason,
            }
        else:
            main_top5 = apply_rerank(unit["query_en_target"], main_dense, reranker, top_k=final_top_k)
            insufficiency_flag, insufficiency_reason = evaluate_evidence_insufficiency(main_top5)
            if mode == "aspect_main_rerank" or not insufficiency_flag:
                final_rows = main_top5
                retrieval_trace = {
                    "mode": mode,
                    "dense_pool_size": len(main_dense),
                    "rerank_input_count": len(main_dense),
                    "fallback_activated": False,
                    "main_insufficiency_flag": insufficiency_flag,
                    "main_insufficiency_reason": insufficiency_reason,
                }
            else:
                fallback_dense, fallback_calls = run_dense(aspect=None, channel="fallback")
                call_count += fallback_calls
                fallback_activated = True
                combined_dense = merge_dense_candidates(main_dense + fallback_dense, top_k=dense_top_k * 2)
                final_rows = apply_rerank(unit["query_en_target"], combined_dense, reranker, top_k=final_top_k)
                retrieval_trace = {
                    "mode": mode,
                    "dense_pool_size": len(main_dense),
                    "rerank_input_count": len(main_dense),
                    "fallback_activated": True,
                    "fallback_dense_pool_size": len(fallback_dense),
                    "combined_rerank_input_count": len(combined_dense),
                    "main_insufficiency_flag": insufficiency_flag,
                    "main_insufficiency_reason": insufficiency_reason,
                }

    latency_ms = round((time.perf_counter() - start) * 1000, 3)
    return {
        "mode": mode,
        "unit": unit,
        "city_test_hotels": city_hotels,
        "rows": final_rows,
        "latency_ms": latency_ms,
        "retrieval_calls": call_count,
        "retrieval_trace": retrieval_trace,
        "main_top5": main_top5,
        "main_dense": main_dense,
        "fallback_dense": fallback_dense,
        "fallback_activated": fallback_activated,
        "insufficiency_reason": insufficiency_reason,
    }


def build_pool_rows(mode_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pooled: dict[tuple[str, str], dict[str, Any]] = {}
    for mode_result in mode_results:
        unit = mode_result["unit"]
        for rank, row in enumerate(mode_result["rows"], start=1):
            key = (unit["unit_id"], row["sentence_id"])
            existing = pooled.get(key)
            if existing is None:
                pooled[key] = {
                    "query_id": unit["query_id"],
                    "city": unit["city"],
                    "query_type": unit["query_type"],
                    "target_aspect": unit["target_aspect"],
                    "target_role": unit["target_role"],
                    "query_text_zh": unit["query_text_zh"],
                    "query_en_target": unit["query_en_target"],
                    "hotel_id": row["hotel_id"],
                    "hotel_name": row["hotel_name"],
                    "sentence_id": row["sentence_id"],
                    "sentence_text": row["sentence_text"],
                    "sentence_aspect": row["sentence_aspect"],
                    "sentence_sentiment": row["sentence_sentiment"],
                    "review_id": row["review_id"],
                    "pooled_from": [mode_result["mode"]],
                    "relevance": "",
                    "aspect_match": "",
                    "polarity_match": "",
                    "notes": "",
                    "_rank_hint": rank,
                }
            else:
                if mode_result["mode"] not in existing["pooled_from"]:
                    existing["pooled_from"].append(mode_result["mode"])
                existing["_rank_hint"] = min(existing["_rank_hint"], rank)

    output_rows = []
    for row in pooled.values():
        pooled_from = sorted(row.pop("pooled_from"))
        rank_hint = row.pop("_rank_hint")
        row["pooled_from"] = ";".join(pooled_from)
        row["_rank_hint"] = rank_hint
        output_rows.append(row)

    output_rows.sort(
        key=lambda item: (
            item["query_id"],
            item["target_role"] != "focus",
            item["target_aspect"],
            item["_rank_hint"],
            item["hotel_name"],
            item["sentence_id"],
        )
    )
    for row in output_rows:
        row.pop("_rank_hint")
    return output_rows


def write_e6_labeling_log(
    target_units: list[dict[str, Any]],
    pool_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    unique_queries = sorted({row["query_id"] for row in target_units})
    unique_units = sorted({row["unit_id"] for row in target_units})
    lines = [
        "# E6 Labeling Log",
        "",
        "## Status",
        "",
        "- [x] qrels pool generated",
        "- [ ] manual labeling completed",
        "- [ ] qrels_evidence.jsonl frozen",
        "",
        "## Pool Summary",
        "",
        f"- Executable queries: {len(unique_queries)}",
        f"- Query-aspect units: {len(unique_units)}",
        f"- Pooled sentence rows: {len(pool_rows)}",
        f"- Official modes: {', '.join(OFFICIAL_MODES)}",
        f"- Pooling depth / mode: Top{POOL_TOP_K}",
        "",
        "## Next Step",
        "",
        f"1. Annotate `{(E6_LABELS_DIR / 'qrels_pool.csv').as_posix()}`.",
        "2. Freeze qrels with `python -m scripts.evaluation.run_experiment_suite --task e6_freeze_qrels`.",
        f"3. Run `e6_retrieval`, `e7_reranker`, `e8_fallback`.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_g_labeling_log(
    target_units: list[dict[str, Any]],
    pool_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    unique_queries = sorted({row["query_id"] for row in target_units})
    unique_units = sorted({row["unit_id"] for row in target_units})
    lines = [
        "# G Retrieval Labeling Log",
        "",
        "## Status",
        "",
        "- [x] G qrels pool generated",
        "- [ ] manual labeling completed",
        "- [ ] g_qrels_evidence.jsonl frozen",
        "",
        "## Pool Summary",
        "",
        f"- Executable queries: {len(unique_queries)}",
        f"- Query-aspect units: {len(unique_units)}",
        f"- Pooled sentence rows: {len(pool_rows)}",
        f"- Query asset: {G_EVAL_QUERY_IDS_PATH.as_posix()}",
        f"- Query types: {', '.join(G_EVAL_QUERY_TYPE_ORDER)}",
        f"- Official modes: {', '.join(OFFICIAL_MODES)}",
        f"- Pooling depth / mode: Top{POOL_TOP_K}",
        "",
        "## Next Step",
        "",
        f"1. Annotate `{G_QRELS_POOL_PATH.as_posix()}`.",
        "2. Freeze qrels with `python -m scripts.evaluation.run_experiment_suite --task g_freeze_qrels`.",
        "3. Run `g_retrieval_eval --retrieval-variant plain` and `g_retrieval_eval --retrieval-variant aspect`.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_qrels_pool_rows(
    *,
    target_units: list[dict[str, Any]],
    cfg: dict[str, Any],
    split_manifest: dict[str, Any],
    review_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    city_test_hotels = build_city_test_hotels(split_manifest, review_df)
    evidence_lookup = build_evidence_lookup(evidence_df)

    PersistentClient = require_chromadb_client()
    sentence_transformer_cls, cross_encoder_cls = require_retrieval_backends()
    client = PersistentClient(path=cfg["embedding"]["chroma_persist_dir"])
    collection = client.get_collection(cfg["embedding"]["chroma_collection"])
    bi_encoder = sentence_transformer_cls(cfg["embedding"]["model"])
    reranker = cross_encoder_cls(cfg["reranker"]["model"])
    normalize_embeddings = bool(cfg["embedding"].get("normalize", True))
    warm_up_models(collection, bi_encoder, reranker, normalize_embeddings)

    pool_rows: list[dict[str, Any]] = []
    for unit in target_units:
        city_hotels = city_test_hotels.get(unit["city"])
        if city_hotels is None:
            raise KeyError(
                f"City '{unit['city']}' for query {unit['query_id']} is missing from city_test_hotels during qrels pool build."
            )
        mode_results = [
            retrieve_official_mode(
                unit=unit,
                mode=mode,
                city_hotels=city_hotels,
                collection=collection,
                bi_encoder=bi_encoder,
                reranker=reranker,
                normalize_embeddings=normalize_embeddings,
                dense_top_k=cfg["reranker"]["top_k_before_rerank"],
                final_top_k=POOL_TOP_K,
                evidence_lookup=evidence_lookup,
            )
            for mode in OFFICIAL_MODES
        ]
        pool_rows.extend(build_pool_rows(mode_results))
    return pool_rows


def _freeze_qrels_from_pool(
    *,
    pool_path: Path,
    output_path: Path,
) -> Path:
    if not pool_path.exists():
        raise FileNotFoundError(f"Missing qrels pool: {pool_path}")

    df = pd.read_csv(pool_path, keep_default_na=False)
    required_cols = {"relevance", "aspect_match", "polarity_match"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise KeyError(f"qrels pool missing columns: {', '.join(missing)}")

    for column in ["relevance", "aspect_match", "polarity_match"]:
        empty_count = int(df[column].astype(str).str.strip().eq("").sum())
        if empty_count:
            raise ValueError(f"`{column}` still has {empty_count} empty rows in {pool_path.name}")

    qrels_rows = []
    for _, row in df.iterrows():
        relevance = int(str(row["relevance"]).strip())
        aspect_match = int(str(row["aspect_match"]).strip())
        polarity_match = int(str(row["polarity_match"]).strip())
        if relevance not in {0, 1, 2}:
            raise ValueError(f"Invalid relevance value for {row['sentence_id']}: {relevance}")
        if aspect_match not in {0, 1}:
            raise ValueError(f"Invalid aspect_match value for {row['sentence_id']}: {aspect_match}")
        if polarity_match not in {0, 1}:
            raise ValueError(f"Invalid polarity_match value for {row['sentence_id']}: {polarity_match}")

        graded_relevance = relevance if aspect_match == 1 and polarity_match == 1 else 0
        binary_relevant = int(graded_relevance >= 1)
        qrels_rows.append(
            {
                "query_id": row["query_id"],
                "target_aspect": row["target_aspect"],
                "target_role": row["target_role"],
                "sentence_id": row["sentence_id"],
                "hotel_id": row["hotel_id"],
                "review_id": row["review_id"],
                "relevance": relevance,
                "aspect_match": aspect_match,
                "polarity_match": polarity_match,
                "graded_relevance": graded_relevance,
                "binary_relevant": binary_relevant,
                "sentence_text": row["sentence_text"],
                "sentence_aspect": row["sentence_aspect"],
                "sentence_sentiment": row["sentence_sentiment"],
                "notes": row.get("notes", ""),
            }
        )

    write_jsonl(output_path, qrels_rows)
    return output_path


def build_e6_qrels_pool(limit_queries: int | None = None) -> Path:
    cfg = load_config()
    split_manifest = load_json(EXPERIMENT_ASSETS_DIR / "frozen_split_manifest.json")
    review_df = cast(pd.DataFrame, pd.read_pickle("data/intermediate/cleaned_reviews.pkl"))
    evidence_df = cast(pd.DataFrame, pd.read_pickle("data/intermediate/evidence_index.pkl"))
    target_units = build_target_units(limit_queries=limit_queries)
    pool_rows = _build_qrels_pool_rows(
        target_units=target_units,
        cfg=cfg,
        split_manifest=split_manifest,
        review_df=review_df,
        evidence_df=evidence_df,
    )

    ensure_dir(E6_LABELS_DIR)
    pool_path = E6_LABELS_DIR / "qrels_pool.csv"
    pd.DataFrame(pool_rows).to_csv(pool_path, index=False, encoding="utf-8-sig")
    write_e6_labeling_log(target_units, pool_rows, E6_LABELS_DIR / "e6_labeling_log.md")
    (E6_LABELS_DIR / "qrels_evidence.jsonl").write_text("\n", encoding="utf-8")
    return pool_path


def freeze_e6_qrels() -> Path:
    return _freeze_qrels_from_pool(
        pool_path=E6_LABELS_DIR / "qrels_pool.csv",
        output_path=E6_LABELS_DIR / "qrels_evidence.jsonl",
    )


def build_g_qrels_pool(limit_queries: int | None = None) -> Path:
    cfg = load_config()
    split_manifest = load_json(EXPERIMENT_ASSETS_DIR / "frozen_split_manifest.json")
    review_df = cast(pd.DataFrame, pd.read_pickle("data/intermediate/cleaned_reviews.pkl"))
    evidence_df = cast(pd.DataFrame, pd.read_pickle("data/intermediate/evidence_index.pkl"))
    target_units = build_g_target_units(limit_queries=limit_queries)
    pool_rows = _build_qrels_pool_rows(
        target_units=target_units,
        cfg=cfg,
        split_manifest=split_manifest,
        review_df=review_df,
        evidence_df=evidence_df,
    )

    ensure_dir(G_QRELS_LABELS_DIR)
    pd.DataFrame(pool_rows).to_csv(G_QRELS_POOL_PATH, index=False, encoding="utf-8-sig")
    write_g_labeling_log(target_units, pool_rows, G_QRELS_LOG_PATH)
    G_QRELS_EVIDENCE_PATH.write_text("\n", encoding="utf-8")
    return G_QRELS_POOL_PATH


def freeze_g_qrels() -> Path:
    return _freeze_qrels_from_pool(
        pool_path=G_QRELS_POOL_PATH,
        output_path=G_QRELS_EVIDENCE_PATH,
    )




def validate_g_qrels(
    qrels_path: Path = G_QRELS_EVIDENCE_PATH,
    *,
    limit_queries: int | None = None,
) -> dict[str, Any]:
    if not qrels_path.exists():
        raise FileNotFoundError(f"Missing G qrels file: {qrels_path}")

    expected_units = build_g_target_units(limit_queries=limit_queries)
    expected_query_ids = {unit["query_id"] for unit in expected_units}
    expected_unit_keys = {
        (unit["query_id"], unit["target_aspect"], unit["target_role"])
        for unit in expected_units
    }

    qrels_rows = load_jsonl(qrels_path)
    if not qrels_rows:
        raise ValueError(f"G qrels file is empty: {qrels_path}")
    qrels_lookup = load_qrels_lookup(qrels_path)
    actual_unit_keys = set(qrels_lookup)
    actual_query_ids = {row["query_id"] for row in qrels_rows}

    missing_unit_keys = sorted(expected_unit_keys - actual_unit_keys)
    unexpected_unit_keys = sorted(actual_unit_keys - expected_unit_keys)
    if missing_unit_keys or unexpected_unit_keys:
        raise ValueError(
            "G qrels target-unit coverage mismatch: "
            f"missing={missing_unit_keys[:5]}, unexpected={unexpected_unit_keys[:5]}"
        )

    missing_query_ids = sorted(expected_query_ids - actual_query_ids)
    unexpected_query_ids = sorted(actual_query_ids - expected_query_ids)
    if missing_query_ids or unexpected_query_ids:
        raise ValueError(
            f"G qrels query coverage mismatch: missing={missing_query_ids}, unexpected={unexpected_query_ids}"
        )

    return {
        "qrels_path": str(qrels_path),
        "query_count": len(actual_query_ids),
        "target_unit_count": len(actual_unit_keys),
        "row_count": len(qrels_rows),
    }

def load_qrels_lookup(qrels_path: Path) -> dict[tuple[str, str, str], dict[str, dict[str, Any]]]:
    rows = load_jsonl(qrels_path)
    if not rows:
        raise ValueError(f"Qrels file is empty: {qrels_path}")

    lookup: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["query_id"], row["target_aspect"], row["target_role"])
        lookup.setdefault(key, {})[row["sentence_id"]] = row
    return lookup


def dcg_at_k(relevances: list[int], k: int) -> float:
    score = 0.0
    for idx, rel in enumerate(relevances[:k], start=1):
        score += (2**rel - 1) / math.log2(idx + 1)
    return score


def ndcg_at_k(relevances: list[int], k: int) -> float:
    actual = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def evaluate_ranked_rows(
    rows: list[dict[str, Any]],
    qrels_by_sentence: dict[str, dict[str, Any]],
    k: int = 5,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    graded: list[int] = []
    binary: list[int] = []
    enriched_rows: list[dict[str, Any]] = []
    for row in rows[:k]:
        qrel = qrels_by_sentence.get(row["sentence_id"])
        graded_rel = int(qrel["graded_relevance"]) if qrel else 0
        binary_rel = int(qrel["binary_relevant"]) if qrel else 0
        graded.append(graded_rel)
        binary.append(binary_rel)
        enriched = dict(row)
        enriched["graded_relevance"] = graded_rel
        enriched["binary_relevant"] = binary_rel
        enriched["aspect_match"] = int(qrel["aspect_match"]) if qrel else 0
        enriched["polarity_match"] = int(qrel["polarity_match"]) if qrel else 0
        enriched_rows.append(enriched)

    ideal_relevances = [int(item["graded_relevance"]) for item in qrels_by_sentence.values()]
    metrics = {
        "aspect_recall_at_5": float(any(binary)),
        "ndcg_at_5": round(
            0.0
            if not ideal_relevances
            else dcg_at_k(graded, k) / max(dcg_at_k(sorted(ideal_relevances, reverse=True), k), 1e-12),
            4,
        ),
        "precision_at_5": round(sum(binary) / k, 4),
        "evidence_diversity_at_5": round(len({row["review_id"] for row in rows[:k]}) / max(len(rows[:k]), 1), 4),
    }

    reciprocal_rank = 0.0
    for idx, rel in enumerate(binary, start=1):
        if rel:
            reciprocal_rank = 1.0 / idx
            break
    metrics["mrr_at_5"] = round(reciprocal_rank, 4)
    return metrics, enriched_rows


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["| none |", "|---|"]
    headers = list(rows[0].keys())
    table = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        table.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return table


def pairwise_cases(log_rows: list[dict[str, Any]], group_a: str, group_b: str) -> tuple[list[dict], list[dict]]:
    by_key: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in log_rows:
        unit = row["intermediate_objects"]["query_unit"]
        key = (unit["query_id"], unit["target_aspect"], unit["target_role"])
        by_key.setdefault(key, {})[row["group_id"]] = row

    improvements = []
    regressions = []
    for grouped in by_key.values():
        left = grouped.get(group_a)
        right = grouped.get(group_b)
        if not left or not right:
            continue
        delta = right["intermediate_objects"]["metrics"]["ndcg_at_5"] - left["intermediate_objects"]["metrics"]["ndcg_at_5"]
        payload = {
            "query_id": right["intermediate_objects"]["query_unit"]["query_id"],
            "target_aspect": right["intermediate_objects"]["query_unit"]["target_aspect"],
            "target_role": right["intermediate_objects"]["query_unit"]["target_role"],
            "query_text_zh": right["intermediate_objects"]["query_unit"]["query_text_zh"],
            "delta_ndcg_at_5": round(delta, 4),
            "left_top_sentence": left["intermediate_objects"]["ranked_rows"][0]["sentence_text"] if left["intermediate_objects"]["ranked_rows"] else "",
            "right_top_sentence": right["intermediate_objects"]["ranked_rows"][0]["sentence_text"] if right["intermediate_objects"]["ranked_rows"] else "",
        }
        if delta > 0:
            improvements.append(payload)
        elif delta < 0:
            regressions.append(payload)

    improvements.sort(key=lambda item: (-item["delta_ndcg_at_5"], item["query_id"], item["target_aspect"]))
    regressions.sort(key=lambda item: (item["delta_ndcg_at_5"], item["query_id"], item["target_aspect"]))
    return improvements[:3], regressions[:3]


def write_analysis_md(experiment_id: str, run_dir: Path, summary_rows: list[dict[str, Any]], log_rows: list[dict[str, Any]]) -> None:
    group_a, group_b = TASK_TO_MODES[experiment_id]
    lines = [
        f"# {experiment_id} Retrieval Result",
        "",
        "## Summary Table",
        "",
    ]
    lines.extend(markdown_table(summary_rows))
    lines.extend(["", "## Notes", ""])
    if experiment_id == "E6":
        lines.append("- Compare plain retrieval against aspect-aware retrieval under the same city-test candidate set.")
    elif experiment_id == "E7":
        lines.append("- Compare dense-only ranking against dense + cross-encoder reranking.")
    else:
        lines.append("- Compare strict main-channel retrieval against main + fallback retrieval.")
    lines.extend(
        [
            "- Unified retrieval metrics reported for all retrieval-side runs: Aspect Recall@5, nDCG@5, Precision@5, MRR@5, Evidence Diversity@5, Retrieval Latency.",
        ]
    )

    improvements, regressions = pairwise_cases(log_rows, group_a, group_b)
    lines.extend(["", "## Representative Improvements", ""])
    if improvements:
        for item in improvements:
            lines.extend(
                [
                    f"- `{item['query_id']}` | {item['target_role']}:{item['target_aspect']} | ΔnDCG@5={item['delta_ndcg_at_5']}",
                    f"  query: {item['query_text_zh']}",
                ]
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Representative Regressions", ""])
    if regressions:
        for item in regressions:
            lines.extend(
                [
                    f"- `{item['query_id']}` | {item['target_role']}:{item['target_aspect']} | ΔnDCG@5={item['delta_ndcg_at_5']}",
                    f"  query: {item['query_text_zh']}",
                ]
            )
    else:
        lines.append("- none")

    if experiment_id == "E8":
        fallback_rows = [
            row for row in log_rows
            if row["group_id"] == "aspect_main_fallback_rerank"
            and row["intermediate_objects"]["retrieval_trace"].get("fallback_activated")
        ]
        lines.extend(["", "## Fallback Cases", ""])
        if fallback_rows:
            for row in fallback_rows[:5]:
                unit = row["intermediate_objects"]["query_unit"]
                trace = row["intermediate_objects"]["retrieval_trace"]
                lines.append(
                    f"- `{unit['query_id']}` | {unit['target_role']}:{unit['target_aspect']} | reason={trace.get('main_insufficiency_reason', '')}"
                )
        else:
            lines.append("- none")

    (run_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")




def write_g_retrieval_analysis_md(
    retrieval_variant: str,
    run_dir: Path,
    summary_row: dict[str, Any],
) -> None:
    lines = [
        f"# G-Series {retrieval_variant.title()} Retrieval Evaluation",
        "",
        "## Summary Table",
        "",
    ]
    lines.extend(markdown_table([summary_row]))
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Retrieval variant: `{retrieval_variant}`",
            f"- Retrieval mode: `{summary_row.get('retrieval_mode', '')}`",
            f"- Candidate policy: `{summary_row.get('candidate_policy', '')}`",
            "- Metrics are from formal qrels-based retrieval evaluation over the G-series query set.",
        ]
    )
    (run_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")

def run_retrieval_eval(
    experiment_id: str,
    output_root: Path,
    limit_queries: int | None = None,
) -> Path:
    cfg = load_config()
    split_manifest = load_json(EXPERIMENT_ASSETS_DIR / "frozen_split_manifest.json")
    review_df = cast(pd.DataFrame, pd.read_pickle("data/intermediate/cleaned_reviews.pkl"))
    evidence_df = cast(pd.DataFrame, pd.read_pickle("data/intermediate/evidence_index.pkl"))
    city_test_hotels = build_city_test_hotels(split_manifest, review_df)
    evidence_lookup = build_evidence_lookup(evidence_df)
    target_units = build_target_units(limit_queries=limit_queries)
    qrels_path = E6_LABELS_DIR / "qrels_evidence.jsonl"
    qrels_lookup = load_qrels_lookup(qrels_path)
    qrels_rows = load_jsonl(qrels_path)

    PersistentClient = require_chromadb_client()
    sentence_transformer_cls, cross_encoder_cls = require_retrieval_backends()
    client = PersistentClient(path=cfg["embedding"]["chroma_persist_dir"])
    collection = client.get_collection(cfg["embedding"]["chroma_collection"])
    bi_encoder = sentence_transformer_cls(cfg["embedding"]["model"])
    reranker = cross_encoder_cls(cfg["reranker"]["model"])
    normalize_embeddings = bool(cfg["embedding"].get("normalize", True))
    warm_up_models(collection, bi_encoder, reranker, normalize_embeddings)

    task_modes = TASK_TO_MODES[experiment_id]
    stable_run_config = {
        "task": experiment_id,
        "split_config_hash": split_manifest["meta"]["config_hash"],
        "query_types": QUERY_SCOPE,
        "query_count": len({unit["query_id"] for unit in target_units}),
        "target_unit_count": len(target_units),
        "target_scope": TARGET_SCOPE,
        "candidate_set_policy": "city_test_hotels",
        "dense_top_k": cfg["reranker"]["top_k_before_rerank"],
        "final_top_k": cfg["reranker"]["top_k_after_rerank"],
        "embedding_model": cfg["embedding"]["model"],
        "reranker_model": cfg["reranker"]["model"],
        "collection": cfg["embedding"]["chroma_collection"],
        "fallback_rule": {
            "min_sentences": FALLBACK_MIN_SENTENCES,
            "min_unique_reviews": FALLBACK_MIN_UNIQUE_REVIEWS,
        },
        "qrels_hash": stable_hash({"rows": qrels_rows}),
        "official_modes": task_modes,
    }

    run_started_at = utc_now_iso()
    run_id = f"{experiment_id.lower()}_{stable_hash(stable_run_config)}_{run_started_at.replace(':', '').replace('-', '')}"
    run_dir = output_root / run_id
    ensure_dir(run_dir)

    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "generated_at": run_started_at,
                "stable_run_config": stable_run_config,
                "official_modes": task_modes,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    log_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for mode in task_modes:
        metric_rows = []
        fallback_sentence_total = 0
        fallback_sentence_nonrelevant = 0
        fallback_activation_flags = []
        insufficiency_flags = []
        latencies = []

        for unit in target_units:
            city_hotels = city_test_hotels.get(unit["city"])
            if city_hotels is None:
                raise KeyError(
                    f"City '{unit['city']}' for query {unit['query_id']} is missing from city_test_hotels during {experiment_id}."
                )
            mode_result = retrieve_official_mode(
                unit=unit,
                mode=mode,
                city_hotels=city_hotels,
                collection=collection,
                bi_encoder=bi_encoder,
                reranker=reranker,
                normalize_embeddings=normalize_embeddings,
                dense_top_k=cfg["reranker"]["top_k_before_rerank"],
                final_top_k=cfg["reranker"]["top_k_after_rerank"],
                evidence_lookup=evidence_lookup,
            )
            qrels_by_sentence = qrels_lookup.get((unit["query_id"], unit["target_aspect"], unit["target_role"]), {})
            metrics, ranked_rows = evaluate_ranked_rows(mode_result["rows"], qrels_by_sentence)
            metric_rows.append(metrics)
            latencies.append(mode_result["latency_ms"])

            retrieval_trace = dict(mode_result["retrieval_trace"])
            retrieval_trace["candidate_hotels"] = city_hotels
            retrieval_trace["dense_returned_count"] = len(mode_result["main_dense"])
            retrieval_trace["fallback_dense_returned_count"] = len(mode_result["fallback_dense"])
            retrieval_trace["main_unique_reviews_top5"] = len({row["review_id"] for row in mode_result["main_top5"]})

            fallback_activation = bool(retrieval_trace.get("fallback_activated"))
            fallback_activation_flags.append(int(fallback_activation))
            insufficiency_flags.append(int(bool(retrieval_trace.get("main_insufficiency_flag"))))

            if experiment_id == "E8" and fallback_activation:
                for row in ranked_rows:
                    if row["channel"] != "fallback":
                        continue
                    fallback_sentence_total += 1
                    if not row["binary_relevant"]:
                        fallback_sentence_nonrelevant += 1

            log_entry = RunLogEntry(
                run_id=run_id,
                group_id=mode,
                query_id=unit["query_id"],
                retrieval_mode=mode,
                candidate_mode=CANDIDATE_MODE,
                config_hash=stable_hash(stable_run_config | {"retrieval_mode": mode}),
                latency_ms=mode_result["latency_ms"],
                intermediate_objects={
                    "query_unit": unit,
                    "retrieval_trace": retrieval_trace,
                    "metrics": metrics,
                    "ranked_rows": ranked_rows,
                    "candidate_hotels": city_hotels,
                },
            )
            log_rows.append(log_entry.model_dump())

        extra_fields: dict[str, Any] = {}
        if experiment_id == "E8":
            extra_fields.update(
                {
                    "evidence_insufficiency_rate": round(sum(insufficiency_flags) / max(len(insufficiency_flags), 1), 4),
                    "fallback_activation_rate": round(sum(fallback_activation_flags) / max(len(fallback_activation_flags), 1), 4),
                    "fallback_noise_rate": round(
                        fallback_sentence_nonrelevant / max(fallback_sentence_total, 1),
                        4,
                    ),
                }
            )
        summary = build_retrieval_summary_row(
            group_id=mode,
            query_count=len({unit["query_id"] for unit in target_units}),
            target_unit_count=len(target_units),
            latencies=latencies,
            metric_rows=metric_rows,
            config_hash=stable_hash(stable_run_config | {"retrieval_mode": mode}),
            extra_fields=extra_fields,
        )

        summary_rows.append(summary)

    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as handle:
        for row in log_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")
    write_analysis_md(experiment_id, run_dir, summary_rows, log_rows)
    return run_dir


def run_g_retrieval_eval(
    retrieval_variant: str,
    output_root: Path,
    limit_queries: int | None = None,
) -> Path:
    if retrieval_variant not in G_RETRIEVAL_VARIANT_SPECS:
        raise ValueError(
            f"Unsupported G retrieval variant: {retrieval_variant}. "
            f"Expected one of {sorted(G_RETRIEVAL_VARIANT_SPECS)}"
        )

    spec = G_RETRIEVAL_VARIANT_SPECS[retrieval_variant]
    cfg = load_config()
    split_manifest = load_json(EXPERIMENT_ASSETS_DIR / "frozen_split_manifest.json")
    review_df = cast(pd.DataFrame, pd.read_pickle("data/intermediate/cleaned_reviews.pkl"))
    evidence_df = cast(pd.DataFrame, pd.read_pickle("data/intermediate/evidence_index.pkl"))
    city_test_hotels = build_city_test_hotels(split_manifest, review_df)
    evidence_lookup = build_evidence_lookup(evidence_df)
    target_units = build_g_target_units(limit_queries=limit_queries)
    qrels_path = G_QRELS_EVIDENCE_PATH
    qrels_lookup = load_qrels_lookup(qrels_path)
    qrels_rows = load_jsonl(qrels_path)

    PersistentClient = require_chromadb_client()
    sentence_transformer_cls, cross_encoder_cls = require_retrieval_backends()
    client = PersistentClient(path=cfg["embedding"]["chroma_persist_dir"])
    collection = client.get_collection(cfg["embedding"]["chroma_collection"])
    bi_encoder = sentence_transformer_cls(cfg["embedding"]["model"])
    reranker = cross_encoder_cls(cfg["reranker"]["model"])
    normalize_embeddings = bool(cfg["embedding"].get("normalize", True))
    warm_up_models(collection, bi_encoder, reranker, normalize_embeddings)

    stable_run_config = {
        "task": "G_retrieval_eval",
        "retrieval_variant": retrieval_variant,
        "split_config_hash": split_manifest["meta"]["config_hash"],
        "query_types": list(G_EVAL_QUERY_TYPE_ORDER),
        "query_count": len({unit["query_id"] for unit in target_units}),
        "target_unit_count": len(target_units),
        "target_scope": TARGET_SCOPE,
        "candidate_set_policy": "city_test_hotels",
        "dense_top_k": cfg["reranker"]["top_k_before_rerank"],
        "final_top_k": cfg["reranker"]["top_k_after_rerank"],
        "embedding_model": cfg["embedding"]["model"],
        "reranker_model": cfg["reranker"]["model"],
        "collection": cfg["embedding"]["chroma_collection"],
        "fallback_rule": {
            "min_sentences": FALLBACK_MIN_SENTENCES,
            "min_unique_reviews": FALLBACK_MIN_UNIQUE_REVIEWS,
        },
        "qrels_hash": stable_hash({"rows": qrels_rows}),
        "retrieval_mode": spec["retrieval_mode"],
        "candidate_policy": spec["candidate_policy"],
    }

    run_started_at = utc_now_iso()
    run_id = f"gret_{retrieval_variant}_{stable_hash(stable_run_config)}_{run_started_at.replace(':', '').replace('-', '')}"
    run_dir = output_root / run_id
    ensure_dir(run_dir)

    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "generated_at": run_started_at,
                "stable_run_config": stable_run_config,
                "retrieval_variant": retrieval_variant,
                "retrieval_mode": spec["retrieval_mode"],
                "candidate_policy": spec["candidate_policy"],
                "qrels_path": str(qrels_path),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    metric_rows: list[dict[str, float]] = []
    latencies: list[float] = []
    log_rows: list[dict[str, Any]] = []
    for unit in target_units:
        city_hotels = city_test_hotels.get(unit["city"])
        if city_hotels is None:
            raise KeyError(
                f"City '{unit['city']}' for query {unit['query_id']} is missing from city_test_hotels during G retrieval eval."
            )
        mode_result = retrieve_official_mode(
            unit=unit,
            mode=str(spec["retrieval_mode"]),
            city_hotels=city_hotels,
            collection=collection,
            bi_encoder=bi_encoder,
            reranker=reranker,
            normalize_embeddings=normalize_embeddings,
            dense_top_k=cfg["reranker"]["top_k_before_rerank"],
            final_top_k=cfg["reranker"]["top_k_after_rerank"],
            evidence_lookup=evidence_lookup,
        )
        qrels_by_sentence = qrels_lookup.get((unit["query_id"], unit["target_aspect"], unit["target_role"]), {})
        metrics, ranked_rows = evaluate_ranked_rows(mode_result["rows"], qrels_by_sentence)
        metric_rows.append(metrics)
        latencies.append(mode_result["latency_ms"])

        retrieval_trace = dict(mode_result["retrieval_trace"])
        retrieval_trace["candidate_hotels"] = city_hotels
        retrieval_trace["dense_returned_count"] = len(mode_result["main_dense"])
        retrieval_trace["fallback_dense_returned_count"] = len(mode_result["fallback_dense"])
        retrieval_trace["main_unique_reviews_top5"] = len({row["review_id"] for row in mode_result["main_top5"]})
        retrieval_trace["retrieval_variant"] = retrieval_variant

        log_entry = RunLogEntry(
            run_id=run_id,
            group_id=str(spec["summary_group_id"]),
            query_id=unit["query_id"],
            retrieval_mode=str(spec["retrieval_mode"]),
            candidate_mode=str(spec["candidate_policy"]),
            config_hash=stable_hash(stable_run_config | {"query_unit_id": unit["unit_id"]}),
            latency_ms=mode_result["latency_ms"],
            intermediate_objects={
                "query_unit": unit,
                "retrieval_trace": retrieval_trace,
                "metrics": metrics,
                "ranked_rows": ranked_rows,
                "candidate_hotels": city_hotels,
            },
        )
        log_rows.append(log_entry.model_dump())

    summary_row = build_retrieval_summary_row(
        group_id=str(spec["summary_group_id"]),
        query_count=len({unit["query_id"] for unit in target_units}),
        target_unit_count=len(target_units),
        latencies=latencies,
        metric_rows=metric_rows,
        config_hash=stable_hash(stable_run_config),
        extra_fields={
            "retrieval_variant": retrieval_variant,
            "retrieval_mode": spec["retrieval_mode"],
            "candidate_policy": spec["candidate_policy"],
            "retrieval_summary_source": "formal_retrieval_eval",
        },
    )

    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as handle:
        for row in log_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame([summary_row]).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")
    write_g_retrieval_analysis_md(retrieval_variant, run_dir, summary_row)
    return run_dir


def _hotel_candidates_from_ranked_df(top_candidates: pd.DataFrame) -> list[HotelCandidate]:
    rows: list[HotelCandidate] = []
    for _, candidate in top_candidates.iterrows():
        score_breakdown_raw: dict[str, Any] = {}
        score_breakdown_value = candidate["score_breakdown"] if "score_breakdown" in candidate else None
        if isinstance(score_breakdown_value, dict):
            score_breakdown_raw = score_breakdown_value
        score_breakdown = {
            str(key): float(value)
            for key, value in score_breakdown_raw.items()
        }
        rows.append(
            HotelCandidate(
                hotel_id=str(candidate["hotel_id"]),
                hotel_name=str(candidate["hotel_name"]),
                score_total=float(cast(Any, candidate["score_total"])),
                score_breakdown=score_breakdown,
            )
        )
    return rows


def freeze_g_retrieval_assets(
    *,
    output_path: Path,
    retrieval_mode: str,
    candidate_policy: str,
    candidate_mode: str = "B_final_aspect_score",
    query_ids: list[str] | None = None,
    limit_queries: int | None = None,
    top_k: int = 5,
) -> Path:
    from scripts.evaluation.evaluate_e2_candidate_selection import (
        build_hotel_summary,
        build_profile_tables,
        candidate_rank,
    )

    cfg = load_config()
    split_manifest = load_json(EXPERIMENT_ASSETS_DIR / "frozen_split_manifest.json")
    judged_queries = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "judged_queries.jsonl")}
    slot_gold = load_slot_gold_lookup()
    clarify_gold = load_clarify_gold_lookup()
    query_ids = list(query_ids or load_g_eval_query_ids())
    if limit_queries is not None:
        query_ids = query_ids[:limit_queries]
    if not query_ids:
        raise ValueError("No G-series query ids available for retrieval asset freeze after applying filters.")
    duplicate_query_ids = sorted({query_id for query_id in query_ids if query_ids.count(query_id) > 1})
    if duplicate_query_ids:
        raise ValueError(f"Duplicate query ids provided for G retrieval asset freeze: {duplicate_query_ids}")
    missing_judged = sorted(query_id for query_id in query_ids if query_id not in judged_queries)
    missing_slot = sorted(query_id for query_id in query_ids if query_id not in slot_gold)
    missing_clarify = sorted(query_id for query_id in query_ids if query_id not in clarify_gold)
    if missing_judged or missing_slot or missing_clarify:
        raise ValueError(
            "Invalid G retrieval query ids: "
            f"missing in judged_queries={missing_judged}, "
            f"missing in slot_gold={missing_slot}, "
            f"missing in clarify_gold={missing_clarify}"
        )

    review_df = cast(pd.DataFrame, pd.read_pickle("data/intermediate/cleaned_reviews.pkl"))
    profile_df = cast(pd.DataFrame, pd.read_pickle("data/intermediate/hotel_profiles.pkl"))
    evidence_df = cast(pd.DataFrame, pd.read_pickle("data/intermediate/evidence_index.pkl"))
    hotel_summary = build_hotel_summary(review_df)
    profile_current, profile_alt = build_profile_tables(profile_df)
    hotel_summary = cast(
        pd.DataFrame,
        hotel_summary[hotel_summary["hotel_id"].isin(list(split_manifest["splits"]["test"]))].copy(),
    )
    city_test_hotels = build_city_test_hotels(split_manifest, review_df)
    evidence_lookup = build_evidence_lookup(evidence_df)

    PersistentClient = require_chromadb_client()
    sentence_transformer_cls, cross_encoder_cls = require_retrieval_backends()
    client = PersistentClient(path=cfg["embedding"]["chroma_persist_dir"])
    collection = client.get_collection(cfg["embedding"]["chroma_collection"])
    bi_encoder = sentence_transformer_cls(cfg["embedding"]["model"])
    reranker = cross_encoder_cls(cfg["reranker"]["model"])
    normalize_embeddings = bool(cfg["embedding"].get("normalize", True))
    warm_up_models(collection, bi_encoder, reranker, normalize_embeddings)

    stable_run_config = {
        "task": "G_retrieval_asset_freeze",
        "retrieval_mode": retrieval_mode,
        "candidate_mode": candidate_mode,
        "candidate_policy": candidate_policy,
        "query_ids": query_ids,
        "query_count": len(query_ids),
        "top_k": top_k,
        "dense_top_k": cfg["reranker"]["top_k_before_rerank"],
        "final_top_k": cfg["reranker"]["top_k_after_rerank"],
        "embedding_model": cfg["embedding"]["model"],
        "reranker_model": cfg["reranker"]["model"],
        "split_config_hash": split_manifest["meta"]["config_hash"],
    }
    config_hash = stable_hash(stable_run_config)

    units: list[dict[str, Any]] = []
    for query_id in query_ids:
        query_row = judged_queries[query_id]
        slot_row = slot_gold[query_id]
        clarify_row = clarify_gold[query_id]
        if clarify_row.get("clarify_needed"):
            raise AssertionError(f"Clarification-required query should not enter G retrieval assets: {query_id}")
        city = slot_row["city"]
        if city not in city_test_hotels:
            raise KeyError(f"City '{city}' for query {query_id} is missing from city_test_hotels during G retrieval asset freeze.")
        city_hotels_df = cast(pd.DataFrame, hotel_summary[hotel_summary["city"] == city].copy())
        ranked = candidate_rank(
            city_hotels_df,
            profile_current,
            profile_alt,
            slot_row.get("focus_aspects", []),
            slot_row.get("avoid_aspects", []),
            candidate_mode,
        )
        ranked_candidates = _hotel_candidates_from_ranked_df(cast(pd.DataFrame, ranked.copy()))
        candidate_hotels: list[HotelCandidate] = []
        evidence_packs: list[EvidencePack] = []
        skipped_hotels_without_evidence: list[str] = []
        for hotel in ranked_candidates:
            evidence_by_aspect: dict[str, list[SentenceCandidate]] = {}
            all_sentence_ids: list[str] = []
            retrieval_trace: dict[str, Any] = {
                "mode": retrieval_mode,
                "query_type": query_row["query_type"],
                "candidate_policy": candidate_policy,
                "source_query_id": query_id,
                "hotel_name": hotel.hotel_name,
                "aspect_roles": {},
                "per_aspect_traces": {},
                "fallback_enabled": retrieval_mode == "aspect_main_fallback_rerank",
            }

            aspect_targets = [(aspect, "focus") for aspect in slot_row.get("focus_aspects", [])] + [
                (aspect, "avoid") for aspect in slot_row.get("avoid_aspects", [])
            ]

            for aspect, target_role in aspect_targets:
                unit = {
                    "query_id": query_id,
                    "city": city,
                    "query_type": query_row["query_type"],
                    "target_aspect": aspect,
                    "target_role": target_role,
                    "query_text_zh": query_row["query_text_zh"],
                    "query_en_full": slot_row["query_en"],
                    "query_en_target": build_query_en_target(city, aspect, target_role),
                }
                mode_result = retrieve_official_mode(
                    unit=unit,
                    mode=retrieval_mode,
                    city_hotels=[{"hotel_id": hotel.hotel_id, "hotel_name": hotel.hotel_name}],
                    collection=collection,
                    bi_encoder=bi_encoder,
                    reranker=reranker,
                    normalize_embeddings=normalize_embeddings,
                    dense_top_k=cfg["reranker"]["top_k_before_rerank"],
                    final_top_k=cfg["reranker"]["top_k_after_rerank"],
                    evidence_lookup=evidence_lookup,
                )
                retrieval_trace["aspect_roles"].setdefault(aspect, []).append(target_role)
                retrieval_trace["per_aspect_traces"].setdefault(aspect, {})[target_role] = dict(mode_result["retrieval_trace"])
                evidence_by_aspect[aspect] = [
                    SentenceCandidate.model_validate(
                        {
                            "sentence_id": row["sentence_id"],
                            "sentence_text": row["sentence_text"],
                            "aspect": row.get("sentence_aspect") or aspect,
                            "sentiment": row.get("sentence_sentiment", "neutral"),
                            "review_date": row.get("review_date"),
                            "score_dense": row.get("score_dense"),
                            "score_rerank": row.get("score_rerank"),
                        }
                    )
                    for row in mode_result["rows"]
                ]
                all_sentence_ids.extend(sentence.sentence_id for sentence in evidence_by_aspect[aspect])

            evidence_packs.append(
                EvidencePack(
                    hotel_id=hotel.hotel_id,
                    query_en=slot_row["query_en"],
                    evidence_by_aspect=evidence_by_aspect,
                    all_sentence_ids=list(dict.fromkeys(all_sentence_ids)),
                    retrieval_trace=retrieval_trace,
                )
            )
            if not all_sentence_ids:
                evidence_packs.pop()
                skipped_hotels_without_evidence.append(hotel.hotel_id)
                continue
            candidate_hotels.append(hotel)
            if len(candidate_hotels) >= top_k:
                break

        if not candidate_hotels:
            raise ValueError(
                f"No evidence-backed candidate hotels produced for query {query_id} in city '{city}' under candidate_mode={candidate_mode} and top_k={top_k}. "
                f"skipped_hotels_without_evidence={skipped_hotels_without_evidence}"
            )

        generation_unit = generation_unit_from_retrieval_assets(
            query_row=query_row,
            slot_row=slot_row,
            candidate_hotels=candidate_hotels,
            evidence_packs=evidence_packs,
            retrieval_mode=retrieval_mode,
            candidate_policy=candidate_policy,
            config_hash=config_hash,
        )
        units.append(generation_unit.model_dump())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, units)
    return output_path


def freeze_g_plain_retrieval_assets(
    output_path: Path = G_PLAIN_RETRIEVAL_UNITS_PATH,
    *,
    query_ids: list[str] | None = None,
    limit_queries: int | None = None,
) -> Path:
    return freeze_g_retrieval_assets(
        output_path=output_path,
        retrieval_mode="plain_city_test_rerank",
        candidate_policy="G_plain_retrieval_top5",
        query_ids=query_ids,
        limit_queries=limit_queries,
    )


def freeze_g_aspect_retrieval_assets(
    output_path: Path = G_ASPECT_RETRIEVAL_UNITS_PATH,
    *,
    query_ids: list[str] | None = None,
    limit_queries: int | None = None,
) -> Path:
    return freeze_g_retrieval_assets(
        output_path=output_path,
        retrieval_mode="aspect_main_no_rerank",
        candidate_policy="G_aspect_retrieval_top5",
        query_ids=query_ids,
        limit_queries=limit_queries,
    )


def validate_g_retrieval_assets(
    asset_path: Path,
    *,
    expected_retrieval_mode: str | None = None,
    expected_candidate_policy: str | None = None,
    expected_query_ids: list[str] | None = None,
) -> dict[str, Any]:
    if not asset_path.exists():
        raise FileNotFoundError(f"Missing G retrieval asset file: {asset_path}")

    expected_query_ids = list(expected_query_ids or load_g_eval_query_ids())
    expected_query_id_set = set(expected_query_ids)
    rows = load_jsonl(asset_path)
    if not rows:
        raise ValueError(f"G retrieval asset file is empty: {asset_path}")

    units = [GenerationEvalUnit.model_validate(row) for row in rows]
    query_ids = [unit.query_id for unit in units]
    duplicate_query_ids = sorted({query_id for query_id in query_ids if query_ids.count(query_id) > 1})
    if duplicate_query_ids:
        raise ValueError(f"Duplicate query ids found in {asset_path}: {duplicate_query_ids}")

    actual_query_id_set = set(query_ids)
    missing_query_ids = sorted(expected_query_id_set - actual_query_id_set)
    unexpected_query_ids = sorted(actual_query_id_set - expected_query_id_set)
    if missing_query_ids or unexpected_query_ids:
        raise ValueError(
            f"Query id mismatch in {asset_path}: missing={missing_query_ids}, unexpected={unexpected_query_ids}"
        )

    for unit in units:
        if expected_retrieval_mode and unit.retrieval_mode != expected_retrieval_mode:
            raise ValueError(
                f"Unexpected retrieval_mode for query {unit.query_id}: expected {expected_retrieval_mode}, got {unit.retrieval_mode}"
            )
        if expected_candidate_policy and unit.candidate_policy != expected_candidate_policy:
            raise ValueError(
                f"Unexpected candidate_policy for query {unit.query_id}: expected {expected_candidate_policy}, got {unit.candidate_policy}"
            )
        if not unit.candidate_hotels:
            raise ValueError(f"Query {unit.query_id} has no candidate_hotels in {asset_path}")
        if len(unit.candidate_hotels) != len(unit.evidence_packs):
            raise ValueError(
                f"Query {unit.query_id} has candidate/evidence count mismatch in {asset_path}: "
                f"{len(unit.candidate_hotels)} candidates vs {len(unit.evidence_packs)} evidence packs"
            )
        if not unit.user_preference_gold.focus_aspects and not unit.user_preference_gold.avoid_aspects:
            raise ValueError(f"Query {unit.query_id} has empty user preference aspects in {asset_path}")

        candidate_ids = [candidate.hotel_id for candidate in unit.candidate_hotels]
        evidence_ids = [pack.hotel_id for pack in unit.evidence_packs]
        if candidate_ids != evidence_ids:
            raise ValueError(
                f"Query {unit.query_id} has candidate/evidence hotel order mismatch in {asset_path}: "
                f"candidates={candidate_ids}, evidence_packs={evidence_ids}"
            )

        expected_aspects = {str(aspect) for aspect in unit.user_preference_gold.focus_aspects} | {
            str(aspect) for aspect in unit.user_preference_gold.avoid_aspects
        }
        for pack in unit.evidence_packs:
            available_aspects = {str(aspect) for aspect in pack.evidence_by_aspect}
            missing_aspects = sorted(expected_aspects - available_aspects)
            if missing_aspects:
                raise ValueError(
                    f"Query {unit.query_id} hotel {pack.hotel_id} is missing evidence aspects {missing_aspects} in {asset_path}"
                )
            if not pack.all_sentence_ids:
                raise ValueError(f"Query {unit.query_id} hotel {pack.hotel_id} has empty all_sentence_ids in {asset_path}")

    return {
        "asset_path": str(asset_path),
        "query_count": len(units),
        "retrieval_mode": expected_retrieval_mode or units[0].retrieval_mode,
        "candidate_policy": expected_candidate_policy or units[0].candidate_policy,
        "query_ids": query_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=[
            "build_qrels_pool",
            "freeze_qrels",
            "build_g_eval_query_ids",
            "freeze_g_plain_assets",
            "freeze_g_aspect_assets",
            "validate_g_plain_assets",
            "validate_g_aspect_assets",
            "build_g_qrels_pool",
            "freeze_g_qrels",
            "validate_g_qrels",
            "run_g_retrieval",
            "run_e6",
            "run_e7",
            "run_e8",
        ],
        required=True,
    )
    parser.add_argument("--output-root", default=str(EXPERIMENT_RUNS_DIR))
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--retrieval-variant", choices=["plain", "aspect"], default=None)
    args = parser.parse_args()

    if args.action == "build_qrels_pool":
        path = build_e6_qrels_pool(limit_queries=args.limit_queries)
        print(f"[OK] qrels pool written to {path}")
        return
    if args.action == "freeze_qrels":
        path = freeze_e6_qrels()
        print(f"[OK] qrels frozen to {path}")
        return
    if args.action == "build_g_eval_query_ids":
        path = write_g_eval_query_ids_asset()
        print(f"[OK] G-series query ids written to {path}")
        return
    if args.action == "freeze_g_plain_assets":
        path = freeze_g_plain_retrieval_assets(limit_queries=args.limit_queries)
        print(f"[OK] G plain retrieval assets written to {path}")
        return
    if args.action == "freeze_g_aspect_assets":
        path = freeze_g_aspect_retrieval_assets(limit_queries=args.limit_queries)
        print(f"[OK] G aspect retrieval assets written to {path}")
        return
    if args.action == "validate_g_plain_assets":
        summary = validate_g_retrieval_assets(
            G_PLAIN_RETRIEVAL_UNITS_PATH,
            expected_retrieval_mode="plain_city_test_rerank",
            expected_candidate_policy="G_plain_retrieval_top5",
        )
        print(f"[OK] G plain retrieval assets validated: {summary}")
        return
    if args.action == "validate_g_aspect_assets":
        summary = validate_g_retrieval_assets(
            G_ASPECT_RETRIEVAL_UNITS_PATH,
            expected_retrieval_mode="aspect_main_no_rerank",
            expected_candidate_policy="G_aspect_retrieval_top5",
        )
        print(f"[OK] G aspect retrieval assets validated: {summary}")
        return
    if args.action == "build_g_qrels_pool":
        path = build_g_qrels_pool(limit_queries=args.limit_queries)
        print(f"[OK] G qrels pool written to {path}")
        return
    if args.action == "freeze_g_qrels":
        path = freeze_g_qrels()
        print(f"[OK] G qrels frozen to {path}")
        return
    if args.action == "validate_g_qrels":
        summary = validate_g_qrels(limit_queries=args.limit_queries)
        print(f"[OK] G qrels validated: {summary}")
        return

    output_root = Path(args.output_root)
    if args.action == "run_g_retrieval":
        if not args.retrieval_variant:
            raise ValueError("run_g_retrieval requires --retrieval-variant plain|aspect")
        run_dir = run_g_retrieval_eval(args.retrieval_variant, output_root=output_root, limit_queries=args.limit_queries)
    elif args.action == "run_e6":
        run_dir = run_retrieval_eval("E6", output_root=output_root, limit_queries=args.limit_queries)
    elif args.action == "run_e7":
        run_dir = run_retrieval_eval("E7", output_root=output_root, limit_queries=args.limit_queries)
    else:
        run_dir = run_retrieval_eval("E8", output_root=output_root, limit_queries=args.limit_queries)
    print(f"[OK] run saved to {run_dir}")


if __name__ == "__main__":
    main()

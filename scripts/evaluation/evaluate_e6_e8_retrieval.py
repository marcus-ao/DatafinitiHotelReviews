"""Shared retrieval evaluation engine for E6, E7, and E8."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

from scripts.shared.experiment_schemas import RunLogEntry
from scripts.shared.experiment_utils import (
    E6_LABELS_DIR,
    EXPERIMENT_ASSETS_DIR,
    EXPERIMENT_RUNS_DIR,
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


def build_target_units(limit_queries: int | None = None) -> list[dict[str, Any]]:
    judged_queries = load_jsonl(EXPERIMENT_ASSETS_DIR / "judged_queries.jsonl")
    slot_gold = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "slot_gold.jsonl")}
    clarify_gold = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "clarify_gold.jsonl")}

    eligible_queries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in judged_queries:
        slot = slot_gold[row["query_id"]]
        clarify = clarify_gold[row["query_id"]]
        if clarify["clarify_needed"]:
            continue
        if not slot["city"] or not (slot["focus_aspects"] or slot["avoid_aspects"]):
            continue
        if row["query_type"] not in ALLOWED_QUERY_TYPES:
            continue
        eligible_queries.append((row, slot))

    if limit_queries:
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


def build_city_test_hotels(split_manifest: dict, review_df: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    hotel_meta = (
        review_df[["hotel_id", "city", "hotel_name"]]
        .drop_duplicates("hotel_id")
        .sort_values(["city", "hotel_name", "hotel_id"])
    )
    test_ids = set(split_manifest["splits"]["test"])
    hotel_meta = hotel_meta[hotel_meta["hotel_id"].isin(test_ids)].copy()

    city_map: dict[str, list[dict[str, str]]] = {}
    for city, group in hotel_meta.groupby("city", sort=True):
        city_map[city] = [
            {
                "hotel_id": row["hotel_id"],
                "hotel_name": row["hotel_name"],
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
        rows.append(
            {
                "hotel_id": hotel_id,
                "hotel_name": hotel_name,
                "sentence_id": sentence_id,
                "sentence_text": meta["sentence_text"],
                "sentence_aspect": meta["aspect"],
                "sentence_sentiment": meta["sentiment"],
                "review_id": meta["review_id"],
                "review_date": pd.Timestamp(meta["review_date"]).date().isoformat() if pd.notna(meta["review_date"]) else None,
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
    reranker: CrossEncoder,
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


def warm_up_models(collection, bi_encoder: SentenceTransformer, reranker: CrossEncoder, normalize_embeddings: bool) -> None:
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
    bi_encoder: SentenceTransformer,
    reranker: CrossEncoder,
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


def build_e6_qrels_pool(limit_queries: int | None = None) -> Path:
    cfg = load_config()
    split_manifest = load_json(EXPERIMENT_ASSETS_DIR / "frozen_split_manifest.json")
    review_df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
    evidence_df = pd.read_pickle("data/intermediate/evidence_index.pkl")
    city_test_hotels = build_city_test_hotels(split_manifest, review_df)
    evidence_lookup = build_evidence_lookup(evidence_df)
    target_units = build_target_units(limit_queries=limit_queries)

    from chromadb import PersistentClient

    client = PersistentClient(path=cfg["embedding"]["chroma_persist_dir"])
    collection = client.get_collection(cfg["embedding"]["chroma_collection"])
    bi_encoder = SentenceTransformer(cfg["embedding"]["model"])
    reranker = CrossEncoder(cfg["reranker"]["model"])
    normalize_embeddings = bool(cfg["embedding"].get("normalize", True))
    warm_up_models(collection, bi_encoder, reranker, normalize_embeddings)

    pool_rows: list[dict[str, Any]] = []
    for unit in target_units:
        city_hotels = city_test_hotels[unit["city"]]
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

    ensure_dir(E6_LABELS_DIR)
    pool_path = E6_LABELS_DIR / "qrels_pool.csv"
    pd.DataFrame(pool_rows).to_csv(pool_path, index=False, encoding="utf-8-sig")
    write_e6_labeling_log(target_units, pool_rows, E6_LABELS_DIR / "e6_labeling_log.md")
    (E6_LABELS_DIR / "qrels_evidence.jsonl").write_text("\n", encoding="utf-8")
    return pool_path


def freeze_e6_qrels() -> Path:
    pool_path = E6_LABELS_DIR / "qrels_pool.csv"
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
            raise ValueError(f"`{column}` still has {empty_count} empty rows in qrels_pool.csv")

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

    qrels_path = E6_LABELS_DIR / "qrels_evidence.jsonl"
    write_jsonl(qrels_path, qrels_rows)
    return qrels_path


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


def run_retrieval_eval(
    experiment_id: str,
    output_root: Path,
    limit_queries: int | None = None,
) -> Path:
    cfg = load_config()
    split_manifest = load_json(EXPERIMENT_ASSETS_DIR / "frozen_split_manifest.json")
    review_df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
    evidence_df = pd.read_pickle("data/intermediate/evidence_index.pkl")
    city_test_hotels = build_city_test_hotels(split_manifest, review_df)
    evidence_lookup = build_evidence_lookup(evidence_df)
    target_units = build_target_units(limit_queries=limit_queries)
    qrels_path = E6_LABELS_DIR / "qrels_evidence.jsonl"
    qrels_lookup = load_qrels_lookup(qrels_path)

    from chromadb import PersistentClient

    client = PersistentClient(path=cfg["embedding"]["chroma_persist_dir"])
    collection = client.get_collection(cfg["embedding"]["chroma_collection"])
    bi_encoder = SentenceTransformer(cfg["embedding"]["model"])
    reranker = CrossEncoder(cfg["reranker"]["model"])
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
        "qrels_hash": stable_hash(load_jsonl(qrels_path)),
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
            city_hotels = city_test_hotels[unit["city"]]
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

        summary = {
            "group_id": mode,
            "query_count": len({unit["query_id"] for unit in target_units}),
            "target_unit_count": len(target_units),
            "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 3),
            "config_hash": stable_hash(stable_run_config | {"retrieval_mode": mode}),
        }

        if experiment_id == "E6":
            summary.update(
                {
                    "aspect_recall_at_5": round(sum(row["aspect_recall_at_5"] for row in metric_rows) / max(len(metric_rows), 1), 4),
                    "ndcg_at_5": round(sum(row["ndcg_at_5"] for row in metric_rows) / max(len(metric_rows), 1), 4),
                    "mrr_at_5": round(sum(row["mrr_at_5"] for row in metric_rows) / max(len(metric_rows), 1), 4),
                    "precision_at_5": round(sum(row["precision_at_5"] for row in metric_rows) / max(len(metric_rows), 1), 4),
                    "evidence_diversity_at_5": round(sum(row["evidence_diversity_at_5"] for row in metric_rows) / max(len(metric_rows), 1), 4),
                }
            )
        elif experiment_id == "E7":
            summary.update(
                {
                    "ndcg_at_5": round(sum(row["ndcg_at_5"] for row in metric_rows) / max(len(metric_rows), 1), 4),
                    "mrr_at_5": round(sum(row["mrr_at_5"] for row in metric_rows) / max(len(metric_rows), 1), 4),
                    "precision_at_5": round(sum(row["precision_at_5"] for row in metric_rows) / max(len(metric_rows), 1), 4),
                }
            )
        else:
            summary.update(
                {
                    "ndcg_at_5": round(sum(row["ndcg_at_5"] for row in metric_rows) / max(len(metric_rows), 1), 4),
                    "precision_at_5": round(sum(row["precision_at_5"] for row in metric_rows) / max(len(metric_rows), 1), 4),
                    "evidence_insufficiency_rate": round(sum(insufficiency_flags) / max(len(insufficiency_flags), 1), 4),
                    "fallback_activation_rate": round(sum(fallback_activation_flags) / max(len(fallback_activation_flags), 1), 4),
                    "fallback_noise_rate": round(
                        fallback_sentence_nonrelevant / max(fallback_sentence_total, 1),
                        4,
                    ),
                }
            )

        summary_rows.append(summary)

    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as handle:
        for row in log_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")
    write_analysis_md(experiment_id, run_dir, summary_rows, log_rows)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=["build_qrels_pool", "freeze_qrels", "run_e6", "run_e7", "run_e8"],
        required=True,
    )
    parser.add_argument("--output-root", default=str(EXPERIMENT_RUNS_DIR))
    parser.add_argument("--limit-queries", type=int, default=None)
    args = parser.parse_args()

    if args.action == "build_qrels_pool":
        path = build_e6_qrels_pool(limit_queries=args.limit_queries)
        print(f"[OK] qrels pool written to {path}")
        return
    if args.action == "freeze_qrels":
        path = freeze_e6_qrels()
        print(f"[OK] qrels frozen to {path}")
        return

    output_root = Path(args.output_root)
    if args.action == "run_e6":
        run_dir = run_retrieval_eval("E6", output_root=output_root, limit_queries=args.limit_queries)
    elif args.action == "run_e7":
        run_dir = run_retrieval_eval("E7", output_root=output_root, limit_queries=args.limit_queries)
    else:
        run_dir = run_retrieval_eval("E8", output_root=output_root, limit_queries=args.limit_queries)
    print(f"[OK] run saved to {run_dir}")


if __name__ == "__main__":
    main()

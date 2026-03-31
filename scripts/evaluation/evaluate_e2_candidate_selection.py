"""Minimal candidate ranking and retrieval benchmark for Aspect-KB E2."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from scripts.shared.experiment_schemas import RunLogEntry
from scripts.shared.experiment_utils import EXPERIMENT_ASSETS_DIR, load_jsonl, stable_hash, utc_now_iso
from scripts.shared.project_utils import load_config


ALLOWED_QUERY_TYPES = {
    "single_aspect",
    "multi_aspect",
    "focus_and_avoid",
    "multi_aspect_strong",
}
DEFAULT_CANDIDATE_MODES = ["A_rating_review_count", "B_final_aspect_score"]
OPTIONAL_CANDIDATE_MODES = ["C_no_controversy_penalty"]
RETRIEVAL_MODE = "aspect_filtered_dense_no_rerank"


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build_hotel_summary(review_df: pd.DataFrame) -> pd.DataFrame:
    return (
        review_df.groupby("hotel_id", as_index=False)
        .agg(
            hotel_name=("hotel_name", "first"),
            city=("city", "first"),
            avg_rating=("rating", "mean"),
            review_count=("review_id", "nunique"),
        )
    )


def build_profile_tables(profile_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = profile_df.pivot(index="hotel_id", columns="aspect", values="final_aspect_score").fillna(0.0)
    no_penalty_df = profile_df.copy()
    no_penalty_df["alt_score_no_penalty"] = (
        no_penalty_df["recency_weighted_pos"] - no_penalty_df["recency_weighted_neg"]
    )
    alt = no_penalty_df.pivot(index="hotel_id", columns="aspect", values="alt_score_no_penalty").fillna(0.0)
    return current, alt


def candidate_rank(
    city_hotels: pd.DataFrame,
    profile_current: pd.DataFrame,
    profile_alt: pd.DataFrame,
    focus_aspects: list[str],
    avoid_aspects: list[str],
    mode: str,
) -> pd.DataFrame:
    df = city_hotels.copy()
    if mode == "A_rating_review_count":
        max_reviews = max(df["review_count"].max(), 1)
        review_norm = df["review_count"] / max_reviews
        df["score_total"] = df["avg_rating"] + review_norm
        df["score_breakdown"] = [
            {
                "avg_rating": round(float(avg_rating), 4),
                "review_count_norm": round(float(review_norm_value), 4),
            }
            for avg_rating, review_norm_value in zip(df["avg_rating"], review_norm)
        ]
    else:
        source = profile_current if mode == "B_final_aspect_score" else profile_alt
        rows = []
        breakdowns = []
        for _, row in df.iterrows():
            hotel_id = row["hotel_id"]
            total = 0.0
            breakdown = {}
            if hotel_id in source.index:
                for aspect in focus_aspects:
                    value = float(source.loc[hotel_id].get(aspect, 0.0))
                    total += value
                    breakdown[f"focus_{aspect}"] = round(value, 4)
                for aspect in avoid_aspects:
                    value = float(source.loc[hotel_id].get(aspect, 0.0))
                    total -= value
                    breakdown[f"avoid_{aspect}"] = round(-value, 4)
            breakdown["score_total"] = round(total, 4)
            rows.append(total)
            breakdowns.append(breakdown)
        df["score_total"] = rows
        df["score_breakdown"] = breakdowns
    return df.sort_values(["score_total", "avg_rating", "review_count"], ascending=False).reset_index(drop=True)


def review_id_from_sentence_id(sentence_id: str) -> str:
    if "_s" in sentence_id:
        return sentence_id.rsplit("_s", 1)[0]
    return sentence_id.split("_", 1)[0]


def retrieve_support(
    collection,
    model,
    query_en: str,
    hotel_id: str,
    city: str,
    focus_aspects: list[str],
    top_k: int,
    normalize_embeddings: bool,
) -> tuple[dict[str, dict], int]:
    embedding = model.encode([query_en], normalize_embeddings=normalize_embeddings).tolist()
    support: dict[str, dict] = {}
    calls = 0
    for aspect in focus_aspects:
        result = collection.query(
            query_embeddings=embedding,
            n_results=top_k,
            where={"$and": [{"hotel_id": str(hotel_id)}, {"city": city}, {"aspect": aspect}]},
            include=["distances"],
        )
        sentence_ids = result.get("ids", [[]])[0]
        distances = [round(float(value), 6) for value in result.get("distances", [[]])[0]]
        review_ids = sorted({review_id_from_sentence_id(sentence_id) for sentence_id in sentence_ids})
        support[aspect] = {
            "hotel_id": str(hotel_id),
            "sentence_ids": sentence_ids,
            "distances": distances,
            "review_ids": review_ids,
            "returned_count": len(sentence_ids),
            "unique_review_count": len(review_ids),
        }
        calls += 1
    return support, calls


def evaluate_candidate_support(
    support: dict[str, dict],
    min_sentence_support: int,
    min_unique_reviews: int,
) -> tuple[bool, str]:
    failures = []
    for aspect, detail in support.items():
        if detail["returned_count"] < min_sentence_support:
            failures.append(f"{aspect}: sentence_count<{min_sentence_support}")
        elif detail["unique_review_count"] < min_unique_reviews:
            failures.append(f"{aspect}: unique_reviews<{min_unique_reviews}")
    if failures:
        return False, "; ".join(failures)
    return True, "all focus aspects satisfy proxy support"


def warm_up_retrieval(collection, model, normalize_embeddings: bool) -> None:
    warmup_embedding = model.encode(
        ["hotel recommendation warmup query"],
        normalize_embeddings=normalize_embeddings,
    ).tolist()
    collection.query(
        query_embeddings=warmup_embedding,
        n_results=1,
        include=["distances"],
    )


def run_e2(
    output_root: Path,
    limit_queries: int | None = None,
    include_ablation: bool = False,
) -> Path:
    cfg = load_config()
    split_manifest = load_json(EXPERIMENT_ASSETS_DIR / "frozen_split_manifest.json")
    judged_queries = load_jsonl(EXPERIMENT_ASSETS_DIR / "judged_queries.jsonl")
    slot_gold = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "slot_gold.jsonl")}
    clarify_gold = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "clarify_gold.jsonl")}

    review_df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
    profile_df = pd.read_pickle("data/intermediate/hotel_profiles.pkl")
    evidence_df = pd.read_pickle("data/intermediate/evidence_index.pkl")

    hotel_summary = build_hotel_summary(review_df)
    profile_current, profile_alt = build_profile_tables(profile_df)
    hotel_summary = hotel_summary[hotel_summary["hotel_id"].isin(set(split_manifest["splits"]["test"]))].copy()

    from chromadb import PersistentClient

    client = PersistentClient(path=cfg["embedding"]["chroma_persist_dir"])
    collection = client.get_collection(cfg["embedding"]["chroma_collection"])
    model = SentenceTransformer(cfg["embedding"]["model"])
    normalize_embeddings = bool(cfg["embedding"].get("normalize", True))
    warm_up_retrieval(collection, model, normalize_embeddings)

    eligible_queries = []
    for row in judged_queries:
        slot = slot_gold[row["query_id"]]
        clarify = clarify_gold[row["query_id"]]
        if clarify["clarify_needed"]:
            continue
        if not slot["city"] or not slot["focus_aspects"]:
            continue
        if row["query_type"] not in ALLOWED_QUERY_TYPES:
            continue
        eligible_queries.append((row, slot))
    if limit_queries:
        eligible_queries = eligible_queries[:limit_queries]

    stable_run_config = {
        "task": "E2",
        "split_config_hash": split_manifest["meta"]["config_hash"],
        "query_types": sorted(ALLOWED_QUERY_TYPES),
        "query_count": len(eligible_queries),
        "top_n": 5,
        "min_sentence_support": 2,
        "min_unique_reviews": 2,
        "retrieval_mode": RETRIEVAL_MODE,
        "dense_top_k": cfg["reranker"]["top_k_before_rerank"],
        "embedding_model": cfg["embedding"]["model"],
        "collection": cfg["embedding"]["chroma_collection"],
        "warmup_applied": True,
    }
    run_started_at = utc_now_iso()
    run_id = f"e2_{stable_hash(stable_run_config)}_{run_started_at.replace(':', '').replace('-', '')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate_modes = list(DEFAULT_CANDIDATE_MODES)
    if include_ablation:
        candidate_modes.extend(OPTIONAL_CANDIDATE_MODES)
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "generated_at": run_started_at,
                "stable_run_config": stable_run_config,
                "candidate_modes": candidate_modes,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    log_rows = []
    summary_rows = []
    for mode in candidate_modes:
        hit_flags = []
        latency_values = []
        retrieval_calls = []
        dense_returned_values = []
        dense_unique_review_values = []
        candidates_checked_values = []

        for query_row, slot in eligible_queries:
            city_hotels = hotel_summary[hotel_summary["city"] == slot["city"]].copy()
            start = time.perf_counter()
            ranked = candidate_rank(city_hotels, profile_current, profile_alt, slot["focus_aspects"], slot["avoid_aspects"], mode)
            top_candidates = ranked.head(5)
            winner = None
            winner_reason = "no candidate satisfied proxy support"
            support_payload = {}
            calls_total = 0
            candidate_records = []
            aspect_returned_counts = []
            aspect_unique_review_counts = []
            for _, candidate in top_candidates.iterrows():
                support, calls = retrieve_support(
                    collection,
                    model,
                    slot["query_en"],
                    candidate["hotel_id"],
                    slot["city"],
                    slot["focus_aspects"],
                    cfg["reranker"]["top_k_before_rerank"],
                    normalize_embeddings,
                )
                support_payload[candidate["hotel_id"]] = support
                calls_total += calls
                for detail in support.values():
                    aspect_returned_counts.append(detail["returned_count"])
                    aspect_unique_review_counts.append(detail["unique_review_count"])
                proxy_pass, proxy_reason = evaluate_candidate_support(
                    support,
                    min_sentence_support=stable_run_config["min_sentence_support"],
                    min_unique_reviews=stable_run_config["min_unique_reviews"],
                )
                candidate_record = {
                    "hotel_id": candidate["hotel_id"],
                    "hotel_name": candidate["hotel_name"],
                    "score_total": round(float(candidate["score_total"]), 4),
                    "score_breakdown": candidate["score_breakdown"],
                    "support_by_aspect": support,
                    "proxy_pass": proxy_pass,
                    "proxy_reason": proxy_reason,
                }
                candidate_records.append(candidate_record)
                if proxy_pass:
                    winner = candidate["hotel_id"]
                    winner_reason = proxy_reason
                    break
            latency_ms = round((time.perf_counter() - start) * 1000, 3)
            hit_flags.append(int(winner is not None))
            latency_values.append(latency_ms)
            retrieval_calls.append(calls_total)
            dense_returned_values.append(
                round(sum(aspect_returned_counts) / max(len(aspect_returned_counts), 1), 4)
            )
            dense_unique_review_values.append(
                round(sum(aspect_unique_review_counts) / max(len(aspect_unique_review_counts), 1), 4)
            )
            candidates_checked_values.append(len(candidate_records))

            log_entry = RunLogEntry(
                run_id=run_id,
                group_id=mode,
                query_id=query_row["query_id"],
                retrieval_mode=RETRIEVAL_MODE,
                candidate_mode=mode,
                config_hash=stable_hash(stable_run_config | {"candidate_mode": mode}),
                latency_ms=latency_ms,
                intermediate_objects={
                    "query": query_row,
                    "slot_gold": slot,
                    "top_candidates": candidate_records,
                    "support_payload": support_payload,
                    "winner_hotel_id": winner,
                    "winner_reason": winner_reason,
                    "candidates_checked": len(candidate_records),
                    "retrieval_calls": calls_total,
                },
            )
            log_rows.append(log_entry.model_dump())

        summary_rows.append(
            {
                "candidate_mode": mode,
                "query_count": len(hit_flags),
                "candidate_hit_at_5_proxy": round(sum(hit_flags) / max(len(hit_flags), 1), 4),
                "avg_latency_ms": round(sum(latency_values) / max(len(latency_values), 1), 3),
                "avg_retrieval_calls": round(sum(retrieval_calls) / max(len(retrieval_calls), 1), 3),
                "avg_candidates_checked": round(sum(candidates_checked_values) / max(len(candidates_checked_values), 1), 3),
                "avg_dense_returned_per_aspect": round(sum(dense_returned_values) / max(len(dense_returned_values), 1), 3),
                "avg_dense_unique_reviews_per_aspect": round(sum(dense_unique_review_values) / max(len(dense_unique_review_values), 1), 3),
                "config_hash": stable_hash(stable_run_config | {"candidate_mode": mode}),
            }
        )

    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as handle:
        for row in log_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# E2 Candidate Ranking Result",
        "",
        "> 当前结果使用 aspect-filtered dense retrieval 作为最小证据检索模块，`Candidate Hit@5` 采用“每个目标方面至少 2 句、且来自至少 2 个不同 review”的代理规则。最终论文版仍建议在 E6 qrels 完成后复算。",
        "",
    ]
    for row in summary_rows:
        lines.extend(
            [
                f"## {row['candidate_mode']}",
                "",
                f"- Candidate Hit@5 (proxy): {row['candidate_hit_at_5_proxy']}",
                f"- Avg latency (ms): {row['avg_latency_ms']}",
                f"- Avg retrieval calls: {row['avg_retrieval_calls']}",
                f"- Avg candidates checked: {row['avg_candidates_checked']}",
                f"- Avg dense returned / aspect: {row['avg_dense_returned_per_aspect']}",
                f"- Avg dense unique reviews / aspect: {row['avg_dense_unique_reviews_per_aspect']}",
                f"- Config hash: `{row['config_hash']}`",
                "",
            ]
        )
    (run_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="experiments/runs")
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--include-ablation", action="store_true")
    args = parser.parse_args()

    run_dir = run_e2(
        Path(args.output_root),
        limit_queries=args.limit_queries,
        include_ablation=args.include_ablation,
    )
    print(f"[OK] E2 outputs written to {run_dir}")


if __name__ == "__main__":
    main()

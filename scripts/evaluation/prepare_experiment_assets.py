"""Prepare frozen split/query assets and E1 bootstrap files for Aspect-KB experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from scripts.shared.experiment_utils import (
    E1_LABELS_DIR,
    EXPERIMENT_ASSETS_DIR,
    EXPERIMENT_RUNS_DIR,
    city_state_map,
    ensure_dir,
    stable_hash,
    utc_now_iso,
    write_json,
    write_jsonl,
    write_yaml,
)
from scripts.shared.project_utils import ASPECT_CATEGORIES, ensure_columns, load_config, log_step


SEED = 20260326
SPLIT_RATIOS = {"train": 0.70, "dev": 0.15, "test": 0.15}

ASPECT_ZH = {
    "location_transport": "位置交通",
    "cleanliness": "卫生干净",
    "service": "服务",
    "room_facilities": "房间设施",
    "quiet_sleep": "安静睡眠",
    "value": "性价比",
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


def deterministic_order(values: list[str], salt: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest(),
    )


def allocate_counts(total: int) -> dict[str, int]:
    if total <= 0:
        return {"train": 0, "dev": 0, "test": 0}
    if total == 1:
        return {"train": 1, "dev": 0, "test": 0}
    if total == 2:
        return {"train": 1, "dev": 0, "test": 1}

    counts = {"train": 1, "dev": 1, "test": 1}
    remaining = total - 3
    raw = {name: SPLIT_RATIOS[name] * remaining for name in counts}
    for name, value in raw.items():
        counts[name] += int(value)
    remainder = total - sum(counts.values())
    fractions = sorted(
        counts,
        key=lambda name: (raw[name] - int(raw[name]), SPLIT_RATIOS[name]),
        reverse=True,
    )
    for idx in range(remainder):
        counts[fractions[idx % len(fractions)]] += 1
    return counts


def build_split_manifest(review_df: pd.DataFrame) -> dict:
    hotel_df = (
        review_df.groupby("hotel_id", as_index=False)
        .agg(
            hotel_name=("hotel_name", "first"),
            city=("city", "first"),
            state=("state", "first"),
            review_count=("review_id", "nunique"),
        )
        .sort_values(["city", "hotel_id"])
    )

    split_lookup: dict[str, str] = {}
    split_lists = {"train": [], "dev": [], "test": []}
    city_summary: dict[str, dict[str, int]] = {}

    for city, group in hotel_df.groupby("city", sort=True):
        hotel_ids = deterministic_order(group["hotel_id"].tolist(), f"{SEED}|{city}")
        counts = allocate_counts(len(hotel_ids))
        city_summary[city] = counts.copy()

        cursor = 0
        for split_name in ["train", "dev", "test"]:
            for hotel_id in hotel_ids[cursor : cursor + counts[split_name]]:
                split_lookup[hotel_id] = split_name
                split_lists[split_name].append(hotel_id)
            cursor += counts[split_name]

    hotel_rows = []
    for _, row in hotel_df.iterrows():
        hotel_rows.append(
            {
                "hotel_id": row["hotel_id"],
                "hotel_name": row["hotel_name"],
                "city": row["city"],
                "state": row["state"],
                "review_count": int(row["review_count"]),
                "split": split_lookup[row["hotel_id"]],
            }
        )

    payload = {
        "meta": {
            "seed": SEED,
            "ratios": SPLIT_RATIOS,
            "generated_at": utc_now_iso(),
            "hotel_count": len(hotel_rows),
        },
        "splits": split_lists,
        "hotels": hotel_rows,
        "city_summary": city_summary,
    }
    payload["meta"]["config_hash"] = stable_hash(
        {
            "seed": SEED,
            "ratios": SPLIT_RATIOS,
            "splits": split_lists,
            "hotels": hotel_rows,
        }
    )
    return payload


def join_en_parts(city: str | None, focus: list[str], avoid: list[str], unsupported: list[str]) -> str:
    bits: list[str] = []
    bits.append(f"hotel in {city}" if city else "hotel")
    if focus:
        bits.append("with " + " and ".join(ASPECT_EN[item] for item in focus))
    if avoid:
        bits.append("avoiding " + " and ".join(ASPECT_EN_AVOID[item] for item in avoid))
    if "budget" in unsupported:
        bits.append("with a budget requirement")
    if "distance_to_landmark" in unsupported:
        bits.append("close to a landmark")
    if "checkin_date" in unsupported:
        bits.append("for a specific check-in date")
    return " ".join(bits)


def build_query_assets(review_df: pd.DataFrame) -> tuple[list[dict], list[dict], list[dict]]:
    city_to_state = city_state_map(review_df)
    cities = sorted(city_to_state)

    judged: list[dict] = []
    slot_gold: list[dict] = []
    clarify_gold: list[dict] = []

    def add_query(
        query_id: str,
        query_text_zh: str,
        query_type: str,
        city: str | None,
        focus: list[str],
        avoid: list[str],
        unsupported: list[str],
        clarify_needed: bool,
        clarify_reason: str,
        target_slots: list[str],
    ) -> None:
        judged.append(
            {
                "query_id": query_id,
                "turn_type": "initial",
                "query_text_zh": query_text_zh,
                "query_type": query_type,
                "city_hint": city,
            }
        )
        slot_gold.append(
            {
                "query_id": query_id,
                "city": city,
                "state": city_to_state.get(city) if city else None,
                "hotel_category": None,
                "focus_aspects": focus,
                "avoid_aspects": avoid,
                "unsupported_requests": unsupported,
                "query_en": join_en_parts(city, focus, avoid, unsupported),
            }
        )
        clarify_gold.append(
            {
                "query_id": query_id,
                "clarify_needed": clarify_needed,
                "clarify_reason": clarify_reason,
                "target_slots": target_slots,
            }
        )

    idx = 1
    for city_idx, city in enumerate(cities):
        a1 = ASPECT_CATEGORIES[city_idx % len(ASPECT_CATEGORIES)]
        a2 = ASPECT_CATEGORIES[(city_idx + 1) % len(ASPECT_CATEGORIES)]
        a3 = ASPECT_CATEGORIES[(city_idx + 2) % len(ASPECT_CATEGORIES)]

        add_query(f"q{idx:03d}", f"我想在{city}找一家{ASPECT_ZH[a1]}比较好的酒店。", "single_aspect", city, [a1], [], [], False, "", [])
        idx += 1
        add_query(f"q{idx:03d}", f"请推荐{city}{ASPECT_ZH[a1]}和{ASPECT_ZH[a2]}都不错的酒店。", "multi_aspect", city, [a1, a2], [], [], False, "", [])
        idx += 1
        add_query(f"q{idx:03d}", f"我在{city}想住得安静一点，但不要{ASPECT_ZH[a3]}太差的酒店。", "focus_and_avoid", city, ["quiet_sleep"], [a3], [], False, "", [])
        idx += 1
        add_query(f"q{idx:03d}", f"帮我找{city}预算在 600 元以内，而且{ASPECT_ZH[a1]}不错的酒店。", "unsupported_budget", city, [a1], [], ["budget"], False, "", [])
        idx += 1
        add_query(f"q{idx:03d}", f"我想在{city}找一家离景点步行 10 分钟内、而且{ASPECT_ZH[a2]}好的酒店。", "unsupported_distance", city, [a2], [], ["distance_to_landmark"], False, "", [])
        idx += 1

    for aspect in ASPECT_CATEGORIES:
        add_query(f"q{idx:03d}", f"我想找一家{ASPECT_ZH[aspect]}好的酒店，你先帮我想想。", "missing_city", None, [aspect], [], [], True, "missing_city", ["city"])
        idx += 1

    for city_idx, city in enumerate(cities):
        aspect = ASPECT_CATEGORIES[city_idx % len(ASPECT_CATEGORIES)]
        add_query(
            f"q{idx:03d}",
            f"我想在{city}找一家{ASPECT_ZH[aspect]}很好，但又最好别太强调{ASPECT_ZH[aspect]}的酒店。",
            "conflict",
            city,
            [aspect],
            [aspect],
            [],
            True,
            "aspect_conflict",
            ["focus_aspects", "avoid_aspects"],
        )
        idx += 1

    unsupported_templates = [
        ("budget", "预算控制在 500 元以内"),
        ("distance_to_landmark", "离景点步行 10 分钟以内"),
        ("checkin_date", "下周五能入住"),
    ]
    for city_idx, city in enumerate(cities):
        unsupported_kind, unsupported_text = unsupported_templates[city_idx % len(unsupported_templates)]
        aspect = ASPECT_CATEGORIES[(city_idx + 3) % len(ASPECT_CATEGORIES)]
        add_query(
            f"q{idx:03d}",
            f"我想去{city}，要求{unsupported_text}，同时{ASPECT_ZH[aspect]}也要好。",
            "unsupported_heavy",
            city,
            [aspect],
            [],
            [unsupported_kind],
            False,
            "",
            [],
        )
        idx += 1

    for city_idx, city in enumerate(cities):
        aspects = [
            ASPECT_CATEGORIES[city_idx % len(ASPECT_CATEGORIES)],
            ASPECT_CATEGORIES[(city_idx + 2) % len(ASPECT_CATEGORIES)],
            ASPECT_CATEGORIES[(city_idx + 4) % len(ASPECT_CATEGORIES)],
        ]
        zh_bits = "、".join(ASPECT_ZH[item] for item in aspects)
        add_query(f"q{idx:03d}", f"请推荐{city}在{zh_bits}三方面都比较均衡的酒店。", "multi_aspect_strong", city, aspects, [], [], False, "", [])
        idx += 1

    return judged, slot_gold, clarify_gold


def build_e1_sample(
    sent_df: pd.DataFrame,
    aspect_df: pd.DataFrame,
    split_manifest: dict,
) -> tuple[pd.DataFrame, dict]:
    test_hotels = set(split_manifest["splits"]["test"])
    sent_subset = sent_df[sent_df["hotel_id"].isin(test_hotels)].copy()
    aspect_subset = aspect_df[aspect_df["hotel_id"].isin(test_hotels)].copy()
    core_rows = aspect_subset[aspect_subset["aspect"].isin(ASPECT_CATEGORIES)].copy()
    sentence_meta = (
        sent_subset[["sentence_id", "hotel_id", "sentence_text", "city"]]
        .drop_duplicates("sentence_id")
        .copy()
    )

    selected_ids: set[str] = set()
    samples: list[pd.DataFrame] = []
    difficult = (
        core_rows.groupby("sentence_id")["aspect"]
        .nunique()
        .reset_index(name="core_aspect_count")
    )
    difficult = difficult[difficult["core_aspect_count"] >= 2]
    difficult = difficult.merge(
        sent_subset[["sentence_id", "hotel_id", "sentence_text", "city"]],
        on="sentence_id",
        how="left",
        validate="one_to_one",
    )
    difficult = difficult.merge(
        core_rows.groupby("sentence_id")["aspect"]
        .apply(lambda values: ";".join(sorted(set(values))))
        .reset_index(name="existing_aspects"),
        on="sentence_id",
        how="left",
    )
    difficult = difficult.drop_duplicates("sentence_id").dropna(subset=["hotel_id"])
    difficult = difficult.sample(n=min(60, len(difficult)), random_state=SEED, replace=False)
    difficult["target_aspect"] = "multi"
    difficult["sample_bucket"] = "difficult"
    difficult["sentiment"] = ""
    difficult = difficult[
        ["sentence_id", "hotel_id", "target_aspect", "sentiment", "sample_bucket", "existing_aspects", "sentence_text", "city"]
    ]
    selected_ids.update(difficult["sentence_id"].tolist())
    samples.append(difficult)

    aspect_summary: dict[str, dict[str, int]] = {}
    for offset, aspect in enumerate(ASPECT_CATEGORIES, start=1):
        aspect_sentences = (
            core_rows[core_rows["aspect"] == aspect][["sentence_id", "hotel_id", "aspect", "sentiment"]]
            .drop_duplicates("sentence_id")
        )
        aspect_sentences = aspect_sentences[~aspect_sentences["sentence_id"].isin(selected_ids)]
        take = min(50, len(aspect_sentences))
        aspect_summary[aspect] = {"target": 50, "actual": take, "shortage": max(0, 50 - take)}
        if take == 0:
            continue
        picked = aspect_sentences.sample(n=take, random_state=SEED + offset, replace=False)
        picked = picked.rename(columns={"aspect": "target_aspect"})
        picked["sample_bucket"] = "core"
        picked = picked.merge(
            sentence_meta,
            on=["sentence_id", "hotel_id"],
            how="left",
            validate="one_to_one",
        )
        picked["existing_aspects"] = picked["target_aspect"]
        picked = picked[
            ["sentence_id", "hotel_id", "target_aspect", "sentiment", "sample_bucket", "existing_aspects", "sentence_text", "city"]
        ]
        selected_ids.update(picked["sentence_id"].tolist())
        samples.append(picked)

    final_df = (
        pd.concat(samples, ignore_index=True)
        .drop_duplicates("sentence_id")
        .sort_values(["sample_bucket", "target_aspect", "sentence_id"])
        .reset_index(drop=True)
    )
    summary = {
        "target_total": 360,
        "actual_total": len(final_df),
        "difficult_target": 60,
        "difficult_actual": len(difficult),
        "difficult_shortage": max(0, 60 - len(difficult)),
        "per_aspect": aspect_summary,
    }
    return final_df, summary


def build_gold_template(sample_df: pd.DataFrame, existing_path: Path | None = None) -> pd.DataFrame:
    gold_df = sample_df[["sentence_id", "hotel_id", "city", "sentence_text"]].copy()
    gold_df["aspect_gold"] = ""
    gold_df["sentiment_gold"] = ""
    gold_df["is_multi_aspect"] = ""
    gold_df["notes"] = ""
    if existing_path and existing_path.exists():
        existing_df = pd.read_csv(existing_path, keep_default_na=False)
        required = {"sentence_id", "aspect_gold", "sentiment_gold", "is_multi_aspect", "notes"}
        if required.issubset(existing_df.columns):
            existing_df = existing_df[["sentence_id", "aspect_gold", "sentiment_gold", "is_multi_aspect", "notes"]]
            existing_df = existing_df.drop_duplicates("sentence_id")
            gold_df = gold_df.merge(existing_df, on="sentence_id", how="left", suffixes=("", "_existing"))
            for col in ["aspect_gold", "sentiment_gold", "is_multi_aspect", "notes"]:
                gold_df[col] = gold_df[f"{col}_existing"].fillna(gold_df[col])
                gold_df = gold_df.drop(columns=[f"{col}_existing"])
    return gold_df


def build_frozen_config(cfg: dict) -> dict:
    return {
        "scope": {
            "cities": cfg["data"]["experiment_cities"],
            "core_aspects": cfg["aspect"]["categories"],
            "single_clarification": True,
            "single_feedback_update": True,
            "knowledge_source": "English hotel reviews knowledge base",
            "explicitly_unsupported": ["budget", "distance_to_landmark", "checkin_date"],
        },
        "split": {"seed": SEED, "by": "hotel_id", "ratios": SPLIT_RATIOS},
        "retrieval": {
            "collection": cfg["embedding"]["chroma_collection"],
            "embedding_model": cfg["embedding"]["model"],
            "reranker_model": cfg["reranker"]["model"],
            "top_k_before_rerank": cfg["reranker"]["top_k_before_rerank"],
            "top_k_after_rerank": cfg["reranker"]["top_k_after_rerank"],
        },
        "workflow": {
            "default_retrieval_mode": cfg["workflow"]["default_retrieval_mode"],
            "enable_fallback": bool(cfg["workflow"]["enable_fallback"]),
        },
        "behavior": {
            "base_model": cfg["behavior"]["base_model"],
            "prompt_versions": cfg["behavior"]["prompt_versions"],
        },
        "generated_at": utc_now_iso(),
    }


def write_e1_report_template(path: Path, sample_df: pd.DataFrame, summary: dict) -> None:
    aspect_counts = sample_df["target_aspect"].value_counts().to_dict()
    shortage_lines = []
    if summary["difficult_shortage"]:
        shortage_lines.append(
            f"- difficult shortage: {summary['difficult_shortage']} (target=60, actual={summary['difficult_actual']})"
        )
    for aspect in ASPECT_CATEGORIES:
        item = summary["per_aspect"][aspect]
        if item["shortage"]:
            shortage_lines.append(
                f"- {aspect}: shortage {item['shortage']} (target={item['target']}, actual={item['actual']})"
            )
    if not shortage_lines:
        shortage_lines.append("- none")
    content = f"""# E1 Report Template

## Status

- [x] `aspect_sentiment_eval_sample.csv` generated
- [ ] `aspect_sentiment_gold.csv` manually annotated (pending)
- [ ] rule-only / zeroshot-only / hybrid predictions completed
- [ ] metrics and confusion matrix exported

## Sample Summary

- Total sampled sentences: {len(sample_df)} / target {summary['target_total']}
- Distribution: {aspect_counts}

## Sampling Shortage

{chr(10).join(shortage_lines)}

## Planned Outputs

- Aspect macro-F1
- Difficult-set multi-label Jaccard
- Sentiment macro-F1
- Confusion matrix
- Representative error cases

## Notes

当前样本采用“困难集优先 + 核心方面去重补齐”的强控制策略，不再使用 silent topup 掩盖配额不足。

将人工标注完成后的 `aspect_sentiment_gold.csv` 作为唯一金标输入，再运行 `python -m scripts.evaluation.evaluate_e1_aspect_reliability` 产出正式结果。
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    cfg = load_config()
    review_df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
    sent_df = pd.read_pickle("data/intermediate/sentences.pkl")
    aspect_df = pd.read_pickle("data/intermediate/aspect_sentiment.pkl")

    ensure_columns(review_df, ["hotel_id", "hotel_name", "city", "state", "review_id"], "cleaned_reviews.pkl")
    ensure_columns(sent_df, ["sentence_id", "hotel_id", "sentence_text", "city"], "sentences.pkl")
    ensure_columns(aspect_df, ["sentence_id", "hotel_id", "aspect", "sentiment"], "aspect_sentiment.pkl")

    assets_dir = ensure_dir(EXPERIMENT_ASSETS_DIR)
    e1_dir = ensure_dir(E1_LABELS_DIR)
    ensure_dir(EXPERIMENT_RUNS_DIR)

    log_step("EXP-A", "生成 frozen_config.yaml")
    write_yaml(assets_dir / "frozen_config.yaml", build_frozen_config(cfg))

    log_step("EXP-B", "生成 frozen_split_manifest.json")
    split_manifest = build_split_manifest(review_df)
    write_json(assets_dir / "frozen_split_manifest.json", split_manifest)

    log_step("EXP-C", "生成 judged_queries / slot_gold / clarify_gold")
    judged, slot_gold, clarify_gold = build_query_assets(review_df)
    write_jsonl(assets_dir / "judged_queries.jsonl", judged)
    write_jsonl(assets_dir / "slot_gold.jsonl", slot_gold)
    write_jsonl(assets_dir / "clarify_gold.jsonl", clarify_gold)

    log_step("EXP-D", "生成 E1 样本与标注模板")
    sample_df, e1_summary = build_e1_sample(sent_df, aspect_df, split_manifest)
    sample_df.to_csv(e1_dir / "aspect_sentiment_eval_sample.csv", index=False, encoding="utf-8-sig")
    gold_path = e1_dir / "aspect_sentiment_gold.csv"
    build_gold_template(sample_df, gold_path).to_csv(gold_path, index=False, encoding="utf-8-sig")
    write_e1_report_template(e1_dir / "e1_report.md", sample_df, e1_summary)

    print("\n[OK] 实验底座资产已生成")
    print(f"   split hotels: train={len(split_manifest['splits']['train'])}, dev={len(split_manifest['splits']['dev'])}, test={len(split_manifest['splits']['test'])}")
    print(f"   queries: {len(judged)}")
    print(f"   E1 sample size: {len(sample_df)}")


if __name__ == "__main__":
    main()

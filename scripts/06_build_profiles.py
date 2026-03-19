"""
06_build_profiles.py
步骤 12a：聚合酒店方面画像
输入:  data/intermediate/aspect_sentiment.pkl
        data/intermediate/cleaned_reviews.pkl
输出: data/intermediate/hotel_profiles.pkl
"""

from pathlib import Path

import pandas as pd

from utils import ASPECT_CATEGORIES, ensure_columns, load_config, log_step


def compute_profile_row(hotel_id: str, aspect: str, group: pd.DataFrame) -> dict:
    pos = group[group["sentiment"] == "positive"]
    neg = group[group["sentiment"] == "negative"]
    neu = group[group["sentiment"] == "neutral"]

    pos_count = int(len(pos))
    neg_count = int(len(neg))
    neu_count = int(len(neu))
    total_count = int(len(group))

    recency_weighted_pos = round(float(pos["recency_weight"].sum()), 3)
    recency_weighted_neg = round(float(neg["recency_weight"].sum()), 3)
    controversy_score = round(
        min(pos_count, neg_count) / max(max(pos_count, neg_count), 1),
        3,
    )
    final_aspect_score = round(
        recency_weighted_pos - recency_weighted_neg - controversy_score * 0.3,
        3,
    )

    return {
        "hotel_id": hotel_id,
        "aspect": aspect,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "neu_count": neu_count,
        "total_count": total_count,
        "recency_weighted_pos": recency_weighted_pos,
        "recency_weighted_neg": recency_weighted_neg,
        "controversy_score": controversy_score,
        "final_aspect_score": final_aspect_score,
    }


def main():
    cfg = load_config()
    aspect_df = pd.read_pickle("data/intermediate/aspect_sentiment.pkl")
    review_df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
    ensure_columns(
        aspect_df,
        ["review_id", "hotel_id", "aspect", "sentiment"],
        "aspect_sentiment.pkl",
    )
    ensure_columns(review_df, ["review_id", "hotel_id", "recency_bucket"], "cleaned_reviews.pkl")
    log_step("STEP12a", f"标签: {len(aspect_df)}, 评论: {len(review_df)}")

    review_meta = review_df[["review_id", "recency_bucket"]].drop_duplicates("review_id")
    aspect_full = aspect_df.merge(review_meta, on="review_id", how="left", validate="many_to_one")
    aspect_full["recency_weight"] = aspect_full["recency_bucket"].map(
        cfg["recency"]["buckets"]
    ).fillna(cfg["recency"]["buckets"].get("unknown", 1.0))

    core_df = aspect_full[aspect_full["aspect"].isin(ASPECT_CATEGORIES)].copy()
    hotel_ids = sorted(review_df["hotel_id"].unique().tolist())

    rows = []
    for hotel_id in hotel_ids:
        hotel_group = core_df[core_df["hotel_id"] == hotel_id]
        for aspect in ASPECT_CATEGORIES:
            aspect_group = hotel_group[hotel_group["aspect"] == aspect]
            rows.append(compute_profile_row(hotel_id, aspect, aspect_group))

    profile_df = pd.DataFrame(rows)
    expected_rows = len(hotel_ids) * len(ASPECT_CATEGORIES)
    if len(profile_df) != expected_rows:
        raise AssertionError(
            f"hotel_profiles 行数异常: {len(profile_df)} != {expected_rows}"
        )

    print(f"\n[OK] 酒店方面画像: {len(profile_df)} 行 ({len(hotel_ids)} 家酒店 × {len(ASPECT_CATEGORIES)} 方面)")
    print(f"   final_aspect_score 均值: {profile_df['final_aspect_score'].mean():.3f}")

    out_path = Path("data/intermediate/hotel_profiles.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile_df.to_pickle(out_path)
    print(f"[SAVE] {out_path} ({len(profile_df)} 行)")


if __name__ == "__main__":
    main()

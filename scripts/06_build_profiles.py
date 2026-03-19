"""
06_build_profiles.py
步骤 11：聚合酒店方面情感画像
输入:  data/intermediate/sentiment_classified.pkl
        data/intermediate/cleaned_reviews.pkl
输出: data/intermediate/hotel_profiles.pkl
"""

import pandas as pd
from pathlib import Path
from utils import load_config, log_step


ASPECTS = [
    "location_transport", "cleanliness", "service",
    "room_facilities", "quiet_sleep", "value", "general"
]


def compute_aspect_sentiment_vector(group: pd.DataFrame) -> dict:
    """计算单酒店的方面情感分布向量"""
    vec = {}
    for aspect in ASPECTS:
        sub = group[group["primary_aspect"] == aspect]
        if len(sub) == 0:
            vec[f"{aspect}_pos"] = None
            vec[f"{aspect}_neg"] = None
            vec[f"{aspect}_neu"] = None
            vec[f"{aspect}_n"]   = 0
            continue
        n = len(sub)
        vec[f"{aspect}_pos"] = round((sub["sentiment"] == "positive").sum() / n, 4)
        vec[f"{aspect}_neg"] = round((sub["sentiment"] == "negative").sum() / n, 4)
        vec[f"{aspect}_neu"] = round((sub["sentiment"] == "neutral").sum() / n, 4)
        vec[f"{aspect}_n"]   = n
    return vec


def main():
    cfg = load_config()
    sent_df    = pd.read_pickle("data/intermediate/sentiment_classified.pkl")
    reviews_df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
    log_step("STEP11", f"句子: {len(sent_df)}, 评论: {len(reviews_df)}")

    # 酒店元数据（从 reviews_df 聚合）
    hotel_meta = reviews_df.groupby("hotel_id").agg(
        hotel_name=("hotel_name",   "first"),
        city=       ("city",         "first"),
        state=      ("state",        "first"),
        address=    ("address",      "first"),
        latitude=   ("latitude",     "first"),
        longitude=  ("longitude",    "first"),
        n_reviews=  ("review_id",    "count"),
        avg_rating= ("rating",       "mean"),
        min_review_date=("review_date", "min"),
        max_review_date=("review_date", "max"),
    ).reset_index()
    hotel_meta["avg_rating"] = hotel_meta["avg_rating"].round(2)

    # 方面情感向量
    profile_rows = []
    for hotel_id, group in sent_df.groupby("hotel_id"):
        row = {"hotel_id": hotel_id, "n_sentences": len(group)}
        row.update(compute_aspect_sentiment_vector(group))
        # 主导情感（跨所有方面）
        all_sent = group["sentiment"]
        row["overall_pos"] = round((all_sent == "positive").sum() / len(group), 4)
        row["overall_neg"] = round((all_sent == "negative").sum() / len(group), 4)
        row["overall_neu"] = round((all_sent == "neutral").sum() / len(group), 4)
        profile_rows.append(row)

    profile_df = pd.DataFrame(profile_rows)
    hotel_profiles = hotel_meta.merge(profile_df, on="hotel_id", how="left")

    print(f"\n✅ 酒店画像: {len(hotel_profiles)} 家酒店")
    print(f"   均值 n_sentences: {hotel_profiles['n_sentences'].mean():.1f}")
    print(f"   overall_pos 均值: {hotel_profiles['overall_pos'].mean():.3f}")
    print(f"   overall_neg 均值: {hotel_profiles['overall_neg'].mean():.3f}")

    out_path = Path("data/intermediate/hotel_profiles.pkl")
    hotel_profiles.to_pickle(out_path)
    print(f"💾 保存: {out_path} ({len(hotel_profiles)} 行)")


if __name__ == "__main__":
    main()

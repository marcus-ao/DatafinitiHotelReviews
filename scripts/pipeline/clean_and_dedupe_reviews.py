"""
clean_and_dedupe_reviews.py
步骤 5-7：无效评论过滤 -> 管理者回复去除 -> 精确去重 -> 日期与评分标准化 -> 酒店过滤
输入:  data/intermediate/city_filtered.pkl
输出: data/intermediate/cleaned_reviews.pkl
"""

from pathlib import Path

import pandas as pd

from scripts.shared.project_utils import (
    assert_unique,
    assert_shape,
    assign_recency_bucket,
    ensure_columns,
    load_config,
    log_step,
    normalize_whitespace,
    rating_to_weak_sentiment,
    remove_manager_response,
    safe_parse_timestamp,
)


def main():
    cfg = load_config()
    df = pd.read_pickle("data/intermediate/city_filtered.pkl")
    ensure_columns(
        df,
        [
            "hotel_id",
            "review_id",
            "review_date_raw",
            "review_title",
            "review_text_raw",
            "rating",
            "hotel_name",
            "city",
        ],
        "city_filtered.pkl",
    )
    log_step("STEP5", f"输入: {len(df)} 行")

    log_step("STEP5a", "无效评论过滤")
    min_len = cfg["cleaning"]["min_text_length"]
    noise = set(cfg["cleaning"]["noise_exact_matches"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").round().astype("Int64")
    valid_mask = (
        df["review_text_raw"].notna()
        & (df["review_text_raw"].astype(str).str.strip().str.len() >= min_len)
        & (~df["review_text_raw"].astype(str).str.strip().isin(noise))
        & df["rating"].between(1, 5)
    )
    df = df[valid_mask].copy()
    print(f"[OK] 有效评论保留: {len(df)} 条")

    log_step("STEP6", "去除管理者回复并清洗文本")
    results = df["review_text_raw"].apply(
        lambda text: remove_manager_response(text, cfg["cleaning"]["min_preserve_length"])
    )
    df["review_text_clean"] = [normalize_whitespace(item[0]) for item in results]
    df["has_manager_reply"] = [item[1] for item in results]
    df["char_len_raw"] = df["review_text_raw"].astype(str).str.len()
    df["char_len_clean"] = df["review_text_clean"].str.len()
    print(
        f"[OK] 检测到管理者回复: {int(df['has_manager_reply'].sum())} / {len(df)} "
        f"({df['has_manager_reply'].mean():.1%})"
    )

    log_step("STEP6b", "二次长度过滤")
    before_len = len(df)
    df = df[df["char_len_clean"] >= min_len].copy()
    print(f"[OK] 清洗后长度不足删除: {before_len - len(df)} 条")

    log_step("STEP7", "精确去重")
    before_dedup = len(df)
    df = df.drop_duplicates(
        subset=["hotel_id", "review_title", "review_text_clean"],
        keep="first",
    ).copy()
    print(f"[OK] 去重删除: {before_dedup - len(df)} 条，剩余 {len(df)} 条")

    log_step("STEP8", "日期与评分标准化")
    df["review_date"] = df["review_date_raw"].apply(safe_parse_timestamp)
    valid_dates = int(df["review_date"].notna().sum())
    if valid_dates / len(df) < 0.99:
        raise AssertionError("日期有效率低于 99%，请检查解析逻辑")
    print(f"[OK] 有效日期: {valid_dates}/{len(df)} ({valid_dates / len(df):.1%})")

    ref_date = pd.Timestamp(cfg["recency"]["reference_date"])
    df["review_year"] = df["review_date"].dt.year.astype("Int64")
    df["review_month"] = df["review_date"].dt.month.astype("Int64")
    df["recency_bucket"] = df["review_date"].apply(
        lambda value: assign_recency_bucket(value, ref_date, cfg["recency"]["buckets"])
    )
    df["sentiment_weak"] = df["rating"].apply(rating_to_weak_sentiment)
    df["full_text"] = (
        df["review_title"].fillna("").astype(str).str.strip() + " " + df["review_text_clean"]
    ).str.strip()
    print("   时间桶分布:")
    print(df["recency_bucket"].value_counts().to_string())

    log_step("STEP9", "酒店级过滤")
    hotel_stats = df.groupby("hotel_id").agg(
        n_reviews=("review_id", "count"),
        avg_len=("review_text_clean", lambda series: series.str.len().mean()),
    )
    valid_hotels = hotel_stats[
        (hotel_stats["n_reviews"] >= cfg["hotel_filter"]["min_reviews_per_hotel"])
        & (hotel_stats["avg_len"] >= cfg["hotel_filter"]["min_avg_text_length"])
    ].index
    before_hotel_filter = len(df)
    df = df[df["hotel_id"].isin(valid_hotels)].copy()
    print(
        f"[OK] 酒店过滤后: {df['hotel_id'].nunique()} 家酒店，{len(df)} 条评论 "
        f"(移除 {before_hotel_filter - len(df)} 条)"
    )
    assert_unique(df[["review_id"]], ["review_id"], "去重后 review_id")

    assert_shape(df, expected_rows=5850, tolerance=0.08, label="清洗后评论")
    out_path = Path("data/intermediate/cleaned_reviews.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_path)
    print(f"[SAVE] {out_path} ({len(df)} 行)")


if __name__ == "__main__":
    main()

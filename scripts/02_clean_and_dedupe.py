"""
02_clean_and_dedupe.py
步骤 5–7：日期解析 → 管理者回复去除 → 精确去重 → 评分标准化 → 酒店过滤
输入:  data/intermediate/city_filtered.pkl
输出: data/intermediate/cleaned_reviews.pkl
"""

import re
import pandas as pd
from pathlib import Path
from utils import (
    load_config, log_step, assert_shape,
    safe_parse_timestamp, remove_manager_response,
    assign_recency_bucket, rating_to_weak_sentiment
)


def main():
    cfg = load_config()
    df = pd.read_pickle("data/intermediate/city_filtered.pkl")
    log_step("STEP5", f"输入: {len(df)} 行")

    # ── Step 5a: 日期解析（逐行，修复 .000Z bug）──
    log_step("STEP5a", "日期解析")
    df["review_date"] = df["review_date_raw"].apply(safe_parse_timestamp)
    n_valid = df["review_date"].notna().sum()
    print(f"✅ 有效日期: {n_valid}/{len(df)} ({n_valid/len(df)*100:.1f}%)")
    assert n_valid / len(df) > 0.99, "日期有效率低于 99%，请检查解析逻辑"

    # 衍生时间字段
    ref_date = pd.Timestamp(cfg["recency"]["reference_date"])
    df["review_year"]  = df["review_date"].dt.year
    df["review_month"] = df["review_date"].dt.month
    df["recency_bucket"] = df["review_date"].apply(
        lambda d: assign_recency_bucket(d, ref_date, cfg["recency"]["buckets"])
    )
    print(f"   时间桶分布:")
    print(df["recency_bucket"].value_counts().to_string())

    # ── Step 5b: 评分标准化 ──
    log_step("STEP5b", "评分标准化")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").round().astype("Int64")
    df = df[df["rating"].between(1, 5)].copy()  # 删除无效评分
    df["sentiment_weak"] = df["rating"].apply(rating_to_weak_sentiment)
    print(f"✅ 评分分布: {df['rating'].value_counts().sort_index().to_dict()}")

    # ── Step 6: 管理者回复去除 ──
    log_step("STEP6", "去除管理者回复")
    results = df["review_text_raw"].apply(
        lambda t: remove_manager_response(t, cfg["cleaning"]["min_preserve_length"])
    )
    df["review_text_clean"] = [r[0] for r in results]
    df["has_manager_reply"] = [r[1] for r in results]
    n_mgr = df["has_manager_reply"].sum()
    print(f"✅ 检测到管理者回复: {n_mgr} / {len(df)} ({n_mgr/len(df)*100:.1f}%)")

    # ── Step 6b: 噪声清洗 ──
    log_step("STEP6b", "清洗噪声文本")
    noise_re = "|".join(
        re.escape(s) for s in cfg["cleaning"]["noise_exact_matches"]
    )
    df["review_text_clean"] = (
        df["review_text_clean"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # ── Step 6c: 长度过滤 ──
    min_len = cfg["cleaning"]["min_text_length"]
    before = len(df)
    df = df[df["review_text_clean"].str.len() >= min_len].copy()
    print(f"✅ 长度过滤 (<{min_len}字符): 删除 {before - len(df)} 条，剩余 {len(df)} 条")

    # ── Step 7: 精确去重 ──
    log_step("STEP7", "精确去重")
    df["dedup_key"] = (
        df["hotel_id"] + "|"
        + df["review_date_raw"].fillna("").astype(str) + "|"
        + df["review_text_raw"].fillna("").str[:200]
    )
    before = len(df)
    df = df.drop_duplicates(subset="dedup_key").drop(columns=["dedup_key"]).copy()
    print(f"✅ 去重: 删除 {before - len(df)} 条，剩余 {len(df)} 条")

    # ── Step 7b: 酒店级过滤 ──
    log_step("STEP7b", "酒店级过滤")
    hotel_review_count = df.groupby("hotel_id").size()
    hotel_avg_len = df.groupby("hotel_id")["review_text_clean"].apply(
        lambda x: x.str.len().mean()
    )
    min_rv = cfg["hotel_filter"]["min_reviews_per_hotel"]
    min_avg = cfg["hotel_filter"]["min_avg_text_length"]
    valid_hotels = hotel_review_count[
        (hotel_review_count >= min_rv) &
        (hotel_avg_len >= min_avg)
    ].index
    before = len(df)
    df = df[df["hotel_id"].isin(valid_hotels)].copy()
    print(f"✅ 酒店过滤: 保留 {df['hotel_id'].nunique()} 家酒店，{len(df)} 条评论")
    print(f"   (删除了 {before - len(df)} 条来自低质量酒店的评论)")

    # 字符长度字段
    df["char_len_clean"] = df["review_text_clean"].str.len()

    assert_shape(df, expected_rows=5700, tolerance=0.08, label="清洗后")
    out_path = Path("data/intermediate/cleaned_reviews.pkl")
    df.to_pickle(out_path)
    print(f"💾 保存: {out_path} ({len(df)} 行)")


if __name__ == "__main__":
    main()

"""
01_load_filter.py
步骤 1–4: 加载 CSV → 字段筛选 → 城市过滤 → 主键生成
输入:  raw_data/Datafiniti_Hotel_Reviews.csv
输出: data/intermediate/city_filtered.pkl
"""

import sys
import pandas as pd
from pathlib import Path

# 支持从 scripts/ 目录运行
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, make_hotel_id, make_review_id, log_step, assert_shape


COLUMN_MAP = {
    "id":                   "source_id",
    "keys":                 "hotel_key",
    "name":                 "hotel_name",
    "address":              "address",
    "city":                 "city",
    "province":             "state",
    "country":              "country",
    "postalCode":           "postal_code",
    "latitude":             "lat",
    "longitude":            "lng",
    "categories":           "categories",
    "primaryCategories":    "primary_category",
    "reviews.date":         "review_date_raw",
    "reviews.rating":       "rating",
    "reviews.title":        "review_title",
    "reviews.text":         "review_text_raw",
    "reviews.sourceURLs":   "review_source_url",
    "websites":             "hotel_website",
}


def main():
    cfg = load_config()

    # ── Step 1: 加载 ──
    log_step("STEP1", f"加载: {cfg['data']['raw_file']}")
    raw_df = pd.read_csv(cfg["data"]["raw_file"], low_memory=False)
    assert raw_df.shape == (10000, 26), f"数据完整性校验失败: {raw_df.shape}"
    print(f"✅ 加载完成: {raw_df.shape}")

    # ── Step 2: 字段筛选 ──
    log_step("STEP2", "字段筛选")
    core_df = raw_df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP).copy()
    print(f"✅ 保留 {len(core_df.columns)} 列")

    # ── Step 3: 城市过滤 ──
    log_step("STEP3", "城市过滤")
    cities = cfg["data"]["experiment_cities"]
    city_df = core_df[core_df["city"].isin(cities)].copy()
    assert_shape(city_df, expected_rows=6171, tolerance=0.03, label="城市过滤后")
    print("   城市分布:")
    for city, cnt in city_df["city"].value_counts().items():
        print(f"     {city}: {cnt}")

    # ── Step 4: 主键生成 ──
    log_step("STEP4", "生成主键")
    city_df["hotel_id"] = city_df["hotel_key"].apply(make_hotel_id)
    city_df["review_id"] = city_df.apply(
        lambda r: make_review_id(
            r["hotel_id"], str(r["review_date_raw"]),
            str(r["review_title"]), str(r["review_text_raw"])
        ), axis=1
    )
    n_hotels  = city_df["hotel_id"].nunique()
    n_reviews = city_df["review_id"].nunique()
    print(f"✅ 唯一酒店: {n_hotels} (预期 211)")
    print(f"✅ 唯一评论: {n_reviews}")
    assert n_hotels == 211, f"酒店数异常: {n_hotels}"

    # ── 保存 ──
    out = Path("data/intermediate/city_filtered.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    city_df.to_pickle(out)
    print(f"💾 保存: {out} ({len(city_df)} 行)")


if __name__ == "__main__":
    main()

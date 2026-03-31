"""
load_and_filter_reviews.py
步骤 1-4：加载 CSV -> 字段筛选 -> 城市过滤 -> 主键生成
输入:  raw_data/Datafiniti_Hotel_Reviews.csv
输出: data/intermediate/city_filtered.pkl
"""

from pathlib import Path

import pandas as pd

from scripts.shared.project_utils import (
    assert_shape,
    ensure_columns,
    load_config,
    log_step,
    make_hotel_id,
    make_review_id,
)


COLUMN_MAP = {
    "id": "source_id",
    "keys": "hotel_key",
    "name": "hotel_name",
    "address": "address",
    "city": "city",
    "province": "state",
    "country": "country",
    "postalCode": "postal_code",
    "latitude": "lat",
    "longitude": "lng",
    "categories": "categories",
    "primaryCategories": "primary_category",
    "reviews.date": "review_date_raw",
    "reviews.rating": "rating",
    "reviews.title": "review_title",
    "reviews.text": "review_text_raw",
    "reviews.sourceURLs": "review_source_url",
    "websites": "hotel_website",
}


def main():
    cfg = load_config()

    log_step("STEP1", f"加载: {cfg['data']['raw_file']}")
    raw_df = pd.read_csv(cfg["data"]["raw_file"], low_memory=False)
    assert raw_df.shape == (10000, 26), f"数据完整性校验失败: {raw_df.shape}"
    ensure_columns(raw_df, COLUMN_MAP.keys(), "原始 CSV")
    print(f"[OK] 加载完成: {raw_df.shape}")

    log_step("STEP2", "字段筛选与统一命名")
    city_df = raw_df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP).copy()
    ensure_columns(city_df, COLUMN_MAP.values(), "字段筛选后")
    print(f"[OK] 保留 {len(city_df.columns)} 列")

    log_step("STEP3", "城市过滤")
    city_df = city_df[city_df["city"].isin(cfg["data"]["experiment_cities"])].copy()
    assert_shape(city_df, expected_rows=6171, tolerance=0.03, label="城市过滤后")
    print("   城市分布:")
    for city, count in city_df["city"].value_counts().items():
        print(f"     {city}: {count}")

    log_step("STEP4", "生成主键")
    city_df["hotel_id"] = city_df["hotel_key"].apply(make_hotel_id)
    city_df["review_id"] = city_df.apply(
        lambda row: make_review_id(
            row["hotel_id"],
            str(row["review_date_raw"]),
            str(row["review_title"]),
            str(row["review_text_raw"]),
        ),
        axis=1,
    )
    n_hotels = city_df["hotel_id"].nunique()
    n_unique_reviews = city_df["review_id"].nunique()
    print(f"[OK] 唯一酒店: {n_hotels} (预期 211)")
    print(f"[OK] review_id 去重前唯一值: {n_unique_reviews} / {len(city_df)}")
    if n_hotels != 211:
        raise AssertionError(f"酒店数异常: {n_hotels}")

    out_path = Path("data/intermediate/city_filtered.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    city_df.to_pickle(out_path)
    print(f"[SAVE] {out_path} ({len(city_df)} 行)")


if __name__ == "__main__":
    main()

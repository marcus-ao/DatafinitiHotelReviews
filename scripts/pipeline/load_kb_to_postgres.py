"""
load_kb_to_postgres.py
步骤 13：将所有中间数据写入 PostgreSQL
输入:  data/intermediate/*.pkl
输出: PostgreSQL kb schema 中的 6 张核心表
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from psycopg2.extras import execute_values

from scripts.shared.project_utils import ensure_columns, get_pg_conn, load_db_config, log_step


def _rows_from_df(df: pd.DataFrame, columns: Iterable[str]) -> list[tuple]:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            tuple(None if pd.isna(row[column]) else row[column] for column in columns)
        )
    return rows


def build_hotel_df(review_df: pd.DataFrame, sentence_df: pd.DataFrame) -> pd.DataFrame:
    hotel_df = (
        review_df.groupby("hotel_id")
        .agg(
            hotel_key=("hotel_key", "first"),
            hotel_name=("hotel_name", "first"),
            address=("address", "first"),
            city=("city", "first"),
            state=("state", "first"),
            country=("country", "first"),
            postal_code=("postal_code", "first"),
            lat=("lat", "first"),
            lng=("lng", "first"),
            hotel_category=("primary_category", "first"),
            categories_raw=("categories", "first"),
            hotel_website=("hotel_website", "first"),
            n_reviews=("review_id", "count"),
            avg_rating=("rating", "mean"),
            rating_std=("rating", "std"),
        )
        .reset_index()
    )
    sentence_counts = sentence_df.groupby("hotel_id").size().rename("n_sentences").reset_index()
    hotel_df = hotel_df.merge(sentence_counts, on="hotel_id", how="left")
    hotel_df["avg_rating"] = hotel_df["avg_rating"].round(2)
    hotel_df["rating_std"] = hotel_df["rating_std"].round(2)
    hotel_df["n_sentences"] = hotel_df["n_sentences"].fillna(0).astype(int)
    return hotel_df


def truncate_tables(conn, schema: str) -> None:
    log_step("PG", "清空旧表数据")
    with conn.cursor() as cur:
        cur.execute(
            f"""
            TRUNCATE TABLE
                {schema}.evidence_index,
                {schema}.hotel_aspect_profile,
                {schema}.aspect_sentiment,
                {schema}.sentence,
                {schema}.review,
                {schema}.hotel
            CASCADE
            """
        )
    conn.commit()


def load_table(conn, schema: str, table: str, df: pd.DataFrame, columns: list[str], page_size: int = 1000) -> None:
    ensure_columns(df, columns, f"{table} load df")
    rows = _rows_from_df(df, columns)
    sql = f"INSERT INTO {schema}.{table} ({', '.join(columns)}) VALUES %s"
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=page_size)
    conn.commit()
    print(f"   [OK] {table}: {len(rows)} 行")


def main():
    db_cfg = load_db_config()
    schema = db_cfg["schema"]
    conn = get_pg_conn(db_cfg)

    review_df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
    sentence_df = pd.read_pickle("data/intermediate/sentences.pkl")
    aspect_df = pd.read_pickle("data/intermediate/aspect_sentiment.pkl")
    profile_df = pd.read_pickle("data/intermediate/hotel_profiles.pkl")
    evidence_df = pd.read_pickle("data/intermediate/evidence_index.pkl")

    ensure_columns(
        review_df,
        [
            "review_id",
            "hotel_id",
            "hotel_key",
            "hotel_name",
            "address",
            "city",
            "state",
            "country",
            "postal_code",
            "lat",
            "lng",
            "primary_category",
            "categories",
            "hotel_website",
            "review_date",
            "review_year",
            "review_month",
            "recency_bucket",
            "rating",
            "sentiment_weak",
            "review_title",
            "review_text_raw",
            "review_text_clean",
            "full_text",
            "char_len_raw",
            "char_len_clean",
            "has_manager_reply",
            "review_source_url",
        ],
        "cleaned_reviews.pkl",
    )
    ensure_columns(
        sentence_df,
        ["sentence_id", "review_id", "hotel_id", "sentence_text", "sentence_order", "char_len", "token_count", "city"],
        "sentences.pkl",
    )
    ensure_columns(
        aspect_df,
        ["sentence_id", "hotel_id", "aspect", "sentiment", "confidence", "label_source", "evidence_span"],
        "aspect_sentiment.pkl",
    )
    ensure_columns(
        profile_df,
        [
            "hotel_id",
            "aspect",
            "pos_count",
            "neg_count",
            "neu_count",
            "total_count",
            "recency_weighted_pos",
            "recency_weighted_neg",
            "controversy_score",
            "final_aspect_score",
        ],
        "hotel_profiles.pkl",
    )
    ensure_columns(
        evidence_df,
        [
            "text_id",
            "hotel_id",
            "sentence_id",
            "sentence_text",
            "aspect",
            "sentiment",
            "city",
            "hotel_name",
            "review_date",
            "rating",
            "recency_bucket",
            "embedding_id",
        ],
        "evidence_index.pkl",
    )

    hotel_df = build_hotel_df(review_df, sentence_df)

    review_out = review_df[
        [
            "review_id",
            "hotel_id",
            "review_date",
            "review_year",
            "review_month",
            "recency_bucket",
            "rating",
            "sentiment_weak",
            "review_title",
            "review_text_raw",
            "review_text_clean",
            "full_text",
            "char_len_raw",
            "char_len_clean",
            "has_manager_reply",
            "review_source_url",
            "city",
            "hotel_name",
        ]
    ].copy()

    sentence_out = sentence_df[
        [
            "sentence_id",
            "review_id",
            "hotel_id",
            "sentence_text",
            "sentence_order",
            "char_len",
            "token_count",
            "city",
        ]
    ].copy()

    aspect_out = aspect_df[
        [
            "sentence_id",
            "hotel_id",
            "aspect",
            "sentiment",
            "confidence",
            "label_source",
            "evidence_span",
        ]
    ].copy()

    truncate_tables(conn, schema)
    load_table(
        conn,
        schema,
        "hotel",
        hotel_df,
        [
            "hotel_id",
            "hotel_key",
            "hotel_name",
            "address",
            "city",
            "state",
            "country",
            "postal_code",
            "lat",
            "lng",
            "hotel_category",
            "categories_raw",
            "hotel_website",
            "n_reviews",
            "avg_rating",
            "rating_std",
        ],
    )
    load_table(
        conn,
        schema,
        "review",
        review_out,
        list(review_out.columns),
        page_size=500,
    )
    load_table(
        conn,
        schema,
        "sentence",
        sentence_out,
        list(sentence_out.columns),
        page_size=1000,
    )
    load_table(
        conn,
        schema,
        "aspect_sentiment",
        aspect_out,
        list(aspect_out.columns),
        page_size=1000,
    )
    load_table(
        conn,
        schema,
        "hotel_aspect_profile",
        profile_df,
        list(profile_df.columns),
        page_size=1000,
    )
    load_table(
        conn,
        schema,
        "evidence_index",
        evidence_df,
        list(evidence_df.columns),
        page_size=1000,
    )

    with conn.cursor() as cur:
        for table in [
            "hotel",
            "review",
            "sentence",
            "aspect_sentiment",
            "hotel_aspect_profile",
            "evidence_index",
        ]:
            cur.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
            count = cur.fetchone()[0]
            print(f"   {schema}.{table}: {count} 行")

    conn.close()
    print("\n[OK] PostgreSQL 数据加载完成")


if __name__ == "__main__":
    main()

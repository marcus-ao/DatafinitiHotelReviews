"""
08_load_to_postgres.py
步骤 13：将所有中间数据写入 PostgreSQL
输入:  data/intermediate/*.pkl
输出: PostgreSQL kb schema 中的 6 张表
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from utils import load_config, log_step, get_pg_conn


def upsert_hotels(conn, hotel_profiles: pd.DataFrame):
    log_step("PG", "写入 hotel 表")
    cols = [
        "hotel_id", "hotel_name", "city", "state", "address",
        "latitude", "longitude", "n_reviews", "avg_rating",
        "min_review_date", "max_review_date", "n_sentences",
        "overall_pos", "overall_neg", "overall_neu"
    ]
    rows = [
        tuple(row[c] if pd.notna(row.get(c)) else None for c in cols)
        for _, row in hotel_profiles.iterrows()
    ]
    sql = f"""
        INSERT INTO kb.hotel ({', '.join(cols)})
        VALUES %s
        ON CONFLICT (hotel_id) DO UPDATE SET
            n_reviews      = EXCLUDED.n_reviews,
            avg_rating     = EXCLUDED.avg_rating,
            n_sentences    = EXCLUDED.n_sentences,
            overall_pos    = EXCLUDED.overall_pos,
            overall_neg    = EXCLUDED.overall_neg,
            overall_neu    = EXCLUDED.overall_neu,
            updated_at     = NOW()
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()
    print(f"   ✅ hotel: {len(rows)} 行")


def upsert_reviews(conn, reviews_df: pd.DataFrame):
    log_step("PG", "写入 review 表")
    cols = [
        "review_id", "hotel_id", "review_date", "review_year",
        "review_month", "recency_bucket", "rating", "sentiment_weak",
        "has_manager_reply", "char_len_clean", "review_text_clean",
        "username", "user_city"
    ]
    rows = []
    for _, row in reviews_df.iterrows():
        rows.append(tuple(
            row.get(c) if pd.notna(row.get(c) if row.get(c) is not None else float('nan')) else None
            for c in cols
        ))
    sql = f"""
        INSERT INTO kb.review ({', '.join(cols)})
        VALUES %s
        ON CONFLICT (review_id) DO NOTHING
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=500)
    conn.commit()
    print(f"   ✅ review: {len(rows)} 行")


def upsert_sentences(conn, sent_df: pd.DataFrame):
    log_step("PG", "写入 sentence 表")
    cols = [
        "sentence_id", "review_id", "hotel_id",
        "sentence_order", "sentence_text", "char_len", "token_count"
    ]
    rows = [
        tuple(row.get(c) for c in cols)
        for _, row in sent_df.iterrows()
    ]
    sql = f"""
        INSERT INTO kb.sentence ({', '.join(cols)})
        VALUES %s
        ON CONFLICT (sentence_id) DO NOTHING
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=1000)
    conn.commit()
    print(f"   ✅ sentence: {len(rows)} 行")


def upsert_aspect_sentiment(conn, classified_df: pd.DataFrame):
    log_step("PG", "写入 aspect_sentiment 表")
    cols = [
        "sentence_id", "review_id", "hotel_id",
        "aspect", "sentiment", "label_source", "confidence"
    ]
    rows = []
    for _, row in classified_df.iterrows():
        # 主方面写一条
        rows.append((
            row["sentence_id"], row["review_id"], row["hotel_id"],
            row["primary_aspect"], row["sentiment"],
            row["label_source"], 0.85 if row["label_source"] == "rule" else 0.65
        ))
    sql = f"""
        INSERT INTO kb.aspect_sentiment ({', '.join(cols)})
        VALUES %s
        ON CONFLICT (sentence_id, aspect) DO UPDATE SET
            sentiment    = EXCLUDED.sentiment,
            label_source = EXCLUDED.label_source,
            confidence   = EXCLUDED.confidence
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=1000)
    conn.commit()
    print(f"   ✅ aspect_sentiment: {len(rows)} 行")


def upsert_hotel_aspect_profile(conn, profiles_df: pd.DataFrame):
    log_step("PG", "写入 hotel_aspect_profile 表")
    ASPECTS = [
        "location_transport", "cleanliness", "service",
        "room_facilities", "quiet_sleep", "value", "general"
    ]
    rows = []
    for _, row in profiles_df.iterrows():
        for asp in ASPECTS:
            n = row.get(f"{asp}_n", 0)
            if not n:
                continue
            rows.append((
                row["hotel_id"], asp,
                row.get(f"{asp}_pos"), row.get(f"{asp}_neg"),
                row.get(f"{asp}_neu"), int(n)
            ))
    sql = """
        INSERT INTO kb.hotel_aspect_profile
            (hotel_id, aspect, pos_ratio, neg_ratio, neu_ratio, n_sentences)
        VALUES %s
        ON CONFLICT (hotel_id, aspect) DO UPDATE SET
            pos_ratio   = EXCLUDED.pos_ratio,
            neg_ratio   = EXCLUDED.neg_ratio,
            neu_ratio   = EXCLUDED.neu_ratio,
            n_sentences = EXCLUDED.n_sentences,
            updated_at  = NOW()
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()
    print(f"   ✅ hotel_aspect_profile: {len(rows)} 行")


def main():
    cfg = load_config()
    conn = get_pg_conn(cfg)

    reviews_df  = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
    classified  = pd.read_pickle("data/intermediate/sentiment_classified.pkl")
    profiles    = pd.read_pickle("data/intermediate/hotel_profiles.pkl")

    # sentence 表只需基础字段
    sent_cols = ["sentence_id", "review_id", "hotel_id",
                 "sentence_order", "sentence_text", "char_len", "token_count"]
    sent_df = classified[sent_cols].drop_duplicates(subset=["sentence_id"])

    upsert_hotels(conn, profiles)
    upsert_reviews(conn, reviews_df)
    upsert_sentences(conn, sent_df)
    upsert_aspect_sentiment(conn, classified)
    upsert_hotel_aspect_profile(conn, profiles)

    # 汇总
    with conn.cursor() as cur:
        for table in ["hotel", "review", "sentence", "aspect_sentiment", "hotel_aspect_profile"]:
            cur.execute(f"SELECT COUNT(*) FROM kb.{table}")
            cnt = cur.fetchone()[0]
            print(f"   kb.{table}: {cnt} 行")

    conn.close()
    print("\n✅ PostgreSQL 数据加载完成")


if __name__ == "__main__":
    main()

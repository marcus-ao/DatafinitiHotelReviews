"""
09_validate.py
步骤 14：端到端验证 — 检查数据契约、ChromaDB 和 PostgreSQL
"""

from __future__ import annotations

import os

import pandas as pd

from utils import ensure_columns, get_pg_conn, load_config, load_db_config, log_step


def check(condition: bool, name: str, detail: str = "") -> bool:
    icon = "[OK]" if condition else "[FAIL]"
    print(f"  {icon} {name}" + (f": {detail}" if detail else ""))
    return condition


def main():
    cfg = load_config()
    db_cfg = load_db_config()
    passed: list[bool] = []

    required_files = [
        "data/intermediate/cleaned_reviews.pkl",
        "data/intermediate/sentences.pkl",
        "data/intermediate/aspect_labels.pkl",
        "data/intermediate/aspect_sentiment.pkl",
        "data/intermediate/hotel_profiles.pkl",
        "data/intermediate/evidence_index.pkl",
    ]

    log_step("VALIDATE", "检查中间文件")
    for path in required_files:
        passed.append(check(os.path.exists(path), f"文件存在: {path}"))

    log_step("VALIDATE", "加载中间产物")
    reviews = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
    sentences = pd.read_pickle("data/intermediate/sentences.pkl")
    aspect_labels = pd.read_pickle("data/intermediate/aspect_labels.pkl")
    aspect_sentiment = pd.read_pickle("data/intermediate/aspect_sentiment.pkl")
    profiles = pd.read_pickle("data/intermediate/hotel_profiles.pkl")
    evidence = pd.read_pickle("data/intermediate/evidence_index.pkl")

    ensure_columns(reviews, ["review_id", "hotel_id", "city", "state", "char_len_clean", "rating"], "cleaned_reviews.pkl")
    ensure_columns(sentences, ["sentence_id", "review_id", "hotel_id", "sentence_text", "city"], "sentences.pkl")
    ensure_columns(aspect_labels, ["sentence_id", "aspect", "label_source"], "aspect_labels.pkl")
    ensure_columns(aspect_sentiment, ["sentence_id", "aspect", "sentiment"], "aspect_sentiment.pkl")
    ensure_columns(profiles, ["hotel_id", "aspect", "final_aspect_score"], "hotel_profiles.pkl")
    ensure_columns(evidence, ["sentence_id", "city", "aspect", "sentiment", "embedding_id"], "evidence_index.pkl")

    log_step("VALIDATE", "检查数据量与契约")
    n_hotels = reviews["hotel_id"].nunique()
    passed.append(check(5400 <= len(reviews) <= 6200, "评论数落在合理区间", str(len(reviews))))
    passed.append(check(n_hotels >= 140, "酒店数 >= 140", str(n_hotels)))
    passed.append(check(len(sentences) >= 30000, "句子数 >= 30000", str(len(sentences))))
    passed.append(check(aspect_labels["sentence_id"].nunique() == len(sentences), "aspect_labels 覆盖全部句子"))
    passed.append(check(aspect_sentiment["sentence_id"].nunique() == len(sentences), "aspect_sentiment 覆盖全部句子"))
    passed.append(check(len(evidence) == len(sentences), "evidence_index 行数 = sentence 行数", str(len(evidence))))
    passed.append(check(len(profiles) == n_hotels * 6, "hotel_aspect_profile = 酒店数 × 6", str(len(profiles))))
    passed.append(check(reviews["city"].nunique() == 10, "覆盖城市 = 10", str(reviews["city"].nunique())))
    passed.append(check(reviews["state"].nunique() == 8, "覆盖州 = 8", str(reviews["state"].nunique())))
    passed.append(check(reviews["rating"].notna().all(), "所有评论有评分"))
    passed.append(check((reviews["char_len_clean"] >= 50).all(), "所有评论 >= 50 字符"))

    log_step("VALIDATE", "检查标签与情感分布")
    aspect_dist = aspect_sentiment["aspect"].value_counts(normalize=True)
    sent_dist = aspect_sentiment["sentiment"].value_counts(normalize=True)
    general_ratio = aspect_dist.get("general", 0.0)
    passed.append(check(general_ratio <= 0.35, "general 比例 <= 35%", f"{general_ratio:.1%}"))
    passed.append(check(sent_dist.get("positive", 0) >= 0.40, "positive >= 40%", f"{sent_dist.get('positive', 0):.1%}"))
    passed.append(check(sent_dist.get("negative", 0) >= 0.05, "negative >= 5%", f"{sent_dist.get('negative', 0):.1%}"))

    log_step("VALIDATE", "检查 ChromaDB")
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        ecfg = cfg["embedding"]
        client = chromadb.PersistentClient(path=ecfg["chroma_persist_dir"])
        collection = client.get_collection(ecfg["chroma_collection"])
        passed.append(check(collection.count() == len(evidence), "ChromaDB 记录数 = evidence_index 行数", str(collection.count())))

        model = SentenceTransformer(ecfg["model"])
        query_embedding = model.encode(
            ["quiet hotel near downtown with good service"],
            normalize_embeddings=bool(ecfg.get("normalize", True)),
        ).tolist()
        result = collection.query(
            query_embeddings=query_embedding,
            n_results=1,
            where={"city": "San Diego", "aspect": "location_transport", "sentiment": "positive"},
        )
        passed.append(check(len(result["ids"][0]) == 1, "ChromaDB city/aspect/sentiment 过滤查询正常"))
    except Exception as exc:
        passed.append(check(False, "ChromaDB 检查", str(exc)))

    log_step("VALIDATE", "检查 PostgreSQL")
    try:
        conn = get_pg_conn(db_cfg)
        schema = db_cfg["schema"]
        expected_counts = {
            "hotel": n_hotels,
            "review": len(reviews),
            "sentence": len(sentences),
            "aspect_sentiment": len(aspect_sentiment),
            "hotel_aspect_profile": len(profiles),
            "evidence_index": len(evidence),
        }
        with conn.cursor() as cur:
            for table, expected in expected_counts.items():
                cur.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
                actual = cur.fetchone()[0]
                passed.append(check(actual == expected, f"{schema}.{table} 行数一致", f"{actual}"))
        conn.close()
    except Exception as exc:
        passed.append(check(False, "PostgreSQL 检查", str(exc)))

    total = len(passed)
    success = sum(passed)
    failed = total - success
    print(f"\n{'=' * 50}")
    print(f"验证结果: {success}/{total} 通过" + (" [OK] 全部通过！" if failed == 0 else f" [FAIL] {failed} 项失败"))
    print("=" * 50)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

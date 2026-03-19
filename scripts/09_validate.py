"""
09_validate.py
步骤 14：端到端验证 — 检查所有数据质量门控指标
"""

import pandas as pd
from utils import load_config, log_step, get_pg_conn


def check(condition: bool, name: str, detail: str = ""):
    icon = "✅" if condition else "❌"
    print(f"  {icon} {name}" + (f": {detail}" if detail else ""))
    return condition


def main():
    cfg = load_config()
    passed = []

    # ── 1. 中间文件检查 ──
    log_step("VALIDATE", "检查中间文件")
    import os
    for fname in [
        "data/intermediate/cleaned_reviews.pkl",
        "data/intermediate/sentences.pkl",
        "data/intermediate/aspect_classified.pkl",
        "data/intermediate/sentiment_classified.pkl",
        "data/intermediate/hotel_profiles.pkl",
    ]:
        passed.append(check(os.path.exists(fname), f"文件存在: {fname}"))

    # ── 2. 数据量检查 ──
    log_step("VALIDATE", "检查数据量")
    reviews = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
    sentences = pd.read_pickle("data/intermediate/sentiment_classified.pkl")
    profiles = pd.read_pickle("data/intermediate/hotel_profiles.pkl")

    passed.append(check(len(reviews) >= 5000,   "评论数 >= 5000",   str(len(reviews))))
    passed.append(check(len(profiles) >= 100,   "酒店数 >= 100",    str(len(profiles))))
    passed.append(check(len(sentences) >= 30000, "句子数 >= 30000", str(len(sentences))))
    passed.append(check(reviews["rating"].notna().all(), "所有评论有评分"))
    passed.append(check(
        (reviews["char_len_clean"] >= 50).all(),
        "所有评论 >= 50 字符"
    ))

    # ── 3. 方面标签检查 ──
    log_step("VALIDATE", "检查方面分布")
    aspect_dist = sentences["primary_aspect"].value_counts(normalize=True)
    general_ratio = aspect_dist.get("general", 0)
    passed.append(check(general_ratio <= 0.25, f"general 比例 <= 25%", f"{general_ratio:.1%}"))
    label_src = sentences["label_source"].value_counts(normalize=True)
    print(f"     label_source 分布: {dict(label_src.round(3))}")

    # ── 4. 情感分布检查 ──
    log_step("VALIDATE", "检查情感分布")
    sent_dist = sentences["sentiment"].value_counts(normalize=True)
    passed.append(check(sent_dist.get("positive", 0) >= 0.40, "positive >= 40%",
                        f"{sent_dist.get('positive',0):.1%}"))
    passed.append(check(sent_dist.get("negative", 0) >= 0.05, "negative >= 5%",
                        f"{sent_dist.get('negative',0):.1%}"))

    # ── 5. ChromaDB 检查 ──
    log_step("VALIDATE", "检查 ChromaDB")
    try:
        import chromadb
        ecfg = cfg["embedding"]
        client = chromadb.PersistentClient(path=ecfg["chroma_persist_dir"])
        col = client.get_collection(ecfg["chroma_collection"])
        count = col.count()
        passed.append(check(count >= 30000, f"ChromaDB 记录数 >= 30000", str(count)))
        # 抽样查询
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(ecfg["model"])
        q_emb = model.encode(["great location near subway"], normalize_embeddings=True).tolist()
        result = col.query(query_embeddings=q_emb, n_results=1,
                           where={"aspect": "location_transport"})
        passed.append(check(len(result["ids"][0]) == 1, "ChromaDB 过滤查询正常"))
    except Exception as e:
        passed.append(check(False, "ChromaDB 检查", str(e)))

    # ── 6. PostgreSQL 检查 ──
    log_step("VALIDATE", "检查 PostgreSQL")
    try:
        conn = get_pg_conn(cfg)
        with conn.cursor() as cur:
            for table, min_rows in [
                ("hotel",               100),
                ("review",              5000),
                ("sentence",            30000),
                ("aspect_sentiment",    30000),
                ("hotel_aspect_profile",500),
            ]:
                cur.execute(f"SELECT COUNT(*) FROM kb.{table}")
                cnt = cur.fetchone()[0]
                passed.append(check(cnt >= min_rows,
                                    f"kb.{table} >= {min_rows}", str(cnt)))
        conn.close()
    except Exception as e:
        passed.append(check(False, "PostgreSQL 连接", str(e)))

    # ── 汇总 ──
    total  = len(passed)
    ok     = sum(passed)
    failed = total - ok
    print(f"\n{'='*50}")
    print(f"验证结果: {ok}/{total} 通过" + (" ✅ 全部通过！" if failed == 0 else f" ❌ {failed} 项失败"))
    print('='*50)
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

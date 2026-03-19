"""
07_build_vector_index.py
步骤 12b-12c：构建 evidence_index 元数据并写入 ChromaDB
输入:  data/intermediate/sentences.pkl
        data/intermediate/aspect_sentiment.pkl
        data/intermediate/cleaned_reviews.pkl
输出: data/intermediate/evidence_index.pkl
        data/chroma_db/
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils import ensure_columns, iso_date, load_config, log_step


def build_primary_meta(aspect_df: pd.DataFrame) -> pd.DataFrame:
    # 这里不用 groupby.apply，是为了避免不同 pandas 版本下
    # sentence_id 被放进索引或直接丢失，导致后续 merge 时报 KeyError。
    # 通过排序后按 sentence_id 去重，可以稳定地为每个句子选出一个主标签。
    ranked = aspect_df.copy()
    ranked["general_rank"] = (ranked["aspect"] == "general").astype(int)
    ranked = ranked.sort_values(
        by=["sentence_id", "general_rank", "confidence", "sentiment_confidence"],
        ascending=[True, True, False, False],
    )
    primary_meta = ranked.drop_duplicates(subset=["sentence_id"], keep="first").copy()
    return primary_meta[
        [
            "sentence_id",
            "aspect",
            "sentiment",
            "confidence",
            "label_source",
        ]
    ]


def main():
    cfg = load_config()
    ecfg = cfg["embedding"]

    sent_df = pd.read_pickle("data/intermediate/sentences.pkl")
    aspect_df = pd.read_pickle("data/intermediate/aspect_sentiment.pkl")
    review_df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
    ensure_columns(
        sent_df,
        ["sentence_id", "review_id", "hotel_id", "sentence_text", "city"],
        "sentences.pkl",
    )
    ensure_columns(
        aspect_df,
        ["sentence_id", "aspect", "sentiment", "confidence", "label_source"],
        "aspect_sentiment.pkl",
    )
    ensure_columns(
        review_df,
        ["review_id", "hotel_name", "review_date", "rating", "recency_bucket"],
        "cleaned_reviews.pkl",
    )
    log_step("STEP12b", f"句子: {len(sent_df)}，标签: {len(aspect_df)}")

    primary_meta = build_primary_meta(aspect_df)
    if primary_meta["sentence_id"].duplicated().any():
        raise AssertionError("primary_meta 中存在重复 sentence_id")

    # 先拼接句子级标签，再关联评论级补充字段，统一生成 evidence_index。
    evidence_df = (
        sent_df.merge(
            primary_meta,
            on="sentence_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            review_df[
                [
                    "review_id",
                    "hotel_name",
                    "review_date",
                    "rating",
                    "recency_bucket",
                    "city",
                ]
            ].drop_duplicates("review_id"),
            on="review_id",
            how="left",
            suffixes=("", "_review"),
            validate="many_to_one",
        )
    )

    evidence_df["aspect"] = evidence_df["aspect"].fillna("general")
    evidence_df["sentiment"] = evidence_df["sentiment"].fillna("neutral")
    evidence_df["text_id"] = evidence_df["sentence_id"]
    evidence_df["city"] = evidence_df["city"].fillna(evidence_df["city_review"])
    evidence_df["embedding_id"] = range(len(evidence_df))
    evidence_df = evidence_df[
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
        ]
    ].copy()

    out_meta = Path("data/intermediate/evidence_index.pkl")
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    evidence_df.to_pickle(out_meta)
    print(f"[SAVE] {out_meta} ({len(evidence_df)} 行)")

    log_step("STEP12c", "编码句向量并写入 ChromaDB")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(ecfg["model"])
    texts = evidence_df["sentence_text"].tolist()
    embeddings = model.encode(
        texts,
        normalize_embeddings=bool(ecfg.get("normalize", True)),
        show_progress_bar=True,
        batch_size=ecfg["encode_batch_size"],
    )

    import chromadb

    persist_dir = Path(ecfg["chroma_persist_dir"])
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    try:
        client.delete_collection(ecfg["chroma_collection"])
    except Exception:
        pass

    collection = client.create_collection(
        name=ecfg["chroma_collection"],
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 5000
    ids = evidence_df["sentence_id"].tolist()
    for start in tqdm(range(0, len(evidence_df), batch_size), desc="写入 ChromaDB"):
        batch = evidence_df.iloc[start : start + batch_size]
        collection.add(
            ids=ids[start : start + batch_size],
            embeddings=embeddings[start : start + batch_size].tolist(),
            documents=batch["sentence_text"].tolist(),
            metadatas=[
                {
                    "hotel_id": str(row["hotel_id"]),
                    "city": str(row["city"]),
                    "hotel_name": str(row["hotel_name"]),
                    "aspect": str(row["aspect"]),
                    "sentiment": str(row["sentiment"]),
                    "rating": int(row["rating"]) if pd.notna(row["rating"]) else 0,
                    "review_date": iso_date(row["review_date"]) or "",
                    "recency_bucket": str(row["recency_bucket"]) if pd.notna(row["recency_bucket"]) else "unknown",
                }
                for _, row in batch.iterrows()
            ],
        )

    print(f"\n[OK] ChromaDB 写入完成: {collection.count()} 条记录")

    test_query = "quiet hotel near downtown with good service"
    test_result = collection.query(
        query_embeddings=model.encode(
            [test_query],
            normalize_embeddings=bool(ecfg.get("normalize", True)),
        ).tolist(),
        n_results=3,
        where={"aspect": "location_transport"},
    )
    print(f"\n[QUERY] 检索测试: {test_query}")
    for doc, meta in zip(test_result["documents"][0], test_result["metadatas"][0]):
        print(f"   [{meta['aspect']}|{meta['sentiment']}] {meta['hotel_name']}: {doc[:80]}...")


if __name__ == "__main__":
    main()

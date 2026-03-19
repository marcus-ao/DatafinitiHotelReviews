"""
07_build_vector_index.py
步骤 12：句子向量化 → 写入 ChromaDB
输入:  data/intermediate/sentiment_classified.pkl
输出: data/chroma_db/ (ChromaDB 持久化目录)
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from utils import load_config, log_step


def main():
    cfg = load_config()
    ecfg = cfg["embedding"]

    df = pd.read_pickle("data/intermediate/sentiment_classified.pkl")
    log_step("STEP12", f"输入: {len(df)} 句子")

    # ── 编码 ──
    from sentence_transformers import SentenceTransformer
    print(f"   加载 Embedding 模型: {ecfg['model']}")
    model = SentenceTransformer(ecfg["model"])

    texts = df["sentence_text"].tolist()
    print(f"   编码 {len(texts)} 句子 (batch_size={ecfg['encode_batch_size']})...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=ecfg["encode_batch_size"]
    )
    print(f"✅ 编码完成: shape={embeddings.shape}")

    # ── ChromaDB ──
    import chromadb
    persist_dir = ecfg["chroma_persist_dir"]
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    # 删除旧集合
    try:
        client.delete_collection(ecfg["chroma_collection"])
        print(f"   已删除旧集合: {ecfg['chroma_collection']}")
    except Exception:
        pass

    collection = client.create_collection(
        name=ecfg["chroma_collection"],
        metadata={"hnsw:space": "cosine"}
    )

    # 构建 metadata
    metadatas = []
    for _, row in df.iterrows():
        metadatas.append({
            "review_id":      str(row["review_id"]),
            "hotel_id":       str(row["hotel_id"]),
            "city":           str(row.get("city", "")),
            "aspect":         str(row["primary_aspect"]),
            "sentiment":      str(row["sentiment"]),
            "label_source":   str(row["label_source"]),
            "sentence_order": int(row["sentence_order"]),
            "char_len":       int(row["char_len"]),
        })

    # 分批写入
    BATCH = 5000
    ids = df["sentence_id"].tolist()
    for i in tqdm(range(0, len(df), BATCH), desc="写入 ChromaDB"):
        collection.add(
            ids=ids[i : i + BATCH],
            embeddings=embeddings[i : i + BATCH].tolist(),
            documents=texts[i : i + BATCH],
            metadatas=metadatas[i : i + BATCH]
        )

    count = collection.count()
    print(f"\n✅ ChromaDB 写入完成: {count} 条记录")
    print(f"   持久化目录: {persist_dir}")

    # 快速检索测试
    test_query = "The location was very convenient near downtown"
    test_result = collection.query(
        query_embeddings=model.encode([test_query], normalize_embeddings=True).tolist(),
        n_results=3,
        where={"aspect": "location_transport"}
    )
    print(f"\n🔍 检索测试 ('{test_query[:40]}...')")
    for doc, meta in zip(test_result["documents"][0], test_result["metadatas"][0]):
        print(f"   [{meta['aspect']} / {meta['sentiment']}] {doc[:80]}...")


if __name__ == "__main__":
    main()

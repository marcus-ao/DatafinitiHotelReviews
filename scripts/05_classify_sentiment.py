"""
05_classify_sentiment.py
步骤 10：句子 → 情感分类（nlptown/bert-base-multilingual-uncased-sentiment）
输入:  data/intermediate/aspect_classified.pkl
输出: data/intermediate/sentiment_classified.pkl
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from utils import load_config, log_step, assert_shape


STAR_TO_SENTIMENT = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}


def stars_from_label(label: str) -> int:
    """nlptown 输出格式: '4 stars' → 4"""
    return int(label.split()[0])


def batch_sentiment(texts: list[str], classifier, batch_size: int) -> list[str]:
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="情感分类"):
        batch = texts[i : i + batch_size]
        outputs = classifier(batch, truncation=True, max_length=128)
        if isinstance(outputs, dict):
            outputs = [outputs]
        for out in outputs:
            stars = stars_from_label(out["label"])
            results.append(STAR_TO_SENTIMENT[stars])
    return results


def main():
    cfg = load_config()
    scfg = cfg["sentiment"]

    df = pd.read_pickle("data/intermediate/aspect_classified.pkl")
    log_step("STEP10", f"输入: {len(df)} 句子")

    from transformers import pipeline
    print(f"   加载模型: {scfg['model']}")
    try:
        sentiment_pipe = pipeline(
            "text-classification",
            model=scfg["model"],
            device=0,
            batch_size=scfg["batch_size"]
        )
    except Exception:
        print("   GPU 不可用，使用 CPU")
        sentiment_pipe = pipeline(
            "text-classification",
            model=scfg["model"],
            device=-1,
            batch_size=32
        )

    texts = df["sentence_text"].tolist()
    sentiments = batch_sentiment(texts, sentiment_pipe, scfg["batch_size"])
    df["sentiment"] = sentiments

    print("\n✅ 情感分布:")
    print(df["sentiment"].value_counts().to_string())

    # 一致性检验：sentence sentiment vs review weak sentiment
    match_mask = (
        (df["sentiment"] == "positive") & (df["sentiment_weak"] == "positive") |
        (df["sentiment"] == "negative") & (df["sentiment_weak"] == "negative") |
        (df["sentiment"] == "neutral")  & (df["sentiment_weak"] == "neutral")
    )
    consistency = match_mask.sum() / len(df)
    print(f"\n   sentence vs review 一致率: {consistency:.1%}")
    if consistency < 0.55:
        print("   ⚠️  一致率偏低（<55%），请检查模型是否正常加载")

    assert_shape(df, expected_rows=46000, tolerance=0.20, label="情感分类后")
    out_path = Path("data/intermediate/sentiment_classified.pkl")
    df.to_pickle(out_path)
    print(f"\n💾 保存: {out_path} ({len(df)} 行)")


if __name__ == "__main__":
    main()

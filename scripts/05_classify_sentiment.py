"""
05_classify_sentiment.py
步骤 11c：句子 -> 情感分类
输入:  data/intermediate/aspect_labels.pkl
输出: data/intermediate/aspect_sentiment.pkl
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils import ensure_columns, load_config, log_step

STAR_TO_SENTIMENT = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}


def get_transformers_device() -> int:
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def stars_from_label(label: str) -> int:
    return int(label.split()[0])


def batch_sentiment(texts: list[str], classifier, batch_size: int) -> list[tuple[str, float]]:
    results: list[tuple[str, float]] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="情感分类"):
        batch = texts[start : start + batch_size]
        outputs = classifier(batch, truncation=True, max_length=128)
        if isinstance(outputs, dict):
            outputs = [outputs]
        for output in outputs:
            stars = stars_from_label(output["label"])
            results.append((STAR_TO_SENTIMENT[stars], round(float(output["score"]), 3)))
    return results


def main():
    cfg = load_config()
    scfg = cfg["sentiment"]

    label_df = pd.read_pickle("data/intermediate/aspect_labels.pkl")
    sent_df = pd.read_pickle("data/intermediate/sentences.pkl")
    review_df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
    ensure_columns(
        label_df,
        ["sentence_id", "review_id", "hotel_id", "aspect", "confidence", "label_source"],
        "aspect_labels.pkl",
    )
    ensure_columns(sent_df, ["sentence_id", "sentence_text"], "sentences.pkl")
    ensure_columns(review_df, ["review_id", "sentiment_weak"], "cleaned_reviews.pkl")
    log_step("STEP11c", f"输入: {len(label_df)} 条方面标签")

    unique_sentences = sent_df[["sentence_id", "sentence_text"]].drop_duplicates("sentence_id")

    from transformers import pipeline

    sentiment_pipe = pipeline(
        "text-classification",
        model=scfg["model"],
        device=get_transformers_device(),
        batch_size=scfg["batch_size"],
    )
    sentiment_pairs = batch_sentiment(
        unique_sentences["sentence_text"].tolist(),
        sentiment_pipe,
        scfg["batch_size"],
    )
    sentence_sentiment_df = unique_sentences[["sentence_id"]].copy()
    sentence_sentiment_df["sentiment"] = [item[0] for item in sentiment_pairs]
    sentence_sentiment_df["sentiment_confidence"] = [item[1] for item in sentiment_pairs]

    aspect_sentiment_df = (
        label_df.merge(sentence_sentiment_df, on="sentence_id", how="left", validate="many_to_one")
        .merge(review_df[["review_id", "sentiment_weak"]], on="review_id", how="left", validate="many_to_one")
    )

    print("\n[OK] 情感分布:")
    print(aspect_sentiment_df["sentiment"].value_counts().to_string())

    match_mask = aspect_sentiment_df["sentiment"] == aspect_sentiment_df["sentiment_weak"]
    consistency = match_mask.mean()
    print(f"\n   sentence vs review 一致率: {consistency:.1%}")
    if consistency < 0.55:
        print("   [WARN] 一致率偏低（<55%），请检查模型是否正常加载")

    out_path = Path("data/intermediate/aspect_sentiment.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aspect_sentiment_df.to_pickle(out_path)
    print(f"\n[SAVE] {out_path} ({len(aspect_sentiment_df)} 行)")


if __name__ == "__main__":
    main()

"""
split_reviews_into_sentences.py
步骤 10：评论正文 -> 句子级拆分
输入:  data/intermediate/cleaned_reviews.pkl
输出: data/intermediate/sentences.pkl
"""

import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scripts.shared.project_utils import assert_shape, ensure_columns, load_config, log_step

_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
        except ImportError as exc:
            raise ImportError(
                "请先安装 spacy 并下载模型: python -m spacy download en_core_web_sm"
            ) from exc

        try:
            _NLP = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
            print("[OK] 使用 spaCy 模型: en_core_web_sm")
        except OSError:
            print(
                "[WARN] 未检测到 en_core_web_sm，回退到 spacy.blank('en') + sentencizer。"
            )
            print(
                "[WARN] 如需更稳定的英文分句效果，请执行: python -m spacy download en_core_web_sm"
            )
            _NLP = spacy.blank("en")
            if "sentencizer" not in _NLP.pipe_names:
                _NLP.add_pipe("sentencizer")
        _NLP.max_length = 2_000_000
    return _NLP


def merge_fragments(raw_sents: list[str], merge_len: int) -> list[str]:
    """先合并碎片句，再交给长度过滤处理。"""
    merged: list[str] = []
    leading_buffer = ""

    for sentence in raw_sents:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) < merge_len:
            if merged:
                merged[-1] = merged[-1].rstrip() + " " + sentence.lstrip()
            else:
                leading_buffer = (
                    f"{leading_buffer} {sentence}".strip() if leading_buffer else sentence
                )
            continue

        if leading_buffer:
            sentence = f"{leading_buffer} {sentence}".strip()
            leading_buffer = ""
        merged.append(sentence)

    if leading_buffer:
        if merged:
            merged[-1] = f"{merged[-1]} {leading_buffer}".strip()
        else:
            merged.append(leading_buffer)

    return merged


def split_review(text: str, min_len: int, merge_len: int, nlp) -> list[str]:
    if not text or not text.strip():
        return []

    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
    raw_sents = [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]
    merged = merge_fragments(raw_sents, merge_len)
    return [sentence for sentence in merged if len(sentence) >= min_len]


def main():
    cfg = load_config()
    scfg = cfg["sentence"]
    min_len = scfg["min_sentence_length"]
    merge_len = scfg["min_fragment_merge_length"]

    df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
    ensure_columns(
        df,
        ["review_id", "hotel_id", "review_text_clean", "city"],
        "cleaned_reviews.pkl",
    )
    log_step("STEP10", f"输入: {len(df)} 条评论 -> 句子拆分")

    nlp = get_nlp()
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="分句"):
        sentences = split_review(row["review_text_clean"], min_len, merge_len, nlp)
        for order, sentence in enumerate(sentences):
            rows.append(
                {
                    "sentence_id": f"{row['review_id']}_s{order:03d}",
                    "review_id": row["review_id"],
                    "hotel_id": row["hotel_id"],
                    "sentence_order": order,
                    "sentence_text": sentence,
                    "char_len": len(sentence),
                    "token_count": len(sentence.split()),
                    "city": row["city"],
                }
            )

    sent_df = pd.DataFrame(rows)
    log_step(
        "STEP10",
        f"拆分完成: {len(sent_df)} 句子, 均值 {len(sent_df) / len(df):.1f} 句/评论",
    )
    print(
        f"   char_len 分布: min={sent_df['char_len'].min()}, "
        f"median={sent_df['char_len'].median():.0f}, "
        f"mean={sent_df['char_len'].mean():.0f}, "
        f"max={sent_df['char_len'].max()}"
    )

    assert_shape(sent_df, expected_rows=44000, tolerance=0.20, label="句子总数")
    out_path = Path("data/intermediate/sentences.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sent_df.to_pickle(out_path)
    print(f"[SAVE] {out_path} ({len(sent_df)} 行)")


if __name__ == "__main__":
    main()

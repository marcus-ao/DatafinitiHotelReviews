"""
03_split_sentences.py
步骤 8：评论正文 → 句子级拆分
输入:  data/intermediate/cleaned_reviews.pkl
输出: data/intermediate/sentences.pkl
"""

import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from utils import load_config, log_step, assert_shape

try:
    import spacy
    _NLP = None
    def get_nlp():
        global _NLP
        if _NLP is None:
            _NLP = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
            # 增大管道 max_length 防止超长文本报错
            _NLP.max_length = 2_000_000
        return _NLP
except ImportError:
    raise ImportError("请先安装 spacy 并下载模型: python -m spacy download en_core_web_sm")


def _merge_fragments(sents: list[str], min_len: int) -> list[str]:
    """将过短碎片句合并到前一句"""
    if not sents:
        return sents
    merged = [sents[0]]
    for s in sents[1:]:
        if len(s) < min_len and merged:
            merged[-1] = merged[-1].rstrip() + " " + s.lstrip()
        else:
            merged.append(s)
    return merged


def split_review(text: str, min_len: int, merge_len: int, nlp) -> list[str]:
    """将单条评论拆分为句子列表"""
    if not text or not text.strip():
        return []
    # 预处理：多余空白
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
    raw_sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    filtered = [s for s in raw_sents if len(s) >= min_len]
    merged = _merge_fragments(filtered, merge_len)
    return [s for s in merged if len(s) >= min_len]


def main():
    cfg = load_config()
    scfg = cfg["sentence"]
    min_len   = scfg["min_sentence_length"]
    merge_len = scfg["min_fragment_merge_length"]

    df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
    log_step("STEP8", f"输入: {len(df)} 条评论 → 句子拆分")

    nlp = get_nlp()
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="分句"):
        sents = split_review(
            row["review_text_clean"], min_len, merge_len, nlp
        )
        for order, sent in enumerate(sents):
            rows.append({
                "sentence_id":    f"{row['review_id']}_s{order:03d}",
                "review_id":      row["review_id"],
                "hotel_id":       row["hotel_id"],
                "sentence_order": order,
                "sentence_text":  sent,
                "char_len":       len(sent),
                "token_count":    len(sent.split()),
            })

    sent_df = pd.DataFrame(rows)
    log_step("STEP8", f"拆分完成: {len(sent_df)} 句子, 均值 {len(sent_df)/len(df):.1f} 句/评论")

    # 统计
    print(f"   char_len 分布: min={sent_df['char_len'].min()}, "
          f"median={sent_df['char_len'].median():.0f}, "
          f"mean={sent_df['char_len'].mean():.0f}, "
          f"max={sent_df['char_len'].max()}")

    assert_shape(sent_df, expected_rows=46000, tolerance=0.20, label="句子总数")
    out_path = Path("data/intermediate/sentences.pkl")
    sent_df.to_pickle(out_path)
    print(f"💾 保存: {out_path} ({len(sent_df)} 行)")


if __name__ == "__main__":
    main()

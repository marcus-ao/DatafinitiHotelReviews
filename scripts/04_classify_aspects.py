"""
04_classify_aspects.py
步骤 11：句子 -> 方面分类（关键词规则 + zero-shot）
输入:  data/intermediate/sentences.pkl
输出: data/intermediate/aspect_labels.pkl
"""

from collections import Counter
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils import (
    ALL_ASPECT_CATEGORIES,
    ASPECT_CATEGORIES,
    ensure_columns,
    load_config,
    log_step,
)

ASPECT_KEYWORDS: dict[str, list[str]] = {
    "location_transport": [
        r"\b(?:location|located|area|neighborhood|district|proximity|close\s+to|near|nearby|"
        r"walk(?:ing|able)?|transit|metro|subway|bus|tram|transport(?:ation)?|"
        r"airport|downtown|central|convenient|access(?:ible)?|commute|distance)\b"
    ],
    "cleanliness": [
        r"\b(?:clean|cleaning|cleanliness|spotless|hygienic|hygiene|dirty|filthy|dust|dusty|"
        r"stain|stained|mold|mould|odor|odour|smell|smelly|musty|fresh|sanitary|tidy|untidy|"
        r"gross|disgusting|sanitiz)\b"
    ],
    "service": [
        r"\b(?:staff|service|reception|front\s*desk|concierge|housekeep(?:er|ing)|"
        r"check[- ]?(?:in|out)|bellman|bellhop|valet|doorman|manager|employee|"
        r"helpful|friendly|rude|professional|attentive|responsive|courteous|polite|impolite|"
        r"accommodat(?:ing|ion)|assist(?:ance|ed|ing)?|welcom|greeting)\b"
    ],
    "room_facilities": [
        r"\b(?:room|suite|bed(?:ding|room|s)?|pillow|mattress|linen|sheet|towel|bathroom|shower|"
        r"bathtub|tub|toilet|sink|fridge|refrigerator|tv|television|wifi|wi-fi|internet|"
        r"air\s*con(?:dition(?:ing|er)?)?|ac|heater|heat(?:ing)?|balcon(?:y|ies)|view|window|"
        r"closet|wardrobe|safe|iron|hairdryer|minibar|amenity|amenities|furnish|furniture|"
        r"couch|sofa|chair|desk|kitchen(?:ette)?|microwave|coffee|kettle|pool|gym|spa|elevator|"
        r"lift|parking|lobby|lounge|facility|facilities)\b"
    ],
    "quiet_sleep": [
        r"\b(?:quiet|noise|noisy|loud|sound|sleep|sleeping|rest|disturb(?:ance|ed|ing)?|"
        r"peaceful|tranquil|calm|silent|wall(?:s)?|thin\s+wall|neighbor|party|music|"
        r"traffic|street\s+noise|snore|insomnia|late\s+night|wake\s+up|woke\s+up|kept\s+awake)\b"
    ],
    "value": [
        r"\b(?:value|price|cost|expensive|cheap|affordable|overpriced|worth|money|rate|rates|"
        r"deal|bargain|budget|luxury|fee|charge|billing|receipt|paid|pay|pricey|reasonable|"
        r"quality(?:\s+for\s+(?:the\s+)?price)?|bang\s+for|\$)\b"
    ],
}

ZS_LABELS = [
    "location and transportation",
    "cleanliness and hygiene",
    "staff and service",
    "room and facilities",
    "noise level and sleep quality",
    "value for money",
]
ZS_LABEL_MAP = dict(zip(ZS_LABELS, ASPECT_CATEGORIES))


def match_aspects_rule(text: str) -> list[dict]:
    matched: list[dict] = []
    lowered = text.lower()
    for aspect, patterns in ASPECT_KEYWORDS.items():
        for pattern in patterns:
            match = re.search(pattern, lowered, re.IGNORECASE)
            if match:
                matched.append(
                    {
                        "aspect": aspect,
                        "confidence": 0.85,
                        "label_source": "rule",
                        "evidence_span": match.group(0),
                    }
                )
                break
    return matched


def get_transformers_device() -> int:
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def batch_zeroshot(
    records: list[tuple[str, str]],
    classifier,
    threshold: float,
    batch_size: int,
) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {}
    ids = [record[0] for record in records]
    texts = [record[1] for record in records]

    for start in tqdm(range(0, len(texts), batch_size), desc="ZeroShot"):
        batch_ids = ids[start : start + batch_size]
        batch_texts = texts[start : start + batch_size]
        outputs = classifier(
            batch_texts,
            candidate_labels=ZS_LABELS,
            multi_label=True,
            hypothesis_template="This sentence is about the hotel's {}.",
        )
        if isinstance(outputs, dict):
            outputs = [outputs]

        for sentence_id, output in zip(batch_ids, outputs):
            labels = []
            for label, score in zip(output["labels"], output["scores"]):
                if score >= threshold:
                    labels.append(
                        {
                            "aspect": ZS_LABEL_MAP[label],
                            "confidence": round(float(score), 3),
                            "label_source": "zeroshot",
                            "evidence_span": "",
                        }
                    )
            results[sentence_id] = labels
    return results


def build_label_rows(sent_df: pd.DataFrame, zeroshot_map: dict[str, list[dict]]) -> list[dict]:
    rows: list[dict] = []
    for _, row in sent_df.iterrows():
        labels = row["rule_labels"]
        if not labels:
            labels = zeroshot_map.get(row["sentence_id"], [])
        if not labels:
            labels = [
                {
                    "aspect": "general",
                    "confidence": 0.5,
                    "label_source": "rule",
                    "evidence_span": "",
                }
            ]

        for label in labels:
            rows.append(
                {
                    "sentence_id": row["sentence_id"],
                    "review_id": row["review_id"],
                    "hotel_id": row["hotel_id"],
                    "aspect": label["aspect"],
                    "confidence": label["confidence"],
                    "label_source": label["label_source"],
                    "evidence_span": label["evidence_span"],
                }
            )
    return rows


def main():
    cfg = load_config()
    acfg = cfg["aspect"]
    threshold = acfg["zeroshot_threshold"]
    batch_size = acfg["zeroshot_batch_size"]
    min_char = acfg["zeroshot_min_char_len"]

    sent_df = pd.read_pickle("data/intermediate/sentences.pkl")
    ensure_columns(
        sent_df,
        ["sentence_id", "review_id", "hotel_id", "sentence_text", "char_len"],
        "sentences.pkl",
    )
    log_step("STEP11", f"输入: {len(sent_df)} 句子")

    log_step("STEP11a", "关键词规则分类")
    sent_df["rule_labels"] = sent_df["sentence_text"].apply(match_aspects_rule)
    sent_df["rule_hit"] = sent_df["rule_labels"].apply(bool)
    print(
        f"[OK] 规则命中: {int(sent_df['rule_hit'].sum())} / {len(sent_df)} "
        f"({sent_df['rule_hit'].mean():.1%})"
    )

    needs_zs = sent_df[(~sent_df["rule_hit"]) & (sent_df["char_len"] >= min_char)].copy()
    zeroshot_map: dict[str, list[dict]] = {}

    if not needs_zs.empty:
        log_step("STEP11b", f"Zero-shot 分类: {len(needs_zs)} 句")
        from transformers import pipeline

        device = get_transformers_device()
        classifier = pipeline(
            "zero-shot-classification",
            model=acfg["zeroshot_model"],
            device=device,
            batch_size=batch_size,
        )
        zeroshot_map = batch_zeroshot(
            list(needs_zs[["sentence_id", "sentence_text"]].itertuples(index=False, name=None)),
            classifier,
            threshold,
            batch_size,
        )

    label_rows = build_label_rows(sent_df, zeroshot_map)
    label_df = pd.DataFrame(label_rows)
    ensure_columns(
        label_df,
        ["sentence_id", "review_id", "hotel_id", "aspect", "confidence", "label_source"],
        "aspect_labels",
    )

    unique_sentence_count = label_df["sentence_id"].nunique()
    if unique_sentence_count != len(sent_df):
        raise AssertionError(
            f"方面标签未覆盖全部句子: {unique_sentence_count} / {len(sent_df)}"
        )

    print("\n[OK] 方面分布:")
    aspect_counter = Counter(label_df["aspect"])
    for aspect, count in aspect_counter.most_common():
        print(f"   {aspect}: {count}")

    unknown_aspects = set(label_df["aspect"]) - set(ALL_ASPECT_CATEGORIES)
    if unknown_aspects:
        raise AssertionError(f"检测到未定义方面标签: {sorted(unknown_aspects)}")

    print("\n   label_source 分布:")
    print(label_df["label_source"].value_counts().to_string())

    out_path = Path("data/intermediate/aspect_labels.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    label_df.to_pickle(out_path)
    print(f"\n[SAVE] {out_path} ({len(label_df)} 行)")


if __name__ == "__main__":
    main()

"""
04_classify_aspects.py
步骤 9：句子 → 方面分类（关键词规则 + zero-shot DeBERTa）
输入:  data/intermediate/sentences.pkl
输出: data/intermediate/aspect_classified.pkl
"""

import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from utils import load_config, log_step, assert_shape

# ──────────────────────────────────────────────
# 方面关键词规则表（英文）
# ──────────────────────────────────────────────
ASPECT_KEYWORDS: dict[str, list[str]] = {
    "location_transport": [
        r"\b(?:location|located|area|neighborhood|district|proximity|close to|near|nearby|"
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
        r"accommodat(?:ing|ion)|assist(?:ance|ed|ing)?|welcom|greeting|check\s+in|check\s+out)\b"
    ],
    "room_facilities": [
        r"\b(?:room|suite|bed(?:ding|room|s)?|pillow|mattress|linen|sheet|towel|bathroom|shower|"
        r"bathtub|tub|toilet|sink|fridge|refrigerator|TV|television|wifi|wi-fi|internet|"
        r"air\s*con(?:dition(?:ing|er)?)?|AC|heater|heat(?:ing)?|balcon(?:y|ies)|view|window|"
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


def match_aspects_rule(text: str) -> list[str]:
    """关键词规则分类，返回匹配到的方面列表（可多标签）"""
    matched = []
    text_lower = text.lower()
    for aspect, patterns in ASPECT_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, text_lower, re.IGNORECASE):
                matched.append(aspect)
                break
    return matched if matched else ["general"]


def batch_zeroshot(texts: list[str], classifier, threshold: float,
                   candidate_labels: list[str], batch_size: int) -> list[list[str]]:
    """批量 zero-shot 分类，返回超过阈值的方面列表"""
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="ZeroShot"):
        batch = texts[i : i + batch_size]
        outputs = classifier(
            batch,
            candidate_labels=candidate_labels,
            multi_label=True,
            hypothesis_template="This sentence is about the hotel's {}."
        )
        if isinstance(outputs, dict):   # 单条返回 dict
            outputs = [outputs]
        for out in outputs:
            matched = [
                label for label, score in zip(out["labels"], out["scores"])
                if score >= threshold
            ]
            results.append(matched if matched else ["general"])
    return results


def main():
    cfg = load_config()
    acfg = cfg["aspect"]
    threshold  = acfg["zeroshot_threshold"]
    batch_size = acfg["zeroshot_batch_size"]
    min_char   = acfg["zeroshot_min_char_len"]

    df = pd.read_pickle("data/intermediate/sentences.pkl")
    log_step("STEP9", f"输入: {len(df)} 句子")

    # ── 阶段 1：关键词规则 ──
    log_step("STEP9a", "关键词规则分类")
    df["aspects_rule"] = df["sentence_text"].apply(match_aspects_rule)
    df["rule_unclassified"] = df["aspects_rule"].apply(lambda x: x == ["general"])
    n_rule = (~df["rule_unclassified"]).sum()
    n_unclassified = df["rule_unclassified"].sum()
    print(f"✅ 规则命中: {n_rule} ({n_rule/len(df)*100:.1f}%), "
          f"未命中: {n_unclassified} ({n_unclassified/len(df)*100:.1f}%)")

    # ── 阶段 2：zero-shot（仅对规则未命中 + 足够长的句子）──
    needs_zs = df[df["rule_unclassified"] & (df["char_len"] >= min_char)]
    log_step("STEP9b", f"Zero-shot 分类: {len(needs_zs)} 句")

    df["aspects_zeroshot"] = [["general"]] * len(df)
    df["label_source"] = "rule"

    if len(needs_zs) > 0:
        from transformers import pipeline
        print(f"   加载模型: {acfg['zeroshot_model']}")
        try:
            classifier = pipeline(
                "zero-shot-classification",
                model=acfg["zeroshot_model"],
                device=0,        # GPU；改为 -1 使用 CPU
                batch_size=batch_size
            )
        except Exception:
            print("   GPU 不可用，使用 CPU")
            classifier = pipeline(
                "zero-shot-classification",
                model=acfg["zeroshot_model"],
                device=-1,
                batch_size=16
            )

        # 候选标签（人类可读，映射到字段名）
        label_display = [
            "location and transportation",
            "cleanliness and hygiene",
            "staff and service",
            "room and facilities",
            "noise level and sleep quality",
            "value for money",
        ]
        label_map = dict(zip(label_display, acfg["categories"]))

        zs_texts = needs_zs["sentence_text"].tolist()
        zs_results_raw = batch_zeroshot(
            zs_texts, classifier, threshold, label_display, batch_size
        )
        # 映射回字段名
        zs_results = [
            [label_map.get(l, "general") for l in res]
            for res in zs_results_raw
        ]

        # 回填
        for idx, aspects in zip(needs_zs.index, zs_results):
            df.at[idx, "aspects_zeroshot"] = aspects
            df.at[idx, "label_source"] = "zeroshot"

    # ── 合并 aspects ──
    def merge_aspects(row):
        if row["label_source"] == "rule":
            return row["aspects_rule"]
        else:
            a = row["aspects_zeroshot"]
            return a if a != ["general"] else row["aspects_rule"]

    df["aspects"] = df.apply(merge_aspects, axis=1)
    df["primary_aspect"] = df["aspects"].apply(lambda x: x[0] if x else "general")

    # 分布统计
    from collections import Counter
    all_aspects = [a for lst in df["aspects"] for a in lst]
    print("\n✅ 方面分布:")
    for aspect, cnt in Counter(all_aspects).most_common():
        print(f"   {aspect}: {cnt} ({cnt/len(df)*100:.1f}%)")
    print(f"\n   label_source 分布:")
    print(df["label_source"].value_counts().to_string())

    assert_shape(df, expected_rows=46000, tolerance=0.20, label="方面分类后")
    out_path = Path("data/intermediate/aspect_classified.pkl")
    df.to_pickle(out_path)
    print(f"\n💾 保存: {out_path} ({len(df)} 行)")


if __name__ == "__main__":
    main()

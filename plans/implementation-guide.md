# 从 0 到 1 实施落地指南

> **版本**: v1.0
> **前置文档**: `plans/plan.md`（战略计划 v3.0）
> **目标**: 一步一步、可直接执行的操作手册

---

## 第0步：环境准备

### 0.1 Python 环境

```bash
# 创建虚拟环境
python -m venv .venv
# Windows 激活
.venv\Scripts\activate
# Linux/Mac 激活
# source .venv/bin/activate

# 安装依赖
pip install pandas numpy spacy scikit-learn tqdm pyyaml
pip install transformers sentence-transformers
pip install vaderSentiment      # 仅作辅助参考，主情感用 nlptown
pip install chromadb            # 向量数据库
pip install psycopg2-binary sqlalchemy  # PostgreSQL
pip install jupyter             # notebook 分析用

# spaCy 英文模型
python -m spacy download en_core_web_sm
```

### 0.2 HuggingFace 模型预下载（可选，避免运行时等待）

```python
# 在 Python 中预下载
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer

# 零样本分类
AutoTokenizer.from_pretrained("MoritzLaurer/deberta-v3-base-zeroshot-v2.0")
AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/deberta-v3-base-zeroshot-v2.0")

# 情感分析（5分类，150K评论训练）
AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Embedding
SentenceTransformer("BAAI/bge-small-en-v1.5")

# Reranker（后续 RAG 阶段才用，可暂不下载）
# AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
```

### 0.3 项目目录初始化

```bash
mkdir -p data/intermediate
mkdir -p scripts
mkdir -p notebooks
mkdir -p configs
mkdir -p models     # 本地缓存模型（可选）
```

### 0.4 配置文件

创建 `configs/params.yaml`：

```yaml
data:
  raw_file: "raw_data/Datafiniti_Hotel_Reviews.csv"
  experiment_cities:
    - "San Diego"
    - "San Francisco"
    - "New Orleans"
    - "Atlanta"
    - "Orlando"
    - "Seattle"
    - "Chicago"
    - "Honolulu"
    - "Dallas"
    - "Anaheim"
  db_url: "postgresql://hotel_user:password@localhost:5432/hotel_reviews_kb"

cleaning:
  min_text_length: 50
  min_preserve_length: 100
  noise_exact_matches: ["MoreMore", "More", "N/A", "NA"]

hotel_filter:
  min_reviews_per_hotel: 5
  min_avg_text_length: 100

sentence:
  min_sentence_length: 10
  max_sentence_length_for_split: 200
  min_fragment_merge_length: 15
  splitter: "spacy"

aspect:
  categories:
    - location_transport
    - cleanliness
    - service
    - room_facilities
    - quiet_sleep
    - value
  zeroshot_model: "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
  zeroshot_threshold: 0.45
  zeroshot_min_char_len: 30
  zeroshot_batch_size: 32

sentiment:
  model: "nlptown/bert-base-multilingual-uncased-sentiment"
  batch_size: 64
  # 5分类 → 3分类映射: 1-2→negative, 3→neutral, 4-5→positive

recency:
  reference_date: "2019-01-30"
  buckets:
    recent_90d: 1.2
    recent_1y: 1.0
    recent_2y: 0.9
    older: 0.8

embedding:
  model: "BAAI/bge-small-en-v1.5"
  dimension: 384
  chroma_collection: "hotel_evidence"
  chroma_persist_dir: "data/chroma_db"

reranker:
  model: "BAAI/bge-reranker-base"
```

---

## 第一步：加载与城市筛选

**脚本**: `scripts/01_load_and_filter.py`
**输入**: `raw_data/Datafiniti_Hotel_Reviews.csv`（10,000 行）
**输出**: `data/intermediate/city_filtered.pkl`（~6,171 行）

```python
import pandas as pd
import hashlib
import yaml
from pathlib import Path

# 加载配置
with open("configs/params.yaml") as f:
    cfg = yaml.safe_load(f)

# Step 1: 加载
raw_df = pd.read_csv(cfg["data"]["raw_file"], low_memory=False)
assert raw_df.shape == (10000, 26), f"数据完整性校验失败: {raw_df.shape}"
print(f"✅ 加载完成: {raw_df.shape}")

# Step 2: 字段筛选与重命名
COLUMN_MAP = {
    "id": "source_id", "keys": "hotel_key", "name": "hotel_name",
    "address": "address", "city": "city", "province": "state",
    "country": "country", "postalCode": "postal_code",
    "latitude": "lat", "longitude": "lng",
    "categories": "categories", "primaryCategories": "primary_category",
    "reviews.date": "review_date_raw", "reviews.rating": "rating",
    "reviews.title": "review_title", "reviews.text": "review_text_raw",
    "reviews.sourceURLs": "review_source_url", "websites": "hotel_website",
}
core_df = raw_df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)

# Step 3: 城市过滤
city_df = core_df[core_df["city"].isin(cfg["data"]["experiment_cities"])].copy()
print(f"✅ 城市过滤: {len(raw_df)} → {len(city_df)} ({len(city_df)} 行)")
print(f"   城市分布: {dict(city_df['city'].value_counts())}")

# Step 4: 主键生成
def make_hotel_id(key: str) -> str:
    return hashlib.md5(key.encode()).hexdigest()[:12]

def make_review_id(hotel_id: str, date: str, title: str, text: str) -> str:
    raw = f"{hotel_id}|{date}|{title}|{text[:200]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

city_df["hotel_id"] = city_df["hotel_key"].apply(make_hotel_id)
city_df["review_id"] = city_df.apply(
    lambda r: make_review_id(r["hotel_id"], str(r["review_date_raw"]),
                              str(r["review_title"]), str(r["review_text_raw"])), axis=1)

print(f"✅ 主键生成: {city_df['hotel_id'].nunique()} 家酒店, {city_df['review_id'].nunique()} 条评论")

# 保存中间结果
city_df.to_pickle("data/intermediate/city_filtered.pkl")
print(f"💾 保存: data/intermediate/city_filtered.pkl")
```

**检查点**：
```
✅ 加载完成: (10000, 26)
✅ 城市过滤: 10000 → 6171
✅ 主键生成: 211 家酒店, ~6171 条评论
```

---

## 第二步：评论清洗

**脚本**: `scripts/02_clean_reviews.py`
**输入**: `data/intermediate/city_filtered.pkl`
**输出**: `data/intermediate/cleaned_reviews.pkl`

```python
import pandas as pd
import re
import yaml

with open("configs/params.yaml") as f:
    cfg = yaml.safe_load(f)

city_df = pd.read_pickle("data/intermediate/city_filtered.pkl")
print(f"📥 加载: {len(city_df)} 行")

# ── Step 5: 无效评论过滤 ──
noise = cfg["cleaning"]["noise_exact_matches"]
min_len = cfg["cleaning"]["min_text_length"]

valid_mask = (
    city_df["review_text_raw"].notna() &
    (city_df["review_text_raw"].str.strip().str.len() >= min_len) &
    (~city_df["review_text_raw"].str.strip().isin(noise)) &
    city_df["rating"].notna() &
    city_df["rating"].between(1, 5)
)
valid_df = city_df[valid_mask].copy()
print(f"✅ 无效过滤: {len(city_df)} → {len(valid_df)} (删除 {len(city_df)-len(valid_df)} 条)")

# ── Step 6: 管理者回复去除 ──
MANAGER_PATTERNS = [
    r"(?:Dear|Hello)\s+(?:Guest|Traveler|Sir|Madam|valued\s+guest|Mr\.|Mrs\.|Ms\.)",
    r"On behalf of (?:the|our)\s+(?:staff|team|management|hotel|entire)",
    r"Thank(?:s|\s+you)\s+(?:for|so\s+much\s+for)\s+(?:your|the|taking)\s+(?:review|feedback|comment|kind|candid|recent|time|visit|stay|choosing|sharing|staying|visiting)",
    r"(?:We|I)\s+(?:appreciate|value)\s+(?:your|the)\s+(?:feedback|review|comment|time|patronage|kind\s+words)",
    r"(?:We|I)(?:'re|'m| are| am)\s+(?:sorry|happy|glad|thrilled|delighted|pleased|so glad|very sorry)\s+(?:to\s+hear|that\s+you|you\s+(?:had|enjoyed|experienced))",
    r"(?:We|I)\s+(?:hope|look forward)\s+to\s+(?:see|seeing|welcome|welcoming)\s+you",
    r"(?:Please|Do not hesitate to)\s+contact\s+(?:us|me)",
    r"(?:Sincerely|Kind\s+Regards|Best\s+Regards|Warm\s+Regards|Respectfully),?\s*(?:\n|\s)*\w",
]

def remove_manager_response(text: str) -> tuple[str, bool]:
    """返回 (cleaned_text, had_manager_reply)"""
    if pd.isna(text):
        return text, False
    min_preserve = cfg["cleaning"]["min_preserve_length"]
    earliest = len(text)
    for pat in MANAGER_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m and m.start() > min_preserve and m.start() < earliest:
            earliest = m.start()
    had_reply = earliest < len(text)
    cleaned = text[:earliest].rstrip()
    cleaned = re.sub(r'\.\.\.\s*More\s*$', '', cleaned, flags=re.IGNORECASE).rstrip()
    return cleaned, had_reply

results = valid_df["review_text_raw"].apply(remove_manager_response)
valid_df["review_text_clean"] = results.apply(lambda x: x[0])
valid_df["has_manager_reply"] = results.apply(lambda x: x[1])
valid_df["char_len_raw"] = valid_df["review_text_raw"].str.len()
valid_df["char_len_clean"] = valid_df["review_text_clean"].str.len()

mgr_count = valid_df["has_manager_reply"].sum()
avg_reduction = (valid_df["char_len_raw"] - valid_df["char_len_clean"]).mean()
print(f"✅ 管理者回复去除: {mgr_count} 条 ({mgr_count/len(valid_df)*100:.1f}%), 平均截短 {avg_reduction:.0f} 字符")

# ── Step 7: 去重 ──
before = len(valid_df)
dedup_df = valid_df.drop_duplicates(
    subset=["hotel_id", "review_title", "review_text_clean"], keep="first"
).copy()
print(f"✅ 去重: {before} → {len(dedup_df)} (删除 {before - len(dedup_df)} 条)")

# ── Step 8: 日期与评分标准化 ──
dedup_df["review_date"] = dedup_df["review_date_raw"].apply(
    lambda d: pd.Timestamp(d) if pd.notna(d) else pd.NaT  # 修复 .000Z 解析
)
dedup_df["review_year"] = dedup_df["review_date"].dt.year.astype("Int64")
dedup_df["review_month"] = dedup_df["review_date"].dt.month.astype("Int64")

REF = pd.Timestamp(cfg["recency"]["reference_date"])
def recency_bucket(d):
    if pd.isna(d): return "unknown"
    delta = (REF - d).days
    if delta <= 90: return "recent_90d"
    elif delta <= 365: return "recent_1y"
    elif delta <= 730: return "recent_2y"
    else: return "older"

dedup_df["recency_bucket"] = dedup_df["review_date"].apply(recency_bucket)
dedup_df["sentiment_weak"] = dedup_df["rating"].apply(
    lambda r: "positive" if r >= 4 else ("negative" if r <= 2 else "neutral"))
dedup_df["full_text"] = dedup_df["review_title"].fillna("") + " " + dedup_df["review_text_clean"]

print(f"✅ 日期标准化: {dedup_df['review_date'].notna().sum()}/{len(dedup_df)} 有效")
print(f"   时间桶分布: {dict(dedup_df['recency_bucket'].value_counts())}")

# ── Step 9: 酒店质量过滤 ──
hotel_stats = dedup_df.groupby("hotel_id").agg(
    n=("review_id", "count"),
    avg_len=("review_text_clean", lambda x: x.str.len().mean())
).reset_index()

min_rev = cfg["hotel_filter"]["min_reviews_per_hotel"]
min_avglen = cfg["hotel_filter"]["min_avg_text_length"]
qualified = hotel_stats[(hotel_stats["n"] >= min_rev) & (hotel_stats["avg_len"] >= min_avglen)]["hotel_id"]
experiment_df = dedup_df[dedup_df["hotel_id"].isin(qualified)].copy()

print(f"✅ 酒店过滤(≥{min_rev}评论): {dedup_df['hotel_id'].nunique()} → {experiment_df['hotel_id'].nunique()} 家酒店")
print(f"   评论: {len(dedup_df)} → {len(experiment_df)}")

experiment_df.to_pickle("data/intermediate/cleaned_reviews.pkl")
print(f"💾 保存: data/intermediate/cleaned_reviews.pkl")
```

**检查点**：
```
✅ 无效过滤: 6171 → ~6155
✅ 管理者回复去除: ~1200 条 (19.4%)
✅ 去重: ~6155 → ~5990
✅ 日期标准化: 100% 有效
✅ 酒店过滤: 211 → ~146 家, ~5850 条评论
```

---

## 第三步：分句

**脚本**: `scripts/03_split_sentences.py`
**输入**: `data/intermediate/cleaned_reviews.pkl`
**输出**: `data/intermediate/sentences.pkl`

```python
import pandas as pd
import spacy
import yaml
from tqdm import tqdm

with open("configs/params.yaml") as f:
    cfg = yaml.safe_load(f)

experiment_df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
print(f"📥 加载: {len(experiment_df)} 条评论")

nlp = spacy.load("en_core_web_sm")
min_len = cfg["sentence"]["min_sentence_length"]
merge_len = cfg["sentence"]["min_fragment_merge_length"]

sentence_records = []
for _, row in tqdm(experiment_df.iterrows(), total=len(experiment_df), desc="分句"):
    doc = nlp(row["review_text_clean"])
    sents = [s.text.strip() for s in doc.sents]

    # 后处理：合并碎片句
    merged = []
    for s in sents:
        if len(s) < merge_len and merged:
            merged[-1] = merged[-1] + " " + s  # 粘到前一句
        elif len(s) >= min_len:
            merged.append(s)

    for order, sent_text in enumerate(merged):
        sentence_records.append({
            "sentence_id": f"{row['review_id']}_{order:02d}",
            "review_id": row["review_id"],
            "hotel_id": row["hotel_id"],
            "sentence_text": sent_text,
            "sentence_order": order,
            "char_len": len(sent_text),
            "token_count": len(sent_text.split()),
            "city": row["city"],
        })

sentence_df = pd.DataFrame(sentence_records)
print(f"✅ 分句完成: {len(experiment_df)} 评论 → {len(sentence_df)} 句")
print(f"   均 {len(sentence_df)/len(experiment_df):.1f} 句/评论")
print(f"   句长分布: 中位 {sentence_df['char_len'].median():.0f}, 均值 {sentence_df['char_len'].mean():.0f}")

sentence_df.to_pickle("data/intermediate/sentences.pkl")
print(f"💾 保存: data/intermediate/sentences.pkl")
```

**检查点**：
```
✅ ~5,850 评论 → ~43,000–46,000 句
✅ 均 ~7.5 句/评论
```

---

## 第四步：方面分类

**脚本**: `scripts/04_classify_aspects.py`
**输入**: `data/intermediate/sentences.pkl`
**输出**: `data/intermediate/aspect_labels.pkl`

```python
import pandas as pd
import re
import yaml
from tqdm import tqdm
from transformers import pipeline

with open("configs/params.yaml") as f:
    cfg = yaml.safe_load(f)

sentence_df = pd.read_pickle("data/intermediate/sentences.pkl")
print(f"📥 加载: {len(sentence_df)} 句")

# ━━━━━━━━━━━━━━━━━━━━━━━━━
# 阶段 1: 关键词规则分类
# ━━━━━━━━━━━━━━━━━━━━━━━━━
ASPECT_PATTERNS = {
    "location_transport": r"\b(?:location|located|downtown|walk(?:ing|ed)?|subway|metro|bus\s+(?:stop|station)|airport|beach|block(?:s)?|convenient(?:ly)?|parking|drive|uber|taxi|mile(?:s)?|close\s+to|minutes?\s+(?:from|to|away)|near(?:by)?|district|neighborhood|street|transit)\b",
    "cleanliness": r"\b(?:clean(?:ed|liness)?|dirty|stain(?:s|ed)?|smell(?:s|ed|y)?|mold(?:y)?|housekeep(?:ing|er)?|spotless|filth(?:y)?|tidy|gross|dust(?:y)?|sanitar(?:y|ize)|hair\s+(?:in|on)|bug(?:s)?|cockroach|bed\s*bug|pest)\b",
    "service": r"\b(?:staff|friendly|helpful|rude|front\s+desk|check[\s-]?in|check[\s-]?out|concierge|(?:room\s+)?service|manager|attentive|responsive|accommodat(?:ing|ed)|valet|bellman|recept(?:ion|ionist)|courteous|polite|unprofessional|welcom(?:e|ing))\b",
    "room_facilities": r"\b(?:room(?:s)?|bed(?:s)?|bathroom|shower|pool|gym|fitness|wi[\s-]?fi|wifi|internet|tv|television|a[\s/]?c|air\s*condition(?:ing|er)?|elevator|towel(?:s)?|pillow(?:s)?|comfort(?:able)?|spacious|tiny|small|large|suite|king|queen|minibar|fridge|refrigerator|microwave|balcony|view|breakfast|renovate(?:d|ion)?|update(?:d)?|decor|furnish|carpet|curtain)\b",
    "quiet_sleep": r"\b(?:quiet|noisy|noise|loud|sleep|slept|soundproof|thin\s+wall(?:s)?|neighbor(?:s)?|street\s+noise|peaceful|disturb(?:ed|ing)?|earplug(?:s)?|rest(?:ful|ed)?|silent|tranquil|party|music|construction|traffic\s+noise)\b",
    "value": r"\b(?:price(?:d|s)?|value|expensive|cheap|worth|overpriced|deal|money(?:'s)?|afford(?:able)?|cost(?:s|ly)?|rate(?:s)?|pay(?:ing)?|dollar(?:s)?|bargain|reasonable|budget|splurge|rip[\s-]?off|bang\s+for)\b",
}

def rule_classify(text: str) -> list[dict]:
    lower = text.lower()
    results = []
    for asp, pat in ASPECT_PATTERNS.items():
        m = re.search(pat, lower)
        if m:
            results.append({"aspect": asp, "source": "rule", "confidence": 0.85, "span": m.group()})
    return results

print("⏳ 阶段 1: 关键词规则分类...")
sentence_df["rule_aspects"] = sentence_df["sentence_text"].apply(rule_classify)
classified_mask = sentence_df["rule_aspects"].apply(len) > 0
n_classified = classified_mask.sum()
print(f"✅ 规则分类: {n_classified}/{len(sentence_df)} ({n_classified/len(sentence_df)*100:.1f}%)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━
# 阶段 2: Zero-shot 补充
# ━━━━━━━━━━━━━━━━━━━━━━━━━
unclassified = sentence_df[~classified_mask].copy()
# 仅对 ≥30 字符的句子做 zero-shot（过滤碎片）
zs_min_len = cfg["aspect"]["zeroshot_min_char_len"]
zs_candidates = unclassified[unclassified["char_len"] >= zs_min_len]
zs_fragments = unclassified[unclassified["char_len"] < zs_min_len]
print(f"⏳ 阶段 2: Zero-shot 分类 {len(zs_candidates)} 句 (跳过 {len(zs_fragments)} 短句)...")

classifier = pipeline(
    "zero-shot-classification",
    model=cfg["aspect"]["zeroshot_model"],
    device=0 if __import__("torch").cuda.is_available() else -1,
    batch_size=cfg["aspect"]["zeroshot_batch_size"],
)

ZS_LABELS = [
    "hotel location and transportation",
    "room cleanliness and hygiene",
    "staff service and hospitality",
    "room facilities and amenities",
    "noise level and sleep quality",
    "price and value for money",
]
LABEL_MAP = dict(zip(ZS_LABELS, cfg["aspect"]["categories"]))

threshold = cfg["aspect"]["zeroshot_threshold"]
zs_results = {}
texts = zs_candidates["sentence_text"].tolist()
ids = zs_candidates["sentence_id"].tolist()

for i in tqdm(range(0, len(texts), 32), desc="Zero-shot"):
    batch = texts[i:i+32]
    batch_ids = ids[i:i+32]
    outputs = classifier(batch, ZS_LABELS, multi_label=True)
    if not isinstance(outputs, list):
        outputs = [outputs]
    for sid, out in zip(batch_ids, outputs):
        aspects = []
        for label, score in zip(out["labels"], out["scores"]):
            if score >= threshold:
                aspects.append({"aspect": LABEL_MAP[label], "source": "zeroshot",
                                "confidence": round(score, 3), "span": ""})
        zs_results[sid] = aspects

sentence_df["zs_aspects"] = sentence_df["sentence_id"].map(zs_results).fillna("").apply(
    lambda x: x if isinstance(x, list) else [])

# ━━━━━━━━━━━━━━━━━━━━━━━━━
# 合并 + 构建 aspect_label 表
# ━━━━━━━━━━━━━━━━━━━━━━━━━
aspect_records = []
for _, row in sentence_df.iterrows():
    labels = row["rule_aspects"] if row["rule_aspects"] else row["zs_aspects"]
    if not labels:
        labels = [{"aspect": "general", "source": "rule", "confidence": 0.5, "span": ""}]
    for lab in labels:
        aspect_records.append({
            "sentence_id": row["sentence_id"],
            "hotel_id": row["hotel_id"],
            "aspect": lab["aspect"],
            "confidence": lab["confidence"],
            "label_source": lab["source"],
            "evidence_span": lab.get("span", ""),
        })

aspect_df = pd.DataFrame(aspect_records)
print(f"✅ 方面标签总数: {len(aspect_df)}")
print(f"   分布:\n{aspect_df['aspect'].value_counts().to_string()}")

aspect_df.to_pickle("data/intermediate/aspect_labels.pkl")
print(f"💾 保存: data/intermediate/aspect_labels.pkl")
```

**检查点**：
```
✅ 规则分类: ~55% 句子
✅ Zero-shot: ~13,000 句补充
✅ 方面标签总数: ~35,000–40,000
✅ room_facilities 占比最高
```

---

## 第五步：情感分类

**脚本**: `scripts/05_classify_sentiment.py`
**输入**: `data/intermediate/aspect_labels.pkl` + `sentences.pkl`
**输出**: `data/intermediate/aspect_sentiment.pkl`

```python
import pandas as pd
import yaml
from tqdm import tqdm
from transformers import pipeline

with open("configs/params.yaml") as f:
    cfg = yaml.safe_load(f)

aspect_df = pd.read_pickle("data/intermediate/aspect_labels.pkl")
sentence_df = pd.read_pickle("data/intermediate/sentences.pkl")
print(f"📥 加载: {len(aspect_df)} 标签, {len(sentence_df)} 句")

# 对每个唯一句子做一次情感分类（避免重复推理）
unique_sents = sentence_df[["sentence_id", "sentence_text"]].drop_duplicates("sentence_id")
print(f"⏳ 情感分类: {len(unique_sents)} 唯一句子...")

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=cfg["sentiment"]["model"],  # nlptown/bert-base-multilingual-uncased-sentiment
    device=0 if __import__("torch").cuda.is_available() else -1,
    batch_size=cfg["sentiment"]["batch_size"],
    truncation=True,
    max_length=512,
)

# 5分类 → 3分类映射
def map_star_to_sentiment(label: str, score: float) -> tuple[str, float]:
    """'1 star'~'5 stars' → positive/neutral/negative"""
    star = int(label[0])
    if star >= 4:
        return "positive", score
    elif star <= 2:
        return "negative", score
    else:
        return "neutral", score

texts = unique_sents["sentence_text"].tolist()
ids = unique_sents["sentence_id"].tolist()
sent_map = {}

for i in tqdm(range(0, len(texts), 64), desc="情感分析"):
    batch = texts[i:i+64]
    batch_ids = ids[i:i+64]
    outputs = sentiment_pipe(batch)
    for sid, out in zip(batch_ids, outputs):
        sentiment, confidence = map_star_to_sentiment(out["label"], out["score"])
        sent_map[sid] = {"sentiment": sentiment, "confidence": round(confidence, 3)}

# 合并到 aspect 表
aspect_df["sentiment"] = aspect_df["sentence_id"].map(lambda x: sent_map.get(x, {}).get("sentiment", "neutral"))
aspect_df["sent_confidence"] = aspect_df["sentence_id"].map(lambda x: sent_map.get(x, {}).get("confidence", 0.5))

print(f"✅ 情感分布:\n{aspect_df['sentiment'].value_counts().to_string()}")

aspect_df.to_pickle("data/intermediate/aspect_sentiment.pkl")
print(f"💾 保存: data/intermediate/aspect_sentiment.pkl")
```

**检查点**：
```
✅ positive ~60-65%, neutral ~15-20%, negative ~18-22%
```

---

## 第六步：聚合酒店方面画像

**脚本**: `scripts/06_build_profiles.py`
**输入**: `data/intermediate/aspect_sentiment.pkl` + `cleaned_reviews.pkl`
**输出**: `data/intermediate/hotel_profiles.pkl`

```python
import pandas as pd
import yaml

with open("configs/params.yaml") as f:
    cfg = yaml.safe_load(f)

aspect_df = pd.read_pickle("data/intermediate/aspect_sentiment.pkl")
review_df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
print(f"📥 加载: {len(aspect_df)} 标签")

RECENCY_W = cfg["recency"]["buckets"]
ASPECTS = cfg["aspect"]["categories"]  # 不含 general

# 关联 recency_bucket
sent_review = pd.read_pickle("data/intermediate/sentences.pkl")[["sentence_id", "review_id"]]
aspect_full = aspect_df.merge(sent_review, on="sentence_id")
aspect_full = aspect_full.merge(review_df[["review_id", "recency_bucket"]], on="review_id")
aspect_full["recency_weight"] = aspect_full["recency_bucket"].map(RECENCY_W)

# 仅对 6 个核心方面聚合（排除 general）
core = aspect_full[aspect_full["aspect"].isin(ASPECTS)]

profiles = []
for (hid, asp), grp in core.groupby(["hotel_id", "aspect"]):
    pos = grp[grp["sentiment"] == "positive"]
    neg = grp[grp["sentiment"] == "negative"]
    neu = grp[grp["sentiment"] == "neutral"]
    w_pos = (pos["recency_weight"]).sum()
    w_neg = (neg["recency_weight"]).sum()
    controversy = min(len(pos), len(neg)) / max(len(pos), len(neg), 1)
    score = round(w_pos - w_neg - controversy * 0.3, 3)
    profiles.append({
        "hotel_id": hid, "aspect": asp,
        "pos_count": len(pos), "neg_count": len(neg), "neu_count": len(neu),
        "total_count": len(grp),
        "recency_weighted_pos": round(w_pos, 3),
        "recency_weighted_neg": round(w_neg, 3),
        "controversy_score": round(controversy, 3),
        "final_aspect_score": score,
    })

profile_df = pd.DataFrame(profiles)
print(f"✅ 画像行数: {len(profile_df)} (预期 {review_df['hotel_id'].nunique() * 6})")
print(f"   方面得分示例:\n{profile_df.head(12).to_string(index=False)}")

profile_df.to_pickle("data/intermediate/hotel_profiles.pkl")
print(f"💾 保存: data/intermediate/hotel_profiles.pkl")
```

---

## 第七步：向量编码与 ChromaDB 构建

**脚本**: `scripts/07_build_vector_index.py`
**输入**: `data/intermediate/sentences.pkl` + `aspect_sentiment.pkl`
**输出**: `data/chroma_db/`（ChromaDB 持久化目录）

```python
import pandas as pd
import yaml
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

with open("configs/params.yaml") as f:
    cfg = yaml.safe_load(f)

sentence_df = pd.read_pickle("data/intermediate/sentences.pkl")
aspect_df = pd.read_pickle("data/intermediate/aspect_sentiment.pkl")
review_df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
print(f"📥 加载: {len(sentence_df)} 句")

# 构建元数据：每个句子的主方面 + 情感 + 酒店信息
# 取每个 sentence_id 的第一个方面作为主方面
primary_aspect = aspect_df.sort_values("confidence", ascending=False).drop_duplicates("sentence_id")
meta_df = sentence_df.merge(
    primary_aspect[["sentence_id", "aspect", "sentiment"]], on="sentence_id", how="left"
).merge(
    review_df[["review_id", "review_date", "rating", "recency_bucket", "hotel_name"]], on="review_id"
)
meta_df["aspect"] = meta_df["aspect"].fillna("general")
meta_df["sentiment"] = meta_df["sentiment"].fillna("neutral")

# Embedding 编码
print("⏳ 编码向量...")
model = SentenceTransformer(cfg["embedding"]["model"])
texts = meta_df["sentence_text"].tolist()
embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=256)
print(f"✅ 编码完成: {embeddings.shape}")

# ChromaDB 构建
print("⏳ 写入 ChromaDB...")
client = chromadb.PersistentClient(path=cfg["embedding"]["chroma_persist_dir"])
# 删除旧集合（如有）
try:
    client.delete_collection(cfg["embedding"]["chroma_collection"])
except:
    pass

collection = client.create_collection(
    name=cfg["embedding"]["chroma_collection"],
    metadata={"hnsw:space": "cosine"}
)

# 分批写入（ChromaDB 单次限制 ~41000 条）
BATCH = 5000
for i in tqdm(range(0, len(meta_df), BATCH), desc="写入 Chroma"):
    batch = meta_df.iloc[i:i+BATCH]
    batch_emb = embeddings[i:i+BATCH].tolist()
    collection.add(
        ids=batch["sentence_id"].tolist(),
        embeddings=batch_emb,
        documents=batch["sentence_text"].tolist(),
        metadatas=[{
            "hotel_id": r["hotel_id"],
            "city": r["city"],
            "aspect": r["aspect"],
            "sentiment": r["sentiment"],
            "rating": int(r["rating"]) if pd.notna(r["rating"]) else 0,
            "hotel_name": str(r["hotel_name"]),
            "review_date": str(r["review_date"])[:10] if pd.notna(r["review_date"]) else "",
            "recency_bucket": r.get("recency_bucket", "unknown"),
        } for _, r in batch.iterrows()]
    )

print(f"✅ ChromaDB 构建完成: {collection.count()} 向量")
print(f"💾 保存: {cfg['embedding']['chroma_persist_dir']}/")

# 快速验证
test_q = "quiet hotel near downtown with good service"
results = collection.query(query_texts=[test_q], n_results=5)
print(f"\n🔍 测试查询: '{test_q}'")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['aspect']}|{meta['sentiment']}] {meta['hotel_name']}: {doc[:80]}...")
```

---

## 第八步：写入 PostgreSQL 数据库

**脚本**: `scripts/08_export_to_postgres.py`
**输入**: 所有 `data/intermediate/*.pkl`
**输出**: PostgreSQL `hotel_reviews_kb` 数据库 `kb` schema

### 8.1 首次初始化（仅执行一次）

```sql
-- 在 psql 中执行
CREATE DATABASE hotel_reviews_kb ENCODING 'UTF8';
CREATE USER hotel_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE hotel_reviews_kb TO hotel_user;

-- 连接到 hotel_reviews_kb 后
\c hotel_reviews_kb
CREATE SCHEMA IF NOT EXISTS kb;
GRANT ALL ON SCHEMA kb TO hotel_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA kb GRANT ALL ON TABLES TO hotel_user;
```

### 8.2 建表 DDL（仅执行一次）

```sql
SET search_path TO kb;

CREATE TABLE kb.hotel (
    hotel_id        VARCHAR(12)     PRIMARY KEY,
    hotel_key       TEXT            NOT NULL UNIQUE,
    hotel_name      VARCHAR(255)    NOT NULL,
    address         TEXT,
    city            VARCHAR(100)    NOT NULL,
    state           CHAR(2)         NOT NULL,
    country         CHAR(2)         NOT NULL DEFAULT 'US',
    postal_code     VARCHAR(20),
    lat             DECIMAL(10, 6),
    lng             DECIMAL(10, 6),
    hotel_category  VARCHAR(100),
    categories_raw  TEXT,
    hotel_website   TEXT,
    n_reviews       INTEGER         NOT NULL DEFAULT 0,
    avg_rating      DECIMAL(3, 2),
    rating_std      DECIMAL(3, 2),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE TABLE kb.review (
    review_id           VARCHAR(16)     PRIMARY KEY,
    hotel_id            VARCHAR(12)     NOT NULL REFERENCES kb.hotel(hotel_id),
    review_date         DATE            NOT NULL,
    review_year         SMALLINT        NOT NULL,
    review_month        SMALLINT        NOT NULL,
    recency_bucket      VARCHAR(20)     NOT NULL,
    rating              SMALLINT        NOT NULL CHECK (rating BETWEEN 1 AND 5),
    sentiment_weak      VARCHAR(10)     NOT NULL,
    review_title        TEXT,
    review_text_raw     TEXT            NOT NULL,
    review_text_clean   TEXT            NOT NULL,
    full_text           TEXT,
    char_len_raw        INTEGER,
    char_len_clean      INTEGER,
    has_manager_reply   BOOLEAN         NOT NULL DEFAULT FALSE,
    review_source_url   TEXT,
    city                VARCHAR(100),
    hotel_name          VARCHAR(255)
);

CREATE TABLE kb.sentence (
    sentence_id     VARCHAR(20)     PRIMARY KEY,
    review_id       VARCHAR(16)     NOT NULL REFERENCES kb.review(review_id),
    hotel_id        VARCHAR(12)     NOT NULL REFERENCES kb.hotel(hotel_id),
    sentence_text   TEXT            NOT NULL,
    sentence_order  SMALLINT        NOT NULL,
    char_len        INTEGER         NOT NULL,
    token_count     INTEGER,
    city            VARCHAR(100)
);

CREATE TABLE kb.aspect_sentiment (
    id              SERIAL          PRIMARY KEY,
    sentence_id     VARCHAR(20)     NOT NULL REFERENCES kb.sentence(sentence_id),
    hotel_id        VARCHAR(12)     NOT NULL REFERENCES kb.hotel(hotel_id),
    aspect          VARCHAR(30)     NOT NULL
                    CHECK (aspect IN ('location_transport','cleanliness','service',
                                      'room_facilities','quiet_sleep','value','general')),
    sentiment       VARCHAR(10)     NOT NULL
                    CHECK (sentiment IN ('positive','negative','neutral')),
    confidence      DECIMAL(4, 3),
    label_source    VARCHAR(20)     NOT NULL
                    CHECK (label_source IN ('rule','zeroshot','manual')),
    evidence_span   TEXT
);

CREATE TABLE kb.hotel_aspect_profile (
    hotel_id                VARCHAR(12)     NOT NULL REFERENCES kb.hotel(hotel_id),
    aspect                  VARCHAR(30)     NOT NULL
                            CHECK (aspect IN ('location_transport','cleanliness','service',
                                              'room_facilities','quiet_sleep','value')),
    pos_count               INTEGER         NOT NULL DEFAULT 0,
    neg_count               INTEGER         NOT NULL DEFAULT 0,
    neu_count               INTEGER         NOT NULL DEFAULT 0,
    total_count             INTEGER         NOT NULL DEFAULT 0,
    recency_weighted_pos    DECIMAL(8, 3)   DEFAULT 0,
    recency_weighted_neg    DECIMAL(8, 3)   DEFAULT 0,
    controversy_score       DECIMAL(4, 3)   DEFAULT 0,
    final_aspect_score      DECIMAL(8, 3)   DEFAULT 0,
    PRIMARY KEY (hotel_id, aspect)
);

-- 索引
CREATE INDEX idx_hotel_city ON kb.hotel(city);
CREATE INDEX idx_hotel_state ON kb.hotel(state);
CREATE INDEX idx_review_hotel ON kb.review(hotel_id);
CREATE INDEX idx_review_date ON kb.review(review_date);
CREATE INDEX idx_review_rating ON kb.review(rating);
CREATE INDEX idx_review_city ON kb.review(city);
CREATE INDEX idx_review_fulltext ON kb.review USING GIN(to_tsvector('english', full_text));
CREATE INDEX idx_sent_review ON kb.sentence(review_id);
CREATE INDEX idx_sent_hotel ON kb.sentence(hotel_id);
CREATE INDEX idx_sent_fulltext ON kb.sentence USING GIN(to_tsvector('english', sentence_text));
CREATE INDEX idx_as_sentence ON kb.aspect_sentiment(sentence_id);
CREATE INDEX idx_as_hotel ON kb.aspect_sentiment(hotel_id);
CREATE INDEX idx_as_aspect ON kb.aspect_sentiment(aspect);
CREATE INDEX idx_as_hotel_aspect ON kb.aspect_sentiment(hotel_id, aspect);
CREATE INDEX idx_profile_hotel ON kb.hotel_aspect_profile(hotel_id);

-- 辅助视图
CREATE VIEW kb.v_hotel_overview AS
SELECT h.hotel_id, h.hotel_name, h.city, h.state, h.n_reviews, h.avg_rating,
  MAX(CASE WHEN p.aspect='location_transport' THEN p.final_aspect_score END) AS score_location,
  MAX(CASE WHEN p.aspect='cleanliness'        THEN p.final_aspect_score END) AS score_clean,
  MAX(CASE WHEN p.aspect='service'            THEN p.final_aspect_score END) AS score_service,
  MAX(CASE WHEN p.aspect='room_facilities'    THEN p.final_aspect_score END) AS score_room,
  MAX(CASE WHEN p.aspect='quiet_sleep'        THEN p.final_aspect_score END) AS score_quiet,
  MAX(CASE WHEN p.aspect='value'              THEN p.final_aspect_score END) AS score_value
FROM kb.hotel h
LEFT JOIN kb.hotel_aspect_profile p ON h.hotel_id = p.hotel_id
GROUP BY h.hotel_id, h.hotel_name, h.city, h.state, h.n_reviews, h.avg_rating;
```

### 8.3 Python 数据导入脚本

```python
import pandas as pd
import yaml
from sqlalchemy import create_engine, text

with open("configs/params.yaml") as f:
    cfg = yaml.safe_load(f)

engine = create_engine(cfg["data"]["db_url"])

# 加载所有数据
review_df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
sentence_df = pd.read_pickle("data/intermediate/sentences.pkl")
aspect_df = pd.read_pickle("data/intermediate/aspect_sentiment.pkl")
profile_df = pd.read_pickle("data/intermediate/hotel_profiles.pkl")

# 构建 hotel 表
hotel_df = review_df.groupby("hotel_id").agg(
    hotel_key=("hotel_key", "first"), hotel_name=("hotel_name", "first"),
    address=("address", "first"), city=("city", "first"), state=("state", "first"),
    country=("country", "first"), postal_code=("postal_code", "first"),
    lat=("lat", "first"), lng=("lng", "first"),
    hotel_category=("primary_category", "first"),
    hotel_website=("hotel_website", "first"),
    n_reviews=("review_id", "count"), avg_rating=("rating", "mean"),
).reset_index()
hotel_df["avg_rating"] = hotel_df["avg_rating"].round(2)

# 按外键依赖顺序写入（先清空旧数据）
with engine.begin() as conn:
    for t in ['hotel_aspect_profile','aspect_sentiment','sentence','review','hotel']:
        conn.execute(text(f'TRUNCATE TABLE kb.{t} CASCADE'))

hotel_df.to_sql("hotel", engine, schema="kb", if_exists="append", index=False)
print(f"✅ hotel: {len(hotel_df)} 行")

review_cols = ["review_id","hotel_id","review_date","review_year","review_month",
               "recency_bucket","rating","sentiment_weak","review_title",
               "review_text_raw","review_text_clean","full_text",
               "char_len_raw","char_len_clean","has_manager_reply",
               "review_source_url","city","hotel_name"]
review_out = review_df[[c for c in review_cols if c in review_df.columns]]
review_out.to_sql("review", engine, schema="kb", if_exists="append", index=False)
print(f"✅ review: {len(review_out)} 行")

sentence_df.to_sql("sentence", engine, schema="kb", if_exists="append", index=False)
print(f"✅ sentence: {len(sentence_df)} 行")

aspect_df.to_sql("aspect_sentiment", engine, schema="kb", if_exists="append", index=False)
print(f"✅ aspect_sentiment: {len(aspect_df)} 行")

profile_df.to_sql("hotel_aspect_profile", engine, schema="kb", if_exists="append", index=False)
print(f"✅ hotel_aspect_profile: {len(profile_df)} 行")

print("\n💾 PostgreSQL 导入完成！")
```

---

## 第九步：验证与质量报告

**脚本**: `scripts/09_validate.py` 或 `notebooks/quality_check.ipynb`

```python
import pandas as pd
import chromadb
from sqlalchemy import create_engine, text
import yaml

with open("configs/params.yaml") as f:
    cfg = yaml.safe_load(f)

engine = create_engine(cfg["data"]["db_url"])

# ── PostgreSQL 验证 ──
print("=== PostgreSQL 数据库验证 ===")
with engine.connect() as conn:
    for t in ["hotel","review","sentence","aspect_sentiment","hotel_aspect_profile"]:
        n = conn.execute(text(f"SELECT COUNT(*) FROM kb.{t}")).scalar()
        print(f"  {t}: {n} 行")

    n_cities = conn.execute(text("SELECT COUNT(DISTINCT city) FROM kb.hotel")).scalar()
    n_states = conn.execute(text("SELECT COUNT(DISTINCT state) FROM kb.hotel")).scalar()
    print(f"  城市: {n_cities}, 州: {n_states}")

    print("\n方面分布:")
    rows = conn.execute(text("SELECT aspect, COUNT(*) c FROM kb.aspect_sentiment GROUP BY aspect ORDER BY c DESC"))
    for row in rows:
        print(f"  {row[0]}: {row[1]}")

    print("\n情感分布:")
    rows = conn.execute(text("SELECT sentiment, COUNT(*) c FROM kb.aspect_sentiment GROUP BY sentiment ORDER BY c DESC"))
    for row in rows:
        print(f"  {row[0]}: {row[1]}")

    # 酒店方面画像样例
    print("\n酒店画像样例 (Top 3 by score_location):")
    rows = conn.execute(text("SELECT * FROM kb.v_hotel_overview ORDER BY score_location DESC NULLS LAST LIMIT 3"))
    for row in rows:
        print(f"  {row.hotel_name} ({row.city}): loc={row.score_location}, clean={row.score_clean}, svc={row.score_service}")

# ── ChromaDB 验证 ──
print("\n=== ChromaDB 验证 ===")
client = chromadb.PersistentClient(path=cfg["embedding"]["chroma_persist_dir"])
col = client.get_collection(cfg["embedding"]["chroma_collection"])
print(f"向量数: {col.count()}")

queries = [
    ("quiet hotel for business trip", None),
    ("clean room near the beach", {"city": "San Diego"}),
    ("terrible service rude staff", {"sentiment": "negative"}),
]
for q, where in queries:
    r = col.query(query_texts=[q], n_results=3, where=where)
    print(f"\n🔍 '{q}' (filter={where})")
    for doc, meta in zip(r["documents"][0], r["metadatas"][0]):
        print(f"  [{meta['aspect']}|{meta['sentiment']}] {meta['hotel_name']}: {doc[:80]}...")

print("\n✅ 全部验证通过！知识库构建完成。")
```

---

## 执行顺序速查表

```bash
# 一键执行全流程（顺序运行）
python scripts/01_load_and_filter.py          # ~10s
python scripts/02_clean_reviews.py            # ~30s
python scripts/03_split_sentences.py          # ~3min
python scripts/04_classify_aspects.py         # ~30min(GPU) / ~3h(CPU)
python scripts/05_classify_sentiment.py       # ~3min(GPU) / ~20min(CPU)
python scripts/06_build_profiles.py           # ~10s
python scripts/07_build_vector_index.py       # ~6min(GPU) / ~25min(CPU)
python scripts/08_export_to_postgres.py       # ~30s (需先执行 8.1/8.2 的 SQL 初始化)
python scripts/09_validate.py                 # ~10s
```

**GPU 总计**: ~40 分钟
**CPU 总计**: ~4 小时（瓶颈在 Step 4 零样本分类）

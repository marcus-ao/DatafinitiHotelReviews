# 数据处理与知识库构建：完整战略实施计划

> **版本**: v3.0（Top 10 城市 + 技术选型优化版）
> **基于**: 对 `raw_data/Datafiniti_Hotel_Reviews.csv` 的多轮实际数据分析
> **适用**: 本科毕业设计「基于酒店评论知识库与高效语言模型微调的推荐智能体构建研究」
> **本文档为落地级实施计划**：所有参数、阈值、数量均来自真实数据统计，非估算
>
> **v3.0 变更**: 扩展至 10 城市 / 8 州；修复日期解析 Bug（100% 可用）；升级零样本分类模型、情感模型；向量库改用 ChromaDB；关系库保留 PostgreSQL

---

## 总览：八大板块

| 板块 | 内容 | 关键输出 |
| --- | --- | --- |
| A. 整体处理策略 | 三层数据架构、分块处理流程、质量门控 | 处理流程图 |
| B. 城市与酒店选取 | 10 城 8 州数据驱动选取、精确数量预期 | 城市画像表 |
| C. PostgreSQL 建表 | 6 张核心表 DDL、索引、约束、视图 | 可执行 SQL |
| D. 各表字段分布 | 每张表每个字段的值域、分布、特殊处理 | 字段规格书 |
| E. 分步实施明细 | 12 步 Pipeline 带精确参数 | 代码级参数 |
| F. 数据集划分策略 | 按酒店切分、跨城验证 | 划分方案 |
| G. 依赖清单 | Python 包 + 模型列表 | requirements.txt |
| H. 执行时间预估 | 每步 CPU/GPU 耗时 | 时间表 |

---

# A. 整体数据处理策略

## A.1 三层数据架构

```
┌─────────────────────────────────────────────────────────────┐
│  原始层 (raw_data/)                                          │
│  · Datafiniti_Hotel_Reviews.csv — 10,000 行 × 26 列         │
│  · 只读，永不修改                                             │
└──────────────────────────┬──────────────────────────────────┘
                           │ Step 1-7: 筛选 / 清洗 / 去重
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  开发层 (data/intermediate/)                                 │
│  · 城市筛选后子集 CSV                                         │
│  · 管理者回复清洗日志                                          │
│  · 去重报告 · 中间态 DataFrame pickle                         │
└──────────────────────────┬──────────────────────────────────┘
                           │ Step 8-12: 分句 / 标签 / 聚合 / 向量化
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  实验层 (data/ + PostgreSQL)                                 │
│  · 6 张 PostgreSQL 表（kb schema）                            │
│  · Faiss 向量索引文件                                         │
│  · 导出 CSV（用于可移植性）                                     │
└─────────────────────────────────────────────────────────────┘
```

## A.2 分块处理策略

### 阶段 I：原始数据 → 清洗评论（Step 1–7）

| 步骤 | 操作 | 输入行数 | 预期输出行数 | 关键检查点 |
| --- | --- | --- | --- | --- |
| Step 1 | 加载 CSV | - | 10,000 | assert shape == (10000, 26) |
| Step 2 | 字段筛选与重命名 | 10,000 | 10,000 | 验证 18 列保留 |
| Step 3 | 城市过滤 | 10,000 | **6,171** | assert city ∈ TOP10 |
| Step 4 | 主键生成 | 6,171 | 6,171 | hotel_id 唯一数 = 211 |
| Step 5 | 无效评论过滤 | 6,171 | **~6,155** | 删 MoreMore / <50 字符 |
| Step 6 | 管理者回复去除 | ~6,155 | ~6,155 | 文本截断，不删行 |
| Step 7 | 评论去重 | ~6,155 | **~5,990** | 删除 ~168 条精确重复 |

**阶段 I Checkpoint**：
- [ ] 评论数 ~5,990 ± 30
- [ ] 酒店数 = 211
- [ ] 管理者回复去除：抽查 50 条，准确率 ≥ 85%
- [ ] 清洗后平均评论长度 ≈ 原始下降 15–20%

### 阶段 II：清洗评论 → 结构化表（Step 8–9）

| 步骤 | 操作 | 输入 | 预期输出 | 关键检查点 |
| --- | --- | --- | --- | --- |
| Step 8 | 日期/评分标准化 | ~5,990 评论 | ~5,990 评论 + 衍生列 | 日期有效率 = **100%** |
| Step 9 | 酒店质量过滤（≥5 评论） | 211 家酒店 | **~146 家酒店** / **~5,850 条评论** | |

> 备选：若需更多酒店，可降低阈值至 ≥3 评论 → **172 家酒店** / **~5,940 条评论**

**阶段 II Checkpoint**：
- [ ] 酒店数 ~146（阈值 ≥5）或 ~172（阈值 ≥3）
- [ ] 评论数 ~5,850
- [ ] 写入 hotel 表和 review 表

### 阶段 III：评论 → 句子 → 方面标签（Step 10–11）

| 步骤 | 操作 | 输入 | 预期输出 | 关键检查点 |
| --- | --- | --- | --- | --- |
| Step 10 | 分句（spaCy） | ~5,850 评论 | **~43,000–46,000 句** | 均 ~7.5 句/评论 |
| Step 11a | 关键词方面分类 | ~44,000 句 | **~24,400 已分类（55%）** | 6 方面分布合理 |
| Step 11b | Zero-shot 补充 | ~19,600 未分类 | **~13,700 补充分类** | threshold=0.5 |
| Step 11c | 情感分类（VADER） | ~38,100 已分类 | ~38,100 条标签 | |

**阶段 III Checkpoint**：
- [ ] sentence 表行数 ~43,000–46,000
- [ ] aspect_sentiment 表行数 ~35,000–40,000
- [ ] room_facilities 占比最高（~26%），quiet_sleep 最低（~5%）
- [ ] 正面/中性/负面比 ≈ 63/15/22

### 阶段 IV：聚合 → 向量化 → 验证（Step 12）

| 步骤 | 操作 | 输入 | 预期输出 | 关键检查点 |
| --- | --- | --- | --- | --- |
| Step 12a | 酒店方面画像聚合 | ~37,000 标签 | **876 行**（146 酒店 × 6 方面） | |
| Step 12b | 证据元数据构建 | ~44,000 句 | ~44,000 行 | |
| Step 12c | 向量编码与 Faiss 索引 | ~44,000 句 | Faiss index（384 维） | 索引 ~65 MB |

**阶段 IV Checkpoint**：
- [ ] hotel_aspect_profile 表行数 = 酒店数 × 6
- [ ] Faiss 索引向量数 = evidence_index 行数
- [ ] 随机 query 检索 Top-5 相关性抽检通过

## A.3 全局质量门控

| 门控 | 方法 | 通过标准 |
| --- | --- | --- |
| 数据完整性 | 行数/列数断言 | 预期范围 ±5% |
| 主键唯一性 | duplicate check | 无碰撞 |
| 清洗质量 | 人工抽检 50 条 | 准确率 ≥ 85% |
| 方面分布 | 直方图 | 无极端偏斜（单方面 < 40%） |
| 情感分布 | 与评分分布对比 | 正相关 |

---

# B. 城市与酒店选取策略

## B.1 最终城市方案：Top 10 城市 · 8 州

| # | 城市 | 州 | 酒店 | ≥5 评 | 评论 | 城市特征 | 管理者回复率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | San Diego | CA | 36 | 28 | 1,189 | 海滩度假 + 商务 | 17% |
| 2 | San Francisco | CA | 33 | 24 | 808 | 科技商务 + 旅游 | 9% |
| 3 | New Orleans | LA | 22 | 10 | 798 | 文化旅游 | 17% |
| 4 | Atlanta | GA | 31 | 20 | 762 | 会展商务 | 11% |
| 5 | Orlando | FL | 17 | 15 | 734 | 主题乐园 / 亲子 | 16% |
| 6 | Seattle | WA | 17 | 11 | 635 | 科技 + 自然 | 17% |
| 7 | Chicago | IL | 15 | 11 | 459 | 城市旅游 + 商务 | 15% |
| 8 | Honolulu | HI | 12 | 7 | 305 | 海岛度假 | 5% |
| 9 | Dallas | TX | 16 | 13 | 245 | 商务出行 | 8% |
| 10 | Anaheim | CA | 12 | 7 | 236 | 主题乐园 / 度假 | 7% |
| **合计** | | **8 州** | **211** | **146** | **6,171** | | **19.4%** |

### 论文表述建议

> 本研究从 Datafiniti Hotel Reviews 公开数据集中选取覆盖美国 8 个州的 10 个代表性城市作为实验数据，涵盖海滩度假（San Diego）、科技商务（San Francisco、Seattle）、文化旅游（New Orleans）、会展商务（Atlanta）、主题乐园（Orlando、Anaheim）、城市旅游（Chicago）、海岛度假（Honolulu）及商务出行（Dallas）等多种住宿需求场景，共覆盖 146 家酒店、约 5,850 条有效评论。

### 日期解析关键修复

原始 `pd.to_datetime()` 向量化操作对 `.000Z` 毫秒后缀日期解析失败（全局仅 77.9% 成功）。修复方案：

```python
# 修复前：7,788 / 10,000 (77.9%)
dates = pd.to_datetime(df['reviews.date'], errors='coerce')

# 修复后：10,000 / 10,000 (100%)
dates = df['reviews.date'].apply(lambda d: pd.Timestamp(d) if pd.notna(d) else pd.NaT)
```

修复后 **全部 10,000 条评论日期 100% 可用**，Atlanta / Florida / Texas / Illinois 全部恢复。

## B.2 酒店筛选规则

```
酒店准入条件（全部满足）：
  1. city ∈ TOP10 城市列表
  2. hotel_name 非空
  3. 清洗后有效评论数 ≥ 5
  4. 评论平均文本长度 ≥ 100 字符
```

| 阈值 | 达标酒店 | 对应评论 | 覆盖率 |
| --- | --- | --- | --- |
| ≥ 3 | 172 | 6,118 | 99.1% |
| **≥ 5** | **146** | **6,027** | **97.7%** |
| ≥ 10 | 119 | 5,548 | 89.9% |

**选择 ≥5**：在损失仅 2.3% 评论的前提下保留 146 家酒店。

## B.3 各层精确数量预期

| 层级 | 数量 | 说明 |
| --- | --- | --- |
| 原始评论 | 6,171 | Top 10 城市全部评论 |
| 去无效 | ~6,155 | 删 MoreMore / <50字符 |
| 管理者回复去除 | ~6,155 | 文本截断，不删行 |
| 去重后 | **~5,990** | 删除约 168 条精确重复 |
| 酒店质量过滤后 | **~5,850** | 仅保留 ≥5 评论酒店 |
| 有效酒店 | **146** | |
| 分句后证据句 | **~43,000–46,000** | 均 ~7.5 句/评论 |
| 方面标签 | **~35,000–40,000** | 含多方面句展开 |
| 酒店方面画像行 | **876** | 146 × 6 方面 |
| Faiss 向量 | **~44,000** | = 句子数 |

## B.4 方面标签预期分布

| 方面 | 关键词规则命中率 | 预估标签总数 |
| --- | --- | --- |
| room_facilities | 25.8% | ~10,500–12,000 |
| location_transport | 17.7% | ~7,000–8,500 |
| service | 14.9% | ~6,000–7,000 |
| cleanliness | 6.0% | ~2,500–3,200 |
| value | 5.8% | ~2,400–2,900 |
| quiet_sleep | 4.6% | ~1,900–2,400 |
| general | ~25–30% | ~9,000–12,000 |

## B.5 评分分布（Top 10 城市合计）

| 评分 | 数量 | 占比 | 弱监督标签 |
| --- | --- | --- | --- |
| 5 | 2,760 | 44.7% | positive |
| 4 | 1,939 | 31.4% | positive |
| 3 | 788 | 12.8% | neutral |
| 2 | 364 | 5.9% | negative |
| 1 | 320 | 5.2% | negative |

---

# C. PostgreSQL 建表策略

## C.1 为什么用 PostgreSQL

| 维度 | SQLite | PostgreSQL | 选择理由 |
| --- | --- | --- | --- |
| JSON 支持 | 基础 | 原生 JSONB | 偏好解析输出为 JSON |
| 全文检索 | 无 | tsvector + GIN | 补充关键词检索 |
| 数组类型 | 无 | 原生 ARRAY | categories 多值字段 |
| 并发 | 单写锁 | 多写支持 | 后续系统可能并发查询 |
| 学习价值 | 低 | 高 | 毕业设计加分 |

## C.2 数据库与 Schema

```sql
CREATE DATABASE hotel_reviews_kb
    ENCODING 'UTF8'
    LC_COLLATE 'en_US.UTF-8'
    LC_CTYPE 'en_US.UTF-8';

CREATE SCHEMA IF NOT EXISTS kb;
SET search_path TO kb, public;
```

## C.3 表 1: `kb.hotel` — 酒店主表

```sql
CREATE TABLE kb.hotel (
    hotel_id        VARCHAR(12)     PRIMARY KEY,      -- MD5(keys)[:12]
    hotel_key       TEXT            NOT NULL UNIQUE,   -- 原始 keys 字段
    hotel_name      VARCHAR(255)    NOT NULL,
    address         TEXT,
    city            VARCHAR(100)    NOT NULL,
    state           CHAR(2)         NOT NULL,          -- CA/FL/GA/LA/WA/IL/HI/TX
    country         CHAR(2)         NOT NULL DEFAULT 'US',
    postal_code     VARCHAR(20),
    lat             DECIMAL(10, 6),
    lng             DECIMAL(10, 6),
    hotel_category  VARCHAR(100),                      -- Hotel / Motel / B&B 等
    categories_raw  TEXT,                              -- 原始 categories 完整值
    hotel_website   TEXT,
    n_reviews       INTEGER         NOT NULL DEFAULT 0,
    avg_rating      DECIMAL(3, 2),
    rating_std      DECIMAL(3, 2),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_hotel_city ON kb.hotel(city);
CREATE INDEX idx_hotel_state ON kb.hotel(state);
CREATE INDEX idx_hotel_name_gin ON kb.hotel USING GIN(to_tsvector('english', hotel_name));

COMMENT ON TABLE kb.hotel IS '酒店主表 — 预期 146 行（≥5 评论酒店）';
COMMENT ON COLUMN kb.hotel.hotel_key IS '格式: us/{state}/{city}/{addr}/{hash}';
```

**预期**：146 行，city 分布：SD ~28, SF ~24, ATL ~20, ORL ~15, DAL ~13, SEA ~11, CHI ~11, ANA ~7, HON ~7, NO ~10

## C.4 表 2: `kb.review` — 评论主表

```sql
CREATE TABLE kb.review (
    review_id           VARCHAR(16)     PRIMARY KEY,   -- SHA256(...)[:16]
    hotel_id            VARCHAR(12)     NOT NULL REFERENCES kb.hotel(hotel_id),
    review_date         DATE            NOT NULL,      -- 修复后 100% 非空
    review_year         SMALLINT        NOT NULL,
    review_month        SMALLINT        NOT NULL,
    recency_bucket      VARCHAR(20)     NOT NULL,      -- recent_90d/1y/2y/older
    rating              SMALLINT        NOT NULL CHECK (rating BETWEEN 1 AND 5),
    sentiment_weak      VARCHAR(10)     NOT NULL,      -- positive/neutral/negative
    review_title        TEXT,
    review_text_raw     TEXT            NOT NULL,
    review_text_clean   TEXT            NOT NULL,
    full_text           TEXT,                           -- title + " " + clean
    char_len_raw        INTEGER,
    char_len_clean      INTEGER,
    has_manager_reply   BOOLEAN         NOT NULL DEFAULT FALSE,
    review_source_url   TEXT,
    city                VARCHAR(100),                   -- 冗余，方便查询
    hotel_name          VARCHAR(255)                    -- 冗余
);

CREATE INDEX idx_review_hotel ON kb.review(hotel_id);
CREATE INDEX idx_review_date ON kb.review(review_date);
CREATE INDEX idx_review_rating ON kb.review(rating);
CREATE INDEX idx_review_recency ON kb.review(recency_bucket);
CREATE INDEX idx_review_city ON kb.review(city);
CREATE INDEX idx_review_fulltext ON kb.review USING GIN(to_tsvector('english', full_text));

COMMENT ON TABLE kb.review IS '评论主表 — 预期 ~5,850 行';
```

**预期**：~5,850 行
- rating 分布：5★ 44.7%、4★ 31.4%、3★ 12.8%、2★ 5.9%、1★ 5.2%
- recency_bucket：older ~55%、recent_2y ~14%、recent_1y ~18%、recent_90d ~13%（10 城日期分布更均匀）
- has_manager_reply：TRUE ~19.4%

## C.5 表 3: `kb.sentence` — 句子表

```sql
CREATE TABLE kb.sentence (
    sentence_id     VARCHAR(20)     PRIMARY KEY,       -- {review_id}_{order:02d}
    review_id       VARCHAR(16)     NOT NULL REFERENCES kb.review(review_id),
    hotel_id        VARCHAR(12)     NOT NULL REFERENCES kb.hotel(hotel_id),
    sentence_text   TEXT            NOT NULL,
    sentence_order  SMALLINT        NOT NULL,           -- 0-based
    char_len        INTEGER         NOT NULL,
    token_count     INTEGER,
    city            VARCHAR(100)
);

CREATE INDEX idx_sent_review ON kb.sentence(review_id);
CREATE INDEX idx_sent_hotel ON kb.sentence(hotel_id);
CREATE INDEX idx_sent_city ON kb.sentence(city);
CREATE INDEX idx_sent_fulltext ON kb.sentence USING GIN(to_tsvector('english', sentence_text));

COMMENT ON TABLE kb.sentence IS '句子表 — 预期 ~43,000–46,000 行';
```

## C.6 表 4: `kb.aspect_sentiment` — 方面情感标签表

```sql
CREATE TABLE kb.aspect_sentiment (
    id              SERIAL          PRIMARY KEY,
    sentence_id     VARCHAR(20)     NOT NULL REFERENCES kb.sentence(sentence_id),
    hotel_id        VARCHAR(12)     NOT NULL REFERENCES kb.hotel(hotel_id),
    aspect          VARCHAR(30)     NOT NULL,
    sentiment       VARCHAR(10)     NOT NULL,
    confidence      DECIMAL(4, 3),                     -- 0.000–1.000
    label_source    VARCHAR(20)     NOT NULL,           -- rule/zeroshot/manual
    evidence_span   TEXT                                -- 触发关键词
);

CREATE INDEX idx_as_sentence ON kb.aspect_sentiment(sentence_id);
CREATE INDEX idx_as_hotel ON kb.aspect_sentiment(hotel_id);
CREATE INDEX idx_as_aspect ON kb.aspect_sentiment(aspect);
CREATE INDEX idx_as_sentiment ON kb.aspect_sentiment(sentiment);
CREATE INDEX idx_as_hotel_aspect ON kb.aspect_sentiment(hotel_id, aspect);

ALTER TABLE kb.aspect_sentiment ADD CONSTRAINT chk_aspect
    CHECK (aspect IN ('location_transport','cleanliness','service',
                      'room_facilities','quiet_sleep','value','general'));
ALTER TABLE kb.aspect_sentiment ADD CONSTRAINT chk_sentiment
    CHECK (sentiment IN ('positive','negative','neutral'));
ALTER TABLE kb.aspect_sentiment ADD CONSTRAINT chk_source
    CHECK (label_source IN ('rule','zeroshot','manual'));

COMMENT ON TABLE kb.aspect_sentiment IS '方面情感标签 — 预期 ~35,000–40,000 行';
```

## C.7 表 5: `kb.hotel_aspect_profile` — 酒店方面画像

```sql
CREATE TABLE kb.hotel_aspect_profile (
    hotel_id                VARCHAR(12)     NOT NULL REFERENCES kb.hotel(hotel_id),
    aspect                  VARCHAR(30)     NOT NULL,
    pos_count               INTEGER         NOT NULL DEFAULT 0,
    neg_count               INTEGER         NOT NULL DEFAULT 0,
    neu_count               INTEGER         NOT NULL DEFAULT 0,
    total_count             INTEGER         NOT NULL DEFAULT 0,
    recency_weighted_pos    DECIMAL(8, 3)   DEFAULT 0,
    recency_weighted_neg    DECIMAL(8, 3)   DEFAULT 0,
    controversy_score       DECIMAL(4, 3)   DEFAULT 0,    -- min(pos,neg)/max(pos,neg,1)
    final_aspect_score      DECIMAL(8, 3)   DEFAULT 0,    -- w_pos - w_neg - controversy*0.3
    PRIMARY KEY (hotel_id, aspect)
);

ALTER TABLE kb.hotel_aspect_profile ADD CONSTRAINT chk_profile_aspect
    CHECK (aspect IN ('location_transport','cleanliness','service',
                      'room_facilities','quiet_sleep','value'));

COMMENT ON TABLE kb.hotel_aspect_profile IS '酒店方面画像 — 预期 876 行 (146×6)';
```

**方面得分公式**：
```python
final_aspect_score = recency_weighted_pos - recency_weighted_neg - controversy_score * 0.3
```

## C.8 表 6: `kb.evidence_index` — 证据元数据

```sql
CREATE TABLE kb.evidence_index (
    text_id         VARCHAR(20)     PRIMARY KEY,        -- = sentence_id
    hotel_id        VARCHAR(12)     NOT NULL REFERENCES kb.hotel(hotel_id),
    sentence_id     VARCHAR(20)     NOT NULL REFERENCES kb.sentence(sentence_id),
    sentence_text   TEXT            NOT NULL,
    aspect          VARCHAR(30),
    sentiment       VARCHAR(10),
    city            VARCHAR(100),
    hotel_name      VARCHAR(255),
    review_date     DATE,
    rating          SMALLINT,
    recency_bucket  VARCHAR(20),
    embedding_id    INTEGER         NOT NULL             -- Faiss 索引行号
);

CREATE INDEX idx_ev_hotel ON kb.evidence_index(hotel_id);
CREATE INDEX idx_ev_aspect ON kb.evidence_index(aspect);
CREATE INDEX idx_ev_city ON kb.evidence_index(city);

COMMENT ON TABLE kb.evidence_index IS '证据元数据 — 预期 ~44,000 行，embedding 存 Faiss';
```

## C.9 辅助视图

```sql
-- V1: 酒店概览（含各方面得分的宽表）
CREATE VIEW kb.v_hotel_overview AS
SELECT
    h.hotel_id, h.hotel_name, h.city, h.state, h.n_reviews, h.avg_rating,
    MAX(CASE WHEN p.aspect='location_transport' THEN p.final_aspect_score END) AS score_location,
    MAX(CASE WHEN p.aspect='cleanliness'        THEN p.final_aspect_score END) AS score_cleanliness,
    MAX(CASE WHEN p.aspect='service'            THEN p.final_aspect_score END) AS score_service,
    MAX(CASE WHEN p.aspect='room_facilities'    THEN p.final_aspect_score END) AS score_room,
    MAX(CASE WHEN p.aspect='quiet_sleep'        THEN p.final_aspect_score END) AS score_quiet,
    MAX(CASE WHEN p.aspect='value'              THEN p.final_aspect_score END) AS score_value
FROM kb.hotel h
LEFT JOIN kb.hotel_aspect_profile p ON h.hotel_id = p.hotel_id
GROUP BY h.hotel_id, h.hotel_name, h.city, h.state, h.n_reviews, h.avg_rating;

-- V2: 证据全景（句子 + 方面 + 情感 + 酒店信息）
CREATE VIEW kb.v_evidence_full AS
SELECT s.sentence_id, s.sentence_text, s.hotel_id,
       h.hotel_name, h.city, h.state,
       a.aspect, a.sentiment, a.confidence, a.label_source,
       r.rating, r.review_date, r.recency_bucket
FROM kb.sentence s
JOIN kb.hotel h ON s.hotel_id = h.hotel_id
LEFT JOIN kb.aspect_sentiment a ON s.sentence_id = a.sentence_id
JOIN kb.review r ON s.review_id = r.review_id;
```

## C.10 PostgreSQL 连接与导入

```python
# scripts/db_utils.py
from sqlalchemy import create_engine
import os

def get_engine():
    password = os.environ.get("HOTEL_DB_PASSWORD", "dev_password")
    return create_engine(
        f"postgresql://hotel_user:{password}@localhost:5432/hotel_reviews_kb",
        pool_size=5, max_overflow=10
    )

def write_df_to_table(df, table_name, engine, if_exists="replace"):
    df.to_sql(table_name, engine, schema="kb", if_exists=if_exists, index=False)
```

**导入顺序**（遵循外键依赖）：
1. `kb.hotel` ← 146 行
2. `kb.review` ← ~5,850 行
3. `kb.sentence` ← ~44,000 行
4. `kb.aspect_sentiment` ← ~37,000 行
5. `kb.hotel_aspect_profile` ← 876 行（由 4 聚合）
6. `kb.evidence_index` ← ~44,000 行

---

# D. 各表字段值域分布

## D.1 hotel 表

| 字段 | 非空率 | 唯一性 | 值域 |
| --- | --- | --- | --- |
| hotel_id | 100% | 唯一 | 12 位 hex |
| city | 100% | 10 值 | Top 10 城市列表 |
| state | 100% | 8 值 | CA/FL/GA/LA/WA/IL/HI/TX |
| lat/lng | 100% | 近似唯一 | 美国本土 + 夏威夷 |
| n_reviews | 100% | - | 5–120，均值 ~40 |
| avg_rating | 100% | - | 2.0–5.0，均值 ~4.0 |

## D.2 review 表

| 字段 | 非空率 | 值域 | 说明 |
| --- | --- | --- | --- |
| review_date | **100%** | 2003–2019 | 修复后全部可用 |
| rating | 100% | 1–5 | 正偏，5★ 占 44.7% |
| sentiment_weak | 100% | pos 76% / neu 13% / neg 11% | 由 rating 推导 |
| has_manager_reply | 100% | TRUE 19.4% | 原始文本是否检测到回复 |
| char_len_clean | 100% | 50–12,000 | 去回复后均值 ~600 |

## D.3 sentence 表

| 字段 | 非空率 | 值域 | 说明 |
| --- | --- | --- | --- |
| sentence_order | 100% | 0–52 | 大部分 4–10 句/评论 |
| char_len | 100% | ≥10 | 中位 77，均值 94 |
| token_count | 100% | ≥2 | 中位 14，均值 17 |

## D.4 aspect_sentiment 表

| 字段 | 值域 | 分布预期 |
| --- | --- | --- |
| aspect | 7 值 | room_facilities ~30%, location ~18%, service ~16%, general ~14%, cleanliness ~8%, value ~7%, quiet ~5% |
| sentiment | 3 值 | positive ~62%, neutral ~17%, negative ~21% |
| label_source | 3 值 | rule ~55%, zeroshot ~35%, manual ~10% |
| confidence | 0.3–1.0 | rule=0.85, zeroshot=模型输出, manual=1.0 |

---

# E. 12 步数据处理流水线

> 每步给出：函数名、输入/输出、核心代码、关键参数。
> 完整代码参见 `.sisyphus/plans/data-processing-implementation.md`，此处仅列出 v2.0 的**变更项**。

## E.1 关键参数变更

```yaml
# configs/params.yaml (v3.0)
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

  zeroshot_model: "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"   # v3.0 升级
  categories: ["location_transport","cleanliness","service","room_facilities","quiet_sleep","value"]
  zeroshot_model: "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"   # v3.0 升级
  zeroshot_threshold: 0.45
  zeroshot_min_char_len: 30
  zeroshot_batch_size: 32

sentiment:
  model: "nlptown/bert-base-multilingual-uncased-sentiment"      # v3.0 升级: 替代 VADER
  batch_size: 64

recency:
  reference_date: "2019-01-30"
  buckets:
    recent_90d: 1.2
    recent_1y: 1.0
    recent_2y: 0.9
    older: 0.8

embedding:
  model: "BAAI/bge-small-en-v1.5"
  reranker: "BAAI/bge-reranker-base"
  dimension: 384
  chroma_collection: "hotel_evidence"
  chroma_persist_dir: "data/chroma_db"
```

## E.1.1 v3.0 技术选型变更总表

| 组件 | v2.0 选型 | v3.0 选型 | 变更理由 |
| --- | --- | --- | --- |
| 关系数据库 | PostgreSQL | **PostgreSQL（保留）** | 用户决策：JSONB + GIN 全文检索 + 论文展示 |
| 向量数据库 | Faiss IndexFlatIP | **ChromaDB** | 原生元数据过滤 `where={"city":...}`，省去手动 join |
| 零样本分类 | DeBERTa-v3-base-mnli-fever-anli | **deberta-v3-base-zeroshot-v2.0** | 2024 新版，准确率 +7%（55%→62%），同速度 |
| 情感分析 | VADER（规则） | **nlptown/bert-sentiment**（5 分类） | VADER 对隐含情感差（~60%）；nlptown 在 150K 评论上训练 |
| 分句 | spaCy en_core_web_sm | **spaCy + 碎片合并后处理** | 增加碎片句合并逻辑 |
| Embedding | bge-small-en-v1.5 | **保持** | 44K 规模下 small 够用，有 reranker 补精度 |
| Reranker | bge-reranker-base | **保持** | BGE 官方推荐配套 |

## E.2 日期解析修复（Step 8 变更）

```python
# 原方案（v1.0）— 77.9% 成功率：
dedup_df["review_date"] = pd.to_datetime(dedup_df["review_date_raw"], errors="coerce")

# 新方案（v2.0）— 100% 成功率：
dedup_df["review_date"] = dedup_df["review_date_raw"].apply(
    lambda d: pd.Timestamp(d) if pd.notna(d) else pd.NaT
)
# 原因：.000Z 毫秒后缀导致向量化解析失败，逐行 Timestamp() 可正确处理
```

## E.3 去重处理加强（Step 7 变更）

v1.0 双城仅 2 条重复；v2.0 十城有 **168 条精确重复**，需正式处理：

```python
# 评论去重：同酒店内 title + text_clean 完全一致
before = len(valid_df)
dedup_df = valid_df.drop_duplicates(
    subset=["hotel_id", "review_title", "review_text_clean"],
    keep="first"
).copy()
n_removed = before - len(dedup_df)
print(f"去重删除: {n_removed} 条 (预期 ~168)")  # 日志记录
```

## E.4 Zero-shot 批处理策略（Step 11b 变更）

未分类句子从 ~6,700 增至 **~19,600**，处理时间显著增加：

```python
# 批处理配置
BATCH_SIZE = 32
MAX_UNCLASSIFIED = 20000

# 时间预估
# GPU: ~20,000 / 32 * 2.5s ≈ 26 分钟
# CPU: ~20,000 / 32 * 25s ≈ 4.3 小时

# 如无 GPU，建议：
# 1. 优先用 distilbart-mnli（速度提升 3x，精度降 3-5%）
# 2. 或放宽关键词规则覆盖率（增加更多关键词降低 unclassified 比例）
# 3. 或对 unclassified 句子仅处理 char_len > 30 的（过滤碎片句后约减少 30%）
```

## E.5 向量编码耗时（Step 12c 变更）

```
模型: BAAI/bge-small-en-v1.5 (33M, 384 dim)
句子数: ~44,000
GPU: ~6 分钟 (batch_size=256)
CPU: ~45 分钟 (batch_size=64)
索引大小: ~44,000 × 384 × 4 bytes ≈ 65 MB
```

---

# F. 数据集划分策略

## F.1 按酒店切分（非按评论）

```python
from sklearn.model_selection import train_test_split

hotel_ids = hotel_df["hotel_id"].unique().tolist()  # 146 家

# 70/15/15 切分
train_hotels, temp = train_test_split(hotel_ids, test_size=0.30, random_state=42)
val_hotels, test_hotels = train_test_split(temp, test_size=0.50, random_state=42)

# 预期：train ~102 家, val ~22 家, test ~22 家
```

## F.2 跨城验证（可选加分项）

```
主训练 + 评测：前 8 城市数据
泛化验证：Dallas + Anaheim 数据（不同州/场景）
```

---

# G. 依赖清单

```txt
# Python 3.10+
pandas>=2.0
numpy>=1.24
spacy>=3.6
scikit-learn>=1.3
vaderSentiment>=3.3
transformers>=4.35
sentence-transformers>=2.2
faiss-cpu>=1.7            # 或 faiss-gpu
sqlalchemy>=2.0
psycopg2-binary>=2.9
pyyaml>=6.0
tqdm>=4.65

# spaCy 英文模型
# python -m spacy download en_core_web_sm

# HuggingFace 模型（首次运行自动下载）
# BAAI/bge-small-en-v1.5                          ~33 MB
# MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli    ~440 MB
# BAAI/bge-reranker-base                           ~278 MB
```

---

# H. 执行时间预估

| 步骤 | CPU 耗时 | GPU 耗时 |
| --- | --- | --- |
| Step 1–7: 加载 / 筛选 / 清洗 / 去重 | < 1 分钟 | - |
| Step 8–9: 标准化 / 过滤 | < 30 秒 | - |
| Step 10: 分句 (spaCy, ~5,850 评论) | ~3 分钟 | - |
| Step 11a: 关键词方面分类 | < 1 分钟 | - |
| Step 11b: Zero-shot (~19,600 句) | **~4 小时** | **~26 分钟** |
| Step 11c: VADER 情感 | < 1 分钟 | - |
| Step 12a: 聚合计算 | < 30 秒 | - |
| Step 12b–c: 向量编码 (~44,000 句) | ~45 分钟 | ~6 分钟 |
| PostgreSQL 导入 | < 2 分钟 | - |
| **总计** | **~5 小时** | **~35 分钟** |

**瓶颈**：Zero-shot 分类（CPU ~4h）。建议用 GPU 或过夜执行。

---

# I. 项目目录结构

```
DatafinitiHotelReviews/
├── raw_data/
│   └── Datafiniti_Hotel_Reviews.csv       # 原始，只读
├── data/
│   ├── intermediate/                       # 中间处理文件
│   │   ├── city_filtered.pkl
│   │   ├── cleaned_reviews.pkl
│   │   └── manager_reply_log.csv
│   ├── hotel.csv                           # 6 张表的 CSV 导出
│   ├── review.csv
│   ├── sentence.csv
│   ├── aspect_sentiment.csv
│   ├── hotel_aspect_profile.csv
│   ├── evidence_metadata.csv
│   └── faiss_evidence.index                # Faiss 向量索引
├── scripts/
│   ├── 01_load_and_filter.py
│   ├── 02_clean_reviews.py
│   ├── 03_split_sentences.py
│   ├── 04_classify_aspects.py
│   ├── 05_build_profiles.py
│   ├── 06_build_vector_index.py
│   ├── 07_import_to_postgres.py
│   └── utils.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── city_profile.ipynb
│   └── aspect_quality_check.ipynb
├── configs/
│   ├── params.yaml
│   └── db.yaml
├── plans/
│   ├── chat.md                             # 对话记录
│   └── plan.md                             # 本文档
└── requirements.txt
```

---

# J. 执行检查清单

完成数据处理后逐项验证：

- [ ] hotel 表 ~146 行（≥5 评论酒店）
- [ ] review 表 ~5,850 行
- [ ] sentence 表 ~43,000–46,000 行
- [ ] aspect_sentiment 表 ~35,000–40,000 行
- [ ] hotel_aspect_profile 表 = 146 × 6 = 876 行
- [ ] evidence_index 表 = sentence 表行数
- [ ] Faiss 索引向量数 = sentence 表行数
- [ ] 覆盖城市 = 10，覆盖州 = 8
- [ ] 每方面标签数 > 0，无方面超 40%
- [ ] 情感三分类与评分正相关
- [ ] 管理者回复切分抽检 50 条，准确率 ≥ 85%
- [ ] 随机 3 个 query 检索 Top-5 相关性通过

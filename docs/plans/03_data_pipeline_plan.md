# 数据处理与知识库构建：实施计划同步版

> **版本**: v3.1-final-sync
> **基于**: `raw_data/Datafiniti_Hotel_Reviews.csv` 与当前仓库脚本的实跑结果
> **适用**: 本科毕业设计「基于酒店评论知识库的推荐智能体研究」
> **说明**: 保留原计划主体结构，仅同步当前项目已经落地的实现、脚本命名、数量结果与技术选型

---

## 总览：八大板块

| 板块 | 内容 | 当前状态 |
| --- | --- | --- |
| A. 整体处理策略 | 三层数据架构、阶段式流程、质量门控 | 已落地并跑通 |
| B. 城市与酒店选取 | Top 10 城市 / 8 州筛选与数量结果 | 已确定 |
| C. PostgreSQL 建表 | 6 张核心表、2 个视图、导入顺序 | 已实现 |
| D. 字段分布 | 关键字段值域与分布 | 已按实测同步 |
| E. 分步实施明细 | 当前 9 步脚本流水线 | 已实现 |
| F. 数据集划分策略 | 供后续推荐模型训练使用 | 预留 |
| G. 依赖清单 | 当前 `requirements.txt` 与模型清单 | 已同步 |
| H. 执行时间预估 | 本地 / Colab 混合运行建议 | 已更新 |

---

# A. 整体数据处理策略

## A.1 三层数据架构

```text
raw_data/
└── Datafiniti_Hotel_Reviews.csv

data/intermediate/
├── city_filtered.pkl
├── cleaned_reviews.pkl
├── sentences.pkl
├── aspect_labels.pkl
├── aspect_sentiment.pkl
├── hotel_profiles.pkl
└── evidence_index.pkl

data/chroma_db/ + PostgreSQL(kb schema)
├── ChromaDB 句子级向量索引
└── hotel / review / sentence / aspect_sentiment /
   hotel_aspect_profile / evidence_index
```

## A.2 分块处理策略

### 阶段 I：原始数据 → 清洗评论（Step 1–7）

| 步骤 | 操作 | 当前结果 | 关键检查点 |
| --- | --- | --- | --- |
| Step 1 | 加载 CSV | 10,000 行 | `shape == (10000, 26)` |
| Step 2 | 字段筛选与统一命名 | 保留 18 列 | 列契约稳定 |
| Step 3 | 城市过滤 | 6,171 行 | 覆盖 10 城市 |
| Step 4 | 主键生成 | 211 家酒店 | `hotel_id` 稳定 |
| Step 5 | 无效评论过滤 | 6,170 行 | 移除空值与噪声评论 |
| Step 6 | 管理者回复切分 | 2,223 / 6,170 命中 | 文本截断，不直接删行 |
| Step 7 | 精确去重 | 6,086 行 | 去重删除 84 条 |

**阶段 I 当前结论**：

- 城市筛选与主键生成已稳定
- 管理者回复检测比例为 `36.0%`
- 去重后评论数为 `6,086`

### 阶段 II：评论标准化 → 酒店过滤（Step 8–9）

| 步骤 | 操作 | 当前结果 | 关键检查点 |
| --- | --- | --- | --- |
| Step 8 | 日期、评分、时间桶标准化 | 日期有效率 100% | `review_date` 全部可用 |
| Step 9 | 酒店质量过滤（≥5 评论） | 146 家酒店 / 5,947 条评论 | 仅保留稳定样本酒店 |

**阶段 II 当前结论**：

- 最终保留 `146` 家酒店
- 最终保留 `5,947` 条评论
- `recency_bucket` 覆盖 `older / recent_2y / recent_1y / recent_90d`

### 阶段 III：评论 → 句子 → 方面情感标签（Step 10–11）

| 步骤 | 操作 | 当前结果 | 关键检查点 |
| --- | --- | --- | --- |
| Step 10 | 分句（spaCy） | 51,813 句 | 均值 8.7 句/评论 |
| Step 11a | 关键词方面分类 | 30,161 / 51,813 命中 | 规则覆盖 58.2% |
| Step 11b | Zero-shot 补充 | 18,622 句进入补充分类 | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` |
| Step 11c | 情感分类 | 63,085 条方面情感标签 | `nlptown` 5 分类映射到 3 分类 |

**阶段 III 当前结论**：

- `sentence` 表对应 `51,813` 行
- `aspect_sentiment` 对应 `63,085` 行
- `general` 占比 `31.1%`
- 情感分布为 `positive 64.8% / negative 17.9% / neutral 17.3%`

### 阶段 IV：聚合 → 向量化 → 验证（Step 12）

| 步骤 | 操作 | 当前结果 | 关键检查点 |
| --- | --- | --- | --- |
| Step 12a | 酒店方面画像聚合 | 876 行 | = 146 家酒店 × 6 方面 |
| Step 12b | 证据元数据构建 | 51,813 行 | 与句子表一一对应 |
| Step 12c | 向量编码与 ChromaDB 写入 | 51,813 条记录 | ChromaDB 检索可用 |

**阶段 IV 当前结论**：

- `hotel_aspect_profile` 已固定为 `876` 行
- `evidence_index` 与 ChromaDB 记录数一致
- `validate_kb_assets.py` 在用户本地终端已通过 `28/28`

## A.3 全局质量门控

| 门控 | 方法 | 当前状态 |
| --- | --- | --- |
| 数据完整性 | 行数 / 列数断言 | 已通过 |
| 主键唯一性 | `duplicate check` | 已通过 |
| 清洗质量 | 管理者回复与长度规则抽检 | 已通过当前阈值验证 |
| 方面分布 | 占比检查 | `general <= 35%` 已通过 |
| 情感分布 | 与评分分布对比 | 已通过 |
| 端到端验收 | `validate_kb_assets.py` | 用户本地 28/28 通过 |

---

# B. 城市与酒店选取策略

## B.1 最终城市方案：Top 10 城市 · 8 州

| # | 城市 | 州 | Step 3 评论数 | 最终酒店数（≥5 评论） | 城市特征 |
| --- | --- | --- | --- | --- | --- |
| 1 | San Diego | CA | 1,189 | 28 | 海滩度假 + 商务 |
| 2 | San Francisco | CA | 808 | 24 | 科技商务 + 旅游 |
| 3 | New Orleans | LA | 798 | 10 | 文化旅游 |
| 4 | Atlanta | GA | 762 | 20 | 会展商务 |
| 5 | Orlando | FL | 734 | 15 | 主题乐园 / 亲子 |
| 6 | Seattle | WA | 635 | 11 | 科技 + 自然 |
| 7 | Chicago | IL | 459 | 11 | 城市旅游 + 商务 |
| 8 | Honolulu | HI | 305 | 7 | 海岛度假 |
| 9 | Dallas | TX | 245 | 13 | 商务出行 |
| 10 | Anaheim | CA | 236 | 7 | 主题乐园 / 度假 |
| **合计** |  | **8 州** | **6,171** | **146** |  |

### 论文表述建议

> 本研究从 Datafiniti Hotel Reviews 公开数据集中筛选覆盖美国 8 个州的 10 个代表性城市作为实验对象，形成 6,171 条城市过滤后评论；在进一步完成清洗、去重和酒店质量过滤后，最终保留 146 家酒店、5,947 条有效评论，覆盖海滩度假、科技商务、文化旅游、会展商务、主题乐园、城市旅游、海岛度假与商务出行等多种住宿场景。

### 日期解析关键修复

原始向量化 `pd.to_datetime()` 对带 `.000Z` 后缀的日期解析不稳定，当前实现已统一改为逐行 `pd.Timestamp()` 解析。  
修复后评论日期有效率为 **100%**。

## B.2 酒店筛选规则

```text
酒店准入条件（全部满足）：
  1. city ∈ Top 10 城市列表
  2. hotel_name 非空
  3. 清洗后有效评论数 ≥ 5
  4. 评论平均文本长度 ≥ 100 字符
```

当前采用 `≥5 评论` 规则，最终保留：

- 酒店数：`146`
- 评论数：`5,947`

## B.3 各层精确数量结果

| 层级 | 当前结果 | 说明 |
| --- | --- | --- |
| 原始评论 | 10,000 | 原始 CSV |
| 城市过滤后 | 6,171 | Top 10 城市 |
| 无效评论过滤后 | 6,170 | 删除明显无效评论 |
| 去重后 | 6,086 | 删除 84 条精确重复 |
| 酒店质量过滤后 | 5,947 | 仅保留 ≥5 评论酒店 |
| 有效酒店 | 146 | 最终知识库酒店数 |
| 分句后证据句 | 51,813 | `sentences.pkl` |
| 方面标签 | 63,085 | `aspect_labels.pkl / aspect_sentiment.pkl` |
| 酒店方面画像 | 876 | 146 × 6 |
| ChromaDB 记录数 | 51,813 | 与 `evidence_index` 一致 |

## B.4 方面标签实际分布

| 方面 | 标签数 | 占比 |
| --- | --- | --- |
| general | 19,644 | 31.1% |
| room_facilities | 15,713 | 24.9% |
| service | 9,070 | 14.4% |
| location_transport | 8,784 | 13.9% |
| value | 3,743 | 5.9% |
| cleanliness | 3,581 | 5.7% |
| quiet_sleep | 2,550 | 4.0% |

## B.5 评分分布（最终清洗样本）

| 评分 | 数量 | 占比 | 弱监督标签 |
| --- | --- | --- | --- |
| 5 | 2,667 | 44.8% | positive |
| 4 | 1,890 | 31.8% | positive |
| 3 | 758 | 12.7% | neutral |
| 2 | 335 | 5.6% | negative |
| 1 | 297 | 5.0% | negative |

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

COMMENT ON TABLE kb.hotel IS '酒店主表 — 当前实测 146 行（≥5 评论酒店）';
COMMENT ON COLUMN kb.hotel.hotel_key IS '格式: us/{state}/{city}/{addr}/{hash}';
```

**当前结果**：146 行；按酒店数分布为 San Diego 28、San Francisco 24、Atlanta 20、Orlando 15、Dallas 13、Seattle 11、Chicago 11、New Orleans 10、Anaheim 7、Honolulu 7。

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

COMMENT ON TABLE kb.review IS '评论主表 — 当前实测 5,947 行';
```

**当前结果**：5,947 行
- rating 分布：5★ 44.8%、4★ 31.8%、3★ 12.7%、2★ 5.6%、1★ 5.0%
- recency_bucket：older 73.5%、recent_2y 16.0%、recent_1y 9.6%、recent_90d 0.9%
- has_manager_reply：TRUE 36.5%（2168 / 5947）

## C.5 表 3: `kb.sentence` — 句子表

```sql
CREATE TABLE kb.sentence (
    sentence_id     VARCHAR(24)     PRIMARY KEY,       -- {review_id}_{order:02d}
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

COMMENT ON TABLE kb.sentence IS '句子表 — 当前实测 51,813 行';
```

## C.6 表 4: `kb.aspect_sentiment` — 方面情感标签表

```sql
CREATE TABLE kb.aspect_sentiment (
    id              BIGSERIAL       PRIMARY KEY,
    sentence_id     VARCHAR(24)     NOT NULL REFERENCES kb.sentence(sentence_id),
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

COMMENT ON TABLE kb.aspect_sentiment IS '方面情感标签 — 当前实测 63,085 行';
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

COMMENT ON TABLE kb.hotel_aspect_profile IS '酒店方面画像 — 当前实测 876 行 (146×6)';
```

**方面得分公式**：
```python
final_aspect_score = recency_weighted_pos - recency_weighted_neg - controversy_score * 0.3
```

## C.8 表 6: `kb.evidence_index` — 证据元数据

```sql
CREATE TABLE kb.evidence_index (
    text_id         VARCHAR(24)     PRIMARY KEY,        -- = sentence_id
    hotel_id        VARCHAR(12)     NOT NULL REFERENCES kb.hotel(hotel_id),
    sentence_id     VARCHAR(24)     NOT NULL REFERENCES kb.sentence(sentence_id),
    sentence_text   TEXT            NOT NULL,
    aspect          VARCHAR(30),
    sentiment       VARCHAR(10),
    city            VARCHAR(100),
    hotel_name      VARCHAR(255),
    review_date     DATE,
    rating          SMALLINT,
    recency_bucket  VARCHAR(20),
    embedding_id    INTEGER         NOT NULL             -- ChromaDB 写入批次中的顺序编号
);

CREATE INDEX idx_ev_hotel ON kb.evidence_index(hotel_id);
CREATE INDEX idx_ev_aspect ON kb.evidence_index(aspect);
CREATE INDEX idx_ev_city ON kb.evidence_index(city);

COMMENT ON TABLE kb.evidence_index IS '证据元数据 — 当前实测 51,813 行，与 ChromaDB 一一对应';
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

当前 PostgreSQL 导入已经统一为：

- 配置来源：`configs/db.yaml`
- 密码覆盖：`HOTEL_DB_PASSWORD`
- 导入脚本：`scripts/pipeline/load_kb_to_postgres.py`
- 连接库：`psycopg2`

**当前导入顺序**（遵循外键依赖）：
1. `kb.hotel` ← 146 行
2. `kb.review` ← 5,947 行
3. `kb.sentence` ← 51,813 行
4. `kb.aspect_sentiment` ← 63,085 行
5. `kb.hotel_aspect_profile` ← 876 行
6. `kb.evidence_index` ← 51,813 行

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
| review_date | **100%** | 2003–2019 | 当前全部可用 |
| rating | 100% | 1–5 | 正偏，5★ 占 44.8% |
| sentiment_weak | 100% | pos 76.6% / neu 12.7% / neg 10.6% | 由 rating 推导 |
| has_manager_reply | 100% | TRUE 36.5% | 清洗后保留样本中的检测结果 |
| char_len_clean | 100% | ≥50 | 过滤阈值已稳定执行 |

## D.3 sentence 表

| 字段 | 非空率 | 值域 | 说明 |
| --- | --- | --- | --- |
| sentence_order | 100% | 0 起始递增 | 大部分评论为 4–10 句 |
| char_len | 100% | 15–1014 | 中位 69，均值 79.3 |
| token_count | 100% | 1–187 | 中位 13，均值 14.7 |

## D.4 aspect_sentiment 表

| 字段 | 值域 | 当前分布 |
| --- | --- | --- |
| aspect | 7 值 | general 31.1%、room_facilities 24.9%、service 14.4%、location_transport 13.9%、value 5.9%、cleanliness 5.7%、quiet_sleep 4.0% |
| sentiment | 3 值 | positive 64.8%、negative 17.9%、neutral 17.3% |
| label_source | 2 值 | rule 96.6%、zeroshot 3.4% |
| confidence | 0.0–1.0 | rule 与 zeroshot 均保留模型或规则置信度 |

---

# E. 数据处理流水线

> 当前仓库已经固定为 9 个脚本步骤，计划文档仅保留现行实现。

## E.1 当前脚本链路

| 步骤 | 脚本 | 输入 | 输出 | 当前结果 |
| --- | --- | --- | --- | --- |
| 1 | `load_and_filter_reviews.py` | 原始 CSV | `city_filtered.pkl` | 6,171 |
| 2 | `clean_and_dedupe_reviews.py` | `city_filtered.pkl` | `cleaned_reviews.pkl` | 5,947 |
| 3 | `split_reviews_into_sentences.py` | `cleaned_reviews.pkl` | `sentences.pkl` | 51,813 |
| 4 | `classify_sentence_aspects.py` | `sentences.pkl` | `aspect_labels.pkl` | 63,085 |
| 5 | `classify_aspect_sentiment.py` | `aspect_labels.pkl` | `aspect_sentiment.pkl` | 63,085 |
| 6 | `build_hotel_aspect_profiles.py` | `aspect_sentiment.pkl` | `hotel_profiles.pkl` | 876 |
| 7 | `build_evidence_vector_index.py` | 句子/评论/标签 | `evidence_index.pkl` + ChromaDB | 51,813 |
| 8 | `load_kb_to_postgres.py` | 全部中间产物 | PostgreSQL | 6 张核心表导入完成 |
| 9 | `validate_kb_assets.py` | 中间产物 + PG + Chroma | 验收报告 | 28/28 通过 |

## E.2 当前关键参数

```yaml
cleaning:
  min_text_length: 50
  min_preserve_length: 100

hotel_filter:
  min_reviews_per_hotel: 5
  min_avg_text_length: 100

sentence:
  splitter: "spacy"
  min_sentence_length: 15
  min_fragment_merge_length: 15

aspect:
  zeroshot_model: "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
  zeroshot_threshold: 0.45
  zeroshot_min_char_len: 30
  zeroshot_batch_size: 32

sentiment:
  model: "nlptown/bert-base-multilingual-uncased-sentiment"
  batch_size: 64

embedding:
  model: "BAAI/bge-small-en-v1.5"
  chroma_collection: "hotel_evidence"
  chroma_persist_dir: "data/chroma_db"
```

## E.3 当前实现要点

- 日期解析已修复为 100% 可用
- 分句逻辑已改为“先分句、再合并短碎片、最后长度过滤”
- 方面分类输出为长表，不再只保留单一主方面
- 情感分类先按唯一句子推理，再回填到长表标签
- 向量库最终实现为 ChromaDB，并支持元数据过滤检索

---

# F. 数据集划分策略

## F.1 按酒店切分（非按评论）

当前数据处理模块尚未执行训练集 / 验证集 / 测试集划分，但后续若进入推荐模型或排序模型训练阶段，仍建议沿用“按酒店切分、不按评论切分”的原则，避免同一酒店同时出现在训练集和测试集中造成信息泄漏。

## F.2 跨城验证（可选加分项）

可继续保留“跨城泛化验证”作为后续实验设计，例如将 Dallas / Anaheim 作为额外泛化验证城市。

---

# G. 依赖清单

```txt
# Python 3.10+
pandas
numpy
tqdm
pyyaml
scikit-learn
spacy
transformers
torch
sentence-transformers
chromadb
psycopg2-binary
python-dotenv
jupyter
ipykernel
matplotlib
seaborn
pyarrow
```

说明：

- `pyarrow` 已纳入依赖，用于兼容部分 Colab 生成的 `.pkl`
- 英文分句建议额外安装 `en_core_web_sm`
- 当前依赖已与仓库中的 `requirements.txt` 保持一致

---

# H. 执行时间预估

| 步骤 | 本地 CPU | Colab / GPU |
| --- | --- | --- |
| `01`–`03` | 分钟级 | 通常无必要上 GPU |
| `classify_sentence_aspects.py` | 最慢，可能数小时 | 推荐在 Colab 或云端执行 |
| `classify_aspect_sentiment.py` | 十几分钟到更久 | 已实测约数分钟 |
| `build_hotel_aspect_profiles.py` | 秒级 | 秒级 |
| `build_evidence_vector_index.py` | 取决于编码速度 | GPU 下几分钟级 |
| `08`–`09` | 分钟级 | 推荐回本地执行 |

**当前经验**：最主要瓶颈仍然是 Step 4 的 zero-shot 分类，因此推荐采用“Colab 跑 04–06，本地跑 07–09”的混合方案。

---

# I. 项目目录结构

```text
DatafinitiHotelReviews/
├── raw_data/
│   └── Datafiniti_Hotel_Reviews.csv
├── data/
│   ├── intermediate/
│   │   ├── city_filtered.pkl
│   │   ├── cleaned_reviews.pkl
│   │   ├── sentences.pkl
│   │   ├── aspect_labels.pkl
│   │   ├── aspect_sentiment.pkl
│   │   ├── hotel_profiles.pkl
│   │   └── evidence_index.pkl
│   └── chroma_db/
├── scripts/
│   ├── pipeline/
│   │   ├── load_and_filter_reviews.py
│   │   ├── clean_and_dedupe_reviews.py
│   │   ├── split_reviews_into_sentences.py
│   │   ├── classify_sentence_aspects.py
│   │   ├── classify_aspect_sentiment.py
│   │   ├── build_hotel_aspect_profiles.py
│   │   ├── build_evidence_vector_index.py
│   │   ├── load_kb_to_postgres.py
│   │   └── validate_kb_assets.py
│   └── shared/
│       └── project_utils.py
├── sql/
│   └── init_schema.sql
├── configs/
│   ├── params.yaml
│   └── db.yaml
├── docs/plans/
│   ├── data_pipeline_plan.md
│   ├── data_pipeline_implementation_guide.md
│   └── thesis_overall_plan.md
└── requirements.txt
```

---

# J. 执行检查清单

当前项目以以下结果作为通过标准：

- [x] `hotel` 表 146 行
- [x] `review` 表 5,947 行
- [x] `sentence` 表 51,813 行
- [x] `aspect_sentiment` 表 63,085 行
- [x] `hotel_aspect_profile` 表 876 行
- [x] `evidence_index` 表 = `sentence` 表行数
- [x] ChromaDB 记录数 = `sentence` 表行数
- [x] 覆盖城市 = 10，覆盖州 = 8
- [x] 每个方面标签数均 > 0，且 `general <= 35%`
- [x] 情感三分类与评分分布方向一致
- [x] ChromaDB 支持 `city + aspect + sentiment` 过滤查询
- [x] `validate_kb_assets.py` 在用户本地终端通过 `28/28`

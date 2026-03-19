-- ============================================================
-- Hotel Reviews Knowledge Base — PostgreSQL Schema
-- 适用: 本科毕业设计「基于酒店评论知识库的推荐智能体研究」
-- 执行: psql -U hotel_user -d hotel_reviews_kb -f sql/init_schema.sql
-- ============================================================

CREATE SCHEMA IF NOT EXISTS kb;

-- ────────────────────────────────────────────────
-- 1. hotel 表 — 酒店主体信息 + 聚合统计
-- ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS kb.hotel (
    hotel_id         TEXT        PRIMARY KEY,
    hotel_name       TEXT        NOT NULL,
    city             TEXT        NOT NULL,
    state            CHAR(2)     NOT NULL,
    address          TEXT,
    latitude         NUMERIC(9,6),
    longitude        NUMERIC(9,6),
    n_reviews        INTEGER     NOT NULL DEFAULT 0,
    avg_rating       NUMERIC(3,2),
    min_review_date  DATE,
    max_review_date  DATE,
    n_sentences      INTEGER     DEFAULT 0,
    overall_pos      NUMERIC(4,3),   -- positive 句子比例
    overall_neg      NUMERIC(4,3),
    overall_neu      NUMERIC(4,3),
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hotel_city    ON kb.hotel (city);
CREATE INDEX IF NOT EXISTS idx_hotel_state   ON kb.hotel (state);
CREATE INDEX IF NOT EXISTS idx_hotel_rating  ON kb.hotel (avg_rating DESC);

-- ────────────────────────────────────────────────
-- 2. review 表 — 清洗后评论主体
-- ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS kb.review (
    review_id          TEXT        PRIMARY KEY,
    hotel_id           TEXT        NOT NULL REFERENCES kb.hotel(hotel_id),
    review_date        DATE,
    review_year        SMALLINT,
    review_month       SMALLINT,
    recency_bucket     TEXT,           -- recent_90d / recent_1y / recent_2y / older
    rating             SMALLINT        CHECK (rating BETWEEN 1 AND 5),
    sentiment_weak     TEXT            CHECK (sentiment_weak IN ('positive','neutral','negative')),
    has_manager_reply  BOOLEAN         DEFAULT FALSE,
    char_len_clean     INTEGER,
    review_text_clean  TEXT,
    username           TEXT,
    user_city          TEXT,
    created_at         TIMESTAMPTZ     DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_review_hotel       ON kb.review (hotel_id);
CREATE INDEX IF NOT EXISTS idx_review_date        ON kb.review (review_date DESC);
CREATE INDEX IF NOT EXISTS idx_review_rating      ON kb.review (rating);
CREATE INDEX IF NOT EXISTS idx_review_recency     ON kb.review (recency_bucket);
-- 全文检索索引
CREATE INDEX IF NOT EXISTS idx_review_fts ON kb.review
    USING GIN (to_tsvector('english', COALESCE(review_text_clean, '')));

-- ────────────────────────────────────────────────
-- 3. sentence 表 — 句子粒度文本
-- ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS kb.sentence (
    sentence_id     TEXT        PRIMARY KEY,   -- {review_id}_s{order}
    review_id       TEXT        NOT NULL REFERENCES kb.review(review_id),
    hotel_id        TEXT        NOT NULL,
    sentence_order  SMALLINT    NOT NULL,
    sentence_text   TEXT        NOT NULL,
    char_len        SMALLINT    NOT NULL,
    token_count     SMALLINT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sentence_review  ON kb.sentence (review_id);
CREATE INDEX IF NOT EXISTS idx_sentence_hotel   ON kb.sentence (hotel_id);
CREATE INDEX IF NOT EXISTS idx_sentence_fts ON kb.sentence
    USING GIN (to_tsvector('english', sentence_text));

-- ────────────────────────────────────────────────
-- 4. aspect_sentiment 表 — 句子方面 + 情感标签
-- ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS kb.aspect_sentiment (
    id              BIGSERIAL   PRIMARY KEY,
    sentence_id     TEXT        NOT NULL REFERENCES kb.sentence(sentence_id),
    review_id       TEXT        NOT NULL,
    hotel_id        TEXT        NOT NULL,
    aspect          TEXT        NOT NULL
                    CHECK (aspect IN (
                        'location_transport','cleanliness','service',
                        'room_facilities','quiet_sleep','value','general'
                    )),
    sentiment       TEXT        NOT NULL
                    CHECK (sentiment IN ('positive','negative','neutral')),
    label_source    TEXT        NOT NULL
                    CHECK (label_source IN ('rule','zeroshot','manual')),
    confidence      NUMERIC(4,3),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (sentence_id, aspect)
);

CREATE INDEX IF NOT EXISTS idx_as_hotel    ON kb.aspect_sentiment (hotel_id);
CREATE INDEX IF NOT EXISTS idx_as_aspect   ON kb.aspect_sentiment (aspect);
CREATE INDEX IF NOT EXISTS idx_as_sent     ON kb.aspect_sentiment (sentiment);
CREATE INDEX IF NOT EXISTS idx_as_hotel_aspect ON kb.aspect_sentiment (hotel_id, aspect);

-- ────────────────────────────────────────────────
-- 5. hotel_aspect_profile 表 — 酒店 × 方面聚合画像
-- ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS kb.hotel_aspect_profile (
    hotel_id     TEXT           NOT NULL REFERENCES kb.hotel(hotel_id),
    aspect       TEXT           NOT NULL,
    pos_ratio    NUMERIC(4,3),
    neg_ratio    NUMERIC(4,3),
    neu_ratio    NUMERIC(4,3),
    n_sentences  INTEGER        DEFAULT 0,
    updated_at   TIMESTAMPTZ    DEFAULT NOW(),
    PRIMARY KEY (hotel_id, aspect)
);

CREATE INDEX IF NOT EXISTS idx_hap_aspect ON kb.hotel_aspect_profile (aspect);
CREATE INDEX IF NOT EXISTS idx_hap_pos    ON kb.hotel_aspect_profile (pos_ratio DESC);

-- ────────────────────────────────────────────────
-- 6. embedding_meta 表 — ChromaDB 条目对应关系
-- ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS kb.embedding_meta (
    sentence_id      TEXT        PRIMARY KEY REFERENCES kb.sentence(sentence_id),
    chroma_id        TEXT        NOT NULL UNIQUE,
    embedding_model  TEXT        NOT NULL DEFAULT 'BAAI/bge-small-en-v1.5',
    indexed_at       TIMESTAMPTZ DEFAULT NOW()
);

-- ────────────────────────────────────────────────
-- 视图 1: hotel_overview — 酒店综合视图
-- ────────────────────────────────────────────────
CREATE OR REPLACE VIEW kb.hotel_overview AS
SELECT
    h.hotel_id,
    h.hotel_name,
    h.city,
    h.state,
    h.n_reviews,
    ROUND(h.avg_rating, 2)   AS avg_rating,
    h.n_sentences,
    h.overall_pos,
    h.overall_neg,
    h.max_review_date        AS latest_review
FROM kb.hotel h
ORDER BY h.n_reviews DESC;

-- ────────────────────────────────────────────────
-- 视图 2: aspect_evidence — RAG 证据查询视图
-- ────────────────────────────────────────────────
CREATE OR REPLACE VIEW kb.aspect_evidence AS
SELECT
    s.sentence_id,
    s.sentence_text,
    a.aspect,
    a.sentiment,
    a.label_source,
    a.confidence,
    r.rating,
    r.review_date,
    r.recency_bucket,
    h.hotel_name,
    h.city,
    h.state
FROM kb.aspect_sentiment a
JOIN kb.sentence s ON a.sentence_id = s.sentence_id
JOIN kb.review   r ON a.review_id   = r.review_id
JOIN kb.hotel    h ON a.hotel_id    = h.hotel_id;

-- done
SELECT 'Schema initialized successfully' AS status;

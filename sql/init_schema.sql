-- ============================================================
-- Hotel Reviews Knowledge Base — PostgreSQL Schema
-- 适用: 本科毕业设计「基于酒店评论知识库的推荐智能体研究」
-- 执行: psql -U <db_user> -d hotel_reviews_kb -f sql/init_schema.sql
-- 说明: 数据库连接参数由 configs/db.yaml 提供，HOTEL_DB_PASSWORD 可临时覆盖密码字段
-- ============================================================

CREATE SCHEMA IF NOT EXISTS kb;
SET search_path TO kb, public;

DROP VIEW IF EXISTS kb.v_evidence_full;
DROP VIEW IF EXISTS kb.v_hotel_overview;

DROP TABLE IF EXISTS kb.evidence_index CASCADE;
DROP TABLE IF EXISTS kb.hotel_aspect_profile CASCADE;
DROP TABLE IF EXISTS kb.aspect_sentiment CASCADE;
DROP TABLE IF EXISTS kb.sentence CASCADE;
DROP TABLE IF EXISTS kb.review CASCADE;
DROP TABLE IF EXISTS kb.hotel CASCADE;

CREATE TABLE kb.hotel (
    hotel_id        VARCHAR(12)     PRIMARY KEY,
    hotel_key       TEXT            NOT NULL UNIQUE,
    hotel_name      VARCHAR(255)    NOT NULL,
    address         TEXT,
    city            VARCHAR(100)    NOT NULL,
    state           CHAR(2)         NOT NULL,
    country         CHAR(2)         NOT NULL DEFAULT 'US',
    postal_code     VARCHAR(20),
    lat             NUMERIC(10, 6),
    lng             NUMERIC(10, 6),
    hotel_category  VARCHAR(100),
    categories_raw  TEXT,
    hotel_website   TEXT,
    n_reviews       INTEGER         NOT NULL DEFAULT 0,
    avg_rating      NUMERIC(3, 2),
    rating_std      NUMERIC(3, 2),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_hotel_city ON kb.hotel(city);
CREATE INDEX idx_hotel_state ON kb.hotel(state);
CREATE INDEX idx_hotel_name_gin ON kb.hotel USING GIN (to_tsvector('english', hotel_name));

CREATE TABLE kb.review (
    review_id           VARCHAR(16)     PRIMARY KEY,
    hotel_id            VARCHAR(12)     NOT NULL REFERENCES kb.hotel(hotel_id),
    review_date         DATE            NOT NULL,
    review_year         SMALLINT        NOT NULL,
    review_month        SMALLINT        NOT NULL,
    recency_bucket      VARCHAR(20)     NOT NULL,
    rating              SMALLINT        NOT NULL CHECK (rating BETWEEN 1 AND 5),
    sentiment_weak      VARCHAR(10)     NOT NULL CHECK (sentiment_weak IN ('positive', 'neutral', 'negative')),
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

CREATE INDEX idx_review_hotel ON kb.review(hotel_id);
CREATE INDEX idx_review_date ON kb.review(review_date);
CREATE INDEX idx_review_rating ON kb.review(rating);
CREATE INDEX idx_review_recency ON kb.review(recency_bucket);
CREATE INDEX idx_review_city ON kb.review(city);
CREATE INDEX idx_review_fulltext ON kb.review USING GIN (to_tsvector('english', COALESCE(full_text, '')));

CREATE TABLE kb.sentence (
    sentence_id     VARCHAR(24)     PRIMARY KEY,
    review_id       VARCHAR(16)     NOT NULL REFERENCES kb.review(review_id),
    hotel_id        VARCHAR(12)     NOT NULL REFERENCES kb.hotel(hotel_id),
    sentence_text   TEXT            NOT NULL,
    sentence_order  SMALLINT        NOT NULL,
    char_len        INTEGER         NOT NULL,
    token_count     INTEGER,
    city            VARCHAR(100)
);

CREATE INDEX idx_sent_review ON kb.sentence(review_id);
CREATE INDEX idx_sent_hotel ON kb.sentence(hotel_id);
CREATE INDEX idx_sent_city ON kb.sentence(city);
CREATE INDEX idx_sent_fulltext ON kb.sentence USING GIN (to_tsvector('english', sentence_text));

CREATE TABLE kb.aspect_sentiment (
    id              BIGSERIAL       PRIMARY KEY,
    sentence_id     VARCHAR(24)     NOT NULL REFERENCES kb.sentence(sentence_id),
    hotel_id        VARCHAR(12)     NOT NULL REFERENCES kb.hotel(hotel_id),
    aspect          VARCHAR(30)     NOT NULL,
    sentiment       VARCHAR(10)     NOT NULL,
    confidence      NUMERIC(4, 3),
    label_source    VARCHAR(20)     NOT NULL,
    evidence_span   TEXT,
    CONSTRAINT chk_aspect
        CHECK (aspect IN ('location_transport', 'cleanliness', 'service', 'room_facilities', 'quiet_sleep', 'value', 'general')),
    CONSTRAINT chk_sentiment
        CHECK (sentiment IN ('positive', 'negative', 'neutral')),
    CONSTRAINT chk_label_source
        CHECK (label_source IN ('rule', 'zeroshot', 'manual')),
    CONSTRAINT uq_sentence_aspect UNIQUE (sentence_id, aspect)
);

CREATE INDEX idx_as_sentence ON kb.aspect_sentiment(sentence_id);
CREATE INDEX idx_as_hotel ON kb.aspect_sentiment(hotel_id);
CREATE INDEX idx_as_aspect ON kb.aspect_sentiment(aspect);
CREATE INDEX idx_as_sentiment ON kb.aspect_sentiment(sentiment);
CREATE INDEX idx_as_hotel_aspect ON kb.aspect_sentiment(hotel_id, aspect);

CREATE TABLE kb.hotel_aspect_profile (
    hotel_id                VARCHAR(12)     NOT NULL REFERENCES kb.hotel(hotel_id),
    aspect                  VARCHAR(30)     NOT NULL,
    pos_count               INTEGER         NOT NULL DEFAULT 0,
    neg_count               INTEGER         NOT NULL DEFAULT 0,
    neu_count               INTEGER         NOT NULL DEFAULT 0,
    total_count             INTEGER         NOT NULL DEFAULT 0,
    recency_weighted_pos    NUMERIC(8, 3)   DEFAULT 0,
    recency_weighted_neg    NUMERIC(8, 3)   DEFAULT 0,
    controversy_score       NUMERIC(4, 3)   DEFAULT 0,
    final_aspect_score      NUMERIC(8, 3)   DEFAULT 0,
    PRIMARY KEY (hotel_id, aspect),
    CONSTRAINT chk_profile_aspect
        CHECK (aspect IN ('location_transport', 'cleanliness', 'service', 'room_facilities', 'quiet_sleep', 'value'))
);

CREATE INDEX idx_profile_hotel ON kb.hotel_aspect_profile(hotel_id);
CREATE INDEX idx_profile_aspect ON kb.hotel_aspect_profile(aspect);

CREATE TABLE kb.evidence_index (
    text_id         VARCHAR(24)     PRIMARY KEY,
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
    embedding_id    INTEGER         NOT NULL
);

CREATE INDEX idx_ev_hotel ON kb.evidence_index(hotel_id);
CREATE INDEX idx_ev_aspect ON kb.evidence_index(aspect);
CREATE INDEX idx_ev_city ON kb.evidence_index(city);

CREATE VIEW kb.v_hotel_overview AS
SELECT
    h.hotel_id,
    h.hotel_name,
    h.city,
    h.state,
    h.n_reviews,
    h.avg_rating,
    MAX(CASE WHEN p.aspect = 'location_transport' THEN p.final_aspect_score END) AS score_location,
    MAX(CASE WHEN p.aspect = 'cleanliness' THEN p.final_aspect_score END) AS score_cleanliness,
    MAX(CASE WHEN p.aspect = 'service' THEN p.final_aspect_score END) AS score_service,
    MAX(CASE WHEN p.aspect = 'room_facilities' THEN p.final_aspect_score END) AS score_room,
    MAX(CASE WHEN p.aspect = 'quiet_sleep' THEN p.final_aspect_score END) AS score_quiet,
    MAX(CASE WHEN p.aspect = 'value' THEN p.final_aspect_score END) AS score_value
FROM kb.hotel h
LEFT JOIN kb.hotel_aspect_profile p ON h.hotel_id = p.hotel_id
GROUP BY h.hotel_id, h.hotel_name, h.city, h.state, h.n_reviews, h.avg_rating;

CREATE VIEW kb.v_evidence_full AS
SELECT
    s.sentence_id,
    s.sentence_text,
    s.hotel_id,
    h.hotel_name,
    h.city,
    h.state,
    a.aspect,
    a.sentiment,
    a.confidence,
    a.label_source,
    r.rating,
    r.review_date,
    r.recency_bucket
FROM kb.sentence s
JOIN kb.hotel h ON s.hotel_id = h.hotel_id
LEFT JOIN kb.aspect_sentiment a ON s.sentence_id = a.sentence_id
JOIN kb.review r ON s.review_id = r.review_id;

SELECT 'Schema initialized successfully' AS status;

"""
utils.py — 公用工具函数
所有脚本共用的辅助函数集中在此处
"""

import hashlib
import re
import yaml
import pandas as pd
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────
# 配置加载
# ──────────────────────────────────────────────

def load_config(config_path: str = "configs/params.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# 主键生成
# ──────────────────────────────────────────────

def make_hotel_id(hotel_key: str) -> str:
    """keys 字段 → 12 位 MD5 定长主键"""
    return hashlib.md5(hotel_key.encode("utf-8")).hexdigest()[:12]


def make_review_id(hotel_id: str, date_raw: str, title: str, text: str) -> str:
    """组合字段 → 16 位 SHA256 评论唯一标识"""
    raw = f"{hotel_id}|{date_raw}|{title}|{text[:200]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ──────────────────────────────────────────────
# 日期解析（修复 .000Z 解析 Bug）
# ──────────────────────────────────────────────

def safe_parse_timestamp(date_str) -> Optional[pd.Timestamp]:
    """
    修复 pd.to_datetime 对 '2018-11-27T00:00:00.000Z' 的向量化解析失败。
    改用逐行 pd.Timestamp() 解析，确保 100% 成功率。
    """
    if pd.isna(date_str):
        return pd.NaT
    try:
        return pd.Timestamp(str(date_str))
    except Exception:
        return pd.NaT


# ──────────────────────────────────────────────
# 管理者回复去除
# ──────────────────────────────────────────────

MANAGER_PATTERNS = [
    r"(?:Dear|Hello)\s+(?:Guest|Traveler|Sir|Madam|valued\s+guest|Mr\.|Mrs\.|Ms\.)",
    r"On behalf of (?:the|our)\s+(?:staff|team|management|hotel|entire)",
    r"Thank(?:s|\s+you)\s+(?:for|so\s+much\s+for)\s+(?:your|the|taking)\s+"
    r"(?:review|feedback|comment|kind|candid|recent|time|visit|stay|choosing|sharing|staying|visiting)",
    r"(?:We|I)\s+(?:appreciate|value)\s+(?:your|the)\s+(?:feedback|review|comment|time|patronage|kind\s+words)",
    r"(?:We|I)(?:'re|'m| are| am)\s+(?:sorry|happy|glad|thrilled|delighted|pleased|so glad|very sorry)\s+"
    r"(?:to\s+hear|that\s+you|you\s+(?:had|enjoyed|experienced))",
    r"(?:We|I)\s+(?:hope|look forward)\s+to\s+(?:see|seeing|welcome|welcoming)\s+you",
    r"(?:Please|Do not hesitate to)\s+contact\s+(?:us|me)",
    r"(?:Sincerely|Kind\s+Regards|Best\s+Regards|Warm\s+Regards|Respectfully),?\s*(?:\n|\s)*\w",
]

TAIL_NOISE_PATTERNS = [
    r"\.\.\.\s*More\s*$",
    r"(?:Read\s+)?(?:Full\s+)?Review\s*$",
]


def remove_manager_response(text: str, min_preserve: int = 100) -> tuple:
    """
    去除管理者回复，返回 (cleaned_text, had_manager_reply)。
    """
    if pd.isna(text) or not isinstance(text, str):
        return (str(text) if not pd.isna(text) else ""), False

    earliest = len(text)
    for pat in MANAGER_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m and m.start() > min_preserve and m.start() < earliest:
            earliest = m.start()

    had_reply = earliest < len(text)
    cleaned = text[:earliest].rstrip()
    for pat in TAIL_NOISE_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).rstrip()

    return cleaned, had_reply


# ──────────────────────────────────────────────
# 时间桶
# ──────────────────────────────────────────────

def assign_recency_bucket(review_date, reference_date: pd.Timestamp, buckets: dict) -> str:
    if pd.isna(review_date):
        return "unknown"
    try:
        delta = (reference_date - pd.Timestamp(review_date)).days
    except Exception:
        return "unknown"
    if delta <= 90:   return "recent_90d"
    elif delta <= 365: return "recent_1y"
    elif delta <= 730: return "recent_2y"
    else:              return "older"


def rating_to_weak_sentiment(rating) -> str:
    try:
        r = int(rating)
        return "positive" if r >= 4 else ("negative" if r <= 2 else "neutral")
    except (ValueError, TypeError):
        return "neutral"


# ──────────────────────────────────────────────
# 数据库工具
# ──────────────────────────────────────────────

def get_pg_conn(config: dict):
    """从配置创建 psycopg2 连接"""
    import os
    import psycopg2
    import re as _re
    db_url = config["data"]["db_url"]
    pw = os.environ.get("HOTEL_DB_PASSWORD")
    if pw:
        db_url = _re.sub(r"(?<=://.+:)[^@]+(?=@)", pw, db_url)
    return psycopg2.connect(db_url)


# ──────────────────────────────────────────────
# 日志 + 断言
# ──────────────────────────────────────────────

def log_step(step: str, msg: str) -> None:
    print(f"[{step}] {msg}")


def assert_shape(df: pd.DataFrame, expected_rows: int,
                 tolerance: float = 0.10, label: str = "") -> None:
    lo = int(expected_rows * (1 - tolerance))
    hi = int(expected_rows * (1 + tolerance))
    actual = len(df)
    assert lo <= actual <= hi, (
        f"[ASSERT] {label}: 期望 ~{expected_rows}±{tolerance*100:.0f}% 行，实际 {actual}"
    )
    print(f"✅ {label}: {actual} 行 (预期 ~{expected_rows})")

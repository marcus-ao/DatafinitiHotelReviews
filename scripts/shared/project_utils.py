"""Shared helpers for the hotel reviews data pipeline."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]


ASPECT_CATEGORIES = [
    "location_transport",
    "cleanliness",
    "service",
    "room_facilities",
    "quiet_sleep",
    "value",
]

ALL_ASPECT_CATEGORIES = ASPECT_CATEGORIES + ["general"]

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


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return ROOT_DIR / resolved


def load_config(config_path: str = "configs/params.yaml") -> dict:
    return load_yaml(resolve_repo_path(config_path))


def load_db_config(config_path: str = "configs/db.yaml") -> dict:
    path = resolve_repo_path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Database config not found: {path}. "
            "Create it from configs/db.example.yaml first."
        )

    raw = load_yaml(path)
    postgres = raw.get("postgres", {})
    required = {"host", "port", "dbname", "user", "password", "schema"}
    missing = sorted(required - set(postgres))
    if missing:
        raise KeyError(f"configs/db.yaml 缺少字段: {', '.join(missing)}")

    password = os.environ.get("HOTEL_DB_PASSWORD", postgres["password"])
    return {
        "host": postgres["host"],
        "port": int(postgres["port"]),
        "dbname": postgres["dbname"],
        "user": postgres["user"],
        "password": password,
        "schema": postgres["schema"],
        "connect_timeout": int(postgres.get("connect_timeout", 30)),
        "pool_size": int(postgres.get("pool_size", 5)),
        "max_overflow": int(postgres.get("max_overflow", 10)),
    }


def get_pg_conn(db_config: Optional[dict] = None):
    import psycopg2

    cfg = db_config or load_db_config()
    return psycopg2.connect(
        host=cfg["host"],
        port=cfg["port"],
        dbname=cfg["dbname"],
        user=cfg["user"],
        password=cfg["password"],
        connect_timeout=cfg["connect_timeout"],
    )


def make_hotel_id(hotel_key: str) -> str:
    return hashlib.md5(str(hotel_key).encode("utf-8")).hexdigest()[:12]


def make_review_id(hotel_id: str, date_raw: str, title: str, text: str) -> str:
    raw = f"{hotel_id}|{date_raw}|{title}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def safe_parse_timestamp(date_str) -> Optional[pd.Timestamp]:
    if pd.isna(date_str):
        return pd.NaT
    try:
        timestamp = pd.Timestamp(str(date_str))
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        return timestamp
    except Exception:
        return pd.NaT


def remove_manager_response(text: str, min_preserve: int = 100) -> tuple[str, bool]:
    if pd.isna(text) or not isinstance(text, str):
        return "", False

    earliest = len(text)
    for pattern in MANAGER_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and min_preserve < match.start() < earliest:
            earliest = match.start()

    cleaned = text[:earliest].rstrip()
    for pattern in TAIL_NOISE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).rstrip()

    return cleaned, earliest < len(text)


def assign_recency_bucket(
    review_date,
    reference_date: pd.Timestamp,
    buckets: dict,
) -> str:
    if pd.isna(review_date):
        return "unknown"

    try:
        delta_days = (reference_date - pd.Timestamp(review_date)).days
    except Exception:
        return "unknown"

    if delta_days <= 90:
        return "recent_90d"
    if delta_days <= 365:
        return "recent_1y"
    if delta_days <= 730:
        return "recent_2y"
    return "older"


def rating_to_weak_sentiment(rating) -> str:
    try:
        value = int(rating)
    except (TypeError, ValueError):
        return "neutral"
    return "positive" if value >= 4 else ("negative" if value <= 2 else "neutral")


def normalize_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def log_step(step: str, msg: str) -> None:
    print(f"[{step}] {msg}")


def ensure_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{label} 缺少字段: {', '.join(missing)}")


def assert_shape(
    df: pd.DataFrame,
    expected_rows: int,
    tolerance: float = 0.10,
    label: str = "",
) -> None:
    low = int(expected_rows * (1 - tolerance))
    high = int(expected_rows * (1 + tolerance))
    actual = len(df)
    if not low <= actual <= high:
        raise AssertionError(
            f"[ASSERT] {label}: 期望 ~{expected_rows}±{tolerance * 100:.0f}% 行，实际 {actual}"
        )
    print(f"[OK] {label}: {actual} 行 (预期 ~{expected_rows})")


def assert_unique(df: pd.DataFrame, subset: list[str], label: str) -> None:
    dup_count = int(df.duplicated(subset=subset).sum())
    if dup_count:
        raise AssertionError(f"{label} 存在 {dup_count} 条重复记录: {subset}")
    print(f"[OK] {label}: 主键唯一 ({', '.join(subset)})")


def iso_date(value) -> Optional[str]:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()

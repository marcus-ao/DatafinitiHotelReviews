"""Lightweight normalization helpers for E3/E4 behavior experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ASPECT_CATEGORIES = [
    "location_transport",
    "cleanliness",
    "service",
    "room_facilities",
    "quiet_sleep",
    "value",
]


NULL_LIKE_STRINGS = {
    "",
    "null",
    "none",
    "n/a",
    "na",
    "无",
    "没有",
    "未提及",
    "未知",
}


def stable_sorted_unique(values: list[str]) -> list[str]:
    return sorted(set(values))


def normalize_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def is_null_like(value: Any) -> bool:
    if value is None:
        return True
    text = normalize_whitespace(str(value)).lower()
    return text in NULL_LIKE_STRINGS


def canonicalize_label(value: Any) -> str:
    text = normalize_whitespace(str(value)).lower()
    text = text.replace("：", ":").replace("，", ",")
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\u4e00-\u9fff ]+", " ", text, flags=re.UNICODE)
    return normalize_whitespace(text)


def canonicalize_city_key(value: Any) -> str:
    text = normalize_whitespace(str(value)).lower()
    text = text.replace("：", ":").replace("，", ",")
    text = re.sub(r"\s*:\s*", ":", text)
    text = re.sub(r"\s*,\s*", ",", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = text.replace("(", "").replace(")", "")
    return normalize_whitespace(text)


def build_alias_lookup(alias_map: dict[str, list[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            lookup[canonicalize_label(alias)] = canonical
    return lookup


ASPECT_ALIAS_LOOKUP = build_alias_lookup(
    {
        "location_transport": [
            "location_transport",
            "location transport",
            "location",
            "transport",
            "交通位置",
            "位置交通",
            "地理位置",
        ],
        "cleanliness": [
            "cleanliness",
            "clean",
            "clean clean",
            "hygiene",
            "卫生",
            "干净",
            "卫生干净",
        ],
        "service": [
            "service",
            "services",
            "服务",
            "服务质量",
        ],
        "room_facilities": [
            "room_facilities",
            "room facilities",
            "room facility",
            "room",
            "facilities",
            "房间设施",
            "房间",
            "设施",
        ],
        "quiet_sleep": [
            "quiet_sleep",
            "quiet sleep",
            "quiet",
            "sleep quality",
            "noise",
            "quiet stay",
            "安静",
            "睡眠",
            "安静睡眠",
            "住得安静",
        ],
        "value": [
            "value",
            "cost performance",
            "cost effective",
            "price performance",
            "price value",
            "性价比",
            "划算",
            "值不值",
        ],
    }
)

UNSUPPORTED_ALIAS_LOOKUP = build_alias_lookup(
    {
        "budget": [
            "budget",
            "budget requirement",
            "price limit",
            "price cap",
            "预算",
            "价格限制",
            "预算限制",
        ],
        "distance_to_landmark": [
            "distance_to_landmark",
            "distance to landmark",
            "distance to attraction",
            "near landmark",
            "near attraction",
            "walking distance to attraction",
            "离景点",
            "离地标",
            "步行距离",
            "景点距离",
        ],
        "checkin_date": [
            "checkin_date",
            "check in date",
            "check-in date",
            "stay date",
            "入住日期",
            "入住时间",
            "入住日期要求",
            "指定入住日期",
        ],
    }
)

DECISION_LABEL_LOOKUP = build_alias_lookup(
    {
        "missing_city": [
            "missing_city",
            "missing city",
            "city missing",
            "缺城市",
            "缺少城市",
            "城市缺失",
        ],
        "aspect_conflict": [
            "aspect_conflict",
            "aspect conflict",
            "conflict",
            "方面冲突",
            "偏好冲突",
        ],
        "none": [
            "none",
            "no clarification",
            "no_clarification",
            "normal",
            "无需澄清",
            "不需要澄清",
        ],
    }
)


def coerce_string_list(values: Any) -> list[str] | None:
    if values is None or is_null_like(values):
        return []
    if isinstance(values, (list, tuple, set)):
        cleaned: list[str] = []
        for item in values:
            if item is None or is_null_like(item):
                continue
            cleaned.append(normalize_whitespace(str(item)))
        return cleaned
    if isinstance(values, str):
        text = normalize_whitespace(values)
        if is_null_like(text):
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                decoded = None
            if decoded is not None:
                return coerce_string_list(decoded)
        if any(sep in text for sep in ["，", ",", "、", ";", "；", "|", "/"]):
            parts = re.split(r"[，,、;；|/]+", text)
            return [normalize_whitespace(part) for part in parts if normalize_whitespace(part)]
        return [text]
    return None


def normalize_city_value(raw_value: Any, city_to_state: dict[str, str]) -> str | None:
    if raw_value is None or is_null_like(raw_value):
        return None

    normalized_input = canonicalize_city_key(raw_value)
    for city, state in city_to_state.items():
        variants = {
            canonicalize_city_key(city),
            canonicalize_city_key(f"{city}:{state}"),
            canonicalize_city_key(f"{city}, {state}"),
            canonicalize_city_key(f"{city} {state}"),
            canonicalize_city_key(f"{city}-{state}"),
            canonicalize_city_key(f"{city} ({state})"),
        }
        if normalized_input in variants:
            return city
    return None


def normalize_aspect_values(values: Any) -> tuple[list[str] | None, list[str]]:
    raw_values = coerce_string_list(values)
    if raw_values is None:
        return None, []

    cleaned: list[str] = []
    unknown: list[str] = []
    for raw_value in raw_values:
        normalized = ASPECT_ALIAS_LOOKUP.get(canonicalize_label(raw_value))
        if normalized in ASPECT_CATEGORIES:
            cleaned.append(normalized)
        else:
            unknown.append(str(raw_value))
    return stable_sorted_unique(cleaned), unknown


def normalize_unsupported_values(values: Any) -> tuple[list[str] | None, list[str]]:
    raw_values = coerce_string_list(values)
    if raw_values is None:
        return None, []

    cleaned: list[str] = []
    unknown: list[str] = []
    for raw_value in raw_values:
        normalized = UNSUPPORTED_ALIAS_LOOKUP.get(canonicalize_label(raw_value))
        if normalized is not None:
            cleaned.append(normalized)
        else:
            unknown.append(str(raw_value))
    return stable_sorted_unique(cleaned), unknown


def parse_payload_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = normalize_whitespace(str(value)).lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def normalize_decision_label(value: Any) -> str | None:
    if value is None:
        return None
    normalized = canonicalize_label(value)
    if not normalized:
        return None
    return DECISION_LABEL_LOOKUP.get(normalized)


def load_query_ids_from_file(path: str | Path) -> list[str]:
    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if source.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            values = payload.get("query_ids", [])
        elif isinstance(payload, list):
            values = payload
        else:
            raise ValueError(f"{source} 必须是 JSON list 或包含 query_ids 的 JSON object。")
    else:
        values = text.splitlines()

    query_ids = [normalize_whitespace(str(value)) for value in values if normalize_whitespace(str(value))]
    deduped: list[str] = []
    seen: set[str] = set()
    for query_id in query_ids:
        if query_id not in seen:
            deduped.append(query_id)
            seen.add(query_id)
    return deduped

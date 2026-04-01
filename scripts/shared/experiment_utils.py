"""Shared helpers for experiment assets and experiment runners."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
EXPERIMENT_ASSETS_DIR = EXPERIMENTS_DIR / "assets"
EXPERIMENT_LABELS_DIR = EXPERIMENTS_DIR / "labels"
EXPERIMENT_REPORTS_DIR = EXPERIMENTS_DIR / "reports"
EXPERIMENT_RUNS_DIR = EXPERIMENTS_DIR / "runs"
E1_LABELS_DIR = EXPERIMENT_LABELS_DIR / "e1_aspect_reliability"
E4_LABELS_DIR = EXPERIMENT_LABELS_DIR / "e4_clarification"
E9_LABELS_DIR = EXPERIMENT_LABELS_DIR / "e9_generation"
E6_LABELS_DIR = EXPERIMENT_LABELS_DIR / "e6_qrels"


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: str | Path, payload: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_yaml(path: str | Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def stable_hash(payload: dict) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def city_state_map(review_df: pd.DataFrame) -> dict[str, str]:
    pairs = (
        review_df[["city", "state"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["city", "state"])
    )
    return dict(zip(pairs["city"], pairs["state"]))

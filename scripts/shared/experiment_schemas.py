"""Pydantic models for frozen experiment assets and run logs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ASPECT_NAME = Literal[
    "location_transport",
    "cleanliness",
    "service",
    "room_facilities",
    "quiet_sleep",
    "value",
    "general",
]

UNSUPPORTED_REQUEST = Literal[
    "budget",
    "distance_to_landmark",
    "checkin_date",
]


class UserPreference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: str | None = None
    state: str | None = None
    hotel_category: str | None = None
    focus_aspects: list[ASPECT_NAME] = Field(default_factory=list)
    avoid_aspects: list[ASPECT_NAME] = Field(default_factory=list)
    unsupported_requests: list[UNSUPPORTED_REQUEST] = Field(default_factory=list)
    query_en: str


class PreferenceParseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    group_id: str
    preference: UserPreference
    schema_valid: bool
    raw_response: str
    error_types: list[str] = Field(default_factory=list)


class ClarificationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    group_id: str
    clarify_needed: bool
    clarify_reason: str = ""
    target_slots: list[str] = Field(default_factory=list)
    question: str = ""
    schema_valid: bool
    raw_response: str


class BridgeQueryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    target_aspect: str
    target_role: str
    query_text_zh: str
    query_en_source: str
    retrieval_mode: str
    ranked_rows: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class HotelCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hotel_id: str
    hotel_name: str
    score_total: float
    score_breakdown: dict[str, float]


class SentenceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentence_id: str
    sentence_text: str
    aspect: ASPECT_NAME
    sentiment: Literal["positive", "negative", "neutral"]
    review_date: str | None = None
    score_dense: float | None = None
    score_rerank: float | None = None


class EvidencePack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hotel_id: str
    query_en: str
    evidence_by_aspect: dict[str, list[SentenceCandidate]]
    retrieval_trace: dict[str, Any]


class WorkflowState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn: int = 1
    query_id: str
    preference: UserPreference
    retrieval_mode: str = "aspect_main_no_rerank"
    fallback_enabled: bool = False
    run_config_hash: str | None = None
    pending_clarification: bool = False
    last_candidates: list[HotelCandidate] = Field(default_factory=list)
    last_evidence_packs: list[EvidencePack] = Field(default_factory=list)


class RunLogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    group_id: str
    query_id: str
    retrieval_mode: str
    candidate_mode: str
    config_hash: str
    latency_ms: float
    intermediate_objects: dict[str, Any]

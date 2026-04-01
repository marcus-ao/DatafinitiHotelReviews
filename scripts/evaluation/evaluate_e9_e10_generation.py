"""Generation-stage evaluation utilities for E9 and E10."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.evaluation.evaluate_e2_candidate_selection import (
    build_hotel_summary,
    build_profile_tables,
    candidate_rank,
)
from scripts.evaluation.evaluate_e3_e5_behavior import (
    build_behavior_backend,
    build_rule_clarification,
    load_json,
    parse_json_with_repair,
)
from scripts.evaluation.evaluate_e6_e8_retrieval import (
    build_evidence_lookup,
    build_query_en_target,
    dense_query_hotel,
    markdown_table,
    merge_dense_candidates,
)
from scripts.shared.behavior_postprocess import normalize_aspect_values
from scripts.shared.behavior_runtime import resolve_behavior_runtime_config
from scripts.shared.experiment_schemas import (
    BehaviorRuntimeConfig,
    EvidencePack,
    GenerationEvalUnit,
    HotelCandidate,
    RecommendationItem,
    RecommendationReason,
    RecommendationResponse,
    RunLogEntry,
    SFTManifestRecord,
    SentenceCandidate,
    CitationVerificationResult,
    UserPreference,
)
from scripts.shared.experiment_utils import (
    E9_LABELS_DIR,
    EXPERIMENT_ASSETS_DIR,
    EXPERIMENT_RUNS_DIR,
    ensure_dir,
    load_jsonl,
    stable_hash,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from scripts.shared.project_utils import load_config, resolve_repo_path


E9_GROUPS = [
    "A_free_generation",
    "B_grounded_generation",
    "C_grounded_generation_with_verifier",
]
E10_GROUPS = ["A_base_4b_grounded", "B_peft_4b_grounded"]

E9_RETRIEVAL_MODE = "aspect_main_no_rerank"
E9_CANDIDATE_POLICY = "E2_B_final_aspect_score_top5"
E9_CANDIDATE_MODE = "e9_frozen_top5"
E10_CANDIDATE_MODE = "e10_frozen_assets"
E9_GENERATION_MAX_NEW_TOKENS = 512
E9_MAX_RECOMMENDATIONS = 2
E9_MAX_REASONS_PER_ITEM = 2
E9_PROMPT_SENTENCE_LIMIT_FREE = 2
E9_PROMPT_SENTENCE_LIMIT_GROUNDED = 3
E9_RETRY_LIMIT = 1
E9_OUTPUT_FIELDS = '{"summary":"","recommendations":[{"hotel_id":"","hotel_name":"","reasons":[{"aspect":"service","reason_text":"","sentence_id":null}]}],"unsupported_notice":""}'
E9_QUERY_IDS_PATH = EXPERIMENT_ASSETS_DIR / "e9_generation_eval_query_ids.json"
E9_UNITS_PATH = EXPERIMENT_ASSETS_DIR / "e9_generation_eval_units.jsonl"
SFT_TRAIN_MANIFEST_PATH = EXPERIMENT_ASSETS_DIR / "sft_train_manifest.jsonl"
SFT_DEV_MANIFEST_PATH = EXPERIMENT_ASSETS_DIR / "sft_dev_manifest.jsonl"
E10_TRAIN_CONFIG_TEMPLATE_PATH = EXPERIMENT_ASSETS_DIR / "e10_train_config_template.json"
E10_ADAPTER_METADATA_TEMPLATE_PATH = EXPERIMENT_ASSETS_DIR / "e10_adapter_metadata.template.json"

ASPECT_ZH_LABELS = {
    "location_transport": "位置交通",
    "cleanliness": "卫生干净",
    "service": "服务",
    "room_facilities": "房间设施",
    "quiet_sleep": "安静睡眠",
    "value": "性价比",
}

UNSUPPORTED_ZH_LABELS = {
    "budget": "预算",
    "distance_to_landmark": "离景点距离",
    "checkin_date": "入住日期",
}

E10_ADAPTER_METADATA_REQUIRED_FIELDS = {
    "adapter_name",
    "base_model_id",
    "served_model_id",
    "adapter_path",
    "backend",
}


def review_id_from_sentence_id(sentence_id: str) -> str:
    if "_s" in sentence_id:
        return sentence_id.rsplit("_s", 1)[0]
    return sentence_id.split("_", 1)[0]


def resolve_generation_asset_paths(limit_queries: int | None = None) -> tuple[Path, Path]:
    if limit_queries is None:
        return E9_UNITS_PATH, E9_QUERY_IDS_PATH
    suffix = f"head_{limit_queries}"
    return (
        EXPERIMENT_ASSETS_DIR / f"e9_generation_eval_units.{suffix}.jsonl",
        EXPERIMENT_ASSETS_DIR / f"e9_generation_eval_query_ids.{suffix}.json",
    )


def load_generation_eval_units(path: Path) -> list[GenerationEvalUnit]:
    return [GenerationEvalUnit.model_validate(row) for row in load_jsonl(path)]


def resolve_repo_relative_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    return resolve_repo_path(path_value)


def load_adapter_metadata(path: str | Path) -> dict[str, Any]:
    metadata_path = resolve_repo_path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing adapter metadata: {metadata_path}. "
            "请先准备 adapter metadata 文件，再运行 e10_base_vs_peft。"
        )
    payload = load_json(metadata_path)
    missing = sorted(E10_ADAPTER_METADATA_REQUIRED_FIELDS - set(payload))
    if missing:
        raise KeyError(
            f"Adapter metadata 缺少字段: {', '.join(missing)}"
        )
    payload["_metadata_path"] = str(metadata_path)
    payload["_resolved_adapter_path"] = str(resolve_repo_path(payload["adapter_path"]))
    return payload


def build_peft_runtime_config(
    base_runtime_config: BehaviorRuntimeConfig,
    adapter_metadata: dict[str, Any],
) -> BehaviorRuntimeConfig:
    if adapter_metadata["backend"] != base_runtime_config.llm_backend:
        raise ValueError(
            "Adapter metadata backend 与当前 behavior.llm_backend 不一致。"
        )
    if base_runtime_config.llm_backend != "api":
        raise NotImplementedError(
            "当前 E10 骨架仅支持通过 API backend 对比 Base 4B 与已部署的 PEFT 模型。"
        )
    return base_runtime_config.model_copy(
        update={
            "model_id": str(adapter_metadata["served_model_id"]),
            "use_peft_adapter": True,
            "adapter_path": str(adapter_metadata["_resolved_adapter_path"]),
            "adapter_metadata_path": str(adapter_metadata["_metadata_path"]),
        }
    )


def build_e10_metric_row(
    group_id: str,
    rows: list[dict[str, Any]],
    stable_run_config: dict[str, Any],
) -> dict[str, Any]:
    unsupported_rows = [row["unsupported_honesty"] for row in rows if row["unsupported_honesty"] is not None]
    support_scores = [
        audit_row["support_score"]
        for row in rows
        for audit_row in row["audit_rows"]
    ]
    return {
        "group_id": group_id,
        "query_count": len(rows),
        "citation_precision": round(
            sum(row["verification"].citation_precision for row in rows) / max(len(rows), 1),
            4,
        ),
        "evidence_verifiability_mean": round(sum(support_scores) / max(len(support_scores), 1), 4),
        "unsupported_honesty_rate": round(
            1.0 if not unsupported_rows else sum(unsupported_rows) / len(unsupported_rows),
            4,
        ),
        "schema_valid_rate": round(sum(int(row["response"].schema_valid) for row in rows) / max(len(rows), 1), 4),
        "avg_latency_ms": round(sum(row["latency_ms"] for row in rows) / max(len(rows), 1), 3),
        "config_hash": stable_hash(stable_run_config | {"group_id": group_id}),
    }


def build_e10_analysis_md(
    run_dir: Path,
    summary_rows: list[dict[str, Any]],
    grouped_rows: dict[str, list[dict[str, Any]]],
    adapter_metadata: dict[str, Any],
) -> None:
    base_rows = {row["query_id"]: row for row in grouped_rows["A_base_4b_grounded"]}
    peft_rows = {row["query_id"]: row for row in grouped_rows["B_peft_4b_grounded"]}
    improved: list[dict[str, Any]] = []
    regressed: list[dict[str, Any]] = []
    for query_id in sorted(base_rows):
        base_row = base_rows[query_id]
        peft_row = peft_rows[query_id]
        delta = round(
            peft_row["verification"].citation_precision - base_row["verification"].citation_precision,
            4,
        )
        payload = {
            "query_id": query_id,
            "delta_citation_precision": delta,
            "base_recommendations": len(base_row["response"].recommendations),
            "peft_recommendations": len(peft_row["response"].recommendations),
            "base_summary": base_row["response"].summary or "",
            "peft_summary": peft_row["response"].summary or "",
        }
        if delta > 0:
            improved.append(payload)
        elif delta < 0:
            regressed.append(payload)

    lines = [
        "# E10 Base vs PEFT Result",
        "",
        "## Summary Table",
        "",
    ]
    lines.extend(markdown_table(summary_rows))
    lines.extend(
        [
            "",
            "## Adapter Metadata",
            "",
            f"- adapter_name: {adapter_metadata['adapter_name']}",
            f"- base_model_id: {adapter_metadata['base_model_id']}",
            f"- served_model_id: {adapter_metadata['served_model_id']}",
            f"- adapter_path: {adapter_metadata['_resolved_adapter_path']}",
            "",
            "## Representative Improvements",
            "",
        ]
    )
    if improved:
        for row in improved[:5]:
            lines.append(
                f"- `{row['query_id']}` | Δcitation_precision={row['delta_citation_precision']} | base_recs={row['base_recommendations']} | peft_recs={row['peft_recommendations']}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Representative Regressions", ""])
    if regressed:
        for row in regressed[:5]:
            lines.append(
                f"- `{row['query_id']}` | Δcitation_precision={row['delta_citation_precision']} | base_recs={row['base_recommendations']} | peft_recs={row['peft_recommendations']}"
            )
    else:
        lines.append("- none")

    (run_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def build_e9_query_rows(limit_queries: int | None = None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    judged_queries = load_jsonl(EXPERIMENT_ASSETS_DIR / "judged_queries.jsonl")
    slot_gold = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "slot_gold.jsonl")}
    clarify_gold = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "clarify_gold.jsonl")}
    eligible_ids = []
    for row in judged_queries:
        slot = slot_gold[row["query_id"]]
        clarify = clarify_gold[row["query_id"]]
        if clarify["clarify_needed"]:
            continue
        if not slot["city"] or not (slot["focus_aspects"] or slot["avoid_aspects"]):
            continue
        if row["query_type"] not in {"single_aspect", "multi_aspect", "focus_and_avoid", "multi_aspect_strong"}:
            continue
        eligible_ids.append((row, slot))
    if limit_queries is not None:
        eligible_ids = eligible_ids[:limit_queries]
    return eligible_ids


def build_split_hotel_lookup(split_manifest: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    by_split_city: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in split_manifest["hotels"]:
        by_split_city[row["split"]][row["city"]].append(row["hotel_id"])
    for split_lookup in by_split_city.values():
        for hotel_ids in split_lookup.values():
            hotel_ids.sort()
    return {split: dict(city_map) for split, city_map in by_split_city.items()}


def build_evidence_pack_for_candidate(
    collection,
    bi_encoder,
    normalize_embeddings: bool,
    evidence_lookup: dict[str, dict[str, Any]],
    preference: UserPreference,
    hotel: HotelCandidate,
    dense_top_k: int,
    final_top_k: int,
) -> EvidencePack:
    evidence_by_aspect: dict[str, list[SentenceCandidate]] = {}
    all_sentence_ids: list[str] = []
    retrieval_aspect_roles: dict[str, set[str]] = defaultdict(set)
    for aspect in preference.focus_aspects:
        retrieval_aspect_roles[aspect].add("focus")
    for aspect in preference.avoid_aspects:
        retrieval_aspect_roles[aspect].add("avoid")

    target_aspects = sorted(retrieval_aspect_roles)
    for aspect in target_aspects:
        aspect_dense_rows: list[dict[str, Any]] = []
        for role in sorted(retrieval_aspect_roles[aspect]):
            query_embedding = bi_encoder.encode(
                [build_query_en_target(preference.city or "", aspect, role)],
                normalize_embeddings=normalize_embeddings,
            ).tolist()
            aspect_dense_rows.extend(
                dense_query_hotel(
                    collection=collection,
                    query_embedding=query_embedding,
                    hotel_id=hotel.hotel_id,
                    city=preference.city or "",
                    hotel_name=hotel.hotel_name,
                    top_k=dense_top_k,
                    evidence_lookup=evidence_lookup,
                    aspect=aspect,
                    channel="main",
                )
            )
        dense_rows = merge_dense_candidates(aspect_dense_rows, top_k=dense_top_k)[:final_top_k]
        sentence_rows: list[SentenceCandidate] = []
        for row in dense_rows:
            sentence_rows.append(
                SentenceCandidate(
                    sentence_id=row["sentence_id"],
                    sentence_text=row["sentence_text"],
                    aspect=row["sentence_aspect"],
                    sentiment=row["sentence_sentiment"],
                    review_date=row["review_date"],
                    score_dense=row["score_dense"],
                    score_rerank=row["score_rerank"],
                )
            )
            all_sentence_ids.append(row["sentence_id"])
        evidence_by_aspect[aspect] = sentence_rows

    all_sentence_ids = sorted(set(all_sentence_ids))
    retrieval_trace = {
        "mode": E9_RETRIEVAL_MODE,
        "dense_top_k": dense_top_k,
        "final_top_k": final_top_k,
        "aspect_count": len(target_aspects),
        "aspect_roles": {aspect: sorted(roles) for aspect, roles in retrieval_aspect_roles.items()},
        "candidate_policy": E9_CANDIDATE_POLICY,
        "fallback_enabled": False,
    }
    return EvidencePack(
        hotel_id=hotel.hotel_id,
        query_en=preference.query_en,
        evidence_by_aspect=evidence_by_aspect,
        all_sentence_ids=all_sentence_ids,
        retrieval_trace=retrieval_trace,
    )


def freeze_e9_assets(limit_queries: int | None = None) -> tuple[Path, Path]:
    cfg = load_config()
    frozen_config = load_json(EXPERIMENT_ASSETS_DIR / "frozen_config.yaml")
    split_manifest = load_json(EXPERIMENT_ASSETS_DIR / "frozen_split_manifest.json")
    review_df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
    profile_df = pd.read_pickle("data/intermediate/hotel_profiles.pkl")
    evidence_df = pd.read_pickle("data/intermediate/evidence_index.pkl")

    hotel_summary = build_hotel_summary(review_df)
    profile_current, profile_alt = build_profile_tables(profile_df)
    del profile_alt
    hotel_summary = hotel_summary[hotel_summary["hotel_id"].isin(set(split_manifest["splits"]["test"]))].copy()
    evidence_lookup = build_evidence_lookup(evidence_df)
    query_rows = build_e9_query_rows(limit_queries=limit_queries)

    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer

    client = PersistentClient(path=cfg["embedding"]["chroma_persist_dir"])
    collection = client.get_collection(cfg["embedding"]["chroma_collection"])
    try:
        bi_encoder = SentenceTransformer(cfg["embedding"]["model"], local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "E9 资产冻结需要本地可用的 embedding 模型缓存；当前环境未能在离线模式下加载 "
            f"{cfg['embedding']['model']}。请先在有网环境缓存该模型后再重试。"
        ) from exc
    normalize_embeddings = bool(cfg["embedding"].get("normalize", True))

    stable_asset_config = {
        "task": "E9_assets",
        "split_config_hash": split_manifest["meta"]["config_hash"],
        "query_count": len(query_rows),
        "retrieval_mode": E9_RETRIEVAL_MODE,
        "candidate_policy": E9_CANDIDATE_POLICY,
        "dense_top_k": cfg["reranker"]["top_k_before_rerank"],
        "final_top_k": cfg["reranker"]["top_k_after_rerank"],
        "embedding_model": cfg["embedding"]["model"],
        "collection": cfg["embedding"]["chroma_collection"],
        "fallback_enabled": frozen_config["workflow"]["enable_fallback"],
        "behavior_base_model": frozen_config["behavior"]["base_model"],
    }

    units: list[dict[str, Any]] = []
    query_ids: list[str] = []
    for query_row, slot_row in query_rows:
        city_hotels = hotel_summary[hotel_summary["city"] == slot_row["city"]].copy()
        ranked = candidate_rank(
            city_hotels=city_hotels,
            profile_current=profile_current,
            profile_alt=profile_current,
            focus_aspects=slot_row["focus_aspects"],
            avoid_aspects=slot_row["avoid_aspects"],
            mode="B_final_aspect_score",
        ).head(5)
        candidate_hotels: list[HotelCandidate] = []
        for _, hotel_row in ranked.iterrows():
            candidate_hotels.append(
                HotelCandidate(
                    hotel_id=str(hotel_row["hotel_id"]),
                    hotel_name=str(hotel_row["hotel_name"]),
                    score_total=round(float(hotel_row["score_total"]), 4),
                    score_breakdown={k: float(v) for k, v in dict(hotel_row["score_breakdown"]).items()},
                )
            )

        preference = UserPreference(
            city=slot_row["city"],
            state=slot_row["state"],
            hotel_category=slot_row["hotel_category"],
            focus_aspects=slot_row["focus_aspects"],
            avoid_aspects=slot_row["avoid_aspects"],
            unsupported_requests=slot_row["unsupported_requests"],
            query_en=slot_row["query_en"],
        )
        evidence_packs = [
            build_evidence_pack_for_candidate(
                collection=collection,
                bi_encoder=bi_encoder,
                normalize_embeddings=normalize_embeddings,
                evidence_lookup=evidence_lookup,
                preference=preference,
                hotel=hotel,
                dense_top_k=cfg["reranker"]["top_k_before_rerank"],
                final_top_k=cfg["reranker"]["top_k_after_rerank"],
            )
            for hotel in candidate_hotels
        ]

        unit_payload = GenerationEvalUnit(
            query_id=query_row["query_id"],
            query_text_zh=query_row["query_text_zh"],
            query_type=query_row["query_type"],
            user_preference_gold=preference,
            unsupported_requests=preference.unsupported_requests,
            candidate_hotels=candidate_hotels,
            evidence_packs=evidence_packs,
            retrieval_mode=E9_RETRIEVAL_MODE,
            candidate_policy=E9_CANDIDATE_POLICY,
            config_hash=stable_hash(
                stable_asset_config
                | {
                    "query_id": query_row["query_id"],
                    "candidate_hotel_ids": [hotel.hotel_id for hotel in candidate_hotels],
                    "all_sentence_ids": [pack.all_sentence_ids for pack in evidence_packs],
                }
            ),
        )
        units.append(unit_payload.model_dump())
        query_ids.append(query_row["query_id"])

    units_path, query_ids_path = resolve_generation_asset_paths(limit_queries=limit_queries)
    write_jsonl(units_path, units)
    write_json(query_ids_path, query_ids)
    ensure_dir(E9_LABELS_DIR)
    return units_path, query_ids_path


def format_candidate_lines(unit: GenerationEvalUnit) -> list[str]:
    lines = []
    for hotel in unit.candidate_hotels:
        lines.append(
            f'- hotel_id={hotel.hotel_id} | hotel_name={hotel.hotel_name} | score_total={round(hotel.score_total, 4)}'
        )
    return lines


def format_evidence_pack_lines(unit: GenerationEvalUnit, grounded_mode: bool) -> list[str]:
    lines = []
    for pack in unit.evidence_packs:
        hotel_lookup = {hotel.hotel_id: hotel.hotel_name for hotel in unit.candidate_hotels}
        lines.append(f"Hotel {pack.hotel_id} | {hotel_lookup.get(pack.hotel_id, pack.hotel_id)}")
        shown_sentence_ids: list[str] = []
        for aspect in sorted(pack.evidence_by_aspect):
            lines.append(f"Aspect {aspect}:")
            sentence_rows = pack.evidence_by_aspect[aspect]
            sentence_limit = E9_PROMPT_SENTENCE_LIMIT_GROUNDED if grounded_mode else E9_PROMPT_SENTENCE_LIMIT_FREE
            sentence_rows = sentence_rows[:sentence_limit]
            for sentence in sentence_rows:
                shown_sentence_ids.append(sentence.sentence_id)
                lines.append(
                    f'- sentence_id={sentence.sentence_id} | text="{sentence.sentence_text}"'
                )
        if grounded_mode:
            lines.append(f"allowed_sentence_ids={','.join(shown_sentence_ids)}")
        lines.append("")
    return lines


def build_generation_prompts(unit: GenerationEvalUnit, group_id: str) -> tuple[str, str]:
    preference_payload = json.dumps(unit.user_preference_gold.model_dump(), ensure_ascii=False)
    candidate_lines = "\n".join(format_candidate_lines(unit))
    grounded_mode = group_id in {
        "B_grounded_generation",
        "C_grounded_generation_with_verifier",
        "A_base_4b_grounded",
        "B_peft_4b_grounded",
    }
    evidence_lines = "\n".join(format_evidence_pack_lines(unit, grounded_mode=grounded_mode))
    if group_id == "A_free_generation":
        system_prompt = (
            "你是酒店推荐工作流里的推荐解释生成器。\n"
            "你只返回 JSON，不要解释。\n"
            "统一输出 schema："
            f"{E9_OUTPUT_FIELDS}\n"
            f"最多返回 {E9_MAX_RECOMMENDATIONS} 家酒店；每家酒店最多 {E9_MAX_REASONS_PER_ITEM} 条理由。\n"
            "summary 只写一句短句，尽量不超过 50 个汉字。\n"
            "reason_text 必须短，不要复述整段证据。\n"
            "recommendation item 里只允许 hotel_id、hotel_name、reasons 三个字段。\n"
            "不要输出空的 reasons 数组；没有把握的酒店直接省略。\n"
            "unsupported_notice 只能出现在根级字段，不能出现在酒店项内部。\n"
            "A 组允许 sentence_id 为 null，但仍然必须优先基于给定证据摘要组织表达，不要编造酒店事实。"
        )
    else:
        system_prompt = (
            "你是证据约束的酒店推荐生成器。\n"
            "你只返回 JSON，不要解释。\n"
            "统一输出 schema："
            f"{E9_OUTPUT_FIELDS}\n"
            f"最多返回 {E9_MAX_RECOMMENDATIONS} 家酒店；每家酒店最多 {E9_MAX_REASONS_PER_ITEM} 条理由。\n"
            "summary 只写一句短句，尽量不超过 50 个汉字。\n"
            "reason_text 必须短，不要复述整段证据。\n"
            "recommendation item 里只允许 hotel_id、hotel_name、reasons 三个字段。\n"
            "不要输出空的 reasons 数组；没有至少 1 条可验证理由的酒店直接省略。\n"
            "unsupported_notice 只能出现在根级字段，不能出现在酒店项内部。\n"
            "每条理由必须包含单个 sentence_id，且该 sentence_id 必须来自当前 hotel 的 EvidencePack。\n"
            "如果给定证据不足以支持推荐，就减少推荐数量或在 unsupported_notice 中诚实说明。"
        )
    user_prompt = (
        f"Query ID: {unit.query_id}\n"
        f"用户中文需求: {unit.query_text_zh}\n"
        f"结构化偏好: {preference_payload}\n"
        f"Unsupported requests: {', '.join(unit.unsupported_requests) or 'none'}\n"
        "候选酒店如下：\n"
        f"{candidate_lines}\n\n"
        "当前证据如下：\n"
        f"{evidence_lines}\n"
        "请优先保留最有把握的 1-2 家酒店；宁可少推，不要凑满。"
    )
    return system_prompt, user_prompt


def build_retry_prompt(
    unit: GenerationEvalUnit,
    invalid_sentence_ids: list[str],
    out_of_pack_sentence_ids: list[str],
) -> str:
    messages = []
    if invalid_sentence_ids:
        messages.append(f"无效 sentence_id: {', '.join(sorted(set(invalid_sentence_ids)))}")
    if out_of_pack_sentence_ids:
        messages.append(f"越权 sentence_id: {', '.join(sorted(set(out_of_pack_sentence_ids)))}")
    issue_text = "；".join(messages) or "引用不合法"
    return (
        f"上一次输出的引用校验失败：{issue_text}。\n"
        f"请重新生成完整 JSON，并确保所有理由都只引用当前 EvidencePack 中真实存在的 sentence_id。\n"
        f"最多只保留 {E9_MAX_RECOMMENDATIONS} 家最有把握的酒店。\n"
        "不要输出空的 reasons 数组；没有证据就直接省略该酒店。\n"
        "unsupported_notice 只能放在根级字段。"
    )


def build_unsupported_notice(unsupported_requests: list[str]) -> str:
    if not unsupported_requests:
        return ""
    labels = [UNSUPPORTED_ZH_LABELS.get(item, item) for item in unsupported_requests]
    return f"当前系统不会直接执行{'、'.join(labels)}这类约束，我会只基于可验证的城市与评论证据给出推荐。"


def empty_generation_response(query_id: str, group_id: str, raw_response: str, schema_valid: bool) -> RecommendationResponse:
    return RecommendationResponse(
        query_id=query_id,
        group_id=group_id,
        summary="",
        recommendations=[],
        unsupported_notice="",
        schema_valid=schema_valid,
        raw_response=raw_response,
    )


def coerce_generation_payload(
    payload: dict[str, Any] | None,
    unit: GenerationEvalUnit,
    group_id: str,
    raw_response: str,
) -> RecommendationResponse:
    if payload is None:
        return empty_generation_response(unit.query_id, group_id, raw_response, False)

    candidate_lookup = {hotel.hotel_id: hotel for hotel in unit.candidate_hotels}
    summary = str(payload.get("summary", "") or "").strip()
    unsupported_notice = str(payload.get("unsupported_notice", "") or "").strip()
    notice_fragments = [unsupported_notice] if unsupported_notice else []
    recommendations_raw = payload.get("recommendations", [])
    schema_valid = True
    if not isinstance(recommendations_raw, list):
        recommendations_raw = []
        schema_valid = False

    recommendation_items: list[RecommendationItem] = []
    for raw_item in recommendations_raw[:E9_MAX_RECOMMENDATIONS]:
        if not isinstance(raw_item, dict):
            schema_valid = False
            continue
        hotel_id = str(raw_item.get("hotel_id", "") or "").strip()
        if hotel_id not in candidate_lookup:
            schema_valid = False
            continue
        hotel = candidate_lookup[hotel_id]
        reasons_raw = raw_item.get("reasons", [])
        item_notice = str(raw_item.get("unsupported_notice", "") or "").strip()
        if item_notice:
            notice_fragments.append(item_notice)
        if not isinstance(reasons_raw, list):
            reasons_raw = []
            schema_valid = False
        reasons: list[RecommendationReason] = []
        for raw_reason in reasons_raw[:E9_MAX_REASONS_PER_ITEM]:
            if not isinstance(raw_reason, dict):
                schema_valid = False
                continue
            aspects, _ = normalize_aspect_values([raw_reason.get("aspect")])
            aspect = aspects[0] if aspects else None
            reason_text = str(raw_reason.get("reason_text", "") or "").strip()
            sentence_id_raw = raw_reason.get("sentence_id")
            sentence_id = str(sentence_id_raw).strip() if sentence_id_raw not in {None, ""} else None
            if aspect is None or not reason_text:
                schema_valid = False
                continue
            if group_id != "A_free_generation" and sentence_id is None:
                schema_valid = False
            reasons.append(
                RecommendationReason(
                    aspect=aspect,
                    reason_text=reason_text,
                    sentence_id=sentence_id,
                )
            )
        if reasons:
            recommendation_items.append(
                RecommendationItem(
                    hotel_id=hotel.hotel_id,
                    hotel_name=hotel.hotel_name,
                    reasons=reasons,
                )
            )
        elif item_notice:
            continue
        else:
            schema_valid = False

    normalized_notice_parts: list[str] = []
    seen_notice_parts: set[str] = set()
    for part in notice_fragments:
        if not part or part in seen_notice_parts:
            continue
        normalized_notice_parts.append(part)
        seen_notice_parts.add(part)

    return RecommendationResponse(
        query_id=unit.query_id,
        group_id=group_id,
        summary=summary,
        recommendations=recommendation_items,
        unsupported_notice=" ".join(normalized_notice_parts),
        schema_valid=schema_valid,
        raw_response=raw_response,
    )


def verify_response_citations(
    response: RecommendationResponse,
    unit: GenerationEvalUnit,
    evidence_lookup: dict[str, dict[str, Any]],
    retry_triggered: bool = False,
    fallback_to_honest_notice: bool = False,
) -> tuple[CitationVerificationResult, list[dict[str, Any]]]:
    pack_lookup = {pack.hotel_id: set(pack.all_sentence_ids) for pack in unit.evidence_packs}
    invalid_sentence_ids: list[str] = []
    out_of_pack_sentence_ids: list[str] = []
    citation_count = 0
    valid_count = 0
    audit_rows: list[dict[str, Any]] = []

    for recommendation in response.recommendations:
        pack_sentence_ids = pack_lookup.get(recommendation.hotel_id, set())
        for reason in recommendation.reasons:
            sentence_id = reason.sentence_id
            citation_exists = 0
            in_current_pack = 0
            support_score = 0
            if sentence_id:
                citation_count += 1
                if sentence_id in evidence_lookup:
                    citation_exists = 1
                    if sentence_id in pack_sentence_ids:
                        in_current_pack = 1
                        valid_count += 1
                        evidence_row = evidence_lookup[sentence_id]
                        support_score = 2 if evidence_row["aspect"] == reason.aspect else 1
                    else:
                        out_of_pack_sentence_ids.append(sentence_id)
                else:
                    invalid_sentence_ids.append(sentence_id)
            audit_rows.append(
                {
                    "query_id": unit.query_id,
                    "group_id": response.group_id,
                    "hotel_id": recommendation.hotel_id,
                    "sentence_id": sentence_id or "",
                    "reason_text": reason.reason_text,
                    "citation_exists": citation_exists,
                    "in_current_evidence_pack": in_current_pack,
                    "support_score": support_score,
                    "notes": "",
                }
            )

    precision = 0.0 if citation_count == 0 else round(valid_count / citation_count, 4)
    verification = CitationVerificationResult(
        query_id=unit.query_id,
        group_id=response.group_id,
        citation_precision=precision,
        invalid_sentence_ids=sorted(set(invalid_sentence_ids)),
        out_of_pack_sentence_ids=sorted(set(out_of_pack_sentence_ids)),
        retry_triggered=retry_triggered,
        fallback_to_honest_notice=fallback_to_honest_notice,
    )
    return verification, audit_rows


def build_honest_fallback_response(
    unit: GenerationEvalUnit,
    group_id: str,
    raw_response: str,
) -> RecommendationResponse:
    return RecommendationResponse(
        query_id=unit.query_id,
        group_id=group_id,
        summary="当前证据引用未通过校验，因此暂不返回不可验证的酒店推荐。",
        recommendations=[],
        unsupported_notice=build_unsupported_notice(unit.unsupported_requests),
        schema_valid=True,
        raw_response=raw_response,
    )


def generate_group_response(
    llm_runner,
    unit: GenerationEvalUnit,
    group_id: str,
    max_new_tokens: int,
    evidence_lookup: dict[str, dict[str, Any]],
) -> tuple[RecommendationResponse, CitationVerificationResult, list[dict[str, Any]], dict[str, Any]]:
    system_prompt, user_prompt = build_generation_prompts(unit, group_id)
    raw_response = llm_runner.generate_json(system_prompt, user_prompt, max_new_tokens=max_new_tokens)
    payload, repaired = parse_json_with_repair(raw_response)
    response = coerce_generation_payload(payload, unit, group_id, raw_response)
    if repaired:
        response.schema_valid = False

    verification, audit_rows = verify_response_citations(response, unit, evidence_lookup)
    debug_payload: dict[str, Any] = {
        "raw_response_initial": raw_response,
        "retry_raw_response": "",
    }

    if group_id != "C_grounded_generation_with_verifier":
        return response, verification, audit_rows, debug_payload

    needs_retry = bool(verification.invalid_sentence_ids or verification.out_of_pack_sentence_ids)
    needs_retry = needs_retry or not response.schema_valid
    if not needs_retry:
        return response, verification, audit_rows, debug_payload

    retry_raw_response = raw_response
    for _ in range(E9_RETRY_LIMIT):
        retry_prompt = build_retry_prompt(
            unit,
            invalid_sentence_ids=verification.invalid_sentence_ids,
            out_of_pack_sentence_ids=verification.out_of_pack_sentence_ids,
        )
        retry_raw_response = llm_runner.generate_json(system_prompt, f"{user_prompt}\n\n{retry_prompt}", max_new_tokens=max_new_tokens)
        payload, repaired = parse_json_with_repair(retry_raw_response)
        retry_response = coerce_generation_payload(payload, unit, group_id, retry_raw_response)
        if repaired:
            retry_response.schema_valid = False
        retry_verification, retry_audit_rows = verify_response_citations(
            retry_response,
            unit,
            evidence_lookup,
            retry_triggered=True,
        )
        response = retry_response
        verification = retry_verification
        audit_rows = retry_audit_rows
        debug_payload["retry_raw_response"] = retry_raw_response
        if (
            response.schema_valid
            and not verification.invalid_sentence_ids
            and not verification.out_of_pack_sentence_ids
        ):
            return response, verification, audit_rows, debug_payload

    response = build_honest_fallback_response(unit, group_id, retry_raw_response)
    verification, audit_rows = verify_response_citations(
        response,
        unit,
        evidence_lookup,
        retry_triggered=True,
        fallback_to_honest_notice=True,
    )
    debug_payload["retry_raw_response"] = retry_raw_response
    return response, verification, audit_rows, debug_payload


def build_e9_metric_row(
    group_id: str,
    rows: list[dict[str, Any]],
    stable_run_config: dict[str, Any],
) -> dict[str, Any]:
    unsupported_rows = [row["unsupported_honesty"] for row in rows if row["unsupported_honesty"] is not None]
    support_scores = [
        audit_row["support_score"]
        for row in rows
        for audit_row in row["audit_rows"]
    ]
    return {
        "group_id": group_id,
        "query_count": len(rows),
        "citation_precision": round(
            sum(row["verification"].citation_precision for row in rows) / max(len(rows), 1),
            4,
        ),
        "evidence_verifiability_mean": round(sum(support_scores) / max(len(support_scores), 1), 4),
        "unsupported_honesty_rate": round(
            1.0 if not unsupported_rows else sum(unsupported_rows) / len(unsupported_rows),
            4,
        ),
        "schema_valid_rate": round(sum(int(row["response"].schema_valid) for row in rows) / max(len(rows), 1), 4),
        "avg_latency_ms": round(sum(row["latency_ms"] for row in rows) / max(len(rows), 1), 3),
        "retry_trigger_rate": round(
            sum(int(row["verification"].retry_triggered) for row in rows) / max(len(rows), 1),
            4,
        ),
        "fallback_to_honest_notice_rate": round(
            sum(int(row["verification"].fallback_to_honest_notice) for row in rows) / max(len(rows), 1),
            4,
        ),
        "config_hash": stable_hash(stable_run_config | {"group_id": group_id}),
    }


def select_representative_rows(group_rows: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    if kind == "drift":
        rows = [
            row for row in group_rows
            if row["verification"].invalid_sentence_ids or row["verification"].out_of_pack_sentence_ids
        ]
        if not rows:
            rows = [
                row for row in group_rows
                if row["response"].recommendations and row["verification"].citation_precision == 0
            ]
        return rows[:3]
    if kind == "positive":
        rows = [
            row for row in group_rows
            if row["response"].recommendations
            and row["verification"].citation_precision == 1.0
            and not row["verification"].fallback_to_honest_notice
        ]
        return rows[:3]
    rows = sorted(group_rows, key=lambda item: (-item["verification"].citation_precision, item["query_id"]))
    return rows[:3]


def build_e9_analysis_md(
    run_dir: Path,
    summary_rows: list[dict[str, Any]],
    grouped_rows: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# E9 Generation Constraint Result",
        "",
        "## Summary Table",
        "",
    ]
    lines.extend(markdown_table(summary_rows))
    lines.extend(["", "## Verifier Notes", ""])
    verifier_rows = grouped_rows.get("C_grounded_generation_with_verifier", [])
    retry_count = sum(int(row["verification"].retry_triggered) for row in verifier_rows)
    fallback_count = sum(int(row["verification"].fallback_to_honest_notice) for row in verifier_rows)
    lines.append(f"- verifier retry count: {retry_count}")
    lines.append(f"- honest fallback count: {fallback_count}")

    lines.extend(["", "## Schema Failures", ""])
    for group_id in E9_GROUPS:
        schema_failures = [row for row in grouped_rows[group_id] if not row["response"].schema_valid]
        lines.append(f"- {group_id}: {len(schema_failures)}")

    for group_id in E9_GROUPS:
        lines.extend(["", f"## {group_id}", ""])
        rep_rows = select_representative_rows(grouped_rows[group_id], kind="top")
        if rep_rows:
            for row in rep_rows:
                lines.append(
                    f"- `{row['query_id']}` | citation_precision={row['verification'].citation_precision} | summary={row['response'].summary or 'n/a'}"
                )
        else:
            lines.append("- none")

    lines.extend(["", "## Free Generation Citation Drift", ""])
    drift_rows = select_representative_rows(grouped_rows["A_free_generation"], kind="drift")
    if drift_rows:
        for row in drift_rows:
            lines.append(
                f"- `{row['query_id']}` | invalid={','.join(row['verification'].invalid_sentence_ids) or 'none'} | out_of_pack={','.join(row['verification'].out_of_pack_sentence_ids) or 'none'}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Grounded + Verifier Positive Cases", ""])
    positive_rows = select_representative_rows(grouped_rows["C_grounded_generation_with_verifier"], kind="positive")
    if positive_rows:
        for row in positive_rows:
            lines.append(
                f"- `{row['query_id']}` | recommendations={len(row['response'].recommendations)} | citation_precision={row['verification'].citation_precision}"
            )
    else:
        lines.append("- none")

    (run_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def run_e9_generation_constraints(
    output_root: Path,
    limit_queries: int | None = None,
) -> Path:
    cfg = load_config()
    frozen_config = load_json(EXPERIMENT_ASSETS_DIR / "frozen_config.yaml")
    behavior_runtime_config, behavior_api_key = resolve_behavior_runtime_config(cfg, frozen_config)
    split_manifest = load_json(EXPERIMENT_ASSETS_DIR / "frozen_split_manifest.json")
    eval_units_path = E9_UNITS_PATH
    if not eval_units_path.exists():
        freeze_e9_assets(limit_queries=None)
    eval_units = load_generation_eval_units(eval_units_path)
    if limit_queries is not None:
        eval_units = eval_units[:limit_queries]

    evidence_df = pd.read_pickle("data/intermediate/evidence_index.pkl")
    evidence_lookup = build_evidence_lookup(evidence_df)
    llm_runner = build_behavior_backend(behavior_runtime_config, behavior_api_key)
    generation_max_new_tokens = E9_GENERATION_MAX_NEW_TOKENS

    stable_run_config = {
        "task": "E9",
        "split_config_hash": split_manifest["meta"]["config_hash"],
        "query_count": len(eval_units),
        "retrieval_mode": E9_RETRIEVAL_MODE,
        "candidate_policy": E9_CANDIDATE_POLICY,
        "behavior_backend": behavior_runtime_config.llm_backend,
        "base_model_id": behavior_runtime_config.model_id,
        "behavior_api_base_url": behavior_runtime_config.api_base_url,
        "behavior_enable_thinking": behavior_runtime_config.enable_thinking,
        "behavior_temperature": behavior_runtime_config.temperature,
        "behavior_max_new_tokens": generation_max_new_tokens,
        "official_group_ids": E9_GROUPS,
        "eval_units_hash": stable_hash([unit.config_hash for unit in eval_units]),
        "fallback_enabled": False,
    }
    run_started_at = utc_now_iso()
    run_id = f"e9_{stable_hash(stable_run_config)}_{run_started_at.replace(':', '').replace('-', '')}"
    run_dir = ensure_dir(output_root / run_id)
    ensure_dir(E9_LABELS_DIR)

    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "generated_at": run_started_at,
                "stable_run_config": stable_run_config,
                "behavior_runtime_config": behavior_runtime_config.model_dump(),
                "selected_query_ids": [unit.query_id for unit in eval_units],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    log_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group_id in E9_GROUPS:
        for unit in eval_units:
            start = time.perf_counter()
            response, verification, response_audit_rows, debug_payload = generate_group_response(
                llm_runner=llm_runner,
                unit=unit,
                group_id=group_id,
                max_new_tokens=generation_max_new_tokens,
                evidence_lookup=evidence_lookup,
            )
            latency_ms = round((time.perf_counter() - start) * 1000, 3)
            unsupported_honesty = None
            if unit.unsupported_requests:
                unsupported_honesty = int(bool(response.unsupported_notice.strip()))

            grouped_entry = {
                "query_id": unit.query_id,
                "latency_ms": latency_ms,
                "response": response,
                "verification": verification,
                "audit_rows": response_audit_rows,
                "unsupported_honesty": unsupported_honesty,
            }
            grouped_rows[group_id].append(grouped_entry)
            audit_rows.extend(response_audit_rows)
            log_rows.append(
                RunLogEntry(
                    run_id=run_id,
                    group_id=group_id,
                    query_id=unit.query_id,
                    retrieval_mode=unit.retrieval_mode,
                    candidate_mode=E9_CANDIDATE_MODE,
                    config_hash=stable_hash(stable_run_config | {"group_id": group_id}),
                    latency_ms=latency_ms,
                    intermediate_objects={
                        "eval_unit": unit.model_dump(),
                        "response": response.model_dump(),
                        "citation_verification": verification.model_dump(),
                        "audit_rows": response_audit_rows,
                        "debug_payload": debug_payload,
                        "unsupported_honesty": unsupported_honesty,
                        "behavior_runtime_config": behavior_runtime_config.model_dump(),
                    },
                ).model_dump()
            )

    summary_rows = [
        build_e9_metric_row(group_id, grouped_rows[group_id], stable_run_config)
        for group_id in E9_GROUPS
    ]

    write_jsonl(run_dir / "results.jsonl", log_rows)
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(run_dir / "citation_verifiability_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(E9_LABELS_DIR / "citation_verifiability_audit.csv", index=False, encoding="utf-8-sig")
    build_e9_analysis_md(run_dir, summary_rows, grouped_rows)
    return run_dir


def assign_manifest_split(query_id: str) -> str:
    return "dev" if int(stable_hash({"query_id": query_id}), 16) % 5 == 0 else "train"


def build_feedback_update_pair(slot_row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    focus_aspects = list(slot_row["focus_aspects"])
    avoid_aspects = list(slot_row["avoid_aspects"])
    if focus_aspects:
        aspect = focus_aspects[0]
        prior = dict(slot_row)
        prior["focus_aspects"] = focus_aspects[1:]
        user_feedback = f"补充一下，请把{ASPECT_ZH_LABELS[aspect]}作为重点考虑方面。"
        target = slot_row
    elif avoid_aspects:
        aspect = avoid_aspects[0]
        prior = dict(slot_row)
        prior["avoid_aspects"] = avoid_aspects[1:]
        user_feedback = f"补充一下，请把{ASPECT_ZH_LABELS[aspect]}作为需要避免的方面。"
        target = slot_row
    else:
        return None
    return {
        "prior_preference": {
            "city": prior["city"],
            "hotel_category": prior["hotel_category"],
            "focus_aspects": prior["focus_aspects"],
            "avoid_aspects": prior["avoid_aspects"],
            "unsupported_requests": prior["unsupported_requests"],
        },
        "user_feedback": user_feedback,
    }, {
        "updated_preference": {
            "city": target["city"],
            "hotel_category": target["hotel_category"],
            "focus_aspects": target["focus_aspects"],
            "avoid_aspects": target["avoid_aspects"],
            "unsupported_requests": target["unsupported_requests"],
        }
    }


def build_sft_manifest_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    judged_queries = load_jsonl(EXPERIMENT_ASSETS_DIR / "judged_queries.jsonl")
    slot_gold = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "slot_gold.jsonl")}
    clarify_gold = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "clarify_gold.jsonl")}

    train_records: list[dict[str, Any]] = []
    dev_records: list[dict[str, Any]] = []
    manifest_config = {"task": "E10_manifest", "version": 1, "behavior_only": True}

    for query_row in judged_queries:
        query_id = query_row["query_id"]
        split = assign_manifest_split(query_id)
        bucket = train_records if split == "train" else dev_records
        slot_row = slot_gold[query_id]
        clarify_row = clarify_gold[query_id]

        preference_record = SFTManifestRecord(
            record_id=stable_hash({"task_type": "preference_parse", "query_id": query_id, "split": split}),
            split=split,
            task_type="preference_parse",
            hotel_id=None,
            query_id=query_id,
            source_asset="experiments/assets/slot_gold.jsonl",
            input_payload={"query_text_zh": query_row["query_text_zh"]},
            target_payload={
                "city": slot_row["city"],
                "hotel_category": slot_row["hotel_category"],
                "focus_aspects": slot_row["focus_aspects"],
                "avoid_aspects": slot_row["avoid_aspects"],
                "unsupported_requests": slot_row["unsupported_requests"],
            },
            config_hash=stable_hash(manifest_config | {"task_type": "preference_parse"}),
        )
        bucket.append(preference_record.model_dump())

        clarification_question = ""
        if clarify_row["clarify_needed"]:
            clarification_question = build_rule_clarification(
                query_row["query_text_zh"],
                {slot_row["city"]: slot_row["state"]} if slot_row["city"] and slot_row["state"] else {},
            ).question
        clarification_record = SFTManifestRecord(
            record_id=stable_hash({"task_type": "clarification", "query_id": query_id, "split": split}),
            split=split,
            task_type="clarification",
            hotel_id=None,
            query_id=query_id,
            source_asset="experiments/assets/clarify_gold.jsonl",
            input_payload={"query_text_zh": query_row["query_text_zh"]},
            target_payload={
                "clarify_needed": clarify_row["clarify_needed"],
                "clarify_reason": clarify_row["clarify_reason"],
                "target_slots": clarify_row["target_slots"],
                "question": clarification_question if clarify_row["clarify_needed"] else "",
            },
            config_hash=stable_hash(manifest_config | {"task_type": "clarification"}),
        )
        bucket.append(clarification_record.model_dump())

        if slot_row["unsupported_requests"]:
            honesty_record = SFTManifestRecord(
                record_id=stable_hash({"task_type": "constraint_honesty", "query_id": query_id, "split": split}),
                split=split,
                task_type="constraint_honesty",
                hotel_id=None,
                query_id=query_id,
                source_asset="experiments/assets/slot_gold.jsonl",
                input_payload={
                    "query_text_zh": query_row["query_text_zh"],
                    "unsupported_requests": slot_row["unsupported_requests"],
                },
                target_payload={
                    "unsupported_notice": build_unsupported_notice(slot_row["unsupported_requests"]),
                },
                config_hash=stable_hash(manifest_config | {"task_type": "constraint_honesty"}),
            )
            bucket.append(honesty_record.model_dump())

        feedback_pair = build_feedback_update_pair(slot_row)
        if feedback_pair is not None:
            input_payload, target_payload = feedback_pair
            feedback_record = SFTManifestRecord(
                record_id=stable_hash({"task_type": "feedback_update", "query_id": query_id, "split": split}),
                split=split,
                task_type="feedback_update",
                hotel_id=None,
                query_id=query_id,
                source_asset="synthetic_feedback_update_from_slot_gold",
                input_payload=input_payload,
                target_payload=target_payload,
                config_hash=stable_hash(manifest_config | {"task_type": "feedback_update"}),
            )
            bucket.append(feedback_record.model_dump())

    return train_records, dev_records


def prepare_e10_manifests() -> tuple[Path, Path]:
    train_records, dev_records = build_sft_manifest_records()
    write_jsonl(SFT_TRAIN_MANIFEST_PATH, train_records)
    write_jsonl(SFT_DEV_MANIFEST_PATH, dev_records)
    return SFT_TRAIN_MANIFEST_PATH, SFT_DEV_MANIFEST_PATH


def run_e10_base_vs_peft(output_root: Path, limit_queries: int | None = None) -> Path:
    cfg = load_config()
    frozen_config = load_json(EXPERIMENT_ASSETS_DIR / "frozen_config.yaml")
    split_manifest = load_json(EXPERIMENT_ASSETS_DIR / "frozen_split_manifest.json")
    base_runtime_config, behavior_api_key = resolve_behavior_runtime_config(cfg, frozen_config)
    frozen_base_model_id = str(frozen_config["behavior"]["base_model"])
    if base_runtime_config.model_id != frozen_base_model_id:
        raise ValueError(
            "E10 固定要求 Base 组使用冻结的正式行为模型 "
            f"{frozen_base_model_id}；当前解析到的 model_id 为 {base_runtime_config.model_id}。"
        )

    if not E9_UNITS_PATH.exists():
        raise FileNotFoundError(
            f"Missing frozen E9 eval units: {E9_UNITS_PATH}. 请先完成 E9 资产冻结。"
        )
    if not SFT_TRAIN_MANIFEST_PATH.exists() or not SFT_DEV_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Missing E10 SFT manifests. 请先运行 `python -m scripts.evaluation.run_experiment_suite --task e10_prepare_manifests`。"
        )
    if not base_runtime_config.adapter_metadata_path:
        raise ValueError(
            "运行 e10_base_vs_peft 需要提供 adapter metadata。"
            "请设置 BEHAVIOR_ADAPTER_METADATA_PATH 或在 behavior 配置中提供 adapter_metadata_path。"
        )

    eval_units = load_generation_eval_units(E9_UNITS_PATH)
    if limit_queries is not None:
        eval_units = eval_units[:limit_queries]
    evidence_df = pd.read_pickle("data/intermediate/evidence_index.pkl")
    evidence_lookup = build_evidence_lookup(evidence_df)

    adapter_metadata = load_adapter_metadata(base_runtime_config.adapter_metadata_path)
    if str(adapter_metadata["base_model_id"]) != frozen_base_model_id:
        raise ValueError(
            "Adapter metadata 中的 base_model_id 与当前冻结主模型不一致："
            f"{adapter_metadata['base_model_id']} != {frozen_base_model_id}"
        )
    peft_runtime_config = build_peft_runtime_config(base_runtime_config, adapter_metadata)
    base_runtime_config = base_runtime_config.model_copy(
        update={
            "use_peft_adapter": False,
            "adapter_path": None,
            "adapter_metadata_path": None,
            "max_new_tokens": E9_GENERATION_MAX_NEW_TOKENS,
        }
    )
    peft_runtime_config = peft_runtime_config.model_copy(
        update={"max_new_tokens": E9_GENERATION_MAX_NEW_TOKENS}
    )

    group_runtime_configs = {
        "A_base_4b_grounded": base_runtime_config,
        "B_peft_4b_grounded": peft_runtime_config,
    }
    group_runners = {
        group_id: build_behavior_backend(runtime_config, behavior_api_key)
        for group_id, runtime_config in group_runtime_configs.items()
    }

    stable_run_config = {
        "task": "E10",
        "split_config_hash": split_manifest["meta"]["config_hash"],
        "query_count": len(eval_units),
        "retrieval_mode": E9_RETRIEVAL_MODE,
        "candidate_policy": E9_CANDIDATE_POLICY,
        "base_model_id": frozen_base_model_id,
        "peft_model_id": peft_runtime_config.model_id,
        "behavior_backend": base_runtime_config.llm_backend,
        "behavior_api_base_url": base_runtime_config.api_base_url,
        "behavior_enable_thinking": base_runtime_config.enable_thinking,
        "behavior_temperature": base_runtime_config.temperature,
        "behavior_max_new_tokens": E9_GENERATION_MAX_NEW_TOKENS,
        "official_group_ids": E10_GROUPS,
        "eval_units_hash": stable_hash([unit.config_hash for unit in eval_units]),
        "adapter_metadata_hash": stable_hash(
            {k: v for k, v in adapter_metadata.items() if not str(k).startswith("_")}
        ),
        "fallback_enabled": False,
    }
    run_started_at = utc_now_iso()
    run_id = f"e10_{stable_hash(stable_run_config)}_{run_started_at.replace(':', '').replace('-', '')}"
    run_dir = ensure_dir(output_root / run_id)
    ensure_dir(E9_LABELS_DIR)

    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "generated_at": run_started_at,
                "stable_run_config": stable_run_config,
                "selected_query_ids": [unit.query_id for unit in eval_units],
                "base_runtime_config": base_runtime_config.model_dump(),
                "peft_runtime_config": peft_runtime_config.model_dump(),
                "adapter_metadata": {k: v for k, v in adapter_metadata.items() if not str(k).startswith("_")},
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    log_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group_id in E10_GROUPS:
        llm_runner = group_runners[group_id]
        runtime_config = group_runtime_configs[group_id]
        for unit in eval_units:
            start = time.perf_counter()
            response, verification, response_audit_rows, debug_payload = generate_group_response(
                llm_runner=llm_runner,
                unit=unit,
                group_id=group_id,
                max_new_tokens=E9_GENERATION_MAX_NEW_TOKENS,
                evidence_lookup=evidence_lookup,
            )
            latency_ms = round((time.perf_counter() - start) * 1000, 3)
            unsupported_honesty = None
            if unit.unsupported_requests:
                unsupported_honesty = int(bool(response.unsupported_notice.strip()))

            grouped_entry = {
                "query_id": unit.query_id,
                "latency_ms": latency_ms,
                "response": response,
                "verification": verification,
                "audit_rows": response_audit_rows,
                "unsupported_honesty": unsupported_honesty,
            }
            grouped_rows[group_id].append(grouped_entry)
            audit_rows.extend(response_audit_rows)
            log_rows.append(
                RunLogEntry(
                    run_id=run_id,
                    group_id=group_id,
                    query_id=unit.query_id,
                    retrieval_mode=unit.retrieval_mode,
                    candidate_mode=E10_CANDIDATE_MODE,
                    config_hash=stable_hash(stable_run_config | {"group_id": group_id}),
                    latency_ms=latency_ms,
                    intermediate_objects={
                        "eval_unit": unit.model_dump(),
                        "response": response.model_dump(),
                        "citation_verification": verification.model_dump(),
                        "audit_rows": response_audit_rows,
                        "debug_payload": debug_payload,
                        "unsupported_honesty": unsupported_honesty,
                        "behavior_runtime_config": runtime_config.model_dump(),
                    },
                ).model_dump()
            )

    summary_rows = [
        build_e10_metric_row(group_id, grouped_rows[group_id], stable_run_config)
        for group_id in E10_GROUPS
    ]

    write_jsonl(run_dir / "results.jsonl", log_rows)
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(run_dir / "citation_verifiability_audit.csv", index=False, encoding="utf-8-sig")
    build_e10_analysis_md(run_dir, summary_rows, grouped_rows, adapter_metadata)
    return run_dir

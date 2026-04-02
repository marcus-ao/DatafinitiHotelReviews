"""Shared behavior evaluation engine for E3, E4, and E5."""

from __future__ import annotations

import argparse
import inspect
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.evaluation.evaluate_e6_e8_retrieval import (
    build_city_test_hotels,
    build_evidence_lookup,
    build_target_units,
    evaluate_ranked_rows,
    load_qrels_lookup,
    markdown_table,
    retrieve_official_mode,
)
from scripts.shared.experiment_schemas import (
    BehaviorRuntimeConfig,
    BridgeQueryRecord,
    ClarificationDecision,
    PreferenceParseResult,
    RunLogEntry,
    UserPreference,
)
from scripts.shared.behavior_postprocess import (
    load_query_ids_from_file,
    normalize_aspect_values,
    normalize_city_value,
    normalize_decision_label,
    normalize_unsupported_values,
    parse_payload_bool,
)
from scripts.shared.behavior_runtime import flatten_openai_content, resolve_behavior_runtime_config
from scripts.shared.experiment_utils import (
    E4_LABELS_DIR,
    E6_LABELS_DIR,
    EXPERIMENT_ASSETS_DIR,
    EXPERIMENT_RUNS_DIR,
    ensure_dir,
    load_jsonl,
    stable_hash,
    utc_now_iso,
    write_jsonl,
)
from scripts.shared.project_utils import ASPECT_CATEGORIES, load_config


E3_GROUPS = ["A_rule_parser", "B_base_llm_structured"]
E4_GROUPS = ["A_rule_clarify", "B_base_llm_clarify"]
E5_GROUPS = ["A_zh_direct_dense_no_rerank", "B_structured_query_en_dense_no_rerank"]
BEHAVIOR_CANDIDATE_MODE = "behavior_only"
ASPECT_ZH_TERMS = {
    "location_transport": ["位置交通", "位置", "交通"],
    "cleanliness": ["卫生干净", "卫生", "干净"],
    "service": ["服务"],
    "room_facilities": ["房间设施", "房间", "设施"],
    "quiet_sleep": ["安静睡眠", "住得安静", "安静一点", "安静", "睡眠"],
    "value": ["性价比"],
}

UNSUPPORTED_PATTERNS = {
    "budget": [r"预算", r"\d+\s*元", r"便宜", r"价格"],
    "distance_to_landmark": [r"离景点", r"步行\s*\d+\s*分钟", r"景点", r"地标", r"附近"],
    "checkin_date": [r"入住", r"下周", r"周[一二三四五六日天]", r"\d+月\d+日"],
}

REASONING_LEAK_PREFIXES = (
    "Thinking Process:",
    "Reasoning:",
    "Thought Process:",
    "Chain of Thought:",
)


class BehaviorLLMBackend(Protocol):
    runtime_config: BehaviorRuntimeConfig

    def generate_json(self, system_prompt: str, user_prompt: str, max_new_tokens: int | None = None) -> str:
        ...


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        if str(path).endswith((".yaml", ".yml")):
            return yaml.safe_load(handle)
        return json.load(handle)


def build_query_en_from_slots(
    city: str | None,
    focus_aspects: list[str],
    avoid_aspects: list[str],
    unsupported_requests: list[str],
) -> str:
    from scripts.evaluation.evaluate_e6_e8_retrieval import ASPECT_EN, ASPECT_EN_AVOID

    parts: list[str] = []
    parts.append(f"hotel in {city}" if city else "hotel")
    if focus_aspects:
        parts.append("with " + " and ".join(ASPECT_EN[aspect] for aspect in focus_aspects))
    if avoid_aspects:
        parts.append("avoiding " + " and ".join(ASPECT_EN_AVOID[aspect] for aspect in avoid_aspects))
    if "budget" in unsupported_requests:
        parts.append("with a budget requirement")
    if "distance_to_landmark" in unsupported_requests:
        parts.append("close to a landmark")
    if "checkin_date" in unsupported_requests:
        parts.append("for a specific check-in date")
    return " ".join(parts)


def stable_sorted_unique(values: list[str]) -> list[str]:
    return sorted(set(values))


def detect_city(query_text: str, allowed_cities: list[str]) -> str | None:
    for city in sorted(allowed_cities, key=len, reverse=True):
        if city in query_text:
            return city
    return None


def detect_unsupported_requests(query_text: str) -> list[str]:
    detected = []
    for request_name, patterns in UNSUPPORTED_PATTERNS.items():
        if any(re.search(pattern, query_text, flags=re.IGNORECASE) for pattern in patterns):
            detected.append(request_name)
    return stable_sorted_unique(detected)


def find_aspects_in_text(text: str) -> list[str]:
    found = []
    normalized = text.strip()
    if not normalized:
        return found
    for aspect in ASPECT_CATEGORIES:
        terms = sorted(ASPECT_ZH_TERMS[aspect], key=len, reverse=True)
        if any(term in normalized for term in terms):
            found.append(aspect)
    return stable_sorted_unique(found)


def split_focus_and_avoid_segments(query_text: str) -> tuple[str, str]:
    for marker in ["但不要", "但是不要", "但又最好别太强调", "但别"]:
        if marker in query_text:
            left, right = query_text.split(marker, 1)
            return left, marker + right
    return query_text, ""


def parse_rule_preference(query_text: str, city_to_state: dict[str, str]) -> UserPreference:
    city = detect_city(query_text, list(city_to_state))
    unsupported_requests = detect_unsupported_requests(query_text)
    focus_segment, avoid_segment = split_focus_and_avoid_segments(query_text)
    focus_aspects = find_aspects_in_text(focus_segment)
    avoid_aspects = find_aspects_in_text(avoid_segment)

    if "quiet_sleep" not in focus_aspects and re.search(r"住得安静|安静一点", focus_segment):
        focus_aspects.append("quiet_sleep")
    if "quiet_sleep" not in avoid_aspects and re.search(r"安静睡眠.*太差", avoid_segment):
        avoid_aspects.append("quiet_sleep")

    focus_aspects = stable_sorted_unique(focus_aspects)
    avoid_aspects = stable_sorted_unique(avoid_aspects)
    return UserPreference(
        city=city,
        state=city_to_state.get(city),
        hotel_category=None,
        focus_aspects=focus_aspects,
        avoid_aspects=avoid_aspects,
        unsupported_requests=unsupported_requests,
        query_en=build_query_en_from_slots(city, focus_aspects, avoid_aspects, unsupported_requests),
    )


def build_rule_clarification(query_text: str, city_to_state: dict[str, str]) -> ClarificationDecision:
    preference = parse_rule_preference(query_text, city_to_state)
    overlap = sorted(set(preference.focus_aspects) & set(preference.avoid_aspects))
    if not preference.city:
        question = "请先告诉我你想入住哪个城市？"
        return ClarificationDecision(
            query_id="",
            group_id="A_rule_clarify",
            clarify_needed=True,
            clarify_reason="missing_city",
            target_slots=["city"],
            question=question,
            schema_valid=True,
            raw_response=question,
        )
    if overlap:
        overlap_text = "、".join(overlap)
        question = f"你同时把 {overlap_text} 设为重点和避免项，能再明确你真正更在意哪一边吗？"
        return ClarificationDecision(
            query_id="",
            group_id="A_rule_clarify",
            clarify_needed=True,
            clarify_reason="aspect_conflict",
            target_slots=["focus_aspects", "avoid_aspects"],
            question=question,
            schema_valid=True,
            raw_response=question,
        )
    return ClarificationDecision(
        query_id="",
        group_id="A_rule_clarify",
        clarify_needed=False,
        clarify_reason="",
        target_slots=[],
        question="",
        schema_valid=True,
        raw_response="",
    )


def clean_json_candidate(raw_text: str) -> str:
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def parse_json_with_repair(raw_text: str) -> tuple[dict[str, Any] | None, bool]:
    candidate = clean_json_candidate(raw_text)
    try:
        return json.loads(candidate), False
    except json.JSONDecodeError:
        repaired = candidate.replace("\n", " ")
        repaired = re.sub(r"\s+", " ", repaired)
        try:
            return json.loads(repaired), True
        except json.JSONDecodeError:
            return None, False


def build_behavior_backend(
    runtime_config: BehaviorRuntimeConfig,
    api_key: str | None,
) -> BehaviorLLMBackend:
    if runtime_config.llm_backend == "api":
        return OpenAICompatibleModel(runtime_config, api_key)
    return LocalBaseModel(runtime_config)


def detect_reasoning_leak(raw_text: str) -> str | None:
    stripped = (raw_text or "").lstrip()
    for prefix in REASONING_LEAK_PREFIXES:
        if stripped.startswith(prefix):
            return "reasoning_leak"
    return None


def prepare_chat_template_tensors(
    tokenizer,
    messages: list[dict[str, str]],
    device: str,
    enable_thinking: bool | None = None,
):
    apply_kwargs = {
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    thinking_control_supported = False
    if enable_thinking is not None:
        try:
            signature = inspect.signature(tokenizer.apply_chat_template)
            accepts_enable_thinking = "enable_thinking" in signature.parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        except (TypeError, ValueError):
            accepts_enable_thinking = True

        if accepts_enable_thinking:
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    **apply_kwargs,
                    enable_thinking=enable_thinking,
                )
                thinking_control_supported = True
            except TypeError:
                rendered = tokenizer.apply_chat_template(messages, **apply_kwargs)
        else:
            rendered = tokenizer.apply_chat_template(messages, **apply_kwargs)
    else:
        rendered = tokenizer.apply_chat_template(messages, **apply_kwargs)
    if hasattr(rendered, "to"):
        rendered = rendered.to(device)
    if hasattr(rendered, "__getitem__") and hasattr(rendered, "get") and "input_ids" in rendered:
        input_ids = rendered["input_ids"]
        attention_mask = rendered.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask, thinking_control_supported

    input_ids = rendered
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, thinking_control_supported


class LocalBaseModel:
    def __init__(self, runtime_config: BehaviorRuntimeConfig) -> None:
        self.runtime_config = runtime_config
        self.model_id = runtime_config.model_id
        self.last_generation_debug: dict[str, Any] = {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            dtype = torch.float16
        else:
            self.device = "cpu"
            dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

    def generate_json(self, system_prompt: str, user_prompt: str, max_new_tokens: int | None = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_ids, attention_mask, thinking_control_supported = prepare_chat_template_tensors(
            self.tokenizer,
            messages,
            self.device,
            enable_thinking=self.runtime_config.enable_thinking,
        )
        target_max_tokens = max_new_tokens or self.runtime_config.max_new_tokens
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": target_max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.runtime_config.temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.runtime_config.temperature
        else:
            generate_kwargs["do_sample"] = False
        with torch.inference_mode():
            output = self.model.generate(**generate_kwargs)
        generated = output[0][input_ids.shape[-1] :]
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        response_error_type = detect_reasoning_leak(decoded)
        self.last_generation_debug = {
            "response_error_type": response_error_type,
            "thinking_control_supported": thinking_control_supported,
            "raw_response_prefix": decoded[:80],
        }
        return decoded


class OpenAICompatibleModel:
    def __init__(self, runtime_config: BehaviorRuntimeConfig, api_key: str | None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "当前环境缺少 openai 依赖，请先执行 `pip install openai` 或 `pip install -r requirements.txt`。"
            ) from exc

        self.runtime_config = runtime_config
        self.model_id = runtime_config.model_id
        self.last_generation_debug: dict[str, Any] = {}
        self.client = OpenAI(
            api_key=api_key or "EMPTY",
            base_url=runtime_config.api_base_url,
            timeout=runtime_config.api_timeout_seconds,
        )

    def generate_json(self, system_prompt: str, user_prompt: str, max_new_tokens: int | None = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            temperature=self.runtime_config.temperature,
            max_tokens=max_new_tokens or self.runtime_config.max_new_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": self.runtime_config.enable_thinking,
                }
            },
        )
        flattened = flatten_openai_content(response.choices[0].message.content)
        self.last_generation_debug = {
            "response_error_type": detect_reasoning_leak(flattened),
            "thinking_control_supported": True,
            "raw_response_prefix": flattened[:80],
        }
        return flattened


def select_query_rows(
    judged_queries: list[dict[str, Any]],
    limit_queries: int | None = None,
    query_id_file: str | Path | None = None,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    if query_id_file:
        requested_ids = load_query_ids_from_file(query_id_file)
        row_lookup = {row["query_id"]: row for row in judged_queries}
        selected_rows = [row_lookup[query_id] for query_id in requested_ids if query_id in row_lookup]
        return selected_rows, f"query_id_file:{Path(query_id_file).name}", requested_ids

    if limit_queries:
        selected_rows = judged_queries[:limit_queries]
        return selected_rows, f"head_{limit_queries}_queries", [row["query_id"] for row in selected_rows]

    return judged_queries, "all_86_queries", [row["query_id"] for row in judged_queries]


def build_preference_prompts_v1(
    query_text: str,
    city_to_state: dict[str, str],
) -> tuple[str, str]:
    city_lines = ", ".join(f"{city}:{state}" for city, state in sorted(city_to_state.items()))
    system_prompt = (
        "You are a structured preference parser for a hotel recommendation workflow. "
        "Return JSON only. Do not output explanations or markdown. "
        "Allowed aspects: location_transport, cleanliness, service, room_facilities, quiet_sleep, value. "
        "Allowed unsupported_requests: budget, distance_to_landmark, checkin_date. "
        "Allowed cities and states: "
        f"{city_lines}. "
        "When the city is missing, set city/state to null. "
        "hotel_category must be null unless the user explicitly asks for a category. "
        "Use arrays for focus_aspects, avoid_aspects, unsupported_requests. "
        "query_en must be a concise English retrieval summary."
    )
    user_prompt = (
        "Extract the user preference from the following Chinese query.\n"
        "Return this JSON schema exactly:\n"
        '{"city": null, "state": null, "hotel_category": null, "focus_aspects": [], '
        '"avoid_aspects": [], "unsupported_requests": [], "query_en": ""}\n'
        f"Query: {query_text}"
    )
    return system_prompt, user_prompt


def build_preference_prompts_v2(
    query_text: str,
    allowed_cities: list[str],
) -> tuple[str, str]:
    city_text = "、".join(sorted(allowed_cities))
    system_prompt = (
        "你是酒店推荐工作流里的结构化偏好解析器。\n"
        "你只返回 JSON，不要解释，不要 markdown，不要补充说明。\n"
        "你只抽取 5 个字段：city、hotel_category、focus_aspects、avoid_aspects、unsupported_requests。\n"
        "city 只能从这 10 个城市里选一个，或者填 null："
        f"{city_text}。\n"
        "focus_aspects 和 avoid_aspects 只能使用 6 个 canonical id："
        "location_transport, cleanliness, service, room_facilities, quiet_sleep, value。\n"
        "unsupported_requests 只能使用 3 个 canonical id：budget, distance_to_landmark, checkin_date。\n"
        "预算、离景点距离、入住日期都属于 unsupported，不要把它们吸收到 value 或 location_transport 里。\n"
        "如果城市缺失，city 填 null；不要输出 Anaheim:CA 这类 city:state 形式。"
    )
    user_prompt = (
        "请抽取下面中文酒店需求的结构化槽位。\n"
        "输出 JSON schema：\n"
        '{"city": null, "hotel_category": null, "focus_aspects": [], "avoid_aspects": [], "unsupported_requests": []}\n'
        "下面是示例：\n"
        '示例1 Query: 我想在Anaheim找一家位置交通比较好的酒店。\n'
        '示例1 JSON: {"city":"Anaheim","hotel_category":null,"focus_aspects":["location_transport"],"avoid_aspects":[],"unsupported_requests":[]}\n'
        '示例2 Query: 帮我找Anaheim预算在 600 元以内，而且位置交通不错的酒店。\n'
        '示例2 JSON: {"city":"Anaheim","hotel_category":null,"focus_aspects":["location_transport"],"avoid_aspects":[],"unsupported_requests":["budget"]}\n'
        '示例3 Query: 我想在Anaheim找一家离景点步行 10 分钟内、而且卫生干净好的酒店。\n'
        '示例3 JSON: {"city":"Anaheim","hotel_category":null,"focus_aspects":["cleanliness"],"avoid_aspects":[],"unsupported_requests":["distance_to_landmark"]}\n'
        '示例4 Query: 我想去Chicago，要求下周五能入住，同时性价比也要好。\n'
        '示例4 JSON: {"city":"Chicago","hotel_category":null,"focus_aspects":["value"],"avoid_aspects":[],"unsupported_requests":["checkin_date"]}\n'
        '示例5 Query: 我在Anaheim想住得安静一点，但不要服务太差的酒店。\n'
        '示例5 JSON: {"city":"Anaheim","hotel_category":null,"focus_aspects":["quiet_sleep"],"avoid_aspects":["service"],"unsupported_requests":[]}\n'
        f"现在请处理这条 Query: {query_text}"
    )
    return system_prompt, user_prompt


def build_preference_prompts(
    query_text: str,
    allowed_cities: list[str],
    city_to_state: dict[str, str],
    prompt_version_id: str,
) -> tuple[str, str]:
    if prompt_version_id == "e3_v1_structured_preference":
        return build_preference_prompts_v1(query_text, city_to_state)
    if prompt_version_id == "e3_v2_cn_slots_only":
        return build_preference_prompts_v2(query_text, allowed_cities)
    raise ValueError(f"Unsupported E3 prompt_version_id: {prompt_version_id}")


def build_clarification_prompts_v1(
    query_text: str,
    allowed_cities: list[str],
) -> tuple[str, str]:
    system_prompt = (
        "You decide whether a hotel recommendation workflow needs one clarification question. "
        "Return JSON only. Allowed clarify_reason values: '', 'missing_city', 'aspect_conflict'. "
        "Allowed target_slots values: city, focus_aspects, avoid_aspects. "
        "Ask at most one short Chinese question. "
        "Do not ask for unsupported requests such as budget, distance_to_landmark, or checkin_date "
        "when the query already has executable city/aspect information."
    )
    user_prompt = (
        "Given the following Chinese hotel query, decide whether clarification is needed.\n"
        "Return this JSON schema exactly:\n"
        '{"clarify_needed": false, "clarify_reason": "", "target_slots": [], "question": ""}\n'
        f"Allowed cities: {', '.join(allowed_cities)}\n"
        f"Query: {query_text}"
    )
    return system_prompt, user_prompt


def build_clarification_prompts_v2(
    query_text: str,
    allowed_cities: list[str],
) -> tuple[str, str]:
    city_text = "、".join(sorted(allowed_cities))
    system_prompt = (
        "你是酒店推荐工作流里的澄清触发分类器。\n"
        "你要先判断 decision_label，再决定是否需要提问。\n"
        "你只返回 JSON，不要解释。\n"
        "decision_label 只能是 missing_city、aspect_conflict、none 三者之一。\n"
        "规则固定如下：\n"
        "1. 只要缺少城市，就必须输出 missing_city。\n"
        "2. 只要同一方面同时出现在关注项和避免项，就必须输出 aspect_conflict。\n"
        "3. 预算、离景点距离、入住日期属于 unsupported；只要 query 仍然有可执行的城市和方面，就输出 none，不要为 unsupported 追问。\n"
        f"允许城市只有：{city_text}。"
    )
    user_prompt = (
        "请根据规则判断下面中文需求是否需要澄清。\n"
        "输出 JSON schema：\n"
        '{"decision_label":"none","question":""}\n'
        "下面是示例：\n"
        '示例1 Query: 我想找一家位置交通好的酒店，你先帮我想想。\n'
        '示例1 JSON: {"decision_label":"missing_city","question":"请先告诉我你想入住哪个城市？"}\n'
        '示例2 Query: 我想找一家卫生干净好的酒店，你先帮我想想。\n'
        '示例2 JSON: {"decision_label":"missing_city","question":"请先告诉我你想入住哪个城市？"}\n'
        '示例3 Query: 我想在Anaheim找一家位置交通很好，但又最好别太强调位置交通的酒店。\n'
        '示例3 JSON: {"decision_label":"aspect_conflict","question":"你是更看重位置交通，还是希望不要太强调位置交通？请再明确一下。"}\n'
        '示例4 Query: 我想在Atlanta找一家卫生干净很好，但又最好别太强调卫生干净的酒店。\n'
        '示例4 JSON: {"decision_label":"aspect_conflict","question":"你是更看重卫生干净，还是希望不要太强调卫生干净？请再明确一下。"}\n'
        '示例5 Query: 帮我找Anaheim预算在 600 元以内，而且位置交通不错的酒店。\n'
        '示例5 JSON: {"decision_label":"none","question":""}\n'
        '示例6 Query: 我在Anaheim想住得安静一点，但不要服务太差的酒店。\n'
        '示例6 JSON: {"decision_label":"none","question":""}\n'
        f"现在请处理这条 Query: {query_text}"
    )
    return system_prompt, user_prompt


def build_clarification_prompts(
    query_text: str,
    allowed_cities: list[str],
    prompt_version_id: str,
) -> tuple[str, str]:
    if prompt_version_id == "e4_v1_clarify_decision":
        return build_clarification_prompts_v1(query_text, allowed_cities)
    if prompt_version_id == "e4_v2_cn_decision_label_fewshot":
        return build_clarification_prompts_v2(query_text, allowed_cities)
    raise ValueError(f"Unsupported E4 prompt_version_id: {prompt_version_id}")


def coerce_preference_payload(
    payload: dict[str, Any] | None,
    query_text: str,
    city_to_state: dict[str, str],
) -> tuple[UserPreference, list[str], bool]:
    error_types: list[str] = []
    if payload is None:
        error_types.append("json_parse_failed")
        return parse_rule_preference(query_text, city_to_state), error_types, False

    raw_city = payload.get("city")
    city = normalize_city_value(raw_city, city_to_state)
    raw_city_text = str(raw_city).strip().lower() if raw_city is not None else ""
    if city is None and raw_city_text not in {"", "null", "none"}:
        error_types.append("unknown_city_value")
    state = city_to_state.get(city)

    hotel_category = payload.get("hotel_category")
    if isinstance(hotel_category, str) and hotel_category.strip().lower() in {"", "null", "none"}:
        hotel_category = None

    focus_aspects, focus_unknown = normalize_aspect_values(payload.get("focus_aspects", []))
    if focus_aspects is None:
        error_types.append("invalid_aspect_array")
        focus_aspects = []
    if focus_unknown:
        error_types.append("unknown_aspect_label")

    avoid_aspects, avoid_unknown = normalize_aspect_values(payload.get("avoid_aspects", []))
    if avoid_aspects is None:
        error_types.append("invalid_aspect_array")
        avoid_aspects = []
    if avoid_unknown:
        error_types.append("unknown_aspect_label")

    unsupported_requests, unsupported_unknown = normalize_unsupported_values(
        payload.get("unsupported_requests", [])
    )
    if unsupported_requests is None:
        error_types.append("invalid_unsupported_array")
        unsupported_requests = []
    if unsupported_unknown:
        error_types.append("unknown_unsupported_label")

    preference = UserPreference(
        city=city,
        state=state,
        hotel_category=hotel_category,
        focus_aspects=focus_aspects,
        avoid_aspects=avoid_aspects,
        unsupported_requests=unsupported_requests,
        query_en=build_query_en_from_slots(city, focus_aspects, avoid_aspects, unsupported_requests),
    )
    return preference, stable_sorted_unique(error_types), True


def coerce_clarification_payload(
    payload: dict[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    empty_payload = {
        "clarify_needed": False,
        "clarify_reason": "",
        "target_slots": [],
        "question": "",
    }
    if payload is None:
        return empty_payload, False

    question = str(payload.get("question", "") or "").strip()
    decision_label = normalize_decision_label(payload.get("decision_label"))
    if decision_label == "missing_city":
        return {
            "clarify_needed": True,
            "clarify_reason": "missing_city",
            "target_slots": ["city"],
            "question": question,
        }, True
    if decision_label == "aspect_conflict":
        return {
            "clarify_needed": True,
            "clarify_reason": "aspect_conflict",
            "target_slots": ["focus_aspects", "avoid_aspects"],
            "question": question,
        }, True
    if decision_label == "none":
        return empty_payload, True

    clarify_needed = parse_payload_bool(payload.get("clarify_needed"), default=False)
    clarify_reason = str(payload.get("clarify_reason", "") or "")
    if clarify_reason not in {"", "missing_city", "aspect_conflict"}:
        clarify_reason = ""
    target_slots = payload.get("target_slots", [])
    if not isinstance(target_slots, list):
        target_slots = []
    allowed_slots = {"city", "focus_aspects", "avoid_aspects"}
    target_slots = [slot for slot in target_slots if slot in allowed_slots]
    if not clarify_needed:
        return empty_payload, True
    return {
        "clarify_needed": clarify_needed,
        "clarify_reason": clarify_reason,
        "target_slots": target_slots,
        "question": question,
    }, True


def set_f1(gold_sets: list[set[str]], pred_sets: list[set[str]]) -> float:
    tp = fp = fn = 0
    for gold, pred in zip(gold_sets, pred_sets):
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def single_slot_score(gold_values: list[str | None], pred_values: list[str | None]) -> float:
    correct = sum(int(gold == pred) for gold, pred in zip(gold_values, pred_values))
    return correct / max(len(gold_values), 1)


def build_balanced_e4_subset(
    judged_queries: list[dict[str, Any]],
    clarify_lookup: dict[str, dict[str, Any]],
) -> list[str]:
    positives = [row["query_id"] for row in judged_queries if clarify_lookup[row["query_id"]]["clarify_needed"]]
    negatives_by_type: dict[str, list[str]] = defaultdict(list)
    for row in judged_queries:
        if clarify_lookup[row["query_id"]]["clarify_needed"]:
            continue
        negatives_by_type[row["query_type"]].append(row["query_id"])
    for query_ids in negatives_by_type.values():
        query_ids.sort()

    negatives: list[str] = []
    query_types = sorted(negatives_by_type)
    cursor = 0
    while len(negatives) < 16 and query_types:
        query_type = query_types[cursor % len(query_types)]
        bucket = negatives_by_type[query_type]
        if bucket:
            negatives.append(bucket.pop(0))
        cursor += 1
        if cursor > 256:
            break
    return positives + negatives[:16]


def run_e3_preference_eval(
    output_root: Path,
    limit_queries: int | None = None,
    query_id_file: str | Path | None = None,
) -> Path:
    cfg = load_config()
    frozen_config = load_json(EXPERIMENT_ASSETS_DIR / "frozen_config.yaml")
    behavior_runtime_config, behavior_api_key = resolve_behavior_runtime_config(cfg, frozen_config)
    split_manifest = load_json(EXPERIMENT_ASSETS_DIR / "frozen_split_manifest.json")
    judged_queries = load_jsonl(EXPERIMENT_ASSETS_DIR / "judged_queries.jsonl")
    slot_lookup = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "slot_gold.jsonl")}
    city_to_state = {
        row["city"]: row["state"]
        for row in slot_lookup.values()
        if row["city"] and row["state"]
    }
    judged_queries, query_scope, selected_query_ids = select_query_rows(
        judged_queries,
        limit_queries=limit_queries,
        query_id_file=query_id_file,
    )
    prompt_version_id = frozen_config["behavior"]["prompt_versions"]["e3_preference"]

    stable_run_config = {
        "task": "E3",
        "split_config_hash": split_manifest["meta"]["config_hash"],
        "query_scope": query_scope,
        "query_count": len(judged_queries),
        "query_id_selection_hash": stable_hash({"query_ids": selected_query_ids}),
        "behavior_backend": behavior_runtime_config.llm_backend,
        "base_model_id": behavior_runtime_config.model_id,
        "behavior_api_base_url": behavior_runtime_config.api_base_url,
        "behavior_enable_thinking": behavior_runtime_config.enable_thinking,
        "behavior_temperature": behavior_runtime_config.temperature,
        "behavior_max_new_tokens": behavior_runtime_config.max_new_tokens,
        "prompt_version_id": prompt_version_id,
        "default_retrieval_mode": frozen_config["workflow"]["default_retrieval_mode"],
        "fallback_enabled": frozen_config["workflow"]["enable_fallback"],
        "official_group_ids": E3_GROUPS,
    }
    run_started_at = utc_now_iso()
    run_id = f"e3_{stable_hash(stable_run_config)}_{run_started_at.replace(':', '').replace('-', '')}"
    run_dir = ensure_dir(output_root / run_id)

    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "generated_at": run_started_at,
                "stable_run_config": stable_run_config,
                "official_group_ids": E3_GROUPS,
                "selected_query_ids": selected_query_ids,
                "behavior_runtime_config": behavior_runtime_config.model_dump(),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    llm_runner: BehaviorLLMBackend | None = None
    log_rows: list[dict[str, Any]] = []
    parsed_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for group_id in E3_GROUPS:
        if group_id == "B_base_llm_structured" and llm_runner is None:
            llm_runner = build_behavior_backend(behavior_runtime_config, behavior_api_key)

        for query_row in judged_queries:
            query_id = query_row["query_id"]
            raw_response = ""
            error_types: list[str] = []
            start = time.perf_counter()
            if group_id == "A_rule_parser":
                preference = parse_rule_preference(query_row["query_text_zh"], city_to_state)
                schema_valid = True
            else:
                system_prompt, user_prompt = build_preference_prompts(
                    query_row["query_text_zh"],
                    list(city_to_state),
                    city_to_state,
                    prompt_version_id=prompt_version_id,
                )
                raw_response = llm_runner.generate_json(
                    system_prompt,
                    user_prompt,
                    max_new_tokens=behavior_runtime_config.max_new_tokens,
                )
                payload, repaired = parse_json_with_repair(raw_response)
                if repaired:
                    error_types.append("json_repaired")
                preference, error_types_from_payload, schema_valid = coerce_preference_payload(
                    payload,
                    query_row["query_text_zh"],
                    city_to_state,
                )
                error_types.extend(error_types_from_payload)
            latency_ms = round((time.perf_counter() - start) * 1000, 3)

            parse_result = PreferenceParseResult(
                query_id=query_id,
                group_id=group_id,
                preference=preference,
                schema_valid=schema_valid,
                raw_response=raw_response,
                error_types=stable_sorted_unique(error_types),
            )
            gold = slot_lookup[query_id]
            predicted = parse_result.preference.model_dump()
            gold_focus = set(gold["focus_aspects"])
            pred_focus = set(predicted["focus_aspects"])
            gold_avoid = set(gold["avoid_aspects"])
            pred_avoid = set(predicted["avoid_aspects"])
            gold_unsupported = set(gold["unsupported_requests"])
            pred_unsupported = set(predicted["unsupported_requests"])

            effective_errors = list(parse_result.error_types)
            if gold["city"] != predicted["city"]:
                effective_errors.append("city_missing")
            if gold_focus != pred_focus or gold_avoid != pred_avoid:
                effective_errors.append("aspect_mapping_error")
            if gold_unsupported - pred_unsupported:
                effective_errors.append("unsupported_missed")
            if gold_unsupported and not pred_unsupported and schema_valid:
                effective_errors.append("unsupported_as_supported")
            parse_result.error_types = stable_sorted_unique(effective_errors)
            parsed_by_group[group_id].append(
                {
                    "query_id": query_id,
                    "gold": gold,
                    "pred": predicted,
                    "schema_valid": schema_valid,
                    "error_types": parse_result.error_types,
                    "query_text_zh": query_row["query_text_zh"],
                }
            )

            log_rows.append(
                RunLogEntry(
                    run_id=run_id,
                    group_id=group_id,
                    query_id=query_id,
                    retrieval_mode=stable_run_config["default_retrieval_mode"],
                    candidate_mode=BEHAVIOR_CANDIDATE_MODE,
                    config_hash=stable_hash(stable_run_config | {"group_id": group_id}),
                    latency_ms=latency_ms,
                    intermediate_objects={
                        "query": query_row,
                        "gold": gold,
                        "result": parse_result.model_dump(),
                        "behavior_runtime_config": behavior_runtime_config.model_dump(),
                    },
                ).model_dump()
            )

    summary_rows: list[dict[str, Any]] = []
    for group_id in E3_GROUPS:
        rows = parsed_by_group[group_id]
        gold_city = [row["gold"]["city"] for row in rows]
        pred_city = [row["pred"]["city"] for row in rows]
        gold_cat = [row["gold"]["hotel_category"] for row in rows]
        pred_cat = [row["pred"]["hotel_category"] for row in rows]
        gold_focus = [set(row["gold"]["focus_aspects"]) for row in rows]
        pred_focus = [set(row["pred"]["focus_aspects"]) for row in rows]
        gold_avoid = [set(row["gold"]["avoid_aspects"]) for row in rows]
        pred_avoid = [set(row["pred"]["avoid_aspects"]) for row in rows]
        gold_unsupported = [set(row["gold"]["unsupported_requests"]) for row in rows]
        pred_unsupported = [set(row["pred"]["unsupported_requests"]) for row in rows]
        exact_match = sum(
            int(
                row["gold"]["city"] == row["pred"]["city"]
                and row["gold"]["hotel_category"] == row["pred"]["hotel_category"]
                and set(row["gold"]["focus_aspects"]) == set(row["pred"]["focus_aspects"])
                and set(row["gold"]["avoid_aspects"]) == set(row["pred"]["avoid_aspects"])
                and set(row["gold"]["unsupported_requests"]) == set(row["pred"]["unsupported_requests"])
            )
            for row in rows
        ) / max(len(rows), 1)
        unsupported_tp = sum(
            len(set(row["gold"]["unsupported_requests"]) & set(row["pred"]["unsupported_requests"]))
            for row in rows
        )
        unsupported_total = sum(len(row["gold"]["unsupported_requests"]) for row in rows)
        summary_rows.append(
            {
                "group_id": group_id,
                "query_count": len(rows),
                "schema_valid_rate": round(sum(int(row["schema_valid"]) for row in rows) / max(len(rows), 1), 4),
                "exact_match_rate": round(exact_match, 4),
                "unsupported_detection_recall": round(unsupported_tp / max(unsupported_total, 1), 4),
                "city_slot_f1": round(single_slot_score(gold_city, pred_city), 4),
                "hotel_category_slot_f1": round(single_slot_score(gold_cat, pred_cat), 4),
                "focus_aspects_slot_f1": round(set_f1(gold_focus, pred_focus), 4),
                "avoid_aspects_slot_f1": round(set_f1(gold_avoid, pred_avoid), 4),
                "unsupported_requests_slot_f1": round(set_f1(gold_unsupported, pred_unsupported), 4),
                "avg_latency_ms": round(
                    sum(row["latency_ms"] for row in log_rows if row["group_id"] == group_id) / max(len(rows), 1),
                    3,
                ),
                "config_hash": stable_hash(stable_run_config | {"group_id": group_id}),
            }
        )

    write_jsonl(run_dir / "results.jsonl", log_rows)
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# E3 Preference Parsing Result",
        "",
        "## Summary Table",
        "",
    ]
    lines.extend(markdown_table(summary_rows))
    lines.extend(["", "## Error Breakdown", ""])
    for group_id in E3_GROUPS:
        rows = parsed_by_group[group_id]
        error_counter = Counter()
        for row in rows:
            error_counter.update(row["error_types"])
        lines.append(f"### {group_id}")
        lines.append("")
        for key in ["city_missing", "aspect_mapping_error", "unsupported_missed", "unsupported_as_supported"]:
            lines.append(f"- {key}: {error_counter.get(key, 0)}")
        lines.append("")
        representative_rows = [row for row in rows if row["error_types"]][:3]
        if representative_rows:
            lines.append("Representative cases:")
            for row in representative_rows:
                lines.append(
                    f"- `{row['query_id']}` {row['query_text_zh']} | errors={','.join(row['error_types'])}"
                )
        else:
            lines.append("- none")
        lines.append("")
    (run_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return run_dir


def run_e4_clarification_eval(
    output_root: Path,
    limit_queries: int | None = None,
    query_id_file: str | Path | None = None,
) -> Path:
    cfg = load_config()
    frozen_config = load_json(EXPERIMENT_ASSETS_DIR / "frozen_config.yaml")
    behavior_runtime_config, behavior_api_key = resolve_behavior_runtime_config(cfg, frozen_config)
    split_manifest = load_json(EXPERIMENT_ASSETS_DIR / "frozen_split_manifest.json")
    judged_queries = load_jsonl(EXPERIMENT_ASSETS_DIR / "judged_queries.jsonl")
    clarify_lookup = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "clarify_gold.jsonl")}
    slot_lookup = {row["query_id"]: row for row in load_jsonl(EXPERIMENT_ASSETS_DIR / "slot_gold.jsonl")}
    city_to_state = {
        row["city"]: row["state"]
        for row in slot_lookup.values()
        if row["city"] and row["state"]
    }
    judged_queries, query_scope, selected_query_ids = select_query_rows(
        judged_queries,
        limit_queries=limit_queries,
        query_id_file=query_id_file,
    )

    balanced_subset = build_balanced_e4_subset(judged_queries, clarify_lookup)
    prompt_version_id = frozen_config["behavior"]["prompt_versions"]["e4_clarification"]
    stable_run_config = {
        "task": "E4",
        "split_config_hash": split_manifest["meta"]["config_hash"],
        "query_scope": query_scope,
        "query_count": len(judged_queries),
        "query_id_selection_hash": stable_hash({"query_ids": selected_query_ids}),
        "balanced_subset_size": len(balanced_subset),
        "behavior_backend": behavior_runtime_config.llm_backend,
        "base_model_id": behavior_runtime_config.model_id,
        "behavior_api_base_url": behavior_runtime_config.api_base_url,
        "behavior_enable_thinking": behavior_runtime_config.enable_thinking,
        "behavior_temperature": behavior_runtime_config.temperature,
        "behavior_max_new_tokens": behavior_runtime_config.max_new_tokens,
        "prompt_version_id": prompt_version_id,
        "default_retrieval_mode": frozen_config["workflow"]["default_retrieval_mode"],
        "fallback_enabled": frozen_config["workflow"]["enable_fallback"],
        "official_group_ids": E4_GROUPS,
    }
    run_started_at = utc_now_iso()
    run_id = f"e4_{stable_hash(stable_run_config)}_{run_started_at.replace(':', '').replace('-', '')}"
    run_dir = ensure_dir(output_root / run_id)
    ensure_dir(E4_LABELS_DIR)

    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "generated_at": run_started_at,
                "stable_run_config": stable_run_config,
                "balanced_subset_query_ids": balanced_subset,
                "selected_query_ids": selected_query_ids,
                "behavior_runtime_config": behavior_runtime_config.model_dump(),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    llm_runner: BehaviorLLMBackend | None = None
    log_rows: list[dict[str, Any]] = []
    decisions_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for group_id in E4_GROUPS:
        if group_id == "B_base_llm_clarify" and llm_runner is None:
            llm_runner = build_behavior_backend(behavior_runtime_config, behavior_api_key)

        for query_row in judged_queries:
            query_id = query_row["query_id"]
            raw_response = ""
            start = time.perf_counter()
            if group_id == "A_rule_clarify":
                decision = build_rule_clarification(query_row["query_text_zh"], city_to_state)
                decision.query_id = query_id
                schema_valid = True
            else:
                system_prompt, user_prompt = build_clarification_prompts(
                    query_row["query_text_zh"],
                    list(city_to_state),
                    prompt_version_id=prompt_version_id,
                )
                raw_response = llm_runner.generate_json(
                    system_prompt,
                    user_prompt,
                    max_new_tokens=behavior_runtime_config.max_new_tokens,
                )
                payload, _ = parse_json_with_repair(raw_response)
                coerced, schema_valid = coerce_clarification_payload(payload)
                decision = ClarificationDecision(
                    query_id=query_id,
                    group_id=group_id,
                    clarify_needed=coerced["clarify_needed"],
                    clarify_reason=coerced["clarify_reason"],
                    target_slots=coerced["target_slots"],
                    question=coerced["question"],
                    schema_valid=schema_valid,
                    raw_response=raw_response,
                )
            latency_ms = round((time.perf_counter() - start) * 1000, 3)
            gold = clarify_lookup[query_id]
            decisions_by_group[group_id].append(
                {
                    "query_id": query_id,
                    "gold": gold,
                    "result": decision.model_dump(),
                    "query_text_zh": query_row["query_text_zh"],
                }
            )
            log_rows.append(
                RunLogEntry(
                    run_id=run_id,
                    group_id=group_id,
                    query_id=query_id,
                    retrieval_mode=stable_run_config["default_retrieval_mode"],
                    candidate_mode=BEHAVIOR_CANDIDATE_MODE,
                    config_hash=stable_hash(stable_run_config | {"group_id": group_id}),
                    latency_ms=latency_ms,
                    intermediate_objects={
                        "query": query_row,
                        "gold": gold,
                        "result": decision.model_dump(),
                        "behavior_runtime_config": behavior_runtime_config.model_dump(),
                    },
                ).model_dump()
            )

    summary_rows: list[dict[str, Any]] = []
    balanced_rows: list[dict[str, Any]] = []

    def metric_row(rows: list[dict[str, Any]], group_id: str) -> dict[str, Any]:
        gold_flags = [int(row["gold"]["clarify_needed"]) for row in rows]
        pred_flags = [int(row["result"]["clarify_needed"]) for row in rows]
        tp = sum(int(g == 1 and p == 1) for g, p in zip(gold_flags, pred_flags))
        fp = sum(int(g == 0 and p == 1) for g, p in zip(gold_flags, pred_flags))
        fn = sum(int(g == 1 and p == 0) for g, p in zip(gold_flags, pred_flags))
        tn = sum(int(g == 0 and p == 0) for g, p in zip(gold_flags, pred_flags))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
        return {
            "group_id": group_id,
            "query_count": len(rows),
            "clarification_accuracy": round((tp + tn) / max(len(rows), 1), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "over_clarification_rate": round(fp / max((fp + tn), 1), 4),
            "under_clarification_rate": round(fn / max((tp + fn), 1), 4),
            "schema_valid_rate": round(sum(int(row["result"]["schema_valid"]) for row in rows) / max(len(rows), 1), 4),
            "avg_latency_ms": round(
                sum(row["latency_ms"] for row in log_rows if row["group_id"] == group_id) / max(len(rows), 1),
                3,
            ),
            "config_hash": stable_hash(stable_run_config | {"group_id": group_id}),
        }

    for group_id in E4_GROUPS:
        rows = decisions_by_group[group_id]
        summary_rows.append(metric_row(rows, group_id))
        balanced_rows.append(metric_row([row for row in rows if row["query_id"] in balanced_subset], group_id))

    audit_rows = []
    for group_id in E4_GROUPS:
        for row in decisions_by_group[group_id]:
            if not clarify_lookup[row["query_id"]]["clarify_needed"]:
                continue
            audit_rows.append(
                {
                    "query_id": row["query_id"],
                    "group_id": group_id,
                    "question": row["result"]["question"],
                    "answerable_score": "",
                    "targeted_score": "",
                    "notes": "",
                }
            )

    audit_df = pd.DataFrame(audit_rows)
    write_jsonl(run_dir / "results.jsonl", log_rows)
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(balanced_rows).to_csv(run_dir / "balanced_summary.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(run_dir / "clarification_question_audit.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(E4_LABELS_DIR / "clarification_question_audit.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# E4 Clarification Result",
        "",
        "## Main Summary",
        "",
    ]
    lines.extend(markdown_table(summary_rows))
    lines.extend(["", "## Balanced Diagnostic Subset", ""])
    lines.extend(markdown_table(balanced_rows))
    lines.extend(["", "## Representative Cases", ""])
    for group_id in E4_GROUPS:
        lines.append(f"### {group_id}")
        lines.append("")
        group_rows = decisions_by_group[group_id]
        under = [
            row for row in group_rows
            if row["gold"]["clarify_needed"] and not row["result"]["clarify_needed"]
        ][:2]
        over = [
            row for row in group_rows
            if not row["gold"]["clarify_needed"] and row["result"]["clarify_needed"]
        ][:2]
        if under:
            lines.append("- Under-clarification:")
            for row in under:
                lines.append(f"  - `{row['query_id']}` {row['query_text_zh']}")
        if over:
            lines.append("- Over-clarification:")
            for row in over:
                lines.append(f"  - `{row['query_id']}` {row['query_text_zh']}")
        if not under and not over:
            lines.append("- none")
        lines.append("")
    lines.extend(
        [
            "## Audit Asset",
            "",
            f"- Run-local: `{run_dir.name}/clarification_question_audit.csv`",
            "- Latest copy: `experiments/labels/e4_clarification/clarification_question_audit.csv`",
        ]
    )
    (run_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return run_dir


def run_e5_query_bridge_eval(output_root: Path, limit_queries: int | None = None) -> Path:
    cfg = load_config()
    frozen_config = load_json(EXPERIMENT_ASSETS_DIR / "frozen_config.yaml")
    split_manifest = load_json(EXPERIMENT_ASSETS_DIR / "frozen_split_manifest.json")
    review_df = pd.read_pickle("data/intermediate/cleaned_reviews.pkl")
    evidence_df = pd.read_pickle("data/intermediate/evidence_index.pkl")
    city_test_hotels = build_city_test_hotels(split_manifest, review_df)
    evidence_lookup = build_evidence_lookup(evidence_df)
    target_units = build_target_units(limit_queries=limit_queries)
    qrels_lookup = load_qrels_lookup(E6_LABELS_DIR / "qrels_evidence.jsonl")

    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer

    client = PersistentClient(path=cfg["embedding"]["chroma_persist_dir"])
    collection = client.get_collection(cfg["embedding"]["chroma_collection"])
    bi_encoder = SentenceTransformer(cfg["embedding"]["model"])
    normalize_embeddings = bool(cfg["embedding"].get("normalize", True))

    stable_run_config = {
        "task": "E5",
        "split_config_hash": split_manifest["meta"]["config_hash"],
        "query_scope": "e6_executable_query_units",
        "query_count": len({unit["query_id"] for unit in target_units}),
        "target_unit_count": len(target_units),
        "base_model_id": frozen_config["behavior"]["base_model"],
        "prompt_version_id": frozen_config["behavior"]["prompt_versions"]["e5_query_bridge"],
        "default_retrieval_mode": frozen_config["workflow"]["default_retrieval_mode"],
        "fallback_enabled": frozen_config["workflow"]["enable_fallback"],
        "official_group_ids": E5_GROUPS,
        "query_strategy": {
            "A": "zh_direct_dense_no_rerank",
            "B": "structured_query_en_dense_no_rerank",
        },
    }
    run_started_at = utc_now_iso()
    run_id = f"e5_{stable_hash(stable_run_config)}_{run_started_at.replace(':', '').replace('-', '')}"
    run_dir = ensure_dir(output_root / run_id)
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "generated_at": run_started_at,
                "stable_run_config": stable_run_config,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(cfg["reranker"]["model"])
    from scripts.evaluation.evaluate_e6_e8_retrieval import warm_up_models

    warm_up_models(collection, bi_encoder, reranker, normalize_embeddings)

    log_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for group_id in E5_GROUPS:
        metric_rows = []
        latencies = []
        for unit in target_units:
            query_unit = dict(unit)
            query_unit["query_en_source"] = group_id
            query_unit["query_en_target"] = (
                query_unit["query_text_zh"]
                if group_id == "A_zh_direct_dense_no_rerank"
                else query_unit["query_en_target"]
            )
            city_hotels = city_test_hotels[query_unit["city"]]
            mode_result = retrieve_official_mode(
                unit=query_unit,
                mode="aspect_main_no_rerank",
                city_hotels=city_hotels,
                collection=collection,
                bi_encoder=bi_encoder,
                reranker=reranker,
                normalize_embeddings=normalize_embeddings,
                dense_top_k=cfg["reranker"]["top_k_before_rerank"],
                final_top_k=cfg["reranker"]["top_k_after_rerank"],
                evidence_lookup=evidence_lookup,
            )
            qrels_by_sentence = qrels_lookup.get(
                (query_unit["query_id"], query_unit["target_aspect"], query_unit["target_role"]),
                {},
            )
            metrics, ranked_rows = evaluate_ranked_rows(mode_result["rows"], qrels_by_sentence)
            metric_rows.append(metrics)
            latencies.append(mode_result["latency_ms"])

            bridge_record = BridgeQueryRecord(
                query_id=query_unit["query_id"],
                target_aspect=query_unit["target_aspect"],
                target_role=query_unit["target_role"],
                query_text_zh=query_unit["query_text_zh"],
                query_en_source=query_unit["query_en_target"],
                retrieval_mode="aspect_main_no_rerank",
                ranked_rows=ranked_rows,
                metrics=metrics,
            )
            log_rows.append(
                RunLogEntry(
                    run_id=run_id,
                    group_id=group_id,
                    query_id=query_unit["query_id"],
                    retrieval_mode="aspect_main_no_rerank",
                    candidate_mode="city_test_all",
                    config_hash=stable_hash(stable_run_config | {"group_id": group_id}),
                    latency_ms=mode_result["latency_ms"],
                    intermediate_objects={
                        "query_unit": query_unit,
                        "bridge_record": bridge_record.model_dump(),
                        "retrieval_trace": mode_result["retrieval_trace"],
                        "ranked_rows": ranked_rows,
                        "metrics": metrics,
                    },
                ).model_dump()
            )

        summary_rows.append(
            {
                "group_id": group_id,
                "query_count": len({unit["query_id"] for unit in target_units}),
                "target_unit_count": len(target_units),
                "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 3),
                "aspect_recall_at_5": round(sum(row["aspect_recall_at_5"] for row in metric_rows) / max(len(metric_rows), 1), 4),
                "ndcg_at_5": round(sum(row["ndcg_at_5"] for row in metric_rows) / max(len(metric_rows), 1), 4),
                "mrr_at_5": round(sum(row["mrr_at_5"] for row in metric_rows) / max(len(metric_rows), 1), 4),
                "precision_at_5": round(sum(row["precision_at_5"] for row in metric_rows) / max(len(metric_rows), 1), 4),
                "config_hash": stable_hash(stable_run_config | {"group_id": group_id}),
            }
        )

    write_jsonl(run_dir / "results.jsonl", log_rows)
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")

    by_group_role: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    by_group_aspect: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in log_rows:
        unit = row["intermediate_objects"]["query_unit"]
        metrics = row["intermediate_objects"]["metrics"]
        by_group_role[row["group_id"]][unit["target_role"]].append(metrics)
        by_group_aspect[row["group_id"]][unit["target_aspect"]].append(metrics)

    lines = [
        "# E5 Query Bridge Result",
        "",
        "## Summary Table",
        "",
    ]
    lines.extend(markdown_table(summary_rows))
    lines.extend(["", "## Role Breakdown", ""])
    for group_id in E5_GROUPS:
        lines.append(f"### {group_id}")
        lines.append("")
        for role in sorted(by_group_role[group_id]):
            metrics = by_group_role[group_id][role]
            lines.append(
                f"- {role}: nDCG@5={round(sum(item['ndcg_at_5'] for item in metrics) / len(metrics), 4)}, "
                f"Precision@5={round(sum(item['precision_at_5'] for item in metrics) / len(metrics), 4)}"
            )
        lines.append("")
    lines.extend(["## Aspect Dependence", ""])
    for group_id in E5_GROUPS:
        aspect_scores = []
        for aspect, metrics in by_group_aspect[group_id].items():
            aspect_scores.append(
                (
                    round(sum(item["ndcg_at_5"] for item in metrics) / len(metrics), 4),
                    aspect,
                )
            )
        aspect_scores.sort()
        lines.append(f"### {group_id}")
        lines.append("")
        lines.append(f"- worst aspects: {aspect_scores[:3]}")
        lines.append(f"- best aspects: {aspect_scores[-3:]}")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- This run compares direct Chinese dense retrieval against structured English retrieval under the same candidate set and `aspect_main_no_rerank` backend.",
            "- If `avoid` and `quiet_sleep` remain weak in both groups, the bottleneck is evidence coverage rather than bridge language alone.",
        ]
    )
    (run_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=["run_e3", "run_e4", "run_e5"],
        required=True,
    )
    parser.add_argument("--output-root", default=str(EXPERIMENT_RUNS_DIR))
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--query-id-file", default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.action == "run_e3":
        run_dir = run_e3_preference_eval(
            output_root=output_root,
            limit_queries=args.limit_queries,
            query_id_file=args.query_id_file,
        )
    elif args.action == "run_e4":
        run_dir = run_e4_clarification_eval(
            output_root=output_root,
            limit_queries=args.limit_queries,
            query_id_file=args.query_id_file,
        )
    else:
        run_dir = run_e5_query_bridge_eval(output_root=output_root, limit_queries=args.limit_queries)
    print(f"[OK] run saved to {run_dir}")


if __name__ == "__main__":
    main()

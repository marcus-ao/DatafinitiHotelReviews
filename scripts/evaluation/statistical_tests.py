from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


DEFAULT_GROUP_ORDER = ("G1", "G2", "G3", "G4")
LOWER_IS_BETTER_KEYWORDS = (
    "hallucination",
    "latency",
    "error",
    "leak",
    "invalid",
    "out_of_pack",
    "retry_trigger",
)
SIGNIFICANCE_THRESHOLDS = (
    (0.001, "***"),
    (0.01, "**"),
    (0.05, "*"),
)
PAIRWISE_TEST_COLUMNS = [
    "group_a",
    "group_b",
    "metric",
    "pairing_mode",
    "overlap_n",
    "dropped_from_a",
    "dropped_from_b",
    "n",
    "non_zero_n",
    "wins_group_a",
    "wins_group_b",
    "ties",
    "higher_is_better",
    "better_group",
    "mean_a",
    "mean_b",
    "median_a",
    "median_b",
    "std_a",
    "std_b",
    "mean_delta",
    "median_delta",
    "std_delta",
    "wilcoxon_statistic",
    "p_value",
    "significance",
    "p_value_adj",
    "significance_adj",
    "p_adjust_method",
    "ci_level",
    "ci_low",
    "ci_high",
    "cohens_d",
    "cohens_d_magnitude",
    "rank_biserial",
    "note",
]


def _to_1d_float_array(scores: Iterable[float], *, label: str) -> np.ndarray:
    array = np.asarray(list(scores), dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{label} 必须是一维数值序列。")
    if array.size == 0:
        raise ValueError(f"{label} 不能为空。")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} 不能包含 NaN 或无穷大。")
    return array


def _normalize_metric_series(
    scores_or_payload: Iterable[float] | Mapping[str, Any],
    *,
    label: str,
) -> tuple[np.ndarray, list[str] | None]:
    if isinstance(scores_or_payload, Mapping):
        if "scores" in scores_or_payload:
            score_values = scores_or_payload["scores"]
        elif "values" in scores_or_payload:
            score_values = scores_or_payload["values"]
        else:
            raise ValueError(f"{label} 缺少 scores/values 字段。")
        score_array = _to_1d_float_array(score_values, label=f"{label}.scores")
        query_ids_raw = scores_or_payload.get("query_ids", scores_or_payload.get("ids"))
        if query_ids_raw is None:
            return score_array, None
        query_ids = [str(query_id) for query_id in query_ids_raw]
        if len(query_ids) != score_array.size:
            raise ValueError(f"{label} 的 query_ids 数量与 scores 数量不一致。")
        if len(set(query_ids)) != len(query_ids):
            raise ValueError(f"{label} 的 query_ids 存在重复值。")
        return score_array, query_ids
    return _to_1d_float_array(scores_or_payload, label=label), None


def _pair_metric_series(
    group_a_scores: Iterable[float] | Mapping[str, Any],
    group_b_scores: Iterable[float] | Mapping[str, Any],
    *,
    label_a: str,
    label_b: str,
) -> tuple[np.ndarray, np.ndarray, str, int, int, int]:
    scores_a, query_ids_a = _normalize_metric_series(group_a_scores, label=label_a)
    scores_b, query_ids_b = _normalize_metric_series(group_b_scores, label=label_b)
    if query_ids_a is None and query_ids_b is None:
        if scores_a.size != scores_b.size:
            raise ValueError(f"{label_a} 与 {label_b} 的配对比较要求长度一致。")
        return scores_a, scores_b, "position", int(scores_a.size), 0, 0
    if query_ids_a is None or query_ids_b is None:
        raise ValueError(f"{label_a} 与 {label_b} 必须同时提供 query_ids，或同时不提供。")

    lookup_a = dict(zip(query_ids_a, scores_a, strict=True))
    lookup_b = dict(zip(query_ids_b, scores_b, strict=True))
    query_id_set_a = set(lookup_a)
    query_id_set_b = set(lookup_b)
    overlap_query_ids = [query_id for query_id in query_ids_a if query_id in query_id_set_b]
    if not overlap_query_ids:
        missing_in_b = sorted(query_id_set_a - query_id_set_b)
        missing_in_a = sorted(query_id_set_b - query_id_set_a)
        raise ValueError(
            f"{label_a} 与 {label_b} 没有可配对的 query_ids："
            f"missing_in_b={missing_in_b}, missing_in_a={missing_in_a}"
        )

    ordered_query_ids = overlap_query_ids
    aligned_a = np.asarray([lookup_a[query_id] for query_id in ordered_query_ids], dtype=float)
    aligned_b = np.asarray([lookup_b[query_id] for query_id in ordered_query_ids], dtype=float)
    return (
        aligned_a,
        aligned_b,
        "query_id",
        int(len(ordered_query_ids)),
        int(len(query_id_set_a - set(ordered_query_ids))),
        int(len(query_id_set_b - set(ordered_query_ids))),
    )


def _default_group_pairs(group_names: Iterable[str]) -> list[tuple[str, str]]:
    ordered_group_names = list(dict.fromkeys(str(group_name) for group_name in group_names))
    canonical_group_names = [group_name for group_name in DEFAULT_GROUP_ORDER if group_name in ordered_group_names]
    if canonical_group_names and set(canonical_group_names) == set(ordered_group_names):
        return list(combinations(canonical_group_names, 2))
    return list(combinations(sorted(ordered_group_names), 2))


def _infer_higher_is_better(metric: str, override: Mapping[str, bool] | None = None) -> bool:
    if override and metric in override:
        return bool(override[metric])
    metric_lower = metric.lower()
    return not any(keyword in metric_lower for keyword in LOWER_IS_BETTER_KEYWORDS)


def significance_stars(p_value: float | None) -> str:
    if p_value is None or not np.isfinite(float(p_value)):
        return "n/a"
    p_value = float(p_value)
    for threshold, label in SIGNIFICANCE_THRESHOLDS:
        if p_value < threshold:
            return label
    return "ns"


def _cohens_d_magnitude(effect_size: float) -> str:
    if not np.isfinite(float(effect_size)):
        return "infinite"
    absolute_effect_size = abs(float(effect_size))
    if absolute_effect_size < 0.2:
        return "negligible"
    if absolute_effect_size < 0.5:
        return "small"
    if absolute_effect_size < 0.8:
        return "medium"
    return "large"


def _adjust_p_values(p_values: list[float], *, method: str) -> list[float]:
    if method == "none":
        return [round(float(p_value), 6) for p_value in p_values]
    if method == "bonferroni":
        count = len(p_values)
        return [round(min(float(p_value) * count, 1.0), 6) for p_value in p_values]
    if method != "holm":
        raise ValueError(f"Unsupported p-value adjustment method: {method}")

    count = len(p_values)
    if count == 0:
        return []
    indexed_p_values = sorted(enumerate(float(p_value) for p_value in p_values), key=lambda item: item[1])
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, (original_index, p_value) in enumerate(indexed_p_values):
        adjusted_value = min((count - rank) * p_value, 1.0)
        running_max = max(running_max, adjusted_value)
        adjusted[original_index] = round(running_max, 6)
    return adjusted


def wilcoxon_signed_rank(
    group_a_scores: Iterable[float],
    group_b_scores: Iterable[float],
    *,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    group_a = _to_1d_float_array(group_a_scores, label="group_a_scores")
    group_b = _to_1d_float_array(group_b_scores, label="group_b_scores")
    if group_a.size != group_b.size:
        raise ValueError("Wilcoxon 配对检验要求两组输入长度一致。")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative 必须是 two-sided / greater / less。")

    deltas = group_b - group_a
    non_zero_deltas = deltas[np.abs(deltas) > 1e-12]
    result = {
        "test": "wilcoxon_signed_rank",
        "n": int(group_a.size),
        "non_zero_n": int(non_zero_deltas.size),
        "zero_diff_count": int(group_a.size - non_zero_deltas.size),
        "alternative": alternative,
    }
    if non_zero_deltas.size == 0:
        return result | {
            "statistic": 0.0,
            "p_value": 1.0,
            "note": "all_zero_differences",
        }
    if non_zero_deltas.size < 2:
        return result | {
            "statistic": 0.0,
            "p_value": 1.0,
            "note": "insufficient_non_zero_pairs",
        }

    statistic, p_value = wilcoxon(
        group_a,
        group_b,
        zero_method="wilcox",
        alternative=alternative,
        method="auto",
    )
    return result | {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "note": "ok",
    }


def bootstrap_ci(
    scores: Iterable[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    values = _to_1d_float_array(scores, label="scores")
    if n_resamples <= 0:
        raise ValueError("n_resamples 必须为正整数。")
    if not 0 < ci < 1:
        raise ValueError("ci 必须在 (0, 1) 区间内。")
    if values.size == 1:
        value = round(float(values[0]), 4)
        return value, value

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_resamples, dtype=float)
    for idx in range(n_resamples):
        sampled = rng.choice(values, size=values.size, replace=True)
        bootstrap_means[idx] = float(np.mean(sampled))

    alpha = (1.0 - ci) / 2.0
    lower = round(float(np.quantile(bootstrap_means, alpha)), 4)
    upper = round(float(np.quantile(bootstrap_means, 1.0 - alpha)), 4)
    return lower, upper


def cohens_d(group_a_scores: Iterable[float], group_b_scores: Iterable[float]) -> float:
    group_a = _to_1d_float_array(group_a_scores, label="group_a_scores")
    group_b = _to_1d_float_array(group_b_scores, label="group_b_scores")
    if group_a.size != group_b.size:
        raise ValueError("cohens_d 在当前实现中要求两组输入长度一致。")
    delta = group_b - group_a
    if group_a.size == 1 and group_b.size == 1:
        return 0.0 if float(delta[0]) == 0.0 else round(float(delta[0]), 4)
    mean_delta = float(np.mean(delta))
    delta_std = float(np.std(delta, ddof=1)) if delta.size > 1 else 0.0
    if delta_std <= 1e-12:
        if abs(mean_delta) <= 1e-12:
            return 0.0
        return float("inf") if mean_delta > 0 else float("-inf")
    return round(float(mean_delta / delta_std), 4)


def rank_biserial_correlation(group_a_scores: Iterable[float], group_b_scores: Iterable[float]) -> float:
    group_a = _to_1d_float_array(group_a_scores, label="group_a_scores")
    group_b = _to_1d_float_array(group_b_scores, label="group_b_scores")
    if group_a.size != group_b.size:
        raise ValueError("rank_biserial_correlation 在当前实现中要求两组输入长度一致。")
    delta = group_b - group_a
    non_zero_mask = np.abs(delta) > 1e-12
    non_zero_delta = delta[non_zero_mask]
    if non_zero_delta.size == 0:
        return 0.0
    ranks = rankdata(np.abs(non_zero_delta), method="average")
    positive_rank_sum = float(np.sum(ranks[non_zero_delta > 0]))
    negative_rank_sum = float(np.sum(ranks[non_zero_delta < 0]))
    total_rank_sum = positive_rank_sum + negative_rank_sum
    if total_rank_sum <= 1e-12:
        return 0.0
    return round((positive_rank_sum - negative_rank_sum) / total_rank_sum, 4)


def compute_pairwise_tests(
    group_score_map: Mapping[str, Mapping[str, Iterable[float] | Mapping[str, Any]]],
    metrics: Iterable[str],
    group_pairs: list[tuple[str, str]] | None = None,
    *,
    higher_is_better: Mapping[str, bool] | None = None,
    p_adjust: str = "holm",
    ci: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
    alternative: str = "two-sided",
) -> pd.DataFrame:
    if not group_score_map:
        raise ValueError("group_score_map 不能为空。")
    metric_names = list(metrics)
    if not metric_names:
        raise ValueError("metrics 不能为空。")
    if p_adjust not in {"none", "holm", "bonferroni"}:
        raise ValueError("p_adjust 必须是 none / holm / bonferroni。")

    group_names = list(group_score_map)
    pair_list = group_pairs or _default_group_pairs(group_names)
    rows: list[dict[str, Any]] = []
    for group_a, group_b in pair_list:
        if group_a not in group_score_map or group_b not in group_score_map:
            raise KeyError(f"未知分组：{group_a}, {group_b}")
        for metric in metric_names:
            if metric not in group_score_map[group_a]:
                raise KeyError(f"{group_a} 缺少指标：{metric}")
            if metric not in group_score_map[group_b]:
                raise KeyError(f"{group_b} 缺少指标：{metric}")

            scores_a, scores_b, pairing_mode, overlap_n, dropped_from_a, dropped_from_b = _pair_metric_series(
                group_score_map[group_a][metric],
                group_score_map[group_b][metric],
                label_a=f"{group_a}.{metric}",
                label_b=f"{group_b}.{metric}",
            )
            delta = scores_b - scores_a
            wilcoxon_result = wilcoxon_signed_rank(scores_a, scores_b, alternative=alternative)
            ci_low, ci_high = bootstrap_ci(delta, n_resamples=n_resamples, ci=ci, seed=seed)
            higher_is_better_flag = _infer_higher_is_better(metric, override=higher_is_better)
            mean_delta = round(float(np.mean(delta)), 4)
            if abs(mean_delta) <= 1e-12:
                better_group = "tie"
            elif higher_is_better_flag:
                better_group = group_b if mean_delta > 0 else group_a
            else:
                better_group = group_a if mean_delta > 0 else group_b

            cohens_d_value = cohens_d(scores_a, scores_b)
            rows.append(
                {
                    "group_a": group_a,
                    "group_b": group_b,
                    "metric": metric,
                    "pairing_mode": pairing_mode,
                    "overlap_n": overlap_n,
                    "dropped_from_a": dropped_from_a,
                    "dropped_from_b": dropped_from_b,
                    "n": int(scores_a.size),
                    "non_zero_n": int(wilcoxon_result["non_zero_n"]),
                    "wins_group_a": int(np.sum(delta < -1e-12)),
                    "wins_group_b": int(np.sum(delta > 1e-12)),
                    "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                    "higher_is_better": higher_is_better_flag,
                    "better_group": better_group,
                    "mean_a": round(float(np.mean(scores_a)), 4),
                    "mean_b": round(float(np.mean(scores_b)), 4),
                    "median_a": round(float(np.median(scores_a)), 4),
                    "median_b": round(float(np.median(scores_b)), 4),
                    "std_a": round(float(np.std(scores_a, ddof=1)) if scores_a.size > 1 else 0.0, 4),
                    "std_b": round(float(np.std(scores_b, ddof=1)) if scores_b.size > 1 else 0.0, 4),
                    "mean_delta": mean_delta,
                    "median_delta": round(float(np.median(delta)), 4),
                    "std_delta": round(float(np.std(delta, ddof=1)) if delta.size > 1 else 0.0, 4),
                    "wilcoxon_statistic": round(float(wilcoxon_result["statistic"]), 6),
                    "p_value": round(float(wilcoxon_result["p_value"]), 6),
                    "significance": significance_stars(float(wilcoxon_result["p_value"])),
                    "p_value_adj": round(float(wilcoxon_result["p_value"]), 6),
                    "significance_adj": significance_stars(float(wilcoxon_result["p_value"])),
                    "p_adjust_method": p_adjust,
                    "ci_level": round(float(ci), 4),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "cohens_d": cohens_d_value,
                    "cohens_d_magnitude": _cohens_d_magnitude(cohens_d_value),
                    "rank_biserial": rank_biserial_correlation(scores_a, scores_b),
                    "note": str(wilcoxon_result["note"]),
                }
            )

    result_df = pd.DataFrame(rows, columns=PAIRWISE_TEST_COLUMNS)
    if result_df.empty:
        return result_df

    if p_adjust != "none":
        for metric, metric_df in result_df.groupby("metric", sort=False):
            adjusted_values = _adjust_p_values(metric_df["p_value"].tolist(), method=p_adjust)
            result_df.loc[metric_df.index, "p_value_adj"] = adjusted_values
    result_df["p_value_adj"] = result_df["p_value_adj"].astype(float).round(6)
    result_df["significance_adj"] = result_df["p_value_adj"].apply(significance_stars)
    return result_df[PAIRWISE_TEST_COLUMNS]

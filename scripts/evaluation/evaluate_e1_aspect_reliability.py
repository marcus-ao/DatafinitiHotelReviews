"""Evaluate E1 aspect predictions against the finalized manual gold labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from scripts.pipeline import classify_sentence_aspects as aspect_module
from scripts.shared.experiment_utils import E1_LABELS_DIR, ROOT_DIR
from scripts.shared.project_utils import ASPECT_CATEGORIES, load_config


def parse_aspect_set(text: str) -> set[str]:
    if pd.isna(text):
        return set()
    items = [item.strip() for item in str(text).split(";") if item.strip()]
    return set(items)


def ordered_aspect_prediction(rows: list[dict]) -> list[str]:
    if not rows:
        return ["general"]
    ranked = sorted(
        rows,
        key=lambda item: (
            item.get("aspect") == "general",
            -float(item.get("confidence", 0.0)),
            item.get("aspect", ""),
        ),
    )
    return [item["aspect"] for item in ranked]


def validate_gold_annotations(merged: pd.DataFrame) -> None:
    required_cols = ["aspect_gold", "sentiment_gold", "is_multi_aspect"]
    missing_summary = {}
    for col in required_cols:
        missing = int(merged[col].fillna("").astype(str).str.strip().eq("").sum())
        if missing:
            missing_summary[col] = missing
    if missing_summary:
        details = ", ".join(f"{name}={count}" for name, count in missing_summary.items())
        raise ValueError(
            "E1 gold annotations are incomplete. "
            f"Please finish `aspect_sentiment_gold.csv` before evaluation: {details}."
        )


def collect_error_cases(
    merged: pd.DataFrame,
    predictions: dict[str, dict[str, list[str] | str]],
    strategy: str,
    limit: int = 5,
) -> list[dict]:
    cases = []
    for _, row in merged.iterrows():
        gold_set = row["gold_set"]
        predicted = predictions[row["sentence_id"]][strategy]
        predicted_main = predicted[0] if predicted else "general"
        gold_main = sorted(gold_set)[0] if gold_set else "general"
        if row["is_multi_aspect"]:
            pred_set = set(predicted) - {"general"}
            gold_core = set(gold_set) - {"general"}
            if pred_set != gold_core:
                cases.append(
                    {
                        "sentence_id": row["sentence_id"],
                        "city": row["city"],
                        "sample_bucket": row.get("sample_bucket", ""),
                        "gold_aspects": sorted(gold_set),
                        "predicted_aspects": predicted,
                        "sentence_text": row["sentence_text"],
                    }
                )
        elif predicted_main != gold_main:
            cases.append(
                {
                    "sentence_id": row["sentence_id"],
                    "city": row["city"],
                    "sample_bucket": row.get("sample_bucket", ""),
                    "gold_aspects": sorted(gold_set),
                    "predicted_aspects": predicted,
                    "sentence_text": row["sentence_text"],
                }
            )
        if len(cases) >= limit:
            break
    return cases


def sentence_sentiment_lookup() -> dict[str, str]:
    aspect_sentiment = pd.read_pickle(ROOT_DIR / "data/intermediate/aspect_sentiment.pkl")
    sentence_df = aspect_sentiment[["sentence_id", "sentiment"]].drop_duplicates("sentence_id")
    return dict(zip(sentence_df["sentence_id"], sentence_df["sentiment"]))


def compare_metric(hybrid_value: float, baseline_value: float) -> str:
    if hybrid_value > baseline_value:
        return "higher"
    if hybrid_value < baseline_value:
        return "lower"
    return "tied"


def top_confusions(labels: list[str], matrix: list[list[int]], top_n: int = 3) -> list[dict[str, int | str]]:
    confusions: list[dict[str, int | str]] = []
    for gold_idx, gold_label in enumerate(labels):
        for pred_idx, pred_label in enumerate(labels):
            if gold_idx == pred_idx:
                continue
            count = matrix[gold_idx][pred_idx]
            if not count:
                continue
            confusions.append(
                {
                    "gold": gold_label,
                    "pred": pred_label,
                    "count": count,
                }
            )
    confusions.sort(key=lambda item: (-int(item["count"]), str(item["gold"]), str(item["pred"])))
    return confusions[:top_n]


def format_confusion_table(labels: list[str], matrix: list[list[int]]) -> list[str]:
    header = "| gold \u2193 / pred \u2192 | " + " | ".join(labels) + " |"
    separator = "|" + "---|" * (len(labels) + 1)
    rows = [header, separator]
    for gold_label, row in zip(labels, matrix):
        rows.append("| " + gold_label + " | " + " | ".join(str(value) for value in row) + " |")
    return rows


def build_predictions(sample_df: pd.DataFrame, cfg: dict) -> dict[str, dict[str, list[str] | str]]:
    predictions: dict[str, dict[str, list[str] | str]] = {}
    needs_zs: list[tuple[str, str]] = []

    for _, row in sample_df.iterrows():
        sentence_id = row["sentence_id"]
        text = row["sentence_text"]
        rule_rows = aspect_module.match_aspects_rule(text)
        predictions[sentence_id] = {
            "rule_only": ordered_aspect_prediction(rule_rows),
            "hybrid": [],
        }
        if not rule_rows and len(text) >= cfg["aspect"]["zeroshot_min_char_len"]:
            needs_zs.append((sentence_id, text))
        else:
            predictions[sentence_id]["hybrid"] = ["general"] if not rule_rows else ordered_aspect_prediction(rule_rows)

    zeroshot_map: dict[str, list[dict]] = {}
    if needs_zs:
        from transformers import pipeline

        classifier = pipeline(
            "zero-shot-classification",
            model=cfg["aspect"]["zeroshot_model"],
            device=aspect_module.get_transformers_device(),
            batch_size=cfg["aspect"]["zeroshot_batch_size"],
        )
        zeroshot_map = aspect_module.batch_zeroshot(
            needs_zs,
            classifier,
            cfg["aspect"]["zeroshot_threshold"],
            cfg["aspect"]["zeroshot_batch_size"],
        )

    for _, row in sample_df.iterrows():
        sentence_id = row["sentence_id"]
        text = row["sentence_text"]
        zs_rows = zeroshot_map.get(sentence_id, [])
        predictions[sentence_id]["zeroshot_only"] = ordered_aspect_prediction(zs_rows)
        if not predictions[sentence_id]["hybrid"]:
            predictions[sentence_id]["hybrid"] = ordered_aspect_prediction(zs_rows)
        if len(text) < cfg["aspect"]["zeroshot_min_char_len"] and not zs_rows:
            predictions[sentence_id]["zeroshot_only"] = ["general"]

    sentiment_lookup = sentence_sentiment_lookup()
    for sentence_id in predictions:
        predictions[sentence_id]["sentiment"] = sentiment_lookup.get(sentence_id, "neutral")

    return predictions


def evaluate(sample_df: pd.DataFrame, gold_df: pd.DataFrame, output_dir: Path) -> None:
    cfg = load_config()
    merged = sample_df.merge(
        gold_df,
        on=["sentence_id", "hotel_id", "city", "sentence_text"],
        how="inner",
    )
    validate_gold_annotations(merged)
    sample_count = int(len(sample_df))
    gold_count = int(len(gold_df))
    merged_count = int(len(merged))
    merged["gold_set"] = merged["aspect_gold"].apply(parse_aspect_set)
    merged["is_multi_aspect"] = (
        merged["is_multi_aspect"].astype(str).str.strip().isin(["1", "true", "True", "TRUE"])
    )

    predictions = build_predictions(sample_df, cfg)
    main_mask = (~merged["is_multi_aspect"]) & (merged["gold_set"].map(len) == 1)
    difficult_mask = merged["is_multi_aspect"]
    labels = ASPECT_CATEGORIES + ["general"]

    metrics: dict[str, dict] = {}
    for strategy in ["rule_only", "zeroshot_only", "hybrid"]:
        y_true = [next(iter(items)) for items in merged.loc[main_mask, "gold_set"]]
        y_pred = [predictions[sentence_id][strategy][0] for sentence_id in merged.loc[main_mask, "sentence_id"]]
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()

        jaccard_scores: list[float] = []
        for _, row in merged.loc[difficult_mask, ["sentence_id", "gold_set"]].iterrows():
            pred_set = set(predictions[row["sentence_id"]][strategy]) - {"general"}
            gold_set = set(row["gold_set"]) - {"general"}
            union = pred_set | gold_set
            score = 0.0 if not union else len(pred_set & gold_set) / len(union)
            jaccard_scores.append(score)

        metrics[strategy] = {
            "aspect_macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
            "difficult_jaccard": round(sum(jaccard_scores) / max(len(jaccard_scores), 1), 4),
            "confusion_labels": labels,
            "confusion_matrix": matrix,
            "representative_errors": collect_error_cases(
                merged.loc[main_mask | difficult_mask].copy(),
                predictions,
                strategy,
            ),
        }

    sentiment_true = merged["sentiment_gold"].fillna("unclear").tolist()
    sentiment_pred = [predictions[sentence_id]["sentiment"] for sentence_id in merged["sentence_id"]]
    sentiment_labels = ["positive", "negative", "neutral", "unclear"]
    metrics["sentiment"] = {
        "macro_f1": round(
            float(
                f1_score(
                    sentiment_true,
                    sentiment_pred,
                    labels=sentiment_labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        )
    }
    metrics["data_summary"] = {
        "sample_rows": sample_count,
        "gold_rows": gold_count,
        "merged_rows": merged_count,
        "excluded_rows": sample_count - merged_count,
        "main_eval_rows": int(main_mask.sum()),
        "difficult_eval_rows": int(difficult_mask.sum()),
    }

    hybrid_confusions = top_confusions(
        metrics["hybrid"]["confusion_labels"],
        metrics["hybrid"]["confusion_matrix"],
    )
    metrics["report_summary"] = {
        "hybrid_vs_rule_only_aspect_macro_f1": compare_metric(
            metrics["hybrid"]["aspect_macro_f1"],
            metrics["rule_only"]["aspect_macro_f1"],
        ),
        "hybrid_vs_zeroshot_only_aspect_macro_f1": compare_metric(
            metrics["hybrid"]["aspect_macro_f1"],
            metrics["zeroshot_only"]["aspect_macro_f1"],
        ),
        "hybrid_vs_rule_only_difficult_jaccard": compare_metric(
            metrics["hybrid"]["difficult_jaccard"],
            metrics["rule_only"]["difficult_jaccard"],
        ),
        "hybrid_vs_zeroshot_only_difficult_jaccard": compare_metric(
            metrics["hybrid"]["difficult_jaccard"],
            metrics["zeroshot_only"]["difficult_jaccard"],
        ),
        "top_confusions_hybrid": hybrid_confusions,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "e1_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    lines = [
        "# E1 Evaluation Result",
        "",
        "## Data Summary",
        "",
        f"- Sample rows: {sample_count}",
        f"- Official gold rows: {gold_count}",
        f"- Strictly merged evaluation rows: {merged_count}",
        f"- Excluded rows after strict merge: {sample_count - merged_count}",
        f"- Main-set rows: {int(main_mask.sum())}",
        f"- Difficult-set rows: {int(difficult_mask.sum())}",
        "",
        "## Aspect Metrics",
        "",
    ]
    for strategy in ["rule_only", "zeroshot_only", "hybrid"]:
        lines.extend(
            [
                f"### {strategy}",
                "",
                f"- Aspect macro-F1: {metrics[strategy]['aspect_macro_f1']}",
                f"- Difficult-set Jaccard: {metrics[strategy]['difficult_jaccard']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Sentiment",
            "",
            f"- Sentiment macro-F1: {metrics['sentiment']['macro_f1']}",
            "",
            "## Key Findings",
            "",
            (
                "- Hybrid vs rule_only: "
                f"Aspect macro-F1 is {metrics['report_summary']['hybrid_vs_rule_only_aspect_macro_f1']}, "
                f"Difficult-set Jaccard is {metrics['report_summary']['hybrid_vs_rule_only_difficult_jaccard']}."
            ),
            (
                "- Hybrid vs zeroshot_only: "
                f"Aspect macro-F1 is {metrics['report_summary']['hybrid_vs_zeroshot_only_aspect_macro_f1']}, "
                f"Difficult-set Jaccard is {metrics['report_summary']['hybrid_vs_zeroshot_only_difficult_jaccard']}."
            ),
        ]
    )
    if hybrid_confusions:
        lines.append("- Most confused aspect pairs under hybrid:")
        for item in hybrid_confusions:
            lines.append(f"  - gold={item['gold']} -> pred={item['pred']} ({item['count']})")
        lines.append("")
    else:
        lines.extend(["- Most confused aspect pairs under hybrid: none", ""])
    lines.extend(["## Confusion Matrix", ""])
    for strategy in ["rule_only", "zeroshot_only", "hybrid"]:
        lines.append(f"### {strategy}")
        lines.append("")
        lines.extend(format_confusion_table(metrics[strategy]["confusion_labels"], metrics[strategy]["confusion_matrix"]))
        lines.append("")
    lines.extend(["## Representative Error Cases", ""])
    for strategy in ["rule_only", "zeroshot_only", "hybrid"]:
        lines.append(f"### {strategy}")
        lines.append("")
        errors = metrics[strategy]["representative_errors"]
        if not errors:
            lines.append("- none")
            lines.append("")
            continue
        for idx, case in enumerate(errors, start=1):
            lines.extend(
                [
                    f"{idx}. `{case['sentence_id']}` | gold={case['gold_aspects']} | pred={case['predicted_aspects']}",
                    f"   {case['sentence_text']}",
                ]
            )
        lines.append("")
    lines.extend(["## Notes", "", "- `aspect_sentiment_gold.csv` is the only official E1 gold.", "- Merchant-reply rows remain excluded from the final gold set."])
    (output_dir / "e1_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-csv", default=str(E1_LABELS_DIR / "aspect_sentiment_eval_sample.csv"))
    parser.add_argument("--gold-csv", default=str(E1_LABELS_DIR / "aspect_sentiment_gold.csv"))
    parser.add_argument("--output-dir", default=str(E1_LABELS_DIR))
    args = parser.parse_args()

    sample_df = pd.read_csv(args.sample_csv)
    gold_df = pd.read_csv(args.gold_csv)
    evaluate(sample_df, gold_df, Path(args.output_dir))
    print(f"[OK] E1 results written to {args.output_dir}")


if __name__ == "__main__":
    main()

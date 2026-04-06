# E4 Clarification Result

## Main Summary

| group_id | query_count | clarification_accuracy | precision | recall | f1 | over_clarification_rate | under_clarification_rate | schema_valid_rate | avg_latency_ms | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|
| A_rule_clarify | 70 | 0.9714 | 0.0 | 0.0 | 0.0 | 0.0286 | 0.0 | 1.0 | 0.046 | 8267fde1784a2747 |
| B_base_llm_clarify | 70 | 0.9857 | 0.0 | 0.0 | 0.0 | 0.0143 | 0.0 | 1.0 | 368.348 | 442ef87210acceb0 |

## Balanced Diagnostic Subset

| group_id | query_count | clarification_accuracy | precision | recall | f1 | over_clarification_rate | under_clarification_rate | schema_valid_rate | avg_latency_ms | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|
| A_rule_clarify | 16 | 0.9375 | 0.0 | 0.0 | 0.0 | 0.0625 | 0.0 | 1.0 | 0.059 | 8267fde1784a2747 |
| B_base_llm_clarify | 16 | 0.9375 | 0.0 | 0.0 | 0.0 | 0.0625 | 0.0 | 1.0 | 392.298 | 442ef87210acceb0 |

## Representative Cases

### A_rule_clarify

- Over-clarification:
  - `q013` 我在Chicago想住得安静一点，但不要安静睡眠太差的酒店。
  - `q043` 我在San Francisco想住得安静一点，但不要安静睡眠太差的酒店。

### B_base_llm_clarify

- Over-clarification:
  - `q013` 我在Chicago想住得安静一点，但不要安静睡眠太差的酒店。

## Audit Asset

- Run-local: `e4_29957e53b85462b3_20260406T114447+0000/clarification_question_audit.csv`
- Latest copy: `experiments/labels/e4_clarification/clarification_question_audit.csv`
# E4 Clarification Result

## Main Summary

| group_id | query_count | clarification_accuracy | precision | recall | f1 | over_clarification_rate | under_clarification_rate | schema_valid_rate | avg_latency_ms | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|
| A_rule_clarify | 86 | 0.9767 | 0.8889 | 1.0 | 0.9412 | 0.0286 | 0.0 | 1.0 | 0.032 | 209504874e1291b2 |
| B_base_llm_clarify | 86 | 0.814 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 240.776 | 9c43909a34453b81 |

## Balanced Diagnostic Subset

| group_id | query_count | clarification_accuracy | precision | recall | f1 | over_clarification_rate | under_clarification_rate | schema_valid_rate | avg_latency_ms | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|
| A_rule_clarify | 32 | 0.9688 | 0.9412 | 1.0 | 0.9697 | 0.0625 | 0.0 | 1.0 | 0.087 | 209504874e1291b2 |
| B_base_llm_clarify | 32 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 647.086 | 9c43909a34453b81 |

## Representative Cases

### A_rule_clarify

- Over-clarification:
  - `q013` 我在Chicago想住得安静一点，但不要安静睡眠太差的酒店。
  - `q043` 我在San Francisco想住得安静一点，但不要安静睡眠太差的酒店。

### B_base_llm_clarify

- Under-clarification:
  - `q051` 我想找一家位置交通好的酒店，你先帮我想想。
  - `q052` 我想找一家卫生干净好的酒店，你先帮我想想。

## Audit Asset

- `experiments/labels/e4_clarification/clarification_question_audit.csv`
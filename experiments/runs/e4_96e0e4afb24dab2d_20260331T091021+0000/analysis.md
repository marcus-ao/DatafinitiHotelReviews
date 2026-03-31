# E4 Clarification Result

## Main Summary

| group_id | query_count | clarification_accuracy | precision | recall | f1 | over_clarification_rate | under_clarification_rate | schema_valid_rate | avg_latency_ms | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|
| A_rule_clarify | 32 | 0.9688 | 0.9412 | 1.0 | 0.9697 | 0.0625 | 0.0 | 1.0 | 0.043 | d21cd5f3d8eb4a15 |
| B_base_llm_clarify | 32 | 0.7812 | 1.0 | 0.5625 | 0.72 | 0.0 | 0.4375 | 1.0 | 167.173 | a47c1dd3d4b4355f |

## Balanced Diagnostic Subset

| group_id | query_count | clarification_accuracy | precision | recall | f1 | over_clarification_rate | under_clarification_rate | schema_valid_rate | avg_latency_ms | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|
| A_rule_clarify | 32 | 0.9688 | 0.9412 | 1.0 | 0.9697 | 0.0625 | 0.0 | 1.0 | 0.043 | d21cd5f3d8eb4a15 |
| B_base_llm_clarify | 32 | 0.7812 | 1.0 | 0.5625 | 0.72 | 0.0 | 0.4375 | 1.0 | 167.173 | a47c1dd3d4b4355f |

## Representative Cases

### A_rule_clarify

- Over-clarification:
  - `q013` 我在Chicago想住得安静一点，但不要安静睡眠太差的酒店。

### B_base_llm_clarify

- Under-clarification:
  - `q053` 我想找一家服务好的酒店，你先帮我想想。
  - `q054` 我想找一家房间设施好的酒店，你先帮我想想。

## Audit Asset

- Run-local: `e4_96e0e4afb24dab2d_20260331T091021+0000/clarification_question_audit.csv`
- Latest copy: `experiments/labels/e4_clarification/clarification_question_audit.csv`
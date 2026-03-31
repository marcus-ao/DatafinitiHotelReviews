# E4 Clarification Result

## Main Summary

| group_id | query_count | clarification_accuracy | precision | recall | f1 | over_clarification_rate | under_clarification_rate | schema_valid_rate | avg_latency_ms | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|
| A_rule_clarify | 86 | 0.9767 | 0.8889 | 1.0 | 0.9412 | 0.0286 | 0.0 | 1.0 | 0.031 | c0c11bfa8affceb4 |
| B_base_llm_clarify | 86 | 0.9884 | 0.9412 | 1.0 | 0.9697 | 0.0143 | 0.0 | 1.0 | 187.554 | 67d9f4726515fb6d |

## Balanced Diagnostic Subset

| group_id | query_count | clarification_accuracy | precision | recall | f1 | over_clarification_rate | under_clarification_rate | schema_valid_rate | avg_latency_ms | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|
| A_rule_clarify | 32 | 0.9688 | 0.9412 | 1.0 | 0.9697 | 0.0625 | 0.0 | 1.0 | 0.083 | c0c11bfa8affceb4 |
| B_base_llm_clarify | 32 | 0.9688 | 0.9412 | 1.0 | 0.9697 | 0.0625 | 0.0 | 1.0 | 504.051 | 67d9f4726515fb6d |

## Representative Cases

### A_rule_clarify

- Over-clarification:
  - `q013` 我在Chicago想住得安静一点，但不要安静睡眠太差的酒店。
  - `q043` 我在San Francisco想住得安静一点，但不要安静睡眠太差的酒店。

### B_base_llm_clarify

- Over-clarification:
  - `q013` 我在Chicago想住得安静一点，但不要安静睡眠太差的酒店。

## Audit Asset

- Run-local: `e4_55c8021e1119fb77_20260331T122648+0000/clarification_question_audit.csv`
- Latest copy: `experiments/labels/e4_clarification/clarification_question_audit.csv`
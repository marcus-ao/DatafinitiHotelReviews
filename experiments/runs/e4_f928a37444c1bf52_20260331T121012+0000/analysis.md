# E4 Clarification Result

## Main Summary

| group_id | query_count | clarification_accuracy | precision | recall | f1 | over_clarification_rate | under_clarification_rate | schema_valid_rate | avg_latency_ms | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|
| A_rule_clarify | 32 | 0.9688 | 0.9412 | 1.0 | 0.9697 | 0.0625 | 0.0 | 1.0 | 0.044 | 1bcea30ab01e497f |
| B_base_llm_clarify | 32 | 0.9688 | 0.9412 | 1.0 | 0.9697 | 0.0625 | 0.0 | 1.0 | 248.291 | fd9db38b84e79e7e |

## Balanced Diagnostic Subset

| group_id | query_count | clarification_accuracy | precision | recall | f1 | over_clarification_rate | under_clarification_rate | schema_valid_rate | avg_latency_ms | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|
| A_rule_clarify | 32 | 0.9688 | 0.9412 | 1.0 | 0.9697 | 0.0625 | 0.0 | 1.0 | 0.044 | 1bcea30ab01e497f |
| B_base_llm_clarify | 32 | 0.9688 | 0.9412 | 1.0 | 0.9697 | 0.0625 | 0.0 | 1.0 | 248.291 | fd9db38b84e79e7e |

## Representative Cases

### A_rule_clarify

- Over-clarification:
  - `q013` 我在Chicago想住得安静一点，但不要安静睡眠太差的酒店。

### B_base_llm_clarify

- Over-clarification:
  - `q013` 我在Chicago想住得安静一点，但不要安静睡眠太差的酒店。

## Audit Asset

- Run-local: `e4_f928a37444c1bf52_20260331T121012+0000/clarification_question_audit.csv`
- Latest copy: `experiments/labels/e4_clarification/clarification_question_audit.csv`
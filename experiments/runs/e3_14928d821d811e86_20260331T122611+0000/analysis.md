# E3 Preference Parsing Result

## Summary Table

| group_id | query_count | schema_valid_rate | exact_match_rate | unsupported_detection_recall | city_slot_f1 | hotel_category_slot_f1 | focus_aspects_slot_f1 | avoid_aspects_slot_f1 | unsupported_requests_slot_f1 | avg_latency_ms | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_rule_parser | 86 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.029 | b0463460071e582b |
| B_base_llm_structured | 86 | 1.0 | 0.9767 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9474 | 0.9836 | 371.14 | cd6bc98fd141f072 |

## Error Breakdown

### A_rule_parser

- city_missing: 0
- aspect_mapping_error: 0
- unsupported_missed: 0
- unsupported_as_supported: 0

- none

### B_base_llm_structured

- city_missing: 0
- aspect_mapping_error: 2
- unsupported_missed: 0
- unsupported_as_supported: 0

Representative cases:
- `q048` 我在Seattle想住得安静一点，但不要性价比太差的酒店。 | errors=aspect_mapping_error,unknown_unsupported_label
- `q062` 我想在New Orleans找一家性价比很好，但又最好别太强调性价比的酒店。 | errors=aspect_mapping_error

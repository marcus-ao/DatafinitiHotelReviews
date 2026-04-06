# E3 Preference Parsing Result

## Summary Table

| group_id | query_count | schema_valid_rate | exact_match_rate | unsupported_detection_recall | city_slot_f1 | hotel_category_slot_f1 | focus_aspects_slot_f1 | avoid_aspects_slot_f1 | unsupported_requests_slot_f1 | avg_latency_ms | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_rule_parser | 70 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.039 | 1ccf693fde704746 |
| B_base_llm_structured | 70 | 1.0 | 0.9857 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9474 | 1.0 | 586.971 | a061a5f936fba8f1 |

## Error Breakdown

### A_rule_parser

- city_missing: 0
- aspect_mapping_error: 0
- unsupported_missed: 0
- unsupported_as_supported: 0

- none

### B_base_llm_structured

- city_missing: 0
- aspect_mapping_error: 1
- unsupported_missed: 0
- unsupported_as_supported: 0

Representative cases:
- `q048` 我在Seattle想住得安静一点，但不要性价比太差的酒店。 | errors=aspect_mapping_error,unknown_unsupported_label

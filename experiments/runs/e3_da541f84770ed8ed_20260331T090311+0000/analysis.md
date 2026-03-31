# E3 Preference Parsing Result

## Summary Table

| group_id | query_count | schema_valid_rate | exact_match_rate | unsupported_detection_recall | city_slot_f1 | hotel_category_slot_f1 | focus_aspects_slot_f1 | avoid_aspects_slot_f1 | unsupported_requests_slot_f1 | avg_latency_ms | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_rule_parser | 23 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.045 | 69c01eff1a0fadb9 |
| B_base_llm_structured | 23 | 1.0 | 0.5652 | 0.7778 | 0.9565 | 1.0 | 0.9091 | 0.2222 | 0.8235 | 2547.109 | bcfd803a53622a2d |

## Error Breakdown

### A_rule_parser

- city_missing: 0
- aspect_mapping_error: 0
- unsupported_missed: 0
- unsupported_as_supported: 0

- none

### B_base_llm_structured

- city_missing: 1
- aspect_mapping_error: 9
- unsupported_missed: 2
- unsupported_as_supported: 1

Representative cases:
- `q005` 我想在Anaheim找一家离景点步行 10 分钟内、而且卫生干净好的酒店。 | errors=aspect_mapping_error
- `q025` 我想在Honolulu找一家离景点步行 10 分钟内、而且性价比好的酒店。 | errors=aspect_mapping_error,unsupported_missed
- `q040` 我想在San Diego找一家离景点步行 10 分钟内、而且服务好的酒店。 | errors=aspect_mapping_error,unsupported_as_supported,unsupported_missed

# E3 Preference Parsing Result

## Summary Table

| group_id | query_count | schema_valid_rate | exact_match_rate | unsupported_detection_recall | city_slot_f1 | hotel_category_slot_f1 | focus_aspects_slot_f1 | avoid_aspects_slot_f1 | unsupported_requests_slot_f1 | avg_latency_ms | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_rule_parser | 86 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.028 | 84346793ccd0b45f |
| B_base_llm_structured | 86 | 1.0 | 0.5698 | 0.0667 | 0.9186 | 1.0 | 0.8258 | 1.0 | 0.125 | 469.137 | 09a8635adb19baf5 |

## Error Breakdown

### A_rule_parser

- city_missing: 0
- aspect_mapping_error: 0
- unsupported_missed: 0
- unsupported_as_supported: 0

- none

### B_base_llm_structured

- city_missing: 7
- aspect_mapping_error: 31
- unsupported_missed: 28
- unsupported_as_supported: 28

Representative cases:
- `q002` 请推荐Anaheim位置交通和卫生干净都不错的酒店。 | errors=city_missing
- `q003` 我在Anaheim想住得安静一点，但不要服务太差的酒店。 | errors=city_missing
- `q004` 帮我找Anaheim预算在 600 元以内，而且位置交通不错的酒店。 | errors=aspect_mapping_error,city_missing,unsupported_as_supported,unsupported_missed

# E8 Retrieval Result

## Summary Table

| group_id | query_count | target_unit_count | avg_latency_ms | config_hash | ndcg_at_5 | precision_at_5 | evidence_insufficiency_rate | fallback_activation_rate | fallback_noise_rate |
|---|---|---|---|---|---|---|---|---|---|
| aspect_main_rerank | 40 | 80 | 351.632 | 014f03702e566d1f | 0.6378 | 0.6375 | 0.05 | 0.0 | 0.0 |
| aspect_main_fallback_rerank | 40 | 80 | 328.067 | 39d8b220c4febe67 | 0.6378 | 0.6375 | 0.05 | 0.05 | 1.0 |

## Notes

- Compare strict main-channel retrieval against main + fallback retrieval.

## Representative Improvements

- none

## Representative Regressions

- none

## Fallback Cases

- `q021` | focus:quiet_sleep | reason=sentence_count<2; unique_reviews<2
- `q022` | focus:quiet_sleep | reason=sentence_count<2; unique_reviews<2
- `q023` | focus:quiet_sleep | reason=sentence_count<2; unique_reviews<2
- `q081` | focus:quiet_sleep | reason=sentence_count<2; unique_reviews<2
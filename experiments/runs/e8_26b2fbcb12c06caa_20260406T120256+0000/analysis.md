# E8 Retrieval Result

## Summary Table

| group_id | query_count | target_unit_count | avg_latency_ms | config_hash | aspect_recall_at_5 | ndcg_at_5 | precision_at_5 | mrr_at_5 | evidence_diversity_at_5 | evidence_insufficiency_rate | fallback_activation_rate | fallback_noise_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aspect_main_rerank | 40 | 80 | 758.186 | 6614b6bb818d6943 | 0.85 | 0.6378 | 0.6375 | 0.7842 | 0.8925 | 0.05 | 0.0 | 0.0 |
| aspect_main_fallback_rerank | 40 | 80 | 784.804 | 8394541e388ba685 | 0.85 | 0.6378 | 0.6375 | 0.7842 | 0.9425 | 0.05 | 0.05 | 1.0 |

## Notes

- Compare strict main-channel retrieval against main + fallback retrieval.
- Unified retrieval metrics reported for all retrieval-side runs: Aspect Recall@5, nDCG@5, Precision@5, MRR@5, Evidence Diversity@5, Retrieval Latency.

## Representative Improvements

- none

## Representative Regressions

- none

## Fallback Cases

- `q021` | focus:quiet_sleep | reason=sentence_count<2; unique_reviews<2
- `q022` | focus:quiet_sleep | reason=sentence_count<2; unique_reviews<2
- `q023` | focus:quiet_sleep | reason=sentence_count<2; unique_reviews<2
- `q081` | focus:quiet_sleep | reason=sentence_count<2; unique_reviews<2
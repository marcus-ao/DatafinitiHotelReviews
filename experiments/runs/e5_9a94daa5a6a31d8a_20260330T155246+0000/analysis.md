# E5 Query Bridge Result

## Summary Table

| group_id | query_count | target_unit_count | avg_latency_ms | aspect_recall_at_5 | ndcg_at_5 | mrr_at_5 | precision_at_5 | config_hash |
|---|---|---|---|---|---|---|---|---|
| A_zh_direct_dense_no_rerank | 40 | 80 | 138.282 | 0.625 | 0.2643 | 0.4233 | 0.22 | e4082234bbc8e690 |
| B_structured_query_en_dense_no_rerank | 40 | 80 | 130.228 | 0.85 | 0.6457 | 0.7781 | 0.645 | f7fe571e2f5d6edd |

## Role Breakdown

### A_zh_direct_dense_no_rerank

- avoid: nDCG@5=0.0161, Precision@5=0.02
- focus: nDCG@5=0.2998, Precision@5=0.2486

### B_structured_query_en_dense_no_rerank

- avoid: nDCG@5=0.1519, Precision@5=0.12
- focus: nDCG@5=0.7162, Precision@5=0.72

## Aspect Dependence

### A_zh_direct_dense_no_rerank

- worst aspects: [(0.0979, 'room_facilities'), (0.2434, 'service'), (0.2576, 'quiet_sleep')]
- best aspects: [(0.2617, 'cleanliness'), (0.3303, 'location_transport'), (0.4526, 'value')]

### B_structured_query_en_dense_no_rerank

- worst aspects: [(0.5372, 'quiet_sleep'), (0.5937, 'value'), (0.5954, 'location_transport')]
- best aspects: [(0.6426, 'room_facilities'), (0.6981, 'service'), (0.8716, 'cleanliness')]

## Interpretation

- This run compares direct Chinese dense retrieval against structured English retrieval under the same candidate set and `aspect_main_no_rerank` backend.
- If `avoid` and `quiet_sleep` remain weak in both groups, the bottleneck is evidence coverage rather than bridge language alone.
# E5 Query Bridge Result

## Summary Table

| group_id | query_count | target_unit_count | avg_latency_ms | config_hash | aspect_recall_at_5 | ndcg_at_5 | precision_at_5 | mrr_at_5 | evidence_diversity_at_5 |
|---|---|---|---|---|---|---|---|---|---|
| A_zh_direct_dense_no_rerank | 40 | 80 | 195.182 | 40e0f7fe6c36ab5e | 0.6 | 0.2735 | 0.23 | 0.4054 | 0.8675 |
| B_structured_query_en_dense_no_rerank | 40 | 80 | 197.333 | 097d76b0e4ed4c03 | 0.85 | 0.6457 | 0.645 | 0.7781 | 0.875 |

## Role Breakdown

### A_zh_direct_dense_no_rerank

- avoid: Recall@5=0.1, nDCG@5=0.0348, Precision@5=0.04, MRR@5=0.1, Diversity@5=0.8867

- focus: Recall@5=0.6714, nDCG@5=0.3076, Precision@5=0.2571, MRR@5=0.449, Diversity@5=0.8648

### B_structured_query_en_dense_no_rerank

- avoid: Recall@5=0.3, nDCG@5=0.1519, Precision@5=0.12, MRR@5=0.1833, Diversity@5=0.8667

- focus: Recall@5=0.9286, nDCG@5=0.7162, Precision@5=0.72, MRR@5=0.8631, Diversity@5=0.8762

## Aspect Dependence

### A_zh_direct_dense_no_rerank

- worst aspects: [(0.0839, 'room_facilities'), (0.2327, 'service'), (0.25, 'cleanliness')]
- best aspects: [(0.268, 'location_transport'), (0.2694, 'quiet_sleep'), (0.616, 'value')]

### B_structured_query_en_dense_no_rerank

- worst aspects: [(0.5372, 'quiet_sleep'), (0.5937, 'value'), (0.5954, 'location_transport')]
- best aspects: [(0.6426, 'room_facilities'), (0.6981, 'service'), (0.8716, 'cleanliness')]

## Interpretation

- This run compares direct Chinese dense retrieval against structured English retrieval under the same candidate set and `aspect_main_no_rerank` backend.
- All retrieval-side summaries now follow the unified six-metric schema: Aspect Recall@5, nDCG@5, Precision@5, MRR@5, Evidence Diversity@5, Retrieval Latency.
- If `avoid` and `quiet_sleep` remain weak in both groups, the bottleneck is evidence coverage rather than bridge language alone.
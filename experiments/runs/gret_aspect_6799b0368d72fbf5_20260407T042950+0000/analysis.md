# G-Series Aspect Retrieval Evaluation

## Summary Table

| group_id | query_count | target_unit_count | avg_latency_ms | config_hash | aspect_recall_at_5 | ndcg_at_5 | precision_at_5 | mrr_at_5 | evidence_diversity_at_5 | retrieval_variant | retrieval_mode | candidate_policy | retrieval_summary_source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aspect_retrieval | 68 | 108 | 238.948 | 6799b0368d72fbf5 | 0.8981 | 0.6996 | 0.6981 | 0.8333 | 0.9037 | aspect | aspect_main_no_rerank | G_aspect_retrieval_top5 | formal_retrieval_eval |

## Notes

- Retrieval variant: `aspect`
- Retrieval mode: `aspect_main_no_rerank`
- Candidate policy: `G_aspect_retrieval_top5`
- Metrics are from formal qrels-based retrieval evaluation over the G-series query set.
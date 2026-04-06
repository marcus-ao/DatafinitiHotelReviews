# E6 Retrieval Result

## Summary Table

| group_id | query_count | target_unit_count | avg_latency_ms | config_hash | aspect_recall_at_5 | ndcg_at_5 | precision_at_5 | mrr_at_5 | evidence_diversity_at_5 |
|---|---|---|---|---|---|---|---|---|---|
| plain_city_test_rerank | 40 | 80 | 831.788 | 157265443c5b953d | 0.7 | 0.3307 | 0.335 | 0.575 | 0.9475 |
| aspect_main_rerank | 40 | 80 | 755.667 | d61bf5bf7cc2b468 | 0.85 | 0.6378 | 0.6375 | 0.7842 | 0.8925 |

## Notes

- Compare plain retrieval against aspect-aware retrieval under the same city-test candidate set.
- Unified retrieval metrics reported for all retrieval-side runs: Aspect Recall@5, nDCG@5, Precision@5, MRR@5, Evidence Diversity@5, Retrieval Latency.

## Representative Improvements

- `q079` | focus:quiet_sleep | ΔnDCG@5=1.0
  query: 请推荐Chicago在服务、安静睡眠、位置交通三方面都比较均衡的酒店。
- `q080` | focus:value | ΔnDCG@5=1.0
  query: 请推荐Dallas在房间设施、性价比、卫生干净三方面都比较均衡的酒店。
- `q084` | focus:value | ΔnDCG@5=1.0
  query: 请推荐San Diego在卫生干净、房间设施、性价比三方面都比较均衡的酒店。

## Representative Regressions

- `q033` | avoid:service | ΔnDCG@5=-0.4307
  query: 我在Orlando想住得安静一点，但不要服务太差的酒店。
- `q008` | avoid:room_facilities | ΔnDCG@5=-0.3066
  query: 我在Atlanta想住得安静一点，但不要房间设施太差的酒店。
- `q017` | focus:quiet_sleep | ΔnDCG@5=-0.2262
  query: 请推荐Dallas房间设施和安静睡眠都不错的酒店。
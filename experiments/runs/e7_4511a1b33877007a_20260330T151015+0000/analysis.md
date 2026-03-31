# E7 Retrieval Result

## Summary Table

| group_id | query_count | target_unit_count | avg_latency_ms | config_hash | ndcg_at_5 | mrr_at_5 | precision_at_5 |
|---|---|---|---|---|---|---|---|
| aspect_main_no_rerank | 40 | 80 | 133.256 | 9480fc69e334fb8d | 0.6457 | 0.7781 | 0.645 |
| aspect_main_rerank | 40 | 80 | 345.117 | 9f882aa589c8b883 | 0.6378 | 0.7842 | 0.6375 |

## Notes

- Compare dense-only ranking against dense + cross-encoder reranking.

## Representative Improvements

- `q048` | avoid:value | ΔnDCG@5=0.6309
  query: 我在Seattle想住得安静一点，但不要性价比太差的酒店。
- `q080` | focus:value | ΔnDCG@5=0.5693
  query: 请推荐Dallas在房间设施、性价比、卫生干净三方面都比较均衡的酒店。
- `q037` | focus:service | ΔnDCG@5=0.3008
  query: 请推荐San Diego卫生干净和服务都不错的酒店。

## Representative Regressions

- `q079` | focus:service | ΔnDCG@5=-0.4142
  query: 请推荐Chicago在服务、安静睡眠、位置交通三方面都比较均衡的酒店。
- `q081` | focus:location_transport | ΔnDCG@5=-0.3162
  query: 请推荐Honolulu在安静睡眠、位置交通、服务三方面都比较均衡的酒店。
- `q008` | avoid:room_facilities | ΔnDCG@5=-0.3066
  query: 我在Atlanta想住得安静一点，但不要房间设施太差的酒店。
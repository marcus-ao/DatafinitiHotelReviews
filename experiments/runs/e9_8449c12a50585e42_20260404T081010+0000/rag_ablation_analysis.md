# E9 RAG Ablation Result

## Summary Table

| group_id | compare_role | query_count | citation_precision | evidence_verifiability_mean | unsupported_honesty_rate | schema_valid_rate | recommendation_coverage | avg_latency_ms | retry_trigger_rate | fallback_to_honest_notice_rate | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B_grounded_generation | with_rag | 40 | 0.95 | 1.9924 | 1.0 | 1.0 | 0.95 | 2182.405 | 0.0 | 0.0 | 1360500f65c2aa26 |
| D_no_evidence_generation | without_rag | 40 | 0.0 | 0.0 | 1.0 | 0.975 | 0.825 | 1052.184 | 0.0 | 0.0 | 01226814e2603934 |

## Primary Conclusion

- primary compare: `B_grounded_generation` vs `D_no_evidence_generation`
- recommendation_coverage: with-RAG=0.95 | without-RAG=0.825 | Δ=0.125
- schema_valid_rate: with-RAG=1.0 | without-RAG=0.975 | Δ=0.025
- citation_precision (auxiliary only): with-RAG=0.95 | without-RAG=0.0
- evidence_verifiability_mean (auxiliary only): with-RAG=1.9924 | without-RAG=0.0
- interpretation: `D_no_evidence_generation` 不看证据，因此 citation/evidence 指标只作辅助解释；主判断应放在 recommendation coverage 与 schema stability。

## Recommendation Recovery Cases

- `q008` | rag_recs=2 | no_rag_recs=0 | Δrecs=2 | rag_summary=推荐亚特兰大安静且设施良好的酒店。 | no_rag_summary=基于名称和评分，无法提供符合安静且设施优良要求的可靠推荐。
- `q033` | rag_recs=2 | no_rag_recs=0 | Δrecs=2 | rag_summary=Orlando 安静且服务良好的酒店推荐 | no_rag_summary=Orlando 安静且服务尚可的酒店推荐
- `q003` | rag_recs=1 | no_rag_recs=0 | Δrecs=1 | rag_summary=该酒店安静且服务良好，符合用户偏好。 | no_rag_summary=基于名称推荐，但缺乏服务与安静证据。
- `q013` | rag_recs=1 | no_rag_recs=0 | Δrecs=1 | rag_summary=仅推荐有睡眠相关证据的酒店。 | no_rag_summary=基于名称推荐两家芝加哥酒店，但缺乏具体评论证据。
- `q043` | rag_recs=1 | no_rag_recs=0 | Δrecs=1 | rag_summary=Hotel Diva 相对安静，但存在街道噪音。 | no_rag_summary=基于名称推荐旧金山酒店，但缺乏具体评价证据。

## Matched Abstentions

- `q023` | rag_notice=用户要求安静睡眠，但候选酒店 Ramada Plaza By Wyndham Waikiki 的可用证据仅支持位置便利，未提供任何关于房间安静或睡眠质量的正面描述，因此无法支持该推荐。 | no_rag_notice=当前无评论证据，无法基于用户偏好提供有依据的酒店推荐。

## Suspicious No-RAG Wins

- `q021` | rag_recs=0 | no_rag_recs=1 | rag_summary=n/a | no_rag_summary=基于名称推测，推荐Honolulu的Ramada Plaza By Wyndham Waikiki酒店。
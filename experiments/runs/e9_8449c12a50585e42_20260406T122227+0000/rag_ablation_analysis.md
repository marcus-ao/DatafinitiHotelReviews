# E9 RAG Ablation Result

## Summary Table

| group_id | compare_role | query_count | citation_precision | evidence_verifiability_mean | unsupported_honesty_rate | schema_valid_rate | recommendation_coverage | aspect_alignment_rate | hallucination_rate | avg_latency_ms | retry_trigger_rate | fallback_to_honest_notice_rate | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_grounded_generation | with_rag | 40 | 0.9688 | 1.9187 | 1.0 | 1.0 | 0.975 | 0.8792 | 0.0063 | 2420.519 | 0.0 | 0.0 | 1360500f65c2aa26 |
| D_no_evidence_generation | without_rag | 40 | 0.0 | 0.0 | 1.0 | 0.975 | 0.85 | 0.7333 | 0.85 | 1304.697 | 0.0 | 0.0 | 01226814e2603934 |

## Primary Conclusion

- primary compare: `B_grounded_generation` vs `D_no_evidence_generation`
- recommendation_coverage: with-RAG=0.975 | without-RAG=0.85 | Δ=0.125
- schema_valid_rate: with-RAG=1.0 | without-RAG=0.975 | Δ=0.025
- aspect_alignment_rate: with-RAG=0.8792 | without-RAG=0.7333
- hallucination_rate: with-RAG=0.0063 | without-RAG=0.85
- citation_precision (auxiliary only): with-RAG=0.9688 | without-RAG=0.0
- evidence_verifiability_mean (auxiliary only): with-RAG=1.9187 | without-RAG=0.0
- interpretation: `D_no_evidence_generation` 不看证据，因此 citation/evidence 指标只作辅助解释；主判断应放在 recommendation coverage 与 schema stability。

## Recommendation Recovery Cases

- `q008` | rag_recs=2 | no_rag_recs=0 | Δrecs=2 | rag_summary=推荐亚特兰大安静且设施良好的酒店。 | no_rag_summary=基于名称和评分，无法提供符合安静且设施优良要求的可靠推荐。
- `q033` | rag_recs=2 | no_rag_recs=0 | Δrecs=2 | rag_summary=Orlando 安静且服务良好的酒店推荐 | no_rag_summary=Orlando 安静且服务尚可的酒店推荐
- `q048` | rag_recs=2 | no_rag_recs=0 | Δrecs=2 | rag_summary=推荐两家Seattle酒店，兼顾安静睡眠与性价比。 | no_rag_summary=基于名称推测，推荐两家西雅图酒店，但缺乏具体评价证据。
- `q043` | rag_recs=1 | no_rag_recs=0 | Δrecs=1 | rag_summary=Hotel Diva 相对安静，但存在街道噪音。 | no_rag_summary=基于名称推荐旧金山酒店，但缺乏具体评价证据。
- `q081` | rag_recs=1 | no_rag_recs=0 | Δrecs=1 | rag_summary=Ramada Plaza By Wyndham Waikiki 在位置、服务和性价比上表现均衡。 | no_rag_summary=基于名称推荐Honolulu酒店，但缺乏具体评论证据。

## Matched Abstentions

- none

## Suspicious No-RAG Wins

- `q021` | rag_recs=0 | no_rag_recs=1 | rag_summary=n/a | no_rag_summary=基于名称推测，推荐Honolulu的Ramada Plaza By Wyndham Waikiki酒店。
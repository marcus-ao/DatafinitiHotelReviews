# E9 Generation Constraint Result

## Summary Table

| group_id | query_count | citation_precision | evidence_verifiability_mean | unsupported_honesty_rate | schema_valid_rate | recommendation_coverage | avg_latency_ms | retry_trigger_rate | fallback_to_honest_notice_rate | config_hash |
|---|---|---|---|---|---|---|---|---|---|---|
| A_free_generation | 40 | 0.9437 | 1.9773 | 1.0 | 1.0 | 0.95 | 2180.3 | 0.0 | 0.0 | a02eee685d569bda |
| B_grounded_generation | 40 | 0.95 | 1.9924 | 1.0 | 1.0 | 0.95 | 2182.405 | 0.0 | 0.0 | 1360500f65c2aa26 |
| C_grounded_generation_with_verifier | 40 | 0.95 | 1.9924 | 1.0 | 1.0 | 0.95 | 2185.01 | 0.0 | 0.0 | 25d4dbb89dc34f17 |
| D_no_evidence_generation | 40 | 0.0 | 0.0 | 1.0 | 0.975 | 0.825 | 1052.184 | 0.0 | 0.0 | 01226814e2603934 |

## Verifier Notes

- verifier retry count: 0
- honest fallback count: 0

## Schema Failures

- A_free_generation: 0
- B_grounded_generation: 0
- C_grounded_generation_with_verifier: 0
- D_no_evidence_generation: 1

## A_free_generation

- `q001` | citation_precision=1.0 | summary=推荐交通位置便利的酒店
- `q002` | citation_precision=1.0 | summary=推荐Anaheim交通便捷且卫生干净的酒店。
- `q003` | citation_precision=1.0 | summary=该酒店安静且服务良好，适合追求睡眠质量的用户。

## B_grounded_generation

- `q001` | citation_precision=1.0 | summary=推荐Anaheim交通便利的酒店。
- `q002` | citation_precision=1.0 | summary=推荐TownePlace Suites Anaheim，交通与卫生均获好评。
- `q003` | citation_precision=1.0 | summary=该酒店安静且服务良好，符合用户偏好。

## C_grounded_generation_with_verifier

- `q001` | citation_precision=1.0 | summary=推荐Anaheim交通便利的酒店。
- `q002` | citation_precision=1.0 | summary=推荐TownePlace Suites Anaheim，交通与卫生均获好评。
- `q003` | citation_precision=1.0 | summary=该酒店安静且服务良好，符合用户偏好。

## D_no_evidence_generation

- `q001` | citation_precision=0.0 | summary=推荐Anaheim交通便利的酒店。
- `q002` | citation_precision=0.0 | summary=推荐Anaheim交通与卫生俱佳的酒店。
- `q003` | citation_precision=0.0 | summary=基于名称推荐，但缺乏服务与安静证据。

## Free Generation Citation Drift

- `q084` | invalid=59b538e701e0032_s003 | out_of_pack=none

## Grounded + Verifier Positive Cases

- `q001` | recommendations=1 | citation_precision=1.0
- `q002` | recommendations=1 | citation_precision=1.0
- `q003` | recommendations=1 | citation_precision=1.0
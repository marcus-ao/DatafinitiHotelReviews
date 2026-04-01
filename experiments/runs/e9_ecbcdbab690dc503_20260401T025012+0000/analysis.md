# E9 Generation Constraint Result

## Summary Table

| group_id | query_count | citation_precision | evidence_verifiability_mean | unsupported_honesty_rate | schema_valid_rate | avg_latency_ms | retry_trigger_rate | fallback_to_honest_notice_rate | config_hash |
|---|---|---|---|---|---|---|---|---|---|
| A_free_generation | 40 | 0.9437 | 1.9697 | 1.0 | 1.0 | 2187.283 | 0.0 | 0.0 | e5ca3705f733d6b7 |
| B_grounded_generation | 40 | 0.9437 | 1.9773 | 1.0 | 1.0 | 2181.737 | 0.0 | 0.0 | 6ec836f74e964ec7 |
| C_grounded_generation_with_verifier | 40 | 0.925 | 1.9922 | 1.0 | 1.0 | 2248.381 | 0.025 | 0.025 | e82ef8e851d9895a |

## Verifier Notes

- verifier retry count: 1
- honest fallback count: 1

## Schema Failures

- A_free_generation: 0
- B_grounded_generation: 0
- C_grounded_generation_with_verifier: 0

## A_free_generation

- `q001` | citation_precision=1.0 | summary=推荐交通位置便利的酒店。
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

## Free Generation Citation Drift

- `q084` | invalid=59b538e701e0032_s003 | out_of_pack=none

## Grounded + Verifier Positive Cases

- `q001` | recommendations=1 | citation_precision=1.0
- `q002` | recommendations=1 | citation_precision=1.0
- `q003` | recommendations=1 | citation_precision=1.0
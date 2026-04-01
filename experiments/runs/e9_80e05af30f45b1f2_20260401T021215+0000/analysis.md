# E9 Generation Constraint Result

## Summary Table

| group_id | query_count | citation_precision | evidence_verifiability_mean | unsupported_honesty_rate | schema_valid_rate | avg_latency_ms | retry_trigger_rate | fallback_to_honest_notice_rate | config_hash |
|---|---|---|---|---|---|---|---|---|---|
| A_free_generation | 40 | 0.3 | 2.0 | 1.0 | 0.25 | 3725.96 | 0.0 | 0.0 | ad5e30a7f237149a |
| B_grounded_generation | 40 | 0.45 | 1.98 | 1.0 | 0.475 | 2690.252 | 0.0 | 0.0 | 550582bd131a6c1f |
| C_grounded_generation_with_verifier | 40 | 0.55 | 1.9848 | 1.0 | 1.0 | 3767.469 | 0.525 | 0.4 | 18e242a07bf13b28 |

## Verifier Notes

- verifier retry count: 21
- honest fallback count: 16

## A_free_generation

- `q001` | citation_precision=1.0 | summary=推荐 TownePlace Suites Anaheim Maingate Near Angel Stadium，因其位置极佳，非常便利地靠近 Honda Center、Anaheim 体育场、餐厅和星巴克。
- `q002` | citation_precision=1.0 | summary=推荐 TownePlace Suites Anaheim Maingate Near Angel Stadium，因其位置交通便利且卫生干净。
- `q003` | citation_precision=1.0 | summary=推荐 TownePlace Suites Anaheim Maingate Near Angel Stadium，因其服务表现良好，符合用户对服务的要求。关于安静睡眠，现有证据仅建议携带便携风扇以阻挡噪音，未直接证实房间本身安静，因此该酒店不完全符合‘住得安静’的核心需求，但可作为备选。

## B_grounded_generation

- `q001` | citation_precision=1.0 | summary=推荐 TownePlace Suites Anaheim Maingate Near Angel Stadium，因其位置交通极其便利，靠近 Honda Center、Anaheim 体育场及餐厅。
- `q002` | citation_precision=1.0 | summary=推荐 TownePlace Suites Anaheim Maingate Near Angel Stadium，因其位置便利且卫生干净。
- `q003` | citation_precision=1.0 | summary=推荐 TownePlace Suites Anaheim Maingate Near Angel Stadium，因其部分住客反馈在特定楼层或位置（如庭院侧）可保持安静，且服务评价良好。

## C_grounded_generation_with_verifier

- `q001` | citation_precision=1.0 | summary=推荐 TownePlace Suites Anaheim Maingate Near Angel Stadium，因其位置交通极其便利，靠近 Honda Center、Anaheim 体育场及餐厅。
- `q002` | citation_precision=1.0 | summary=推荐 TownePlace Suites Anaheim Maingate Near Angel Stadium，因其位置便利且卫生干净。
- `q003` | citation_precision=1.0 | summary=推荐 TownePlace Suites Anaheim Maingate Near Angel Stadium，因其部分住客反馈在特定楼层或位置（如庭院侧）可保持安静，且服务评价良好。

## Free Generation Citation Drift

- none

## Grounded + Verifier Positive Cases

- `q001` | recommendations=1 | citation_precision=1.0
- `q002` | recommendations=1 | citation_precision=1.0
- `q003` | recommendations=1 | citation_precision=1.0
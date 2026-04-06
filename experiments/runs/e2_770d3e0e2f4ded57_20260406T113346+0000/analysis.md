# E2 Candidate Ranking Result

> 当前结果使用 aspect-filtered dense retrieval 作为最小证据检索模块，`Candidate Hit@5` 采用“每个目标方面至少 2 句、且来自至少 2 个不同 review”的代理规则。最终论文版仍建议在 E6 qrels 完成后复算。

## A_rating_review_count

- Candidate Hit@5 (proxy): 0.9
- Avg latency (ms): 140.176
- Avg retrieval calls: 1.775
- Avg candidates checked: 1.025
- Avg dense returned / aspect: 14.271
- Avg dense unique reviews / aspect: 11.729
- Config hash: `c6ee494c8a11bafc`

## B_final_aspect_score

- Candidate Hit@5 (proxy): 0.9
- Avg latency (ms): 142.289
- Avg retrieval calls: 1.9
- Avg candidates checked: 1.15
- Avg dense returned / aspect: 13.133
- Avg dense unique reviews / aspect: 10.767
- Config hash: `64aecc42d002d658`

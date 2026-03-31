# E2 Candidate Ranking Result

> 当前结果使用 `aspect_filtered_dense_no_rerank` 作为最小证据检索模块，`Candidate Hit@5` 采用“每个目标方面至少 2 句、且来自至少 2 个不同 review”的代理规则。该结果可作为 Aspect-KB 章节的首轮论文材料，但仍不应替代后续 E6 qrels 评测。

## Result Table

| Group | Candidate Hit@5 (proxy) | Avg latency (ms) | Avg retrieval calls | Avg candidates checked | Avg dense returned / aspect | Avg dense unique reviews / aspect |
|---|---:|---:|---:|---:|---:|---:|
| A_rating_review_count | 0.9 | 111.641 | 1.775 | 1.025 | 14.271 | 11.729 |
| B_final_aspect_score | 0.9 | 102.223 | 1.9 | 1.15 | 13.133 | 10.767 |

## First-Round Interpretation

- 在当前 proxy 指标下，`A_rating_review_count` 与 `B_final_aspect_score` 的命中率同为 `0.9`。
- `B_final_aspect_score` 的平均延迟略低于 `A_rating_review_count`，但其平均候选检查数反而略高，因此当前还不能据此写成 “Aspect-KB 明显优于基线”。
- 这一轮结果更适合作为“Aspect-KB 已能支持候选缩圈实验，并在效率上显示出轻微优势”的首轮证据，而不是最终 superiority claim。

## Shared Failures

两组共同失败的 query 为：

- `q021`: 我想在 Honolulu 找一家安静睡眠比较好的酒店。
- `q022`: 请推荐 Honolulu 安静睡眠和性价比都不错的酒店。
- `q023`: 我在 Honolulu 想住得安静一点，但不要位置交通太差的酒店。
- `q081`: 请推荐 Honolulu 在安静睡眠、位置交通、服务三方面都比较均衡的酒店。

这 4 条 query 的共同失败原因一致：当前 test 候选池里唯一被检查到的酒店是 `Ramada Plaza By Wyndham Waikiki`，其 `quiet_sleep` 支持度始终不满足代理规则。

## Failure Diagnosis

### 1. 当前直接瓶颈是 test split 下的城市候选稀疏

- 冻结切分中，Honolulu 全市共有 `7` 家酒店，但 `test split` 中只有 `1` 家，即 `Ramada Plaza By Wyndham Waikiki`。
- 因此 E2 在这 4 条 query 上并不是在“多个 Honolulu 候选之间竞争失败”，而是在“只有 1 家可评候选时，该酒店本身缺少 quiet_sleep 支持证据”。

### 2. `Ramada Plaza By Wyndham Waikiki` 的 `quiet_sleep` 证据为 0

该酒店在当前证据索引中的方面覆盖如下：

- `location_transport`: `43` 句 / `25` 个 review
- `service`: `22` 句 / `14` 个 review
- `cleanliness`: `12` 句 / `9` 个 review
- `room_facilities`: `10` 句 / `7` 个 review
- `value`: `5` 句 / `5` 个 review
- `quiet_sleep`: `0` 句 / `0` 个 review

所以对 `q021 / q022 / q023 / q081` 来说，失败日志里出现的 `quiet_sleep: sentence_count<2` 是与底层证据索引一致的，不是 runner 统计错误。

### 3. 这不是“Honolulu 整体没有 quiet_sleep 数据”

Honolulu 全市并非没有 `quiet_sleep` 证据。当前数据中至少有以下酒店满足该方面覆盖：

- `Hyatt Centric Waikiki Beach`: `2` 句 / `2` 个 review
- `Luana Waikiki Hotel & Suites`: `9` 句 / `8` 个 review
- `Waikiki Resort Hotel`: `6` 句 / `6` 个 review

但这些酒店分别落在 `dev/train` 中，没有进入当前 E2 的 `test` 候选池。因此，本轮失败首先是**切分后的城市候选稀疏**问题，而不是 Honolulu 原始数据完全不足。

### 4. 还存在次级的主标签机制限制

在 `Ramada Plaza By Wyndham Waikiki` 的原始句子中，仍能找到一句带有安静语义的文本：

`Quieter, less congested and easy to walk to where you want along the beach.`

但它当前被标为 `location_transport`，而不是 `quiet_sleep`。这说明除了 test split 稀疏外，当前主标签机制在“安静感受”与“位置/拥挤度描述”交叉时，也会造成一部分 `quiet_sleep` 召回流失。不过，这类句子在该酒店上数量仍然不足以满足当前 `2` 句 / `2` review 的支持阈值，因此它是次级原因，不是主因。

## Boundary for Writing

- 当前可以写的结论是：`B_final_aspect_score` 在本轮 proxy 评测下与弱基线命中率持平，平均延迟略低，说明 Aspect-KB 画像已具备进入候选缩圈链路的可行性。
- 当前不能写的结论是：Aspect-KB 已经在 E2 上显著优于基线。
- 更稳妥的论文表述应是：E2 已提供首轮支持性证据，但受限于 test split 的城市候选覆盖和 `quiet_sleep` 标签稀疏，最终检索质量结论仍需等待 E6/E7/E8 的进一步验证。

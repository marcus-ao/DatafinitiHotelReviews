# E9 Generation Audit Assets

本目录用于保存 `E9` 证据约束生成阶段的人工审计材料。

当前约定：

- `citation_verifiability_audit.csv`
  每轮正式 `E9` run 导出的最新审计表副本。
- `citation_verifiability_audit_e9_ecbcdbab690dc503_qwen35_4b_reviewed.csv`
  `E9` 第二轮正式结果的 reviewed 冻结快照。
- 论文主表默认引用自动指标。
- 涉及“Evidence Verifiability”最终结论、案例分析和章节写作时，应优先引用人工 reviewed 结果，而不是直接把弱自动近似当最终事实。

固定字段：

- `query_id`
- `group_id`
- `hotel_id`
- `sentence_id`
- `reason_text`
- `citation_exists`
- `in_current_evidence_pack`
- `support_score`
- `notes`

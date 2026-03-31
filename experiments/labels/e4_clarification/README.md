# E4 Clarification Audit

本目录用于存放 `E4` 正式运行后生成的人工问题质量审计资产。

当前约定：

- 每轮 `E4` 正式 run 都会在自己的 run 目录内写一份 `clarification_question_audit.csv`
- 当前目录中的 `clarification_question_audit.csv` 只保留最新副本，方便继续人工填写
- 历史关键 run 如果需要冻结，会额外保存成带 run id 的快照文件
- 人工填写列仍然是：
  - `answerable_score`
  - `targeted_score`
  - `notes`

# E4 Clarification Audit

本目录用于存放 `E4` 正式运行后生成的人工问题质量审计资产。

当前约定：

- 正式审计文件名固定为 `clarification_question_audit.csv`
- 该文件由 `python -m scripts.evaluation.run_experiment_suite --task e4_clarification` 生成
- 生成后再由人工填写 `answerable_score`、`targeted_score` 与 `notes`

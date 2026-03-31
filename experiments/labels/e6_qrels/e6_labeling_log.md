# E6 Labeling Log

## Status

- [x] qrels pool generated
- [x] manual labeling completed
- [x] qrels_evidence.jsonl frozen

## Pool Summary

- Executable queries: 40
- Query-aspect units: 80
- Focus units: 70
- Avoid units: 10
- Pooled sentence rows: 817
- Focus rows: 700
- Avoid rows: 117
- Official modes: plain_city_test_rerank, aspect_main_rerank, aspect_main_no_rerank, aspect_main_fallback_rerank
- Pooling depth / mode: Top5

## Final Artifacts

- Official labeled pool: `experiments/labels/e6_qrels/qrels_pool.csv`
- Frozen evidence: `experiments/labels/e6_qrels/qrels_evidence.jsonl`
- Labeling run archive: local archive `ReviewsLabelling/runs/20260330_194912/`
- Final merged pool snapshot: `ReviewsLabelling/runs/20260330_194912/final_qrels_pool.csv`
- Resolved review subset: `ReviewsLabelling/runs/20260330_194912/review_resolved.csv`

## Next Step

1. Run `python -m scripts.evaluation.run_experiment_suite --task e6_retrieval`.
2. Run `python -m scripts.evaluation.run_experiment_suite --task e7_reranker`.
3. Run `python -m scripts.evaluation.run_experiment_suite --task e8_fallback`.

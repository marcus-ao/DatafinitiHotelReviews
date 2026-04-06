# G Retrieval Labeling Log

## Status

- [x] G qrels pool generated
- [x] manual labeling completed
- [ ] g_qrels_evidence.jsonl frozen

## Pool Summary

- Executable queries: 68
- Query-aspect units: 108
- Pooled sentence rows: 1102
- Query asset: D:/VSCodeWorkspace/GraduationDesign/DatafinitiHotelReviews/experiments/assets/g_eval_query_ids_68.json
- Query types: single_aspect, multi_aspect, focus_and_avoid, multi_aspect_strong, unsupported_budget, unsupported_distance, unsupported_heavy
- Official modes: plain_city_test_rerank, aspect_main_rerank, aspect_main_no_rerank, aspect_main_fallback_rerank
- Pooling depth / mode: Top5

## Labeling Summary

- Reused from `E6 qrels` by exact `(query_id, target_aspect, target_role, hotel_id, sentence_id)` match: 812 rows
- Reused from `E6 qrels` by unique relaxed `(target_aspect, target_role, hotel_id, sentence_id)` match: 205 rows
- Manually re-annotated after the 2-query cleanup: 85 rows
- Removed abnormal queries in this rerun-prep round: `q021`, `q024`

## Next Step

1. Freeze qrels with `python -m scripts.evaluation.run_experiment_suite --task g_freeze_qrels`.
2. Validate qrels with `python -m scripts.evaluation.run_experiment_suite --task g_validate_qrels`.
3. Run `g_retrieval_eval --retrieval-variant plain` and `g_retrieval_eval --retrieval-variant aspect`.

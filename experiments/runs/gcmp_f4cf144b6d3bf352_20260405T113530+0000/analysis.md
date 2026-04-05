# G Compare Result (G2 vs G1)

## Summary Table

| group_id | query_count | citation_precision | evidence_verifiability_mean | schema_valid_rate | recommendation_coverage | aspect_alignment_rate | hallucination_rate | unsupported_honesty_rate | avg_latency_ms | config_hash | source_run_id | source_run_dir | source_label | latency_formally_comparable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 | 70 | 0.9643 | 1.5535 | 0.9857 | 0.9714 | 0.9286 | 0.0047 | 0.9333 | 5152.838 | 2735d67da525349a | ggen_fea86921c7f960ff_20260405T110301+0000 | experiments/runs/ggen_fea86921c7f960ff_20260405T110301+0000 | G1 | True |
| G2 | 70 | 0.9607 | 1.9735 | 1.0 | 0.9714 | 0.9167 | 0.0088 | 0.9667 | 5763.328 | 97a4d07fe799359e | ggen_6076ca68b3b85e10_20260405T110912+0000 | experiments/runs/ggen_6076ca68b3b85e10_20260405T110912+0000 | G2 | True |

## Representative Improvements

- `q040` | Δschema_valid=1 | Δcitation_precision=0.0 | Δevidence=0.0 | g1_recs=2 | g2_recs=2
- `q013` | Δschema_valid=0 | Δcitation_precision=1.0 | Δevidence=2.0 | g1_recs=0 | g2_recs=1
- `q002` | Δschema_valid=0 | Δcitation_precision=0.5 | Δevidence=1.0 | g1_recs=1 | g2_recs=1
- `q003` | Δschema_valid=0 | Δcitation_precision=0.0 | Δevidence=1.0 | g1_recs=1 | g2_recs=1
- `q011` | Δschema_valid=0 | Δcitation_precision=0.0 | Δevidence=1.0 | g1_recs=1 | g2_recs=2

## Representative Regressions

- `q024` | Δschema_valid=0 | Δcitation_precision=-1.0 | Δevidence=-1.0 | g1_recs=1 | g2_recs=0
- `q069` | Δschema_valid=0 | Δcitation_precision=-0.5 | Δevidence=-0.25 | g1_recs=2 | g2_recs=1
- `q079` | Δschema_valid=0 | Δcitation_precision=-0.25 | Δevidence=-0.25 | g1_recs=2 | g2_recs=2
# G Compare Result (G3 vs G1)

## Summary Table

| group_id | query_count | citation_precision | evidence_verifiability_mean | schema_valid_rate | recommendation_coverage | aspect_alignment_rate | hallucination_rate | unsupported_honesty_rate | avg_latency_ms | config_hash | source_run_id | source_run_dir | source_label | latency_formally_comparable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 | 70 | 0.9643 | 1.5535 | 0.9857 | 0.9714 | 0.9286 | 0.0047 | 0.9333 | 5152.838 | 2735d67da525349a | ggen_fea86921c7f960ff_20260405T110301+0000 | experiments/runs/ggen_fea86921c7f960ff_20260405T110301+0000 | G1 | True |
| G3 | 70 | 0.95 | 1.5634 | 0.9714 | 0.9714 | 0.9381 | 0.0141 | 1.0 | 5444.945 | 2c471eb64907a606 | ggen_d2f0a943009a7136_20260405T111912+0000 | experiments/runs/ggen_d2f0a943009a7136_20260405T111912+0000 | G3 | True |

## Representative Improvements

- `q040` | Δschema_valid=1 | Δcitation_precision=0.0 | Δevidence=-0.25 | g1_recs=2 | g3_recs=2
- `q002` | Δschema_valid=0 | Δcitation_precision=0.5 | Δevidence=1.0 | g1_recs=1 | g3_recs=1
- `q009` | Δschema_valid=0 | Δcitation_precision=0.0 | Δevidence=0.5 | g1_recs=2 | g3_recs=1
- `q011` | Δschema_valid=0 | Δcitation_precision=0.0 | Δevidence=0.5 | g1_recs=1 | g3_recs=2
- `q014` | Δschema_valid=0 | Δcitation_precision=0.0 | Δevidence=0.5 | g1_recs=2 | g3_recs=1

## Representative Regressions

- `q030` | Δschema_valid=-1 | Δcitation_precision=0.0 | Δevidence=0.0 | g1_recs=2 | g3_recs=2
- `q035` | Δschema_valid=-1 | Δcitation_precision=0.0 | Δevidence=0.25 | g1_recs=2 | g3_recs=2
- `q019` | Δschema_valid=0 | Δcitation_precision=-0.5 | Δevidence=-1.0 | g1_recs=2 | g3_recs=1
- `q045` | Δschema_valid=0 | Δcitation_precision=-0.5 | Δevidence=-1.0 | g1_recs=2 | g3_recs=1
- `q044` | Δschema_valid=0 | Δcitation_precision=-0.5 | Δevidence=-0.75 | g1_recs=2 | g3_recs=1
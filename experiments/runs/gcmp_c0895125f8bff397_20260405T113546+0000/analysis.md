# G Compare Result (G4 vs G2)

## Summary Table

| group_id | query_count | citation_precision | evidence_verifiability_mean | schema_valid_rate | recommendation_coverage | aspect_alignment_rate | hallucination_rate | unsupported_honesty_rate | avg_latency_ms | config_hash | source_run_id | source_run_dir | source_label | latency_formally_comparable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G2 | 70 | 0.9607 | 1.9735 | 1.0 | 0.9714 | 0.9167 | 0.0088 | 0.9667 | 5763.328 | 97a4d07fe799359e | ggen_6076ca68b3b85e10_20260405T110912+0000 | experiments/runs/ggen_6076ca68b3b85e10_20260405T110912+0000 | G2 | True |
| G4 | 70 | 0.9607 | 1.9545 | 0.9714 | 0.9714 | 0.9381 | 0.0182 | 1.0 | 5581.1 | beb415157f30c50f | ggen_dc21004597b2b385_20260405T112545+0000 | experiments/runs/ggen_dc21004597b2b385_20260405T112545+0000 | G4 | True |

## Representative Improvements

- `q069` | Δschema_valid=0 | Δcitation_precision=0.5 | Δevidence=1.0 | g2_recs=1 | g4_recs=1
- `q079` | Δschema_valid=0 | Δcitation_precision=0.25 | Δevidence=0.5 | g2_recs=2 | g4_recs=2

## Representative Regressions

- `q022` | Δschema_valid=-1 | Δcitation_precision=0.0 | Δevidence=-1.0 | g2_recs=1 | g4_recs=1
- `q018` | Δschema_valid=-1 | Δcitation_precision=0.0 | Δevidence=-0.5 | g2_recs=2 | g4_recs=2
- `q073` | Δschema_valid=0 | Δcitation_precision=-0.5 | Δevidence=-1.0 | g2_recs=2 | g4_recs=1
- `q085` | Δschema_valid=0 | Δcitation_precision=-0.25 | Δevidence=-0.5 | g2_recs=2 | g4_recs=2
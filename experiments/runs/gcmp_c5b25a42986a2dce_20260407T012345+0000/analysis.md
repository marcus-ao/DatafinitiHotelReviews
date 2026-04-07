# G Compare Result (G4 vs G3)

## Summary Table

| group_id | query_count | citation_precision | evidence_verifiability_mean | schema_valid_rate | recommendation_coverage | aspect_alignment_rate | hallucination_rate | unsupported_honesty_rate | avg_latency_ms | config_hash | source_run_id | source_run_dir | source_label | latency_formally_comparable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G3 | 68 | 0.9632 | 1.5515 | 0.9706 | 0.9853 | 0.951 | 0.0221 | 1.0 | 5465.992 | 22b817bc3a40bc81 | ggen_a3e6f7da3d69ef54_20260406T155927+0000 | d:\VSCodeWorkspace\GraduationDesign\DatafinitiHotelReviews\experiments\runs\ggen_a3e6f7da3d69ef54_20260406T155927+0000 | G3 | True |
| G4 | 68 | 0.989 | 1.9449 | 0.9706 | 1.0 | 0.9657 | 0.0221 | 1.0 | 5492.044 | 122d8410e5bd1e5b | ggen_49001bbb5794fd52_20260406T160643+0000 | d:\VSCodeWorkspace\GraduationDesign\DatafinitiHotelReviews\experiments\runs\ggen_49001bbb5794fd52_20260406T160643+0000 | G4 | True |

## Representative Improvements

- `q030` | Δschema_valid=1 | Δcitation_precision=0.0 | Δevidence=0.0 | g3_recs=2 | g4_recs=2
- `q035` | Δschema_valid=1 | Δcitation_precision=0.0 | Δevidence=0.0 | g3_recs=2 | g4_recs=2
- `q013` | Δschema_valid=0 | Δcitation_precision=1.0 | Δevidence=2.0 | g3_recs=0 | g4_recs=1
- `q019` | Δschema_valid=0 | Δcitation_precision=0.5 | Δevidence=1.5 | g3_recs=1 | g4_recs=2
- `q044` | Δschema_valid=0 | Δcitation_precision=0.5 | Δevidence=1.5 | g3_recs=1 | g4_recs=2

## Representative Regressions

- `q022` | Δschema_valid=-1 | Δcitation_precision=0.0 | Δevidence=-0.5 | g3_recs=1 | g4_recs=1
- `q018` | Δschema_valid=-1 | Δcitation_precision=0.0 | Δevidence=0.25 | g3_recs=2 | g4_recs=2
- `q073` | Δschema_valid=0 | Δcitation_precision=-0.5 | Δevidence=-1.0 | g3_recs=1 | g4_recs=1
- `q085` | Δschema_valid=0 | Δcitation_precision=-0.25 | Δevidence=0.0 | g3_recs=2 | g4_recs=2
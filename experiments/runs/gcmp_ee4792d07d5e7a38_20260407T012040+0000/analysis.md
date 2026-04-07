# G Compare Result (G3 vs G1)

## Summary Table

| group_id | query_count | citation_precision | evidence_verifiability_mean | schema_valid_rate | recommendation_coverage | aspect_alignment_rate | hallucination_rate | unsupported_honesty_rate | avg_latency_ms | config_hash | source_run_id | source_run_dir | source_label | latency_formally_comparable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 | 68 | 0.9779 | 1.5404 | 0.9853 | 0.9853 | 0.9412 | 0.0074 | 0.931 | 5395.824 | ec191d7195b72b76 | ggen_1f8cbadcfff55384_20260406T154446+0000 | d:\VSCodeWorkspace\GraduationDesign\DatafinitiHotelReviews\experiments\runs\ggen_1f8cbadcfff55384_20260406T154446+0000 | G1 | True |
| G3 | 68 | 0.9632 | 1.5515 | 0.9706 | 0.9853 | 0.951 | 0.0221 | 1.0 | 5465.992 | 22b817bc3a40bc81 | ggen_a3e6f7da3d69ef54_20260406T155927+0000 | d:\VSCodeWorkspace\GraduationDesign\DatafinitiHotelReviews\experiments\runs\ggen_a3e6f7da3d69ef54_20260406T155927+0000 | G3 | True |

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
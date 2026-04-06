# E10 Base vs PEFT Compare Result

## Summary Table

| group_id | query_count | citation_precision | evidence_verifiability_mean | schema_valid_rate | recommendation_coverage | aspect_alignment_rate | hallucination_rate | unsupported_honesty_rate | avg_latency_ms | config_hash | reasoning_leak_rate | auditable_query_rate | source_run_id | source_run_dir | latency_formally_comparable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_base_4b_grounded | 40 | 0.9688 | 1.9187 | 1.0 | 0.975 | 0.8792 | 0.0063 | n/a (no applicable unsupported-request queries) | 5899.415 | 921e5b6974b091b7 | 0.0 | 1.0 | e10_0dc5c2e6f867c66f_20260406T135631+0000 | experiments/runs/e10_0dc5c2e6f867c66f_20260406T135631+0000 | True |
| B_peft_4b_grounded | 40 | 0.925 | 1.8292 | 0.9 | 0.925 | 0.8125 | 0.0063 | n/a (no applicable unsupported-request queries) | 5729.578 | 8d76bc9f77945640 | 0.0 | 1.0 | e10_927c1d0a2fbc5870_20260406T141302+0000 | experiments/runs/e10_927c1d0a2fbc5870_20260406T141302+0000 | True |

## Source Runs

- base_run_id: e10_0dc5c2e6f867c66f_20260406T135631+0000
- peft_run_id: e10_927c1d0a2fbc5870_20260406T141302+0000
- latency_formally_comparable: yes
- latency_note: 两组使用相同 local backend，可纳入正式时延对照。

## Representative Improvements

- `q079` | Δschema_valid=0 | Δcitation_precision=0.25 | Δevidence=0.5 | base_recs=2 | peft_recs=2

## Representative Regressions

- `q018` | Δschema_valid=-1 | Δcitation_precision=0.0 | Δevidence=-0.5 | base_recs=2 | peft_recs=2
- `q038` | Δschema_valid=-1 | Δcitation_precision=0.0 | Δevidence=-0.0833 | base_recs=2 | peft_recs=2
- `q017` | Δschema_valid=-1 | Δcitation_precision=0.0 | Δevidence=0.0 | base_recs=2 | peft_recs=1
- `q033` | Δschema_valid=-1 | Δcitation_precision=0.0 | Δevidence=0.0 | base_recs=2 | peft_recs=1
- `q013` | Δschema_valid=0 | Δcitation_precision=-1.0 | Δevidence=-2.0 | base_recs=1 | peft_recs=0
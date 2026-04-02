# E10 Base vs PEFT Compare Result

## Summary Table

| group_id | query_count | citation_precision | evidence_verifiability_mean | unsupported_honesty_rate | schema_valid_rate | reasoning_leak_rate | auditable_query_rate | avg_latency_ms | config_hash | source_run_id | source_run_dir | latency_formally_comparable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_base_4b_grounded | 40 | 0.9688 | 1.9704 | n/a (no applicable unsupported-request queries) | 1.0 | 0.0 | 1.0 | 5776.447 | 921e5b6974b091b7 | e10_0dc5c2e6f867c66f_20260402T015230+0000 | /root/autodl-tmp/workspace/DatafinitiHotelReviews/experiments/runs/e10_0dc5c2e6f867c66f_20260402T015230+0000 | True |
| B_peft_4b_grounded | 40 | 0.925 | 1.9915 | n/a (no applicable unsupported-request queries) | 1.0 | 0.0 | 0.975 | 5440.052 | ed4665911e79a1ce | e10_0ef381420c1bd19a_20260402T020120+0000 | /root/autodl-tmp/workspace/DatafinitiHotelReviews/experiments/runs/e10_0ef381420c1bd19a_20260402T020120+0000 | True |

## Source Runs

- base_run_id: e10_0dc5c2e6f867c66f_20260402T015230+0000
- peft_run_id: e10_0ef381420c1bd19a_20260402T020120+0000
- latency_formally_comparable: yes
- latency_note: 两组使用相同 local backend，可纳入正式时延对照。

## Representative Improvements

- `q079` | Δschema_valid=0 | Δcitation_precision=0.25 | Δevidence=0.5 | base_recs=2 | peft_recs=2

## Representative Regressions

- `q013` | Δschema_valid=0 | Δcitation_precision=-1.0 | Δevidence=-2.0 | base_recs=1 | peft_recs=0
- `q023` | Δschema_valid=0 | Δcitation_precision=-1.0 | Δevidence=-1.5 | base_recs=1 | peft_recs=0
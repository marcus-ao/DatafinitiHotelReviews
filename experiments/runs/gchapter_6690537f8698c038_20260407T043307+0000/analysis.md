# G1-G4 Unified Chapter Report

## Retrieval Summary

> Retrieval-side rows below are loaded from formal qrels-based G retrieval evaluation runs.
> The CSV also records the retrieval variant and source run directory for each group.

| group_id | retrieval_variant | aspect_recall_at_5 | ndcg_at_5 | precision_at_5 | mrr_at_5 | evidence_diversity_at_5 | avg_latency_ms | retrieval_summary_source | retrieval_summary_run_dir |
|---|---|---|---|---|---|---|---|---|---|
| G1 | plain | 0.75 | 0.3726 | 0.3704 | 0.6227 | 0.95 | 885.252 | formal_retrieval_eval | experiments\runs\gret_plain_8b50bf28a263c259_20260407T042745+0000 |
| G2 | aspect | 0.8981 | 0.6996 | 0.6981 | 0.8333 | 0.9037 | 238.948 | formal_retrieval_eval | experiments\runs\gret_aspect_6799b0368d72fbf5_20260407T042950+0000 |
| G3 | plain | 0.75 | 0.3726 | 0.3704 | 0.6227 | 0.95 | 885.252 | formal_retrieval_eval | experiments\runs\gret_plain_8b50bf28a263c259_20260407T042745+0000 |
| G4 | aspect | 0.8981 | 0.6996 | 0.6981 | 0.8333 | 0.9037 | 238.948 | formal_retrieval_eval | experiments\runs\gret_aspect_6799b0368d72fbf5_20260407T042950+0000 |

## Generation Summary

| group_id | schema_valid_rate | citation_precision | evidence_verifiability_mean | recommendation_coverage | aspect_alignment_rate | hallucination_rate | unsupported_honesty_rate |
|---|---|---|---|---|---|---|---|
| G1 | 0.9853 | 0.9779 | 1.5404 | 0.9853 | 0.9412 | 0.0074 | 0.931 |
| G2 | 1.0 | 0.9816 | 1.9522 | 0.9853 | 0.9289 | 0.0037 | 0.9655 |
| G3 | 0.9706 | 0.9632 | 1.5515 | 0.9853 | 0.951 | 0.0221 | 1.0 |
| G4 | 0.9706 | 0.989 | 1.9449 | 1.0 | 0.9657 | 0.0221 | 1.0 |

## Pairwise Statistical Tests

| group_a | group_b | metric | pairing_mode | overlap_n | dropped_from_a | dropped_from_b | n | non_zero_n | wins_group_a | wins_group_b | ties | higher_is_better | better_group | mean_a | mean_b | median_a | median_b | std_a | std_b | mean_delta | median_delta | std_delta | wilcoxon_statistic | p_value | significance | p_value_adj | significance_adj | p_adjust_method | ci_level | ci_low | ci_high | cohens_d | cohens_d_magnitude | rank_biserial | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 | G2 | aspect_alignment_rate | query_id | 68 | 0 | 0 | 68 | 4 | 2 | 2 | 64 | True | G1 | 0.9412 | 0.9289 | 1.0 | 1.0 | 0.1623 | 0.2002 | -0.0123 | 0.0 | 0.1477 | 3.0 | 0.461451 | ns | 0.951933 | ns | holm | 0.95 | -0.0515 | 0.0147 | -0.083 | negligible | -0.4 | ok |
| G1 | G2 | citation_precision | query_id | 68 | 0 | 0 | 68 | 2 | 1 | 1 | 66 | True | G2 | 0.9779 | 0.9816 | 1.0 | 1.0 | 0.1348 | 0.1246 | 0.0037 | 0.0 | 0.0682 | 1.0 | 0.654721 | ns | 1.0 | ns | holm | 0.95 | -0.011 | 0.0221 | 0.0539 | negligible | 0.3333 | ok |
| G1 | G2 | evidence_verifiability_mean | query_id | 68 | 0 | 0 | 68 | 47 | 1 | 46 | 21 | True | G2 | 1.5404 | 1.9522 | 1.5 | 2.0 | 0.3973 | 0.2565 | 0.4118 | 0.5 | 0.3608 | 5.0 | 0.0 | *** | 0.0 | *** | holm | 0.95 | 0.3272 | 0.5 | 1.1413 | large | 0.9911 | ok |
| G1 | G2 | hallucination_rate | query_id | 68 | 0 | 0 | 68 | 2 | 1 | 1 | 66 | False | G2 | 0.0074 | 0.0037 | 0.0 | 0.0 | 0.0606 | 0.0303 | -0.0037 | 0.0 | 0.0682 | 1.0 | 0.654721 | ns | 1.0 | ns | holm | 0.95 | -0.0221 | 0.011 | -0.0539 | negligible | -0.3333 | ok |
| G1 | G2 | latency_ms | query_id | 68 | 0 | 0 | 68 | 68 | 16 | 52 | 0 | False | G1 | 5395.8242 | 5745.495 | 6187.528 | 6474.679 | 1514.7603 | 1480.0817 | 349.6708 | 219.5505 | 1144.4101 | 407.0 | 3e-06 | *** | 1.8e-05 | *** | holm | 0.95 | 83.9894 | 628.2318 | 0.3055 | small | 0.653 | ok |
| G1 | G2 | recommendation_coverage | query_id | 68 | 0 | 0 | 68 | 0 | 0 | 0 | 68 | True | tie | 0.9853 | 0.9853 | 1.0 | 1.0 | 0.1213 | 0.1213 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | ns | 1.0 | ns | holm | 0.95 | 0.0 | 0.0 | 0.0 | negligible | 0.0 | all_zero_differences |
| G1 | G2 | schema_valid_rate | query_id | 68 | 0 | 0 | 68 | 1 | 0 | 1 | 67 | True | G2 | 0.9853 | 1.0 | 1.0 | 1.0 | 0.1213 | 0.0 | 0.0147 | 0.0 | 0.1213 | 0.0 | 1.0 | ns | 1.0 | ns | holm | 0.95 | 0.0 | 0.0441 | 0.1213 | negligible | 1.0 | insufficient_non_zero_pairs |
| G1 | G2 | unsupported_honesty_rate | query_id | 29 | 0 | 0 | 29 | 1 | 0 | 1 | 28 | True | G2 | 0.931 | 0.9655 | 1.0 | 1.0 | 0.2579 | 0.1857 | 0.0345 | 0.0 | 0.1857 | 0.0 | 1.0 | ns | 1.0 | ns | holm | 0.95 | 0.0 | 0.1379 | 0.1857 | negligible | 1.0 | insufficient_non_zero_pairs |
| G1 | G3 | aspect_alignment_rate | query_id | 68 | 0 | 0 | 68 | 4 | 1 | 3 | 64 | True | G3 | 0.9412 | 0.951 | 1.0 | 1.0 | 0.1623 | 0.1552 | 0.0098 | 0.0 | 0.0808 | 2.5 | 0.317311 | ns | 0.951933 | ns | holm | 0.95 | -0.0049 | 0.0294 | 0.1213 | negligible | 0.5 | ok |
| G1 | G3 | citation_precision | query_id | 68 | 0 | 0 | 68 | 4 | 3 | 1 | 64 | True | G1 | 0.9779 | 0.9632 | 1.0 | 1.0 | 0.1348 | 0.1573 | -0.0147 | 0.0 | 0.1213 | 2.5 | 0.317311 | ns | 1.0 | ns | holm | 0.95 | -0.0441 | 0.0147 | -0.1213 | negligible | -0.5 | ok |
| G1 | G3 | evidence_verifiability_mean | query_id | 68 | 0 | 0 | 68 | 22 | 8 | 14 | 46 | True | G3 | 1.5404 | 1.5515 | 1.5 | 1.5 | 0.3973 | 0.448 | 0.011 | 0.0 | 0.3187 | 112.5 | 0.644886 | ns | 1.0 | ns | holm | 0.95 | -0.0699 | 0.0847 | 0.0346 | negligible | 0.1107 | ok |
| G1 | G3 | hallucination_rate | query_id | 68 | 0 | 0 | 68 | 4 | 1 | 3 | 64 | False | G1 | 0.0074 | 0.0221 | 0.0 | 0.0 | 0.0606 | 0.1034 | 0.0147 | 0.0 | 0.1213 | 2.5 | 0.317311 | ns | 1.0 | ns | holm | 0.95 | -0.0147 | 0.0441 | 0.1213 | negligible | 0.5 | ok |
| G1 | G3 | latency_ms | query_id | 68 | 0 | 0 | 68 | 68 | 21 | 47 | 0 | False | G1 | 5395.8242 | 5465.9923 | 6187.528 | 6387.845 | 1514.7603 | 1463.6545 | 70.1681 | 161.4215 | 1515.9451 | 752.0 | 0.010098 | * | 0.05049 | ns | holm | 0.95 | -311.5273 | 413.5756 | 0.0463 | negligible | 0.3589 | ok |
| G1 | G3 | recommendation_coverage | query_id | 68 | 0 | 0 | 68 | 0 | 0 | 0 | 68 | True | tie | 0.9853 | 0.9853 | 1.0 | 1.0 | 0.1213 | 0.1213 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | ns | 1.0 | ns | holm | 0.95 | 0.0 | 0.0 | 0.0 | negligible | 0.0 | all_zero_differences |
| G1 | G3 | schema_valid_rate | query_id | 68 | 0 | 0 | 68 | 3 | 2 | 1 | 65 | True | G1 | 0.9853 | 0.9706 | 1.0 | 1.0 | 0.1213 | 0.1702 | -0.0147 | 0.0 | 0.2111 | 2.0 | 0.563703 | ns | 1.0 | ns | holm | 0.95 | -0.0588 | 0.0294 | -0.0697 | negligible | -0.3333 | ok |
| G1 | G3 | unsupported_honesty_rate | query_id | 29 | 0 | 0 | 29 | 2 | 0 | 2 | 27 | True | G3 | 0.931 | 1.0 | 1.0 | 1.0 | 0.2579 | 0.0 | 0.069 | 0.0 | 0.2579 | 0.0 | 0.157299 | ns | 0.943794 | ns | holm | 0.95 | 0.0 | 0.1724 | 0.2674 | small | 1.0 | ok |
| G1 | G4 | aspect_alignment_rate | query_id | 68 | 0 | 0 | 68 | 3 | 0 | 3 | 65 | True | G4 | 0.9412 | 0.9657 | 1.0 | 1.0 | 0.1623 | 0.102 | 0.0245 | 0.0 | 0.1328 | 0.0 | 0.10247 | ns | 0.61482 | ns | holm | 0.95 | 0.0 | 0.0637 | 0.1846 | negligible | 1.0 | ok |
| G1 | G4 | citation_precision | query_id | 68 | 0 | 0 | 68 | 4 | 2 | 2 | 64 | True | G4 | 0.9779 | 0.989 | 1.0 | 1.0 | 0.1348 | 0.0674 | 0.011 | 0.0 | 0.1523 | 3.5 | 0.580712 | ns | 1.0 | ns | holm | 0.95 | -0.0184 | 0.0515 | 0.0724 | negligible | 0.3 | ok |
| G1 | G4 | evidence_verifiability_mean | query_id | 68 | 0 | 0 | 68 | 47 | 2 | 45 | 21 | True | G4 | 1.5404 | 1.9449 | 1.5 | 2.0 | 0.3973 | 0.1972 | 0.4044 | 0.5 | 0.4384 | 50.0 | 0.0 | *** | 0.0 | *** | holm | 0.95 | 0.3051 | 0.511 | 0.9224 | large | 0.9113 | ok |
| G1 | G4 | hallucination_rate | query_id | 68 | 0 | 0 | 68 | 5 | 1 | 4 | 63 | False | G1 | 0.0074 | 0.0221 | 0.0 | 0.0 | 0.0606 | 0.094 | 0.0147 | 0.0 | 0.1133 | 4.0 | 0.333998 | ns | 1.0 | ns | holm | 0.95 | -0.011 | 0.0441 | 0.1298 | negligible | 0.4667 | ok |

## LLM Judge Summary

| group_id | judge_count | relevance | traceability | fluency | completeness | honesty | overall_mean |
|---|---|---|---|---|---|---|---|
| G1 | 68 | 3.725 | 3.5809 | 4.5603 | 2.9809 | 4.4382 | 3.8571 |
| G2 | 68 | 3.9324 | 4.4029 | 4.6456 | 3.2132 | 4.5382 | 4.1465 |
| G3 | 68 | 3.9059 | 3.8868 | 4.5882 | 3.1735 | 4.5647 | 4.0244 |
| G4 | 68 | 4.025 | 4.4265 | 4.6235 | 3.3206 | 4.7029 | 4.2197 |

## Human-Verified Blind Review

- These blind-review annotations were manually checked and accepted by the researcher.
- The current workspace treats them as formal human-verified review results.

## Human-Verified Blind Review Item Summary

| source_group_id | review_count | overall_quality_mean | evidence_credibility_mean | practical_value_mean |
|---|---|---|---|---|
| G1 | 20 | 3.785 | 3.595 | 3.315 |
| G2 | 20 | 3.795 | 3.615 | 3.3200000000000003 |
| G3 | 20 | 3.755 | 3.46 | 3.19 |
| G4 | 20 | 3.75 | 3.5700000000000003 | 3.225 |

## Human-Verified Blind Review Pairwise Summary

| preference_label | count |
|---|---|
| G1>G2 | 2 |
| G1>G3 | 1 |
| G2>G1 | 2 |
| G2>G4 | 9 |
| G3>G2 | 1 |
| G4>G1 | 2 |
| G4>G2 | 2 |
| G4>G3 | 1 |
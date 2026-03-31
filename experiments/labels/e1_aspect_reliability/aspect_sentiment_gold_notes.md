# Aspect Sentiment Gold Data Note

## Current Official File

- Official E1 gold path: `E1/aspect_sentiment_gold.csv`
- Synced from local adjudication archive `ReviewsLabelling/runs/20260330_133939/final_gold.csv` on 2026-03-30.

## Counts

- Original sampled sentences: 360
- Final usable gold sentences: 344
- Excluded merchant reply sentences: 16
- Manually resolved review rows: 76

## Related Files

- `ReviewsLabelling/runs/20260330_133939/final_gold.csv`: finalized adjudicated source snapshot.
- `ReviewsLabelling/runs/20260330_133939/review_resolved.csv`: review adjudication log for the 76 rows in the review queue. Its final decisions have already been merged into `E1/aspect_sentiment_gold.csv`.
- `ReviewsLabelling/runs/20260330_133939/final_excluded.csv`: excluded merchant replies. Do not merge these rows back into the gold set.
- `ReviewsLabelling/runs/20260330_133939/annotated.csv`: auto-labeled baseline before final manual adjudication. Keep for audit only.

## Usage Guidance

- Use only `E1/aspect_sentiment_gold.csv` as the final E1 sentence-level gold input.
- Do not treat `review_resolved.csv` as an additional dataset that still needs to be appended.
- Do not reintroduce rows from `final_excluded.csv`.

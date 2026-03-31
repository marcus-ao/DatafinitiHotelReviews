import unittest

import pandas as pd

from scripts.pipeline import build_evidence_vector_index as vector_mod
from scripts.pipeline import build_hotel_aspect_profiles as profile_mod
from scripts.pipeline import classify_sentence_aspects as aspect_mod


class AspectPipelineTestCase(unittest.TestCase):
    def test_rule_classifier_can_return_multiple_aspects(self):
        labels = aspect_mod.match_aspects_rule(
            "Great location near downtown and the staff were friendly."
        )
        aspects = {label["aspect"] for label in labels}
        self.assertIn("location_transport", aspects)
        self.assertIn("service", aspects)

    def test_compute_profile_row_uses_recency_weights_and_controversy(self):
        group = pd.DataFrame(
            [
                {"sentiment": "positive", "recency_weight": 1.2},
                {"sentiment": "positive", "recency_weight": 1.0},
                {"sentiment": "negative", "recency_weight": 0.8},
                {"sentiment": "neutral", "recency_weight": 0.9},
            ]
        )
        row = profile_mod.compute_profile_row("h1", "service", group)
        self.assertEqual(row["pos_count"], 2)
        self.assertEqual(row["neg_count"], 1)
        self.assertEqual(row["neu_count"], 1)
        self.assertEqual(row["total_count"], 4)
        self.assertAlmostEqual(row["recency_weighted_pos"], 2.2)
        self.assertAlmostEqual(row["recency_weighted_neg"], 0.8)
        self.assertAlmostEqual(row["controversy_score"], 0.5)
        self.assertAlmostEqual(row["final_aspect_score"], 1.25)

    def test_build_primary_meta_prefers_non_general_and_higher_confidence(self):
        rows = pd.DataFrame(
            [
                {
                    "sentence_id": "s1",
                    "aspect": "general",
                    "confidence": 0.95,
                    "sentiment_confidence": 0.8,
                    "sentiment": "positive",
                    "label_source": "rule",
                },
                {
                    "sentence_id": "s1",
                    "aspect": "service",
                    "confidence": 0.80,
                    "sentiment_confidence": 0.7,
                    "sentiment": "positive",
                    "label_source": "rule",
                },
            ]
        )
        primary_meta = vector_mod.build_primary_meta(rows)
        self.assertEqual(primary_meta.iloc[0]["aspect"], "service")


if __name__ == "__main__":
    unittest.main()

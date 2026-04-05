import unittest

from scripts.evaluation import statistical_tests as stats_mod


class StatisticalTestsTestCase(unittest.TestCase):
    def test_wilcoxon_signed_rank_handles_normal_input(self):
        result = stats_mod.wilcoxon_signed_rank([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
        self.assertEqual(result["test"], "wilcoxon_signed_rank")
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["non_zero_n"], 3)
        self.assertGreaterEqual(result["p_value"], 0.0)

    def test_wilcoxon_signed_rank_handles_all_zero_differences(self):
        result = stats_mod.wilcoxon_signed_rank([1.0, 1.0], [1.0, 1.0])
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["note"], "all_zero_differences")
        self.assertEqual(result["zero_diff_count"], 2)

    def test_wilcoxon_signed_rank_handles_single_non_zero_pair(self):
        result = stats_mod.wilcoxon_signed_rank([1.0], [2.0])
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["note"], "insufficient_non_zero_pairs")

    def test_wilcoxon_signed_rank_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            stats_mod.wilcoxon_signed_rank([1.0, 2.0], [1.0])

    def test_bootstrap_ci_returns_ordered_interval(self):
        low, high = stats_mod.bootstrap_ci([1.0, 2.0, 3.0, 4.0], n_resamples=200, seed=7)
        self.assertLessEqual(low, high)

    def test_bootstrap_ci_is_reproducible(self):
        interval_a = stats_mod.bootstrap_ci([1.0, 2.0, 3.0], n_resamples=200, seed=99)
        interval_b = stats_mod.bootstrap_ci([1.0, 2.0, 3.0], n_resamples=200, seed=99)
        self.assertEqual(interval_a, interval_b)

    def test_cohens_d_handles_identical_groups(self):
        self.assertEqual(stats_mod.cohens_d([1.0, 1.0], [1.0, 1.0]), 0.0)

    def test_cohens_d_uses_paired_deltas(self):
        effect_size = stats_mod.cohens_d([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        self.assertEqual(effect_size, float("inf"))

    def test_rank_biserial_correlation_handles_identical_groups(self):
        self.assertEqual(stats_mod.rank_biserial_correlation([1.0, 2.0], [1.0, 2.0]), 0.0)

    def test_significance_stars_formats_thresholds(self):
        self.assertEqual(stats_mod.significance_stars(0.2), "ns")
        self.assertEqual(stats_mod.significance_stars(0.04), "*")
        self.assertEqual(stats_mod.significance_stars(0.009), "**")
        self.assertEqual(stats_mod.significance_stars(0.0009), "***")

    def test_compute_pairwise_tests_outputs_expected_columns(self):
        df = stats_mod.compute_pairwise_tests(
            {
                "G1": {"citation_precision": [0.8, 0.9, 1.0]},
                "G2": {"citation_precision": [0.9, 0.95, 1.0]},
            },
            metrics=["citation_precision"],
        )
        self.assertEqual(len(df), 1)
        self.assertEqual(list(df.columns), stats_mod.PAIRWISE_TEST_COLUMNS)
        self.assertEqual(df.iloc[0]["group_a"], "G1")
        self.assertEqual(df.iloc[0]["group_b"], "G2")
        self.assertEqual(df.iloc[0]["pairing_mode"], "position")
        self.assertEqual(df.iloc[0]["higher_is_better"], True)

    def test_compute_pairwise_tests_aligns_by_query_id_when_available(self):
        df = stats_mod.compute_pairwise_tests(
            {
                "G1": {
                    "citation_precision": {
                        "scores": [0.8, 0.9, 1.0],
                        "query_ids": ["q001", "q002", "q003"],
                    }
                },
                "G2": {
                    "citation_precision": {
                        "scores": [1.0, 0.95, 0.9],
                        "query_ids": ["q003", "q002", "q001"],
                    }
                },
            },
            metrics=["citation_precision"],
        )
        self.assertEqual(df.iloc[0]["pairing_mode"], "query_id")
        self.assertAlmostEqual(df.iloc[0]["mean_delta"], 0.05, places=4)
        self.assertEqual(df.iloc[0]["wins_group_b"], 2)
        self.assertEqual(df.iloc[0]["wins_group_a"], 0)

    def test_compute_pairwise_tests_rejects_query_id_mismatch(self):
        with self.assertRaisesRegex(ValueError, "query_ids 集合不一致"):
            stats_mod.compute_pairwise_tests(
                {
                    "G1": {"citation_precision": {"scores": [0.8], "query_ids": ["q001"]}},
                    "G2": {"citation_precision": {"scores": [0.9], "query_ids": ["q002"]}},
                },
                metrics=["citation_precision"],
            )

    def test_compute_pairwise_tests_marks_lower_is_better_metrics(self):
        df = stats_mod.compute_pairwise_tests(
            {
                "G1": {"hallucination_rate": [0.1, 0.2, 0.1]},
                "G2": {"hallucination_rate": [0.3, 0.2, 0.4]},
            },
            metrics=["hallucination_rate"],
        )
        self.assertEqual(df.iloc[0]["higher_is_better"], False)
        self.assertEqual(df.iloc[0]["better_group"], "G1")

    def test_compute_pairwise_tests_applies_holm_adjustment_per_metric(self):
        df = stats_mod.compute_pairwise_tests(
            {
                "G1": {
                    "citation_precision": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "schema_valid_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
                },
                "G2": {
                    "citation_precision": [0.6, 0.7, 0.8, 0.9, 1.0],
                    "schema_valid_rate": [0.6, 0.7, 0.8, 0.9, 1.0],
                },
                "G3": {
                    "citation_precision": [0.65, 0.75, 0.85, 0.95, 1.0],
                    "schema_valid_rate": [0.65, 0.75, 0.85, 0.95, 1.0],
                },
            },
            metrics=["citation_precision", "schema_valid_rate"],
            p_adjust="holm",
        )
        self.assertTrue((df["p_adjust_method"] == "holm").all())
        self.assertTrue((df["p_value_adj"] >= df["p_value"]).all())

    def test_compute_pairwise_tests_respects_group_pair_order(self):
        df = stats_mod.compute_pairwise_tests(
            {
                "G1": {"citation_precision": [0.8, 0.9]},
                "G2": {"citation_precision": [0.9, 1.0]},
                "G3": {"citation_precision": [0.85, 0.95]},
            },
            metrics=["citation_precision"],
            group_pairs=[("G3", "G1")],
        )
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["group_a"], "G3")
        self.assertEqual(df.iloc[0]["group_b"], "G1")

    def test_compute_pairwise_tests_rejects_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            stats_mod.compute_pairwise_tests(
                {
                    "G1": {"citation_precision": [0.8, float("nan")]},
                    "G2": {"citation_precision": [0.9, 1.0]},
                },
                metrics=["citation_precision"],
            )


if __name__ == "__main__":
    unittest.main()

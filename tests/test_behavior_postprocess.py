import unittest
from pathlib import Path

from scripts.shared.behavior_postprocess import (
    load_query_ids_from_file,
    normalize_aspect_values,
    normalize_city_value,
    normalize_decision_label,
    normalize_unsupported_values,
)


class BehaviorPostprocessTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.city_to_state = {
            "Anaheim": "CA",
            "New Orleans": "LA",
            "San Francisco": "CA",
        }

    def test_normalize_city_value_accepts_city_state_variants(self):
        self.assertEqual(normalize_city_value("Anaheim:CA", self.city_to_state), "Anaheim")
        self.assertEqual(normalize_city_value("New Orleans, LA", self.city_to_state), "New Orleans")
        self.assertEqual(normalize_city_value("san francisco ca", self.city_to_state), "San Francisco")

    def test_normalize_aspect_values_accepts_chinese_labels(self):
        cleaned, unknown = normalize_aspect_values(["位置交通", "安静睡眠", "service"])
        self.assertEqual(cleaned, ["location_transport", "quiet_sleep", "service"])
        self.assertEqual(unknown, [])

    def test_normalize_unsupported_values_accepts_aliases(self):
        cleaned, unknown = normalize_unsupported_values(["预算", "distance to landmark", "入住日期"])
        self.assertEqual(cleaned, ["budget", "checkin_date", "distance_to_landmark"])
        self.assertEqual(unknown, [])

    def test_normalize_decision_label_accepts_aliases(self):
        self.assertEqual(normalize_decision_label("missing city"), "missing_city")
        self.assertEqual(normalize_decision_label("方面冲突"), "aspect_conflict")
        self.assertEqual(normalize_decision_label("none"), "none")

    def test_load_query_ids_from_json_list_preserves_order(self):
        query_ids = load_query_ids_from_file(
            Path(__file__).resolve().parents[1] / "experiments/assets/e4_diagnostic_query_ids.json"
        )
        self.assertEqual(query_ids[:3], ["q051", "q052", "q053"])
        self.assertEqual(len(query_ids), 32)


if __name__ == "__main__":
    unittest.main()

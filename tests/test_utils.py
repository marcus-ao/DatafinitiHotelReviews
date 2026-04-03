import os
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path("scripts").resolve()))

from utils import load_db_config, safe_parse_timestamp


class UtilsTestCase(unittest.TestCase):
    def test_safe_parse_timestamp_handles_millisecond_z_suffix(self):
        value = safe_parse_timestamp("2018-11-27T00:00:00.000Z")
        self.assertTrue(pd.notna(value))
        self.assertEqual(str(value.date()), "2018-11-27")

    def test_load_db_config_respects_env_password_override(self):
        old_password = os.environ.get("HOTEL_DB_PASSWORD")
        os.environ["HOTEL_DB_PASSWORD"] = "env_password"
        try:
            cfg = load_db_config("configs/db.yaml.example")
        finally:
            if old_password is None:
                os.environ.pop("HOTEL_DB_PASSWORD", None)
            else:
                os.environ["HOTEL_DB_PASSWORD"] = old_password

        self.assertEqual(cfg["password"], "env_password")
        self.assertEqual(cfg["schema"], "kb")


if __name__ == "__main__":
    unittest.main()

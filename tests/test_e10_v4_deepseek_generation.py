import os
import unittest

from scripts.evaluation import run_e10_v4_deepseek_generation as deepseek_mod


class DeepSeekGenerationTestCase(unittest.TestCase):
    def test_resolve_deepseek_settings_prefers_deepseek_env(self):
        old_env = {name: os.environ.get(name) for name in [
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_BASE_URL",
            "DEEPSEEK_REASONER_MODEL",
        ]}
        os.environ["DEEPSEEK_API_KEY"] = "test-key"
        os.environ["DEEPSEEK_BASE_URL"] = "https://example.deepseek.test"
        os.environ["DEEPSEEK_REASONER_MODEL"] = "deepseek-reasoner-x"
        try:
            settings = deepseek_mod.resolve_deepseek_settings()
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(settings["api_key"], "test-key")
        self.assertEqual(settings["base_url"], "https://example.deepseek.test")
        self.assertEqual(settings["model_name"], "deepseek-reasoner-x")

    def test_build_query_draft_row_sets_query_generated_status(self):
        request_row = {
            "request_id": "v4qry_001",
            "seed_id": "seed_001",
            "split": "train",
            "source_mode": "silver_deepseek",
            "primary_slice": "control_standard_grounded",
            "secondary_tags": ["single_hotel"],
            "city": "Seattle",
            "temperature": 0.6,
            "top_p": 0.9,
        }
        settings = {
            "model_name": "deepseek-reasoner",
            "base_url": "https://api.deepseek.com",
        }
        row = deepseek_mod.build_query_draft_row(request_row, "请推荐西雅图服务好的酒店。", settings)
        self.assertEqual(row["review_status"], "query_generated")
        self.assertEqual(row["query_text_zh"], "请推荐西雅图服务好的酒店。")
        self.assertEqual(row["provenance"]["generator_provider"], "deepseek")

    def test_build_target_draft_row_marks_parse_status(self):
        request_row = {
            "request_id": "v4tgt_001",
            "seed_id": "seed_001",
            "split": "train",
            "source_mode": "silver_deepseek",
            "primary_slice": "control_standard_grounded",
            "secondary_tags": ["single_hotel"],
            "query_id": "v4s_001",
            "query_text_zh": "请推荐西雅图服务好的酒店。",
            "temperature": 0.2,
            "top_p": 0.8,
        }
        settings = {
            "model_name": "deepseek-reasoner",
            "base_url": "https://api.deepseek.com",
        }
        row = deepseek_mod.build_target_draft_row(
            request_row,
            '{"summary":"ok","recommendations":[],"unsupported_notice":""}',
            settings,
        )
        self.assertEqual(row["review_status"], "target_generated")
        self.assertTrue(row["response_parse_ok"])
        self.assertEqual(row["target_payload"]["summary"], "ok")


if __name__ == "__main__":
    unittest.main()

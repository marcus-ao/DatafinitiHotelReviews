import os
import unittest

from scripts.shared import behavior_runtime as behavior_mod


class BehaviorRuntimeConfigTestCase(unittest.TestCase):
    def test_resolve_behavior_runtime_config_prefers_env_overrides_for_api_backend(self):
        cfg = {
            "behavior": {
                "llm_backend": "local",
                "base_model": "Qwen/Qwen2.5-3B-Instruct",
                "api_base_url": "http://config-host:8000/v1",
                "api_key_env": "OPENAI_API_KEY",
                "enable_thinking": True,
                "temperature": 0.3,
                "max_new_tokens": 128,
                "api_timeout_seconds": 60,
            }
        }
        frozen_config = {
            "behavior": {
                "base_model": "Qwen/Qwen3.5-2B",
            }
        }
        old_env = {name: os.environ.get(name) for name in [
            "BEHAVIOR_LLM_BACKEND",
            "BEHAVIOR_MODEL_ID",
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "BEHAVIOR_ENABLE_THINKING",
            "BEHAVIOR_TEMPERATURE",
            "BEHAVIOR_MAX_NEW_TOKENS",
            "BEHAVIOR_API_TIMEOUT_SECONDS",
        ]}
        os.environ["BEHAVIOR_LLM_BACKEND"] = "api"
        os.environ["BEHAVIOR_MODEL_ID"] = "Qwen/Qwen3.5-9B"
        os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:8000/v1"
        os.environ["OPENAI_API_KEY"] = "EMPTY"
        os.environ["BEHAVIOR_ENABLE_THINKING"] = "false"
        os.environ["BEHAVIOR_TEMPERATURE"] = "0"
        os.environ["BEHAVIOR_MAX_NEW_TOKENS"] = "256"
        os.environ["BEHAVIOR_API_TIMEOUT_SECONDS"] = "120"
        try:
            runtime_config, api_key = behavior_mod.resolve_behavior_runtime_config(cfg, frozen_config)
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(runtime_config.llm_backend, "api")
        self.assertEqual(runtime_config.model_id, "Qwen/Qwen3.5-9B")
        self.assertEqual(runtime_config.api_base_url, "http://127.0.0.1:8000/v1")
        self.assertFalse(runtime_config.enable_thinking)
        self.assertEqual(runtime_config.temperature, 0.0)
        self.assertEqual(runtime_config.max_new_tokens, 256)
        self.assertEqual(runtime_config.api_timeout_seconds, 120)
        self.assertEqual(api_key, "EMPTY")

    def test_flatten_openai_content_handles_mixed_text_blocks(self):
        content = [
            {"type": "text", "text": "{"},
            {"type": "text", "text": '"status"'},
            {"type": "text", "text": ': "ok"}'},
        ]
        self.assertEqual(behavior_mod.flatten_openai_content(content), '{"status": "ok"}')


if __name__ == "__main__":
    unittest.main()

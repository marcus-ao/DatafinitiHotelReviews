import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.training import training_utils as training_mod
from scripts.training import train_e10_peft as train_mod


class TrainingUtilsTestCase(unittest.TestCase):
    def test_load_train_config_rejects_unknown_task_type(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad_config.json"
            path.write_text(
                json.dumps(
                    {
                        "base_model_id": "/root/autodl-tmp/models/base/Qwen3.5-4B",
                        "adapter_type": "qlora",
                        "train_manifest_path": "train.jsonl",
                        "dev_manifest_path": "dev.jsonl",
                        "task_types": ["preference_parse", "unknown_task"],
                        "output_adapter_dir": "/root/autodl-tmp/models/adapters/qwen35_4b_qlora/exp01",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                training_mod.load_train_config(path)

    def test_load_train_config_accepts_grounded_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "good_config.json"
            path.write_text(
                json.dumps(
                    {
                        "base_model_id": "/root/autodl-tmp/models/base/Qwen3.5-4B",
                        "adapter_type": "qlora",
                        "train_manifest_path": "train_v2.jsonl",
                        "dev_manifest_path": "dev_v2.jsonl",
                        "task_types": ["preference_parse", "grounded_recommendation"],
                        "output_adapter_dir": "/root/autodl-tmp/models/adapters/qwen35_4b_qlora/exp02",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            config = training_mod.load_train_config(path)
        self.assertEqual(config.task_types, ["preference_parse", "grounded_recommendation"])

    def test_build_sft_text_sample_contains_task_and_json(self):
        row = {
            "record_id": "rec01",
            "query_id": "q001",
            "task_type": "preference_parse",
            "input_payload": {"query_text_zh": "测试输入"},
            "target_payload": {"city": "Anaheim"},
        }
        sample = training_mod.build_sft_text_sample(row)
        self.assertEqual(sample["record_id"], "rec01")
        self.assertIn("Task: preference_parse", sample["text"])
        self.assertIn('"query_text_zh": "测试输入"', sample["text"])
        self.assertIn('"city": "Anaheim"', sample["text"])

    def test_build_sft_trainer_kwargs_supports_processing_class_signature(self):
        class FakeTrainer:
            def __init__(
                self,
                model,
                args,
                train_dataset,
                eval_dataset,
                peft_config,
                processing_class=None,
                dataset_text_field=None,
                max_length=None,
            ):
                pass

        kwargs = training_mod.build_sft_trainer_kwargs(
            FakeTrainer,
            model="model",
            tokenizer="tok",
            args="args",
            train_dataset="train",
            eval_dataset="dev",
            peft_config="peft",
            max_seq_length=2048,
        )
        self.assertEqual(kwargs["processing_class"], "tok")
        self.assertEqual(kwargs["dataset_text_field"], "text")
        self.assertEqual(kwargs["max_length"], 2048)

    def test_build_sft_trainer_kwargs_supports_tokenizer_signature(self):
        class FakeTrainer:
            def __init__(
                self,
                model,
                args,
                train_dataset,
                eval_dataset,
                peft_config,
                tokenizer=None,
                dataset_text_field=None,
                max_seq_length=None,
            ):
                pass

        kwargs = training_mod.build_sft_trainer_kwargs(
            FakeTrainer,
            model="model",
            tokenizer="tok",
            args="args",
            train_dataset="train",
            eval_dataset="dev",
            peft_config="peft",
            max_seq_length=1024,
        )
        self.assertEqual(kwargs["tokenizer"], "tok")
        self.assertEqual(kwargs["dataset_text_field"], "text")
        self.assertEqual(kwargs["max_seq_length"], 1024)

    def test_load_manifest_records_filters_allowed_task_types(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "manifest.jsonl"
            path.write_text(
                '\n'.join(
                    [
                        json.dumps({"record_id": "r1", "query_id": "q1", "task_type": "preference_parse", "input_payload": {}, "target_payload": {}}),
                        json.dumps({"record_id": "r2", "query_id": "q2", "task_type": "clarification", "input_payload": {}, "target_payload": {}}),
                        json.dumps({"record_id": "r3", "query_id": "q3", "task_type": "ignore_me", "input_payload": {}, "target_payload": {}}),
                    ]
                ),
                encoding="utf-8",
            )
            rows = training_mod.load_manifest_records(path, {"preference_parse", "clarification"})
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["record_id"] for row in rows], ["r1", "r2"])

    def test_write_training_metadata_outputs_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = training_mod.E10TrainConfig(
                base_model_id="/root/autodl-tmp/models/base/Qwen3.5-4B",
                adapter_type="qlora",
                train_manifest_path="experiments/assets/sft_train_manifest.jsonl",
                dev_manifest_path="experiments/assets/sft_dev_manifest.jsonl",
                task_types=["preference_parse"],
                output_adapter_dir=str(Path(tmp_dir) / "models" / "adapters" / "exp01"),
            )
            output_paths = training_mod.ensure_output_dirs(config)
            metadata_path, summary_path = training_mod.write_training_metadata(
                config=config,
                output_paths=output_paths,
                train_rows=[{"record_id": "r1"}],
                dev_rows=[{"record_id": "r2"}],
            )
            self.assertTrue(metadata_path.exists())
            self.assertTrue(summary_path.exists())

    def test_assert_sft_samples_within_max_seq_length_raises_for_overflow(self):
        class FakeTokenizer:
            def __call__(self, text, add_special_tokens=True, truncation=False):
                return {"input_ids": list(range(len(text)))}

        samples = [
            {
                "record_id": "r_overflow",
                "query_id": "q001",
                "task_type": "grounded_recommendation",
                "text": "x" * 3000,
            }
        ]
        with self.assertRaises(ValueError):
            training_mod.assert_sft_samples_within_max_seq_length(
                samples,
                FakeTokenizer(),
                2048,
                dataset_name="train_dataset",
            )

    def test_train_dry_run_does_not_create_cloud_root_paths(self):
        fake_train_rows = [
            {
                "record_id": "r1",
                "query_id": "q1",
                "task_type": "preference_parse",
                "input_payload": {"query_text_zh": "测试"},
                "target_payload": {"city": "Anaheim"},
            }
        ]
        fake_dev_rows = [
            {
                "record_id": "r2",
                "query_id": "q2",
                "task_type": "clarification",
                "input_payload": {"query_text_zh": "测试"},
                "target_payload": {"clarify_needed": False},
            }
        ]
        config = training_mod.E10TrainConfig(
            base_model_id="/root/autodl-tmp/models/base/Qwen3.5-4B",
            adapter_type="qlora",
            train_manifest_path="experiments/assets/sft_train_manifest.jsonl",
            dev_manifest_path="experiments/assets/sft_dev_manifest.jsonl",
            task_types=["preference_parse", "clarification"],
            output_adapter_dir="/root/autodl-tmp/models/adapters/qwen35_4b_qlora/exp01",
        )
        with mock.patch.object(train_mod, "load_train_config", return_value=config), \
             mock.patch.object(train_mod, "load_manifest_records", side_effect=[fake_train_rows, fake_dev_rows]):
            result = train_mod.run_training("ignored.json", dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["train_sample_count"], 1)
        self.assertEqual(result["dev_sample_count"], 1)
        self.assertEqual(result["adapter_metadata_path"], "")


if __name__ == "__main__":
    unittest.main()

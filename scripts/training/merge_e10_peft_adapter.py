"""Merge an E10 PEFT adapter into a standalone model and optionally copy metadata."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from pathlib import PurePosixPath
from pathlib import PureWindowsPath
from typing import Any


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def normalize_external_path_string(path_value: str | Path) -> str:
    path_text = str(path_value).strip()
    if PurePosixPath(path_text).is_absolute() or PureWindowsPath(path_text).is_absolute():
        return path_text
    return str(resolve_repo_path(path_text))


def build_default_report_dir(adapter_path: str | Path) -> str:
    adapter_path_text = str(adapter_path).strip()
    if "/models/adapters/" in adapter_path_text:
        return adapter_path_text.replace("/models/adapters/", "/training/reports/")
    adapter_dir = Path(adapter_path_text)
    return str(adapter_dir).replace("\\models\\adapters\\", "\\training\\reports\\")


def merge_e10_peft_adapter(
    *,
    base_model_path: str | Path,
    adapter_path: str | Path,
    merged_output_path: str | Path,
    report_dir: str | Path | None = None,
    repo_metadata_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    normalized_base_model_path = normalize_external_path_string(base_model_path)
    normalized_adapter_path = normalize_external_path_string(adapter_path)
    normalized_merged_output_path = normalize_external_path_string(merged_output_path)
    normalized_report_dir = (
        normalize_external_path_string(report_dir)
        if report_dir
        else normalize_external_path_string(build_default_report_dir(adapter_path))
    )

    resolved_base_model_path = Path(normalized_base_model_path)
    resolved_adapter_path = Path(normalized_adapter_path)
    resolved_merged_output_path = Path(normalized_merged_output_path)
    resolved_report_dir = Path(normalized_report_dir)
    resolved_report_metadata_path = resolved_report_dir / "adapter_metadata.json"
    resolved_repo_metadata_path = (
        resolve_repo_path(repo_metadata_path) if repo_metadata_path else None
    )
    normalized_report_metadata_path = (
        normalized_report_dir.rstrip("/\\") + "/adapter_metadata.json"
        if PurePosixPath(normalized_report_dir).is_absolute()
        else str(resolved_report_metadata_path)
    )

    result = {
        "base_model_path": normalized_base_model_path,
        "adapter_path": normalized_adapter_path,
        "merged_output_path": normalized_merged_output_path,
        "report_dir": normalized_report_dir,
        "report_metadata_path": normalized_report_metadata_path,
        "repo_metadata_path": str(resolved_repo_metadata_path) if resolved_repo_metadata_path else "",
        "dry_run": dry_run,
    }
    if dry_run:
        return result

    if not resolved_report_metadata_path.exists():
        raise FileNotFoundError(
            f"Missing training report metadata: {resolved_report_metadata_path}"
        )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_merged_output_path.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(resolved_base_model_path),
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(resolved_base_model_path),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, str(resolved_adapter_path))
    model = model.merge_and_unload()
    model.save_pretrained(str(resolved_merged_output_path))
    tokenizer.save_pretrained(str(resolved_merged_output_path))

    if resolved_repo_metadata_path is not None:
        resolved_repo_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved_report_metadata_path, resolved_repo_metadata_path)

    result["status"] = "ok"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--merged-output-path", required=True)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--repo-metadata-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = merge_e10_peft_adapter(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        merged_output_path=args.merged_output_path,
        report_dir=args.report_dir,
        repo_metadata_path=args.repo_metadata_path,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

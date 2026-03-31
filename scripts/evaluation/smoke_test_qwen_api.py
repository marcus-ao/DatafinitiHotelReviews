from __future__ import annotations

import os

from openai import OpenAI


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    model_id = os.getenv("BEHAVIOR_MODEL_ID", "Qwen/Qwen3.5-4B")
    base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    enable_thinking = _parse_bool(os.getenv("BEHAVIOR_ENABLE_THINKING"), default=False)

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_id,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": 'Return a one-line JSON object: {"status":"ok"}'},
        ],
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
            }
        },
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()

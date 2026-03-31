from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-2B",
    temperature=0,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Return a one-line JSON object: {\"status\":\"ok\"}"},
    ],
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    },
)

print(response.choices[0].message.content)

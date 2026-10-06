from __future__ import annotations

import json
import time
from typing import Any

import httpx

from app.config import get_llm_config, get_settings


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise LLMError("Model response did not contain a JSON object.")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"Model returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMError("Model response must be a JSON object.")
    return value


def llm_json(
    *,
    system_prompt: str,
    user_payload: dict[str, Any] | str,
    response_contract: str,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
    retries: int = 2,
) -> dict[str, Any]:
    settings = get_settings()
    model = get_llm_config()
    user_text = user_payload if isinstance(user_payload, str) else json.dumps(user_payload, ensure_ascii=False, default=str)
    messages = [
        {
            "role": "system",
            "content": (
                system_prompt.strip()
                + "\n\nReturn ONLY one valid JSON object. Do not use Markdown or prose outside JSON.\n"
                + "Required JSON contract:\n"
                + response_contract.strip()
            ),
        },
        {"role": "user", "content": user_text},
    ]
    payload = {
        "model": model.id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": min(max_output_tokens or model.maxOutputTokens, model.maxOutputTokens),
    }
    headers = {"Content-Type": "application/json"}
    if settings.indiaai_api_key:
        headers["Authorization"] = f"Bearer {settings.indiaai_api_key}"

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=settings.llm_timeout_seconds) as client:
                response = client.post(str(model.url), headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            return _extract_json(content)
        except (httpx.HTTPError, KeyError, TypeError, ValueError, LLMError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.6 * (2**attempt))
                continue
            break
    raise LLMError(f"IndiaAI inference failed: {last_error}")

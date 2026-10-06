from __future__ import annotations

import json
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


def _message_content(body: dict[str, Any]) -> str:
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        preview = json.dumps(body, ensure_ascii=False)[:1000]
        raise LLMError(f"Unexpected IndiaAI response shape: {preview}") from exc

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts)

    raise LLMError(f"Unsupported message.content type: {type(content).__name__}")


def llm_json(
    *,
    system_prompt: str,
    user_payload: dict[str, Any] | str,
    response_contract: str,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    model = get_llm_config()

    user_text = (
        user_payload
        if isinstance(user_payload, str)
        else json.dumps(user_payload, ensure_ascii=False, default=str)
    )

    payload = {
        "model": model.id,
        "messages": [
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
        ],
        "temperature": temperature,
        "max_tokens": min(max_output_tokens or model.maxOutputTokens, model.maxOutputTokens),
    }

    headers = {"Content-Type": "application/json"}
    if settings.indiaai_api_key:
        headers["Authorization"] = f"Bearer {settings.indiaai_api_key}"

    timeout = httpx.Timeout(
        timeout=settings.llm_timeout_seconds,
        connect=min(settings.llm_timeout_seconds, 10.0),
    )

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(str(model.url), headers=headers, json=payload)
    except httpx.TimeoutException as exc:
        raise LLMError(
            f"IndiaAI request timed out after {settings.llm_timeout_seconds:.0f}s."
        ) from exc
    except httpx.ConnectError as exc:
        raise LLMError(f"Could not connect to IndiaAI endpoint: {exc}") from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"IndiaAI transport error: {exc}") from exc

    if response.status_code >= 400:
        body = response.text.strip().replace("\n", " ")[:1200]
        raise LLMError(
            f"IndiaAI returned HTTP {response.status_code}: {body or '<empty response body>'}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        preview = response.text.strip()[:1200]
        raise LLMError(f"IndiaAI returned non-JSON response: {preview}") from exc

    return _extract_json(_message_content(body))


def probe_llm() -> dict[str, Any]:
    model = get_llm_config()
    result = llm_json(
        system_prompt="You are a connectivity probe. Return the requested JSON exactly.",
        user_payload="Return an object with ok=true.",
        response_contract='{"ok": true}',
        max_output_tokens=64,
    )
    return {"ok": result.get("ok") is True, "model": model.id}

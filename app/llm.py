from __future__ import annotations

import json
from contextvars import ContextVar, Token
from typing import Any, Callable

import httpx

from app.config import get_llm_config, get_settings


class LLMError(RuntimeError):
    pass


LLMEventCallback = Callable[[dict[str, Any]], None]
_event_callback: ContextVar[LLMEventCallback | None] = ContextVar("llm_event_callback", default=None)


def set_llm_event_callback(callback: LLMEventCallback | None) -> Token:
    return _event_callback.set(callback)


def reset_llm_event_callback(token: Token) -> None:
    _event_callback.reset(token)


def _emit(event: dict[str, Any]) -> None:
    callback = _event_callback.get()
    if callback:
        callback(event)


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


def _stream_delta(event: dict[str, Any]) -> tuple[str, int]:
    try:
        choice = event["choices"][0]
    except (KeyError, IndexError, TypeError):
        return "", 0

    delta = choice.get("delta") or {}
    if not isinstance(delta, dict):
        return "", 0

    content = delta.get("content")
    content_text = content if isinstance(content, str) else ""

    reasoning = delta.get("reasoning_content")
    reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0

    return content_text, reasoning_chars


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
        "stream": True,
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if settings.indiaai_api_key:
        headers["Authorization"] = f"Bearer {settings.indiaai_api_key}"

    timeout = httpx.Timeout(
        connect=min(settings.llm_timeout_seconds, 10.0),
        read=settings.llm_timeout_seconds,
        write=min(settings.llm_timeout_seconds, 30.0),
        pool=min(settings.llm_timeout_seconds, 10.0),
    )

    _emit({"type": "llm_request", "model": model.id})

    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", str(model.url), headers=headers, json=payload) as response:
                _emit(
                    {
                        "type": "llm_connected",
                        "status_code": response.status_code,
                        "content_type": response.headers.get("content-type", ""),
                    }
                )

                if response.status_code >= 400:
                    body = response.read().decode("utf-8", errors="replace").strip().replace("\n", " ")[:1200]
                    raise LLMError(
                        f"IndiaAI returned HTTP {response.status_code}: {body or '<empty response body>'}"
                    )

                content_type = response.headers.get("content-type", "").lower()
                if "text/event-stream" not in content_type:
                    preview = response.read().decode("utf-8", errors="replace").strip()[:500]
                    raise LLMError(
                        "IndiaAI did not return an SSE stream for stream=true. "
                        f"content-type={content_type or '<missing>'}; response={preview}"
                    )

                content_parts: list[str] = []
                received_chars = 0
                received_events = 0
                done_seen = False

                for line in response.iter_lines():
                    if not line:
                        continue

                    text = line.strip()
                    if not text.startswith("data:"):
                        continue

                    data = text[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        done_seen = True
                        break

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise LLMError(f"IndiaAI emitted invalid SSE JSON: {data[:300]}") from exc

                    received_events += 1
                    content_delta, reasoning_chars = _stream_delta(event)
                    activity_chars = len(content_delta) + reasoning_chars

                    if content_delta:
                        content_parts.append(content_delta)

                    if activity_chars:
                        received_chars += activity_chars
                        _emit(
                            {
                                "type": "llm_delta",
                                "received_chars": received_chars,
                                "stream_events": received_events,
                            }
                        )

                content = "".join(content_parts).strip()
                _emit(
                    {
                        "type": "llm_complete",
                        "received_chars": received_chars,
                        "stream_events": received_events,
                        "done_seen": done_seen,
                    }
                )

                if not content:
                    raise LLMError(
                        "IndiaAI stream completed without assistant content. "
                        f"Received {received_events} SSE event(s)."
                    )

                return _extract_json(content)

    except LLMError:
        raise
    except httpx.TimeoutException as exc:
        raise LLMError(
            f"IndiaAI stream was idle for {settings.llm_timeout_seconds:.0f}s."
        ) from exc
    except httpx.ConnectError as exc:
        raise LLMError(f"Could not connect to IndiaAI endpoint: {exc}") from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"IndiaAI streaming transport error: {exc}") from exc


def probe_llm() -> dict[str, Any]:
    model = get_llm_config()
    result = llm_json(
        system_prompt="You are a connectivity probe. Return the requested JSON exactly.",
        user_payload="Return an object with ok=true.",
        response_contract='{"ok": true}',
        max_output_tokens=64,
    )
    return {"ok": result.get("ok") is True, "model": model.id}

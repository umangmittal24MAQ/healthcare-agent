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
        preview = " ".join(text.split())[:500]
        raise LLMError(
            "Model response did not contain a JSON object. "
            f"Response preview: {preview or '<empty>'}"
        )

    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        error_at = exc.pos
        preview_start = max(0, error_at - 120)
        preview_end = min(len(text), error_at + 120)
        preview = " ".join(text[preview_start:preview_end].split())
        raise LLMError(
            f"Model returned invalid JSON: {exc}. Around error: {preview}"
        ) from exc

    if not isinstance(value, dict):
        raise LLMError("Model response must be a JSON object.")
    return value


def _stream_delta(event: dict[str, Any]) -> tuple[str, str, str | None]:
    try:
        choice = event["choices"][0]
    except (KeyError, IndexError, TypeError):
        return "", "", None

    if not isinstance(choice, dict):
        return "", "", None

    delta = choice.get("delta") or {}
    if not isinstance(delta, dict):
        delta = {}

    content = delta.get("content")
    content_text = content if isinstance(content, str) else ""

    reasoning = delta.get("reasoning_content")
    reasoning_text = reasoning if isinstance(reasoning, str) else ""

    finish_reason = choice.get("finish_reason")
    finish_text = finish_reason if isinstance(finish_reason, str) else None

    return content_text, reasoning_text, finish_text


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
                    + "\n\nSTRICT OUTPUT REQUIREMENTS:\n"
                    + "- Return exactly one valid JSON object.\n"
                    + "- The first non-whitespace character must be { and the last must be }.\n"
                    + "- Do not output Markdown, code fences, explanations, analysis, reasoning, or <think> tags.\n"
                    + "- Do not prepend or append any text outside the JSON object.\n"
                    + "Required JSON contract:\n"
                    + response_contract.strip()
                ),
            },
            {"role": "user", "content": user_text},
        ],
        "temperature": temperature,
        "max_tokens": min(max_output_tokens or model.maxOutputTokens, model.maxOutputTokens),
        "response_format": {"type": "json_object"},
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
                content_chars = 0
                reasoning_chars = 0
                received_events = 0
                done_seen = False
                finish_reason: str | None = None

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
                    content_delta, reasoning_delta, chunk_finish_reason = _stream_delta(event)

                    if chunk_finish_reason:
                        finish_reason = chunk_finish_reason

                    if content_delta:
                        content_parts.append(content_delta)
                        content_chars += len(content_delta)

                    if reasoning_delta:
                        reasoning_chars += len(reasoning_delta)

                    received_chars = content_chars + reasoning_chars
                    if content_delta or reasoning_delta:
                        _emit(
                            {
                                "type": "llm_delta",
                                "received_chars": received_chars,
                                "content_chars": content_chars,
                                "reasoning_chars": reasoning_chars,
                                "stream_events": received_events,
                                "finish_reason": finish_reason,
                            }
                        )

                content = "".join(content_parts).strip()
                received_chars = content_chars + reasoning_chars
                _emit(
                    {
                        "type": "llm_complete",
                        "received_chars": received_chars,
                        "content_chars": content_chars,
                        "reasoning_chars": reasoning_chars,
                        "stream_events": received_events,
                        "done_seen": done_seen,
                        "finish_reason": finish_reason,
                    }
                )

                if finish_reason == "length":
                    raise LLMError(
                        "IndiaAI exhausted the output-token budget before completing the structured response "
                        f"(content_chars={content_chars}, reasoning_chars={reasoning_chars}, "
                        f"events={received_events}). Increase max_output_tokens or make the stage output more concise."
                    )

                if not content:
                    raise LLMError(
                        "IndiaAI stream completed without final assistant content "
                        f"(reasoning_chars={reasoning_chars}, finish_reason={finish_reason or 'unknown'}, "
                        f"events={received_events})."
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
        max_output_tokens=1024,
    )
    return {"ok": result.get("ok") is True, "model": model.id}

from __future__ import annotations

import json

import pytest

import app.llm as llm


class FakeResponse:
    status_code = 200
    headers = {"content-type": "text/event-stream"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_lines(self):
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "checking"}}]},
            {"choices": [{"delta": {"content": '{"ok":'}}]},
            {"choices": [{"delta": {"content": "true}"}}]},
        ]
        for chunk in chunks:
            yield "data: " + json.dumps(chunk)
        yield "data: [DONE]"

    def read(self):
        return b""


class FakeClient:
    captured = None

    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.get("timeout")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, *, headers, json):
        FakeClient.captured = {
            "method": method,
            "url": url,
            "headers": headers,
            "json": json,
        }
        return FakeResponse()


def test_llm_streaming_client_reassembles_json_and_requests_json_mode(monkeypatch):
    monkeypatch.setattr(llm.httpx, "Client", FakeClient)

    events = []
    token = llm.set_llm_event_callback(events.append)
    try:
        result = llm.llm_json(
            system_prompt="Return probe result.",
            user_payload={"probe": True},
            response_contract='{"ok":true}',
            max_output_tokens=64,
        )
    finally:
        llm.reset_llm_event_callback(token)

    assert result == {"ok": True}
    assert FakeClient.captured["method"] == "POST"
    assert FakeClient.captured["url"] == "https://indiaai.maqsoftware.net/v1/chat/completions"
    assert FakeClient.captured["json"]["model"] == "qwen-3.8-27b"
    assert FakeClient.captured["json"]["stream"] is True
    assert FakeClient.captured["json"]["response_format"] == {"type": "json_object"}
    assert FakeClient.captured["headers"]["Accept"] == "text/event-stream"

    event_types = [event["type"] for event in events]
    assert event_types[0] == "llm_request"
    assert "llm_connected" in event_types
    assert "llm_delta" in event_types
    assert event_types[-1] == "llm_complete"


def test_extract_json_error_contains_response_preview():
    with pytest.raises(llm.LLMError) as exc:
        llm._extract_json("I will explain the answer instead of returning JSON.")

    message = str(exc.value)
    assert "did not contain a JSON object" in message
    assert "I will explain" in message

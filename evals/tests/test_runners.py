from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request

import pytest

from evals.models import load_dataset
from evals.runners import (
    EvaluationError,
    FixtureRunner,
    HttpSseRunner,
    OpenResponse,
    ResponseHeaders,
)

_ROOT = Path(__file__).resolve().parents[2]
_DATASET = load_dataset(_ROOT / "evals" / "datasets" / "phase1-seed.json")
_CASE = _DATASET.cases[0]


class _Headers:
    def __init__(self, content_type: str = "text/event-stream; charset=utf-8") -> None:
        self._content_type = content_type

    def get(self, name: str, default: str | None = None) -> str | None:
        return self._content_type if name.lower() == "content-type" else default


class _Response:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        content_type: str = "text/event-stream; charset=utf-8",
    ) -> None:
        self.status = status
        self.headers: ResponseHeaders = _Headers(content_type)
        self._body = io.BytesIO(body)

    def readline(self, limit: int = -1) -> bytes:
        return self._body.readline(limit)

    def __enter__(self) -> _Response:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool | None:
        del exc_type, exc_value, traceback
        return None


def _sse(*events: tuple[str, dict[str, object]]) -> bytes:
    return "".join(
        f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
        for event, data in events
    ).encode()


def _opener_for(response: _Response) -> Callable[[Request, float], OpenResponse]:
    def opener(_request: Request, _timeout: float) -> OpenResponse:
        return response

    return opener


def test_fixture_runner_returns_expected_answer() -> None:
    result = FixtureRunner().generate(_DATASET, _CASE)

    assert result.text == _CASE.expected_answer
    assert result.finish_reason == "stop"


def test_http_runner_sends_provider_neutral_request_and_parses_sse() -> None:
    captured: dict[str, object] = {}
    body = _sse(
        ("token", {"text": "Take ", "index": 0, "finish_reason": None}),
        ("token", {"text": "10 mg.", "index": 1, "finish_reason": None}),
        ("complete", {"text": "", "index": 2, "finish_reason": "stop"}),
    )

    def opener(request: Request, timeout: float) -> _Response:
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        request_data = request.data
        assert isinstance(request_data, bytes)
        captured["payload"] = json.loads(request_data)
        return _Response(body)

    runner = HttpSseRunner(
        "http://127.0.0.1:8080/api/v1/debug/generate",
        timeout_seconds=3,
        max_tokens=64,
        opener=opener,
    )
    result = runner.generate(_DATASET, _CASE)

    assert result.text == "Take 10 mg."
    assert result.finish_reason == "stop"
    assert result.ttft_ms >= 0
    assert result.latency_ms >= result.ttft_ms
    assert captured["url"] == "http://127.0.0.1:8080/api/v1/debug/generate"
    assert captured["timeout"] == 3
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["metadata"] == {
        "evaluation_dataset": "phase1-seed",
        "evaluation_case": "discharge-medication",
    }
    assert payload["temperature"] == 0.0
    assert _CASE.context in payload["messages"][0]["content"]


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.example.com:443/v1/debug/generate",
        "http://user@localhost:8080/v1/debug/generate",
        "http://localhost:8080/other",
        "http://bad-:8080/v1/debug/generate",
        "http://localhost/v1/debug/generate",
        "http://0.0.0.0:8080/v1/debug/generate",
        "http://169.254.1.1:8080/v1/debug/generate",
        "http://224.0.0.1:8080/v1/debug/generate",
        "http://[::]:8080/v1/debug/generate",
    ],
)
def test_http_runner_rejects_nonlocal_or_noncanonical_endpoints(endpoint: str) -> None:
    with pytest.raises(ValueError, match="endpoint"):
        HttpSseRunner(endpoint)


@pytest.mark.parametrize(
    ("timeout", "max_tokens"),
    [(0.0, 10), (float("nan"), 10), (1.0, 0), (1.0, 4097)],
)
def test_http_runner_rejects_invalid_limits(timeout: float, max_tokens: int) -> None:
    with pytest.raises(ValueError):
        HttpSseRunner(
            "http://localhost:8080/v1/debug/generate",
            timeout_seconds=timeout,
            max_tokens=max_tokens,
        )


@pytest.mark.parametrize(
    "body",
    [
        _sse(("token", {"text": "x", "index": 1, "finish_reason": None})),
        _sse(("unknown", {"text": "x", "index": 0, "finish_reason": None})),
        b"event: token\ndata: not-json\n\n",
        b'event: token\ndata: {"text":"x","text":"y","index":0,"finish_reason":null}\n\n',
        _sse(
            ("token", {"text": "x", "index": 0, "finish_reason": None}),
            ("complete", {"text": "ignored", "index": 1, "finish_reason": "stop"}),
        ),
        b"event: token\ndata: {}\n\n",
        b'event: token\ndata: {"text":"x","index":0,"finish_reason":null}\n',
        _sse(("complete", {"text": "", "index": 0, "finish_reason": "stop"})),
    ],
)
def test_http_runner_rejects_malformed_streams(body: bytes) -> None:
    runner = HttpSseRunner(
        "http://localhost:8080/v1/debug/generate",
        opener=lambda _request, _timeout: _Response(body),
    )

    with pytest.raises(EvaluationError, match="stream"):
        runner.generate(_DATASET, _CASE)


def test_http_runner_errors_are_content_safe() -> None:
    def unavailable(_request: Request, _timeout: float) -> _Response:
        raise URLError(f"must not expose {_CASE.context}")

    runner = HttpSseRunner(
        "http://localhost:8080/v1/debug/generate",
        opener=unavailable,
    )

    with pytest.raises(EvaluationError) as caught:
        runner.generate(_DATASET, _CASE)
    assert _CASE.context not in str(caught.value)


def test_http_runner_rejects_status_content_type_and_size() -> None:
    for response in (
        _Response(b"", status=503),
        _Response(b"", content_type="application/json"),
        _Response(b"", content_type="text/event-stream-malformed"),
        _Response(b"x" * (64 * 1024 + 1)),
    ):
        runner = HttpSseRunner(
            "http://localhost:8080/v1/debug/generate",
            opener=_opener_for(response),
        )
        with pytest.raises(EvaluationError):
            runner.generate(_DATASET, _CASE)

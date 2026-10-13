"""Shared test fixtures: keyless spy model, capture stub server, event readers."""

from __future__ import annotations

import base64
import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from pydantic import PrivateAttr

HINT_HEADER = "X-Agent-Hint"


class SpyChatModel(GenericFakeChatModel):
    """Keyless spy: records the bind kwargs the hint header travels in."""

    _binds: list = PrivateAttr(default_factory=list)

    def bind(self, **kwargs: Any) -> Any:
        self._binds.append(kwargs)
        return self

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        self._binds.append(kwargs)
        return self


def hinted_headers(spy: SpyChatModel) -> Iterator[str]:
    """Encoded X-Agent-Hint values captured from every model bind call."""
    for kwargs in spy._binds:  # noqa: SLF001 (test access to the spy)
        headers = kwargs.get("extra_headers") or {}
        if HINT_HEADER in headers:
            yield headers[HINT_HEADER]


def independent_decode(encoded: str) -> dict[str, Any]:
    """Decode the wire hint without reusing the package's encode path."""
    padded = encoded + "=" * (-len(encoded) % 4)
    raw = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    return json.loads(raw)


def read_status_events(path: Path, session_id: str | None = None) -> list[dict[str, Any]]:
    """Status events from a JSONL sink, optionally filtered by session.

    A missing file means "no events were ever written" (the degradation case).
    """
    if not path.exists():
        return []
    events = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if session_id is None:
        return events
    return [event for event in events if event["session"]["session_id"] == session_id]


class _CaptureHandler(BaseHTTPRequestHandler):
    """Record each request as one JSON line; answer one plain completion."""

    capture_path: Path
    model_name: str = "stub-model"

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8") if length else ""
        record = {
            "path": self.path,
            "headers": {key.lower(): value for key, value in self.headers.items()},
            "body": json.loads(body) if body else None,
        }
        with self.capture_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        payload = {
            "id": "chatcmpl-agent-hints-stub",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "AGENT_HINTS_WIRE_OK"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture()
def stub_server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """In-process OpenAI-compatible capture stub; yields (base_url, capture_path)."""
    capture = tmp_path / "capture.jsonl"
    handler = type("_CapturingHandler", (_CaptureHandler,), {"capture_path": capture})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", capture
    finally:
        server.shutdown()
        server.server_close()

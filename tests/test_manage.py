"""Manage-request channel: wire shape, placeholder body, retry and barrier."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from conftest import independent_decode
from langchain_agent_hints.manage import HttpManageRequestClient, ManageRequestError
from langchain_agent_hints.middleware import AgentHintsMiddleware
from langchain_agent_hints.schema import build_manage_hint, encode_agent_hint


def _records(capture_path: Path) -> list[dict[str, Any]]:
    if not capture_path.exists():
        return []
    return [
        json.loads(line)
        for line in capture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_manage_hint_is_session_plus_context_management_only() -> None:
    hint = build_manage_hint(session_id="s-1", verb="compact", call_index=2, epoch=0)
    wire = independent_decode(encode_agent_hint(hint))
    assert wire["session"]["state"] == "compact"
    assert wire["session"]["call_index"] == 2
    assert wire["session"]["epoch"] == 0
    assert wire["context_management"] == {
        "manage_request": True,
        "edits": [{"type": "evict", "target": "session"}],
    }
    assert set(wire) == {"session", "context_management"}


def test_manage_client_posts_placeholder_body_and_header(stub_server) -> None:
    base_url, capture = stub_server
    client = HttpManageRequestClient(
        url=f"{base_url}/chat/completions", model="stub-model", api_key="control-key"
    )

    assert client.send("pause", "s-1", call_index=3, epoch=1) is True

    records = _records(capture)
    assert len(records) == 1
    record = records[0]
    assert record["body"] == {"model": "stub-model", "messages": [], "stream": False}
    assert record["headers"]["authorization"] == "Bearer control-key"
    wire = independent_decode(record["headers"]["x-agent-hint"])
    assert wire["session"]["state"] == "pause"
    assert wire["context_management"] == {
        "manage_request": True,
        "edits": [{"type": "offload", "target": "session"}],
    }
    assert "priority_control" not in wire and "cache_control" not in wire


class _FlakyHandler(BaseHTTPRequestHandler):
    """First ``failures`` requests get 503; then 200."""

    failures = 1
    received: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8")
        type(self).received.append(
            {"body": json.loads(body), "hint": self.headers.get("X-Agent-Hint")}
        )
        if len(type(self).received) <= type(self).failures:
            self.send_response(503)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture()
def flaky_server() -> Any:
    handler = type("_H", (_FlakyHandler,), {"failures": 1, "received": []})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, handler
    finally:
        server.shutdown()
        server.server_close()


def test_resume_retries_transient_failure(flaky_server) -> None:
    server, handler = flaky_server
    client = HttpManageRequestClient(
        url=f"http://127.0.0.1:{server.server_address[1]}/chat/completions",
        model="m",
        timeout=1.0,
        attempts=3,
    )
    assert client.send("resume", "s-1", call_index=4, epoch=2) is True
    assert len(handler.received) == 2


def test_mutation_does_not_retry_and_isolates_the_session(flaky_server) -> None:
    server, handler = flaky_server
    client = HttpManageRequestClient(
        url=f"http://127.0.0.1:{server.server_address[1]}/chat/completions",
        model="m",
        timeout=1.0,
        attempts=3,
    )
    with pytest.raises(ManageRequestError):
        client.send("pause", "s-1", call_index=3, epoch=0)
    assert len(handler.received) == 1  # no retry for a mutation
    assert client.is_isolated("s-1")
    # The barrier refuses new mainline traffic for an unknown KV state.
    with pytest.raises(ManageRequestError):
        client.ensure_ready("s-1")


# -- middleware integration --------------------------------------------------


class _FakeManage:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.isolated: set[str] = set()

    def ensure_ready(self, session_id: str) -> None:
        if session_id in self.isolated:
            raise ManageRequestError("isolated")

    def send(
        self,
        verb: str,
        session_id: str,
        *,
        call_index: int,
        epoch: int,
        parent_session_id: str | None = None,
    ) -> bool:
        self.sent.append(
            {
                "verb": verb,
                "session_id": session_id,
                "call_index": call_index,
                "epoch": epoch,
            }
        )
        return True


class _Request:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages
        self.model_settings: dict[str, Any] = {}
        self.model = type("M", (), {"model_name": "m"})()
        self.tools: list[Any] = []
        self.system_message = None

    def override(self, **kwargs: Any) -> _Request:
        clone = _Request(self.messages)
        clone.model_settings = kwargs.get("model_settings", self.model_settings)
        return clone


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}, "metadata": {}}


def test_new_session_sends_no_resume(monkeypatch) -> None:
    from langchain_core.messages import HumanMessage

    import langchain_agent_hints.middleware as middleware_mod

    monkeypatch.setattr(middleware_mod, "ensure_config", lambda: _config("new"))
    manage = _FakeManage()
    mw = AgentHintsMiddleware(agent_type="a", manage=manage)  # type: ignore[arg-type]
    mw._hinted_request(_Request([HumanMessage(content="hi")]))
    assert manage.sent == []


def test_adopted_session_sends_resume_before_the_request(monkeypatch) -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    import langchain_agent_hints.middleware as middleware_mod

    monkeypatch.setattr(middleware_mod, "ensure_config", lambda: _config("adopted"))
    manage = _FakeManage()
    mw = AgentHintsMiddleware(agent_type="a", manage=manage)  # type: ignore[arg-type]
    mw._hinted_request(_Request([HumanMessage(content="one"), AIMessage(content="two")]))
    assert manage.sent == [{"verb": "resume", "session_id": "adopted", "call_index": 2, "epoch": 0}]


def test_rewrite_sends_compact_with_the_old_epoch(monkeypatch) -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    import langchain_agent_hints.middleware as middleware_mod

    monkeypatch.setattr(middleware_mod, "ensure_config", lambda: _config("rewrite"))
    manage = _FakeManage()
    mw = AgentHintsMiddleware(agent_type="a", manage=manage)  # type: ignore[arg-type]
    mw._hinted_request(
        _Request(
            [
                HumanMessage(content="a"),
                AIMessage(content="b"),
                HumanMessage(content="c"),
            ]
        )
    )
    manage.sent.clear()
    hinted = mw._hinted_request(_Request([HumanMessage(content="summary")]))
    assert hinted is not None
    assert manage.sent == [
        {"verb": "compact", "session_id": "rewrite", "call_index": 2, "epoch": 0}
    ]
    wire = independent_decode(hinted.model_settings["extra_headers"]["X-Agent-Hint"])
    assert wire["session"]["state"] == "append"
    assert wire["session"]["epoch"] == 1
    assert "context_management" not in wire


def test_pause_then_next_request_resumes(monkeypatch) -> None:
    from langchain_core.messages import HumanMessage, SystemMessage

    import langchain_agent_hints.middleware as middleware_mod

    monkeypatch.setattr(middleware_mod, "ensure_config", lambda: _config("idle"))
    manage = _FakeManage()
    mw = AgentHintsMiddleware(agent_type="a", manage=manage)  # type: ignore[arg-type]
    system = SystemMessage(content="stable system prompt")
    mw._hinted_request(_Request([system, HumanMessage(content="one")]))
    mw.send_manage("pause", "idle")
    mw._hinted_request(_Request([system, HumanMessage(content="two")]))
    assert [item["verb"] for item in manage.sent] == ["pause", "resume"]

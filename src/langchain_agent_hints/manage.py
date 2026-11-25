"""Out-of-band manage requests — the lifecycle channel (protocol §7.3/§8.3).

Lifecycle verbs never ride on an ordinary model request. A ``pause`` /
``resume`` / ``stop`` / ``compact`` is delivered as its own HTTP request whose
body is a placeholder ``{model, messages: [], stream: false}`` and whose only
payload is the ``X-Agent-Hint`` header:

    session + context_management{manage_request: true, edits: [one edit]}

The client is deliberately small and transport-only:

- one ``POST`` per verb, serialized per session (no interleaved mutations);
- only ``resume`` retries transient failures (408/429/5xx/network);
- an unknown result for a mutation (``pause`` / ``stop`` / ``compact``) puts
  the session into isolation: later ordinary requests for that session MUST
  fail explicitly rather than silently racing the unknown KV state
  (protocol §8.4);
- ``HTTP 2xx`` means "accepted", never "KV action completed".

Nothing here is model-visible: a broken control endpoint degrades the
lifecycle channel, not the inference request itself (the middleware decides
whether a mutation failure blocks the run).
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Literal

from langchain_agent_hints.schema import HINT_HEADER, build_manage_hint, encode_agent_hint

ManageVerb = Literal["pause", "resume", "stop", "compact"]

#: HTTP statuses that are explicitly transient for ``resume`` (protocol §8.4).
_RETRYABLE_STATUS = frozenset({408, 429})


class ManageRequestError(RuntimeError):
    """A manage request failed, or the session is isolated by an earlier one."""


class HttpManageRequestClient:
    """Send manage requests to a model or dedicated control endpoint.

    ``url`` is the full chat-completions endpoint (or an explicit control
    endpoint). ``model`` is the placeholder body's model id. ``api_key`` is
    forwarded as a bearer token — when the control endpoint is a different
    origin from the data plane, pass a control-plane key instead (protocol
    §8.3).
    """

    def __init__(
        self,
        *,
        url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 2.0,
        attempts: int = 3,
        headers: dict[str, str] | None = None,
    ) -> None:
        if not url:
            msg = "manage endpoint url is required"
            raise ValueError(msg)
        if not model:
            msg = "manage placeholder model is required"
            raise ValueError(msg)
        self._url = url
        self._model = model
        self._api_key = api_key
        self._timeout = float(timeout)
        self._attempts = max(1, int(attempts))
        self._headers = dict(headers or {})
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._isolated: dict[str, str] = {}

    # -- session guards ------------------------------------------------------

    def _lock_for(self, session_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(session_id, threading.Lock())

    def _isolation_reason(self, session_id: str) -> str | None:
        with self._guard:
            return self._isolated.get(session_id)

    def is_isolated(self, session_id: str) -> bool:
        """Whether an earlier mutation left this session's KV state unknown."""
        return self._isolation_reason(session_id) is not None

    def ensure_ready(self, session_id: str) -> None:
        """Barrier for a mainline request: wait for mutations, refuse isolation.

        Acquiring the per-session lock serializes the check with any in-flight
        manage request; releasing it immediately lets this request proceed.
        """
        reason = self._isolation_reason(session_id)
        if reason is not None:
            msg = (
                f"session {session_id!r} is isolated after an unknown manage "
                f"result ({reason}); start a new session_id to continue"
            )
            raise ManageRequestError(msg)
        lock = self._lock_for(session_id)
        with lock:
            reason = self._isolation_reason(session_id)
            if reason is not None:
                msg = (
                    f"session {session_id!r} is isolated after an unknown manage "
                    f"result ({reason}); start a new session_id to continue"
                )
                raise ManageRequestError(msg)

    def _isolate(self, session_id: str, detail: str) -> None:
        with self._guard:
            self._isolated[session_id] = detail

    def clear_isolation(self, session_id: str) -> None:
        """Operator/test escape hatch: forget an isolation record."""
        with self._guard:
            self._isolated.pop(session_id, None)

    # -- wire ----------------------------------------------------------------

    def _payload(
        self,
        verb: ManageVerb,
        session_id: str,
        *,
        call_index: int,
        epoch: int,
        parent_session_id: str | None,
    ) -> tuple[bytes, dict[str, str]]:
        hint = build_manage_hint(
            session_id=session_id,
            verb=verb,
            call_index=call_index,
            epoch=epoch,
            parent_session_id=parent_session_id,
        )
        body = json.dumps(
            {"model": self._model, "messages": [], "stream": False},
            separators=(",", ":"),
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            HINT_HEADER: encode_agent_hint(hint),
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        headers.update(self._headers)
        return body, headers

    def send(
        self,
        verb: ManageVerb,
        session_id: str,
        *,
        call_index: int,
        epoch: int,
        parent_session_id: str | None = None,
    ) -> bool:
        """Send one manage request.

        Returns ``True`` when the endpoint accepted it. ``resume`` returns
        ``False`` after exhausting its bounded retries (a missed prefetch is
        advisory); mutations raise :class:`ManageRequestError` and isolate the
        session because their outcome is unknown (protocol §8.4).
        """
        body, headers = self._payload(
            verb,
            session_id,
            call_index=call_index,
            epoch=epoch,
            parent_session_id=parent_session_id,
        )
        lock = self._lock_for(session_id)
        with lock:
            reason = self._isolation_reason(session_id)
            if reason is not None:
                msg = (
                    f"session {session_id!r} is isolated after an unknown manage result ({reason})"
                )
                raise ManageRequestError(msg)

            attempts = self._attempts if verb == "resume" else 1
            last_error: Exception | None = None
            for attempt in range(attempts):
                request = urllib.request.Request(
                    self._url, data=body, headers=headers, method="POST"
                )
                try:
                    with urllib.request.urlopen(request, timeout=self._timeout) as response:
                        response.read()
                    return True
                except urllib.error.HTTPError as error:
                    last_error = error
                    if verb == "resume" and (
                        error.code in _RETRYABLE_STATUS or 500 <= error.code < 600
                    ):
                        if attempt + 1 < attempts:
                            time.sleep(min(0.2 * (attempt + 1), 1.0))
                            continue
                        return False
                    detail = f"{verb} HTTP {error.code}"
                    self._isolate(session_id, detail)
                    msg = f"manage {verb} for {session_id!r} failed: HTTP {error.code}"
                    raise ManageRequestError(msg) from error
                except (urllib.error.URLError, OSError) as error:
                    last_error = error
                    if verb == "resume" and attempt + 1 < attempts:
                        time.sleep(min(0.2 * (attempt + 1), 1.0))
                        continue
                    if verb == "resume":
                        return False
                    detail = f"{verb} {type(error).__name__}: {error}"
                    self._isolate(session_id, detail)
                    msg = f"manage {verb} for {session_id!r} failed: {error}"
                    raise ManageRequestError(msg) from error

            if verb == "resume":
                return False
            detail = f"{verb} {last_error}"
            self._isolate(session_id, detail)
            msg = f"manage {verb} for {session_id!r} failed: {last_error}"
            raise ManageRequestError(msg)

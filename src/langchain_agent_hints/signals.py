"""Run signals — the in-process bridge between the status and hint channels.

The hint header and the out-of-band status events observe the same run from
two different middleware hooks. Two derived values need the status channel's
observation to be *real* rather than a request-preparation guess:

- ``agent_status.state_since_ms`` — the instant the current state began (the
  protocol's "tracked transition instant");
- ``priority`` — an approval resume is a user-intervened window, so the next
  model call must report ``3`` / ``user_waiting`` even though its last message
  is a tool result (the approved tool).

Mount both middlewares with the same ``RunSignals`` instance to share them.
Without the status middleware the hint middleware degrades to the trigger
classification and the request wall clock.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class _SessionSignal:
    state_since_ms: int | None = None
    resumed: bool = False


class RunSignals:
    """Thread-safe per-session signals shared by the two middlewares."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, _SessionSignal] = {}

    def observe_transition(
        self,
        session_id: str,
        previous: str | None,
        state: str,
        now_ms: int,
    ) -> None:
        """Record one lifecycle transition (called by the status middleware).

        A transition out of ``pause`` into ``running``/``tool`` is an approval
        resume window: the next hint reports user-intervened priority.
        """
        with self._lock:
            signal = self._sessions.setdefault(session_id, _SessionSignal())
            signal.state_since_ms = now_ms
            if previous == "pause" and state in ("running", "tool"):
                signal.resumed = True

    def state_since_ms(self, session_id: str) -> int | None:
        """Timestamp of the current state's start, or ``None`` (cold start)."""
        with self._lock:
            signal = self._sessions.get(session_id)
            return signal.state_since_ms if signal is not None else None

    def take_resumed(self, session_id: str) -> bool:
        """Consume the one-shot approval-resume flag."""
        with self._lock:
            signal = self._sessions.get(session_id)
            if signal is None or not signal.resumed:
                return False
            signal.resumed = False
            return True

    def finish(self, session_id: str) -> None:
        """Drop the session's signals (run completed)."""
        with self._lock:
            self._sessions.pop(session_id, None)

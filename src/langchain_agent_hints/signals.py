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
    state: str | None = None
    resumed: bool = False
    resume_approval: bool = False
    pause_approval: bool = False


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
        reason: str | None = None,
    ) -> None:
        """Record one lifecycle transition (called by the status middleware).

        A transition out of ``pause``/``suspend``/``tool`` into ``running`` is
        a one-shot release window: the next hint reports ``resume``. Only a
        release from a *human* pause (human approval / checkpoint) is
        user-intervened; a tool release stays ``critical_path``.
        """
        with self._lock:
            signal = self._sessions.setdefault(session_id, _SessionSignal())
            signal.state_since_ms = now_ms
            signal.state = state
            if state == "pause":
                signal.pause_approval = reason in ("human_approval", "checkpoint")
            released_from_wait = (
                previous in ("pause", "suspend") and state in ("running", "tool")
            ) or (previous == "tool" and state == "running")
            if released_from_wait:
                signal.resumed = True
                signal.resume_approval = bool(
                    previous in ("pause", "suspend") and signal.pause_approval
                )
                signal.pause_approval = False

    def state_since_ms(self, session_id: str) -> int | None:
        """Timestamp of the current state's start, or ``None`` (cold start)."""
        with self._lock:
            signal = self._sessions.get(session_id)
            return signal.state_since_ms if signal is not None else None

    def current_state(self, session_id: str) -> str | None:
        """Current internal state (``tool``/``suspend`` included), or None."""
        with self._lock:
            signal = self._sessions.get(session_id)
            return signal.state if signal is not None else None

    def take_resumed(self, session_id: str) -> bool:
        """Consume the one-shot release flag (approval or tool)."""
        with self._lock:
            signal = self._sessions.get(session_id)
            if signal is None or not signal.resumed:
                return False
            signal.resumed = False
            return True

    def take_approval_resume(self, session_id: str) -> bool:
        """Whether the current one-shot release came from a human pause.

        Consumed independently of :meth:`take_resumed` so the header can report
        ``resume`` for a tool release while priority stays ``critical_path``.
        """
        with self._lock:
            signal = self._sessions.get(session_id)
            if signal is None or not signal.resume_approval:
                return False
            signal.resume_approval = False
            return True

    def finish(self, session_id: str) -> None:
        """Drop the session's signals (run completed)."""
        with self._lock:
            self._sessions.pop(session_id, None)

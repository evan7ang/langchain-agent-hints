"""Observation statistics — the in-process serving estimate engine.

LangChain/middleware hooks expose everything a statistics engine needs, so the
estimates are collected **inside the plugin** (unlike Dify, where platform
boundaries force an out-of-band hint service to forward the data):

- ``wrap_model_call`` measures latency, output tokens (``usage_metadata``) and
  whether the response asked for tools;
- ``wrap_tool_call`` measures tool duration and records the tool transition;
- ``after_agent`` closes a session (calls per session, session duration) and
  records the finalize transition.

Aggregation is per ``agent_type`` with a rolling window (bounded memory) and a
per-session scratch record. Emission picks one value per metric (nearest-rank
quantile, default p90) — the protocol's "distribution collapses to a single
value before sending" rule in ``Agent Hint字段&实现定义.md`` §3.

Everything is advisory: with too few samples the metric is omitted (omission
is the encoding for "no estimate"), and a disabled store is a no-op.
"""

from __future__ import annotations

import json
import math
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_agent_hints.schema import NextActionHint, PredictedTool, PredictionHint

DEFAULT_QUANTILE = 0.9
DEFAULT_MIN_SAMPLES = 3
DEFAULT_MAX_SAMPLES = 256


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (no interpolation); ``values`` must be non-empty."""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


class _Series:
    """Bounded rolling sample window."""

    __slots__ = ("_maxlen", "_values")

    def __init__(self, maxlen: int) -> None:
        self._maxlen = max(1, int(maxlen))
        self._values: deque[float] = deque(maxlen=self._maxlen)

    def add(self, value: float | int | None) -> None:
        if value is None:
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if numeric >= 0:
            self._values.append(numeric)

    def __len__(self) -> int:
        return len(self._values)

    def percentile(self, q: float) -> float | None:
        if not self._values:
            return None
        return percentile(list(self._values), q)

    def mean(self) -> float | None:
        if not self._values:
            return None
        return sum(self._values) / len(self._values)

    def values(self) -> list[float]:
        return list(self._values)

    def load(self, values: list[float]) -> None:
        self._values = deque(values[-self._maxlen :], maxlen=self._maxlen)


@dataclass(frozen=True)
class ObservationOptions:
    """Static behavior of the observation engine.

    ``enabled`` is the master switch (a disabled store records nothing and the
    middleware emits no ``prediction``). ``min_samples`` gates each metric: a
    value is only emitted once at least this many samples exist. ``quantile``
    collapses a distribution to one value (p90 by default). ``persist_path``
    optionally carries the aggregate across processes (JSON, atomic writes).
    """

    enabled: bool = True
    quantile: float = DEFAULT_QUANTILE
    min_samples: int = DEFAULT_MIN_SAMPLES
    max_samples: int = DEFAULT_MAX_SAMPLES
    persist_path: Path | str | None = None


class _TypeStats:
    """Per-agent-type aggregates (shared across sessions)."""

    __slots__ = (
        "latency_ms",
        "output_tokens",
        "interval_ms",
        "tool_duration_ms",
        "calls_per_session",
        "session_duration_ms",
        "tool_transitions",
        "finalize_transitions",
        "next_tool_counts",
    )

    def __init__(self, max_samples: int) -> None:
        self.latency_ms = _Series(max_samples)
        self.output_tokens = _Series(max_samples)
        self.interval_ms = _Series(max_samples)
        self.tool_duration_ms: dict[str, _Series] = {}
        self.calls_per_session = _Series(max_samples)
        self.session_duration_ms = _Series(max_samples)
        self.tool_transitions = 0
        self.finalize_transitions = 0
        self.next_tool_counts: dict[str, int] = {}

    def tool_series(self, name: str) -> _Series:
        series = self.tool_duration_ms.get(name)
        if series is None:
            series = _Series(self.latency_ms._maxlen)  # noqa: SLF001 — same window
            self.tool_duration_ms[name] = series
        return series


class _SessionStats:
    """Per-session scratch facts (not persisted; a restart resets them)."""

    __slots__ = (
        "started_ms",
        "last_call_end_ms",
        "last_interval_ms",
        "calls",
        "output_tokens",
        "pending_outcome",
    )

    def __init__(self, now_ms: int) -> None:
        self.started_ms = now_ms
        self.last_call_end_ms: int | None = None
        self.last_interval_ms: int | None = None
        self.calls = 0
        self.output_tokens = 0
        # A model response is awaiting its transition outcome (tool/finalize).
        self.pending_outcome = False


class ObservationStore:
    """Thread-safe rolling statistics for measured serving estimates.

    One instance per composition; all hooks may call it concurrently (parallel
    tool batches, concurrent subagent runs). ``snapshot()`` exposes the raw
    aggregate for diagnostics/tests.
    """

    def __init__(self, options: ObservationOptions | None = None) -> None:
        self._options = options or ObservationOptions()
        self._lock = threading.RLock()
        self._types: dict[str, _TypeStats] = {}
        self._sessions: dict[str, _SessionStats] = {}
        if self._options.persist_path:
            self._load()

    # -- configuration -------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._options.enabled

    @property
    def options(self) -> ObservationOptions:
        return self._options

    def reset(self) -> None:
        """Drop every observation (tests / tenant switch)."""
        with self._lock:
            self._types.clear()
            self._sessions.clear()

    # -- recording -----------------------------------------------------------

    def model_call_started(self, session_id: str, *, now_ms: int) -> int | None:
        """Start a model call; return the measured gap since the previous call ended."""
        if not self._options.enabled:
            return None
        with self._lock:
            session = self._sessions.setdefault(session_id, _SessionStats(now_ms))
            interval: int | None = None
            if session.last_call_end_ms is not None:
                interval = max(0, int(now_ms - session.last_call_end_ms))
                session.last_interval_ms = interval
                self._stats_for(session_id).interval_ms.add(interval)
            return interval

    def model_call_finished(
        self,
        session_id: str,
        agent_type: str,
        *,
        latency_ms: float | int | None,
        output_tokens: int | None,
        had_tool_calls: bool,
        now_ms: int,
    ) -> None:
        if not self._options.enabled:
            return
        with self._lock:
            session = self._sessions.setdefault(session_id, _SessionStats(now_ms))
            session.calls += 1
            session.last_call_end_ms = now_ms
            session.pending_outcome = True
            stats = self._stats_for(agent_type)
            stats.latency_ms.add(latency_ms)
            stats.output_tokens.add(output_tokens)
            if output_tokens:
                session.output_tokens += output_tokens
            # A response without tool calls ends the run; record the outcome
            # now so the aggregate learns even if after_agent never fires.
            if not had_tool_calls:
                self._record_outcome_locked(session, stats, "finalize")

    def tool_started(self, session_id: str, agent_type: str, tool_name: str) -> None:
        """Record that the last model response is followed by a tool call."""
        if not self._options.enabled:
            return
        with self._lock:
            session = self._sessions.setdefault(session_id, _SessionStats(0))
            stats = self._stats_for(agent_type)
            if session.pending_outcome:
                self._record_outcome_locked(session, stats, "tool")
            stats.next_tool_counts[tool_name] = stats.next_tool_counts.get(tool_name, 0) + 1

    def tool_finished(
        self, agent_type: str, tool_name: str, duration_ms: float | int | None
    ) -> None:
        if not self._options.enabled:
            return
        with self._lock:
            self._stats_for(agent_type).tool_series(tool_name).add(duration_ms)

    def run_finished(self, session_id: str, agent_type: str, *, now_ms: int) -> None:
        """Close one session: final transition, calls-per-session, duration."""
        if not self._options.enabled:
            return
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                return
            stats = self._stats_for(agent_type)
            if session.pending_outcome:
                self._record_outcome_locked(session, stats, "finalize")
            if session.calls:
                stats.calls_per_session.add(session.calls)
                stats.session_duration_ms.add(max(0, now_ms - session.started_ms))
            self._flush_locked()

    # -- emission ------------------------------------------------------------

    def prediction(
        self,
        session_id: str,
        agent_type: str,
        *,
        call_index: int,
        now_ms: int,
    ) -> PredictionHint | None:
        """Measured estimates for the request being prepared (may be partial)."""
        if not self._options.enabled:
            return None
        q = self._options.quantile
        min_samples = max(1, self._options.min_samples)
        with self._lock:
            stats = self._types.get(agent_type)
            session = self._sessions.get(session_id)
            if stats is None:
                return None

            def measured(series: _Series) -> float | None:
                return series.percentile(q) if len(series) >= min_samples else None

            latency = measured(stats.latency_ms)
            output = measured(stats.output_tokens)
            interval: float | None = None
            if session is not None and session.last_interval_ms is not None:
                interval = float(session.last_interval_ms)
            else:
                interval = measured(stats.interval_ms)

            remaining_calls: int | None = None
            average_calls = (
                stats.calls_per_session.mean()
                if len(stats.calls_per_session) >= min_samples
                else None
            )
            if average_calls is not None:
                remaining_calls = max(0, int(round(average_calls - call_index)))

            remaining_time: int | None = None
            average_duration = (
                stats.session_duration_ms.mean()
                if len(stats.session_duration_ms) >= min_samples
                else None
            )
            if average_duration is not None and session is not None:
                elapsed = max(0, now_ms - session.started_ms)
                remaining_time = max(0, int(round(average_duration - elapsed)))

            next_action = self._next_action(stats, min_samples)

        if all(
            value is None
            for value in (latency, output, interval, remaining_calls, remaining_time, next_action)
        ):
            return None
        return PredictionHint(
            latency_ms=None if latency is None else int(round(latency)),
            remaining_calls=remaining_calls,
            output_tokens=None if output is None else int(round(output)),
            interval_ms=None if interval is None else int(round(interval)),
            remaining_time_ms=remaining_time,
            next_action=next_action,
        )

    # -- internals -----------------------------------------------------------

    def _stats_for(self, agent_type: str) -> _TypeStats:
        stats = self._types.get(agent_type)
        if stats is None:
            stats = _TypeStats(self._options.max_samples)
            self._types[agent_type] = stats
        return stats

    def _record_outcome_locked(
        self, session: _SessionStats, stats: _TypeStats, outcome: str
    ) -> None:
        session.pending_outcome = False
        if outcome == "tool":
            stats.tool_transitions += 1
        else:
            stats.finalize_transitions += 1

    def _next_action(self, stats: _TypeStats, min_samples: int) -> NextActionHint | None:
        total = stats.tool_transitions + stats.finalize_transitions
        if total < min_samples:
            return None
        tool_share = stats.tool_transitions / total
        if tool_share >= 0.5:
            top_tool = (
                max(stats.next_tool_counts, key=stats.next_tool_counts.__getitem__)
                if stats.next_tool_counts
                else None
            )
            return NextActionHint(
                state="tool",
                tools=(PredictedTool(name=top_tool),) if top_tool else None,
                prob=round(tool_share, 2),
            )
        return NextActionHint(state="finalize", prob=round(1.0 - tool_share, 2))

    # -- optional persistence ------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "types": {
                    name: {
                        "latency_ms": stats.latency_ms.values(),
                        "output_tokens": stats.output_tokens.values(),
                        "interval_ms": stats.interval_ms.values(),
                        "tool_duration_ms": {
                            tool: series.values() for tool, series in stats.tool_duration_ms.items()
                        },
                        "calls_per_session": stats.calls_per_session.values(),
                        "session_duration_ms": stats.session_duration_ms.values(),
                        "tool_transitions": stats.tool_transitions,
                        "finalize_transitions": stats.finalize_transitions,
                        "next_tool_counts": dict(stats.next_tool_counts),
                    }
                    for name, stats in self._types.items()
                }
            }

    def _flush_locked(self) -> None:
        path = self._options.persist_path
        if not path:
            return
        target = Path(path)
        payload = json.dumps(self.snapshot(), ensure_ascii=False)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(target)
        except OSError:
            # Advisory data: a broken stats file must never fail an agent run.
            return

    def _load(self) -> None:
        path = Path(self._options.persist_path)  # type: ignore[arg-type]
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8") or "{}")
        except (OSError, json.JSONDecodeError):
            return
        types = payload.get("types") or {}
        if not isinstance(types, dict):
            return
        with self._lock:
            for name, data in types.items():
                if not isinstance(data, dict):
                    continue
                stats = self._stats_for(str(name))
                stats.latency_ms.load(_numbers(data.get("latency_ms")))
                stats.output_tokens.load(_numbers(data.get("output_tokens")))
                stats.interval_ms.load(_numbers(data.get("interval_ms")))
                stats.calls_per_session.load(_numbers(data.get("calls_per_session")))
                stats.session_duration_ms.load(_numbers(data.get("session_duration_ms")))
                for tool, values in (data.get("tool_duration_ms") or {}).items():
                    stats.tool_series(str(tool)).load(_numbers(values))
                stats.tool_transitions = _count(data.get("tool_transitions"))
                stats.finalize_transitions = _count(data.get("finalize_transitions"))
                counts = data.get("next_tool_counts") or {}
                if isinstance(counts, dict):
                    stats.next_tool_counts = {
                        str(key): _count(value) for key, value in counts.items()
                    }


def _numbers(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    result: list[float] = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            continue
    return result


def _count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0

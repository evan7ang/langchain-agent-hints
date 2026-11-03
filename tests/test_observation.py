"""Observation engine: percentile collapse, gating, outcomes, persistence."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_agent_hints.observation import (
    ObservationOptions,
    ObservationStore,
    percentile,
)


def test_nearest_rank_percentile() -> None:
    values = [10, 20, 30, 40]
    assert percentile(values, 0.5) == 20
    assert percentile(values, 0.9) == 40
    assert percentile([7], 0.9) == 7


def test_store_gates_metrics_below_min_samples() -> None:
    store = ObservationStore(ObservationOptions(min_samples=3))
    store.model_call_started("s", now_ms=0)
    store.model_call_finished(
        "s", "a", latency_ms=10, output_tokens=5, had_tool_calls=False, now_ms=10
    )
    assert store.prediction("s", "a", call_index=1, now_ms=11) is None


def test_store_collects_measured_prediction() -> None:
    store = ObservationStore(ObservationOptions(min_samples=2))
    for index in range(2):
        session = f"s{index}"
        store.model_call_started(session, now_ms=index * 1000)
        store.model_call_finished(
            session,
            "agent",
            latency_ms=40 + index,
            output_tokens=100 + index,
            had_tool_calls=True,
            now_ms=index * 1000 + 50,
        )
        store.tool_started(session, "agent", "web_search")
        store.tool_finished("agent", "web_search", 20)
        store.tool_started(session, "agent", "web_search")  # no pending outcome now
        store.model_call_started(session, now_ms=index * 1000 + 120)
        store.model_call_finished(
            session,
            "agent",
            latency_ms=50,
            output_tokens=120,
            had_tool_calls=False,
            now_ms=index * 1000 + 160,
        )
        store.run_finished(session, "agent", now_ms=index * 1000 + 200)

    prediction = store.prediction("new", "agent", call_index=1, now_ms=5000)
    assert prediction is not None
    assert prediction.latency_ms == 50  # p90 of [40, 50, 50, 50]
    assert prediction.output_tokens == 120  # p90 of [100, 101, 120]
    assert prediction.remaining_calls == 1  # 2 calls per completed session
    assert prediction.next_action is not None
    assert prediction.next_action.state == "running"  # tool folds to running on the wire
    assert prediction.next_action.tools is not None
    assert prediction.next_action.tools[0].name == "web_search"
    assert prediction.next_action.prob == 0.5


def test_finalize_majority_predicts_finalize() -> None:
    store = ObservationStore(ObservationOptions(min_samples=2))
    for index in range(2):
        session = f"f{index}"
        store.model_call_started(session, now_ms=0)
        store.model_call_finished(
            session, "agent", latency_ms=1, output_tokens=1, had_tool_calls=False, now_ms=1
        )
        store.run_finished(session, "agent", now_ms=2)
    prediction = store.prediction("new", "agent", call_index=1, now_ms=3)
    assert prediction is not None and prediction.next_action is not None
    assert prediction.next_action.state == "finalize"
    assert prediction.next_action.prob == 1.0


def test_disabled_store_is_a_noop() -> None:
    store = ObservationStore(ObservationOptions(enabled=False))
    store.model_call_started("s", now_ms=0)
    store.model_call_finished(
        "s", "a", latency_ms=1, output_tokens=1, had_tool_calls=False, now_ms=1
    )
    assert store.prediction("s", "a", call_index=1, now_ms=2) is None


def test_persistence_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    options = ObservationOptions(min_samples=1, persist_path=path)
    store = ObservationStore(options)
    for index in range(2):
        session = f"p{index}"
        store.model_call_started(session, now_ms=0)
        store.model_call_finished(
            session, "agent", latency_ms=30, output_tokens=11, had_tool_calls=False, now_ms=5
        )
        store.run_finished(session, "agent", now_ms=10)
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["types"]["agent"]["latency_ms"] == [30.0, 30.0]

    restored = ObservationStore(options)
    prediction = restored.prediction("new", "agent", call_index=1, now_ms=20)
    assert prediction is not None and prediction.latency_ms == 30

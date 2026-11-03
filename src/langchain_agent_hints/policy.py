"""Cache-control policy — the harness-side rules of the cache_control guide.

Implements ``CACHE_CONTROL_IMPLEMENTATION_GUIDE.md``:

- one declaration only, ``type="ephemeral"`` (``pin`` is retired);
- ``ttl_s`` is mandatory and derived as ``clamp(gap * 2, 60, 3600)``;
- ``range.end`` is the **logical input count** ``[system?] + messages``
  (1-based, inclusive), never the append-only history length;
- a prefix shorter than one engine cache block (128 tokens) is not declared;
- compaction/branch, model change or tool-set change rounds are not declared;
- estimation failures degrade to "no declaration" — never to an error.
"""

from __future__ import annotations

from typing import Any

from langchain_agent_hints.schema import CacheControlHint, CacheControlRange

#: Engine cache block floor (one 128-token block).
MIN_CACHE_TOKENS = 128
#: TTL clamp bounds (the consumer clamps too; declaring the clamp keeps the
#: wire value explicit).
TTL_MIN_S = 60
TTL_MAX_S = 3600
#: Gap fallback when no tool/interval sample exists.
FALLBACK_GAP_S = 300.0
#: Estimator asymmetry: under-estimating loses 100% of the cache, so tolerate
#: an actual gap of up to 2x the estimate.
TTL_FACTOR = 2.0


def estimate_ttl_s(
    *,
    tool_expected_s: float | None = None,
    interval_ms: int | float | None = None,
) -> int:
    """TTL from the next-tool estimate, else the last inter-call gap, else 300 s.

    A tool estimate wins over the interval (the upcoming pause is the tool
    wait, not the previous gap), which is the guide's priority order.
    """
    gap: float
    if tool_expected_s is not None and tool_expected_s > 0:
        gap = float(tool_expected_s)
    elif interval_ms is not None and interval_ms > 0:
        gap = float(interval_ms) / 1000.0
    else:
        gap = FALLBACK_GAP_S
    return max(TTL_MIN_S, min(TTL_MAX_S, int(round(gap * TTL_FACTOR))))


def estimate_tokens(value: Any) -> int:
    """Tokenizer-free prefix size estimate (guide §6.2).

    CJK characters count as one token each; everything else as one token per
    four characters. Walks strings, message objects (``.content``), lists and
    dicts so the same helper serves LangGraph and Dify payloads.
    """
    cjk = 0
    other = 0

    def walk(item: Any) -> None:
        nonlocal cjk, other
        if item is None:
            return
        if isinstance(item, str):
            for char in item:
                code = ord(char)
                if 0x2E80 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF or 0xFF00 <= code <= 0xFFEF:
                    cjk += 1
                else:
                    other += 1
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
        elif isinstance(item, dict):
            for child in item.values():
                walk(child)
        elif hasattr(item, "content"):
            walk(item.content)
        elif hasattr(item, "text"):
            walk(item.text)

    walk(value)
    return cjk + (other + 3) // 4


def logical_message_count(messages: list[Any], *, system_present: bool) -> int:
    """Logical input count for ``range.end``: ``[system?] + messages``.

    A trailing empty assistant placeholder (no content, no tool calls) is not
    part of the logical input and is not counted. Messages that carry no
    ``content`` attribute are still counted (they occupy a wire slot).
    """
    kept = list(messages or [])
    if kept:
        last = kept[-1]
        content = getattr(last, "content", "")
        tool_calls = getattr(last, "tool_calls", None)
        role = str(getattr(last, "type", "") or getattr(last, "role", ""))
        if "ai" in role and not tool_calls and not str(content or "").strip():
            kept = kept[:-1]
    return len(kept) + (1 if system_present else 0)


def build_cache_control(
    *,
    logical_count: int,
    prefix_tokens: int,
    ttl_s: int,
) -> tuple[CacheControlHint, ...] | None:
    """One ``ephemeral`` declaration, or ``None`` when the guide says "don't".

    ``None`` for: unknown/empty input, a prefix below one cache block, or a
    non-positive TTL. Otherwise the TTL is clamped to ``[60, 3600]``.
    """
    if not isinstance(logical_count, int) or logical_count < 1:
        return None
    if prefix_tokens < MIN_CACHE_TOKENS:
        return None
    if not isinstance(ttl_s, int) or ttl_s <= 0:
        return None
    ttl = max(TTL_MIN_S, min(TTL_MAX_S, ttl_s))
    return (
        CacheControlHint(
            type="ephemeral",
            ttl_s=ttl,
            range=CacheControlRange(end=logical_count, unit="message"),
        ),
    )

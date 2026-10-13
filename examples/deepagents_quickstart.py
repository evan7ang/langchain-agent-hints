"""Wire langchain-agent-hints into a deepagents Deep Agent.

Requirements (not declared by the package itself):

    pip install deepagents langchain-openai
    export OPENAI_API_KEY=...   # any tool-calling model works

Any OpenAI-compatible gateway works too: point ``OPENAI_BASE_URL`` at it and
override the model string with ``AGENT_HINTS_EXAMPLE_MODEL`` (default
``openai:gpt-4o-mini``). Status events land in ``AGENT_HINTS_STATUS_SINK`` if
set, else ``examples/status-events.jsonl``.

Run from the repository root:

    python examples/deepagents_quickstart.py
"""

from __future__ import annotations

import os
from pathlib import Path

from deepagents import create_deep_agent

from langchain_agent_hints import (
    AgentHintsMiddleware,
    AgentStatusEventsMiddleware,
    JsonlStatusEventSink,
)


def main() -> None:
    model = os.environ.get("AGENT_HINTS_EXAMPLE_MODEL", "openai:gpt-4o-mini")
    sink_path = Path(
        os.environ.get("AGENT_HINTS_STATUS_SINK") or (Path(__file__).parent / "status-events.jsonl")
    )

    agent = create_deep_agent(
        model=model,
        tools=[],
        middleware=[
            AgentHintsMiddleware(agent_type="quickstart-agent"),
            AgentStatusEventsMiddleware(sink=JsonlStatusEventSink(sink_path)),
        ],
    )

    result = agent.invoke(
        {"messages": [("user", "In one sentence: what is the X-Agent-Hint header for?")]},
        config={"configurable": {"thread_id": "quickstart-1"}},
    )
    print(result["messages"][-1].content)
    print(f"status events: {sink_path}")


if __name__ == "__main__":
    main()

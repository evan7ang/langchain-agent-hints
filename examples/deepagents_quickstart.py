"""Wire langchain-agent-hints into a deepagents Deep Agent.

Requirements (not declared by the package itself):

    pip install deepagents langchain-openai
    export OPENAI_API_KEY=...   # any tool-calling model works

Run from the repository root:

    python examples/deepagents_quickstart.py

The run writes out-of-band status events to ``examples/status-events.jsonl``
(next to this file) and carries one X-Agent-Hint header per model request.
"""

from __future__ import annotations

from pathlib import Path

from deepagents import create_deep_agent

from langchain_agent_hints import (
    AgentHintsMiddleware,
    AgentStatusEventsMiddleware,
    JsonlStatusEventSink,
)


def main() -> None:
    sink_path = Path(__file__).parent / "status-events.jsonl"

    agent = create_deep_agent(
        model="openai:gpt-4o-mini",
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

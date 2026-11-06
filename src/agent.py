"""Custom multi-turn agent loop.

One user message is handled as follows:

1. Reject a blank message before any model call.
2. Append it to the session history and scratchpad.
3. Ask the model for one decision: a tool call or a final answer.
4. Run the tool, append the result to the trace, the history, and the scratchpad.
5. Repeat until a final answer or until ``max_steps`` decisions have been used.

A step is one model decision. A turn that saves or searches then answers needs
two steps. Tool errors stay in the trace and are sent back to the model; they
do not abort the loop by themselves. Side effects of tools that already ran
are kept if the step budget runs out.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from src.llm import LLMError, build_llm
from src.memory_store import MemoryStore
from src.session import Session, new_session_id
from src.tools import ToolRegistry
from src.trace import ToolTrace, TraceRecord

FINAL = "final"
EMPTY_QUERY = "empty_query"
MAX_STEPS = "max_steps"
LLM_ERROR = "llm_error"


@dataclass
class AgentResult:
    answer: str
    trace: ToolTrace
    stopped_reason: str
    ok: bool
    session_id: str


class Agent:
    def __init__(self, session: Session, tools: ToolRegistry, llm, max_steps: int = 6):
        self.session = session
        self.tools = tools
        self.llm = llm
        self.max_steps = max_steps

    def run(self, user_text: str, max_steps: int | None = None) -> AgentResult:
        text = user_text if isinstance(user_text, str) else ""
        if not text.strip():
            return AgentResult(
                answer="Empty query: send a non-empty message.",
                trace=ToolTrace(),
                stopped_reason=EMPTY_QUERY,
                ok=False,
                session_id=self.session.session_id,
            )

        budget = self.max_steps if max_steps is None else max_steps
        trace = ToolTrace()
        self.session.add_user(text)
        if budget < 1:
            return self._stop(max_steps_message(0), trace, MAX_STEPS)

        for step in range(1, budget + 1):
            try:
                decision = self.llm.complete(self.session.model_messages(), self.tools.specs())
            except LLMError as exc:
                return self._stop(f"Model call failed: {exc}", trace, LLM_ERROR)

            if decision.action == "final":
                content = decision.content.strip() or "(empty model response)"
                return self._stop(content, trace, FINAL, ok=True)
            if decision.action != "tool":
                return self._stop(
                    "The model returned an unknown action.",
                    trace,
                    "invalid_model_action",
                )

            name = decision.tool or ""
            args = decision.args
            started = time.perf_counter()
            result = self.tools.execute(name, args, session=self.session)
            duration_ms = (time.perf_counter() - started) * 1000
            trace.add(
                TraceRecord(
                    step=step,
                    tool=name or "(empty)",
                    args=args if isinstance(args, dict) else {},
                    ok=bool(result.get("ok")),
                    result=result,
                    error=None if result.get("ok") else str(result.get("error") or "error"),
                    duration_ms=duration_ms,
                )
            )
            self.session.add_tool(name or "(empty)", result)
            detail = result.get("message") or result.get("id") or result.get("error") or ""
            self.session.note(f"{name or 'tool'}: {detail}")

        return self._stop(max_steps_message(budget), trace, MAX_STEPS)

    def _stop(self, answer: str, trace: ToolTrace, reason: str, ok: bool = False) -> AgentResult:
        self.session.add_assistant(answer)
        return AgentResult(
            answer=answer,
            trace=trace,
            stopped_reason=reason,
            ok=ok,
            session_id=self.session.session_id,
        )


def max_steps_message(count: int) -> str:
    unit = "step" if count == 1 else "steps"
    return f"Stopped after {count} {unit} without a final answer."


def build_agent(
    *,
    memory_dir: Path | str,
    session_id: str | None = None,
    max_steps: int = 6,
    llm=None,
) -> Agent:
    session = Session(session_id=session_id or new_session_id())
    tools = ToolRegistry(MemoryStore(memory_dir))
    return Agent(
        session=session,
        tools=tools,
        llm=llm if llm is not None else build_llm(),
        max_steps=max_steps,
    )

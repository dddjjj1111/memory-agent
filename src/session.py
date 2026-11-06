"""In-process short-term memory for one chat session.

``messages`` is the turn history (user, assistant, tool). ``scratchpad`` is a
short list of working notes. Neither is written to disk unless a tool saves a
long-term memory.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field


def new_session_id() -> str:
    return "sess_" + uuid.uuid4().hex[:8]


@dataclass
class Session:
    session_id: str
    messages: list[dict] = field(default_factory=list)
    scratchpad: list[str] = field(default_factory=list)
    max_notes: int = 40

    def note(self, text: str) -> None:
        clean = " ".join(text.split())
        if not clean:
            return
        if len(clean) > 160:
            clean = clean[:157] + "..."
        self.scratchpad.append(clean)
        if len(self.scratchpad) > self.max_notes:
            del self.scratchpad[: len(self.scratchpad) - self.max_notes]

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self.note(f"user: {text}")

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def add_tool(self, name: str, result: dict) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool": name,
                "content": json.dumps(result, ensure_ascii=False, sort_keys=True),
            }
        )

    def model_messages(self) -> list[dict]:
        """History plus a view of the scratchpad. Does not mutate the session."""

        if not self.scratchpad:
            return list(self.messages)
        pad = "Short-term scratchpad (this session only):\n" + "\n".join(
            f"- {line}" for line in self.scratchpad
        )
        return [*self.messages, {"role": "system", "content": pad}]

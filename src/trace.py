"""Tool-call trace for one user turn."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


def _compact(value: object, limit: int = 160) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


@dataclass
class TraceRecord:
    step: int
    tool: str
    args: dict
    ok: bool
    result: dict
    error: str | None
    duration_ms: float


@dataclass
class ToolTrace:
    """Ordered tool calls made while answering one user message."""

    records: list[TraceRecord] = field(default_factory=list)

    def add(self, record: TraceRecord) -> None:
        self.records.append(record)

    def to_dict(self) -> list[dict]:
        return [asdict(record) for record in self.records]

    def render(self) -> str:
        if not self.records:
            return "[trace] (no tool calls)"
        lines = []
        for record in self.records:
            status = "ok" if record.ok else f"error={record.error}"
            lines.append(
                f"[trace] step={record.step} tool={record.tool} {status} "
                f"duration_ms={record.duration_ms:.1f} "
                f"args={_compact(record.args, 140)} {_brief_result(record.result)}"
            )
        return "\n".join(lines)


def _brief_result(result: object) -> str:
    """Short trace fields. The full payload remains on ``TraceRecord.result``."""

    if not isinstance(result, dict):
        return f"result={_compact(result, 80)}"
    if not result.get("ok"):
        message = result.get("message") or result.get("error") or ""
        return f"message={message}"
    bits: list[str] = []
    if result.get("id"):
        bits.append(f"id={result['id']}")
    if "hit_count" in result:
        bits.append(f"hits={result['hit_count']}")
    elif "count" in result:
        bits.append(f"count={result['count']}")
    if result.get("store_empty"):
        bits.append("store_empty=true")
    message = result.get("message")
    if message:
        bits.append(f"message={message}")
    return " ".join(bits) if bits else f"result={_compact(result, 80)}"

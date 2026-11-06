"""Tools the agent is allowed to call.

The registry checks argument shape before it touches the store. A bad call
returns ``invalid_tool_args`` (or ``missing_memory_id`` / ``write_failure``)
and does not raise into the loop.
"""

from __future__ import annotations

from src.memory_store import MAX_CONTENT_CHARS, MEMORY_ID_RE, MemoryStore
from src.results import fail, succeed
from src.session import Session

TOOL_SPECS: list[dict] = [
    {
        "name": "memory_save",
        "description": "Write one fact to long-term JSON memory. Content must be a non-empty string.",
        "parameters": {
            "content": "string, required",
            "tags": "list of strings, optional",
        },
    },
    {
        "name": "memory_search",
        "description": (
            "Substring search over memory content, tags, and ids. "
            "If memory_id is set, fetch that id instead; a missing id is an error."
        ),
        "parameters": {
            "query": "string, required unless memory_id is set",
            "memory_id": "string, optional, form mem_xxxxxxxx",
        },
    },
    {
        "name": "memory_list",
        "description": "List every long-term memory note. Takes no arguments.",
        "parameters": {},
    },
    {
        "name": "summarize_session",
        "description": (
            "Compress recent user and assistant turns, plus the scratchpad, "
            "into one extractive note. Saves it unless save is false."
        ),
        "parameters": {"save": "boolean, optional, default true"},
    },
]


class ToolRegistry:
    def __init__(self, store: MemoryStore):
        self.store = store

    def specs(self) -> list[dict]:
        return TOOL_SPECS

    def execute(self, name: str, args: dict | None, session: Session | None = None) -> dict:
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return fail("invalid_tool_args", "tool arguments must be a JSON object")
        if name == "memory_save":
            parsed = _validate_save(args)
            if not parsed["ok"]:
                return parsed
            return self.store.save(
                parsed["content"],
                tags=parsed["tags"],
                session_id=session.session_id if session else None,
            )
        if name == "memory_search":
            parsed = _validate_search(args)
            if not parsed["ok"]:
                return parsed
            if "memory_id" in parsed:
                found = self.store.get(parsed["memory_id"])
                if not found["ok"]:
                    return found
                item = found["item"]
                return succeed(
                    query=None,
                    memory_id=item["id"],
                    hits=[item],
                    hit_count=1,
                    store_empty=False,
                    message=f"found {item['id']}",
                )
            return self.store.search(parsed["query"])
        if name == "memory_list":
            if args:
                return fail(
                    "invalid_tool_args",
                    "memory_list does not take arguments",
                )
            return self.store.list_memories()
        if name == "summarize_session":
            parsed = _validate_summarize(args)
            if not parsed["ok"]:
                return parsed
            return _summarize(self.store, session, save=parsed["save"])
        return fail("invalid_tool_args", f"Unknown tool: {name or '(empty)'}")


def _validate_save(args: dict) -> dict:
    unknown = sorted(set(args) - {"content", "tags"})
    if unknown:
        return fail("invalid_tool_args", f"Unexpected arguments: {', '.join(unknown)}")
    content = args.get("content")
    if not isinstance(content, str) or not content.strip():
        return fail("invalid_tool_args", "memory_save requires non-empty string content")
    if len(content.strip()) > MAX_CONTENT_CHARS:
        return fail("invalid_tool_args", f"content exceeds {MAX_CONTENT_CHARS} characters")
    tags = args.get("tags", [])
    if tags is None:
        tags = []
    if not isinstance(tags, list) or not all(isinstance(tag, str) and tag.strip() for tag in tags):
        return fail("invalid_tool_args", "tags must be a list of non-empty strings")
    if len(tags) > 8:
        return fail("invalid_tool_args", "at most 8 tags are allowed")
    clean: list[str] = []
    for tag in tags:
        item = tag.strip()
        if len(item) > 32:
            return fail("invalid_tool_args", "each tag must be at most 32 characters")
        if item not in clean:
            clean.append(item)
    return succeed(content=content.strip(), tags=clean)


def _validate_search(args: dict) -> dict:
    unknown = sorted(set(args) - {"query", "memory_id"})
    if unknown:
        return fail("invalid_tool_args", f"Unexpected arguments: {', '.join(unknown)}")
    memory_id = args.get("memory_id")
    if memory_id is not None:
        if not isinstance(memory_id, str) or not memory_id.strip():
            return fail("invalid_tool_args", "memory_id must be a non-empty string")
        if not MEMORY_ID_RE.fullmatch(memory_id):
            return fail("invalid_tool_args", "memory_id must look like mem_xxxxxxxx")
        return succeed(memory_id=memory_id)
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return fail(
            "invalid_tool_args",
            "memory_search requires a non-empty query or a memory_id",
        )
    return succeed(query=query.strip())


def _validate_summarize(args: dict) -> dict:
    unknown = sorted(set(args) - {"save"})
    if unknown:
        return fail("invalid_tool_args", f"Unexpected arguments: {', '.join(unknown)}")
    save = args.get("save", True)
    if not isinstance(save, bool):
        return fail("invalid_tool_args", "save must be a boolean")
    return succeed(save=save)


def _clip(text: str, limit: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def _summarize(store: MemoryStore, session: Session | None, save: bool) -> dict:
    if session is None:
        return fail("invalid_tool_args", "summarize_session requires an active session")
    conversational = [
        message for message in session.messages if message.get("role") in {"user", "assistant"}
    ]
    user_lines = [
        _clip(str(message.get("content", "")), 140)
        for message in conversational
        if message.get("role") == "user" and str(message.get("content", "")).strip()
    ][-6:]
    last_reply = ""
    for message in reversed(conversational):
        if message.get("role") == "assistant" and str(message.get("content", "")).strip():
            last_reply = _clip(str(message["content"]), 160)
            break
    tool_notes = [note for note in session.scratchpad if not note.startswith("user:")][-4:]
    parts: list[str] = []
    if user_lines:
        parts.append("Users: " + " || ".join(user_lines))
    if last_reply:
        parts.append(f"Last reply: {last_reply}")
    if tool_notes:
        parts.append("Scratchpad: " + "; ".join(tool_notes))
    if not parts:
        return fail("empty_memory", "Nothing in this session to summarize.")
    summary = "Session summary (extractive): " + " | ".join(parts)
    if not save:
        return succeed(summary=summary, saved=False, message="summary not saved")
    stored = store.save(summary, tags=["summary"], session_id=session.session_id)
    if not stored["ok"]:
        stored["summary"] = summary
        stored["saved"] = False
        return stored
    return succeed(
        summary=summary,
        saved=True,
        id=stored["id"],
        content=summary,
        file=stored["file"],
        message=f"saved summary {stored['id']}",
    )

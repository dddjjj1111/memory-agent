"""File-backed long-term memory.

Each note is one JSON file under the memory directory. Search is a
case-insensitive substring over content, tags, and id. There is no ranking
and no embedding index.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.results import fail, succeed

MEMORY_ID_RE = re.compile(r"mem_[0-9a-f]{8}")
MAX_CONTENT_CHARS = 4000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_memory(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    memory_id = data.get("id")
    content = data.get("content")
    if memory_id != path.stem or not isinstance(content, str):
        return None
    raw_tags = data.get("tags")
    tags = [tag for tag in raw_tags if isinstance(tag, str)] if isinstance(raw_tags, list) else []
    created_at = data.get("created_at")
    session_id = data.get("session_id")
    return {
        "id": memory_id,
        "content": content,
        "tags": tags,
        "created_at": created_at if isinstance(created_at, str) else "",
        "session_id": session_id if isinstance(session_id, str) else "",
    }


class MemoryStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def save(
        self,
        content: str,
        tags: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict:
        if not isinstance(content, str) or not content.strip():
            return fail("invalid_tool_args", "memory_save requires non-empty string content")
        text = content.strip()
        if len(text) > MAX_CONTENT_CHARS:
            return fail(
                "invalid_tool_args",
                f"content exceeds {MAX_CONTENT_CHARS} characters",
            )
        clean_tags = []
        for tag in tags or []:
            if isinstance(tag, str) and tag.strip() and tag.strip() not in clean_tags:
                clean_tags.append(tag.strip())
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            memory_id = self._allocate_id()
            record = {
                "id": memory_id,
                "content": text,
                "tags": clean_tags,
                "session_id": session_id or "",
                "created_at": _now(),
            }
            path = self.root / f"{memory_id}.json"
            _atomic_write(path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        except OSError as exc:
            return fail("write_failure", f"Could not write memory: {exc.strerror or exc}")
        return succeed(
            id=record["id"],
            content=record["content"],
            tags=record["tags"],
            created_at=record["created_at"],
            session_id=record["session_id"],
            file=path.name,
            message=f"saved {record['id']}",
        )

    def get(self, memory_id: str) -> dict:
        if not isinstance(memory_id, str) or not MEMORY_ID_RE.fullmatch(memory_id):
            return fail("invalid_tool_args", "memory_id must look like mem_xxxxxxxx")
        path = self.root / f"{memory_id}.json"
        try:
            exists = path.is_file()
        except OSError as exc:
            return fail("write_failure", f"Could not read memory: {exc.strerror or exc}")
        if not exists:
            return fail(
                "missing_memory_id",
                f"No memory with id {memory_id}.",
                memory_id=memory_id,
            )
        item = _read_memory(path)
        if item is None:
            return fail(
                "missing_memory_id",
                f"No readable memory with id {memory_id}.",
                memory_id=memory_id,
            )
        return succeed(item=item, message=f"found {memory_id}")

    def search(self, query: str) -> dict:
        items = self._load_all()
        needle = query.casefold()
        if not items:
            return succeed(
                query=query,
                hits=[],
                hit_count=0,
                store_empty=True,
                message="empty memory",
            )
        hits = [item for item in items if needle in _haystack(item)]
        return succeed(
            query=query,
            hits=hits,
            hit_count=len(hits),
            store_empty=False,
            message="no match" if not hits else f"{len(hits)} match(es)",
        )

    def list_memories(self) -> dict:
        items = self._load_all()
        if not items:
            return succeed(items=[], count=0, store_empty=True, message="empty memory")
        return succeed(
            items=items,
            count=len(items),
            store_empty=False,
            message=f"{len(items)} memory item(s)",
        )

    def _load_all(self) -> list[dict]:
        try:
            if not self.root.is_dir():
                return []
            paths = sorted(self.root.glob("mem_*.json"))
        except OSError:
            return []
        items = []
        for path in paths:
            if not MEMORY_ID_RE.fullmatch(path.stem):
                continue
            item = _read_memory(path)
            if item is not None:
                items.append(item)
        items.sort(key=lambda item: (item.get("created_at") or "", item["id"]))
        return items

    def _allocate_id(self) -> str:
        for _ in range(8):
            memory_id = "mem_" + uuid.uuid4().hex[:8]
            if not (self.root / f"{memory_id}.json").exists():
                return memory_id
        raise OSError("could not allocate a memory id")


def _haystack(item: dict) -> str:
    tags = " ".join(item.get("tags") or [])
    return f"{item.get('content', '')} {tags} {item.get('id', '')}".casefold()


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise

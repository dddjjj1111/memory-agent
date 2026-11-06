"""Model adapter.

``USE_MOCK_LLM=1`` (the default) selects a keyword heuristic. It is not a
language model. It looks at the latest user message and any tool results
already produced for that message, then either calls one tool or answers.

Set ``USE_MOCK_LLM=0`` and ``OPENAI_API_KEY`` to send the same JSON action
protocol to an OpenAI-compatible chat endpoint. Tests and the README demo do
not use that path.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from src.envutil import use_mock_llm

SYSTEM_PROMPT = """You are memory-agent. You keep a short-term session scratchpad and a long-term file memory.
Reply with one JSON object and no markdown.
Call a tool with {"action":"tool","tool":"<name>","args":{...}}.
Answer the user with {"action":"final","content":"..."}.
Tools:
- memory_save: args {"content": string, "tags": optional string list}. Content must be non-empty.
- memory_search: args {"query": string} or {"memory_id": "mem_xxxxxxxx"}. Search is case-insensitive substring, not semantic.
- memory_list: args {}.
- summarize_session: args {"save": optional boolean}. Compresses this session into a memory note.
Use a memory tool when the user asks to remember, recall, list, or summarize.
After a tool result is present, return a final answer unless another call is required.
If the store has no match, say so. Do not invent memories.
"""


class LLMError(RuntimeError):
    """The remote model call failed. The mock heuristic does not raise this."""


@dataclass
class Decision:
    action: str
    content: str = ""
    tool: str | None = None
    args: dict | None = None
    raw: str = ""


class MockLLM:
    """Keyword router used for offline demos and pytest."""

    def complete(self, messages: list[dict], tools: list[dict]) -> Decision:
        del tools
        return route_mock(latest_user_text(messages), tool_observations_since_last_user(messages))


class OpenAICompatibleLLM:
    """Minimal chat client. The endpoint must return the JSON action protocol."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 30.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, messages: list[dict], tools: list[dict]) -> Decision:
        del tools
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": _to_chat_messages(messages),
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            raw = body["choices"][0]["message"]["content"]
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise LLMError(str(exc)) from exc
        if not isinstance(raw, str):
            raise LLMError("model content was not a string")
        return parse_decision(raw)


def build_llm() -> MockLLM | OpenAICompatibleLLM:
    if use_mock_llm():
        return MockLLM()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "USE_MOCK_LLM=0 but OPENAI_API_KEY is empty. "
            "Set the key or leave USE_MOCK_LLM=1 for the offline heuristic."
        )
    return OpenAICompatibleLLM(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    )


def parse_decision(raw: str) -> Decision:
    """Parse a model message. Non-JSON text is treated as a final answer."""

    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return Decision(action="final", content=raw.strip() or "(empty model response)", raw=raw)
    if not isinstance(obj, dict):
        return Decision(action="final", content=raw.strip(), raw=raw)
    action = obj.get("action")
    if action == "tool" and isinstance(obj.get("tool"), str):
        args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
        return Decision(action="tool", tool=obj["tool"], args=args, raw=raw)
    if action == "final" and isinstance(obj.get("content"), str) and obj["content"].strip():
        return Decision(action="final", content=obj["content"].strip(), raw=raw)
    if action == "final":
        return Decision(action="final", content="(empty model response)", raw=raw)
    return Decision(action="final", content=raw.strip() or "(empty model response)", raw=raw)


def latest_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def tool_observations_since_last_user(messages: list[dict]) -> list[dict]:
    last_user = -1
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user = index
    observations = []
    for message in messages[last_user + 1 :]:
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message.get("content") or "{}")
        except json.JSONDecodeError:
            payload = {
                "ok": False,
                "error": "invalid_tool_result",
                "message": "Tool result was not valid JSON.",
            }
        if not isinstance(payload, dict):
            payload = {
                "ok": False,
                "error": "invalid_tool_result",
                "message": "Tool result was not a JSON object.",
            }
        observations.append({"tool": str(message.get("tool") or ""), "payload": payload})
    return observations


def route_mock(user_text: str, observations: list[dict]) -> Decision:
    """Pick the next mock action.

    Match order: forced tool loop, then a final answer if a tool already ran,
    then empty-save, summarize, list, recall, remember, otherwise a direct reply.
    The forced loop ignores tool results so ``max_steps`` can be demonstrated.
    """

    text = user_text.strip()
    if _force_tool_loop(text):
        return _tool("memory_list", {})
    if observations:
        return _final_from_observation(observations[-1])
    if _is_empty_save(text):
        return _tool("memory_save", {"content": ""})
    if _is_summarize(text):
        return _tool("summarize_session", {})
    if _is_list_request(text):
        return _tool("memory_list", {})
    query = _recall_query(text)
    if query is not None:
        return _tool("memory_search", {"query": query})
    fact = _remember_fact(text)
    if fact is not None:
        content, tags = _shape_memory(text, fact)
        return _tool("memory_save", {"content": content, "tags": tags})
    return Decision(action="final", content=_direct_answer(text))


def _tool(name: str, args: dict) -> Decision:
    return Decision(action="tool", tool=name, args=args)


def _final_from_observation(observation: dict) -> Decision:
    tool = observation.get("tool") or "tool"
    payload = observation["payload"]
    if not payload.get("ok"):
        message = str(payload.get("message") or payload.get("error") or "tool failed")
        error = payload.get("error")
        if error == "invalid_tool_args":
            return Decision(action="final", content=f"I could not complete that. {message}")
        if error == "write_failure":
            return Decision(action="final", content=f"Long-term memory write failed. {message}")
        if error == "missing_memory_id":
            return Decision(action="final", content=message)
        if error == "empty_memory":
            return Decision(action="final", content=message)
        return Decision(action="final", content=f"Tool {tool} failed. {message}")

    if tool == "memory_save":
        return Decision(
            action="final",
            content=f"Saved to long-term memory ({payload.get('id')}): {payload.get('content')}",
        )
    if tool == "memory_search":
        hits = payload.get("hits") or []
        if not hits:
            if payload.get("store_empty"):
                return Decision(
                    action="final",
                    content="No matching long-term memory. The store is empty.",
                )
            query = payload.get("query") or ""
            return Decision(
                action="final",
                content=f'No matching long-term memory for "{query}".',
            )
        joined = " ".join(str(hit.get("content", "")) for hit in hits)
        return Decision(action="final", content=f"From long-term memory: {joined}")
    if tool == "memory_list":
        items = payload.get("items") or []
        if not items:
            return Decision(action="final", content="Long-term memory is empty.")
        lines = [f"{item.get('id')}: {item.get('content')}" for item in items]
        return Decision(action="final", content="Long-term memories:\n" + "\n".join(lines))
    if tool == "summarize_session":
        saved_id = payload.get("id") or "not-saved"
        return Decision(
            action="final",
            content=f"Session summary saved ({saved_id}): {payload.get('summary', '')}",
        )
    return Decision(action="final", content=json.dumps(payload, ensure_ascii=False))


_EMPTY_SAVE = re.compile(r"(?i)^(?:please\s+)?(?:remember(?:\s+that)?|save empty)[.!?]*$")
_REMEMBER_LINE = re.compile(r"(?i)^(?:please\s+)?remember(?:\s+that)?\s+(.+)$")
_ZH_REMEMBER_LINE = re.compile(r"^请?记住\s*(.+)$")
_NAME_PATTERNS = (
    re.compile(r"(?i)\bmy name is\s+([A-Za-z][\w'-]{0,40})"),
    re.compile(r"我叫\s*([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9'_-]{0,20})"),
    re.compile(r"我的名字是\s*([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9'_-]{0,20})"),
)
_BAD_NAMES = {"什么", "啥", "谁", "what", "who", "which"}
_ABOUT = re.compile(r"(?i)^what do you remember about\s+(.+)$")
_ZH_ABOUT = re.compile(r"^你记得\s*(.+?)\s*吗$")


def _force_tool_loop(text: str) -> bool:
    lowered = text.casefold()
    return "keep listing" in lowered or "一直列出" in text


def _is_empty_save(text: str) -> bool:
    stripped = text.strip()
    if stripped in {"记住", "请记住"}:
        return True
    return _EMPTY_SAVE.fullmatch(stripped) is not None


def _is_summarize(text: str) -> bool:
    stripped = text.strip().rstrip("？?.!。")
    lowered = stripped.casefold()
    if lowered.startswith("please summarize") or lowered.startswith("please summarise"):
        return True
    if lowered.startswith("summarize") or lowered.startswith("summarise"):
        return True
    return stripped.startswith("总结")


def _is_list_request(text: str) -> bool:
    lowered = text.casefold().strip().rstrip("?.!")
    if lowered in {
        "list memories",
        "list memory",
        "show memories",
        "what do you remember",
    }:
        return True
    return text.strip().rstrip("？?.。") in {"列出记忆", "有哪些记忆", "列出长期记忆"}


def _recall_query(text: str) -> str | None:
    if _is_empty_save(text) or _remember_fact(text) is not None:
        return None
    stripped = text.strip().rstrip("？?.!。")
    lowered = stripped.casefold()
    if lowered in {"what is my name", "what's my name", "whats my name", "who am i"}:
        return "name"
    if stripped in {"我叫什么", "我叫啥", "我的名字是什么", "我的名字", "我是谁", "你知道我叫什么"}:
        return "name"
    if re.fullmatch(r"我叫(?:什么|啥)(?:名字)?", stripped):
        return "name"
    if ("我的名字" in stripped or "我是谁" in stripped or "我叫什么" in stripped) and (
        "吗" in stripped or "什么" in stripped or "谁" in stripped
    ):
        return "name"
    about = _ABOUT.match(stripped)
    if about:
        topic = _clean_fact(about.group(1))
        return topic or None
    zh_about = _ZH_ABOUT.match(stripped)
    if zh_about:
        topic = _clean_fact(zh_about.group(1))
        return topic or None
    return None


def _remember_fact(text: str) -> str | None:
    stripped = text.strip()
    if _is_empty_save(stripped):
        return None
    match = _REMEMBER_LINE.match(stripped) or _ZH_REMEMBER_LINE.match(stripped)
    if match:
        fact = _clean_fact(match.group(1))
        if fact:
            return fact
    name = _extract_name(stripped)
    if name and not _looks_like_name_question(stripped):
        return f"The user's name is {name}."
    return None


def _looks_like_name_question(text: str) -> bool:
    stripped = text.strip().rstrip("？?.!。")
    lowered = stripped.casefold()
    if lowered in {"what is my name", "what's my name", "whats my name", "who am i"}:
        return True
    return "什么" in stripped or "谁" in stripped or lowered.startswith("what ") or lowered.startswith("who ")


def _extract_name(text: str) -> str | None:
    for pattern in _NAME_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        name = match.group(1).strip(" .,!?:;，。！？")
        if not name or name.casefold() in _BAD_NAMES:
            continue
        return name
    return None


def _shape_memory(text: str, fact: str) -> tuple[str, list[str]]:
    name = _extract_name(text)
    if not name:
        return fact, []
    content = fact
    if "name" not in content.casefold() and "名字" not in content:
        content = f"The user's name is {name}. {content}".strip()
    return content, ["name", "名字"]


def _clean_fact(raw: str) -> str:
    fact = raw.strip().strip("\"'`“”‘’")
    fact = re.sub(r"[\s.。!！?？]+$", "", fact)
    return fact.strip()


def _direct_answer(text: str) -> str:
    stripped = text.strip().rstrip("!！.。")
    lowered = stripped.casefold()
    if lowered in {"hello", "hi", "hey"} or stripped in {"你好", "您好"}:
        return (
            "Hello. I can chat in this session and save facts to long-term memory "
            "when you ask me to remember them."
        )
    compact = lowered.replace(" ", "")
    if compact in {"2+2", "2+2=?"} or lowered in {"what is 2+2", "what's 2+2"}:
        return "4"
    return (
        "I have no saved memory for that, and this turn does not ask me to store one. "
        "Ask me to remember a fact, or ask what I remember."
    )


def _to_chat_messages(messages: list[dict]) -> list[dict]:
    chat = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "tool":
            chat.append(
                {
                    "role": "user",
                    "content": f"Tool result ({message.get('tool')}): {content}",
                }
            )
        elif role == "assistant":
            chat.append({"role": "assistant", "content": content})
        else:
            chat.append({"role": "user", "content": content})
    return chat

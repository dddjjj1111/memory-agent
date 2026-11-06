"""Offline tests for the mock agent, the memory tools, and the CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.agent import build_agent
from src.llm import MockLLM, build_llm, parse_decision
from src.memory_store import MemoryStore
from src.session import Session
from src.tools import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _force_mock_llm(monkeypatch):
    monkeypatch.setenv("USE_MOCK_LLM", "1")


def _agent(tmp_path, max_steps: int = 6):
    return build_agent(memory_dir=tmp_path, max_steps=max_steps)


def _run_cli(args, tmp_path):
    env = os.environ.copy()
    env["USE_MOCK_LLM"] = "1"
    env["MEMORY_DIR"] = str(tmp_path)
    return subprocess.run(
        [sys.executable, "-m", "src.main", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("save_turn", "recall_turn", "needle"),
    [
        ("Remember that my name is Ada", "What is my name?", "Ada"),
        ("my name is Ada", "What's my name?", "Ada"),
        ("我叫小明", "我叫什么", "小明"),
        ("记住我喜欢黑咖啡", "你记得黑咖啡吗", "黑咖啡"),
    ],
)
def test_save_and_recall_across_turns(tmp_path, save_turn, recall_turn, needle):
    agent = _agent(tmp_path)
    saved = agent.run(save_turn)
    recalled = agent.run(recall_turn)

    assert saved.ok and saved.stopped_reason == "final"
    assert saved.trace.records[0].tool == "memory_save"
    assert saved.trace.records[0].ok
    assert needle in saved.answer

    assert recalled.ok and recalled.stopped_reason == "final"
    assert recalled.trace.records[0].tool == "memory_search"
    assert recalled.trace.records[0].result["hit_count"] == 1
    assert needle in recalled.answer
    assert sum(message["role"] == "user" for message in agent.session.messages) == 2
    assert agent.session.scratchpad


def test_save_writes_tagged_json(tmp_path):
    agent = _agent(tmp_path)
    result = agent.run("Remember that my name is Ada")
    files = list(tmp_path.glob("mem_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["id"] == files[0].stem
    assert payload["content"] == "my name is Ada"
    assert payload["tags"] == ["name", "名字"]
    assert payload["session_id"] == agent.session.session_id
    assert result.trace.records[0].result["id"] == payload["id"]

    listed = agent.tools.execute("memory_list", {})
    assert listed["count"] == 1
    assert listed["items"][0]["id"] == payload["id"]


def test_long_term_memory_survives_a_new_session(tmp_path):
    first = _agent(tmp_path)
    first.run("Remember that my name is Ada")
    second = _agent(tmp_path)
    recalled = second.run("What is my name?")

    assert second.session.session_id != first.session.session_id
    assert "Ada" in recalled.answer
    assert not any("Remember" in note for note in second.session.scratchpad)


def test_scratchpad_is_not_written_to_disk(tmp_path):
    agent = _agent(tmp_path)
    result = agent.run("Hello")

    assert result.ok
    assert result.stopped_reason == "final"
    assert result.trace.records == []
    assert result.answer.startswith("Hello.")
    assert agent.session.scratchpad
    assert list(tmp_path.iterdir()) == []


def test_search_miss(tmp_path):
    agent = _agent(tmp_path)
    agent.run("Remember that my name is Ada")
    result = agent.run("What do you remember about quantum zucchini?")

    assert result.ok
    record = result.trace.records[0]
    assert record.tool == "memory_search"
    assert record.ok
    assert record.result["store_empty"] is False
    assert record.result["hit_count"] == 0
    assert 'No matching long-term memory for "quantum zucchini".' == result.answer
    assert "Ada" not in result.answer


def test_search_on_empty_store(tmp_path):
    result = _agent(tmp_path).run("What do you remember about quantum zucchini?")

    assert result.ok
    assert result.trace.records[0].result["store_empty"] is True
    assert result.answer == "No matching long-term memory. The store is empty."


def test_search_is_case_insensitive_substring(tmp_path):
    agent = _agent(tmp_path)
    agent.run("Remember that I like Espresso")
    hit = agent.run("What do you remember about espresso?")
    miss = agent.run("What do you remember about coffee?")

    assert "Espresso" in hit.answer
    assert miss.answer == 'No matching long-term memory for "coffee".'


def test_invalid_tool_args_through_agent(tmp_path):
    result = _agent(tmp_path).run("Please remember")

    assert result.ok
    assert result.stopped_reason == "final"
    record = result.trace.records[0]
    assert record.tool == "memory_save"
    assert record.ok is False
    assert record.error == "invalid_tool_args"
    assert "non-empty" in result.answer
    assert list(tmp_path.glob("*.json")) == []


def test_tool_schema_rejects_bad_args(tmp_path):
    registry = ToolRegistry(MemoryStore(tmp_path))
    session = Session(session_id="sess_test")

    empty_query = registry.execute("memory_search", {"query": "  "}, session=session)
    bad_tags = registry.execute("memory_save", {"content": "a fact", "tags": "name"}, session=session)
    unknown = registry.execute("memory_delete", {}, session=session)
    not_object = registry.execute("memory_list", ["nope"], session=session)
    extra = registry.execute("memory_list", {"limit": 1}, session=session)

    assert empty_query["error"] == "invalid_tool_args"
    assert bad_tags["error"] == "invalid_tool_args"
    assert unknown["error"] == "invalid_tool_args"
    assert not_object["error"] == "invalid_tool_args"
    assert extra["error"] == "invalid_tool_args"
    assert list(tmp_path.glob("*.json")) == []


def test_missing_memory_id(tmp_path):
    registry = ToolRegistry(MemoryStore(tmp_path))
    missing = registry.execute("memory_search", {"memory_id": "mem_deadbeef"})
    malformed = registry.execute("memory_search", {"memory_id": "../secret"})

    assert missing["ok"] is False
    assert missing["error"] == "missing_memory_id"
    assert malformed["ok"] is False
    assert malformed["error"] == "invalid_tool_args"


def test_corrupt_memory_file_is_skipped(tmp_path):
    (tmp_path / "mem_abcd1234.json").write_text("{not json", encoding="utf-8")
    store = MemoryStore(tmp_path)

    listed = store.list_memories()
    fetched = store.get("mem_abcd1234")

    assert listed["ok"] and listed["items"] == []
    assert fetched["error"] == "missing_memory_id"


def test_write_failure(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    result = ToolRegistry(MemoryStore(blocker)).execute(
        "memory_save",
        {"content": "a fact"},
        session=Session(session_id="sess_test"),
    )

    assert result["ok"] is False
    assert result["error"] == "write_failure"


def test_max_steps_keeps_listing(tmp_path):
    result = _agent(tmp_path).run("keep listing", max_steps=2)

    assert result.ok is False
    assert result.stopped_reason == "max_steps"
    assert result.answer == "Stopped after 2 steps without a final answer."
    assert len(result.trace.records) == 2
    assert [record.tool for record in result.trace.records] == ["memory_list", "memory_list"]


def test_max_steps_keeps_a_tool_side_effect(tmp_path):
    result = _agent(tmp_path).run("Remember that my name is Ada", max_steps=1)

    assert result.stopped_reason == "max_steps"
    assert result.trace.records[0].tool == "memory_save"
    assert result.trace.records[0].ok
    assert result.answer == "Stopped after 1 step without a final answer."
    stored = json.loads(next(tmp_path.glob("mem_*.json")).read_text(encoding="utf-8"))
    assert stored["content"] == "my name is Ada"


def test_empty_query(tmp_path):
    agent = _agent(tmp_path)
    empty = agent.run("   ")
    follow = agent.run("Hello")

    assert empty.ok is False
    assert empty.stopped_reason == "empty_query"
    assert empty.trace.records == []
    assert agent.session.messages[0]["content"] == "Hello"
    assert follow.ok


def test_summarize_session(tmp_path):
    agent = _agent(tmp_path)
    agent.run("Remember that my name is Ada")
    result = agent.run("Summarize this session")

    assert result.ok
    assert result.trace.records[0].tool == "summarize_session"
    assert result.trace.records[0].ok
    assert "Session summary" in result.answer
    assert len(list(tmp_path.glob("mem_*.json"))) == 2


def test_summarize_empty_session_is_empty_memory(tmp_path):
    result = ToolRegistry(MemoryStore(tmp_path)).execute(
        "summarize_session",
        {},
        session=Session(session_id="sess_empty"),
    )
    assert result["ok"] is False
    assert result["error"] == "empty_memory"


def test_mock_is_the_default(monkeypatch):
    monkeypatch.delenv("USE_MOCK_LLM", raising=False)
    assert isinstance(build_llm(), MockLLM)


def test_real_llm_requires_a_key(monkeypatch):
    monkeypatch.setenv("USE_MOCK_LLM", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        build_llm()


def test_parse_decision_accepts_json_and_plain_text():
    tool = parse_decision('{"action":"tool","tool":"memory_list","args":{}}')
    final = parse_decision('```json\n{"action":"final","content":"Hi"}\n```')
    plain = parse_decision("Just a sentence.")

    assert tool.action == "tool" and tool.tool == "memory_list" and tool.args == {}
    assert final.action == "final" and final.content == "Hi"
    assert plain.action == "final" and plain.content == "Just a sentence."


def test_cli_demo(tmp_path):
    completed = _run_cli(["--demo", "--memory-dir", str(tmp_path)], tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "Saved to long-term memory" in completed.stdout
    assert "From long-term memory: my name is Ada and I prefer black coffee" in completed.stdout
    assert "Session summary saved" in completed.stdout
    assert "tool=memory_save" in completed.stdout
    assert "tool=memory_search" in completed.stdout
    assert "tool=summarize_session" in completed.stdout


def test_cli_two_turns_and_failures(tmp_path):
    recalled = _run_cli(
        [
            "--memory-dir",
            str(tmp_path),
            "--turn",
            "Remember that my name is Ada",
            "--turn",
            "What is my name?",
        ],
        tmp_path,
    )
    empty = _run_cli(["--memory-dir", str(tmp_path), "--turn", "   "], tmp_path)
    limited = _run_cli(
        ["--memory-dir", str(tmp_path), "--max-steps", "2", "--turn", "keep listing"],
        tmp_path,
    )

    assert recalled.returncode == 0
    assert "From long-term memory: my name is Ada" in recalled.stdout
    assert empty.returncode == 1
    assert "Empty query" in empty.stdout
    assert limited.returncode == 1
    assert "Stopped after 2 steps" in limited.stdout

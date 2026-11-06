# Failure cases

These are the boundaries the loop and the memory tools actually implement. The commands assume the repository root, Python 3.10+, and `USE_MOCK_LLM=1` (the default).

Tool failures and loop failures are different. A rejected tool call is stored on the `ToolTrace` and given back to the model, which can still finish the turn. `empty_query`, `max_steps`, and `llm_error` stop the turn. The CLI exits `1` only for those loop failures.

`ok` on a tool result means that call succeeded. `AgentResult.ok` means the turn ended with a final answer.

## Empty query

Trigger: a user message that is empty or only whitespace.

```bash
python -m src.main --turn "   "
```

Result: no model call, no tool call, nothing appended to the session. Answer: `Empty query: send a non-empty message.` Stop reason: `empty_query`. CLI exit code `1`.

The REPL ignores a blank line instead of treating it as a turn.

`memory_search` with a blank `query` and no `memory_id` is a different case: the tool returns `invalid_tool_args` and does not scan the store.

## Empty memory

Trigger: `memory_search` or `memory_list` when the directory has no readable notes. `summarize_session` on a session with no messages and an empty scratchpad uses the same error name.

```bash
python -m src.main --memory-dir /tmp/memory-agent-empty \
  --turn "What do you remember about quantum zucchini?"
```

Result: search succeeds as a call (`ok: true`) with `hits: []`, `store_empty: true`, and message `empty memory`. The mock answers `No matching long-term memory. The store is empty.` The turn itself succeeds.

A session that has never saved anything can still answer. `Hello` is a final reply with an empty tool trace and no files written.

`summarize_session` with nothing to compress returns `ok: false`, `error: empty_memory`. The mock does not hit this on a normal turn, because the user line is already on the session before the tool runs.

## Search miss

Trigger: the store has notes, and the substring is absent. Matching is case-insensitive over content, tags, and id. It is not semantic: a note that says `Espresso` does not match `coffee`.

```bash
python -m src.main --memory-dir /tmp/memory-agent-miss \
  --turn "Remember that my name is Ada" \
  --turn "What do you remember about quantum zucchini?"
```

Result: `store_empty: false`, `hit_count: 0`, message `no match`. The mock answers `No matching long-term memory for "quantum zucchini".` It does not repeat unrelated notes.

## Missing memory id

Trigger: `memory_search` with `memory_id`. The id must be `mem_` plus 8 lowercase hex digits. A well-formed id that is not on disk, or a file that is not readable JSON, returns `missing_memory_id`. Anything else, including a path (`../secret`), returns `invalid_tool_args` and is not joined onto the memory directory.

The mock heuristic does not emit ids. Call the store or the tool:

```python
from src.memory_store import MemoryStore

print(MemoryStore("data/memory").get("mem_deadbeef"))
```

Result: `{"ok": false, "error": "missing_memory_id", ...}`. No exception.

## Invalid tool arguments

Trigger from the mock: `Please remember`, `remember`, `记住`, or `save empty`. Those phrases call `memory_save` with empty content. The registry rejects the call before it creates a file.

```bash
python -m src.main --turn "Please remember"
```

Result: trace record `tool=memory_save error=invalid_tool_args`. The mock then answers that the call was rejected and quotes `memory_save requires non-empty string content`. The turn still ends with a final answer, so the CLI exits `0`.

The same error is returned for an unknown tool name, arguments that are not an object, unexpected keys, tags that are not a list of strings, a blank search query, and content longer than 4000 characters.

## Write failure

Trigger: the memory path cannot be created or replaced. The usual demo case is pointing the store at a file instead of a directory.

```python
from pathlib import Path
from src.memory_store import MemoryStore
from src.session import Session
from src.tools import ToolRegistry

Path("/tmp/memory-agent-blocked").write_text("x")
print(ToolRegistry(MemoryStore("/tmp/memory-agent-blocked")).execute(
    "memory_save",
    {"content": "a fact"},
    session=Session(session_id="sess_test"),
))
```

Result: `error: write_failure` and a message starting `Could not write memory`. The tool does not raise. A failed `summarize_session` save returns the same error and still includes the summary text it built.

Writes use a temporary file in the memory directory, then `replace`. There is no cross-process lock.

## Max steps

A step is one model decision. Saving or searching and then answering takes two steps. The default budget is `MAX_STEPS` or 6.

The mock treats `keep listing` and `一直列出` as a request to call `memory_list` on every step, including after a tool result. That is how an offline run reaches the budget. A real model hits the same stop if it keeps calling tools.

```bash
python -m src.main --max-steps 2 --turn "keep listing"
```

Result: two `memory_list` trace records, stop reason `max_steps`, answer `Stopped after 2 steps without a final answer.`, CLI exit code `1`.

Tools that already ran are not rolled back. `--max-steps 1 --turn "Remember that my name is Ada"` writes the JSON file and then stops, without the "Saved to long-term memory" reply. The next turn can still search that note.

## Unreadable files

`mem_*.json` files that are not valid JSON, or whose `id` does not match the file name, are skipped by list and search. `memory_search` by that id returns `missing_memory_id`. Other files in the directory are ignored.

## Model call failure

With `USE_MOCK_LLM=0`, a network error, a non-JSON body, or a missing `choices[0].message.content` raises `LLMError`. The loop records stop reason `llm_error` and does not keep calling the endpoint on that turn. The mock path does not raise this.

If `USE_MOCK_LLM=0` and `OPENAI_API_KEY` is empty, `build_llm()` raises `RuntimeError` before a session starts. The CLI prints that message and exits `1`.

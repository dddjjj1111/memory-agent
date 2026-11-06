# memory-agent

A small multi-turn chat agent with two memory layers: a short-term session scratchpad that lives in the process, and long-term notes stored as JSON files. The agent loop decides between a tool call and a final answer. It does not use LangChain.

The offline default is a keyword heuristic (`USE_MOCK_LLM=1`), not a language model. It is enough to demo save and recall, and it is what the tests run.

This is the memory slice next to a retrieval demo (mini-rag-agent) and a tool-routing demo (multi-tool-agent).

## Features

- A custom loop for one session. Each user message is kept, then the model may call tools until it answers or runs out of steps.
- Short-term memory: the message list plus a scratchpad of working notes. Both disappear when the process exits unless a tool saves them.
- Long-term memory: one JSON file per note under `data/memory/`, shared by later sessions that use the same directory.
- Tools: `memory_save`, `memory_search`, `memory_list`, and `summarize_session` (an extractive compression of recent turns, written back as a note).
- A `ToolTrace` for every tool call in the turn: name, arguments, ok or error, result, and duration.
- `USE_MOCK_LLM=1` for offline runs. An optional OpenAI-compatible HTTP client is used only when the mock is turned off and `OPENAI_API_KEY` is set.

## Setup

Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The only dependency is pytest. Copying `.env.example` to `.env` is optional. If `USE_MOCK_LLM` is unset, the CLI still uses the mock.

Scripted turns in one session:

```bash
python -m src.main \
  --turn "Remember that my name is Ada and I prefer black coffee." \
  --turn "What is my name?" \
  --turn "What do you remember about coffee?"
```

Fixed demo (the same kind of dialogue, then a session summary):

```bash
python -m src.main --demo
```

Example output from one run. Memory ids change every run.

```
You: Remember that my name is Ada and I prefer black coffee.
Agent: Saved to long-term memory (mem_e60d9bf7): my name is Ada and I prefer black coffee
[trace] step=1 tool=memory_save ok duration_ms=0.3 args={"content": "my name is Ada and I prefer black coffee", "tags": ["name", "名字"]} id=mem_e60d9bf7 message=saved mem_e60d9bf7

You: What is my name?
Agent: From long-term memory: my name is Ada and I prefer black coffee
[trace] step=1 tool=memory_search ok duration_ms=0.2 args={"query": "name"} hits=1 message=1 match(es)

You: What do you remember about coffee?
Agent: From long-term memory: my name is Ada and I prefer black coffee
[trace] step=1 tool=memory_search ok duration_ms=0.1 args={"query": "coffee"} hits=1 message=1 match(es)

You: Summarize this session.
Agent: Session summary saved (mem_8e255561): Session summary (extractive): Users: Remember that my name is Ada and I prefer black coffee. || What is my name? || What do you remember about coffee? || Summarize this session. | Last reply: From long-term memory: my name is Ada and I prefer black coffee | Scratchpad: memory_save: saved mem_e60d9bf7; memory_search: 1 match(es); memory_search: 1 match(es)
[trace] step=1 tool=summarize_session ok duration_ms=0.2 args={} id=mem_8e255561 message=saved summary mem_8e255561
```

REPL (blank lines are ignored; `/exit`, `/trace`, `/memories`, `/scratch`, `/help`):

```bash
python -m src.main
```

Notes are written to `data/memory/` (or `--memory-dir` / `MEMORY_DIR`). Those JSON files are gitignored.

Tests:

```bash
pytest
```

To call an OpenAI-compatible endpoint instead of the mock, set `USE_MOCK_LLM=0`, `OPENAI_API_KEY`, and optionally `OPENAI_BASE_URL` and `OPENAI_MODEL`. The remote model is expected to return one JSON object: `{"action":"tool","tool":"...","args":{...}}` or `{"action":"final","content":"..."}`. Plain text is treated as a final answer. The tests do not hit the network.

## Layout

```
src/agent.py          loop, step budget, session wiring
src/session.py        in-process history and scratchpad
src/memory_store.py   JSON files, substring search
src/tools.py          memory_save, memory_search, memory_list, summarize_session
src/llm.py            mock heuristic and optional HTTP model
src/trace.py          ToolTrace
src/main.py           --turn, --demo, and the REPL
tests/                pytest, mock only
notes/failure_cases.md
data/memory/          runtime notes (not committed)
```

## Notes

Search is a case-insensitive substring over the note body, tags, and id. There is no embedding index and no ranker. A note that says "Espresso" does not match the query "coffee".

`summarize_session` keeps up to six user lines, the last assistant reply, and recent tool notes from the scratchpad. Each piece is truncated. It does not paraphrase.

The mock maps a few phrases and otherwise answers without tools. Match order is: forced tool loop, reply from a tool result, empty save, summarize, list, recall, remember, then a direct reply.

| User text | Mock action |
| --- | --- |
| `remember that ...`, `记住...`, `my name is ...`, `我叫...` (not a question) | `memory_save` |
| `Please remember`, `remember`, `记住`, `save empty` | `memory_save` with empty content, which the tool rejects |
| `What is my name?`, `我叫什么` | `memory_search` with query `name` |
| `What do you remember about TOPIC?`, `你记得TOPIC吗` | `memory_search` with that topic |
| `list memories`, `列出记忆` | `memory_list` |
| `Summarize this session`, `总结...` | `summarize_session` |
| `keep listing`, `一直列出` | `memory_list` on every step, so the step limit can be shown |
| `Hello` | direct reply, no tools |

Name statements are stored with tags `name` and `名字`, so both the English and Chinese name questions search for `name`.

A step is one model decision. The default budget is 6 (`MAX_STEPS` or `--max-steps`). A save or search that then answers the user needs two steps. If the budget runs out, tools that already ran are kept; see `notes/failure_cases.md` for empty input, an empty store, a search miss, a missing id, bad arguments, write errors, and the step limit.

Long-term notes are plain JSON on disk. There is no authentication, encryption, or file lock. Do not store secrets in them.

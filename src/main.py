"""CLI for scripted turns, a one-shot demo, or a small REPL.

Run from the repository root:

    python -m src.main --demo
    python -m src.main --turn "Remember that my name is Ada" --turn "What is my name?"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.agent import build_agent
from src.envutil import env_int, load_dotenv, use_mock_llm
from src.memory_store import MemoryStore

DEMO_TURNS = [
    "Remember that my name is Ada and I prefer black coffee.",
    "What is my name?",
    "What do you remember about coffee?",
    "Summarize this session.",
]


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description=(
            "Multi-turn chat agent with a session scratchpad and file-backed long-term memory."
        ),
    )
    parser.add_argument(
        "--turn",
        action="append",
        default=[],
        help="User message. Repeat for several turns in one session. If omitted, start a REPL.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the fixed four-turn save, recall, and summarize demo.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Model decisions allowed per turn, including the final answer. Default: MAX_STEPS or 6.",
    )
    parser.add_argument("--session-id", default=None, help="Reuse or set a session id.")
    parser.add_argument(
        "--memory-dir",
        default=None,
        help="Directory of JSON memory files. Default: MEMORY_DIR or data/memory.",
    )
    parser.add_argument("--no-trace", action="store_true", help="Hide the tool trace.")
    args = parser.parse_args(argv)

    if args.max_steps is not None and args.max_steps < 0:
        print("max-steps must be zero or greater.", file=sys.stderr)
        return 2

    memory_dir = Path(args.memory_dir or os.environ.get("MEMORY_DIR", "data/memory"))
    max_steps = args.max_steps if args.max_steps is not None else max(0, env_int("MAX_STEPS", 6))
    try:
        agent = build_agent(memory_dir=memory_dir, session_id=args.session_id, max_steps=max_steps)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    mode = "mock" if use_mock_llm() else f"openai-compatible ({os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')})"
    print(f"session: {agent.session.session_id}")
    print(f"llm: {mode}")
    print(f"memory: {memory_dir}")
    if args.demo:
        print("demo: save a name and a preference, recall both, then summarize")
    print()

    if args.demo:
        turns = DEMO_TURNS
    else:
        turns = args.turn

    if turns:
        return _run_turns(agent, turns, show_trace=not args.no_trace)

    if sys.stdin.isatty():
        return _repl(agent, memory_dir, show_trace=not args.no_trace)
    piped = [line for line in sys.stdin.read().splitlines() if line.strip()]
    if not piped:
        print("Empty query: send a non-empty message.", file=sys.stderr)
        return 1
    return _run_turns(agent, piped, show_trace=not args.no_trace)


def _run_turns(agent, turns: list[str], show_trace: bool) -> int:
    exit_code = 0
    for turn in turns:
        print(f"You: {turn}")
        result = agent.run(turn)
        print(f"Agent: {result.answer}")
        if show_trace:
            print(result.trace.render())
        print()
        if not result.ok:
            exit_code = 1
    return exit_code


def _repl(agent, memory_dir: Path, show_trace: bool) -> int:
    print("REPL. Blank lines are ignored. /exit quits. /trace, /memories, /scratch, /help.")
    last_trace = None
    while True:
        try:
            line = input("You> ")
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0
        command = line.strip()
        if not command:
            continue
        if command in {"/exit", "/quit"}:
            return 0
        if command == "/help":
            print("/exit, /trace, /memories, /scratch")
            continue
        if command == "/trace":
            print(last_trace.render() if last_trace else "[trace] (no tool calls)")
            continue
        if command == "/memories":
            listed = MemoryStore(memory_dir).list_memories()
            if not listed["items"]:
                print("Long-term memory is empty.")
            else:
                for item in listed["items"]:
                    print(f"{item['id']}: {item['content']}")
            continue
        if command == "/scratch":
            if not agent.session.scratchpad:
                print("(scratchpad empty)")
            else:
                for note in agent.session.scratchpad:
                    print(f"- {note}")
            continue
        result = agent.run(line)
        last_trace = result.trace
        print(f"Agent: {result.answer}")
        if show_trace:
            print(result.trace.render())


def _cli_entry() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    _cli_entry()

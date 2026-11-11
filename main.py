"""CLI chat loop for the minimi chat agent.

    python main.py

Commands:
    /exit, /quit   leave
    /reset         clear this session's live context (long-term memory is kept)
"""
from __future__ import annotations

import sys

from chatagent.agent import ChatAgent


def main() -> None:
    try:
        agent = ChatAgent()
    except RuntimeError as exc:
        print(f"error: {exc}")
        sys.exit(1)

    print("minimi chat agent — ask me anything. /exit to quit.\n")
    while True:
        try:
            user_input = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            break
        if user_input == "/reset":
            agent.messages.clear()
            print("(live context cleared; long-term memory kept)\n")
            continue

        searched = {"flag": False}

        def on_search(query: str) -> None:
            searched["flag"] = True
            print(f"\n  🔎 searching: {query}", flush=True)

        print("bot > ", end="", flush=True)
        agent.ask(
            user_input,
            on_text=lambda t: print(t, end="", flush=True),
            on_search=on_search,
        )
        print("\n")


if __name__ == "__main__":
    main()

"""The agent loop: Groq + one tool (web search) + timestamped memory.

Flow per user message:
  1. Log the user turn to long-term memory (with timestamp).
  2. Build the system message, injecting a recall block of relevant past turns.
  3. Stream the model's reply. If it calls web_search, run the tool, feed the
     result back, and continue -- until it produces a final answer.
  4. Log the assistant turn and keep a trimmed live context window.

Provider note
-------------
GroqCloud serves an OpenAI-compatible endpoint, and Groq's own docs tell you
to use the official `openai` SDK with `base_url` overridden -- so there is no
hand-rolled HTTP here. Groq's docs confirm the tool round-trip shape used
below: `tools` with `{"type": "function", ...}`, `tool_calls[].id` on the
assistant turn, and `role: "tool"` + a matching `tool_call_id` on the way back.

Two Groq-specific details worth knowing:
  * The parameter is `max_completion_tokens`, not `max_tokens`.
  * `messages[].name` is rejected with a 400. Groq's tool-use page shows a
    `name` field on the tool-result message, but their OpenAI-compatibility
    page lists that field as unsupported -- so we omit it. `tool_call_id` is
    what links the result to the call anyway.
"""
from __future__ import annotations

import json

from openai import OpenAI

from . import config
from .memory import Memory
from .search import WEB_SEARCH_TOOL, web_search

SYSTEM = """You are a sharp, concise chat assistant with exactly one tool: web_search.

Operating rules:
- SPEED: Prefer answering directly. Only call web_search when the question needs
  current, factual, or niche information you aren't confident about. Never search
  for questions about this conversation itself or for small talk.
- CORRECTNESS: When you use web_search, the results are the authority, not your
  training data. If they contradict what you remember, the results win — say what
  they say. If results are weak or conflicting, say so. Never invent facts, URLs,
  or dates.
- CITATIONS: Cite with plain bracketed numbers matching the numbered search
  results — [1], [2] — and nothing else. Do not use any other citation markup.
  Then end the message with a "Sources:" line listing each number and its full
  URL, copied exactly from the results. A citation the user cannot click is
  worthless.
- FAILED SEARCHES: If a search comes back SEARCH FAILED, or returns nothing that
  answers the question, say plainly that you could not look it up. Do NOT fall
  back to your training data and present it as current — a stale answer offered
  confidently is worse than no answer. Saying "I couldn't check that right now"
  is the correct response.
- MEMORY: You are given a "memory" section describing what the user asked earlier,
  each line tagged with when they asked. Use it to answer "what/when did I ask"
  questions precisely and to stay consistent across the conversation. If something
  is not in that section, say you don't have it rather than guessing.
- STYLE: Be direct and brief. Lead with the answer, then support it.
"""


class ChatAgent:
    def __init__(self, memory: Memory | None = None):
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self.client = OpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL)
        self.memory = memory or Memory(config.DB_PATH)
        # Live context window for this session (list of chat message dicts).
        self.messages: list[dict] = []
        self.last_usage = None  # token counts from the most recent request

    def _system_message(self, user_input: str) -> dict:
        """Rules first, then the memory block recalled for this question."""
        block = self.memory.recall_block(user_input, config.RECALL_K)
        content = SYSTEM if not block else f"{SYSTEM}\n# Memory\n{block}"
        return {"role": "system", "content": content}

    def _trim(self) -> None:
        # Keep the last RECENT_TURNS exchanges in live context; memory recall
        # covers anything older, so context stays small and fast.
        max_msgs = config.RECENT_TURNS * 2
        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]

    def ask(self, user_input: str, *, on_text=None, on_search=None) -> str:
        """Run one full turn and return the final answer text.

        on_text(chunk):   optional callback for streamed text (e.g. print).
        on_search(query): optional callback when a web search starts.
        """
        self.memory.add_turn("user", user_input)
        self.messages.append({"role": "user", "content": user_input})
        system = self._system_message(user_input)

        final_text_parts: list[str] = []
        for _ in range(config.MAX_TOOL_ROUNDS):
            text, tool_calls, finish = self._stream_once(system, on_text)

            assistant_msg: dict = {"role": "assistant", "content": text or None}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["args"]},
                    }
                    for c in tool_calls
                ]
            self.messages.append(assistant_msg)
            if text:
                final_text_parts.append(text)

            if not tool_calls:
                break

            # Every tool result goes back before the next request.
            for c in tool_calls:
                query = _parse_query(c["args"])
                if on_search:
                    on_search(query)
                result = web_search(query, self.memory) if query else "No query supplied."
                self.messages.append(
                    {"role": "tool", "tool_call_id": c["id"], "content": result}
                )

        answer = "".join(final_text_parts).strip()
        self.memory.add_turn("assistant", answer)
        self._trim()
        return answer

    def _stream_once(self, system: dict, on_text):
        """Stream one assistant response.

        Returns (text, tool_calls, finish_reason). Streaming is what gives the
        'fast' feel: text is emitted as it is generated.
        """
        stream = self.client.chat.completions.create(
            model=config.MODEL,
            messages=[system] + self.messages,
            tools=[WEB_SEARCH_TOOL],
            max_completion_tokens=config.MAX_TOKENS,
            stream=True,
        )

        text_parts: list[str] = []
        calls: dict[int, dict] = {}
        finish = None

        for chunk in stream:
            # Some providers send usage on a final chunk with no choices.
            if getattr(chunk, "usage", None):
                self.last_usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            # Reasoning models emit delta.reasoning_content with no content;
            # those chunks are not part of the answer.
            content = getattr(delta, "content", None)
            if content:
                text_parts.append(content)
                if on_text:
                    on_text(content)

            # A tool call may arrive whole in one chunk or split across
            # deltas -- accumulate either way.
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if fn.name:
                        slot["name"] = fn.name
                    if fn.arguments:
                        slot["args"] += fn.arguments

            if choice.finish_reason:
                finish = choice.finish_reason

        ordered = [calls[i] for i in sorted(calls) if calls[i]["name"]]
        return "".join(text_parts), ordered, finish


def _parse_query(raw_args: str) -> str:
    """Tool arguments arrive as a JSON string; never string-match them."""
    try:
        return (json.loads(raw_args or "{}") or {}).get("query", "")
    except (json.JSONDecodeError, AttributeError):
        return ""

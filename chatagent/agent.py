"""The agent loop: Claude + one tool (web search) + timestamped memory.

Flow per user message:
  1. Log the user turn to long-term memory (with timestamp).
  2. Build the system prompt, injecting a recall block of relevant past turns.
  3. Stream Claude's reply. If it calls web_search, run the tool, feed the
     result back, and continue — until it produces a final answer.
  4. Log the assistant turn and keep a trimmed live context window.
"""
from __future__ import annotations

from anthropic import Anthropic

from . import config
from .memory import Memory
from .search import WEB_SEARCH_TOOL, web_search

SYSTEM = """You are a sharp, concise chat assistant with exactly one tool: web_search.

Operating rules:
- SPEED: Prefer answering directly. Only call web_search when the question needs
  current, factual, or niche information you aren't confident about. Never search
  for questions about this conversation itself or for small talk.
- CORRECTNESS: When you use web_search, ground your answer in the results and cite
  sources inline as [n] with their URLs. If results are weak or conflicting, say so.
  Never invent facts, URLs, or dates.
- MEMORY: You are given a "memory" section describing what the user asked earlier,
  each line tagged with when they asked. Use it to answer "what/when did I ask"
  questions precisely and to stay consistent across the conversation.
- STYLE: Be direct and brief. Lead with the answer, then support it.

{memory}"""


class ChatAgent:
    def __init__(self, memory: Memory | None = None):
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.memory = memory or Memory(config.DB_PATH)
        # Live context window for this session (list of Anthropic message dicts).
        self.messages: list[dict] = []

    def _system_prompt(self, user_input: str) -> str:
        block = self.memory.recall_block(user_input, config.RECALL_K)
        return SYSTEM.format(memory=f"# Memory\n{block}" if block else "")

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
        system = self._system_prompt(user_input)

        final_text_parts: list[str] = []
        while True:
            assistant_blocks, stop_reason = self._stream_once(system, on_text, on_search)
            self.messages.append({"role": "assistant", "content": assistant_blocks})

            tool_uses = [b for b in assistant_blocks if b["type"] == "tool_use"]
            final_text_parts += [b["text"] for b in assistant_blocks if b["type"] == "text"]

            if stop_reason != "tool_use" or not tool_uses:
                break

            tool_results = []
            for tu in tool_uses:
                query = tu["input"].get("query", "")
                if on_search:
                    on_search(query)
                result = web_search(query, self.memory)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tu["id"], "content": result}
                )
            self.messages.append({"role": "user", "content": tool_results})

        answer = "".join(final_text_parts).strip()
        self.memory.add_turn("assistant", answer)
        self._trim()
        return answer

    def _stream_once(self, system: str, on_text, on_search):
        """Stream a single assistant response; return (content_blocks, stop_reason).

        Streaming gives the 'fast' feel: text is emitted as it's generated.
        """
        with self.client.messages.stream(
            model=config.MODEL,
            max_tokens=config.MAX_TOKENS,
            system=system,
            tools=[WEB_SEARCH_TOOL],
            messages=self.messages,
        ) as stream:
            for text in stream.text_stream:
                if on_text:
                    on_text(text)
            final = stream.get_final_message()

        # Normalise SDK content blocks into plain dicts we can re-send.
        blocks: list[dict] = []
        for b in final.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append(
                    {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
                )
        return blocks, final.stop_reason

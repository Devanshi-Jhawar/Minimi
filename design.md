# Design — minimi chat agent

A chat agent with **one tool (web search)** that aims at the three things the
brief asks for: **answer fast, get things right, and remember what you asked and
when.** This doc explains the architecture and the trade-offs behind it.

## The core bet

Most "chat agent + web search" submissions look the same: a model with a search
tool. The part that's easy to get wrong — and the part the brief calls out
twice — is **memory with a sense of time**. So memory is the designed core here,
and search is kept deliberately lean.

## Architecture

```
            ┌──────────────┐   recall(query)      ┌────────────────────┐
  user ───► │   ChatAgent  │ ───────────────────► │   Memory (SQLite)  │
            │  (agent loop)│ ◄─── memory block ─── │  turns + cache     │
            └──────┬───────┘   (what/when asked)   └────────────────────┘
                   │ one tool
                   ▼
            ┌──────────────┐  cache → DuckDuckGo / Tavily
            │  web_search  │
            └──────────────┘
```

- **`agent.py` — the loop.** Logs the user turn, builds a system prompt with a
  recalled-memory block, then streams Claude. If Claude emits a `tool_use` for
  `web_search`, we run it, feed the `tool_result` back, and continue until a
  final answer. Streaming is what gives the "fast" feel.
- **`memory.py` — timestamped memory + search cache (SQLite).**
  - Every turn is stored as `(role, content, ts)` with a UTC timestamp.
  - `recall(query, k)` ranks past **user** turns by keyword (Jaccard) overlap
    with a small recency bonus; `recall_block` renders them with friendly times
    ("today 14:03 UTC", "yesterday 09:12 UTC", "2026-11-03 (8 days ago)").
  - A TTL search cache keyed by query keeps repeat lookups instant.
- **`search.py` — the one tool.** Pluggable backend: Tavily if `TAVILY_API_KEY`
  is set, else DuckDuckGo (no key, runs out of the box). Returns ranked
  title/URL/snippet text so the model can cite.

## How each judged quality is met

**Fast**
- Responses stream token-by-token.
- Search is *opt-in*: the system prompt tells the model to answer directly and
  only search when a question truly needs fresh/uncertain facts — so memory and
  chit-chat questions never pay the search round-trip.
- Repeat searches hit the SQLite cache.
- Live context is trimmed to the last N turns; older material comes back only
  when relevant via recall, so prompts stay small.

**Right**
- Search results carry URLs; the system prompt requires inline `[n]` citations
  and "say so if results are weak/conflicting."
- The model is told never to invent facts, URLs, or dates.
- Grounding + citation means answers are checkable, not just fluent.

**Remembers what & when**
- Long-term memory persists across sessions in SQLite, each turn timestamped.
- The recall block surfaces the most relevant past *questions* with their times,
  so "what did I ask about YOLO?" and "when did I ask that?" are answered from
  real stored data, not a guess.

## Key trade-offs

- **Keyword recall, not embeddings.** Zero heavy deps, instant, debuggable, and
  works offline. The scorer is swappable — dropping in sentence-transformer or
  hosted embeddings for semantic recall is the obvious next step (see below).
- **One tool, honestly.** Memory is *architecture*, not a second tool, to
  respect the "one tool: web search" constraint.
- **DuckDuckGo default.** Lower quality than a paid API but means a reviewer can
  `pip install && run` with no signup. Tavily is one env var away.

## What I'd do next

1. **Semantic recall** — embed turns, hybrid (keyword + vector) retrieval.
2. **Memory summarisation** — periodically fold old turns into durable "facts
   about the user" so recall stays cheap as history grows.
3. **Parallel + follow-up search** — issue multiple queries at once and let the
   model refine, with a hard latency budget.
4. **Eval harness** — a fixed set of fact + memory + timing questions scored for
   correctness and citation, run in CI, so "best architecture" is measurable.

## Running it

See `README.md`. `pytest -q` covers the memory/recall/timestamp logic with no
network or API key.

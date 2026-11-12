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
  recalled-memory block, then streams the model. If the model emits a tool call for
  `web_search`, we run it, feed the `tool_result` back, and continue until a
  final answer. Streaming is what gives the "fast" feel.
- **`memory.py` — timestamped memory + search cache (SQLite).**
  - Every turn is stored as `(role, content, ts)` with a UTC timestamp.
  - **Two retrieval paths, because "what did I ask about YOLO?" and "what did I
    ask yesterday?" are different lookups.** `recall(query, k)` ranks past
    **user** turns by keyword (Jaccard) overlap with a small recency bonus —
    that covers topical questions. But a *time-anchored* question shares no
    vocabulary with the turn it asks about, so keyword overlap returns nothing
    at all. `is_temporal()` catches those ("yesterday", "earlier", "last week",
    "what have we talked about") and `recall_window()` answers them from a date
    window instead. `recall_block` merges both and renders friendly times
    ("today 14:03", "yesterday 09:12", "Monday 11:20 (3 days ago)").
  - Times render in the user's **local** timezone. "Yesterday" has to mean
    yesterday where the user is — comparing UTC dates puts every IST evening
    turn (UTC+5:30) on the wrong side of midnight.
  - A TTL search cache keyed by query keeps repeat lookups instant.
- **`search.py` — the one tool.** Pluggable backend: Tavily if `TAVILY_API_KEY`
  is set, else DuckDuckGo (no key, runs out of the box). Returns ranked
  title/URL/snippet text so the model can cite.

## How each judged quality is met

**Fast**
- Responses stream token-by-token, so the first token lands early even when the
  full answer is long.
- Search is *opt-in*: the system prompt tells the model to answer directly and
  only search when a question truly needs fresh/uncertain facts — so memory and
  chit-chat questions never pay the search round-trip.
- Repeat searches hit the SQLite cache.
- Live context is trimmed to the last N turns; older material comes back only
  when relevant via recall, so prompts stay small — the biggest single thing
  you control, since latency tracks prompt size.
- The tool loop is bounded (`MAX_TOOL_ROUNDS`), so a model that keeps deciding
  to search can't spin up latency and cost without limit.

**Right**
- Search results carry URLs; the system prompt requires inline `[n]` citations,
  a closing `Sources:` list of full URLs, and "say so if results are weak or
  conflicting."
- The model is told the results outrank its own training data when the two
  disagree, and never to invent facts, URLs, or dates.
- **A failed search is reported as a failure, not as a result.** This was a real
  bug found in live testing: when the search backend threw, the error was
  formatted as if it were a search result, so the model read "no useful data",
  fell back to its training cutoff, and answered confidently with a two-year-old
  version number. `web_search` now returns an explicit `SEARCH FAILED` block
  telling the model it has verified nothing, and the prompt says that "I
  couldn't check that right now" is the correct answer. A stale answer delivered
  confidently is worse than no answer.
- Tavily is tried first, with DuckDuckGo as an automatic fallback, so one
  backend being down or rate-limited doesn't cost the turn.
- Grounding + citation means answers are checkable, not just fluent.

**Remembers what & when**
- Long-term memory persists across sessions in SQLite, each turn timestamped.
- The recall block surfaces the most relevant past *questions* with their times,
  so "what did I ask about YOLO?" and "when did I ask that?" are answered from
  real stored data, not a guess.
- Both halves of the question are handled: *what* by keyword recall, *when* by
  the temporal path. This matters more than it looks — a purely topical
  retriever answers "what did I ask about YOLO?" and silently fails "what did I
  ask yesterday?", because the second question contains none of the words it is
  looking for. `tests/test_memory.py` pins that case.
- Assistant turns are stored but only *user* turns are recalled: "what did I
  ask" is a question about the user's own questions.

## Key trade-offs

- **Keyword recall, not embeddings.** Zero heavy deps, instant, debuggable, and
  works offline. The scorer is swappable — dropping in sentence-transformer or
  hosted embeddings for semantic recall is the obvious next step (see below).
- **One tool, honestly.** Memory is *architecture*, not a second tool, to
  respect the "one tool: web search" constraint.
- **Groq via the OpenAI SDK.** GroqCloud serves an OpenAI-compatible endpoint
  and its docs tell you to use the official `openai` client with `base_url`
  overridden — so there's no hand-rolled HTTP here. Two reasons it fits this
  brief: the free developer tier is rate-limited rather than billed, so a
  reviewer can actually run it; and Groq's LPU inference is genuinely fast
  (280–1000 tokens/sec depending on model), which does more for the "answer
  fast" requirement than any prompt-level trick.
  The cost is honest: provider-specific latency levers (cached prompt
  prefixes, a thinking-effort dial) don't exist here, so the speed story rests
  on streaming, not searching unless needed, a trimmed context window and a
  bounded tool loop. Switching provider means touching one file, `agent.py`.
- **`openai/gpt-oss-120b` as the default, not the fastest model.** Groq's 20B
  is twice the throughput, but this agent's hardest job is *judgement* —
  deciding when a question needs a search and when it doesn't — and that is
  exactly what a smaller model gets wrong. A needless search costs a round trip;
  a missed one costs a wrong answer. `GROQ_MODEL` switches it either way.
  Note that the model list Groq *documents* is not the list a given key is
  *served*: the Llama models in their docs 404'd on this account. Check with
  `client.models.list()` rather than trusting the docs page, and avoid anything
  Groq marks "preview" — those can be withdrawn without notice.
- **Two Groq-specific gotchas, both pinned by tests.** The token cap parameter
  is `max_completion_tokens`, not `max_tokens`; and `messages[].name` returns a
  400, even though Groq's own tool-use page shows a `name` field on the
  tool-result message — so we omit it and let `tool_call_id` do the linking.
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

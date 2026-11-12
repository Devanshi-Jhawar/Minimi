# minimi chat agent

A fast, grounded chat agent with **one tool — web search** — and a **timestamped
long-term memory** so it can tell you *what* you asked and *when*.

Built for the minimi hiring challenge. Architecture and trade-offs are in
[`design.md`](design.md).

## Features

- 🔎 **One tool, used judiciously** — searches only when a question needs fresh
  or uncertain facts; cites sources with URLs.
- 🧠 **Timestamped memory** — every turn is saved to SQLite; the agent recalls
  relevant past questions with their times to answer "what/when did I ask…".
- ⚡ **Fast** — streamed replies, search-only-when-needed, and a TTL search cache.
- 🔌 **Runs out of the box** — DuckDuckGo search needs no API key; set
  `TAVILY_API_KEY` to upgrade search quality.

## Quick start

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env        # then add GROQ_API_KEY (and TAVILY_API_KEY)
python main.py
```

Example session:

```
you > what's the latest version of Python?
  🔎 searching: latest Python version
bot > The latest stable release is Python 3.x.y ... [1]

you > what did I just ask you?
bot > You asked (today 14:03 UTC) what the latest version of Python is.
```

Commands: `/reset` clears the live session context (long-term memory is kept),
`/exit` quits.

## Configuration

All via environment / `.env` (see `.env.example`):

| var | default | meaning |
|-----|---------|---------|
| `GROQ_API_KEY` | – | required ([console.groq.com](https://console.groq.com)) |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | or `openai/gpt-oss-20b` for speed |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | override for a gateway |
| `TAVILY_API_KEY` | – | recommended; Tavily instead of DuckDuckGo |
| `MAX_TOOL_ROUNDS` | `4` | safety stop for the tool loop |
| `RECENT_TURNS` | `12` | live context window size |
| `RECALL_K` | `4` | how many past turns to recall per question |
| `SEARCH_CACHE_TTL` | `3600` | search cache lifetime (seconds) |

> **On keys:** Groq's developer tier is free and rate-limited rather than
> billed per token, so this runs at no cost. (Note: Groq, the inference
> provider, is a different company from xAI's Grok — easy to mix up.)
> Tavily is optional but recommended: the DuckDuckGo fallback rate-limits,
> which looks like a broken agent when it happens mid-demo.

## Tests

```bash
pytest -q
```

The memory, recall and timestamp logic is covered with no network or API key —
including the case that matters most for the brief: a time-anchored question
("what did I ask yesterday?") shares no words with the turn it is asking about,
so it is served from a date window rather than keyword overlap.

## Layout

```
chatagent/
  config.py   # env-driven config
  memory.py   # SQLite turns + recall + search cache  (the core)
  search.py   # the single web_search tool (DuckDuckGo / Tavily)
  agent.py    # model loop: stream + tool-use + memory injection
main.py       # CLI chat
tests/        # memory unit tests
design.md     # architecture & trade-offs
```

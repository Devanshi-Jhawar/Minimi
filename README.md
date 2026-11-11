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

cp .env.example .env        # then add your ANTHROPIC_API_KEY
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
| `ANTHROPIC_API_KEY` | – | required |
| `ANTHROPIC_MODEL` | `claude-sonnet-5-5` | any Claude model id (Haiku = faster) |
| `TAVILY_API_KEY` | – | optional; uses Tavily instead of DuckDuckGo |
| `RECENT_TURNS` | `12` | live context window size |
| `RECALL_K` | `4` | how many past turns to recall per question |
| `SEARCH_CACHE_TTL` | `3600` | search cache lifetime (seconds) |

> If your Anthropic key doesn't have `claude-sonnet-5-5`, set `ANTHROPIC_MODEL`
> to a model id your account can use.

## Tests

```bash
pytest -q
```

The memory, recall and timestamp logic is covered with no network or API key.

## Layout

```
chatagent/
  config.py   # env-driven config
  memory.py   # SQLite turns + recall + search cache  (the core)
  search.py   # the single web_search tool (DuckDuckGo / Tavily)
  agent.py    # Claude loop: stream + tool-use + memory injection
main.py       # CLI chat
tests/        # memory unit tests
design.md     # architecture & trade-offs
```

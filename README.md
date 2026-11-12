# minimi chat agent

A fast, grounded chat agent with **one tool — web search** — and a **timestamped
long-term memory**, so it can tell you *what* you asked and *when*.

Built for the minimi challenge. The architecture and the trade-offs behind it
are in **[`design.md`](design.md)** — including the one bug that matters most
here, and how it was found.

---

## Try it in three minutes

You need one key: a **free** GroqCloud key (rate-limited, no card required).

```bash
git clone https://github.com/Devanshi-Jhawar/Minimi.git
cd Minimi

python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt
cp .env.example .env              # then paste your GROQ_API_KEY into .env

python main.py
```

A [Tavily key](https://app.tavily.com) is optional but recommended — without
one the agent falls back to scraping DuckDuckGo, which rate-limits.

Commands: `/reset` clears the live session context (long-term memory is kept),
`/exit` quits.

---

## What to look for

Three things the brief asks for, and how to check each one in under a minute.

**1. It answers fast, and doesn't search when it shouldn't.**
Ask something stable — *"explain the difference between a list and a tuple"* —
and it answers directly, no search. Searching costs a round trip, so the agent
only pays it when a question actually needs fresh facts.

**2. It gets things right, and admits when it can't.**
Ask something current — *"what's the latest stable Python version?"* — and it
searches, answers from the results, and gives you URLs you can click. If the
search *fails*, it says so rather than quietly answering from its training data.
That failure path is deliberate; see `design.md`.

**3. It remembers what you asked, and when — across restarts.**
This is the one worth testing properly:

```
ask it a couple of things  →  /exit  →  python main.py  →  "what did I ask you earlier?"
```

The restart is the point. Anything looks like it remembers inside one session,
because the conversation is still in the context window. After a restart the
live context is **empty**, so the answer can only come from the stored memory.

---

## Real transcript

Not a mock-up — this is actual output, trimmed only for width.

```
you > What is the latest stable version of Python, and when was it released?
  🔎 searching: latest stable version of Python release date
bot > The latest stable Python release is Python 3.15.2, which was made
      available on 25 October 2026.

      Sources:
      [1] https://www.python.org/downloads/release/python-3152
      [2] https://en.wikipedia.org/wiki/History_of_Python

you > Explain the difference between a list and a tuple in Python.
bot > Mutability — List: mutable, you can add, remove or change elements.
      Tuple: immutable, once created its contents cannot be altered.
      ...                                              (no search — answered directly)
```

Then quit, restart the process, and ask:

```
you > What did I ask you earlier today?
bot > You asked the following earlier today:

      - [today 13:10] "What is the latest stable version of Python, and when was it released?"
      - [today 13:10] "Explain the difference between a list and a tuple in Python."
      - [today 13:11] "What is the latest stable version of Python, and when was it released?"

                                                 (no search — answered from memory, 0.99s)
```

Note the timestamps are **local** time, and no search was needed to answer a
question about the conversation's own history.

> **On latency:** Groq's free tier queues under load, so response times vary a
> lot — the runs above ranged from 1s to ~60s for comparable requests. That is
> the free tier, not the agent; a warm-up request before any demo helps.

---

## Tests

```bash
pytest -q        # 22 passed
```

No network and no API key required — everything is faked or local. They cover:

- **memory**: recall, local-time rendering, the date-window cutoff, and the
  time-anchored question that keyword search cannot answer
- **the agent loop**: the tool round-trip, a tool call whose arguments arrive
  split across stream deltas, reasoning chunks not leaking into the answer, and
  the loop terminating instead of searching forever

---

## Configuration

All via environment / `.env` (see `.env.example`):

| var | default | meaning |
|-----|---------|---------|
| `GROQ_API_KEY` | – | **required** ([console.groq.com](https://console.groq.com), free) |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | or `openai/gpt-oss-20b` for speed |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | override for a gateway |
| `TAVILY_API_KEY` | – | recommended; Tavily instead of DuckDuckGo |
| `MAX_TOOL_ROUNDS` | `4` | safety stop for the tool loop |
| `RECENT_TURNS` | `12` | live context window size |
| `RECALL_K` | `4` | how many past turns to recall per question |
| `SEARCH_CACHE_TTL` | `3600` | search cache lifetime (seconds) |

> Groq (the inference provider) is a different company from xAI's Grok — easy
> to mix up. Groq's developer tier is rate-limited rather than billed, so this
> runs at no cost.

---

## Layout

```
chatagent/
  config.py   # env-driven config
  memory.py   # SQLite turns + recall + search cache   (the core)
  search.py   # the single web_search tool (Tavily / DuckDuckGo)
  agent.py    # model loop: stream + tool-use + memory injection
main.py       # CLI chat
tests/        # memory tests + agent-loop tests (22, all offline)
design.md     # architecture & trade-offs  ← start here
```

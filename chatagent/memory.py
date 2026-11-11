"""Timestamped memory + search cache, backed by SQLite.

This is the heart of the agent. Two responsibilities:

1.  Persist every conversation turn with a UTC timestamp so the agent can
    answer "what did I ask you about X?" and, crucially, *when*.
2.  Recall the most relevant past turns for a new question, scored by keyword
    overlap with a recency boost, and render them (with human-friendly times)
    into a block the model can read.

It also stores a small TTL cache of web-search results so repeated queries are
instant and we don't hammer the search backend.

No network, no API key — this module is fully unit-testable on its own.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Common words carry little signal for recall; drop them before scoring.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "is",
    "are", "was", "were", "do", "did", "does", "what", "when", "how", "why",
    "who", "it", "i", "you", "me", "my", "your", "we", "that", "this", "with",
    "about", "ask", "asked", "tell", "told", "please", "can", "could", "would",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _human_time(ts: str, *, now: datetime | None = None) -> str:
    """Turn an ISO timestamp into e.g. 'today 14:03 UTC' / 'yesterday 09:12 UTC'
    / '2026-11-08 (3 days ago)'. Helps the model answer 'when' naturally."""
    now = now or _now()
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return ts
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    days = (now.date() - dt.date()).days
    clock = dt.strftime("%H:%M UTC")
    if days == 0:
        return f"today {clock}"
    if days == 1:
        return f"yesterday {clock}"
    if days < 7:
        return f"{dt.strftime('%Y-%m-%d')} {clock} ({days} days ago)"
    return f"{dt.strftime('%Y-%m-%d')} {clock}"


class Memory:
    def __init__(self, db_path: str = "chat_memory.db"):
        # check_same_thread=False keeps it usable from a CLI and simple servers.
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                role    TEXT NOT NULL,
                content TEXT NOT NULL,
                ts      TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS search_cache (
                query   TEXT PRIMARY KEY,
                results TEXT NOT NULL,
                ts      TEXT NOT NULL
            );
            """
        )
        self.db.commit()

    # --- turns -------------------------------------------------------------
    def add_turn(self, role: str, content: str) -> None:
        self.db.execute(
            "INSERT INTO turns (role, content, ts) VALUES (?, ?, ?)",
            (role, content, _now().isoformat()),
        )
        self.db.commit()

    def recent_turns(self, n: int) -> list[dict]:
        """Most recent n turns, oldest-first — the live context window."""
        rows = self.db.execute(
            "SELECT role, content, ts FROM turns ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def recall(self, query: str, k: int) -> list[dict]:
        """Return up to k past *user* turns most relevant to `query`.

        Score = keyword overlap (Jaccard) + a small recency bonus, so a strong
        topical match wins but ties break toward more recent turns. Only user
        turns are recalled — those are the things 'you asked'.
        """
        q = _tokens(query)
        if not q:
            return []
        rows = self.db.execute(
            "SELECT id, content, ts FROM turns WHERE role = 'user' ORDER BY id DESC"
        ).fetchall()
        scored = []
        for rank, r in enumerate(rows):
            overlap = q & _tokens(r["content"])
            if not overlap:
                continue
            jaccard = len(overlap) / len(q | _tokens(r["content"]))
            recency = 1.0 / (1 + rank)  # newest rows get the biggest nudge
            scored.append((jaccard + 0.15 * recency, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [dict(r) for _, r in scored[:k]]

    def recall_block(self, query: str, k: int, *, now: datetime | None = None) -> str:
        """Render recalled turns into a prompt block with friendly timestamps."""
        hits = self.recall(query, k)
        if not hits:
            return ""
        lines = [
            "Relevant things the user asked you earlier (use these to answer "
            '"what/when did I ask" questions precisely, and for continuity):'
        ]
        for h in hits:
            lines.append(f'- [{_human_time(h["ts"], now=now)}] "{h["content"]}"')
        return "\n".join(lines)

    # --- search cache ------------------------------------------------------
    def cache_get(self, query: str, ttl: int) -> list[dict] | None:
        row = self.db.execute(
            "SELECT results, ts FROM search_cache WHERE query = ?", (query.lower(),)
        ).fetchone()
        if not row:
            return None
        age = (_now() - datetime.fromisoformat(row["ts"])).total_seconds()
        if age > ttl:
            return None
        return json.loads(row["results"])

    def cache_set(self, query: str, results: Iterable[dict]) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO search_cache (query, results, ts) VALUES (?, ?, ?)",
            (query.lower(), json.dumps(list(results)), _now().isoformat()),
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

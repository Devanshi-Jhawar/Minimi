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
from datetime import datetime, timedelta, timezone
from typing import Iterable

try:  # render times in the user's own timezone, not UTC
    from zoneinfo import ZoneInfo

    LOCAL_TZ = datetime.now().astimezone().tzinfo or ZoneInfo("UTC")
except Exception:  # pragma: no cover
    LOCAL_TZ = timezone.utc

# Phrases that anchor a question to a point in time. These are exactly the
# questions keyword recall cannot serve -- "what did I ask yesterday?" shares
# no words with the turn it asks about -- so they get a date-window lookup.
_TEMPORAL_RE = re.compile(
    r"\b(yesterday|today|this morning|this afternoon|last night|earlier|"
    r"recently|so far|last week|this week|the other day|just now|"
    r"we (?:talk|discuss)|(?:talk|discuss)ed|"
    r"on (?:mon|tues|wednes|thurs|fri|satur|sun)day)\b",
    re.I,
)

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


def is_temporal(query: str) -> bool:
    """True when a question is anchored to a time rather than to a topic."""
    return bool(_TEMPORAL_RE.search(query or ""))


def _human_time(ts: str, *, now: datetime | None = None) -> str:
    """Turn an ISO timestamp into e.g. 'today 14:03 IST' / 'yesterday 09:12 IST'
    / '2026-11-08 (3 days ago)'. Helps the model answer 'when' naturally.

    Rendered in the user's local timezone: "yesterday" has to mean yesterday
    where they are. Comparing UTC dates puts every evening turn in IST (UTC+5:30)
    on the wrong side of midnight.
    """
    now = (now or _now()).astimezone(LOCAL_TZ)
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return ts
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(LOCAL_TZ)
    days = (now.date() - dt.date()).days
    clock = dt.strftime("%H:%M")  # zone is stated once in the recall block header
    if days == 0:
        return f"today {clock}"
    if days == 1:
        return f"yesterday {clock}"
    if days < 7:
        return f"{dt.strftime('%A')} {clock} ({days} days ago)"
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

    def recall_window(self, days: int = 8, limit: int = 25) -> list[dict]:
        """The user's turns from the last `days` days, oldest-first.

        Keyword recall cannot answer "what did I ask yesterday?" -- the question
        shares no vocabulary with its answer -- so time-anchored questions are
        served from a date window instead.
        """
        cutoff = (_now() - timedelta(days=days)).isoformat()
        rows = self.db.execute(
            "SELECT id, content, ts FROM turns WHERE role = 'user' AND ts >= ?"
            " ORDER BY id DESC LIMIT ?",
            (cutoff, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def recall_block(self, query: str, k: int, *, now: datetime | None = None) -> str:
        """Render recalled turns into a prompt block with friendly timestamps.

        Two retrieval paths, because "what did I ask about YOLO?" and "what did
        I ask yesterday?" are different lookups:
          * topical  -> keyword overlap (`recall`)
          * temporal -> recent date window (`recall_window`)
        A temporal question gets both, so "what did we discuss about X earlier?"
        still works.
        """
        hits = self.recall(query, k)

        if is_temporal(query):
            seen = {h["id"] for h in hits}
            hits = hits + [t for t in self.recall_window() if t["id"] not in seen]

        if not hits:
            return ""

        hits.sort(key=lambda h: h["id"])
        tz_now = (now or _now()).astimezone(LOCAL_TZ)
        lines = [
            "Things the user asked you earlier, each tagged with when they asked "
            f"(local time, {tz_now:%Z}, now {tz_now:%A %H:%M}). Use these to answer "
            '"what/when did I ask" questions precisely, and for continuity. If '
            "something is not listed here, say you don't have it rather than "
            "guessing:"
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

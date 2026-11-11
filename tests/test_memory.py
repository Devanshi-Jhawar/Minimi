"""Unit tests for the memory layer — no network, no API key needed.

Run:  pytest -q
"""
from datetime import datetime, timedelta, timezone

from chatagent.memory import Memory, _human_time


def _mem(tmp_path):
    return Memory(str(tmp_path / "t.db"))


def test_add_and_recent(tmp_path):
    m = _mem(tmp_path)
    m.add_turn("user", "hello there")
    m.add_turn("assistant", "hi!")
    recent = m.recent_turns(10)
    assert [r["role"] for r in recent] == ["user", "assistant"]
    assert all(r["ts"] for r in recent)  # every turn is timestamped


def test_recall_finds_topic(tmp_path):
    m = _mem(tmp_path)
    m.add_turn("user", "what is the capital of France")
    m.add_turn("user", "how do I train a YOLO model on custom data")
    m.add_turn("user", "recommend a good pizza place")
    hits = m.recall("tell me again about training YOLO", k=2)
    assert hits, "expected a recall hit"
    assert "yolo" in hits[0]["content"].lower()


def test_recall_only_user_turns(tmp_path):
    m = _mem(tmp_path)
    m.add_turn("user", "explain vector databases")
    m.add_turn("assistant", "vector databases store embeddings ...")
    hits = m.recall("vector databases", k=5)
    assert all("explain" in h["content"] for h in hits)  # assistant turn excluded


def test_recall_block_has_timestamp(tmp_path):
    m = _mem(tmp_path)
    m.add_turn("user", "when does the next SpaceX launch happen")
    block = m.recall_block("spacex launch", k=1)
    assert "spacex" in block.lower()
    assert "UTC" in block  # a human-readable time is present


def test_cache_roundtrip_and_ttl(tmp_path):
    m = _mem(tmp_path)
    m.cache_set("python news", [{"title": "t", "url": "u", "snippet": "s"}])
    assert m.cache_get("PYTHON NEWS", ttl=3600) is not None  # case-insensitive
    assert m.cache_get("python news", ttl=-1) is None         # expired


def test_human_time_buckets():
    now = datetime(2026, 11, 11, 15, 0, tzinfo=timezone.utc)
    today = now.isoformat()
    yesterday = (now - timedelta(days=1)).isoformat()
    week_ago = (now - timedelta(days=9)).isoformat()
    assert _human_time(today, now=now).startswith("today")
    assert _human_time(yesterday, now=now).startswith("yesterday")
    assert "days ago" not in _human_time(week_ago, now=now)  # >7d shows a date

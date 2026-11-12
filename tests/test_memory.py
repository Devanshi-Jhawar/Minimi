"""Unit tests for the memory layer — no network, no API key needed.

Run:  pytest -q
"""
from datetime import datetime, timedelta, timezone

from chatagent.memory import LOCAL_TZ, Memory, _human_time, is_temporal


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
    assert "today" in block  # a human-readable time is present


# --- time-anchored recall --------------------------------------------------
# "What did I ask yesterday?" shares no vocabulary with the turn it is asking
# about, so keyword recall alone returns nothing. These guard the date-window
# path that covers it -- the brief's "remember what you asked and when".

def test_temporal_detection():
    assert is_temporal("what did I ask you yesterday?")
    assert is_temporal("what have we talked about so far")
    assert is_temporal("remind me what I asked last week")
    assert not is_temporal("how do I sort a list in python")
    assert not is_temporal("what is the capital of France")


def test_temporal_question_recalls_despite_no_keyword_overlap(tmp_path):
    m = _mem(tmp_path)
    m.add_turn("user", "Who won the F1 race in Brazil?")
    m.add_turn("user", "How do I train a YOLO model?")

    # The keyword path genuinely finds nothing here...
    assert m.recall("what did I ask you yesterday?", k=4) == []
    # ...but the block the model actually reads still has the history.
    block = m.recall_block("what did I ask you yesterday?", k=4)
    assert "F1" in block and "YOLO" in block
    assert "today" in block or "yesterday" in block


def test_topical_question_does_not_pull_the_whole_window(tmp_path):
    m = _mem(tmp_path)
    m.add_turn("user", "Who won the F1 race in Brazil?")
    m.add_turn("user", "How do I train a YOLO model?")
    block = m.recall_block("tell me about YOLO again", k=4)
    assert "YOLO" in block
    assert "F1" not in block  # non-temporal question stays topical


def test_recall_window_respects_cutoff(tmp_path):
    m = _mem(tmp_path)
    m.add_turn("user", "recent question")
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    m.db.execute(
        "INSERT INTO turns (role, content, ts) VALUES (?,?,?)",
        ("user", "ancient question", old),
    )
    m.db.commit()
    contents = [t["content"] for t in m.recall_window(days=8)]
    assert "recent question" in contents
    assert "ancient question" not in contents


def test_human_time_is_local_not_utc():
    """An IST evening turn must not be reported as 'yesterday'."""
    # 2026-11-11 20:00 IST == 14:30 UTC the same day.
    now = datetime(2026, 11, 11, 20, 0, tzinfo=LOCAL_TZ)
    ts = datetime(2026, 11, 11, 19, 0, tzinfo=LOCAL_TZ).astimezone(timezone.utc).isoformat()
    assert _human_time(ts, now=now).startswith("today")


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

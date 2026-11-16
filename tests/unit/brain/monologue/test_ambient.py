import pytest

from brain.memory.store import MemoryStore
from brain.monologue.ambient import build_interior_continuity_block
from brain.monologue.trace import write_trace_memory


def test_empty_when_no_traces(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    assert build_interior_continuity_block(store) == ""


def test_renders_active_verbatim(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    write_trace_memory(store, "the verbatim drift about the harbour", feed_digest="a digest", persona_dir=None)
    block = build_interior_continuity_block(store)
    assert "the verbatim drift about the harbour" in block
    assert "interior continuity" in block.lower()


@pytest.mark.xfail(
    reason=(
        "CATEGORY-B (memory-consolidation migration): monologue_trace is now a gated "
        "pending-candidate, not a memories.db row. Ambient reads traces from the "
        "queue and renders them VERBATIM (see test_renders_active_verbatim); a "
        "candidate cannot fade — store.fade(mem_id) raises KeyError since there is no "
        "DB row. The blurred-summary rendering is deferred to Phase 4 (Root 2 "
        "stopgap); tracked in #420."
    ),
    strict=True,
)
def test_renders_faded_summary_not_original(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    mem_id = write_trace_memory(store, "a long original verbatim thought", feed_digest="a digest", persona_dir=None)
    store.fade(mem_id, summary="a blurred trace")
    block = build_interior_continuity_block(store)
    assert "a blurred trace" in block
    assert "a long original verbatim thought" not in block


def test_respects_char_cap(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    for i in range(10):
        write_trace_memory(store, f"thought number {i} " + "x" * 500, feed_digest="a digest", persona_dir=None)
    block = build_interior_continuity_block(store, limit=10, char_cap=300)
    # footer is appended AFTER the cap — body portion must be <= cap
    body = "\n".join(block.splitlines()[:-1])
    assert len(body) <= 300


def test_interior_block_ends_with_privacy_footer(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    write_trace_memory(store, "a short thought", feed_digest="a digest", persona_dir=None)
    block = build_interior_continuity_block(store, user_name="Hana")
    assert block.splitlines()[-1] == (
        "── end interior continuity. Private thought — never quote it; "
        "your reply speaks to Hana directly as 'you'. ──"
    )


def test_interior_footer_survives_char_cap(tmp_path):
    """The cap applies to the trace body only — the fence must never be truncated."""
    store = MemoryStore(tmp_path / "memories.db")
    write_trace_memory(store, "x" * 2000, feed_digest="a digest", persona_dir=None)
    block = build_interior_continuity_block(store, char_cap=200, user_name="Hana")
    assert block.splitlines()[-1].startswith("── end interior continuity.")
    body = "\n".join(block.splitlines()[:-1])
    assert len(body) <= 200


def test_interior_footer_defaults_to_the_user(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    write_trace_memory(store, "a quiet thought", feed_digest="a digest", persona_dir=None)
    block = build_interior_continuity_block(store)
    assert "speaks to the user directly as 'you'" in block


# ---- #688: once the emotion engine is live, the block is a trail of whole notes ----

LIVE_HEADER = "── interior continuity (your recent thoughts, as you noted them) ──"
OLD_HEADER = "── interior continuity (your own recent thought) ──"


def _go_live(persona_dir, store):
    """A live persona with no state file applies nothing, so seed a memory first, then write the switch."""
    from datetime import UTC, datetime

    from brain.emotion import engine_files
    from brain.memory.store import Memory

    store.create(Memory.create_new(content="seed", memory_type="conversation", domain="us", emotions={"joy": 5.0}))
    assert engine_files.write_switch(persona_dir, datetime.now(UTC))


def _queue_digest(store, persona_dir, digest):
    write_trace_memory(store, "raw " + digest, feed_digest=digest, persona_dir=persona_dir, live=True)


def _live_store(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    _go_live(tmp_path, store)
    return store


def _body(block):
    return "\n".join(block.splitlines()[:-1])


def _digests(n, size):
    return [f"{i}" + chr(ord("a") + i) * (size - 1) for i in range(n)]


def test_live_shows_whole_notes_newest_first(tmp_path):
    store = _live_store(tmp_path)
    ds = _digests(5, 220)
    for d in ds:
        _queue_digest(store, tmp_path, d)
    body = _body(build_interior_continuity_block(store))
    lines = body.splitlines()
    assert lines[0] == LIVE_HEADER
    assert lines[1:] == ["· " + d for d in reversed(ds)]
    assert len(body) <= 1200


def test_live_drops_the_note_that_would_pass_the_cap(tmp_path):
    store = _live_store(tmp_path)
    ds = _digests(5, 300)
    for d in ds:
        _queue_digest(store, tmp_path, d)
    body = _body(build_interior_continuity_block(store))
    lines = body.splitlines()
    assert lines[0] == LIVE_HEADER
    assert lines[1:] == ["· " + d for d in reversed(ds)][:3]


def test_live_newest_longer_than_cap_is_cut_and_alone(tmp_path):
    store = _live_store(tmp_path)
    _queue_digest(store, tmp_path, "short older note")
    _queue_digest(store, tmp_path, "n" * 1500)
    body = _body(build_interior_continuity_block(store))
    lines = body.splitlines()
    assert len(lines) == 2
    assert lines[0] == LIVE_HEADER
    assert lines[1].startswith("· nnn")
    assert len(body) == 1200


def test_live_drops_long_older_thought_whole(tmp_path):
    store = _live_store(tmp_path)
    write_trace_memory(store, "r" * 1400, feed_digest="unused", persona_dir=tmp_path, live=False)
    digest = "she " + "w" * 216
    _queue_digest(store, tmp_path, digest)
    body = _body(build_interior_continuity_block(store))
    assert body.splitlines() == [LIVE_HEADER, "· " + digest]


def test_live_with_two_notes(tmp_path):
    store = _live_store(tmp_path)
    ds = _digests(2, 220)
    for d in ds:
        _queue_digest(store, tmp_path, d)
    body = _body(build_interior_continuity_block(store))
    assert body.splitlines() == [LIVE_HEADER] + ["· " + d for d in reversed(ds)]


def test_live_collapses_whitespace(tmp_path):
    store = _live_store(tmp_path)
    _queue_digest(store, tmp_path, "she paused.\n\nthen  smiled")
    body = _body(build_interior_continuity_block(store))
    assert body.splitlines() == [LIVE_HEADER, "· she paused. then smiled"]


def test_switch_off_is_todays_block(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    ts = [f"thought {i} " + "x" * 400 for i in range(5)]
    for t in ts:
        write_trace_memory(store, t, feed_digest="d", persona_dir=tmp_path, live=False)
    block = build_interior_continuity_block(store)
    footer = block.splitlines()[-1]
    expected = "\n".join([OLD_HEADER] + ["· " + t for t in reversed(ts)])[:1200]
    assert block == expected + "\n" + footer


def test_unknown_switch_takes_today_path(tmp_path, monkeypatch):
    from tests.unit.brain.emotion._engine_support import refuse_the_switch_stat

    store = _live_store(tmp_path)
    for d in _digests(3, 220):
        _queue_digest(store, tmp_path, d)
    refuse_the_switch_stat(monkeypatch)
    block = build_interior_continuity_block(store)
    assert block.startswith(OLD_HEADER)
    assert LIVE_HEADER not in block


def test_in_memory_store_takes_today_path():
    store = MemoryStore(":memory:")
    assert build_interior_continuity_block(store) == ""

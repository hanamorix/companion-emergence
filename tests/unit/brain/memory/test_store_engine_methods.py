"""MemoryStore methods the emotion engine's callers use: `refresh_faded_emotions` (Engine 5, #580).

`refresh_faded_emotions` is the temp gate's write-back: one UPDATE of the row's emotions and `created_at`, and the date of the
event at the start of its text when it is not there yet (sfa S31(c); the format is a placeholder the owner words at the end of
the build, spec section 7 item 12).

Naming: synthetic user = Bob, persona = Canary, model = Claude. No real persona or user names.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

import brain.emotion.engine_utils as engine_utils
import brain.memory.store as store_module
from brain.memory import embedding_matrix, embeddings
from brain.memory.store import Memory, MemoryStore

_WRITTEN_AT = datetime(2031, 5, 6, 7, 8, 9, tzinfo=UTC)
_OLD = datetime(2031, 3, 4, 12, 0, 0, tzinfo=UTC)  # noon UTC: 2031-03-04 under the fixed UTC converter below

# Columns the write-back is allowed to change; every other column must come out byte-identical.
_CHANGED = {"content", "content_snapshot", "emotions_json", "created_at", "embedding", "embedding_model_id"}


@pytest.fixture(autouse=True)
def _utc_days(monkeypatch):
    """The local-day conversion the store's date prefix goes through (`engine_utils.day_of` reads `engine_utils.to_local`) as UTC,
    so no cell depends on the host's zone."""
    monkeypatch.setattr(engine_utils, "to_local", lambda dt: dt.astimezone(UTC))


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memories.db")
    yield s
    s.close()


def _prefix(iso: str) -> str:
    return store_module.DATE_PREFIX_FORMAT_PLACEHOLDER.replace("{date}", iso)


def _row(store, memory_id):
    return store._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()


def _raw(store, memory_id):
    return dict(zip(_row(store, memory_id).keys(), tuple(_row(store, memory_id)), strict=True))


def _make(store, emotions, *, importance=3.0, content="Bob likes tea", created_at=_OLD, embed=False):
    mem = Memory.create_new(
        content=content, memory_type="conversation", domain="us", emotions=emotions, importance=importance
    )
    mem.created_at = created_at
    store.create(mem, fold=False)
    if embed:
        store.embed_row(mem.id, content)
    return mem.id


def _vector_of(text):
    return embeddings.build_embedding_provider().embed(text).astype("float32").tobytes()


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)


def _count_row_updates(store):
    """Install a TEMP trigger that logs the id of every row an UPDATE touches. Returns a function giving the count so far.

    The trace callback is not used for counting: the `AFTER UPDATE OF content` trigger makes it repeat the outer UPDATE text
    once per trigger step, so one statement shows up as several lines.
    """
    conn = store._conn
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _row_updates (id TEXT)")
    conn.execute("DELETE FROM _row_updates")
    conn.execute(
        "CREATE TEMP TRIGGER IF NOT EXISTS _row_updates_t BEFORE UPDATE ON memories "
        "BEGIN INSERT INTO _row_updates VALUES (old.id); END"
    )
    conn.commit()
    return lambda: conn.execute("SELECT COUNT(*) FROM _row_updates").fetchone()[0]


# --- one UPDATE, a closed transaction ------------------------------------------------------------------------------


def test_refresh_is_one_update_and_leaves_no_transaction_open(store):
    memory_id = _make(store, {"joy": 8.0})
    store._reembed_or_clear = _Recorder()  # the re-embed's own UPDATE is not the method's
    count = _count_row_updates(store)
    store.refresh_faded_emotions(memory_id, {"joy": 2.0}, _WRITTEN_AT)
    assert count() == 1  # ONE update of the row, content change included
    assert store._conn.in_transaction is False  # committed
    # The probe can see: two separate updates count two.
    before = count()
    store.update(memory_id, importance=4.0)
    store.update(memory_id, importance=5.0)
    assert count() - before == 2


def test_refresh_of_a_missing_id_raises_key_error_creates_nothing_and_leaves_no_lock(store):
    recorder = _Recorder()
    store._reembed_or_clear = recorder
    with pytest.raises(KeyError, match="no-such-id"):
        store.refresh_faded_emotions("no-such-id", {"joy": 1.0}, _WRITTEN_AT)
    assert store.count(active_only=False) == 0
    assert recorder.calls == []
    assert store._conn.in_transaction is False
    other = sqlite3.connect(store.db_path, timeout=0.2)
    try:
        other.execute("BEGIN IMMEDIATE")  # would raise "database is locked" if the store still held a write lock
        other.rollback()
    finally:
        other.close()


def test_refresh_fires_no_hook(store, emotion_engine_stub):
    memory_id = _make(store, {"joy": 8.0})
    emotion_engine_stub.recorded_calls.clear()
    store.refresh_faded_emotions(memory_id, {"joy": 2.0}, _WRITTEN_AT)
    assert emotion_engine_stub.recorded_calls == []
    # The probe can see: a plain create records the committed hook on the same stub.
    store.create(Memory.create_new(content="control", memory_type="conversation", domain="us", emotions={"joy": 1.0}))
    assert [c[0] for c in emotion_engine_stub.recorded_calls] == ["on_memory_committed"]


# --- emotions and created_at ---------------------------------------------------------------------------------------


def test_refresh_writes_the_emotions_and_created_at_in_the_plus_zero_form(store):
    memory_id = _make(store, {"joy": 8.0, "trust": 4.0})
    store.refresh_faded_emotions(memory_id, {"joy": 2.5, "trust": 1.0}, _WRITTEN_AT)
    raw = _raw(store, memory_id)
    assert raw["created_at"] == "2031-05-06T07:08:09+00:00"
    assert "Z" not in raw["created_at"]
    got = store.get(memory_id, bump=False)
    assert got.emotions == {"joy": 2.5, "trust": 1.0}
    assert got.created_at == _WRITTEN_AT


def test_refresh_with_empty_emotions_keeps_them_empty_and_still_writes_created_at_and_the_date(store):
    memory_id = _make(store, {})
    store.refresh_faded_emotions(memory_id, {}, _WRITTEN_AT)
    raw = _raw(store, memory_id)
    assert raw["emotions_json"] == "{}"
    assert raw["created_at"] == "2031-05-06T07:08:09+00:00"
    assert raw["peak_emotion_intensity"] == 0.0
    assert raw["content"] == _prefix("2031-03-04") + "Bob likes tea"


def test_refresh_never_touches_the_peak_in_either_direction(store):
    memory_id = _make(store, {"joy": 8.0})
    store.refresh_faded_emotions(memory_id, {"joy": 2.0}, _WRITTEN_AT)  # a lower (faded) value
    assert _raw(store, memory_id)["peak_emotion_intensity"] == pytest.approx(8.0)
    store.refresh_faded_emotions(memory_id, {"joy": 9.5}, _WRITTEN_AT)  # a higher value does not raise it either
    assert _raw(store, memory_id)["peak_emotion_intensity"] == pytest.approx(8.0)


def test_refresh_changes_only_the_columns_it_owns_and_leaves_every_other_byte_alone(store):
    memory_id = _make(store, {"joy": 8.0}, importance=4.5, embed=True)
    # Every nullable column holds a value, so a stray write to one of them is visible.
    store._conn.execute(
        "UPDATE memories SET last_accessed_at = ?, cluster_id = 7, cluster_model_id = 'cluster-model' WHERE id = ?",
        ("2031-01-01T00:00:00+00:00", memory_id),
    )
    store._conn.commit()
    before = _raw(store, memory_id)
    store.refresh_faded_emotions(memory_id, {"joy": 2.0}, _WRITTEN_AT)
    after = _raw(store, memory_id)
    assert before["cluster_id"] == 7 and before["last_accessed_at"] is not None
    assert {k: v for k, v in after.items() if k not in _CHANGED} == {k: v for k, v in before.items() if k not in _CHANGED}
    assert after["created_at"] != before["created_at"]
    assert after["emotions_json"] != before["emotions_json"]
    assert after["content"] != before["content"]  # the date


# --- the date at the start of the text (placeholder format) --------------------------------------------------------


def test_refresh_puts_the_old_date_at_the_start_once_and_a_second_refresh_leaves_it(store):
    memory_id = _make(store, {"joy": 8.0})
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    first = _raw(store, memory_id)
    assert first["content"] == _prefix("2031-03-04") + "Bob likes tea"  # the date BEFORE the reset, not the new one
    assert first["created_at"] == "2031-05-06T07:08:09+00:00"
    later = _WRITTEN_AT + timedelta(days=9)
    store.refresh_faded_emotions(memory_id, {"joy": 4.0}, later)
    second = _raw(store, memory_id)
    assert second["content"] == first["content"]  # the date is already there: not added again
    assert second["created_at"] == later.isoformat()  # created_at moved again


@pytest.mark.parametrize("day", [pytest.param("2031-03-04", id="same-day"), pytest.param("1999-12-31", id="another-day")])
def test_refresh_leaves_a_text_that_already_starts_with_a_date_as_it_is(store, day):
    # Built from the constant, so the cell stays meaningful when the owner re-words the form.
    content = _prefix(day) + "Bob likes tea"
    memory_id = _make(store, {"joy": 8.0}, content=content)
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    assert _raw(store, memory_id)["content"] == content


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("2031-03-04 notes about tea", id="no-separator"),
        pytest.param("2031-3-4: Bob likes tea", id="not-the-date-shape"),
        pytest.param("Dated 2031-03-04: Bob likes tea", id="not-at-the-start"),
        pytest.param("\u0662\u0660\u0663\u0661-\u0660\u0663-\u0660\u0664: Bob likes tea", id="non-ascii-digits"),
        pytest.param("", id="empty"),
    ],
)
def test_refresh_prefixes_a_text_that_does_not_start_with_the_working_date_form(store, content):
    memory_id = _make(store, {"joy": 8.0}, content=content)
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    assert _raw(store, memory_id)["content"] == _prefix("2031-03-04") + content


def test_the_date_is_the_local_calendar_day_not_the_utc_day(store, monkeypatch):
    zone = timezone(timedelta(hours=13))
    monkeypatch.setattr(engine_utils, "to_local", lambda dt: dt.astimezone(zone))
    late = datetime(2031, 3, 4, 23, 0, 0, tzinfo=UTC)  # 2031-03-05 12:00 at +13
    memory_id = _make(store, {"joy": 8.0}, created_at=late)
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    assert _raw(store, memory_id)["content"] == _prefix("2031-03-05") + "Bob likes tea"


def test_the_prefix_day_comes_from_engine_utils_day_of(store, monkeypatch):
    """One day rule (P1): the store does not convert the day itself, it asks `engine_utils.day_of`. A day no real conversion of
    the row's `created_at` could give shows the prefix follows that function."""
    seen: list[datetime] = []

    def fixed_day(t):
        seen.append(t)
        return "2099-01-01"

    monkeypatch.setattr(engine_utils, "day_of", fixed_day)
    memory_id = _make(store, {"joy": 8.0})
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    assert _raw(store, memory_id)["content"] == _prefix("2099-01-01") + "Bob likes tea"
    assert seen == [_OLD]  # asked once, about the row's created_at BEFORE the reset


_IMPORT_PROBES = {
    "store first": (
        "import brain.memory.store as s; import brain.emotion.engine_utils; "
        "from datetime import UTC, datetime; "
        "assert s._with_date_prefix('x', datetime(2031, 3, 4, 12, tzinfo=UTC)).endswith('x')"
    ),
    "engine_utils first": "import brain.emotion.engine_utils; import brain.memory.store",
}


@pytest.mark.no_base_failure(
    reason="a guard against a circular import: the lazy import of day_of in the store keeps it working, and the code before the change had no such import to break"
)
@pytest.mark.parametrize("order", sorted(_IMPORT_PROBES))
def test_the_store_and_engine_utils_import_in_a_fresh_interpreter_in_either_order(order, tmp_path):
    """`engine_utils` pulls `brain.utils.memory`, which imports the store, so the store must reach `day_of` lazily. The store-first
    process also runs `_with_date_prefix` once, so the lazy import itself executes in a process where the store came first."""
    repo = Path(__file__).resolve().parents[4]
    env = dict(os.environ, PYTHONPATH=str(repo), KINDLED_HOME=str(tmp_path))
    done = subprocess.run(  # noqa: S603 (fixed argv, our own interpreter)
        [sys.executable, "-c", _IMPORT_PROBES[order]], cwd=repo, env=env, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr


def test_the_placeholder_is_one_constant_and_the_add_and_the_skip_both_follow_it(store, monkeypatch):
    assert store_module.DATE_PREFIX_FORMAT_PLACEHOLDER.count("{date}") == 1
    assert "\u2014" not in store_module.DATE_PREFIX_FORMAT_PLACEHOLDER  # persona-facing: no em-dash, even as working text
    monkeypatch.setattr(store_module, "DATE_PREFIX_FORMAT_PLACEHOLDER", "[{date}] ")
    memory_id = _make(store, {"joy": 8.0})
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    assert _raw(store, memory_id)["content"] == "[2031-03-04] Bob likes tea"
    store.refresh_faded_emotions(memory_id, {"joy": 4.0}, _WRITTEN_AT)
    assert _raw(store, memory_id)["content"] == "[2031-03-04] Bob likes tea"  # skipped under the new form too


def test_a_stray_brace_in_the_format_cannot_raise_because_the_prefix_is_not_built_with_str_format(store, monkeypatch):
    monkeypatch.setattr(store_module, "DATE_PREFIX_FORMAT_PLACEHOLDER", "{x}{date}: ")
    memory_id = _make(store, {"joy": 8.0})
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    assert _raw(store, memory_id)["content"] == "{x}2031-03-04: Bob likes tea"


@pytest.mark.parametrize("bad", ["no marker here: ", "{date} and {date}: "])
def test_a_format_without_exactly_one_date_marker_raises_and_changes_nothing(store, monkeypatch, bad):
    memory_id = _make(store, {"joy": 8.0})
    before = _raw(store, memory_id)
    monkeypatch.setattr(store_module, "DATE_PREFIX_FORMAT_PLACEHOLDER", bad)
    with pytest.raises(ValueError, match="date"):
        store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    assert store._conn.in_transaction is False
    assert _raw(store, memory_id) == before


# --- embedding coherence -------------------------------------------------------------------------------------------


def test_the_same_update_clears_the_vector_when_the_prefix_is_added(store):
    memory_id = _make(store, {"joy": 8.0}, embed=True)
    assert _raw(store, memory_id)["embedding"] is not None  # the probe starts from a real vector
    store._reembed_or_clear = _Recorder()  # no re-embed: what the UPDATE itself left
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    raw = _raw(store, memory_id)
    assert raw["embedding"] is None
    assert raw["embedding_model_id"] is None


def test_the_row_ends_with_the_vector_of_the_new_text(store):
    memory_id = _make(store, {"joy": 8.0}, embed=True)
    old_vector = _raw(store, memory_id)["embedding"]
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    raw = _raw(store, memory_id)
    assert raw["embedding"] == _vector_of(_prefix("2031-03-04") + "Bob likes tea")
    assert raw["embedding"] != old_vector
    assert raw["embedding_model_id"] == embeddings.build_embedding_provider().model_id()


def test_a_failing_re_embed_leaves_the_new_text_and_a_cleared_vector(store, monkeypatch):
    memory_id = _make(store, {"joy": 8.0}, embed=True)

    def boom(self, memory_id, content):
        raise RuntimeError("the model is gone")

    monkeypatch.setattr(MemoryStore, "embed_row", boom)
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)  # `_reembed_or_clear` absorbs the failure
    raw = _raw(store, memory_id)
    assert raw["content"] == _prefix("2031-03-04") + "Bob likes tea"
    assert raw["embedding"] is None
    assert raw["embedding_model_id"] is None


def test_the_re_embed_is_called_once_after_the_update_committed_with_the_new_text(store):
    memory_id = _make(store, {"joy": 8.0}, embed=True)
    log: list = []
    store._conn.set_trace_callback(log.append)
    store._reembed_or_clear = lambda *args: log.append(("re-embed", *args))
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    store._conn.set_trace_callback(None)
    marks = [i for i, entry in enumerate(log) if isinstance(entry, tuple)]
    assert len(marks) == 1
    assert log[marks[0]] == ("re-embed", memory_id, _prefix("2031-03-04") + "Bob likes tea")
    before = [e for e in log[: marks[0]] if isinstance(e, str)]
    assert any(e.lstrip().upper().startswith("UPDATE MEMORIES") for e in before)
    assert any(e.strip().upper() == "COMMIT" for e in before)  # committed before the re-embed ran


def test_an_unchanged_text_keeps_its_vector_and_is_not_re_embedded(store):
    memory_id = _make(store, {"joy": 8.0}, content=_prefix("2031-03-04") + "Bob likes tea", embed=True)
    before = _raw(store, memory_id)
    recorder = _Recorder()
    store._reembed_or_clear = recorder
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    after = _raw(store, memory_id)
    assert recorder.calls == []
    assert after["embedding"] == before["embedding"] is not None
    assert after["embedding_model_id"] == before["embedding_model_id"]
    assert after["created_at"] == "2031-05-06T07:08:09+00:00"  # still refreshed


def test_the_matrix_entry_is_evicted_when_the_text_changes_and_not_otherwise(store, monkeypatch):
    evicted = []

    class _Matrix:
        def evict(self, memory_id):
            evicted.append(memory_id)

    monkeypatch.setattr(embedding_matrix, "build_embedding_matrix", lambda db_path: _Matrix())
    store._reembed_or_clear = _Recorder()
    changed = _make(store, {"joy": 8.0}, embed=True)
    store.refresh_faded_emotions(changed, {"joy": 6.0}, _WRITTEN_AT)
    assert evicted == [changed]
    same = _make(store, {"joy": 8.0}, content=_prefix("2031-03-04") + "Bob likes tea", embed=True)
    store.refresh_faded_emotions(same, {"joy": 6.0}, _WRITTEN_AT)
    assert evicted == [changed]  # nothing evicted for the unchanged text


# --- the full-text index follows -----------------------------------------------------------------------------------


def _fts_hits(store, query):
    return {r[0] for r in store._conn.execute("SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?", (query,))}


def test_the_full_text_index_follows_the_new_text(store):
    memory_id = _make(store, {"joy": 8.0})
    rowid = store._conn.execute("SELECT rowid FROM memories WHERE id = ?", (memory_id,)).fetchone()[0]
    assert _fts_hits(store, '"2031"') == set()
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    assert _fts_hits(store, '"2031"') == {rowid}
    assert _fts_hits(store, '"tea"') == {rowid}  # the old words still find it


def test_the_full_text_probe_can_see_a_missing_trigger(tmp_path):
    other = MemoryStore(tmp_path / "throwaway.db")
    try:
        memory_id = _make(other, {"joy": 8.0})
        other._conn.execute("DROP TRIGGER memories_fts_au")
        other._conn.commit()
        other._reembed_or_clear = _Recorder()
        other.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
        assert _fts_hits(other, '"2031"') == set()  # without the trigger the index does not follow: the probe sees it
    finally:
        other.close()


# --- a fading row keeps its origin date through an unfade ----------------------------------------------------------


def test_a_fade_a_write_back_and_an_unfade_keep_the_origin_date(store):
    memory_id = _make(store, {"joy": 8.0}, content="Bob likes tea with honey every morning")
    store.fade(memory_id, summary="Bob likes tea")
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    fading = _raw(store, memory_id)
    assert fading["content"] == _prefix("2031-03-04") + "Bob likes tea"
    assert fading["content_snapshot"] == _prefix("2031-03-04") + "Bob likes tea with honey every morning"
    store.unfade(memory_id)
    restored = _raw(store, memory_id)
    assert restored["content"] == _prefix("2031-03-04") + "Bob likes tea with honey every morning"
    assert restored["created_at"] == "2031-05-06T07:08:09+00:00"
    assert restored["content_snapshot"] is None


def test_a_snapshot_that_already_starts_with_a_date_is_left_alone_and_the_summary_is_judged_on_its_own(store):
    dated = _prefix("1999-12-31")
    memory_id = _make(store, {"joy": 8.0}, content=dated + "Bob likes tea with honey")
    store.fade(memory_id, summary="Bob likes tea")  # a summary with no date, a snapshot that has one
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    raw = _raw(store, memory_id)
    assert raw["content_snapshot"] == dated + "Bob likes tea with honey"
    assert raw["content"] == _prefix("2031-03-04") + "Bob likes tea"


def test_a_summary_that_already_starts_with_a_date_is_left_alone_and_the_snapshot_gets_the_prefix(store):
    memory_id = _make(store, {"joy": 8.0}, content="Bob likes tea with honey")
    store.fade(memory_id, summary=_prefix("1999-12-31") + "Bob likes tea")
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    raw = _raw(store, memory_id)
    assert raw["content"] == _prefix("1999-12-31") + "Bob likes tea"
    assert raw["content_snapshot"] == _prefix("2031-03-04") + "Bob likes tea with honey"


def test_a_change_to_the_snapshot_alone_keeps_the_vector_and_does_not_re_embed(store):
    memory_id = _make(store, {"joy": 8.0}, content="Bob likes tea with honey", embed=True)
    store.fade(memory_id, summary=_prefix("1999-12-31") + "Bob likes tea")  # the summary is dated, the snapshot is not
    store.embed_row(memory_id, _prefix("1999-12-31") + "Bob likes tea")  # a vector for the summary the row holds
    before = _raw(store, memory_id)
    assert before["embedding"] is not None
    recorder = _Recorder()
    store._reembed_or_clear = recorder
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    after = _raw(store, memory_id)
    assert after["content_snapshot"] == _prefix("2031-03-04") + "Bob likes tea with honey"  # the snapshot got its date
    assert after["content"] == before["content"]
    assert after["embedding"] == before["embedding"]  # the text the vector describes did not change
    assert after["embedding_model_id"] == before["embedding_model_id"]
    assert recorder.calls == []


def test_an_active_row_writes_no_snapshot(store):
    memory_id = _make(store, {"joy": 8.0})
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    assert _raw(store, memory_id)["content_snapshot"] is None


# --- the transaction's failure paths -------------------------------------------------------------------------------


def test_an_error_after_the_transaction_began_rolls_back_and_propagates(store, monkeypatch):
    memory_id = _make(store, {"joy": 8.0}, embed=True)
    before = _raw(store, memory_id)
    recorder = _Recorder()
    store._reembed_or_clear = recorder

    def boom(dt):
        raise OSError("no local time for this date")

    monkeypatch.setattr(engine_utils, "to_local", boom)
    with pytest.raises(OSError, match="no local time"):
        store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    assert store._conn.in_transaction is False
    assert _raw(store, memory_id) == before  # every column byte-identical
    assert recorder.calls == []


def test_a_connection_already_inside_a_transaction_is_refused(store):
    memory_id = _make(store, {"joy": 8.0})
    before = _raw(store, memory_id)
    store._conn.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(RuntimeError, match="transaction"):
            store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    finally:
        store._conn.rollback()
    assert _raw(store, memory_id) == before


def test_the_read_happens_inside_the_write_transaction_begun_first(store):
    memory_id = _make(store, {"joy": 8.0})
    statements: list[str] = []
    store._conn.set_trace_callback(statements.append)
    store._reembed_or_clear = _Recorder()
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    store._conn.set_trace_callback(None)
    begin = next(i for i, s in enumerate(statements) if s.strip().upper() == "BEGIN IMMEDIATE")
    select = next(i for i, s in enumerate(statements) if s.lstrip().upper().startswith("SELECT CONTENT, CONTENT_SNAPSHOT"))
    update = next(i for i, s in enumerate(statements) if s.lstrip().upper().startswith("UPDATE MEMORIES"))
    assert begin < select < update  # the row is read AFTER the write lock is taken, and written before the lock is released


def test_no_other_writer_can_get_in_between_the_read_and_the_write(store, monkeypatch):
    # The day conversion (`engine_utils.to_local`, through `day_of`) runs after the SELECT and before the UPDATE: from inside it, a second connection tries to take the write lock.
    memory_id = _make(store, {"joy": 8.0})
    outcomes: list[object] = []

    def to_local_that_probes_the_lock(dt):
        other = sqlite3.connect(store.db_path, timeout=0.2, isolation_level=None)
        try:
            other.execute("BEGIN IMMEDIATE")
            other.execute("ROLLBACK")
            outcomes.append("got the lock")
        except sqlite3.OperationalError as exc:
            outcomes.append(str(exc))
        finally:
            other.close()
        return dt.astimezone(UTC)

    monkeypatch.setattr(engine_utils, "to_local", to_local_that_probes_the_lock)
    store._reembed_or_clear = _Recorder()
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    assert len(outcomes) == 1
    assert "locked" in outcomes[0]  # the store held the write lock between its read and its write


def test_a_busy_store_raises_and_leaves_the_row_unchanged(store):
    memory_id = _make(store, {"joy": 8.0})
    before = _raw(store, memory_id)
    store._conn.execute("PRAGMA busy_timeout = 100")  # the real wait is 30 s: lowered to keep the cell fast
    other = sqlite3.connect(store.db_path, timeout=0.2, isolation_level=None)
    try:
        other.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError):
            store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
        other.execute("ROLLBACK")
    finally:
        other.close()
    assert store._conn.in_transaction is False
    assert _raw(store, memory_id) == before


def test_the_row_is_written_from_the_text_it_holds_not_from_a_stale_copy(store):
    # The gate's copy of the row can be seconds old; a content edit that landed since is kept (and dated).
    memory_id = _make(store, {"joy": 8.0}, content="Bob likes tea")
    store.update(memory_id, content="Bob likes coffee")
    store.refresh_faded_emotions(memory_id, {"joy": 6.0}, _WRITTEN_AT)
    assert _raw(store, memory_id)["content"] == _prefix("2031-03-04") + "Bob likes coffee"

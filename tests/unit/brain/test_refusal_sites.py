"""A refused model call ("not now") leaves its item in place and writes nothing (C7 per-site, C38 d, C39, C60; D13, D24).

Each test drives one `generate()` caller with a provider that refuses: `ProviderAuthDeferred` (expired login) or
`ShutdownDeferred` (the bridge is stopping; the flag is set as the call is refused, as when a stop arrives while the job
holds its slot). The provider snapshots the persona directory at the moment of the call (the C60 comparison point) and
the test compares it with the directory after the caller has handled the refusal.

Files named in advance as exempt (C60): append-only logs (`*.log`, `*.log.jsonl`), lock files (`*.lock`, and the
compaction lock `*.compacting`, which the site holds across the call and releases after it), and SQLite's
own sidecars (`-wal`, `-shm`, `-journal`); a `.db` file is compared by its logical content (`iterdump`), because closing
a store checkpoints its write-ahead log into the file without any write by the caller.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge import cli_throttle, provider_auth
from brain.bridge.provider import LLMProvider

REFUSALS = ["auth", "shutdown"]


def _exempt(path: Path) -> bool:
    n = path.name
    return (
        n.endswith((".log", ".log.jsonl", ".lock", ".compacting", "-wal", "-shm", "-journal"))
        or "__pycache__" in path.parts
    )


def _content(path: Path):
    if path.suffix == ".db":
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                return tuple(con.iterdump())
            finally:
                con.close()
        except sqlite3.Error:
            pass
    return path.read_bytes()


def tree(root: Path) -> dict[str, object]:
    return {
        p.relative_to(root).as_posix(): _content(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and not _exempt(p)
    }


def _refusal(kind: str) -> Exception:
    if kind == "shutdown":
        cli_throttle.request_shutdown()
        return cli_throttle.ShutdownDeferred("bridge is stopping")
    return provider_auth.ProviderAuthDeferred("provider auth expired")


class Refusing(LLMProvider):
    """Refuses every `generate`; records the persona directory as it was at the call."""

    def __init__(self, kind: str, root: Path):
        self.kind = kind
        self.root = root
        self.calls = 0
        self.at_call: dict | None = None

    def generate(self, prompt: str, *, system: str | None = None, **_kw) -> str:
        self.calls += 1
        if self.at_call is None:
            self.at_call = tree(self.root)
        raise _refusal(self.kind)

    def name(self) -> str:
        return "refusing"

    def chat(self, messages, *, tools=None, options=None):  # pragma: no cover - never a chat site
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _clean():
    cli_throttle.reset()
    provider_auth.reset()
    yield
    cli_throttle.reset()
    provider_auth.reset()


@pytest.fixture
def persona(tmp_path) -> Path:
    pd = tmp_path / "nell"
    (pd / "active_conversations").mkdir(parents=True)
    return pd


def _refused_type(kind: str):
    return cli_throttle.ShutdownDeferred if kind == "shutdown" else provider_auth.ProviderAuthDeferred


def _assert_unchanged(provider: Refusing) -> None:
    assert provider.calls >= 1, "the refused call was never reached: the test does not exercise the site"
    assert tree(provider.root) == provider.at_call


OLD = "2026-01-01T10:00:00+00:00"


def _buffer(pd: Path, sid: str = "sess_a", n: int = 4, start: str = OLD) -> str:
    from brain.ingest.buffer import ingest_turn

    t0 = datetime.fromisoformat(start)
    for i in range(n):
        ingest_turn(
            pd,
            {
                "session_id": sid,
                "speaker": "user" if i % 2 == 0 else "assistant",
                "text": f"turn {i} about the garden and the rain",
                "ts": (t0 + timedelta(minutes=i)).isoformat(),
            },
        )
    return sid


def _stores(pd: Path):
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    return MemoryStore(pd / "memories.db", integrity_check=False), HebbianMatrix(pd / "hebbian.db")


# ---------------------------------------------------------------- ingest/extract.py (snapshot, finalize)


@pytest.mark.parametrize("kind", REFUSALS)
@pytest.mark.parametrize("sweep", ["snapshot", "finalize"])
def test_session_extraction_refusal_leaves_cursor_and_backoff(persona, kind, sweep):
    from brain.ingest import pipeline

    sid = _buffer(persona)
    store, hebbian = _stores(persona)
    prov = Refusing(kind, persona)
    def _sweep() -> None:
        if sweep == "snapshot":
            pipeline.snapshot_stale_sessions(persona, silence_minutes=0.0, store=store, hebbian=hebbian, provider=prov)
        else:
            pipeline.finalize_stale_sessions(
                persona, finalize_after_hours=0.0, store=store, hebbian=hebbian, provider=prov
            )

    try:
        if kind == "shutdown":
            # #456: the bridge is stopping: the sweep ends with the refusal.
            with pytest.raises(_refused_type(kind)):
                _sweep()
        else:
            # #470/#473 (merged): any other refusal is a deferred extraction, "not now": the sweep goes on, nothing kept
            # changes.
            _sweep()
    finally:
        store.close()
        hebbian.close()
    assert prov.calls == 1, "a refusal is not retried"
    _assert_unchanged(prov)
    from brain.ingest.buffer import read_backoff, read_cursor

    assert read_cursor(persona, sid) is None and read_backoff(persona, sid) is None


@pytest.mark.parametrize("kind", REFUSALS)
@pytest.mark.parametrize("entry", ["close", "snapshot"])
def test_refusal_inside_the_held_extraction_lock_releases_it(persona, kind, entry):
    """#470's close_session / extract_session_snapshot hold the session's extraction lock across the model call. A stop
    refusal raised inside that lock must leave it free (a later pass acquires it) and the session untouched; an auth
    refusal there is a deferred report, also with the lock free."""
    from brain.ingest import pipeline
    from brain.ingest.buffer import extract_lock_path, read_backoff, read_cursor
    from brain.utils.file_lock import file_lock

    sid = _buffer(persona)
    buf = persona / "active_conversations" / f"{sid}.jsonl"
    store, hebbian = _stores(persona)
    prov = Refusing(kind, persona)
    run = pipeline.close_session if entry == "close" else pipeline.extract_session_snapshot
    try:
        if kind == "shutdown":
            with pytest.raises(_refused_type(kind)):
                run(persona, sid, store=store, hebbian=hebbian, provider=prov)
        else:
            assert run(persona, sid, store=store, hebbian=hebbian, provider=prov).deferred is True
    finally:
        store.close()
        hebbian.close()
    assert prov.calls == 1
    with file_lock(extract_lock_path(persona, sid), blocking=False) as acquired:
        assert acquired, "the refusal left the session's extraction lock held"
    assert buf.exists()
    assert read_cursor(persona, sid) is None and read_backoff(persona, sid) is None
    _assert_unchanged(prov)


# ---------------------------------------------------------------- pass 2: monologue (extractor) and attunement (detector)


def _queue_monologue(pd: Path) -> None:
    from brain.chat import pass2_queue

    pass2_queue.enqueue(
        {
            "id": pass2_queue.new_record_id(),
            "kind": "monologue",
            "monologue_text": "I keep thinking about the rain.",
            "visible_reply": "It rained all day.",
            "recent_user_msgs": ["how was your day"],
        },
        persona_dir=pd,
    )


def _queue_attunement(pd: Path) -> None:
    from brain.chat import pass2_queue

    pass2_queue.enqueue(
        {
            "id": pass2_queue.new_record_id(),
            "kind": "attunement",
            "turn_id": "t1",
            "user_message": "I have been feeling quite tired after work lately honestly",
            "reply_text": "That sounds draining.",
            "buffer_slice": [{"id": "t1", "content": "I have been feeling quite tired after work lately honestly"}],
        },
        persona_dir=pd,
    )


def _patch_all_providers(monkeypatch, prov: Refusing) -> None:
    """Route every provider a site builds for itself to the refusing one."""
    import brain.bridge.model_tier as model_tier
    from brain.bridge.provider import ClaudeCliProvider

    monkeypatch.setattr(model_tier, "build_tier_provider", lambda *_a, **_k: prov)
    monkeypatch.setattr(ClaudeCliProvider, "generate", lambda self, *a, **k: prov.generate(*a, **k))


@pytest.mark.parametrize("kind", REFUSALS)
@pytest.mark.parametrize("record", ["monologue", "attunement"])
def test_pass2_refusal_keeps_the_record_at_the_queue_head(persona, monkeypatch, kind, record):
    from brain.chat import pass2_queue

    (_queue_monologue if record == "monologue" else _queue_attunement)(persona)
    before = pass2_queue._peek_head_locked(persona)
    prov = Refusing(kind, persona)
    _patch_all_providers(monkeypatch, prov)
    with pytest.raises(_refused_type(kind)):
        pass2_queue.drain_all_locked(persona)
    assert pass2_queue._peek_head_locked(persona) == before
    assert not (persona / "attunement_errors.jsonl").exists()  # no haiku_call_failed / error row
    _assert_unchanged(prov)


@pytest.mark.parametrize("kind", REFUSALS)
def test_attunement_refusal_leaves_the_daily_budget(persona, monkeypatch, kind):
    from brain.attunement import budget
    from brain.attunement.store import BufferTurn
    from brain.chat import tool_loop

    budget.consume_call(persona, now=datetime.now(UTC))  # an existing budget file for today
    path = budget._budget_path(persona)
    before = path.read_bytes()
    prov = Refusing(kind, persona)
    _patch_all_providers(monkeypatch, prov)
    with pytest.raises(_refused_type(kind)):
        tool_loop._run_attunement_pass2(
            persona, "t1", "a long enough message about work", "ok",
            [BufferTurn(id="t1", content="a long enough message about work")],
        )
    assert path.read_bytes() == before
    _assert_unchanged(prov)


def test_attunement_other_failure_is_charged_to_the_budget(persona, monkeypatch):
    """C39(d) pass-2 half: a call that was not refused but failed still costs one budget call."""
    from brain.attunement import budget
    from brain.attunement.store import BufferTurn
    from brain.bridge.provider import ClaudeCliProvider
    from brain.chat import tool_loop

    def boom(self, *a, **k):
        raise RuntimeError("cli exited 1")

    monkeypatch.setattr(ClaudeCliProvider, "generate", boom)
    now = datetime.now(UTC)
    remaining = budget.get_remaining(persona, now=now)
    tool_loop._run_attunement_pass2(
        persona, "t1", "a long enough message about work", "ok",
        [BufferTurn(id="t1", content="a long enough message about work")],
    )
    assert budget.get_remaining(persona, now=now) == remaining - 1


# ---------------------------------------------------------------- attunement backfill (C39 a-e)


class _Detector:
    """Counts calls and records the categories asked for; refuses from call `refuse_at` on (1-based)."""

    def __init__(self, persona: Path, kind: str, refuse_at: int | None, fail: bool = False):
        self.persona = persona
        self.kind = kind
        self.refuse_at = refuse_at
        self.fail = fail
        self.calls = 0
        self.categories: list[frozenset[str] | None] = []
        self.at_refusal: dict | None = None

    def __call__(self, *, buffer_slice, reply_text, only_categories=None):
        from brain.attunement.detector import _decline_output

        self.calls += 1
        self.categories.append(only_categories)
        if self.refuse_at is not None and self.calls >= self.refuse_at:
            self.at_refusal = tree(self.persona)
            raise _refusal(self.kind)
        if self.fail:
            raise RuntimeError("detector failed")
        return _decline_output(buffer_slice[-1].id)


def _user_turns(pd: Path, n: int) -> None:
    from brain.ingest.buffer import ingest_turn

    t0 = datetime.fromisoformat(OLD)
    for i in range(n):
        ingest_turn(
            pd,
            {
                "session_id": "sess_b",
                "speaker": "user",
                "text": f"distinct topic number {i} " + " ".join(f"w{i}x{j}" for j in range(6)),
                "ts": (t0 + timedelta(minutes=i)).isoformat(),
            },
        )
    # the backfill reads role/content rows
    path = pd / "active_conversations" / "sess_b.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    path.write_text(
        "".join(json.dumps({"id": f"u{i}", "role": "user", "content": r["text"], "ts": r["ts"]}) + "\n"
                for i, r in enumerate(rows)),
        encoding="utf-8",
    )


@pytest.fixture
def backfill_env(persona, monkeypatch):
    from brain.attunement import backfill

    monkeypatch.setattr("brain.body.session_hours.compute_active_session_hours", lambda *a, **k: 0.0)
    _user_turns(persona, 400)
    (persona / "attunement").mkdir(exist_ok=True)
    return backfill


def _run_detector_patch(monkeypatch, det: _Detector) -> None:
    import brain.attunement.detector as detector_mod

    monkeypatch.setattr(
        detector_mod,
        "run_detector",
        lambda *, buffer_slice, reply_text, only_categories=None, **_k: det(
            buffer_slice=buffer_slice, reply_text=reply_text, only_categories=only_categories
        ),
    )


@pytest.mark.parametrize("kind", REFUSALS)
def test_backfill_full_pass_refused_at_first_window_writes_nothing(backfill_env, persona, monkeypatch, kind):
    backfill = backfill_env
    det = _Detector(persona, kind, refuse_at=1)
    _run_detector_patch(monkeypatch, det)
    before = tree(persona)  # comparison point: before run_backfill (the start-of-run save is deferred)
    backfill.run_backfill(persona, delay_s=0)
    assert det.calls == 1  # no later window in this run
    assert tree(persona) == before


@pytest.mark.parametrize("kind", REFUSALS)
def test_backfill_full_pass_refused_later_keeps_state_of_the_last_good_window(
    backfill_env, persona, monkeypatch, kind
):
    backfill = backfill_env
    det = _Detector(persona, kind, refuse_at=3)
    _run_detector_patch(monkeypatch, det)
    backfill.run_backfill(persona, delay_s=0)
    assert det.calls == 3
    assert tree(persona) == det.at_refusal  # (a) state and budget as before the refused window's call


@pytest.mark.parametrize("kind", REFUSALS)
def test_supplementary_refused_at_first_window_keeps_the_complete_record(backfill_env, persona, monkeypatch, kind):
    backfill = backfill_env

    state = backfill._state_path(persona)
    state.write_text(
        json.dumps({
            "started_at": "2026-01-01T00:00:00Z", "total_windows": 3, "sampled_windows": 2, "processed_windows": 2,
            "patterns_emitted": 0, "status": "complete", "last_cursor": "window-1", "schema_version": "0.0.28",
        }),
        encoding="utf-8",
    )
    assert backfill.should_run_supplementary_backfill(persona)
    det = _Detector(persona, kind, refuse_at=1)
    _run_detector_patch(monkeypatch, det)
    before = tree(persona)
    backfill.run_supplementary_backfill(persona, delay_s=0)
    assert tree(persona) == before  # (b) the prior complete record at the old schema survives
    assert backfill.should_run_supplementary_backfill(persona) and not backfill.should_run_backfill(persona)


@pytest.mark.parametrize("kind", REFUSALS)
def test_supplementary_refused_after_some_windows_resumes_as_supplementary(backfill_env, persona, monkeypatch, kind):
    backfill = backfill_env
    from brain.attunement.schemas import SCHEMA_VERSION

    backfill._state_path(persona).write_text(
        json.dumps({
            "started_at": "2026-01-01T00:00:00Z", "total_windows": 3, "sampled_windows": 2, "processed_windows": 2,
            "patterns_emitted": 0, "status": "complete", "last_cursor": "window-1", "schema_version": "0.0.28",
        }),
        encoding="utf-8",
    )
    det = _Detector(persona, kind, refuse_at=3)
    _run_detector_patch(monkeypatch, det)
    backfill.run_supplementary_backfill(persona, delay_s=0)
    saved = json.loads(backfill._state_path(persona).read_text(encoding="utf-8"))
    assert saved["pass_kind"] == "supplementary" and saved["status"] == "running"
    cursor = saved["last_cursor"]
    assert cursor  # (c) the cursor of the last good window
    # next start: still the supplementary pass, resumed from the cursor, only the new categories
    cli_throttle.reset()
    provider_auth.reset()
    assert backfill.should_run_supplementary_backfill(persona) and not backfill.should_run_backfill(persona)
    det2 = _Detector(persona, kind, refuse_at=None)
    _run_detector_patch(monkeypatch, det2)
    final = backfill.run_supplementary_backfill(persona, delay_s=0)
    assert det2.calls >= 1
    assert all(c == backfill._NEW_CATEGORIES for c in det2.categories)
    assert final.status == "complete" and final.schema_version == SCHEMA_VERSION
    sampled = backfill.select_sample(backfill.window_buffer(backfill._read_buffer_turns(persona)))
    assert det2.calls == len(sampled) - backfill._cursor_index(sampled, cursor)


def test_backfill_other_failure_is_charged_to_the_budget(backfill_env, persona, monkeypatch):
    """C39(d) backfill half."""
    from brain.attunement import budget

    backfill = backfill_env
    det = _Detector(persona, "auth", refuse_at=None, fail=True)
    _run_detector_patch(monkeypatch, det)
    now = datetime.now(UTC)
    before = budget.get_remaining(persona, now=now)
    backfill.run_backfill(persona, delay_s=0, now_dt=now)
    assert det.calls >= 1
    assert budget.get_remaining(persona, now=now) == before - det.calls


def test_supplementary_budget_exhausted_defers_as_supplementary(backfill_env, persona, monkeypatch):
    """C39(e)."""
    from brain.attunement import budget

    backfill = backfill_env
    backfill._state_path(persona).write_text(
        json.dumps({
            "started_at": "2026-01-01T00:00:00Z", "total_windows": 3, "sampled_windows": 2, "processed_windows": 2,
            "patterns_emitted": 0, "status": "complete", "last_cursor": "window-1", "schema_version": "0.0.28",
        }),
        encoding="utf-8",
    )
    now = datetime.now(UTC)
    while budget.consume_call(persona, now=now):
        pass
    det = _Detector(persona, "auth", refuse_at=None)
    _run_detector_patch(monkeypatch, det)
    backfill.run_supplementary_backfill(persona, delay_s=0, now_dt=now)
    saved = json.loads(backfill._state_path(persona).read_text(encoding="utf-8"))
    assert det.calls == 0
    assert saved["status"] == "deferred_to_next_day" and saved["pass_kind"] == "supplementary"
    assert backfill.should_run_supplementary_backfill(persona) and not backfill.should_run_backfill(persona)
    det2 = _Detector(persona, "auth", refuse_at=None)
    _run_detector_patch(monkeypatch, det2)
    backfill.run_supplementary_backfill(persona, delay_s=0, now_dt=now + timedelta(days=1))
    assert det2.calls >= 1 and all(c == backfill._NEW_CATEGORIES for c in det2.categories)


# ---------------------------------------------------------------- compaction (#473)


def _compactable(pd: Path, sid: str = "sess_c", n: int = 6) -> str:
    from brain.ingest.buffer import read_session, write_cursor

    _buffer(pd, sid, n=n, start=(datetime.now(UTC) - timedelta(days=4)).isoformat())
    write_cursor(pd, sid, read_session(pd, sid)[-1]["ts"])
    return sid


@pytest.mark.parametrize("kind", REFUSALS)
@pytest.mark.parametrize("path", ["compact", "cascade", "tick"])
def test_compaction_refusal_stores_no_fold_or_stand_in(persona, monkeypatch, kind, path):
    from brain.chat import compaction

    sid = _compactable(persona)
    prov = Refusing(kind, persona)
    with pytest.raises(_refused_type(kind)):
        if path == "compact":
            compaction.compact_conversation(
                persona, sid, older_than=timedelta(0), fold_existing_summary=True, provider=prov, min_keep_tail=0
            )
        elif path == "cascade":
            compaction.cascade_conversation(persona, sid, provider=prov, min_keep_tail=0)
        else:
            import brain.bridge.supervisor as sup

            monkeypatch.setattr("brain.chat.compaction._ROLLOVER_QUIET_GAP", timedelta(days=365))
            monkeypatch.setattr(compaction, "cascade_conversation", lambda *a, **k: compaction.compact_conversation(
                persona, sid, older_than=timedelta(0), fold_existing_summary=True, provider=prov, min_keep_tail=0))
            sup._run_compaction_tick(persona, prov)
    _assert_unchanged(prov)


@pytest.mark.parametrize("kind", REFUSALS)
def test_chat_turn_compaction_refusal_skips_the_fold_and_the_turn_continues(persona, monkeypatch, kind):
    from brain.bridge.provider import ChatMessage
    from brain.chat import budget as chat_budget
    from brain.chat import compaction

    sid = _compactable(persona)
    prov = Refusing(kind, persona)
    monkeypatch.setattr(compaction, "build_compaction_provider", lambda _pd: prov)
    messages = [ChatMessage(role="system", content="sys")] + [
        ChatMessage(role="user" if i % 2 == 0 else "assistant", content="word " * 400) for i in range(12)
    ]
    out = chat_budget.apply_budget(
        messages, max_tokens=500, preserve_tail_msgs=2, provider=prov, persona_dir=persona, session_id=sid
    )
    assert out  # the turn continues with the in-prompt floor
    _assert_unchanged(prov)


# ---------------------------------------------------------------- calibration (relevance judge tiebreak)


@pytest.mark.parametrize("kind", REFUSALS)
def test_calibration_refusal_leaves_the_row_unlabeled(persona, kind):
    from brain.memory.relevance_judge import FakeRelevanceJudgeProvider, label_calibration_sample
    from brain.memory.store import Memory, MemoryStore

    store = MemoryStore(persona / "memories.db", integrity_check=False)
    try:
        mem = Memory.create_new(content="something in between", memory_type="conversation", domain="us")
        store.create(mem)
        store.log_calibration_sample(query="q", candidate_ids=[mem.id], reranker_scores=[0.0], reranker_model_id="m")
        judge = FakeRelevanceJudgeProvider(scores={("q", mem.content): 0.0})  # ambiguous: goes to the tiebreak
        prov = Refusing(kind, persona)
        with pytest.raises(_refused_type(kind)):
            label_calibration_sample(store, provider=prov, judge=judge)
        row = store._conn.execute("SELECT local_judge_label FROM calibration_log").fetchone()
        assert row["local_judge_label"] is None
    finally:
        store.close()
    _assert_unchanged(prov)


# ---------------------------------------------------------------- soul review


def _candidate(i: int) -> dict:
    return {
        "id": f"cand-{i}", "text": f"a meaningful moment {i}", "label": "test", "importance": 8.0,
        "queued_at": "2026-01-01T00:00:00+00:00", "source": "test", "status": "auto_pending",
    }


@pytest.mark.parametrize("kind", REFUSALS)
def test_soul_review_refusal_counts_no_failure_and_writes_nothing(persona, kind):
    from brain.memory.store import MemoryStore
    from brain.soul.review import review_pending_candidates
    from brain.soul.store import SoulStore

    (persona / "soul_candidates.jsonl").write_text(
        "".join(json.dumps(_candidate(i)) + "\n" for i in range(2)), encoding="utf-8"
    )
    store = MemoryStore(persona / "memories.db", integrity_check=False)
    soul = SoulStore(str(persona / "crystallizations.db"))
    prov = Refusing(kind, persona)
    try:
        with pytest.raises(_refused_type(kind)):
            review_pending_candidates(persona, store=store, soul_store=soul, provider=prov)
    finally:
        store.close()
        soul.close()
    assert prov.calls == 1
    _assert_unchanged(prov)


def test_soul_review_refusal_after_a_decision_keeps_that_decision(persona):
    """A refusal at the second candidate: the first candidate's decision (already crystallised) is saved, so it is
    not reviewed again; the refused candidate stays pending and unchanged."""
    from brain.memory.store import MemoryStore
    from brain.soul.review import review_pending_candidates
    from brain.soul.store import SoulStore

    (persona / "soul_candidates.jsonl").write_text(
        "".join(json.dumps(_candidate(i)) + "\n" for i in range(2)), encoding="utf-8"
    )

    class OneThenRefuse(Refusing):
        def generate(self, prompt, *, system=None, **_kw):
            if self.calls == 0:
                self.calls += 1
                return json.dumps({"decision": "reject", "love_type": "craft", "resonance": 5, "confidence": 9,
                                   "reasoning": "not this one", "why_it_matters": ""})
            return super().generate(prompt, system=system)

    store = MemoryStore(persona / "memories.db", integrity_check=False)
    soul = SoulStore(str(persona / "crystallizations.db"))
    try:
        with pytest.raises(provider_auth.ProviderAuthDeferred):
            review_pending_candidates(persona, store=store, soul_store=soul, provider=OneThenRefuse("auth", persona))
    finally:
        store.close()
        soul.close()
    rows = {r["id"]: r for r in map(json.loads, (persona / "soul_candidates.jsonl").read_text().splitlines())}
    assert rows["cand-0"]["status"] != "auto_pending"
    assert rows["cand-1"] == _candidate(1)


# ---------------------------------------------------------------- emotion backfill


@pytest.mark.parametrize("kind", REFUSALS)
def test_emotion_backfill_refusal_leaves_state_cursor_and_budget(persona, kind):
    from brain.ingest import emotion_backfill
    from brain.memory.store import Memory, MemoryStore

    budget = emotion_backfill._budget_path(persona)
    store = MemoryStore(persona / "memories.db", integrity_check=False)
    try:
        for i in range(3):
            store.create(Memory.create_new(content=f"memory {i}", memory_type="conversation", domain="us"))
        prov = Refusing(kind, persona)
        with pytest.raises(_refused_type(kind)):
            emotion_backfill.run_emotion_backfill(persona, provider=prov, store=store, delay_s=0)
    finally:
        store.close()
    assert prov.calls == 1
    assert not budget.exists()  # the refused call was not charged
    _assert_unchanged(prov)


# ---------------------------------------------------------------- self-model articulation


@pytest.mark.parametrize("kind", REFUSALS)
def test_articulate_refusal_raises_and_leaves_the_budget(persona, kind):
    from brain.self_model.articulate import articulate
    from brain.self_model.gap import Gap

    prov = Refusing(kind, persona)
    with pytest.raises(_refused_type(kind)):
        articulate(Gap(per_channel={"joy": 1.0}, magnitude=1.0, unnamed_pressure=0.0), provider=prov, persona_dir=persona)
    _assert_unchanged(prov)


@pytest.mark.parametrize("kind", REFUSALS)
def test_self_model_tick_refusal_leaves_state_and_cadence(persona, monkeypatch, kind):
    import brain.bridge.supervisor as sup
    from brain.self_model.gap import Gap

    monkeypatch.setattr(sup, "compute_gap", lambda *_a, **_k: Gap(per_channel={"joy": 1.0}, magnitude=1.0,
                                                                  unnamed_pressure=0.0))
    prov = Refusing(kind, persona)
    with pytest.raises(_refused_type(kind)):
        sup._run_self_model_tick(persona, provider=prov, event_bus=object())
    _assert_unchanged(prov)


# ---------------------------------------------------------------- growth crystallizers


@pytest.mark.parametrize("kind", REFUSALS)
def test_growth_creative_dna_refusal_persists_no_growth_run(persona, monkeypatch, kind):
    from brain.growth import scheduler
    from brain.memory.store import MemoryStore

    (persona / "persona_config.json").write_text("{}", encoding="utf-8")
    prov = Refusing(kind, persona)
    import brain.bridge.model_tier as model_tier

    monkeypatch.setattr(model_tier, "build_tier_provider", lambda *_a, **_k: prov)
    store = MemoryStore(persona / "memories.db", integrity_check=False)
    try:
        with pytest.raises(_refused_type(kind)):
            scheduler.run_growth_tick(persona, store, datetime.now(UTC))
    finally:
        store.close()
    _assert_unchanged(prov)


@pytest.mark.parametrize("kind", REFUSALS)
def test_heartbeat_growth_refusal_does_not_record_a_growth_run(persona, monkeypatch, kind):
    from brain.engines import heartbeat as hb

    engine = hb.HeartbeatEngine.__new__(hb.HeartbeatEngine)
    engine.interests_path = persona / "interests.json"
    engine.store = None

    def refuse(*_a, **_k):
        raise _refusal(kind)

    monkeypatch.setattr("brain.growth.scheduler.run_growth_tick", refuse)
    import dataclasses

    state = hb.HeartbeatState.fresh("test")
    state.last_growth_at = datetime(2026, 1, 1, tzinfo=UTC)
    last = state.last_growth_at
    cfg = dataclasses.replace(hb.HeartbeatConfig(), growth_enabled=True, growth_every_hours=0.0)
    added, ran, err = engine._try_run_growth(state, datetime.now(UTC), cfg, False)
    assert (added, ran) == (0, False) and err == "deferred"
    assert state.last_growth_at == last


@pytest.mark.parametrize("kind", REFUSALS)
def test_reflex_crystallizer_refusal_raises(persona, kind):
    from brain.growth.crystallizers import reflex as rc
    from brain.memory.store import MemoryStore

    store = MemoryStore(persona / "memories.db", integrity_check=False)
    prov = Refusing(kind, persona)
    try:
        with pytest.raises(_refused_type(kind)):
            rc.crystallize_reflex(
                store=store, provider=prov, persona_dir=persona, current_arcs=[], removed_arc_names=set(),
                persona_name="nell", now=datetime.now(UTC),
            )
    finally:
        store.close()
    _assert_unchanged(prov)


# ---------------------------------------------------------------- interest sweep, initiate review


@pytest.mark.parametrize("kind", REFUSALS)
def test_interest_sweep_refusal_raises(persona, kind):
    from brain.engines import interest_sweep
    from brain.memory.store import MemoryStore

    store = MemoryStore(persona / "memories.db", integrity_check=False)
    prov = Refusing(kind, persona)
    defaults = Path(interest_sweep.__file__).resolve().parent / "default_interests.json"
    try:
        with pytest.raises(_refused_type(kind)):
            interest_sweep.run_sweep_tick(
                store=store, provider=prov, interests_path=persona / "interests.json",
                default_interests_path=defaults, now=datetime.now(UTC),
            )
    finally:
        store.close()
    _assert_unchanged(prov)


@pytest.mark.parametrize("kind", REFUSALS)
def test_initiate_review_refusal_is_re_raised(persona, kind):
    from brain.initiate.review import _make_haiku_call

    prov = Refusing(kind, persona)
    with pytest.raises(_refused_type(kind)):
        _make_haiku_call(prov)(system="s", user="u")
    _assert_unchanged(prov)


# ---------------------------------------------------------------- heartbeat-stage engines and vocab repair (C60 rows
# the plan marked "none expected": the refusal propagates to the stage's handler and nothing is written on the way)


class AnswerThenRefuse(Refusing):
    """Answers the first len(answers) calls, then refuses."""

    def __init__(self, kind, root, answers):
        super().__init__(kind, root)
        self.answers = list(answers)

    def generate(self, prompt, *, system=None, **kw):
        if self.answers:
            return self.answers.pop(0)
        return super().generate(prompt, system=system)


def _seeded_store(pd: Path):
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore

    store = MemoryStore(pd / "memories.db", integrity_check=False)
    store.create(Memory.create_new(content="a seed about the sea", memory_type="conversation", domain="us",
                                   emotions={"love": 8.0}))
    return store, HebbianMatrix(pd / "hebbian.db")


@pytest.mark.parametrize("kind", REFUSALS)
def test_dream_refusal_writes_nothing(persona, kind):
    from brain.engines.dream import DreamEngine

    store, hebbian = _seeded_store(persona)
    prov = Refusing(kind, persona)
    try:
        with pytest.raises(_refused_type(kind)):
            DreamEngine(store=store, hebbian=hebbian, provider=prov, log_path=persona / "dreams.log.jsonl",
                        persona_dir=persona, persona_name="Nell", persona_system_prompt="You are Nell.").run_cycle()
    finally:
        store.close()
        hebbian.close()
    _assert_unchanged(prov)


@pytest.mark.parametrize("kind", REFUSALS)
def test_reflex_refusal_writes_nothing(persona, kind):
    from brain.engines.reflex import ReflexEngine

    arcs = persona / "reflex_arcs.json"
    arcs.write_text(json.dumps({"version": 1, "arcs": [{
        "name": "test_arc", "description": "d", "trigger": {"love": 5}, "days_since_human_min": 0,
        "cooldown_hours": 1.0, "action": "a", "output_memory_type": "reflex_journal", "prompt_template": "Hi.",
    }]}), encoding="utf-8")
    store, hebbian = _seeded_store(persona)
    prov = Refusing(kind, persona)
    defaults = Path(__file__).resolve().parents[3] / "brain" / "engines" / "default_reflex_arcs.json"
    try:
        with pytest.raises(_refused_type(kind)):
            ReflexEngine(store=store, provider=prov, persona_name="Nell", persona_system_prompt="You are Nell.",
                         arcs_path=arcs, log_path=persona / "reflex_log.json", default_arcs_path=defaults).run_tick()
    finally:
        store.close()
        hebbian.close()
    _assert_unchanged(prov)


def _research_engine(pd: Path, store, prov):
    from brain.engines.research import ResearchEngine
    from brain.search.base import NoopWebSearcher

    return ResearchEngine(
        store=store, provider=prov, searcher=NoopWebSearcher(), persona_name="Nell",
        persona_system_prompt="You are Nell.", interests_path=pd / "interests.json",
        research_log_path=pd / "research_log.json",
        default_interests_path=pd / "no_default_interests.json",
    )


@pytest.mark.parametrize("kind", REFUSALS)
@pytest.mark.parametrize("step", ["seed", "select", "fire"])
def test_research_refusal_writes_nothing(persona, kind, step):
    store, hebbian = _seeded_store(persona)
    persona.joinpath("no_default_interests.json").write_text(json.dumps({"interests": []}), encoding="utf-8")
    persona.joinpath("voice.md").write_text("Nell is curious about the sea and old maps.", encoding="utf-8")
    if step != "seed":
        persona.joinpath("interests.json").write_text(json.dumps({"version": 1, "interests": [
            {"id": f"i{i}", "topic": f"topic {i}", "pull_score": 9.0, "scope": "either", "related_keywords": [],
             "notes": "", "first_seen": "2026-01-01T00:00:00+00:00", "last_fed": "2026-01-01T00:00:00+00:00",
             "last_researched_at": None, "feed_count": 3, "source_types": ["conversation"]}
            for i in range(2)]}), encoding="utf-8")
    answers = ['{"choice": "i0", "why": "it pulls"}'] if step == "fire" else []
    prov = AnswerThenRefuse(kind, persona, answers)
    try:
        engine = _research_engine(persona, store, prov)
        if step == "seed":
            engine.run_tick(days_since_human_override=10.0)  # the seed step catches and reports no interests
        else:
            with pytest.raises(_refused_type(kind)):
                engine.run_tick(days_since_human_override=10.0, emotion_state_override=None)
    finally:
        store.close()
        hebbian.close()
    _assert_unchanged(prov)


@pytest.mark.parametrize("kind", REFUSALS)
def test_research_topic_overlap_refusal_raises(persona, kind):
    from brain.engines.research import _compute_topic_overlap_via_haiku

    prov = Refusing(kind, persona)
    with pytest.raises(_refused_type(kind)):
        _compute_topic_overlap_via_haiku(thread_topic="t", thread_summary="s", recent_conversation_excerpt="e",
                                         provider=prov, user_name="Bob", persona_dir=persona)
    _assert_unchanged(prov)


@pytest.mark.parametrize("kind", REFUSALS)
def test_vocab_repair_refusal_writes_nothing(persona, kind):
    from brain.health import vocab_repair
    from brain.health.reconstruct import PLACEHOLDER_DESCRIPTION
    from brain.memory.store import MemoryStore

    (persona / "emotion_vocabulary.json").write_text(json.dumps({"version": 1, "emotions": [
        {"name": "wistful_hope", "description": PLACEHOLDER_DESCRIPTION, "category": "persona_extension",
         "decay_half_life_days": 14.0, "intensity_clamp": 10.0}]}), encoding="utf-8")
    store = MemoryStore(persona / "memories.db", integrity_check=False)
    prov = Refusing(kind, persona)
    try:
        with pytest.raises(_refused_type(kind)):
            vocab_repair.run_vocab_repair(persona, store=store, provider=prov)
    finally:
        store.close()
    _assert_unchanged(prov)

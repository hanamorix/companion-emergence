# Full-Session Context + Sticky Sessions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 20-turn rolling window with full-buffer prompt construction; split the 5-minute extraction sweep from session lifecycle; finalize sessions only at 24h silence or explicit close; add a budget guard for the multi-hour edge case.

**Architecture:** Buffer file becomes the source of truth for in-session context. New cursor sidecar (`<sid>.cursor`) lets snapshot passes extract only newly-added turns without destroying state. Supervisor runs two cadences: minute-tick `snapshot_stale_sessions` (5-min silence → memorise, keep alive) and hourly `finalize_stale_sessions` (24h silence → real close). Engine reads buffer, optional budget guard compresses head when prompt exceeds 190K estimated tokens.

**Tech Stack:** Python 3.13, pytest, existing brain modules (`brain.ingest.*`, `brain.chat.*`, `brain.bridge.supervisor`). No new third-party dependencies.

**Source spec:** `docs/superpowers/specs/2026-05-10-full-session-context-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `brain/ingest/buffer.py` | Modify | Add cursor sidecar helpers + `read_session_after`. |
| `brain/ingest/pipeline.py` | Modify | Add `extract_session_snapshot`, `snapshot_stale_sessions`, `finalize_stale_sessions`. Rewire `close_session` to clear cursor on exit. Deprecate `close_stale_sessions` (callers move to snapshot/finalize). |
| `brain/chat/budget.py` | Create | `apply_budget(messages, *, max_tokens, preserve_tail_msgs, provider) -> list[ChatMessage]`. |
| `brain/chat/engine.py` | Modify | Build messages list from buffer; pass through `apply_budget` before tool loop. |
| `brain/chat/session.py` | Modify | Docstring update demoting `HISTORY_MAX_TURNS`; raise ceiling to a sanity-only value. |
| `brain/bridge/supervisor.py` | Modify | Swap destructive sweep for `snapshot_stale_sessions`; add hourly `finalize_stale_sessions` cadence; publish new event types. |
| `tests/unit/brain/ingest/test_buffer.py` | Modify | Add cursor + `read_session_after` cases. |
| `tests/unit/brain/ingest/test_pipeline.py` | Modify | Add `extract_session_snapshot`, `snapshot_stale_sessions`, `finalize_stale_sessions` cases; assert `close_session` clears cursor. |
| `tests/unit/brain/chat/test_budget.py` | Create | Unit tests for `apply_budget`. |
| `tests/unit/brain/chat/test_engine.py` | Modify | Add buffer-driven prompt, image replay, budget pass-through, buffer-read-failure fallback. |
| `tests/unit/brain/bridge/test_supervisor.py` | Modify | Snapshot sweep keeps buffer + registry; finalize cadence fires; new events. |
| `tests/integration/brain/bridge/test_sticky_session.py` | Create | 50-turn session → 5-min sweep → return → assert full transcript in prompt. |

---

## Phase 1 — Buffer cursor primitives

### Task 1: Cursor read/write/delete + `read_session_after`

**Files:**
- Modify: `brain/ingest/buffer.py`
- Test: `tests/unit/brain/ingest/test_buffer.py`

- [ ] **Step 1: Add failing tests for cursor primitives**

Append to `tests/unit/brain/ingest/test_buffer.py`:

```python
import pytest

from brain.ingest.buffer import (
    delete_cursor,
    ingest_turn,
    read_cursor,
    read_session_after,
    write_cursor,
)


def test_write_and_read_cursor_roundtrip(tmp_path: Path) -> None:
    ingest_turn(tmp_path, {"session_id": "sess_abc", "speaker": "user", "text": "hi"})
    write_cursor(tmp_path, "sess_abc", "2026-05-10T20:00:00+00:00")
    assert read_cursor(tmp_path, "sess_abc") == "2026-05-10T20:00:00+00:00"


def test_read_cursor_missing_returns_none(tmp_path: Path) -> None:
    assert read_cursor(tmp_path, "sess_abc") is None


def test_read_cursor_malformed_returns_none(tmp_path: Path) -> None:
    (tmp_path / "active_conversations").mkdir(parents=True)
    (tmp_path / "active_conversations" / "sess_abc.cursor").write_text(
        "not-a-timestamp", encoding="utf-8"
    )
    assert read_cursor(tmp_path, "sess_abc") is None


def test_write_cursor_rejects_malformed_ts(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_cursor(tmp_path, "sess_abc", "garbage")


def test_delete_cursor_is_idempotent(tmp_path: Path) -> None:
    delete_cursor(tmp_path, "sess_abc")  # missing — must not raise
    write_cursor(tmp_path, "sess_abc", "2026-05-10T20:00:00+00:00")
    delete_cursor(tmp_path, "sess_abc")
    assert read_cursor(tmp_path, "sess_abc") is None


def test_read_session_after_returns_only_post_cursor_turns(tmp_path: Path) -> None:
    sid = "sess_abc"
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "a",
                           "ts": "2026-05-10T20:00:00+00:00"})
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "assistant", "text": "b",
                           "ts": "2026-05-10T20:00:05+00:00"})
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "c",
                           "ts": "2026-05-10T20:01:00+00:00"})
    out = read_session_after(tmp_path, sid, "2026-05-10T20:00:30+00:00")
    assert [t["text"] for t in out] == ["c"]


def test_read_session_after_none_cursor_returns_all(tmp_path: Path) -> None:
    sid = "sess_abc"
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "a",
                           "ts": "2026-05-10T20:00:00+00:00"})
    out = read_session_after(tmp_path, sid, None)
    assert len(out) == 1


def test_read_session_after_malformed_cursor_returns_all(tmp_path: Path) -> None:
    sid = "sess_abc"
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "a",
                           "ts": "2026-05-10T20:00:00+00:00"})
    out = read_session_after(tmp_path, sid, "not-a-ts")
    assert len(out) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/brain/ingest/test_buffer.py -v`
Expected: All seven new tests FAIL with `ImportError: cannot import name 'read_cursor'` (and siblings).

- [ ] **Step 3: Implement cursor primitives in `brain/ingest/buffer.py`**

Append to `brain/ingest/buffer.py` (after `delete_session_buffer`):

```python
import os


def _cursor_path(persona_dir: Path, session_id: str) -> Path:
    """Resolve <persona>/active_conversations/<session_id>.cursor.

    Same session_id validation as _session_path so the cursor file lands
    inside the active_conversations dir, never traversed.
    """
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError(
            f"invalid session_id {session_id!r} — must match "
            f"[A-Za-z0-9_-]{{1,64}}"
        )
    return _active_conversations_dir(persona_dir) / f"{session_id}.cursor"


def read_cursor(persona_dir: Path, session_id: str) -> str | None:
    """Return the cursor ts (ISO string) for a session.

    Returns None when the cursor file is missing, empty, or its content
    doesn't parse as an ISO-8601 timestamp. Callers treat None as
    "extract from the beginning."
    """
    path = _cursor_path(persona_dir, session_id)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        datetime.fromisoformat(text.replace("Z", "+00:00"))
        return text
    except (OSError, ValueError):
        return None


def write_cursor(persona_dir: Path, session_id: str, ts: str) -> None:
    """Atomically write the cursor file. Raises ValueError on bad ts."""
    datetime.fromisoformat(ts.replace("Z", "+00:00"))
    path = _cursor_path(persona_dir, session_id)
    tmp = path.with_suffix(".cursor.tmp")
    tmp.write_text(ts, encoding="utf-8")
    os.replace(tmp, path)


def delete_cursor(persona_dir: Path, session_id: str) -> None:
    """Idempotent unlink of the cursor file."""
    path = _cursor_path(persona_dir, session_id)
    path.unlink(missing_ok=True)


def read_session_after(
    persona_dir: Path, session_id: str, after_ts: str | None
) -> list[dict]:
    """Return turns whose ts > after_ts.

    after_ts=None returns all turns. Malformed after_ts also returns all
    turns (logged at caller layer). Turns whose ts is missing or unparseable
    are skipped silently.
    """
    turns = read_session(persona_dir, session_id)
    if after_ts is None:
        return turns
    try:
        cutoff = datetime.fromisoformat(after_ts.replace("Z", "+00:00"))
    except ValueError:
        return turns
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=UTC)
    out: list[dict] = []
    for t in turns:
        raw = t.get("ts")
        if not raw:
            continue
        try:
            t_dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if t_dt.tzinfo is None:
            t_dt = t_dt.replace(tzinfo=UTC)
        if t_dt > cutoff:
            out.append(t)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/brain/ingest/test_buffer.py -v`
Expected: All tests PASS, including the new seven.

- [ ] **Step 5: Commit**

```bash
git add brain/ingest/buffer.py tests/unit/brain/ingest/test_buffer.py
git commit -m "feat(ingest): add cursor sidecar + read_session_after helpers"
```

---

## Phase 2 — Snapshot extraction

### Task 2: `extract_session_snapshot` — non-destructive ingest

**Files:**
- Modify: `brain/ingest/pipeline.py`
- Test: `tests/unit/brain/ingest/test_pipeline.py`

- [ ] **Step 1: Add failing tests for `extract_session_snapshot`**

Append to `tests/unit/brain/ingest/test_pipeline.py` (imports as needed at top):

```python
from brain.ingest.buffer import (
    delete_cursor,
    ingest_turn,
    read_cursor,
    write_cursor,
)
from brain.ingest.pipeline import extract_session_snapshot


def test_snapshot_preserves_buffer_and_writes_cursor(
    tmp_path: Path, fake_provider, mem_store, hebbian_store
) -> None:
    sid = "sess_abc"
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "I love watercolour"})
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "assistant", "text": "noted"})

    report = extract_session_snapshot(
        tmp_path, sid, store=mem_store, hebbian=hebbian_store, provider=fake_provider,
    )

    buf = tmp_path / "active_conversations" / f"{sid}.jsonl"
    assert buf.exists(), "snapshot must NOT delete the buffer"
    assert report.session_id == sid
    cursor = read_cursor(tmp_path, sid)
    assert cursor is not None, "snapshot must write the cursor"


def test_snapshot_with_cursor_only_extracts_new_turns(
    tmp_path: Path, fake_provider, mem_store, hebbian_store
) -> None:
    sid = "sess_abc"
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "old",
                           "ts": "2026-05-10T20:00:00+00:00"})
    write_cursor(tmp_path, sid, "2026-05-10T20:00:00+00:00")
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "new",
                           "ts": "2026-05-10T20:05:00+00:00"})

    fake_provider.reset_calls()
    extract_session_snapshot(
        tmp_path, sid, store=mem_store, hebbian=hebbian_store, provider=fake_provider,
    )

    assert fake_provider.last_transcript is not None
    assert "new" in fake_provider.last_transcript
    assert "old" not in fake_provider.last_transcript


def test_snapshot_with_no_new_turns_skips_provider_call(
    tmp_path: Path, fake_provider, mem_store, hebbian_store
) -> None:
    sid = "sess_abc"
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "x",
                           "ts": "2026-05-10T20:00:00+00:00"})
    write_cursor(tmp_path, sid, "2026-05-10T20:00:00+00:00")

    fake_provider.reset_calls()
    report = extract_session_snapshot(
        tmp_path, sid, store=mem_store, hebbian=hebbian_store, provider=fake_provider,
    )

    assert fake_provider.call_count == 0
    assert report.extracted == 0


def test_snapshot_malformed_cursor_falls_back_to_full(
    tmp_path: Path, fake_provider, mem_store, hebbian_store
) -> None:
    sid = "sess_abc"
    (tmp_path / "active_conversations").mkdir(parents=True, exist_ok=True)
    (tmp_path / "active_conversations" / f"{sid}.cursor").write_text("garbage")
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "hello"})

    fake_provider.reset_calls()
    extract_session_snapshot(
        tmp_path, sid, store=mem_store, hebbian=hebbian_store, provider=fake_provider,
    )

    assert "hello" in fake_provider.last_transcript


def test_snapshot_on_empty_buffer_returns_empty_report(
    tmp_path: Path, fake_provider, mem_store, hebbian_store
) -> None:
    sid = "sess_abc"
    (tmp_path / "active_conversations").mkdir(parents=True, exist_ok=True)
    (tmp_path / "active_conversations" / f"{sid}.jsonl").touch()

    fake_provider.reset_calls()
    report = extract_session_snapshot(
        tmp_path, sid, store=mem_store, hebbian=hebbian_store, provider=fake_provider,
    )

    assert report.extracted == 0
    assert fake_provider.call_count == 0
```

If `fake_provider` / `mem_store` / `hebbian_store` fixtures don't exist or don't track `call_count` / `last_transcript`, extend the existing fixture file (`tests/unit/brain/ingest/conftest.py` if present, otherwise the existing fake provider used by `test_pipeline.py`). Mirror the shape already in use in this test module.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/brain/ingest/test_pipeline.py -v -k snapshot`
Expected: All five new tests FAIL with `ImportError: cannot import name 'extract_session_snapshot'`.

- [ ] **Step 3: Implement `extract_session_snapshot` in `brain/ingest/pipeline.py`**

Add to `brain/ingest/pipeline.py` after `close_session`:

```python
def extract_session_snapshot(
    persona_dir: Path,
    session_id: str,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    embeddings: EmbeddingCache | None = None,
    config: dict | None = None,
) -> IngestReport:
    """Run BUFFER → EXTRACT → SCORE → DEDUPE → COMMIT → SOUL → LOG without
    deleting the buffer.

    Uses a per-session cursor sidecar to extract only turns added since the
    previous successful snapshot. On a successful pass, advances the cursor
    to the ts of the last extracted turn. On extraction failure, the cursor
    is unchanged so the next pass retries the same turns.

    Returns an IngestReport with the same shape as close_session.
    """
    from brain.ingest.buffer import (
        read_cursor,
        read_session_after,
        write_cursor,
    )

    cfg = config or {}
    report = IngestReport(session_id=session_id)

    cursor = read_cursor(persona_dir, session_id)
    turns = read_session_after(persona_dir, session_id, cursor)

    if not turns:
        logger.info(
            "conversation_snapshot_empty session=%s cursor=%s",
            session_id,
            cursor,
        )
        return report

    user_name = _load_user_name(persona_dir)
    assistant_name = persona_dir.name
    transcript = format_transcript(
        turns,
        max_tokens=int(cfg.get("max_transcript_tokens", 6000)),
        user_name=user_name,
        assistant_name=assistant_name,
    )
    extraction = extract_items_with_status(
        transcript,
        provider=provider,
        max_retries=int(cfg.get("extraction_max_retries", 1)),
        user_name=user_name,
        assistant_name=assistant_name,
    )
    if extraction.failed:
        report.errors += 1
        logger.warning(
            "conversation_snapshot_failed session=%s turns=%d error=%s; cursor unchanged",
            session_id,
            len(turns),
            extraction.error or "unknown extraction failure",
        )
        return report

    items = [it.normalize() for it in extraction.items if it.text]
    report.extracted = len(items)

    dedup_threshold = float(cfg.get("dedup_threshold", DEFAULT_DEDUP_THRESHOLD))
    crystallize_threshold = int(cfg.get("crystallize_threshold", DEFAULT_SOUL_THRESHOLD))

    for item in items:
        if is_duplicate(item.text, store=store, threshold=dedup_threshold, embeddings=embeddings):
            report.deduped += 1
            continue
        mem_id = commit_item(item, session_id=session_id, store=store, hebbian=hebbian)
        if mem_id is None:
            report.errors += 1
            continue
        report.committed += 1
        report.memory_ids.append(mem_id)
        if item.importance >= crystallize_threshold:
            queued = queue_soul_candidate(
                persona_dir, memory_id=mem_id, item=item, session_id=session_id,
            )
            if queued:
                report.soul_candidates += 1
            else:
                report.soul_queue_errors += 1

    # Advance cursor to the ts of the last turn we included. If the last
    # turn lacks a ts (shouldn't happen — ingest_turn writes one — but be
    # defensive), leave the cursor unchanged so the next pass retries.
    last_ts = turns[-1].get("ts")
    if last_ts:
        try:
            write_cursor(persona_dir, session_id, str(last_ts))
        except (OSError, ValueError):
            logger.warning(
                "conversation_snapshot_cursor_write_failed session=%s ts=%s",
                session_id,
                last_ts,
            )

    logger.info(
        "conversation_snapshot session=%s turns=%d extracted=%d committed=%d "
        "deduped=%d soul_candidates=%d cursor=%s",
        session_id,
        len(turns),
        report.extracted,
        report.committed,
        report.deduped,
        report.soul_candidates,
        last_ts,
    )
    return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/brain/ingest/test_pipeline.py -v -k snapshot`
Expected: All five new tests PASS.

- [ ] **Step 5: Commit**

```bash
git add brain/ingest/pipeline.py tests/unit/brain/ingest/test_pipeline.py
git commit -m "feat(ingest): add extract_session_snapshot — non-destructive cursor-driven extract"
```

---

## Phase 3 — Snapshot + finalize sweeps

### Task 3: `snapshot_stale_sessions` (replaces destructive sweep)

**Files:**
- Modify: `brain/ingest/pipeline.py`
- Test: `tests/unit/brain/ingest/test_pipeline.py`

- [ ] **Step 1: Add failing test for `snapshot_stale_sessions`**

Append to `tests/unit/brain/ingest/test_pipeline.py`:

```python
from brain.ingest.pipeline import snapshot_stale_sessions


def test_snapshot_stale_sessions_keeps_buffer_alive(
    tmp_path: Path, fake_provider, mem_store, hebbian_store, freeze_time_5min_ago
) -> None:
    sid = "sess_abc"
    # freeze_time_5min_ago fixture (or equivalent) makes ingest_turn stamp
    # the turn with a ts 5+ min in the past.
    with freeze_time_5min_ago:
        ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "earlier"})

    reports = snapshot_stale_sessions(
        tmp_path,
        silence_minutes=5.0,
        store=mem_store,
        hebbian=hebbian_store,
        provider=fake_provider,
    )

    assert len(reports) == 1
    assert reports[0].session_id == sid
    buf = tmp_path / "active_conversations" / f"{sid}.jsonl"
    assert buf.exists(), "snapshot sweep must NOT delete the buffer"


def test_snapshot_stale_sessions_skips_fresh_sessions(
    tmp_path: Path, fake_provider, mem_store, hebbian_store
) -> None:
    sid = "sess_abc"
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "just now"})

    reports = snapshot_stale_sessions(
        tmp_path,
        silence_minutes=5.0,
        store=mem_store,
        hebbian=hebbian_store,
        provider=fake_provider,
    )

    assert reports == []
```

If a `freeze_time_5min_ago` fixture doesn't exist, ingest with an explicit `ts` 6 minutes in the past:

```python
from datetime import UTC, datetime, timedelta
old_ts = (datetime.now(UTC) - timedelta(minutes=6)).isoformat()
ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "earlier", "ts": old_ts})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/brain/ingest/test_pipeline.py -v -k snapshot_stale`
Expected: FAIL with `ImportError: cannot import name 'snapshot_stale_sessions'`.

- [ ] **Step 3: Implement `snapshot_stale_sessions` in `brain/ingest/pipeline.py`**

Add to `brain/ingest/pipeline.py`:

```python
def snapshot_stale_sessions(
    persona_dir: Path,
    *,
    silence_minutes: float = 5.0,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    embeddings: EmbeddingCache | None = None,
    config: dict | None = None,
) -> list[IngestReport]:
    """Iterate active sessions; snapshot any whose last turn is past silence_minutes.

    Non-destructive: buffer files and the in-memory _SESSIONS registry are
    left intact. The caller (supervisor) should NOT call remove_session()
    for sessions returned here.

    Returns reports for sessions where extract_session_snapshot ran (skips
    fresh sessions; skips ghost files with no readable turns).
    """
    reports: list[IngestReport] = []
    for sid in list_active_sessions(persona_dir):
        turns = read_session(persona_dir, sid)
        if not turns:
            # Ghost — clean it up without a report.
            delete_session_buffer(persona_dir, sid)
            continue
        age = session_silence_minutes(turns)
        if age >= silence_minutes:
            report = extract_session_snapshot(
                persona_dir,
                sid,
                store=store,
                hebbian=hebbian,
                provider=provider,
                embeddings=embeddings,
                config=config,
            )
            reports.append(report)
    return reports
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/brain/ingest/test_pipeline.py -v -k snapshot_stale`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add brain/ingest/pipeline.py tests/unit/brain/ingest/test_pipeline.py
git commit -m "feat(ingest): snapshot_stale_sessions — non-destructive periodic extract"
```

### Task 4: `finalize_stale_sessions` (24h real close)

**Files:**
- Modify: `brain/ingest/pipeline.py`
- Test: `tests/unit/brain/ingest/test_pipeline.py`

- [ ] **Step 1: Add failing tests for `finalize_stale_sessions`**

Append to `tests/unit/brain/ingest/test_pipeline.py`:

```python
from datetime import UTC, datetime, timedelta

from brain.ingest.pipeline import finalize_stale_sessions


def test_finalize_under_threshold_skips(
    tmp_path: Path, fake_provider, mem_store, hebbian_store
) -> None:
    sid = "sess_abc"
    recent = (datetime.now(UTC) - timedelta(hours=10)).isoformat()
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "x", "ts": recent})

    reports = finalize_stale_sessions(
        tmp_path,
        finalize_after_hours=24.0,
        store=mem_store, hebbian=hebbian_store, provider=fake_provider,
    )

    assert reports == []
    buf = tmp_path / "active_conversations" / f"{sid}.jsonl"
    assert buf.exists()


def test_finalize_at_threshold_deletes_buffer_and_cursor(
    tmp_path: Path, fake_provider, mem_store, hebbian_store
) -> None:
    sid = "sess_abc"
    old = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "x", "ts": old})
    write_cursor(tmp_path, sid, old)

    reports = finalize_stale_sessions(
        tmp_path,
        finalize_after_hours=24.0,
        store=mem_store, hebbian=hebbian_store, provider=fake_provider,
    )

    assert len(reports) == 1
    assert reports[0].session_id == sid
    buf = tmp_path / "active_conversations" / f"{sid}.jsonl"
    cursor_file = tmp_path / "active_conversations" / f"{sid}.cursor"
    assert not buf.exists(), "finalize must delete the buffer"
    assert not cursor_file.exists(), "finalize must delete the cursor"


def test_finalize_per_session_error_isolation(
    tmp_path: Path, mem_store, hebbian_store, exploding_provider
) -> None:
    """A provider that raises should not abort the whole sweep."""
    sid_a = "sess_aaa"
    sid_b = "sess_bbb"
    old = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    ingest_turn(tmp_path, {"session_id": sid_a, "speaker": "user", "text": "a", "ts": old})
    ingest_turn(tmp_path, {"session_id": sid_b, "speaker": "user", "text": "b", "ts": old})

    # exploding_provider raises on the first call only — fixture or simple lambda
    reports = finalize_stale_sessions(
        tmp_path,
        finalize_after_hours=24.0,
        store=mem_store, hebbian=hebbian_store, provider=exploding_provider,
    )

    # At least one of the two finalize attempts must have completed (the
    # one that didn't hit the explosion). The other returns a report with
    # errors>=1. Neither raised out of the loop.
    assert len(reports) == 2
```

`exploding_provider`: if no fixture exists, define inline in the test module — a stub provider whose `generate()` raises on first call then returns valid JSON.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/brain/ingest/test_pipeline.py -v -k finalize`
Expected: FAIL with `ImportError: cannot import name 'finalize_stale_sessions'`.

- [ ] **Step 3: Implement `finalize_stale_sessions` + cursor cleanup in `close_session`**

Modify `close_session` in `brain/ingest/pipeline.py`. Find the existing `delete_session_buffer(persona_dir, session_id)` near the end and add a cursor cleanup just before/after it:

```python
    # ── DELETE buffer + cursor sidecar ────────────────────────────────────────
    delete_session_buffer(persona_dir, session_id)
    from brain.ingest.buffer import delete_cursor
    delete_cursor(persona_dir, session_id)

    return report
```

Also handle the empty-session early-return path (turns is empty) — add `delete_cursor` next to its `delete_session_buffer`.

Append `finalize_stale_sessions` to `brain/ingest/pipeline.py`:

```python
def finalize_stale_sessions(
    persona_dir: Path,
    *,
    finalize_after_hours: float = 24.0,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    embeddings: EmbeddingCache | None = None,
    config: dict | None = None,
) -> list[IngestReport]:
    """Iterate active sessions; finalize any whose last turn is past
    ``finalize_after_hours``.

    "Finalize" = run one last snapshot extraction, then delete the buffer
    and the cursor sidecar. Caller (supervisor) follows up with
    remove_session() for each returned report.session_id.

    Per-session try/except so one bad session can't abort the loop.
    """
    from brain.ingest.buffer import delete_cursor

    threshold_minutes = finalize_after_hours * 60.0
    reports: list[IngestReport] = []
    for sid in list_active_sessions(persona_dir):
        turns = read_session(persona_dir, sid)
        if not turns:
            delete_session_buffer(persona_dir, sid)
            delete_cursor(persona_dir, sid)
            continue
        age = session_silence_minutes(turns)
        if age < threshold_minutes:
            continue
        try:
            report = extract_session_snapshot(
                persona_dir, sid,
                store=store, hebbian=hebbian, provider=provider,
                embeddings=embeddings, config=config,
            )
        except Exception:
            logger.exception("finalize_stale_sessions: snapshot failed session=%s", sid)
            report = IngestReport(session_id=sid)
            report.errors += 1
        # Whether snapshot succeeded or failed, the session has been silent
        # past the finalize threshold — drop it. Memories already in
        # MemoryStore from earlier snapshots stay. Buffer leaving means we
        # accept the loss of any turns that failed to ingest.
        delete_session_buffer(persona_dir, sid)
        delete_cursor(persona_dir, sid)
        reports.append(report)
        logger.info(
            "conversation_finalized session=%s silence_hours=%.2f",
            sid,
            age / 60.0,
        )
    return reports
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/brain/ingest/test_pipeline.py -v`
Expected: All tests PASS (including existing close_session tests with the cursor cleanup; if any close_session test asserts on file state, it should still pass because it created no cursor).

- [ ] **Step 5: Commit**

```bash
git add brain/ingest/pipeline.py tests/unit/brain/ingest/test_pipeline.py
git commit -m "feat(ingest): finalize_stale_sessions + close_session cursor cleanup"
```

---

## Phase 4 — Supervisor rewiring

### Task 5: Replace destructive sweep + add hourly finalize cadence

**Files:**
- Modify: `brain/bridge/supervisor.py`
- Test: `tests/unit/brain/bridge/test_supervisor.py`

- [ ] **Step 1: Add failing tests for supervisor's new behaviour**

Append to `tests/unit/brain/bridge/test_supervisor.py`:

```python
def test_supervisor_snapshot_sweep_keeps_session_alive(
    tmp_path: Path, fake_provider, fake_event_bus
) -> None:
    """After a snapshot sweep, the session must remain in _SESSIONS and
    its buffer file on disk."""
    from brain.chat.session import create_session, get_session, reset_registry
    from brain.bridge.supervisor import run_folded
    from brain.ingest.buffer import ingest_turn
    import threading

    reset_registry()
    sess = create_session(tmp_path.name)
    sid = sess.session_id
    old_ts = (datetime.now(UTC) - timedelta(minutes=6)).isoformat()
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "earlier", "ts": old_ts})

    stop = threading.Event()
    t = threading.Thread(
        target=run_folded,
        args=(stop,),
        kwargs={
            "persona_dir": tmp_path,
            "provider": fake_provider,
            "event_bus": fake_event_bus,
            "tick_interval_s": 0.1,
            "silence_minutes": 5.0,
            "heartbeat_interval_s": None,
            "soul_review_interval_s": None,
            "finalize_after_hours": 24.0,
            "finalize_interval_s": None,  # disable finalize during this test
        },
    )
    t.start()
    time.sleep(0.5)
    stop.set()
    t.join(timeout=2.0)

    buf = tmp_path / "active_conversations" / f"{sid}.jsonl"
    assert buf.exists(), "snapshot sweep must NOT delete the buffer"
    assert get_session(sid) is not None, "snapshot sweep must NOT evict from _SESSIONS"
    # And the new event type was published:
    types = [e.get("type") for e in fake_event_bus.events]
    assert "session_snapshot" in types


def test_supervisor_finalize_cadence_drops_old_sessions(
    tmp_path: Path, fake_provider, fake_event_bus
) -> None:
    from brain.chat.session import create_session, get_session, reset_registry
    from brain.bridge.supervisor import run_folded
    from brain.ingest.buffer import ingest_turn
    import threading

    reset_registry()
    sess = create_session(tmp_path.name)
    sid = sess.session_id
    old_ts = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "ancient", "ts": old_ts})

    stop = threading.Event()
    t = threading.Thread(
        target=run_folded,
        args=(stop,),
        kwargs={
            "persona_dir": tmp_path,
            "provider": fake_provider,
            "event_bus": fake_event_bus,
            "tick_interval_s": 0.1,
            "silence_minutes": 5.0,
            "heartbeat_interval_s": None,
            "soul_review_interval_s": None,
            "finalize_after_hours": 24.0,
            "finalize_interval_s": 0.05,  # fire near-immediately for the test
        },
    )
    t.start()
    time.sleep(0.5)
    stop.set()
    t.join(timeout=2.0)

    buf = tmp_path / "active_conversations" / f"{sid}.jsonl"
    assert not buf.exists(), "finalize must delete the buffer"
    assert get_session(sid) is None, "finalize must remove session from _SESSIONS"
    types = [e.get("type") for e in fake_event_bus.events]
    assert "session_finalized" in types
```

Existing tests may assert on `session_closed` from the sweep — update those to expect `session_snapshot` instead (the supervisor only publishes `session_closed` via the explicit-close path in `server.py`, which this thread doesn't touch).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/brain/bridge/test_supervisor.py -v -k "snapshot or finalize"`
Expected: FAIL — `run_folded` doesn't accept `finalize_after_hours` / `finalize_interval_s`, doesn't publish `session_snapshot` / `session_finalized`, and still deletes the buffer.

- [ ] **Step 3: Rewire `brain/bridge/supervisor.py`**

In `brain/bridge/supervisor.py`:

a. Update the import at the top:

```python
from brain.ingest.pipeline import (
    finalize_stale_sessions,
    snapshot_stale_sessions,
)
```

Remove the now-unused `close_stale_sessions` import.

b. Update `run_folded` signature — add two new kwargs:

```python
def run_folded(
    stop_event: threading.Event,
    *,
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus,
    tick_interval_s: float = 60.0,
    silence_minutes: float = 5.0,
    heartbeat_interval_s: float | None = 900.0,
    soul_review_interval_s: float | None = 6 * 3600.0,
    finalize_after_hours: float = 24.0,
    finalize_interval_s: float | None = 3600.0,
) -> None:
```

c. Initialise the finalize bookkeeping variable next to the heartbeat one:

```python
    last_finalize_at = (
        time.monotonic() if finalize_interval_s is not None else None
    )
```

d. Replace the body inside the `with ExitStack()` block. The current call is `close_stale_sessions(...)` followed by a `for sid in closed_session_ids: remove_session(sid)`. Replace it:

```python
                reports = snapshot_stale_sessions(
                    persona_dir,
                    silence_minutes=silence_minutes,
                    store=store,
                    hebbian=hebbian,
                    provider=provider,
                    embeddings=embeddings,
                )
                # Snapshot is NON-destructive — do NOT call remove_session
                # here. Session lifecycle is owned by finalize_stale_sessions
                # below and the explicit /sessions/close path.
                pruned_empty_sessions = prune_empty_sessions(
                    older_than_seconds=silence_minutes * 60.0,
                    persona_name=persona_dir.name,
                )
```

e. Replace the `session_closed` event publish with `session_snapshot`:

```python
            for r in reports:
                event_bus.publish(
                    {
                        "type": "session_snapshot",
                        "session_id": r.session_id,
                        "committed": r.committed,
                        "deduped": r.deduped,
                        "soul_candidates": r.soul_candidates,
                        "errors": r.errors,
                        "at": _now_iso(),
                    }
                )
```

The `supervisor_tick` event publish stays as-is.

f. Add a new finalize cadence block after the soul-review cadence:

```python
        # Finalize cadence — 24h silence (default) or explicit. Each pass
        # does at most one final snapshot per stale session, then deletes
        # buffer + cursor + registry entry. Slow cadence (hourly default)
        # because the threshold is days, not minutes.
        if (
            finalize_interval_s is not None
            and last_finalize_at is not None
            and time.monotonic() - last_finalize_at >= finalize_interval_s
        ):
            try:
                _run_finalize_tick(
                    persona_dir,
                    provider,
                    event_bus,
                    finalize_after_hours=finalize_after_hours,
                )
            except Exception:
                logger.exception("supervisor finalize tick raised")
            last_finalize_at = time.monotonic()
```

g. Add the helper at module level:

```python
def _run_finalize_tick(
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus,
    *,
    finalize_after_hours: float,
) -> None:
    """Run one finalize pass — per-tick stores, then drop registry entries
    for every session that was finalized."""
    with ExitStack() as stack:
        store = MemoryStore(persona_dir / "memories.db")
        stack.callback(store.close)
        hebbian = HebbianMatrix(persona_dir / "hebbian.db")
        stack.callback(hebbian.close)
        embeddings = EmbeddingCache(
            persona_dir / "embeddings.db", FakeEmbeddingProvider(dim=256),
        )
        stack.callback(embeddings.close)

        reports = finalize_stale_sessions(
            persona_dir,
            finalize_after_hours=finalize_after_hours,
            store=store,
            hebbian=hebbian,
            provider=provider,
            embeddings=embeddings,
        )

    for r in reports:
        remove_session(r.session_id)
        event_bus.publish(
            {
                "type": "session_finalized",
                "session_id": r.session_id,
                "committed": r.committed,
                "deduped": r.deduped,
                "errors": r.errors,
                "at": _now_iso(),
            }
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/brain/bridge/test_supervisor.py -v`
Expected: All tests PASS, including the two new ones and any existing tests that previously asserted on `session_closed` (now updated to `session_snapshot`).

- [ ] **Step 5: Commit**

```bash
git add brain/bridge/supervisor.py tests/unit/brain/bridge/test_supervisor.py
git commit -m "feat(supervisor): non-destructive snapshot sweep + 24h finalize cadence"
```

---

## Phase 5 — Budget guard

### Task 6: `apply_budget` module

**Files:**
- Create: `brain/chat/budget.py`
- Test: `tests/unit/brain/chat/test_budget.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/brain/chat/test_budget.py`:

```python
"""Tests for brain.chat.budget — prompt-size guard."""

from __future__ import annotations

from brain.bridge.chat import ChatMessage
from brain.chat.budget import apply_budget


class StubProvider:
    """Minimal LLMProvider stub. apply_budget only calls generate()."""

    def __init__(self, response: str = "[summary of earlier conversation]"):
        self.response = response
        self.calls: list[str] = []

    def name(self) -> str:
        return "stub"

    def generate(self, *, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        return self.response


def test_apply_budget_below_threshold_is_passthrough() -> None:
    msgs = [
        ChatMessage(role="system", content="be Nell"),
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
    ]
    out = apply_budget(msgs, max_tokens=1_000, preserve_tail_msgs=40, provider=StubProvider())
    assert out == msgs


def test_apply_budget_above_threshold_compresses_head() -> None:
    # ~4 chars/token: 200_000 chars ≈ 50K tokens. We want to exceed 1K tokens.
    huge_text = "x" * 8_000  # ~2K tokens estimate
    msgs = [ChatMessage(role="system", content="be Nell")]
    # 60 head messages, each ~2K tokens → ~120K tokens of head
    for i in range(60):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append(ChatMessage(role=role, content=huge_text))
    # 40 tail messages (small)
    for i in range(40):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append(ChatMessage(role=role, content="tail"))

    provider = StubProvider(response="[earlier: lots of x]")
    out = apply_budget(msgs, max_tokens=10_000, preserve_tail_msgs=40, provider=provider)

    # Structure: original system + compressed-head system note + 40 tail msgs
    assert out[0] == msgs[0]
    assert out[1].role == "system"
    assert "earlier" in out[1].content.lower() or "[earlier" in out[1].content
    assert out[-40:] == msgs[-40:]
    assert len(provider.calls) == 1


def test_apply_budget_compression_failure_falls_back_to_truncation() -> None:
    msgs = [ChatMessage(role="system", content="be Nell")]
    huge = "y" * 8_000
    for i in range(60):
        msgs.append(ChatMessage(role="user" if i % 2 == 0 else "assistant", content=huge))
    for i in range(40):
        msgs.append(ChatMessage(role="user" if i % 2 == 0 else "assistant", content="tail"))

    class ExplodingProvider:
        def name(self) -> str:
            return "exploding"

        def generate(self, *, prompt: str, **kwargs) -> str:
            raise RuntimeError("model down")

    out = apply_budget(msgs, max_tokens=10_000, preserve_tail_msgs=40,
                       provider=ExplodingProvider())

    assert out[0] == msgs[0]
    assert out[1].role == "system"
    assert "truncated" in out[1].content.lower()
    assert out[-40:] == msgs[-40:]


def test_apply_budget_preserves_short_session_with_few_tail_msgs() -> None:
    # 5 small messages, well under any budget — return unchanged.
    msgs = [
        ChatMessage(role="system", content="be Nell"),
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hi"),
        ChatMessage(role="user", content="how"),
        ChatMessage(role="assistant", content="good"),
    ]
    out = apply_budget(msgs, max_tokens=190_000, preserve_tail_msgs=40,
                       provider=StubProvider())
    assert out == msgs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/brain/chat/test_budget.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'brain.chat.budget'`.

- [ ] **Step 3: Implement `brain/chat/budget.py`**

```python
"""Prompt-size guard for engine.respond.

apply_budget inspects the message list, estimates its size via len(text)//4,
and when it exceeds `max_tokens`:
  1. Splits into [head_to_compress, preserved_tail (last preserve_tail_msgs)].
  2. Concatenates head text and asks the provider to summarise.
  3. Returns [original_system, compressed_head_system, *preserved_tail].

On provider failure, falls back to a deterministic truncation note. The
original system message is never compressed.
"""

from __future__ import annotations

import logging

from brain.bridge.chat import ChatMessage
from brain.bridge.provider import LLMProvider

logger = logging.getLogger(__name__)

_COMPRESSION_PROMPT = """Summarize the following conversation for context preservation.
Preserve: names of people and places, decisions made, emotional beats,
unresolved threads, anything that would be referenced later.
Drop: pleasantries, repetition, formatting noise.
Output prose only, no headers or lists.

CONVERSATION:
{transcript}

SUMMARY:"""


def _estimate_tokens(messages: list[ChatMessage]) -> int:
    total_chars = 0
    for m in messages:
        c = m.content
        if isinstance(c, str):
            total_chars += len(c)
        else:
            # tuple of content blocks (text + images) — count text only
            for block in c:
                text = getattr(block, "text", None)
                if text:
                    total_chars += len(text)
    return total_chars // 4


def _stringify(message: ChatMessage) -> str:
    c = message.content
    if isinstance(c, str):
        return c
    parts = []
    for block in c:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return " ".join(parts)


def apply_budget(
    messages: list[ChatMessage],
    *,
    max_tokens: int = 190_000,
    preserve_tail_msgs: int = 40,
    provider: LLMProvider,
) -> list[ChatMessage]:
    """Return a message list that fits inside max_tokens.

    Identity transform when the estimate is below max_tokens or when the
    message list is short enough that there's no head to compress (i.e.
    fewer than 2 + preserve_tail_msgs entries — the system message plus
    the preserved tail).
    """
    if _estimate_tokens(messages) <= max_tokens:
        return messages

    if len(messages) < 2 + preserve_tail_msgs:
        # Not enough head to compress — leave as-is (the caller's prompt
        # is huge but small in count, e.g. a single 200K user message).
        return messages

    system_msg = messages[0]
    head = messages[1 : len(messages) - preserve_tail_msgs]
    tail = messages[-preserve_tail_msgs:]

    if not head:
        return messages

    transcript = "\n".join(
        f"{m.role}: {_stringify(m)}" for m in head
    )
    summary_msg: ChatMessage
    try:
        summary = provider.generate(prompt=_COMPRESSION_PROMPT.format(transcript=transcript))
        summary_msg = ChatMessage(
            role="system",
            content=f"[Earlier in this conversation: {summary.strip()}]",
        )
    except Exception:
        logger.exception("apply_budget: provider summarisation failed; falling back")
        summary_msg = ChatMessage(
            role="system",
            content=f"[truncated {len(head)} earlier messages]",
        )

    return [system_msg, summary_msg, *tail]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/brain/chat/test_budget.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add brain/chat/budget.py tests/unit/brain/chat/test_budget.py
git commit -m "feat(chat): budget guard for prompt size with summarise fallback"
```

---

## Phase 6 — Engine reads buffer

### Task 7: Replace `*session.history` with buffer-driven prompt + budget guard

**Files:**
- Modify: `brain/chat/engine.py`
- Test: `tests/unit/brain/chat/test_engine.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/brain/chat/test_engine.py` (adjust fixture names to match the existing module style):

```python
def test_respond_reads_prior_turns_from_buffer_not_history(
    tmp_path, fake_provider, mem_store, hebbian_store, voice_md
) -> None:
    """The prompt sent to the provider must contain prior turns read from
    the buffer file — NOT from session.history."""
    from brain.chat.engine import respond
    from brain.chat.session import create_session
    from brain.ingest.buffer import ingest_turn

    sess = create_session(tmp_path.name)
    # Pre-seed buffer with a prior turn that is NOT in session.history.
    ingest_turn(tmp_path, {"session_id": sess.session_id, "speaker": "user",
                           "text": "I love watercolour"})
    ingest_turn(tmp_path, {"session_id": sess.session_id, "speaker": "assistant",
                           "text": "tell me about the brushes"})
    # session.history is empty for this session — proves the buffer is the source.

    respond(
        tmp_path, "the kolinsky sable",
        store=mem_store, hebbian=hebbian_store, provider=fake_provider,
        session=sess, voice_md_override=voice_md,
    )

    sent = fake_provider.last_messages
    user_texts = [m.content for m in sent if m.role == "user"]
    assert "I love watercolour" in user_texts
    assert "the kolinsky sable" in user_texts


def test_respond_falls_back_to_history_when_buffer_read_fails(
    tmp_path, fake_provider, mem_store, hebbian_store, voice_md, monkeypatch
) -> None:
    from brain.chat.engine import respond
    from brain.chat.session import create_session

    sess = create_session(tmp_path.name)
    sess.append_turn("hi from history", "hi back")

    def boom(*a, **kw):
        raise OSError("disk gone")

    monkeypatch.setattr("brain.chat.engine.read_session", boom)

    respond(
        tmp_path, "next turn",
        store=mem_store, hebbian=hebbian_store, provider=fake_provider,
        session=sess, voice_md_override=voice_md,
    )

    sent = fake_provider.last_messages
    contents = [m.content for m in sent if isinstance(m.content, str)]
    assert "hi from history" in contents


def test_respond_replays_image_turn_from_buffer(
    tmp_path, fake_provider, mem_store, hebbian_store, voice_md, sample_image_sha
) -> None:
    """A prior user turn with image_shas is reconstructed as a content tuple."""
    from brain.bridge.chat import ImageBlock, TextBlock
    from brain.chat.engine import respond
    from brain.chat.session import create_session
    from brain.ingest.buffer import ingest_turn

    sess = create_session(tmp_path.name)
    ingest_turn(tmp_path, {"session_id": sess.session_id, "speaker": "user",
                           "text": "look at this", "image_shas": [sample_image_sha]})

    respond(
        tmp_path, "what do you think?",
        store=mem_store, hebbian=hebbian_store, provider=fake_provider,
        session=sess, voice_md_override=voice_md,
    )

    sent = fake_provider.last_messages
    image_user_msg = next(
        m for m in sent if m.role == "user" and not isinstance(m.content, str)
    )
    blocks = list(image_user_msg.content)
    assert any(isinstance(b, TextBlock) and b.text == "look at this" for b in blocks)
    assert any(isinstance(b, ImageBlock) and b.image_sha == sample_image_sha for b in blocks)
```

`sample_image_sha` should be an existing fixture (the engine tests already exercise images via `_build_user_message`); if not, drop the third test and add a TODO note that image replay coverage moves to the integration test in Task 9.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/brain/chat/test_engine.py -v -k "buffer or replay or fallback"`
Expected: FAIL — engine still uses `*session.history`.

- [ ] **Step 3: Modify `brain/chat/engine.py`**

Add imports near the existing ones:

```python
from brain.chat.budget import apply_budget
from brain.ingest.buffer import read_session
```

Add a helper near the top of the module (after `_build_user_message`):

```python
def _buffer_turns_to_messages(
    persona_dir: Path, turns: list[dict]
) -> list[ChatMessage]:
    """Reconstruct ChatMessage list from buffer JSONL records.

    Image-bearing user turns rebuild a (TextBlock, *ImageBlock) content
    tuple identical to what _build_user_message produces for live turns.
    Missing or unreadable images are skipped with a warning, matching
    _build_user_message's defensive behaviour.
    """
    from brain.bridge.chat import ContentBlock
    from brain.images import media_type_for_sha

    out: list[ChatMessage] = []
    for t in turns:
        speaker = t.get("speaker")
        text = t.get("text", "")
        if speaker == "user":
            role: ChatMessageRole = "user"
        elif speaker == "assistant":
            role = "assistant"
        else:
            continue  # skip unknown speakers defensively

        image_shas = t.get("image_shas") or []
        if role == "user" and image_shas:
            blocks: list[ContentBlock] = []
            if text:
                blocks.append(TextBlock(text=text))
            for sha in image_shas:
                try:
                    media_type = media_type_for_sha(persona_dir, sha)
                    blocks.append(ImageBlock(image_sha=sha, media_type=media_type))
                except (FileNotFoundError, ValueError) as exc:
                    logger.warning(
                        "buffer replay: skipping image_sha=%s: %s",
                        sha[:8] if len(sha) >= 8 else sha,
                        exc,
                    )
            if blocks:
                out.append(ChatMessage(role=role, content=tuple(blocks)))
            elif text:
                out.append(ChatMessage(role=role, content=text))
            continue

        out.append(ChatMessage(role=role, content=text))
    return out
```

`ChatMessageRole` may need to be imported from `brain.bridge.chat` (use the existing import path used elsewhere in this module).

In `respond`, replace the "6. Messages list" block:

```python
    # 6. Messages list — buffer-driven, with budget guard.
    user_msg = _build_user_message(persona_dir, user_input, image_shas)
    try:
        prior_turns = read_session(persona_dir, session.session_id)
        # Exclude the just-appended live turn — buffer write happens in
        # _persist_turn AFTER the model call, so prior_turns is the
        # already-persisted history. (See _persist_turn ordering.)
        history_msgs = _buffer_turns_to_messages(persona_dir, prior_turns)
    except Exception:
        logger.exception(
            "engine.respond: buffer read failed session=%s; falling back to session.history",
            session.session_id,
        )
        history_msgs = list(session.history)

    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=system_msg),
        *history_msgs,
        user_msg,
    ]
    messages = apply_budget(
        messages,
        max_tokens=190_000,
        preserve_tail_msgs=40,
        provider=provider,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/brain/chat/test_engine.py -v`
Expected: PASS, including the new buffer-driven, fallback, and image-replay tests.

- [ ] **Step 5: Commit**

```bash
git add brain/chat/engine.py tests/unit/brain/chat/test_engine.py
git commit -m "feat(chat): engine builds prompt from buffer + applies budget guard"
```

---

## Phase 7 — Session.py demotion + integration test

### Task 8: Demote `HISTORY_MAX_TURNS`

**Files:**
- Modify: `brain/chat/session.py`

- [ ] **Step 1: Update docstring + raise sanity ceiling**

Edit `brain/chat/session.py`. Change line 23-24:

```python
# Sanity ceiling on the in-memory history list. As of 2026-05-10, the
# engine builds its prompt from the buffer file (see brain.chat.engine.
# _buffer_turns_to_messages), so session.history is informational only —
# kept for telemetry, tests, and a fallback path when the buffer read
# fails. A high ceiling is fine; it never hits the prompt unless the
# buffer is unreadable.
HISTORY_MAX_TURNS = 5000
```

- [ ] **Step 2: Run the full chat test suite to confirm no regression**

Run: `uv run pytest tests/unit/brain/chat -v`
Expected: All tests pass. If any test was asserting `HISTORY_MAX_TURNS == 20`, update it to the new value or drop it as no longer load-bearing.

- [ ] **Step 3: Commit**

```bash
git add brain/chat/session.py tests/unit/brain/chat/
git commit -m "refactor(chat): demote HISTORY_MAX_TURNS — buffer is now prompt source"
```

### Task 9: Integration test — sticky session round-trip

**Files:**
- Create: `tests/integration/brain/bridge/test_sticky_session.py`

- [ ] **Step 1: Write the integration test**

Create `tests/integration/brain/bridge/test_sticky_session.py`:

```python
"""Integration: 50-turn session → 5-min silence → snapshot sweep →
user returns → full prior transcript appears in the next prompt."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.bridge.events import EventBus
from brain.chat.engine import respond
from brain.chat.session import create_session, get_session, reset_registry
from brain.ingest.buffer import ingest_turn
from brain.bridge.supervisor import run_folded


def test_sticky_session_survives_snapshot_sweep(
    tmp_path: Path, fake_provider, mem_store, hebbian_store, voice_md
) -> None:
    reset_registry()
    persona_dir = tmp_path
    sess = create_session(persona_dir.name)
    sid = sess.session_id

    # Pre-seed 50 prior turns (25 user + 25 assistant pairs), each stamped
    # 6 minutes in the past so they trip the silence threshold.
    base = datetime.now(UTC) - timedelta(minutes=6)
    for i in range(25):
        ingest_turn(persona_dir, {
            "session_id": sid, "speaker": "user", "text": f"u{i}",
            "ts": (base + timedelta(seconds=i * 2)).isoformat(),
        })
        ingest_turn(persona_dir, {
            "session_id": sid, "speaker": "assistant", "text": f"a{i}",
            "ts": (base + timedelta(seconds=i * 2 + 1)).isoformat(),
        })

    bus = EventBus()
    stop = threading.Event()
    t = threading.Thread(
        target=run_folded,
        args=(stop,),
        kwargs={
            "persona_dir": persona_dir,
            "provider": fake_provider,
            "event_bus": bus,
            "tick_interval_s": 0.1,
            "silence_minutes": 5.0,
            "heartbeat_interval_s": None,
            "soul_review_interval_s": None,
            "finalize_after_hours": 24.0,
            "finalize_interval_s": None,
        },
    )
    t.start()
    time.sleep(0.5)
    stop.set()
    t.join(timeout=2.0)

    # Session must still be alive, buffer intact.
    assert get_session(sid) is not None
    buf = persona_dir / "active_conversations" / f"{sid}.jsonl"
    assert buf.exists()

    # User returns and sends a new message — prompt must contain all 50
    # prior turns (or all of them that fit; budget guard not triggered at
    # this volume).
    fake_provider.reset_calls()
    respond(
        persona_dir, "still here",
        store=mem_store, hebbian=hebbian_store, provider=fake_provider,
        session=sess, voice_md_override=voice_md,
    )

    sent = fake_provider.last_messages
    user_texts = [m.content for m in sent if m.role == "user" and isinstance(m.content, str)]
    assistant_texts = [m.content for m in sent if m.role == "assistant" and isinstance(m.content, str)]
    for i in range(25):
        assert f"u{i}" in user_texts, f"missing prior user turn u{i}"
        assert f"a{i}" in assistant_texts, f"missing prior assistant turn a{i}"
    assert "still here" in user_texts
```

- [ ] **Step 2: Run the integration test**

Run: `uv run pytest tests/integration/brain/bridge/test_sticky_session.py -v`
Expected: PASS.

- [ ] **Step 3: Run the full test suite as a regression gate**

Run: `uv run pytest -x`
Expected: All tests PASS. If failures appear in unrelated modules (e.g. callers of the now-deprecated `close_stale_sessions` returning `session_closed` event), fix them inline — they're part of this change set.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/brain/bridge/test_sticky_session.py
git commit -m "test(integration): sticky session survives snapshot sweep, prompt has full transcript"
```

---

## Phase 8 — Cleanup

### Task 10: Decide on `close_stale_sessions`

**Files:**
- Modify: `brain/ingest/pipeline.py`
- Modify: `tests/unit/brain/ingest/test_pipeline.py`

- [ ] **Step 1: Check for remaining callers**

Run: `git grep -n close_stale_sessions -- brain/ tests/`
Expected: only `brain/ingest/pipeline.py` (definition) + its tests.

- [ ] **Step 2: Remove `close_stale_sessions`**

Since the supervisor no longer uses it and no test outside `test_pipeline.py` references it, delete the function from `brain/ingest/pipeline.py` and remove its tests from `tests/unit/brain/ingest/test_pipeline.py`. If any test is worth keeping (e.g. an empty-buffer ghost cleanup that the snapshot sweep also handles), port it onto `snapshot_stale_sessions` instead.

If grep finds an outside caller, leave `close_stale_sessions` in place as a thin wrapper that emits a deprecation warning and forwards to `snapshot_stale_sessions` — flag it in the commit message.

- [ ] **Step 3: Run full suite**

Run: `uv run pytest -x`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add brain/ingest/pipeline.py tests/unit/brain/ingest/test_pipeline.py
git commit -m "refactor(ingest): remove unused close_stale_sessions"
```

### Task 11: Verify lint + type checks

- [ ] **Step 1: Run ruff**

Run: `uv run ruff check brain/ tests/`
Expected: clean. Fix any new warnings inline.

- [ ] **Step 2: Run the full test suite one last time**

Run: `uv run pytest`
Expected: All tests PASS.

- [ ] **Step 3: Final commit if any lint fixes were applied**

```bash
git add brain/ tests/
git commit -m "chore: ruff cleanup after sticky-session change set"
```

---

## Self-review (done before handoff)

- **Spec coverage:** Every section of the source spec maps to a task.
  - §Architecture/1 Live prompt → Task 7
  - §Architecture/2 Sticky sessions → Tasks 2, 3
  - §Architecture/3 Real close → Task 4 (+ close_session cursor cleanup in Task 4 step 3)
  - §Architecture/4 Budget guard → Task 6
  - §Observability events → Task 5
  - §Migration → handled implicitly (cursor missing = full extract; existing sessions need no migration script)
- **No placeholders:** every test has a body, every implementation block contains the code an engineer pastes.
- **Type consistency:** `extract_session_snapshot`, `snapshot_stale_sessions`, `finalize_stale_sessions` all share the same `IngestReport` return type and same `(persona_dir, *, store, hebbian, provider, embeddings=None, config=None)` keyword shape. `apply_budget` keyword names match between definition (Task 6) and call site (Task 7). `run_folded` signature additions (`finalize_after_hours`, `finalize_interval_s`) match between definition (Task 5) and integration test (Task 9).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-10-full-session-context.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Pairs well with the worktree pattern in `superpowers:using-git-worktrees`.

2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

Which approach?

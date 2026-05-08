"""Tests for brain.setup — pure persona-setup helpers used by `nell init`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.setup import (
    VOICE_TEMPLATES,
    install_voice_template,
    validate_persona_name,
    write_persona_config,
)

# ---- validate_persona_name ----


def test_validate_persona_name_accepts_simple_names() -> None:
    for name in ["nell", "siren", "nell_2", "Nell-test", "x", "a" * 40]:
        validate_persona_name(name)  # no raise


def test_validate_persona_name_rejects_path_traversal_and_garbage() -> None:
    for evil in ["../escape", "a/b", "..", "", "n e l l", "x" * 41,
                 "weird:char", "name.with.dots"]:
        with pytest.raises(ValueError, match="invalid persona name"):
            validate_persona_name(evil)


def test_validate_persona_name_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="invalid persona name"):
        validate_persona_name(None)  # type: ignore[arg-type]


# ---- write_persona_config ----


def test_write_persona_config_creates_file_with_user_name(tmp_path: Path) -> None:
    persona_dir = tmp_path / "siren"
    path = write_persona_config(persona_dir, user_name="Hana")
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["user_name"] == "Hana"
    # Defaults preserved for the other fields
    assert data["provider"] == "claude-cli"


def test_write_persona_config_preserves_existing_fields(tmp_path: Path) -> None:
    """Re-running init shouldn't clobber an existing provider override."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()
    config_path = persona_dir / "persona_config.json"
    config_path.write_text(json.dumps({
        "provider": "ollama",
        "searcher": "ddgs",
        "mcp_audit_log_level": "full",
    }))

    write_persona_config(persona_dir, user_name="Hana")
    data = json.loads(config_path.read_text())
    assert data["user_name"] == "Hana"
    assert data["provider"] == "ollama"  # preserved
    assert data["mcp_audit_log_level"] == "full"  # preserved


def test_write_persona_config_strips_empty_user_name_to_none(tmp_path: Path) -> None:
    persona_dir = tmp_path / "nell"
    write_persona_config(persona_dir, user_name="   ")
    data = json.loads((persona_dir / "persona_config.json").read_text())
    assert data["user_name"] is None


def test_write_persona_config_creates_dir_if_missing(tmp_path: Path) -> None:
    persona_dir = tmp_path / "deep" / "nested" / "nell"
    write_persona_config(persona_dir, user_name="Hana")
    assert (persona_dir / "persona_config.json").exists()


# ---- install_voice_template ----


def test_install_voice_template_default_writes_no_file(tmp_path: Path) -> None:
    """default + skip → no voice.md (DEFAULT_VOICE_TEMPLATE applies on chat)."""
    persona_dir = tmp_path / "siren"
    for template in ("default", "skip"):
        result = install_voice_template(persona_dir, template)
        assert result is None
        assert not (persona_dir / "voice.md").exists()


def test_install_voice_template_nell_example_copies_packaged_file(
    tmp_path: Path,
) -> None:
    """nell-example writes the packaged brain/voice_templates/nell-voice.md.

    The template ships inside the brain wheel so it's available whether
    the framework is installed from source, from a wheel, or from inside
    the Phase 7 bundled NellFace.app — no `repo_root` lookup needed.
    """
    persona_dir = tmp_path / "persona"
    result = install_voice_template(persona_dir, "nell-example")
    assert result == persona_dir / "voice.md"
    content = result.read_text(encoding="utf-8")
    # The shipped Nell voice draft is the canonical one — opens with
    # the section-1 header.
    assert "## 1. Who you are" in content
    assert len(content) > 1000  # not an empty file


def test_install_voice_template_unknown_raises(tmp_path: Path) -> None:
    persona_dir = tmp_path / "nell"
    with pytest.raises(ValueError, match="unknown voice template"):
        install_voice_template(persona_dir, "no-such-template")


def test_voice_templates_keys_are_documented() -> None:
    """Every key has a non-empty human-readable description so the wizard
    can surface the choices to the user."""
    assert set(VOICE_TEMPLATES.keys()) == {"default", "nell-example", "skip"}
    for desc in VOICE_TEMPLATES.values():
        assert isinstance(desc, str) and len(desc) > 20


# ---------------------------------------------------------------------------
# Audit 2026-05-07 P2-1 + P2-2 — concurrent-write race regression tests
# ---------------------------------------------------------------------------


def test_growth_log_concurrent_appends_dont_clobber_each_other(tmp_path: Path) -> None:
    """Two concurrent appenders must each land their event line — the
    previous read-rewrite-replace shape lost one event under contention."""
    import threading
    from datetime import UTC
    from datetime import datetime as _dt

    from brain.growth.log import GrowthLogEvent, append_growth_event, read_growth_log

    log_path = tmp_path / "growth_log.jsonl"
    num_threads = 8
    events_per_thread = 5

    def worker(thread_id: int) -> None:
        for i in range(events_per_thread):
            append_growth_event(
                log_path,
                GrowthLogEvent(
                    timestamp=_dt.now(UTC),
                    type="emotion_added",
                    name=f"thread_{thread_id}_event_{i}",
                    description="concurrent test",
                    decay_half_life_days=None,
                    reason="race regression guard",
                    evidence_memory_ids=(),
                    score=0.0,
                    relational_context=None,
                ),
            )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    events = read_growth_log(log_path)
    expected = num_threads * events_per_thread
    assert len(events) == expected, (
        f"expected {expected} events from concurrent appenders; got {len(events)} "
        f"— races dropped {expected - len(events)} events"
    )
    # Every (thread, event) pair must be present exactly once
    names = {e.name for e in events}
    expected_names = {
        f"thread_{t}_event_{i}"
        for t in range(num_threads)
        for i in range(events_per_thread)
    }
    assert names == expected_names


def test_soul_candidate_review_holds_lock_against_concurrent_queue(tmp_path: Path) -> None:
    """While review is in its read-modify-rewrite window, queue_soul_candidate
    must block until review releases the lock; previously the queued candidate
    could be silently dropped when review replaced the file."""
    import threading
    import time as _time

    from brain.ingest.soul_queue import queue_soul_candidate
    from brain.ingest.types import ExtractedItem
    from brain.utils.file_lock import file_lock

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    candidates_path = persona_dir / "soul_candidates.jsonl"
    candidates_path.write_text("")  # empty file so the lock has a sidecar to attach to

    queue_finished = threading.Event()

    def queue_worker():
        # This MUST block while the lock is held by the test.
        queue_soul_candidate(
            persona_dir,
            memory_id="m1",
            item=ExtractedItem(text="test", label="fact", importance=5),
            session_id="s1",
        )
        queue_finished.set()

    with file_lock(candidates_path):
        t = threading.Thread(target=queue_worker, daemon=True)
        t.start()
        # Give the worker a chance to attempt + block
        _time.sleep(0.2)
        assert not queue_finished.is_set(), (
            "queue_soul_candidate completed while review held the lock — "
            "the file_lock guard isn't engaging"
        )
    # Lock released — worker should now finish promptly
    t.join(timeout=2.0)
    assert queue_finished.is_set(), "queue_soul_candidate didn't finish after lock released"

    # Candidate did persist
    from brain.health.jsonl_reader import read_jsonl_skipping_corrupt

    records = read_jsonl_skipping_corrupt(candidates_path)
    assert len(records) == 1
    assert records[0]["memory_id"] == "m1"

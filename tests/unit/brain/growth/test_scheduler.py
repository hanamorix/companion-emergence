"""Tests for brain.growth.scheduler — orchestrator + atomic apply."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from brain.growth.proposal import EmotionProposal
from brain.growth.scheduler import (
    GrowthTickResult,
    _read_current_vocabulary_names,
    _should_run_growth_tick,
    run_growth_tick,
)
from brain.memory.store import MemoryStore


def _seed_vocab(persona_dir: Path, names: list[str] | None = None) -> None:
    """Seed an emotion_vocabulary.json with the given names as core emotions."""
    if names is None:
        names = ["love", "joy"]
    entries = [
        {
            "name": n,
            "description": f"the feeling of {n}",
            "category": "core",
            "decay_half_life_days": 7.0,
        }
        for n in names
    ]
    (persona_dir / "emotion_vocabulary.json").write_text(
        json.dumps({"version": 1, "emotions": entries}, indent=2),
        encoding="utf-8",
    )


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    pdir = tmp_path / "persona"
    pdir.mkdir()
    return pdir


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


def test_run_growth_tick_no_proposals_returns_zero(persona_dir: Path, store: MemoryStore) -> None:
    """Real crystallizer returns no proposals when the store has no repeated pattern."""
    _seed_vocab(persona_dir)
    result = run_growth_tick(persona_dir, store, datetime.now(UTC))
    assert isinstance(result, GrowthTickResult)
    assert result.emotions_added == 0
    assert result.proposals_seen == 0
    # No log file created when nothing happened
    assert not (persona_dir / "emotion_growth.log.jsonl").exists()


def test_run_growth_tick_applies_proposal_atomically(persona_dir: Path, store: MemoryStore) -> None:
    """When a crystallizer returns a proposal, vocabulary + log update together."""
    _seed_vocab(persona_dir)
    proposal = EmotionProposal(
        name="lingering",
        description="the slow trail of warmth after a loved person leaves the room",
        decay_half_life_days=7.0,
        evidence_memory_ids=("mem_a", "mem_b"),
        score=0.78,
        relational_context="recurred during Hana's tender messages",
    )

    with patch(
        "brain.growth.scheduler.crystallize_vocabulary",
        return_value=[proposal],
    ):
        result = run_growth_tick(persona_dir, store, datetime.now(UTC))

    assert result.emotions_added == 1
    assert result.proposals_seen == 1
    assert result.proposals_rejected == 0

    # Vocabulary file updated
    vocab = json.loads((persona_dir / "emotion_vocabulary.json").read_text(encoding="utf-8"))
    names = [e["name"] for e in vocab["emotions"]]
    assert "lingering" in names
    new_entry = next(e for e in vocab["emotions"] if e["name"] == "lingering")
    assert new_entry["category"] == "persona_extension"
    assert new_entry["decay_half_life_days"] == 7.0

    # Growth log updated
    log_path = persona_dir / "emotion_growth.log.jsonl"
    assert log_path.exists()
    [line] = log_path.read_text(encoding="utf-8").splitlines()
    parsed = json.loads(line)
    assert parsed["type"] == "emotion_added"
    assert parsed["name"] == "lingering"
    assert parsed["relational_context"] == "recurred during Hana's tender messages"


def test_run_growth_tick_skips_proposal_with_existing_name(
    persona_dir: Path, store: MemoryStore
) -> None:
    """Idempotent: a proposal whose name already exists in the vocabulary is skipped silently."""
    _seed_vocab(persona_dir, names=["love", "joy"])
    proposal = EmotionProposal(
        name="love",  # already in vocab
        description="dup",
        decay_half_life_days=None,
        evidence_memory_ids=(),
        score=0.5,
        relational_context=None,
    )
    with patch("brain.growth.scheduler.crystallize_vocabulary", return_value=[proposal]):
        result = run_growth_tick(persona_dir, store, datetime.now(UTC))
    assert result.emotions_added == 0
    assert result.proposals_seen == 1
    # Skipped duplicates aren't counted as rejections — re-proposing is normal.
    assert result.proposals_rejected == 0
    assert not (persona_dir / "emotion_growth.log.jsonl").exists()


def test_run_growth_tick_rejects_proposal_with_invalid_chars(
    persona_dir: Path, store: MemoryStore
) -> None:
    """Names containing path-traversal chars or curly braces are rejected as schema-invalid."""
    _seed_vocab(persona_dir)
    bad_names = ["bad/name", "bad\\name", "bad{name}", ""]
    for bn in bad_names:
        proposal = EmotionProposal(
            name=bn,
            description="x",
            decay_half_life_days=None,
            evidence_memory_ids=(),
            score=0.5,
            relational_context=None,
        )
        with patch("brain.growth.scheduler.crystallize_vocabulary", return_value=[proposal]):
            result = run_growth_tick(persona_dir, store, datetime.now(UTC))
        assert result.emotions_added == 0
        assert result.proposals_rejected == 1


def test_a_dry_growth_tick_leaves_a_corrupt_vocabulary_and_the_heal_marker_as_they_are(
    persona_dir: Path, store: MemoryStore
) -> None:
    """`nell heartbeat --dry-run` runs a dry tick: it reads, and must not quarantine, repair, lock or re-arm anything."""
    from brain.health.json_io import write_json
    from brain.paths import cadence_state_path

    vocab_path = persona_dir / "emotion_vocabulary.json"
    vocab_path.write_text("{not json", encoding="utf-8")
    write_json(cadence_state_path(persona_dir, "vocab_heal_final.json"), {"healed_at": "x", "count": 0})
    before = {p.relative_to(persona_dir).as_posix() for p in persona_dir.rglob("*")}

    with patch("brain.growth.scheduler.crystallize_vocabulary", return_value=[]):
        run_growth_tick(persona_dir, store, datetime.now(UTC), dry_run=True)

    assert vocab_path.read_text(encoding="utf-8") == "{not json"
    assert {p.relative_to(persona_dir).as_posix() for p in persona_dir.rglob("*")} == before


def test_a_dry_tick_on_an_unreadable_vocabulary_does_not_count_a_held_word_as_new(
    persona_dir: Path, store: MemoryStore
) -> None:
    """With the file unreadable, the dry tick falls back to the words this process holds, not to none at all."""
    from brain.emotion import vocabulary
    from brain.emotion.vocabulary import Emotion

    (persona_dir / "emotion_vocabulary.json").write_text("{not json", encoding="utf-8")
    vocabulary._unregister("calm_joy_blend")
    vocabulary.register(Emotion("calm_joy_blend", "calm and joy together", "persona_extension", 14.0))
    proposal = EmotionProposal(
        name="calm_joy_blend",
        description="calm and joy together",
        decay_half_life_days=14.0,
        evidence_memory_ids=(),
        score=0.5,
        relational_context=None,
    )
    with patch("brain.growth.scheduler.crystallize_vocabulary", return_value=[proposal]):
        result = run_growth_tick(persona_dir, store, datetime.now(UTC), dry_run=True)

    assert result.emotions_added == 0


@pytest.mark.no_base_failure(
    reason="guards register_new_word's held-word path: the base scheduler appended to the file itself, so it held there"
)
def test_growth_tick_writes_a_word_this_process_holds_but_the_file_lost(persona_dir: Path, store: MemoryStore) -> None:
    """A repair can drop a word the process still holds; the logged emotion_added must leave the word in the file."""
    from brain.emotion import vocabulary
    from brain.emotion.vocabulary import Emotion

    _seed_vocab(persona_dir)
    vocabulary._unregister("hope_joy_blend")
    vocabulary.register(Emotion("hope_joy_blend", "hope and joy together", "persona_extension", 20.0))
    proposal = EmotionProposal(
        name="hope_joy_blend",
        description="hope and joy together",
        decay_half_life_days=20.0,
        evidence_memory_ids=(),
        score=0.5,
        relational_context=None,
    )
    with patch("brain.growth.scheduler.crystallize_vocabulary", return_value=[proposal]):
        run_growth_tick(persona_dir, store, datetime.now(UTC))

    in_file = {e["name"] for e in json.loads((persona_dir / "emotion_vocabulary.json").read_text(encoding="utf-8"))["emotions"]}
    assert _added_names(persona_dir) == ["hope_joy_blend"]
    assert "hope_joy_blend" in in_file


def test_growth_tick_repairs_the_vocabulary_under_its_lock_and_re_arms_the_heal(
    persona_dir: Path, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scheduler's own repair of a corrupt file holds register_new_word's lock and deletes the heal marker, as the walker does."""
    from brain.growth import scheduler
    from brain.health.json_io import write_json
    from brain.paths import cadence_state_path
    from brain.utils.file_lock import file_lock

    vocab_path = persona_dir / "emotion_vocabulary.json"
    vocab_path.write_text("{not json", encoding="utf-8")
    marker = cadence_state_path(persona_dir, "vocab_heal_final.json")
    write_json(marker, {"healed_at": "x", "count": 0})
    held: list[bool] = []
    real_read = scheduler.read_json_result

    def spy(path, *args, **kwargs):  # noqa: ANN001, ANN202
        if Path(path) == vocab_path:
            with file_lock(vocab_path, blocking=False) as acquired:
                held.append(not acquired)
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(scheduler, "read_json_result", spy)
    with patch("brain.growth.scheduler.crystallize_vocabulary", return_value=[]):
        run_growth_tick(persona_dir, store, datetime.now(UTC))

    assert held == [True]
    assert not marker.exists()


def test_growth_tick_registers_word_immediately(persona_dir: Path, store: MemoryStore) -> None:
    """The crystallized word is usable in this process at once, with no registry reload."""
    from brain.emotion import vocabulary

    _seed_vocab(persona_dir)
    proposal = EmotionProposal(
        name="grief_joy_blend",
        description="grief and joy together",
        decay_half_life_days=31.5,
        evidence_memory_ids=(),
        score=0.5,
        relational_context=None,
    )
    assert vocabulary.get("grief_joy_blend") is None
    with patch("brain.growth.scheduler.crystallize_vocabulary", return_value=[proposal]):
        run_growth_tick(persona_dir, store, datetime.now(UTC))
    assert vocabulary.get("grief_joy_blend") is not None


def test_run_growth_tick_dry_run_does_not_write(persona_dir: Path, store: MemoryStore) -> None:
    """dry_run=True calls crystallizer but writes neither vocabulary nor log."""
    _seed_vocab(persona_dir)
    proposal = EmotionProposal(
        name="lingering",
        description="x",
        decay_half_life_days=None,
        evidence_memory_ids=(),
        score=0.5,
        relational_context=None,
    )
    vocab_before = (persona_dir / "emotion_vocabulary.json").read_text(encoding="utf-8")
    with patch("brain.growth.scheduler.crystallize_vocabulary", return_value=[proposal]):
        result = run_growth_tick(persona_dir, store, datetime.now(UTC), dry_run=True)
    assert result.emotions_added == 1  # would-have-added semantics
    # Files unchanged
    assert (persona_dir / "emotion_vocabulary.json").read_text(encoding="utf-8") == vocab_before
    assert not (persona_dir / "emotion_growth.log.jsonl").exists()


def test_run_growth_tick_handles_multiple_proposals(persona_dir: Path, store: MemoryStore) -> None:
    _seed_vocab(persona_dir)
    proposals = [
        EmotionProposal(
            name=f"p_{letter}",  # the new-word gate allows letters and underscores only, no digits
            description=f"desc {letter}",
            decay_half_life_days=float(i),
            evidence_memory_ids=(),
            score=0.5,
            relational_context=None,
        )
        for i, letter in enumerate("abc")
    ]
    with patch("brain.growth.scheduler.crystallize_vocabulary", return_value=proposals):
        result = run_growth_tick(persona_dir, store, datetime.now(UTC))
    assert result.emotions_added == 3
    log_path = persona_dir / "emotion_growth.log.jsonl"
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 3


# ---- Hardening: corrupt-vocab detection ----


def test_run_growth_tick_corrupt_vocabulary_logs_warning(
    persona_dir: Path, store: MemoryStore, caplog
) -> None:
    """A corrupt emotion_vocabulary.json is detected and surfaces a WARNING.

    Post-T10 the scheduler routes through json_io (HEAL), which quarantines the
    corrupt file and resets to default. The scheduler still completes (returns
    empty set) but the anomaly surfaces a WARNING so the user / forensic-grep
    has a thread to pull on rather than silent degradation.
    """
    import logging

    caplog.set_level(logging.WARNING)

    # Write corrupt JSON to the vocab path.
    (persona_dir / "emotion_vocabulary.json").write_text("{not valid json", encoding="utf-8")

    result = run_growth_tick(persona_dir, store, datetime.now(UTC))

    # Tick still completes (no proposals from stub anyway).
    assert result.emotions_added == 0

    # Warning surfaces the anomaly — message contains "anomaly" and the filename.
    anomaly_warnings = [
        r
        for r in caplog.records
        if "anomaly" in r.getMessage() and "emotion_vocabulary.json" in r.getMessage()
    ]
    assert len(anomaly_warnings) == 1


def test_run_growth_tick_invalid_vocabulary_schema_logs_warning(
    persona_dir: Path, store: MemoryStore, caplog
) -> None:
    """A vocab file with the wrong shape (missing 'emotions' list) surfaces a WARNING."""
    import logging

    caplog.set_level(logging.WARNING)

    (persona_dir / "emotion_vocabulary.json").write_text(
        json.dumps({"version": 1, "wrong_field": []}), encoding="utf-8"
    )

    result = run_growth_tick(persona_dir, store, datetime.now(UTC))
    assert result.emotions_added == 0

    # Post-T10: schema mismatch goes through json_io HEAL; warning message
    # contains "anomaly" and identifies the file.
    schema_warnings = [
        r
        for r in caplog.records
        if "anomaly" in r.getMessage() and "emotion_vocabulary.json" in r.getMessage()
    ]
    assert len(schema_warnings) == 1


# ---- Health T10: json_io HEAL wiring ----


def test_read_current_vocabulary_names_corrupt_quarantines_restores_bak(
    persona_dir: Path, caplog
) -> None:
    """Corrupt vocab + valid .bak1 → names from .bak1 returned + quarantine created."""
    import logging

    vocab_path = persona_dir / "emotion_vocabulary.json"
    bak1 = persona_dir / "emotion_vocabulary.json.bak1"
    bak1.write_text(
        json.dumps({"version": 1, "emotions": [{"name": "joy"}, {"name": "love"}]}),
        encoding="utf-8",
    )
    vocab_path.write_text("{corrupt{{", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        names, anomaly = _read_current_vocabulary_names(vocab_path)

    # Names restored from .bak1
    assert "joy" in names
    assert "love" in names
    # Anomaly captures the heal action so caller can surface to audit log
    assert anomaly is not None
    assert anomaly.action == "restored_from_bak1"
    # Original quarantined
    corrupt_files = list(persona_dir.glob("emotion_vocabulary.json.corrupt-*"))
    assert len(corrupt_files) == 1


def test_read_current_vocabulary_names_corrupt_no_bak_resets_to_default(
    persona_dir: Path, caplog
) -> None:
    """Corrupt vocab + no .bak → empty set returned + warning logged."""
    import logging

    vocab_path = persona_dir / "emotion_vocabulary.json"
    vocab_path.write_text("{corrupt{{", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        names, anomaly = _read_current_vocabulary_names(vocab_path)

    assert names == set()
    assert anomaly is not None
    assert anomaly.action == "reset_to_default"
    # Some warning surfaced (either from json_io or the existing path)
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_run_growth_tick_missing_vocabulary_no_warning(
    persona_dir: Path, store: MemoryStore, caplog
) -> None:
    """A missing vocab file is the expected fresh-persona case — no warning."""
    import logging

    caplog.set_level(logging.WARNING)

    # Don't seed the vocab file at all.
    result = run_growth_tick(persona_dir, store, datetime.now(UTC))
    assert result.emotions_added == 0

    # Crucially: no warning. Missing != corrupt.
    bad_warnings = [
        r
        for r in caplog.records
        if "corrupt JSON" in r.getMessage() or "invalid schema" in r.getMessage()
    ]
    assert bad_warnings == []


# ---- F3: anomaly collector forwarded through run_growth_tick ----


def test_run_growth_tick_with_collector_appends_vocab_anomaly(
    persona_dir: Path, store: MemoryStore
) -> None:
    """When run_growth_tick is passed an anomalies_collector and the
    vocabulary file is corrupt, the anomaly produced by the heal flow
    appears in the collector — surfacing growth-tick-internal anomalies
    to the heartbeat tick's audit log.

    Followup F3 from the brain-health module.
    """
    from brain.health.anomaly import BrainAnomaly

    # Corrupt the vocab file with no .bak so reset_to_default fires.
    (persona_dir / "emotion_vocabulary.json").write_text("{not json", encoding="utf-8")

    collector: list[BrainAnomaly] = []
    result = run_growth_tick(
        persona_dir,
        store,
        datetime.now(UTC),
        anomalies_collector=collector,
    )

    # Tick still completes (vocabulary crystallizer has no repeated pattern to propose).
    assert result.emotions_added == 0

    # Collector received the anomaly
    assert len(collector) == 1
    a = collector[0]
    assert a.file == "emotion_vocabulary.json"
    assert a.action == "reset_to_default"


def test_run_growth_tick_no_collector_does_not_raise(persona_dir: Path, store: MemoryStore) -> None:
    """Calling run_growth_tick without anomalies_collector still works —
    the parameter is opt-in, not required."""
    (persona_dir / "emotion_vocabulary.json").write_text("{not json", encoding="utf-8")
    # No collector passed; previously was the only way to call.
    result = run_growth_tick(persona_dir, store, datetime.now(UTC))
    assert result.emotions_added == 0


def test_run_growth_tick_clean_vocabulary_collector_unchanged(
    persona_dir: Path, store: MemoryStore
) -> None:
    """When the vocabulary file is healthy, the collector receives nothing."""
    from brain.health.anomaly import BrainAnomaly

    _seed_vocab(persona_dir)  # writes a clean vocabulary file

    collector: list[BrainAnomaly] = []
    run_growth_tick(persona_dir, store, datetime.now(UTC), anomalies_collector=collector)
    assert collector == []


# ---------------------------------------------------------------------------
# Task 4 — _should_run_growth_tick throttle predicate tests
# ---------------------------------------------------------------------------


def test_throttle_runs_when_never_ticked() -> None:
    """last_tick=None → always run (first ever growth tick)."""
    assert (
        _should_run_growth_tick(
            last_tick=None,
            now=datetime(2026, 4, 28, tzinfo=UTC),
            throttle_days=7.0,
        )
        is True
    )


def test_throttle_runs_when_window_elapsed() -> None:
    """8 days since last tick with a 7-day throttle → should run."""
    last = datetime(2026, 4, 20, tzinfo=UTC)
    now = datetime(2026, 4, 28, tzinfo=UTC)  # 8 days later
    assert _should_run_growth_tick(last_tick=last, now=now, throttle_days=7.0) is True


def test_throttle_skips_when_window_active() -> None:
    """3 days since last tick with a 7-day throttle → skip."""
    last = datetime(2026, 4, 25, tzinfo=UTC)
    now = datetime(2026, 4, 28, tzinfo=UTC)  # 3 days later
    assert _should_run_growth_tick(last_tick=last, now=now, throttle_days=7.0) is False


def test_throttle_boundary_at_exactly_threshold() -> None:
    """Exactly throttle_days elapsed → inclusive, returns True."""
    from datetime import timedelta

    last = datetime(2026, 4, 21, tzinfo=UTC)
    now = last + timedelta(days=7)  # exactly 7 days
    assert _should_run_growth_tick(last_tick=last, now=now, throttle_days=7.0) is True


def test_vocabulary_crystallization_emits_initiate_candidate(
    persona_dir: Path,
    store: MemoryStore,
) -> None:
    """After vocabulary crystallization commits, emit a candidate.

    Phase 4.2 — the emit fires inside scheduler.run_growth_tick immediately
    after register_new_word + append_growth_event, so it lives at the
    actual SoulStore-commit point for vocabulary changes.
    """
    from brain.initiate.emit import read_candidates

    _seed_vocab(persona_dir)
    proposal = EmotionProposal(
        name="lingering",
        description="the slow trail of warmth after a loved person leaves the room",
        decay_half_life_days=7.0,
        evidence_memory_ids=("mem_a", "mem_b"),
        score=0.78,
        relational_context="recurred during Hana's tender messages",
    )
    with patch(
        "brain.growth.scheduler.crystallize_vocabulary",
        return_value=[proposal],
    ):
        run_growth_tick(persona_dir, store, datetime.now(UTC))

    candidates = read_candidates(persona_dir)
    assert any(
        c.source == "crystallization" and c.source_id == "vocabulary_emotion:lingering"
        for c in candidates
    )


# ---- emotion rework B1 PR 5: a live persona's growth reads the engine ----


def _go_live(persona_dir: Path, store: MemoryStore, **seed: float):
    from brain.emotion import engine as emotion_engine
    from brain.memory.store import Memory
    from tests.unit.brain.emotion._engine_support import make_live

    store.create(Memory.create_new(content="seed", memory_type="conversation", domain="us", emotions=seed or {"joy": 5.0}))
    make_live(persona_dir, datetime.now(UTC))
    return emotion_engine.engine_for(persona_dir)


def _blend_rows(store: MemoryStore, emotions: dict[str, float]) -> None:
    from brain.memory.store import Memory

    for i in range(3):
        store.create(
            Memory.create_new(content=f"braided {i}", memory_type="conversation", domain="us", emotions=emotions, importance=8.0)
        )


def _added_names(persona_dir: Path) -> list[str]:
    log_path = persona_dir / "emotion_growth.log.jsonl"
    if not log_path.exists():
        return []
    return [json.loads(line)["name"] for line in log_path.read_text(encoding="utf-8").splitlines()]


def test_live_initiate_candidate_carries_short_averages(persona_dir: Path) -> None:
    from brain.initiate.emit import read_candidates
    from brain.memory.store import Memory

    _seed_vocab(persona_dir, ["love", "joy", "grief"])
    disk = MemoryStore(persona_dir / "memories.db")
    try:
        eng = _go_live(persona_dir, disk, love=6.0)
        # A making counts for the engine; the old conversation-only aggregate never saw it.
        disk.create(Memory.create_new(content="made", memory_type="making", domain="interior", emotions={"joy": 8.0}))
        _blend_rows(disk, {"love": 9.0, "grief": 8.0})
        eng.note_active(datetime.now(UTC))
        expected = {name: r.short for name, r in eng.read().items() if r.short > 0}
        assert "joy" in expected

        run_growth_tick(persona_dir, disk, datetime.now(UTC))
    finally:
        disk.close()

    [candidate] = [c for c in read_candidates(persona_dir) if c.source_id.startswith("vocabulary_emotion:")]
    assert candidate.emotional_snapshot is not None
    assert candidate.emotional_snapshot.vector == pytest.approx(expected)


def test_old_memories_do_not_form_a_blend_when_live(persona_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from brain.emotion import engine_utils
    from tests.unit.brain.emotion._engine_support import half_fade

    _seed_vocab(persona_dir, ["love", "joy", "grief"])
    disk = MemoryStore(persona_dir / "memories.db")
    try:
        _go_live(persona_dir, disk)
        _blend_rows(disk, {"love": 7.0, "grief": 7.0})
        # Faded to 3.5 each: below the 6.0 floor, so no blend.
        monkeypatch.setattr(engine_utils, "faded_emotion", half_fade)

        result = run_growth_tick(persona_dir, disk, datetime.now(UTC))
    finally:
        disk.close()

    assert result.emotions_added == 0
    assert _added_names(persona_dir) == []


def test_top_two_are_chosen_by_faded_value(persona_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from brain.emotion import engine_utils

    _seed_vocab(persona_dir, ["love", "joy", "grief"])
    disk = MemoryStore(persona_dir / "memories.db")
    try:
        _go_live(persona_dir, disk)
        _blend_rows(disk, {"love": 9.0, "grief": 8.0, "joy": 7.0})
        # love fades to 4.5 while grief and joy hold, so the strongest two are grief and joy.
        monkeypatch.setattr(
            engine_utils,
            "faded_emotion",
            lambda memory, emotion, half_lives, switch_over_at, now: memory._stored_emotions[emotion]
            * (0.5 if emotion == "love" else 1.0),
        )

        run_growth_tick(persona_dir, disk, datetime.now(UTC))
    finally:
        disk.close()

    assert _added_names(persona_dir) == ["grief_joy_blend"]


@pytest.mark.no_base_failure(reason="pins the switch-off tick, which is today's")
def test_off_tick_is_unchanged(persona_dir: Path) -> None:
    _seed_vocab(persona_dir, ["love", "joy", "grief"])
    disk = MemoryStore(persona_dir / "memories.db")
    try:
        _blend_rows(disk, {"love": 9.0, "grief": 8.0, "joy": 7.0})
        run_growth_tick(persona_dir, disk, datetime.now(UTC))
    finally:
        disk.close()

    assert _added_names(persona_dir) == ["grief_love_blend"]

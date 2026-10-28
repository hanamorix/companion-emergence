"""Group D of the json_io phase-2 migration: modules with JSON state reads, heals and backup saves.

Main's rule: a migration never WIDENS what a caller swallows. Each reader below is in the mode the site map
gives it (DEFAULT: a handler swallowed an I/O fault before; CONTENT: only bad content fell back; RAISE: nothing
was swallowed; RESTORE: the two budget files). A directory where the file should be is an I/O fault on every
platform. Writers: the file parses back with the stdlib, a second save replaces it, no temp file is left.
Each test calls the REAL function on a ``tmp_path``. Nothing here touches a persona directory.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.health.json_io import JsonFileError

NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)


def _imp(mod: str):
    import importlib

    return importlib.import_module(mod)


# name -> (relative file, call(pd), mode, what a missing file returns (checked by `==` unless None))
READERS = {
    "forgetting._load_forgetting_state": (
        "forgetting_state.json", lambda pd: _imp("brain.forgetting")._load_forgetting_state(pd), "DEFAULT", {}),
    "forgetting._load_migration_grace": (
        "source-manifest.json", lambda pd: _imp("brain.forgetting")._load_migration_grace(pd), "DEFAULT", (None, 0.0)),
    "crystallizers.reflex._load_emotion_vocabulary": (
        "emotion_vocabulary.json",
        lambda pd: _imp("brain.growth.crystallizers.reflex")._load_emotion_vocabulary(pd), "DEFAULT", set()),
    "soul_candidate_repair._load_state": (
        "soul_candidate_repair_state.json",
        lambda pd: _imp("brain.health.soul_candidate_repair")._load_state(pd), "CONTENT", None),
    "vocab_repair._load_state": (
        "vocab_repair_state.json", lambda pd: _imp("brain.health.vocab_repair")._load_state(pd), "CONTENT", None),
    "emotion_backfill._consume_budget": (
        "emotion_backfill_budget.json",
        lambda pd: _imp("brain.ingest.emotion_backfill")._consume_budget(pd, now=NOW, cap=5), "CONTENT", True),
    "kindled_link.cadence.load_tick_cadence": (
        "kindled_link/tick_cadence.json",
        lambda pd: _imp("brain.kindled_link.cadence").load_tick_cadence(pd), "DEFAULT", {"last_run": None}),
    "recovery.cli._bridge_is_live": (
        "bridge.json", lambda pd: _imp("brain.recovery.cli")._bridge_is_live(pd), "DEFAULT", False),
    "reflection_gate._load": (
        "reflection_state.json", lambda pd: _imp("brain.chat.reflection_gate")._load(pd), "RAISE", {}),
    "migrator.companion_emergence._read_source_lived_age": (
        "felt_time_state.json",
        lambda pd: _imp("brain.migrator.companion_emergence")._read_source_lived_age(pd), "DEFAULT", 0.0),
    "migrator.og_interests.extract_soul_names_best_effort": (
        "nell_soul.json",
        lambda pd: _imp("brain.migrator.og_interests").extract_soul_names_best_effort(pd), "CONTENT", set()),
    "persona_state._build_connection": (
        "heartbeat_state.json", lambda pd: _imp("brain.bridge.persona_state")._build_connection(pd), "DEFAULT", None),
    "persona_state._build_interior": (
        "daemon_state.json", lambda pd: _imp("brain.bridge.persona_state")._build_interior(pd), "DEFAULT", None),
    "self_model.cadence.load_or_init": (
        "cadence/self_model_cadence.json",
        lambda pd: _imp("brain.self_model.cadence").load_or_init(pd, now=NOW), "DEFAULT", None),
    "soul.cadence.load_cadence_state": (
        "cadence/soul_review_cadence.json",
        lambda pd: _imp("brain.soul.cadence").load_cadence_state(pd), "DEFAULT", None),
}


def _file_for(name: str, pd: Path) -> Path:
    rel = READERS[name][0]
    if name.startswith("self_model.cadence"):
        return _imp("brain.self_model.cadence")._state_path(pd)
    if name.startswith("soul.cadence"):
        return _imp("brain.soul.cadence")._state_path(pd)
    return pd / rel


@pytest.mark.parametrize("name", sorted(READERS))
def test_group_d_reader_missing_file_does_not_raise(tmp_path: Path, name: str) -> None:
    _rel, call, _mode, expected = READERS[name]
    pd = tmp_path / "persona"
    pd.mkdir()
    got = call(pd)
    if expected is not None:
        assert got == expected


@pytest.mark.parametrize("name", sorted(READERS))
def test_group_d_reader_corrupt_bytes(tmp_path: Path, name: str) -> None:
    _rel, call, mode, _expected = READERS[name]
    pd = tmp_path / "persona"
    pd.mkdir()
    f = _file_for(name, pd)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"\xff{not json")
    if mode == "RAISE":
        with pytest.raises(ValueError):
            call(pd)
    else:
        call(pd)  # DEFAULT and CONTENT: bad content takes the fallback


@pytest.mark.parametrize("name", sorted(READERS))
def test_group_d_reader_io_fault_follows_its_mode(tmp_path: Path, name: str) -> None:
    _rel, call, mode, _expected = READERS[name]
    pd = tmp_path / "persona"
    pd.mkdir()
    f = _file_for(name, pd)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.mkdir()
    if name == "recovery.cli._bridge_is_live":
        assert call(pd) is False  # not a file: the is_file probe answers
        return
    if name == "self_model.cadence.load_or_init":
        with pytest.raises(OSError):  # the read falls back, then the "last ran now" init cannot be written
            call(pd)
    elif mode == "DEFAULT":
        call(pd)  # a handler swallowed an I/O fault before; the module does now
    else:
        with pytest.raises(OSError) as exc:
            call(pd)
        assert isinstance(exc.value, JsonFileError) and exc.value.kind == "io"


def test_group_d_consume_budget_corrupt_counter_restarts_but_an_unreadable_one_is_not_reset(tmp_path: Path) -> None:
    from brain.ingest.emotion_backfill import _consume_budget

    pd = tmp_path
    path = pd / "emotion_backfill_budget.json"
    for _ in range(3):
        assert _consume_budget(pd, now=NOW, cap=3) is True
    assert _consume_budget(pd, now=NOW, cap=3) is False  # cap reached
    path.write_bytes(b"{torn")  # corrupt content: as before, starts from an empty counter
    assert _consume_budget(pd, now=NOW, cap=3) is True
    path.unlink()
    path.mkdir()  # unreadable: the daily cap must NOT reset
    with pytest.raises(OSError):
        _consume_budget(pd, now=NOW, cap=3)


def test_group_d_files_pending_read(tmp_path: Path) -> None:
    from brain.files.pending import _read

    p = tmp_path / "r.json"
    assert _read(p) is None
    p.write_text('{"id": "a"}')
    assert _read(p) == {"id": "a"}
    p.write_text("[1]")
    assert _read(p) is None  # not a record
    p.write_bytes(b"\xff")
    assert _read(p) is None


def test_group_d_creative_dna_bundled_default_is_read(tmp_path: Path) -> None:
    from brain.creative.dna import _default_factory

    assert "core_voice" in _default_factory()


def test_group_d_og_reader_hashes_the_bytes_it_parses(tmp_path: Path) -> None:
    from brain.migrator.og import OGReader

    d = tmp_path / "og"
    d.mkdir()
    (d / "memories_v2.json").write_bytes(b'[{"a": 1}]')
    reader = OGReader(d)
    assert reader.read_memories() == [{"a": 1}]
    assert [m.size_bytes for m in reader.manifest()] == [len(b'[{"a": 1}]')]


def test_group_d_relevance_judge_snapshot_complete_manifest_fallbacks(tmp_path: Path, caplog) -> None:
    from brain.memory.relevance_judge import _snapshot_complete

    repo = tmp_path / "models--x"
    snap = repo / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    assert _snapshot_complete(snap) is True  # no manifest: trusts the download, with a warning
    trees = repo / "trees"
    trees.mkdir()
    (trees / "rev1.json").write_bytes(b"\xff{")
    assert _snapshot_complete(snap) is True
    (trees / "rev1.json").write_text(json.dumps({"files": {"a.bin": {"size": 3}}}))
    assert _snapshot_complete(snap) is False  # a listed file is absent
    (snap / "a.bin").write_bytes(b"abc")
    assert _snapshot_complete(snap) is True


# ---------------------------------------------------------------------------------------------
# The two budget files: RESTORE (guard R17)
# ---------------------------------------------------------------------------------------------


def test_group_d_attunement_budget_restores_a_corrupt_file_from_the_previous_save_and_fails_closed_without_one(
    tmp_path: Path,
) -> None:
    from brain.attunement import budget

    pd = tmp_path
    for _ in range(2):
        assert budget.consume_call(pd, now=NOW, cap=5) is True
    path = budget._budget_path(pd)
    path.write_bytes(b"{torn")
    assert budget.get_remaining(pd, now=NOW, cap=5) == 4  # restored from the previous save (one call lost)
    path.write_bytes(b"{torn")
    for bak in path.parent.glob("daily_budget.json.bak*"):
        bak.unlink()
    assert budget.get_remaining(pd, now=NOW, cap=5) == 0  # no good copy: still fails CLOSED


def test_group_d_attunement_budget_unreadable_file_propagates(tmp_path: Path) -> None:
    from brain.attunement import budget

    p = budget._budget_path(tmp_path)
    p.parent.mkdir(parents=True)
    p.mkdir()
    with pytest.raises(OSError):
        budget.get_remaining(tmp_path, now=NOW, cap=5)


def test_group_d_articulate_budget_restore_corrupt_and_io(tmp_path: Path) -> None:
    from brain.self_model import articulate

    pd = tmp_path
    articulate._save_budget(pd, {"date": "2026-01-01", "count": 1})
    articulate._save_budget(pd, {"date": "2026-01-01", "count": 2})
    path = articulate._budget_path(pd)
    path.write_bytes(b"{torn")
    assert articulate._load_budget(pd) == {"date": "2026-01-01", "count": 1}  # the previous save
    path.write_bytes(b"{torn")
    for bak in path.parent.glob("*.bak*"):
        bak.unlink()
    with pytest.raises(ValueError):
        articulate._load_budget(pd)  # no good copy: the caller keeps its fail-safe-permissive branch
    path.unlink()
    path.mkdir()
    with pytest.raises(OSError) as exc:
        articulate._load_budget(pd)
    assert not isinstance(exc.value, ValueError)  # an I/O fault must not look like "corrupt: allow"


# ---------------------------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------------------------


def _no_temp_left(directory: Path) -> None:
    left = [p.name for p in directory.iterdir() if ".tmp" in p.name or p.name.endswith(".new")]
    assert not left, left


def test_group_d_writers_roundtrip_and_leave_no_temp(tmp_path: Path) -> None:
    from brain.attunement import budget
    from brain.chat import pass2_queue, reflection_gate
    from brain.forgetting import _persist_forgetting_state
    from brain.health.vocab_repair import _atomic_write_vocab
    from brain.memory.embedding_backfill import _load_cursor, _save_cursor
    from brain.self_model import articulate, resolve
    from brain.soul import cadence as soul_cadence
    from brain.update import overlay

    pd = tmp_path / "persona"
    pd.mkdir()
    _persist_forgetting_state(pd, {"m1": 2})
    _persist_forgetting_state(pd, {"m1": 3})
    assert json.loads((pd / "forgetting_state.json").read_text()) == {"m1": 3}
    assert json.loads((pd / "forgetting_state.json.bak1").read_text()) == {"m1": 2}
    _atomic_write_vocab(pd / "emotion_vocabulary.json", {"version": 1, "emotions": [{"name": "café"}]})
    assert (pd / "emotion_vocabulary.json").read_bytes().isascii()
    budget._save(pd, {"date": "d", "count": 1})
    assert json.loads(budget._budget_path(pd).read_text()) == {"date": "d", "count": 1}
    pass2_queue._save_queue_unlocked(pd, [{"id": "x"}])
    assert pass2_queue._load_queue_unlocked(pd) == [{"id": "x"}]
    reflection_gate._atomic_write(pd, {"k": {"last_reflection_ts": "t"}})
    assert reflection_gate._load(pd) == {"k": {"last_reflection_ts": "t"}}
    _save_cursor(pd, "model-1", ("2026-01-01T00:00:00", "id1"))
    assert _load_cursor(pd, "model-1") == ("2026-01-01T00:00:00", "id1")
    assert _load_cursor(pd, "model-2") is None  # a model swap resets the cursor
    articulate._save_budget(pd, {"date": "d", "count": 1})
    assert articulate._load_budget(pd) == {"date": "d", "count": 1}
    resolve._save_audit(pd, {"gaps_surfaced": 2, "reconciles_called": 1})
    assert resolve.load_audit(pd) == {"gaps_surfaced": 2, "reconciles_called": 1}
    soul_cadence.save_cadence_state(pd, soul_cadence.ReviewCadenceState(next_review_at=NOW, consecutive_failures=2))
    got = soul_cadence.load_cadence_state(pd)
    assert got.next_review_at == NOW and got.consecutive_failures == 2
    root = tmp_path / "overlay"
    overlay._write_state(root, {"dir": "a"}, None)
    assert overlay.read_state(root) == {"active": {"dir": "a"}, "previous": None}
    for d in (pd, pd / "cadence", root):
        if d.exists():
            _no_temp_left(d)


def test_group_d_kit_source_manifest_and_best_effort_writes(tmp_path: Path) -> None:
    from brain.migrator.emergence_kit import _write_kit_source_manifest

    path = tmp_path / "source-manifest.json"
    _write_kit_source_manifest(
        path, input_dir=tmp_path, memories_imported=1, memories_skipped=0,
        crystallizations_imported=2, personality_copied=True,
    )
    assert json.loads(path.read_text())["summary"]["crystallizations_imported"] == 2
    blocker = tmp_path / "file"
    blocker.write_text("x")
    _write_kit_source_manifest(  # best-effort: an unwritable target is a warning, not an error
        blocker / "m.json", input_dir=tmp_path, memories_imported=0, memories_skipped=0,
        crystallizations_imported=0, personality_copied=False,
    )


def test_group_d_scheduler_appends_to_the_vocabulary_and_raises_on_a_corrupt_one(tmp_path: Path) -> None:
    from brain.growth.proposal import EmotionProposal
    from brain.growth.scheduler import _append_to_vocabulary

    vocab = tmp_path / "emotion_vocabulary.json"
    prop = EmotionProposal(
        name="wistful",
        description="a soft longing",
        decay_half_life_days=7.0,
        evidence_memory_ids=(),
        score=0.5,
        relational_context=None,
    )
    _append_to_vocabulary(vocab, prop)  # missing file: starts a fresh vocabulary
    _append_to_vocabulary(vocab, prop)
    assert [e["name"] for e in json.loads(vocab.read_text())["emotions"]] == ["wistful", "wistful"]
    vocab.write_bytes(b"\xff{torn")
    with pytest.raises(ValueError):  # RAISE: a corrupt vocabulary is never silently replaced
        _append_to_vocabulary(vocab, prop)


def test_group_d_interest_spawn_state_is_read_with_a_fresh_day_fallback_and_written_back(tmp_path: Path) -> None:
    from brain.chat.extractor import InterestCandidate, _apply_interest_candidate

    state_file = tmp_path / "interest_spawn_state.json"
    _apply_interest_candidate(InterestCandidate(topic="astronomy", keywords=["stars"], why="loves it"), tmp_path)
    assert json.loads(state_file.read_text())["count"] == 1
    state_file.write_bytes(b"\xff{torn")  # corrupt: a fresh day, as before
    _apply_interest_candidate(InterestCandidate(topic="birdsong", keywords=[], why="w"), tmp_path)
    assert json.loads(state_file.read_text())["count"] == 1

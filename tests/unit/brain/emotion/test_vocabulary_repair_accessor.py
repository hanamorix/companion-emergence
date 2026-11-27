"""#803 step 16 (VC1): the damaged-file repair of her vocabulary goes through the accessor.

The owner (the bridge) repairs a damaged `emotion_vocabulary.json` through `vocabulary_file.repair`: json_io's heal run whole under the
vocabulary file's own lock, ending with the copy matching the repaired file. A process that is not the owner never repairs: `walk_persona`
reads the file without repair and reports the damage as an alarm. All personas here are tmp directories with synthetic entries.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from brain import dev_constants
from brain.cli import main
from brain.emotion import persona_loader, vocabulary, vocabulary_file
from brain.emotion.vocabulary import Emotion
from brain.growth import scheduler
from brain.health import json_io
from brain.health.walker import walk_persona
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.utils import file_lock, file_ops

VOCAB = "emotion_vocabulary.json"


def _entry(name: str, description: str = "her own words") -> dict:
    return {"name": name, "description": description, "category": "persona_extension", "decay_half_life_days": 6.0, "intensity_clamp": 10.0}


def _write_file(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps({"version": 1, "emotions": entries}), encoding="utf-8")


def _register(name: str) -> None:
    vocabulary.register(Emotion(name, "registered by hand", "persona_extension", 6.0))


def _in_registry(name: str) -> bool:
    """Registry membership with the check off, so a lazy reload by the read itself cannot answer for the repair."""
    with vocabulary.suspended():
        return vocabulary.get(name) is not None


@pytest.fixture
def persona(tmp_path: Path) -> Path:
    persona_dir = tmp_path / "personas" / "testpersona"
    persona_dir.mkdir(parents=True)
    MemoryStore(db_path=persona_dir / "memories.db").close()
    HebbianMatrix(db_path=persona_dir / "hebbian.db").close()
    return persona_dir


@pytest.fixture
def path(persona: Path) -> Path:
    return persona / VOCAB


@pytest.fixture
def owner() -> None:
    vocabulary.set_bridge_owner(True)  # the autouse registry fixture puts the flag back


# ----------------------------------------------------------------------------------------------------------------
# Criterion 1: a repair through the accessor updates the copy
# ----------------------------------------------------------------------------------------------------------------


def test_a_reset_to_the_default_drops_the_entries_the_copy_held(path, owner):
    _register("only in the copy")
    path.write_text("{ not json", encoding="utf-8")
    res = vocabulary_file.read(path)  # the HEAL default
    assert res.anomaly is not None and res.anomaly.action == "reset_to_default"
    assert not _in_registry("only in the copy")  # the copy follows the repaired file: an empty one


def test_a_restore_from_a_backup_gives_the_copy_the_backups_entries(path, owner):
    _register("only in the copy")
    _write_file(path.with_name(VOCAB + ".bak1"), [_entry("from the backup")])
    path.write_text("{ not json", encoding="utf-8")
    res = vocabulary_file.read(path)
    assert res.anomaly is not None and res.anomaly.action.startswith("restored_from_bak")
    assert _in_registry("from the backup")
    assert not _in_registry("only in the copy")


def test_the_repair_leaves_a_followed_copy_matching_the_repaired_file(path, owner):
    _write_file(path, [_entry("kept one")])
    persona_loader.load_persona_vocabulary(path)
    path.write_text("{ not json", encoding="utf-8")
    vocabulary_file.read(path)
    # no read of the copy has happened: the key the repair noted is the key of the file it left
    assert vocabulary.copy_matches(path, file_ops.stat_path(path).key)
    assert not _in_registry("kept one")


def test_the_growth_schedulers_read_repairs_through_the_accessor(path, persona, owner):
    _register("only in the copy")
    path.write_text("{ not json", encoding="utf-8")
    names, anomaly = scheduler._read_current_vocabulary_names(persona)
    assert anomaly is not None and anomaly.action == "reset_to_default"
    assert "only_in_the_copy" not in names
    assert not _in_registry("only in the copy")


def test_walk_persona_in_the_bridge_repairs_through_the_accessor(path, persona, owner):
    _register("only in the copy")
    path.write_text("{ not json", encoding="utf-8")
    anomalies = walk_persona(persona)
    vocab = [a for a in anomalies if a.file == VOCAB]
    assert len(vocab) == 1 and vocab[0].action == "reset_to_default"
    assert not _in_registry("only in the copy")
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1, "emotions": []}


def test_the_loaders_heal_branch_leaves_the_copy_matching_the_repaired_file(path, owner):
    _write_file(path, [_entry("kept one")])
    persona_loader.load_persona_vocabulary(path)
    path.write_text("{ not json", encoding="utf-8")
    _count, anomaly = persona_loader.load_persona_vocabulary_with_anomaly(path)
    assert anomaly is not None and anomaly.action == "reset_to_default"
    assert not _in_registry("kept one")
    assert vocabulary.copy_matches(path, file_ops.stat_path(path).key)


# ----------------------------------------------------------------------------------------------------------------
# Criterion 2: a process that is not the owner reports and writes nothing
# ----------------------------------------------------------------------------------------------------------------

_DAMAGE = [
    pytest.param(b"", "json_parse_error", id="empty"),
    pytest.param(b"{ not json", "json_parse_error", id="malformed"),
    pytest.param(b"\xff\xfe\x00bad", "json_parse_error", id="encoding"),
    pytest.param(b"[1, 2]", "schema_mismatch", id="schema-not-a-dict"),
    pytest.param(b'{"version": 1, "emotions": 3}', "schema_mismatch", id="schema-no-list"),
]


@pytest.mark.parametrize(("damaged", "kind"), _DAMAGE)
def test_a_process_that_is_not_the_owner_reports_the_damage_as_an_alarm_and_writes_nothing(persona, path, damaged, kind):
    assert not vocabulary.is_bridge_owner()
    path.write_bytes(damaged)
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    names_before = sorted(p.name for p in persona.iterdir() if p.name.startswith(VOCAB))
    anomalies = [a for a in walk_persona(persona) if a.file == VOCAB]
    assert len(anomalies) == 1
    assert anomalies[0].action == "alarmed_unrecoverable"
    assert anomalies[0].kind == kind
    assert anomalies[0].quarantine_path is None
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before  # no write
    assert sorted(p.name for p in persona.iterdir() if p.name.startswith(VOCAB)) == names_before  # no quarantine file, no backup, no lock sidecar left behind


def test_nell_health_check_beside_a_running_bridge_reports_the_alarm_exits_2_and_writes_nothing(monkeypatch, tmp_path, persona, path, capsys):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    assert not vocabulary.is_bridge_owner()  # the checking process is not the bridge
    path.write_text("{ not json", encoding="utf-8")
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    rc = main(["health", "check", "--persona", "testpersona"])
    out = capsys.readouterr().out
    assert rc == 2
    assert VOCAB in out and "unrecoverable" in out
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert not [p for p in persona.iterdir() if "corrupt" in p.name or ".bak" in p.name or p.name.endswith(".heal.lock")]


def test_a_healthy_or_missing_file_gives_a_process_that_is_not_the_owner_no_anomaly(persona, path):
    assert [a for a in walk_persona(persona) if a.file == VOCAB] == []  # absent
    _write_file(path, [_entry("fine")])
    assert [a for a in walk_persona(persona) if a.file == VOCAB] == []


@pytest.mark.parametrize(
    ("read_kind", "anomaly_kind"),
    [("empty", "json_parse_error"), ("malformed", "json_parse_error"), ("encoding", "json_parse_error"), ("shape", "schema_mismatch"), ("schema", "schema_mismatch")],
)
def test_the_anomaly_kind_of_a_read_kind_is_the_one_mapping_json_io_heals_with(read_kind, anomaly_kind):
    assert json_io.anomaly_kind_for(read_kind) == anomaly_kind


# ----------------------------------------------------------------------------------------------------------------
# Criterion 3: a failed repair
# ----------------------------------------------------------------------------------------------------------------


def test_a_failed_repair_leaves_the_file_and_the_copy_and_logs(path, owner, monkeypatch, caplog):
    _write_file(path, [_entry("kept one")])
    persona_loader.load_persona_vocabulary(path)
    path.write_text("{ not json", encoding="utf-8")
    before = (path.read_bytes(), path.stat().st_mtime_ns)

    def refuse(*_a, **_k):
        raise OSError("no room for the quarantine copy")

    monkeypatch.setattr(json_io, "_quarantine_copy", refuse)
    with caplog.at_level(logging.WARNING), pytest.raises(json_io.JsonFileError):
        vocabulary_file.read(path)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert _in_registry("kept one")  # the copy is as it was
    assert any("repair" in r.getMessage() and VOCAB in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


def test_a_repair_that_cannot_get_the_lock_leaves_the_file_and_the_copy_and_logs(path, owner, monkeypatch, caplog):
    _write_file(path, [_entry("kept one")])
    persona_loader.load_persona_vocabulary(path)
    path.write_text("{ not json", encoding="utf-8")
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    monkeypatch.setattr(dev_constants, "EMOTION_VOCABULARY_LOCK_WAIT_S", 0.05)
    with file_lock.file_lock(path):  # another writer holds her file's lock
        with caplog.at_level(logging.WARNING), pytest.raises(OSError):
            vocabulary_file.read(path)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert _in_registry("kept one")
    assert any("repair" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


# ----------------------------------------------------------------------------------------------------------------
# Criterion 4: an io error on this file
# ----------------------------------------------------------------------------------------------------------------


def test_an_io_error_on_this_file_builds_no_anomaly_is_logged_and_the_walk_goes_on(persona, path, monkeypatch, caplog):
    from brain.health import walker

    (persona / "memories.db").write_bytes(b"not sqlite data at all")  # a file after it in the walk
    (persona / "user_preferences.json").write_text("{not json", encoding="utf-8")  # a file before it
    real = walker.read_json_result

    def read(p, **kwargs):
        if Path(p).name == VOCAB:
            raise json_io.JsonFileError(p, "io", "disk said no")
        return real(p, **kwargs)

    monkeypatch.setattr(walker, "read_json_result", read)
    with caplog.at_level(logging.WARNING):
        anomalies = walk_persona(persona)
    assert not [a for a in anomalies if a.file == VOCAB]
    assert any(a.file == "user_preferences.json" for a in anomalies)
    assert any(a.kind == "sqlite_integrity_fail" for a in anomalies)
    assert any(VOCAB in r.getMessage() and "disk said no" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


# ----------------------------------------------------------------------------------------------------------------
# Criterion 5: the owner's reload repairs a damaged file
# ----------------------------------------------------------------------------------------------------------------


def test_the_owners_reload_repairs_a_damaged_file_and_the_copy_matches(path, owner, caplog):
    _write_file(path, [_entry("kept one")])
    persona_loader.load_persona_vocabulary(path)
    path.write_text("{ not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert vocabulary.get("kept one") is None  # the next read of the copy found the damage, repaired it, and the copy follows
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1, "emotions": []}
    assert [p for p in path.parent.iterdir() if ".corrupt-" in p.name]  # the damaged file was kept aside
    assert any("emotion_vocabulary anomaly detected" in r.getMessage() for r in caplog.records)
    assert vocabulary.copy_matches(path, file_ops.stat_path(path).key)


def test_the_owners_reload_restores_a_backup_and_cleans_what_the_backup_holds(path, owner):
    _write_file(path, [_entry("kept one")])
    persona_loader.load_persona_vocabulary(path)
    _write_file(path.with_name(VOCAB + ".bak1"), [_entry("tender longing"), _entry("tender_longing", "second"), _entry("joy")])
    path.write_text("{ not json", encoding="utf-8")
    assert vocabulary.get("tender longing") is not None
    names = [e["name"] for e in json.loads(path.read_text(encoding="utf-8"))["emotions"]]
    assert names == ["tender longing"]  # the core entry is removed, the twins merged under the first spelling


# ----------------------------------------------------------------------------------------------------------------
# Criterion 6: lock order
# ----------------------------------------------------------------------------------------------------------------


def test_the_heal_runs_inside_her_files_lock(path, owner, monkeypatch):
    path.write_text("{ not json", encoding="utf-8")
    seen: dict[str, bool] = {}
    real = json_io._heal_locked

    def spy(*args, **kwargs):
        with file_lock.file_lock(path, blocking=False) as acquired:
            seen["free"] = acquired  # False: the repair holds the vocabulary file's own lock while json_io heals
        return real(*args, **kwargs)

    monkeypatch.setattr(json_io, "_heal_locked", spy)
    res = vocabulary_file.read(path)
    assert seen == {"free": False}
    assert res.anomaly is not None


# ----------------------------------------------------------------------------------------------------------------
# Criterion 7: a healthy or missing file takes no lock
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.no_base_failure(reason="control: the old HEAL read took no vocabulary lock for a healthy or missing file either; this pins that the repair keeps it")
def test_a_healthy_or_missing_file_takes_no_lock_and_writes_nothing(path, owner, monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("the vocabulary lock was taken")

    monkeypatch.setattr(file_lock, "locked_within", forbidden)
    assert vocabulary_file.read(path).outcome == "missing"
    _write_file(path, [_entry("fine")])
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    res = vocabulary_file.read(path)
    assert res.outcome == "ok" and res.anomaly is None
    assert (path.read_bytes(), os.stat(path).st_mtime_ns) == before

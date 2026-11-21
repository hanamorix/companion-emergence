"""How the vocabulary loader writes `emotion_vocabulary.json` (#784 follow-ups).

Every write the loader makes to the file (the heal, the rebuild of a missing file, the rebuild after a reset) happens under
`file_lock(vocab_path)`, the lock `register_new_word` takes, on data read inside that lock, and with the file's own backup
count. The heal writes only entries the loader can read back.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain.emotion import persona_loader, vocabulary
from brain.emotion.persona_loader import load_persona_vocabulary
from brain.memory.store import Memory, MemoryStore
from brain.utils.file_lock import file_lock


def _store_with(*names: str) -> MemoryStore:
    store = MemoryStore(":memory:")
    for name in names:
        store.create(
            Memory.create_new(content=f"felt {name}", memory_type="conversation", domain="us", emotions={name: 8.0})
        )
    return store


def _lock_is_held(path: Path) -> bool:
    with file_lock(path, blocking=False) as acquired:
        return not acquired


def _entries(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["emotions"]


@pytest.mark.parametrize("scenario", ["heal", "missing_file_rebuild", "reset_rebuild"])
def test_every_loader_write_holds_the_vocabulary_lock_and_uses_the_files_backups(
    scenario: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vocab_path = tmp_path / "emotion_vocabulary.json"
    if scenario == "heal":
        vocab_path.write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")
    elif scenario == "reset_rebuild":
        vocab_path.write_text("{not json", encoding="utf-8")
    vocabulary._unregister("omicron_feel")
    monkeypatch.setattr(persona_loader, "compute_treatment", lambda *_a, **_k: SimpleNamespace(backup_count=7))
    writes: list[tuple[bool, int | None]] = []
    real_write = persona_loader.write_json

    def spy(path, data, **kwargs):  # noqa: ANN001, ANN202
        if Path(path) == vocab_path:
            writes.append((_lock_is_held(vocab_path), kwargs.get("backups")))
        return real_write(path, data, **kwargs)

    monkeypatch.setattr(persona_loader, "write_json", spy)
    store = _store_with("omicron_feel")
    try:
        load_persona_vocabulary(vocab_path, store=store)
    finally:
        store.close()

    assert writes and all(write == (True, 7) for write in writes)
    assert "omicron_feel" in {e["name"] for e in _entries(vocab_path)}


@pytest.mark.no_base_failure(
    reason="guards _vocab_backups' fallback: the base wrote with a fixed 3 backups and never read the audit log"
)
def test_an_unreadable_audit_log_falls_back_to_three_backups_instead_of_failing_the_write(tmp_path: Path) -> None:
    """The backup count reads heartbeats.log.jsonl; a bad line there must not stop the rebuild from memories."""
    (tmp_path / "heartbeats.log.jsonl").write_text('{"anomalies": 5}\n', encoding="utf-8")
    vocab_path = tmp_path / "emotion_vocabulary.json"
    vocabulary._unregister("eta_feel")
    store = _store_with("eta_feel")
    try:
        load_persona_vocabulary(vocab_path, store=store)
    finally:
        store.close()

    assert "eta_feel" in {e["name"] for e in _entries(vocab_path)}


def test_a_load_with_nothing_to_write_takes_no_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every command and bridge start loads with a store; a healthy, already-healed file is only read, so none waits on a writer."""
    from contextlib import contextmanager

    from brain.health.json_io import write_json
    from brain.paths import cadence_state_path

    vocab_path = tmp_path / "emotion_vocabulary.json"
    entry = {"name": "zeta_feel", "description": "d", "category": "persona_extension", "decay_half_life_days": 9.0}
    vocab_path.write_text(json.dumps({"version": 1, "emotions": [entry]}), encoding="utf-8")
    write_json(cadence_state_path(tmp_path, "vocab_heal_final.json"), {"healed_at": "x", "count": 0})
    locked: list[Path] = []

    @contextmanager
    def recording_lock(path, *args, **kwargs):  # noqa: ANN001, ANN202
        locked.append(Path(path))
        yield True

    monkeypatch.setattr(persona_loader, "file_lock", recording_lock)
    vocabulary._unregister("zeta_feel")
    store = _store_with("zeta_feel")
    try:
        load_persona_vocabulary(vocab_path, store=store)
    finally:
        store.close()

    assert locked == []
    assert vocabulary.get("zeta_feel") is not None


def test_a_malformed_entry_does_not_hide_its_word_from_the_heal(tmp_path: Path) -> None:
    """`{"name": x}` can't be registered; the heal replaces it with a whole entry rather than counting it as present."""
    vocab_path = tmp_path / "emotion_vocabulary.json"
    vocab_path.write_text(json.dumps({"version": 1, "emotions": [{"name": "pi_feel"}]}), encoding="utf-8")
    vocabulary._unregister("pi_feel")
    store = _store_with("pi_feel")
    try:
        load_persona_vocabulary(vocab_path, store=store)
    finally:
        store.close()

    assert vocabulary.get("pi_feel") is not None
    pi_entries = [e for e in _entries(vocab_path) if e["name"] == "pi_feel"]
    assert len(pi_entries) == 1 and "description" in pi_entries[0]


def test_a_registered_word_with_a_non_finite_half_life_heals_with_finite_values(tmp_path: Path) -> None:
    """JSON has no NaN: writing the registered fields back as they are would raise on every store load."""
    import math

    from brain.emotion.vocabulary import Emotion

    vocab_path = tmp_path / "emotion_vocabulary.json"
    vocab_path.write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")
    vocabulary._unregister("rho_feel")
    vocabulary.register(Emotion("rho_feel", "d", "persona_extension", float("nan"), float("inf")))
    store = _store_with("rho_feel")
    try:
        load_persona_vocabulary(vocab_path, store=store)
    finally:
        store.close()

    (rho,) = [e for e in _entries(vocab_path) if e["name"] == "rho_feel"]
    assert math.isfinite(rho["decay_half_life_days"]) and math.isfinite(rho["intensity_clamp"])


def test_a_file_entry_with_a_non_finite_number_is_not_registered_and_is_healed(tmp_path: Path) -> None:
    """Python's JSON reads NaN; a NaN half-life in the registry poisons decay and every later write of the file."""
    import math

    vocab_path = tmp_path / "emotion_vocabulary.json"
    bad = {"name": "sigma_feel", "description": "d", "category": "persona_extension", "decay_half_life_days": "NaN"}
    vocab_path.write_text(json.dumps({"version": 1, "emotions": [bad]}).replace('"NaN"', "NaN"), encoding="utf-8")
    vocabulary._unregister("sigma_feel")
    store = _store_with("sigma_feel")
    try:
        load_persona_vocabulary(vocab_path, store=store)
    finally:
        store.close()

    assert math.isfinite(vocabulary.get("sigma_feel").decay_half_life_days)
    (sigma,) = [e for e in _entries(vocab_path) if e["name"] == "sigma_feel"]
    assert math.isfinite(sigma["decay_half_life_days"])


def test_a_broken_entry_no_memory_carries_is_repaired_in_place_keeping_its_description(tmp_path: Path) -> None:
    """A NaN entry for a word no memory carries would make every heal write raise; it's fixed in place, not dropped."""
    import math

    from brain.paths import cadence_state_path

    vocab_path = tmp_path / "emotion_vocabulary.json"
    bad = {"name": "chi_feel", "description": "a crystallized meaning", "category": "persona_extension", "decay_half_life_days": "NaN"}
    vocab_path.write_text(json.dumps({"version": 1, "emotions": [bad]}).replace('"NaN"', "NaN"), encoding="utf-8")
    vocabulary._unregister("chi_feel")
    vocabulary._unregister("phi_feel")
    store = _store_with("phi_feel")
    try:
        load_persona_vocabulary(vocab_path, store=store)
    finally:
        store.close()

    by_name = {e["name"]: e for e in _entries(vocab_path)}
    assert by_name["chi_feel"]["description"] == "a crystallized meaning"
    assert math.isfinite(by_name["chi_feel"]["decay_half_life_days"])
    assert "phi_feel" in by_name
    assert vocabulary.get("chi_feel") is not None
    assert cadence_state_path(tmp_path, "vocab_heal_final.json").exists()


def test_a_non_finite_value_in_an_extra_field_becomes_null_so_the_heal_can_write(tmp_path: Path) -> None:
    """A NaN outside the core fields leaves the entry registrable but the file unwritable: every heal would raise."""
    vocab_path = tmp_path / "emotion_vocabulary.json"
    entry = {"name": "lambda_feel", "description": "d", "category": "persona_extension", "decay_half_life_days": 9.0, "score": "NaN"}
    vocab_path.write_text(json.dumps({"version": 1, "emotions": [entry]}).replace('"NaN"', "NaN"), encoding="utf-8")
    vocabulary._unregister("lambda_feel")
    vocabulary._unregister("mu2_feel")
    store = _store_with("mu2_feel")
    try:
        load_persona_vocabulary(vocab_path, store=store)
    finally:
        store.close()

    by_name = {e["name"]: e for e in _entries(vocab_path)}
    assert "mu2_feel" in by_name
    assert by_name["lambda_feel"]["score"] is None and by_name["lambda_feel"]["description"] == "d"


def test_a_broken_entry_with_a_whole_twin_is_dropped_not_repaired(tmp_path: Path) -> None:
    """A repaired `{"name": x}` ahead of the whole entry would win the registry with placeholder values."""
    from brain.health.json_io import write_json
    from brain.paths import cadence_state_path

    vocab_path = tmp_path / "emotion_vocabulary.json"
    whole = {"name": "kappa_feel", "description": "the real meaning", "category": "persona_extension", "decay_half_life_days": 30.0}
    vocab_path.write_text(json.dumps({"version": 1, "emotions": [{"name": "kappa_feel"}, whole]}), encoding="utf-8")
    write_json(cadence_state_path(tmp_path, "vocab_heal_final.json"), {"healed_at": "x", "count": 0})
    vocabulary._unregister("kappa_feel")
    store = _store_with()
    try:
        load_persona_vocabulary(vocab_path, store=store)
    finally:
        store.close()

    assert vocabulary.get("kappa_feel").description == "the real meaning"
    assert [e for e in _entries(vocab_path) if e["name"] == "kappa_feel"] == [whole]


def test_a_broken_entry_gives_its_real_description_to_a_placeholder_twin(tmp_path: Path) -> None:
    """Dropping the broken twin must not delete the only real description the word has."""
    from brain.health.json_io import write_json
    from brain.health.reconstruct import PLACEHOLDER_DESCRIPTION
    from brain.paths import cadence_state_path

    vocab_path = tmp_path / "emotion_vocabulary.json"
    broken = {"name": "nu_feel", "description": "the real meaning", "category": "persona_extension", "decay_half_life_days": "NaN"}
    whole = {"name": "Nu_Feel", "description": PLACEHOLDER_DESCRIPTION, "category": "persona_extension", "decay_half_life_days": 14.0}
    text = json.dumps({"version": 1, "emotions": [broken, whole]}).replace('"NaN"', "NaN")
    vocab_path.write_text(text, encoding="utf-8")
    write_json(cadence_state_path(tmp_path, "vocab_heal_final.json"), {"healed_at": "x", "count": 0})
    vocabulary._unregister("nu_feel")
    store = _store_with()
    try:
        load_persona_vocabulary(vocab_path, store=store)
    finally:
        store.close()

    assert vocabulary.get("nu_feel").description == "the real meaning"
    nu = [e for e in _entries(vocab_path) if vocabulary.canonical_name(e["name"]) == "nu_feel"]
    assert len(nu) == 1 and nu[0]["description"] == "the real meaning" and nu[0]["decay_half_life_days"] == 14.0


def test_two_broken_copies_of_a_word_become_one_keeping_the_real_description(tmp_path: Path) -> None:
    """With no whole entry, the copy with a real description is the one kept, so the placeholder can't win the registry."""
    vocab_path = tmp_path / "emotion_vocabulary.json"
    vocab_path.write_text(
        json.dumps({"version": 1, "emotions": [{"name": "xi_feel"}, {"name": "xi_feel", "description": "the real meaning"}]}),
        encoding="utf-8",
    )
    vocabulary._unregister("xi_feel")
    load_persona_vocabulary(vocab_path)

    assert vocabulary.get("xi_feel").description == "the real meaning"
    assert [e["description"] for e in _entries(vocab_path) if e["name"] == "xi_feel"] == ["the real meaning"]


def test_a_nested_non_finite_value_becomes_null_so_the_file_can_be_written(tmp_path: Path) -> None:
    """A NaN inside an extra field's object would make every write of the file raise, the same as a top-level one."""
    vocab_path = tmp_path / "emotion_vocabulary.json"
    entry = {"name": "rho2_feel", "description": "d", "category": "persona_extension", "decay_half_life_days": 9.0, "meta": {"score": "NaN"}}
    vocab_path.write_text(json.dumps({"version": 1, "emotions": [entry]}).replace('"NaN"', "NaN"), encoding="utf-8")
    vocabulary._unregister("rho2_feel")
    load_persona_vocabulary(vocab_path)

    (rho,) = _entries(vocab_path)
    assert rho["meta"] == {"score": None}


def test_register_new_word_repairs_a_non_finite_entry_instead_of_being_blocked_by_it(tmp_path: Path) -> None:
    """JSON can't write NaN back, so an unrepaired NaN entry for another word would refuse every new word for good."""
    import math

    from brain.emotion.persona_loader import register_new_word

    vocab_path = tmp_path / "emotion_vocabulary.json"
    bad = {"name": "upsilon_feel", "description": "kept", "category": "persona_extension", "decay_half_life_days": "NaN"}
    vocab_path.write_text(json.dumps({"version": 1, "emotions": [bad]}).replace('"NaN"', "NaN"), encoding="utf-8")
    vocabulary._unregister("tau_feel")

    assert register_new_word(vocab_path, "tau_feel") == "tau_feel"
    assert vocabulary.get("tau_feel") is not None
    by_name = {e["name"]: e for e in _entries(vocab_path)}
    assert "tau_feel" in by_name
    assert by_name["upsilon_feel"]["description"] == "kept"
    assert math.isfinite(by_name["upsilon_feel"]["decay_half_life_days"])


def test_register_new_word_repairs_a_malformed_entry_for_the_same_word(tmp_path: Path) -> None:
    """`{"name": x}` must not count as present: the word would be reported registered while the loader skips it."""
    from brain.emotion.persona_loader import register_new_word

    vocab_path = tmp_path / "emotion_vocabulary.json"
    vocab_path.write_text(json.dumps({"version": 1, "emotions": [{"name": "omega_feel", "description": "kept"}]}), encoding="utf-8")
    vocabulary._unregister("omega_feel")

    assert register_new_word(vocab_path, "omega_feel") == "omega_feel"
    assert vocabulary.get("omega_feel") is not None
    (omega,) = [e for e in _entries(vocab_path) if e["name"] == "omega_feel"]
    assert omega["description"] == "kept" and "decay_half_life_days" in omega

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


def test_the_merge_carries_an_explicit_no_decay_half_life_to_a_broken_copy(tmp_path: Path) -> None:
    """`decay_half_life_days: null` means "no decay"; a copy missing the field must not turn that into 14 days."""
    vocab_path = tmp_path / "emotion_vocabulary.json"
    copies = [{"name": "tau2_feel", "description": "the real meaning"}, {"name": "tau2_feel", "decay_half_life_days": None}]
    vocab_path.write_text(json.dumps({"version": 1, "emotions": copies}), encoding="utf-8")
    vocabulary._unregister("tau2_feel")
    load_persona_vocabulary(vocab_path)

    assert vocabulary.get("tau2_feel").decay_half_life_days is None
    (tau,) = [e for e in _entries(vocab_path) if e["name"] == "tau2_feel"]
    assert tau["decay_half_life_days"] is None and tau["description"] == "the real meaning"


def test_a_non_finite_value_outside_any_named_entry_becomes_null_so_the_heal_can_write(tmp_path: Path) -> None:
    """A NaN in an unnamed entry or a top-level field names no word, but still makes every write of the file raise."""
    vocab_path = tmp_path / "emotion_vocabulary.json"
    data = {"version": 1, "built_at_score": "NaN", "emotions": [{"name": 5, "weight": "NaN"}]}
    vocab_path.write_text(json.dumps(data).replace('"NaN"', "NaN"), encoding="utf-8")
    vocabulary._unregister("psi2_feel")
    store = _store_with("psi2_feel")
    try:
        load_persona_vocabulary(vocab_path, store=store)
    finally:
        store.close()

    written = json.loads(vocab_path.read_text(encoding="utf-8"))
    assert written["built_at_score"] is None
    assert {"name": 5, "weight": None} in written["emotions"]
    assert "psi2_feel" in {e.get("name") for e in written["emotions"]}


def test_a_corrupt_half_life_counts_for_nothing_as_the_engine_ignores_it(tmp_path: Path) -> None:
    """The engine skips the copy with no half-life and the NaN one, and reads the null: the repair keeps the null."""
    vocab_path = tmp_path / "emotion_vocabulary.json"
    copies = [
        {"name": "m2_feel", "description": "real", "category": "persona_extension"},
        {"name": "m2_feel", "decay_half_life_days": "NaN"},
        {"name": "m2_feel", "decay_half_life_days": None},
    ]
    vocab_path.write_text(json.dumps({"version": 1, "emotions": copies}).replace('"NaN"', "NaN"), encoding="utf-8")
    vocabulary._unregister("m2_feel")
    load_persona_vocabulary(vocab_path)

    assert vocabulary.get("m2_feel").decay_half_life_days is None
    (m2,) = [e for e in _entries(vocab_path) if e["name"] == "m2_feel"]
    assert m2["decay_half_life_days"] is None and m2["description"] == "real"
    assert persona_loader._repair_broken_entries(json.loads(vocab_path.read_text(encoding="utf-8"))) == []


def test_the_merge_takes_only_a_half_life_the_engine_accepts(tmp_path: Path) -> None:
    """The engine ignores a half-life of 0 or below, so the repair mustn't write one into the file and the registry."""
    vocab_path = tmp_path / "emotion_vocabulary.json"
    base = {"name": "n2_feel", "description": "real", "category": "persona_extension", "decay_half_life_days": "NaN"}
    copies = [base, {"name": "n2_feel", "decay_half_life_days": -2}, {"name": "n2_feel", "decay_half_life_days": 0}]
    copies.append({"name": "n2_feel", "decay_half_life_days": 5})
    vocab_path.write_text(json.dumps({"version": 1, "emotions": copies}).replace('"NaN"', "NaN"), encoding="utf-8")
    vocabulary._unregister("n2_feel")
    load_persona_vocabulary(vocab_path)

    assert vocabulary.get("n2_feel").decay_half_life_days == 5.0
    assert [e["decay_half_life_days"] for e in _entries(vocab_path) if e["name"] == "n2_feel"] == [5.0]


def test_a_non_finite_description_or_category_becomes_placeholder_text_not_the_word_none(tmp_path: Path) -> None:
    """Nulled, a NaN description or category would register as the text "None"; the placeholders say what's missing."""
    from brain.health.reconstruct import PLACEHOLDER_DESCRIPTION

    vocab_path = tmp_path / "emotion_vocabulary.json"
    entry = {"name": "d2_feel", "description": "NaN", "category": "NaN", "decay_half_life_days": 3.0}
    vocab_path.write_text(json.dumps({"version": 1, "emotions": [entry]}).replace('"NaN"', "NaN"), encoding="utf-8")
    vocabulary._unregister("d2_feel")
    load_persona_vocabulary(vocab_path)

    emotion = vocabulary.get("d2_feel")
    assert (emotion.description, emotion.category) == (PLACEHOLDER_DESCRIPTION, "persona_extension")
    (d2,) = _entries(vocab_path)
    assert (d2["description"], d2["category"]) == (PLACEHOLDER_DESCRIPTION, "persona_extension")
    assert persona_loader._repair_broken_entries(json.loads(vocab_path.read_text(encoding="utf-8"))) == []


@pytest.mark.parametrize("bad", [0, -2, True, "3"])
def test_a_whole_entry_with_a_half_life_the_engine_ignores_is_repaired(tmp_path: Path, bad: object) -> None:
    """The entry registers, but the engine refuses this half-life and uses its built-in one, so the repair writes that."""
    from brain import dev_constants

    vocab_path = tmp_path / "emotion_vocabulary.json"
    entry = {"name": "w2_feel", "description": "real", "category": "persona_extension", "decay_half_life_days": bad}
    vocab_path.write_text(json.dumps({"version": 1, "emotions": [entry]}), encoding="utf-8")
    vocabulary._unregister("w2_feel")
    load_persona_vocabulary(vocab_path)

    expected = dev_constants.EMOTION_DEFAULT_HALF_LIFE_DAYS  # even for "3": the engine ignores a string
    assert vocabulary.get("w2_feel").decay_half_life_days == expected
    (w2,) = _entries(vocab_path)
    assert w2["decay_half_life_days"] == expected and w2["description"] == "real"
    assert persona_loader._repair_broken_entries(json.loads(vocab_path.read_text(encoding="utf-8"))) == []


def test_a_null_description_or_category_becomes_placeholder_text_not_the_word_none(tmp_path: Path) -> None:
    """`str(None)` would register the text "None"; a real description from the word's other copy is kept instead."""
    from brain.health.reconstruct import PLACEHOLDER_DESCRIPTION

    vocab_path = tmp_path / "emotion_vocabulary.json"
    lone = {"name": "v2_feel", "description": None, "category": None, "decay_half_life_days": 3.0}
    twin = {"name": "u2_feel", "description": None, "category": "complex", "decay_half_life_days": 3.0}
    donor = {"name": "u2_feel", "description": "the real meaning", "category": None, "decay_half_life_days": None}
    vocab_path.write_text(json.dumps({"version": 1, "emotions": [lone, twin, donor]}), encoding="utf-8")
    vocabulary._unregister("v2_feel")
    vocabulary._unregister("u2_feel")
    load_persona_vocabulary(vocab_path)

    v2, u2 = vocabulary.get("v2_feel"), vocabulary.get("u2_feel")
    assert (v2.description, v2.category) == (PLACEHOLDER_DESCRIPTION, "persona_extension")
    # Both u2 copies are broken: the base is the one with the real description, but the half-life is the 3 days the
    # engine reads from the first copy.
    assert (u2.description, u2.category, u2.decay_half_life_days) == ("the real meaning", "complex", 3.0)
    assert [(e["name"], e["description"]) for e in _entries(vocab_path)] == [
        ("v2_feel", PLACEHOLDER_DESCRIPTION),
        ("u2_feel", "the real meaning"),
    ]
    assert persona_loader._repair_broken_entries(json.loads(vocab_path.read_text(encoding="utf-8"))) == []


def test_a_read_only_load_registers_what_a_writing_load_would_write_so_nothing_stale_is_written_back(
    tmp_path: Path,
) -> None:
    """The registry skips words it holds, so a value a read-only load registered would outlive the file's repair and be
    written back by `register_new_word` once the file lost the word."""
    from brain.health.reconstruct import PLACEHOLDER_DECAY_DAYS, PLACEHOLDER_DESCRIPTION

    vocab_path = tmp_path / "emotion_vocabulary.json"
    unsound = {"name": "zed_feel", "description": None, "category": None, "decay_half_life_days": 0}
    raw = json.dumps({"version": 1, "emotions": [unsound]})
    vocab_path.write_text(raw, encoding="utf-8")
    vocabulary._unregister("zed_feel")
    load_persona_vocabulary(vocab_path, read_only=True)

    held = vocabulary.get("zed_feel")
    assert (held.description, held.category, held.decay_half_life_days) == (
        PLACEHOLDER_DESCRIPTION,
        "persona_extension",
        PLACEHOLDER_DECAY_DAYS,
    )
    assert vocab_path.read_text(encoding="utf-8") == raw  # read-only: nothing written
    vocab_path.write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")
    assert persona_loader.register_new_word(vocab_path, "zed_feel") == "zed_feel"
    (zed,) = _entries(vocab_path)
    assert (zed["description"], zed["decay_half_life_days"]) == (PLACEHOLDER_DESCRIPTION, PLACEHOLDER_DECAY_DAYS)
    assert persona_loader._repair_broken_entries(json.loads(vocab_path.read_text(encoding="utf-8"))) == []


@pytest.mark.parametrize("bad", [-3.0, 0.0, float("nan")])
def test_register_new_word_writes_the_default_half_life_for_one_the_engine_would_ignore(tmp_path: Path, bad: float) -> None:
    """A half-life of 0 or below would be written and registered, and the next repair would report the word."""
    vocab_path = tmp_path / "emotion_vocabulary.json"
    vocab_path.write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")
    vocabulary._unregister("negword_feel")
    assert persona_loader.register_new_word(vocab_path, "negword_feel", half_life_days=bad) == "negword_feel"

    assert vocabulary.get("negword_feel").decay_half_life_days == 14.0
    assert [e["decay_half_life_days"] for e in _entries(vocab_path)] == [14.0]
    assert persona_loader._repair_broken_entries(json.loads(vocab_path.read_text(encoding="utf-8"))) == []


def test_register_new_word_writes_a_numeric_half_life_as_a_number(tmp_path: Path) -> None:
    """A string such as "3" is a usable half-life, but written as a string the entry is unsound and gets repaired."""
    vocab_path = tmp_path / "emotion_vocabulary.json"
    vocab_path.write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")
    vocabulary._unregister("strword_feel")
    persona_loader.register_new_word(vocab_path, "strword_feel", half_life_days="3")  # type: ignore[arg-type]

    assert [e["decay_half_life_days"] for e in _entries(vocab_path)] == [3.0]
    assert persona_loader._repair_broken_entries(json.loads(vocab_path.read_text(encoding="utf-8"))) == []


def _engine_half_life(persona_dir: Path, name: str) -> float:
    """What the emotion engine decays `name` by, read from the file as `EmotionEngine.half_life` does."""
    from brain import dev_constants
    from brain.emotion.engine_files import VocabularyHalfLives

    days = VocabularyHalfLives(persona_dir).get(name)  # a fresh cache: it reads the file as it is now
    if days is not None:
        return days
    return dev_constants.EMOTION_HALF_LIFE_DAYS.get(name, dev_constants.EMOTION_DEFAULT_HALF_LIFE_DAYS)


@pytest.mark.parametrize(
    ("copies", "registered"),
    [
        # The engine reads the first copy's null (the built-in half-life applies); the later whole copy says 9 days.
        (
            [
                {"name": "tw_feel", "description": "the real meaning", "category": None, "decay_half_life_days": None},
                {"name": "tw_feel", "description": "(reconstructed from memory)", "category": "persona_extension",
                 "decay_half_life_days": 9.0},
            ],
            None,
        ),
        # The engine reads the first copy's 3 days; the copy with the real description says no decay.
        (
            [
                {"name": "tw_feel", "description": None, "category": "complex", "decay_half_life_days": 3.0},
                {"name": "tw_feel", "description": "the real meaning", "category": None, "decay_half_life_days": None},
            ],
            3.0,
        ),
    ],
    ids=["whole_twin_later", "real_description_later"],
)
def test_a_repair_never_changes_the_half_life_the_engine_decays_a_word_by(
    tmp_path: Path, copies: list[dict], registered: float | None
) -> None:
    """The repair keeps the half-life of the first copy, in file order, that the engine accepts, on every copy it keeps."""
    vocab_path = tmp_path / "emotion_vocabulary.json"
    vocab_path.write_text(json.dumps({"version": 1, "emotions": copies}), encoding="utf-8")
    before = _engine_half_life(tmp_path, "tw_feel")
    vocabulary._unregister("tw_feel")
    load_persona_vocabulary(vocab_path)

    assert _engine_half_life(tmp_path, "tw_feel") == before
    assert vocabulary.get("tw_feel").decay_half_life_days == registered
    assert {e["decay_half_life_days"] for e in _entries(vocab_path) if e["name"] == "tw_feel"} == {registered}
    assert vocabulary.get("tw_feel").description == "the real meaning"
    assert persona_loader._repair_broken_entries(json.loads(vocab_path.read_text(encoding="utf-8"))) == []


@pytest.mark.no_base_failure(reason="guards a regression this branch's repair copy introduced; main never copied the file")
def test_a_read_only_load_of_deeply_nested_json_still_registers_its_words(tmp_path: Path) -> None:
    """JSON reads about 1000 levels of nesting; copying it for the repair runs out of stack about halfway there."""
    vocab_path = tmp_path / "emotion_vocabulary.json"
    entry = {"name": "deep_feel", "description": "d", "category": "persona_extension", "decay_half_life_days": 3.0}
    raw = json.dumps({"version": 1, "emotions": [entry]})
    nested = "[" * 800 + "]" * 800
    vocab_path.write_text(raw.replace('"decay_half_life_days"', f'"extra": {nested}, "decay_half_life_days"'), encoding="utf-8")
    vocabulary._unregister("deep_feel")
    load_persona_vocabulary(vocab_path, read_only=True)

    assert vocabulary.get("deep_feel").decay_half_life_days == 3.0


@pytest.mark.no_base_failure(reason="guards a regression this branch's write check introduced; main never copied the file")
def test_a_writing_load_of_deeply_nested_json_still_registers_its_words(tmp_path: Path) -> None:
    """The check for whether a load would write copies the file too: too deep to copy, it can't need a repair."""
    vocab_path = tmp_path / "emotion_vocabulary.json"
    entry = {"name": "deeper_feel", "description": "d", "category": "persona_extension", "decay_half_life_days": 3.0}
    raw = json.dumps({"version": 1, "emotions": [entry]})
    nested = "[" * 800 + "]" * 800
    vocab_path.write_text(raw.replace('"decay_half_life_days"', f'"extra": {nested}, "decay_half_life_days"'), encoding="utf-8")
    vocabulary._unregister("deeper_feel")
    load_persona_vocabulary(vocab_path)

    assert vocabulary.get("deeper_feel").decay_half_life_days == 3.0


def test_a_read_only_load_warns_about_entries_it_registered_repaired(tmp_path: Path, caplog) -> None:
    """It leaves the file as it is, so the warning is the only sign that the file needs a writing load to fix it."""
    import logging

    vocab_path = tmp_path / "emotion_vocabulary.json"
    vocab_path.write_text(json.dumps({"version": 1, "emotions": [{"name": "warn_feel"}]}), encoding="utf-8")
    vocabulary._unregister("warn_feel")
    with caplog.at_level(logging.WARNING, logger="brain.emotion.persona_loader"):
        load_persona_vocabulary(vocab_path, read_only=True)

    assert any("warn_feel" in r.getMessage() and "read-only" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("read_only", [True, False])
def test_an_entry_whose_name_is_not_text_names_no_word(tmp_path: Path, read_only: bool) -> None:
    """The repair leaves such an entry alone, so registering it as the word "5" would register what it never checked."""
    vocab_path = tmp_path / "emotion_vocabulary.json"
    entry = {"name": 5, "description": "NaN", "category": "persona_extension", "decay_half_life_days": "NaN"}
    ok = {"name": "fine_feel", "description": "d", "category": "persona_extension", "decay_half_life_days": 3.0}
    vocab_path.write_text(json.dumps({"version": 1, "emotions": [entry, ok]}).replace('"NaN"', "NaN"), encoding="utf-8")
    vocabulary._unregister("5")
    vocabulary._unregister("fine_feel")
    load_persona_vocabulary(vocab_path, read_only=read_only)

    assert vocabulary.get("5") is None
    assert vocabulary.get("fine_feel") is not None


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_description_never_wins_over_a_real_one(tmp_path: Path, blank: str) -> None:
    """A blank description says no more than the placeholder: as the base, it would drop the twin that says something."""
    vocab_path = tmp_path / "emotion_vocabulary.json"
    copies = [
        {"name": "blank_feel", "description": blank, "decay_half_life_days": 3.0},
        {"name": "Blank_Feel", "description": "another meaning", "decay_half_life_days": 3.0},
    ]
    vocab_path.write_text(json.dumps({"version": 1, "emotions": copies}), encoding="utf-8")
    vocabulary._unregister("blank_feel")
    load_persona_vocabulary(vocab_path)

    assert vocabulary.get("blank_feel").description == "another meaning"
    assert [e["description"] for e in _entries(vocab_path)] == ["another meaning"]


def test_an_integer_too_large_for_a_float_is_repaired_instead_of_crashing_the_load_and_new_words(tmp_path: Path) -> None:
    """JSON reads a 400-digit number as an int, and `float()` of it raises OverflowError rather than returning inf."""
    from brain.health.reconstruct import PLACEHOLDER_DECAY_DAYS

    vocab_path = tmp_path / "emotion_vocabulary.json"
    huge = {"name": "big_feel", "description": "real", "category": "persona_extension", "decay_half_life_days": 7}
    raw = json.dumps({"version": 1, "emotions": [huge]}).replace(": 7}", ": " + "1" + "0" * 400 + ', "intensity_clamp": 1' + "0" * 400 + "}")
    vocab_path.write_text(raw, encoding="utf-8")
    vocabulary._unregister("big_feel")
    vocabulary._unregister("newword_feel")
    load_persona_vocabulary(vocab_path)

    assert vocabulary.get("big_feel").decay_half_life_days == PLACEHOLDER_DECAY_DAYS
    assert vocabulary.get("big_feel").intensity_clamp == 10.0
    persona_loader.register_new_word(vocab_path, "newword_feel")
    assert {"big_feel", "newword_feel"} <= {e["name"] for e in _entries(vocab_path)}
    assert persona_loader._repair_broken_entries(json.loads(vocab_path.read_text(encoding="utf-8"))) == []


def test_a_bare_non_finite_number_in_the_emotions_list_becomes_null_and_the_file_loads(tmp_path: Path) -> None:
    """A bare NaN element isn't an object, so it can't be changed in place: it must be replaced in the list itself."""
    vocab_path = tmp_path / "emotion_vocabulary.json"
    ok = {"name": "ok_feel", "description": "d", "category": "persona_extension", "decay_half_life_days": 3.0}
    vocab_path.write_text(json.dumps({"version": 1, "emotions": ["NaN", ok]}).replace('"NaN"', "NaN"), encoding="utf-8")
    vocabulary._unregister("ok_feel")
    load_persona_vocabulary(vocab_path)

    assert vocabulary.get("ok_feel") is not None
    assert json.loads(vocab_path.read_text(encoding="utf-8"))["emotions"] == [None, ok]
    assert persona_loader._repair_broken_entries(json.loads(vocab_path.read_text(encoding="utf-8"))) == []


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

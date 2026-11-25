"""Properties the vocabulary repair must keep, checked on seeded random vocabulary files (#784).

Four cold reviews of the repair each found a file the hand-written tests missed: a bare NaN in the list, a 400-digit
integer, a null description registered as the text "None", a repair that changed how fast a feeling fades. These tests
check the properties those reviews relied on against a few thousand generated files, through every writer (the repair,
a read-only and a writing load, the heal, `register_new_word`), so the next change is checked the same way. A failure
names its seed and prints the file; `_random_vocabulary(random.Random(seed))` rebuilds it. The seeds are fixed, so every
run checks the same files.

What a reader needs of an entry is stated here (`_meets_contract`), not taken from the repair's own `_is_sound`: a check
built on the repair's rule would weaken with it. The engine's rule is taken from the engine (`_classify`,
`VocabularyHalfLives`), because the property is agreement with what the engine actually does.

The generator mixes what broke the repair before: NaN and infinity at any depth, integers too large for a float,
numeric strings, bools, null against missing against corrupt fields, blank descriptions, clamps of 0 or below, names
that aren't text, spelling twins that are one word, words with a built-in half-life, and whole and broken copies of a
word together. `test_the_generator_makes_*` fail if an edit stops it making them.

Not covered here, on purpose:
  - files that aren't a dict with an `emotions` list, or aren't JSON (the quarantine and heal from backup:
    test_persona_loader_vocab_recovery.py);
  - a category outside the known set: no reader validates it, and the repair keeps it as written;
  - a name that isn't text whose `str()` is a real word's name: the loader skips such an entry, but the engine reads it
    under that text, so the two could pick different copies. The engine's file is outside this change (#784).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
from pathlib import Path

import pytest

from brain import dev_constants
from brain.emotion import persona_loader, vocabulary
from brain.emotion.engine_files import VocabularyHalfLives, _classify, _Fault
from brain.emotion.persona_loader import load_persona_vocabulary
from brain.health.reconstruct import PLACEHOLDER_DESCRIPTION
from brain.memory.store import Memory, MemoryStore

_NAN, _INF = float("nan"), float("inf")
# Five spellings of one word, so the generator makes spelling twins, and two words with a built-in half-life in
# `dev_constants.EMOTION_HALF_LIFE_DAYS` (so the engine's fallback isn't only ever the default). None is a baseline emotion.
_TEXT_NAMES = [
    "prop_a_feel",
    "Prop_A_Feel",
    "prop-a-feel",
    " Prop_A_Feel ",
    "PROP-A-FEEL",
    "prop_b_feel",
    "creative_hunger",
    "Emergence",
]
_WORDS = sorted({vocabulary.canonical_name(n) for n in _TEXT_NAMES})
_NON_TEXT_NAMES: list[object] = [5, None, _NAN, ["x"]]
# What `str()` makes of them: the word each would register as if a loader accepted it.
_NON_TEXT_FORMS = sorted({vocabulary.canonical_name(str(n)) for n in _NON_TEXT_NAMES} | {str(n) for n in _NON_TEXT_NAMES})
_NUMBERS: list[object] = [_NAN, _INF, -_INF, None, "x", "3", "NaN", True, False, 0, -2, 3.0, 7, 10**400, -(10**400)]
_NESTED: list[object] = [[_NAN], {"a": _INF}, [[_NAN, {"b": -_INF}]], {}, [1, "two"]]
_DESCRIPTIONS: list[object] = ["the real meaning", "another meaning", PLACEHOLDER_DESCRIPTION, None, _NAN, 4, ""]
_CATEGORIES: list[object] = ["persona_extension", "complex", None, _NAN, 7]

_PURE_CASES = 3000
_LOAD_CASES = 300  # each writes and loads the file three times
_HEAL_CASES = 150  # each builds a memory store
_NEW_WORD = "propnew_feel"  # letters and underscores only: the gate for new words refuses digits


def _whole_entry(r: random.Random) -> dict:
    """An entry every reader reads as written. Built on purpose: random fields are rarely all valid at once, so without
    these a word would almost never have two whole copies beside a broken one."""
    entry: dict[str, object] = {
        "name": r.choice(_TEXT_NAMES),
        "description": r.choice(["the real meaning", "another meaning", PLACEHOLDER_DESCRIPTION]),
        "category": r.choice(["persona_extension", "complex"]),
        "decay_half_life_days": r.choice([None, 3.0, 7, 30.5]),
    }
    if r.random() < 0.5:
        entry["intensity_clamp"] = r.choice([10.0, 5, 0.5])
    return entry


def _random_entry(r: random.Random) -> object:
    if r.random() < 0.08:
        return r.choice([_NAN, None, 3, "not an object", [_INF]])
    if r.random() < 0.3:
        return _whole_entry(r)
    entry: dict[str, object] = {}
    roll = r.random()
    if roll < 0.90:
        entry["name"] = r.choice(_TEXT_NAMES)
    elif roll < 0.97:
        entry["name"] = r.choice(_NON_TEXT_NAMES)
    fields = [
        ("description", _DESCRIPTIONS),
        ("category", _CATEGORIES),
        ("decay_half_life_days", _NUMBERS),
        ("intensity_clamp", _NUMBERS),
        ("extra", _NUMBERS + _NESTED),
    ]
    for key, pool in fields:
        if r.random() < 0.75:
            entry[key] = copy.deepcopy(r.choice(pool))
    return entry


def _random_vocabulary(r: random.Random) -> dict:
    data: dict[str, object] = {"version": 1, "emotions": [_random_entry(r) for _ in range(r.randint(0, 6))]}
    if r.random() < 0.3:
        data["built_at"] = copy.deepcopy(r.choice([1, "s", _NAN, [_INF], {"x": -_INF}]))
    return data


def _snapshot(data: object) -> str:
    """A comparable form of a file's data: NaN compares unequal to itself, so compare its text instead."""
    return json.dumps(data, sort_keys=True, default=repr)


def _text_words(data: dict) -> set[str]:
    entries = data["emotions"]
    return {vocabulary.canonical_name(e["name"]) for e in entries if isinstance(e, dict) and isinstance(e.get("name"), str)}


def _unregister_words() -> None:
    for word in _WORDS + _NON_TEXT_FORMS:
        vocabulary._unregister(word)


@pytest.fixture(autouse=True)
def _quiet_and_clean():
    # Each damaged file logs warnings by design; thousands of them would drown a real failure's output.
    logging.disable(logging.WARNING)
    _unregister_words()
    yield
    _unregister_words()
    logging.disable(logging.NOTSET)


def _is_number(value: object) -> bool:
    """A finite number a float can hold: not a bool, a numeric string, NaN, infinity or a 400-digit integer."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def _meets_contract(entry: object) -> bool:
    """What every reader needs of a named entry, stated here rather than taken from the repair's own `_is_sound`: a check
    that used the repair's rule would weaken with it and still agree.

      - name, description and category are text (the loader would otherwise register `str(None)`, the text "None");
      - the half-life is null or a number above 0 (the engine ignores anything else and uses its built-in one);
      - the clamp, when present, is a number above 0 (at 0 or below, every intensity exceeds it: the word can't be felt).
    A category outside the known set is not checked: no reader validates it, and the repair keeps it as written.
    """
    if not isinstance(entry, dict):
        return False
    if not all(isinstance(entry.get(key), str) for key in ("name", "description", "category")):
        return False
    if "decay_half_life_days" not in entry:
        return False
    half_life = entry["decay_half_life_days"]
    if half_life is not None and not (_is_number(half_life) and half_life > 0):
        return False
    return "intensity_clamp" not in entry or (_is_number(entry["intensity_clamp"]) and entry["intensity_clamp"] > 0)


def _fail(seed: int, data: dict, why: str) -> None:
    pytest.fail(f"seed {seed}: {why}\n file: {_snapshot(data)}")


def test_the_repair_leaves_every_file_writable_sound_and_stable() -> None:
    """On every file: the result writes as strict JSON, every named entry left is one the loader and the engine both
    read as written, a second pass finds nothing, no word is lost, and a repair that reports nothing changed nothing."""
    for seed in range(_PURE_CASES):
        original = _random_vocabulary(random.Random(seed))
        data = copy.deepcopy(original)
        reported = persona_loader._repair_broken_entries(data)

        try:
            json.dumps(data, allow_nan=False)
        except ValueError:
            _fail(seed, original, "the repaired file can't be written as JSON")
        for entry in data["emotions"]:
            if persona_loader._entry_canonical_name(entry) is None:
                continue
            if not _meets_contract(entry):
                _fail(seed, original, f"an entry is left that a reader would misread: {entry!r}")
            if isinstance(_classify(entry), _Fault):
                _fail(seed, original, f"the engine refuses an entry the repair left: {entry!r}")
        if persona_loader._repair_broken_entries(copy.deepcopy(data)):
            _fail(seed, original, "a second pass still finds something to repair")
        if not _text_words(original) <= _text_words(data):
            _fail(seed, original, f"the repair lost words: {sorted(_text_words(original) - _text_words(data))}")
        if not reported and _snapshot(data) != _snapshot(original):
            _fail(seed, original, "the repair changed the file but reported nothing")
        named_before = {e["name"] for e in original["emotions"] if isinstance(e, dict) and isinstance(e.get("name"), str)}
        if set(reported) - named_before - _PLACE_LABELS:
            _fail(seed, original, f"the repair reported names it didn't touch: {sorted(set(reported) - named_before)}")
        for word in _text_words(original):
            if _real_descriptions(original, word) and not _real_descriptions(data, word):
                _fail(seed, original, f"the repair lost every real description of {word}")
            whole_before = sum(_meets_contract(e) for e in _entries_of(original, word))
            if len(_entries_of(data, word)) < max(1, whole_before):
                _fail(seed, original, f"the repair dropped a whole copy of {word}")


_PLACE_LABELS = {"(an entry with no name)", "(the file's top level)"}


def _entries_of(data: dict, word: str) -> list[dict]:
    return [e for e in data["emotions"] if persona_loader._entry_canonical_name(e) == word]


def _real_descriptions(data: dict, word: str) -> set[str]:
    """A word's descriptions that say what it means: text, not blank, other than the placeholder a reconstruction writes."""
    found = {e.get("description") for e in _entries_of(data, word)}
    return {d for d in found if isinstance(d, str) and d.strip() and d != PLACEHOLDER_DESCRIPTION}


def _engine_half_life(persona_dir: Path, word: str) -> float:
    """What the emotion engine decays `word` by, reading the file as `EmotionEngine.half_life` does."""
    return _with_built_in(VocabularyHalfLives(persona_dir).get(word), word)  # a fresh cache reads the file as it is now


def _with_built_in(days: float | None, word: str) -> float:
    """The engine's lookup after the file: the file's number, else `EMOTION_HALF_LIFE_DAYS`, else the default."""
    if days is not None:
        return days
    return dev_constants.EMOTION_HALF_LIFE_DAYS.get(word, dev_constants.EMOTION_DEFAULT_HALF_LIFE_DAYS)


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")  # Python's JSON writes NaN and Infinity, as a damaged file has


def test_the_repair_never_changes_the_half_life_the_engine_decays_a_word_by(tmp_path: Path) -> None:
    """The engine reads the file itself; a repair that changed what it reads would change how fast a feeling fades."""
    # The engine's own parser, on the data in memory: writing and re-reading two files per case cost most of this test's
    # time, and the suite's CPU budget is 2 s. The tests below still go through the real file.
    engine = VocabularyHalfLives(tmp_path)
    for seed in range(_PURE_CASES):
        original = _random_vocabulary(random.Random(seed))
        data = copy.deepcopy(original)
        persona_loader._repair_broken_entries(data)
        table_before, table_after = engine._parse(original), engine._parse(data)
        for word in sorted(_text_words(original)):
            before = _with_built_in(table_before.get(word), word)
            after = _with_built_in(table_after.get(word), word)
            if before != after:
                _fail(seed, original, f"the engine's half-life for {word} went from {before} to {after}")


def _registered(words: list[str]) -> dict[str, tuple | None]:
    found: dict[str, tuple | None] = {}
    for word in words:
        emotion = vocabulary.get(word)
        found[word] = (
            None
            if emotion is None
            else (emotion.description, emotion.category, emotion.decay_half_life_days, emotion.intensity_clamp)
        )
    return found


def test_a_read_only_load_and_a_writing_load_register_the_same_and_only_the_writing_one_writes(tmp_path: Path) -> None:
    """A read-only process (the MCP server, a dry run, the engine's refresh) and the bridge's writing load read one
    file; if they registered different values, a later write-back could put the read-only process's values on disk."""
    path = tmp_path / "emotion_vocabulary.json"
    for seed in range(_LOAD_CASES):
        original = _random_vocabulary(random.Random(seed))
        _write(path, original)
        raw = path.read_bytes()

        _unregister_words()
        load_persona_vocabulary(path, read_only=True)
        read_only = _registered(_WORDS)
        if path.read_bytes() != raw:
            _fail(seed, original, "a read-only load changed the file")

        _unregister_words()
        load_persona_vocabulary(path)
        writing = _registered(_WORDS)
        if read_only != writing:
            _fail(seed, original, f"read-only registered {read_only}, the writing load {writing}")
        for word, held in writing.items():
            engine_days = VocabularyHalfLives(tmp_path).get(word)  # None: no number in the file, the built-in applies
            if held is not None and held[2] != engine_days:
                _fail(seed, original, f"the registry holds {held[2]} days for {word}, the engine reads {engine_days}")
        stray = {k: v for k, v in _registered(_NON_TEXT_FORMS).items() if v is not None}
        if stray:
            _fail(seed, original, f"an entry whose name isn't text registered as a word: {stray}")

        left = json.loads(path.read_text(encoding="utf-8"))
        if persona_loader._repair_broken_entries(left):
            _fail(seed, original, "the file a writing load left still needs repair")
        _unregister_words()
        load_persona_vocabulary(path, read_only=True)
        if _registered(_WORDS) != writing:
            _fail(seed, original, "loading the repaired file registers different values")


def test_adding_a_new_word_to_any_file_works_and_leaves_the_other_words_as_the_engine_had_them(tmp_path: Path) -> None:
    """`register_new_word` is the growth path's writer: a file it can't repair would stop new words for good."""
    path = tmp_path / "emotion_vocabulary.json"
    for seed in range(_LOAD_CASES):
        r = random.Random(seed)
        original = _random_vocabulary(r)
        _write(path, original)
        before = {word: _engine_half_life(tmp_path, word) for word in _text_words(original)}
        # A word new to both; one the file may already hold, maybe only broken; or one this process already holds
        # (registered by a read-only load, as the MCP server would have) that the file may or may not have.
        word = r.choice([_NEW_WORD, *_WORDS])
        _unregister_words()
        vocabulary._unregister(_NEW_WORD)
        if r.random() < 0.5:
            load_persona_vocabulary(path, read_only=True)

        if persona_loader.register_new_word(path, word) != word:
            _fail(seed, original, f"register_new_word refused {word}")
        if vocabulary.get(word) is None:
            _fail(seed, original, f"register_new_word returned {word} but didn't register it")
        left = json.loads(path.read_text(encoding="utf-8"))
        if persona_loader._repair_broken_entries(copy.deepcopy(left)):
            _fail(seed, original, "the file register_new_word left still needs repair")
        if not _entries_of(left, word):
            _fail(seed, original, f"{word} isn't in the file")
        if word == _NEW_WORD and len(_entries_of(left, word)) != 1:
            _fail(seed, original, "the new word isn't in the file exactly once")
        if not _text_words(original) <= _text_words(left):
            _fail(seed, original, "register_new_word lost words")
        for word, days in before.items():
            if _engine_half_life(tmp_path, word) != days:
                _fail(seed, original, f"the engine's half-life for {word} changed when a new word was added")
    vocabulary._unregister(_NEW_WORD)


def test_the_heal_writes_every_word_the_memories_feel_and_leaves_the_rest_as_the_engine_had_them(tmp_path: Path) -> None:
    """The bridge's first load has the memory store: a word the memories feel but the file lacks (or holds only broken)
    is healed into the file, so the engine and a restart know it. That write must leave a file nothing else changes."""
    for seed in range(_HEAL_CASES):
        r = random.Random(seed)
        original = _random_vocabulary(r)
        persona_dir = tmp_path / str(seed)  # its own folder: the heal runs once per persona, until its marker exists
        persona_dir.mkdir()
        path = persona_dir / "emotion_vocabulary.json"
        _write(path, original)
        before = {word: _engine_half_life(persona_dir, word) for word in _text_words(original)}
        felt = r.sample([*_TEXT_NAMES, _NEW_WORD], r.randint(1, 3))
        store = MemoryStore(":memory:")
        try:
            for name in felt:
                store.create(Memory.create_new(content=f"felt {name}", memory_type="conversation", domain="us",
                                               emotions={name: 6.0}))
            _unregister_words()
            vocabulary._unregister(_NEW_WORD)
            load_persona_vocabulary(path, store=store)
        finally:
            store.close()

        left = json.loads(path.read_text(encoding="utf-8"))
        if persona_loader._repair_broken_entries(copy.deepcopy(left)):
            _fail(seed, original, "the file the heal left still needs repair")
        for name in felt:
            word = vocabulary.canonical_name(name)
            if vocabulary.get(word) is None or not any(_meets_contract(e) for e in _entries_of(left, word)):
                _fail(seed, original, f"the memories feel {word}, but it isn't registered and whole in the file")
        for word, days in before.items():
            if _engine_half_life(persona_dir, word) != days:
                _fail(seed, original, f"the engine's half-life for {word} changed in the heal")
    vocabulary._unregister(_NEW_WORD)


def _anywhere(value: object, test) -> bool:
    if isinstance(value, dict):
        return any(_anywhere(v, test) for v in value.values())
    if isinstance(value, list):
        return any(_anywhere(v, test) for v in value)
    return test(value)


def test_the_generator_makes_the_files_that_broke_the_repair_before() -> None:
    """If an edit stopped the generator making these, the properties above would hold on files that test nothing."""
    files = [_random_vocabulary(random.Random(seed)) for seed in range(_PURE_CASES)]
    entries = [e for data in files for e in data["emotions"]]
    named = [e for e in entries if isinstance(e, dict) and isinstance(e.get("name"), str)]

    def is_nan(value: object) -> bool:
        return isinstance(value, float) and math.isnan(value)

    assert any(is_nan(e) for e in entries), "a bare NaN element"
    assert any(isinstance(e, dict) and _anywhere(e.get("extra"), is_nan) for e in entries), "a nested NaN"
    assert any(e.get("decay_half_life_days") == 10**400 for e in named), "an integer too large for a float"
    assert any(e.get("decay_half_life_days") == "3" for e in named), "a numeric string half-life"
    assert any("description" in e and e["description"] is None for e in named), "a null description"
    assert any(isinstance(e, dict) and "name" in e and not isinstance(e["name"], str) for e in entries), "a non-text name"
    spellings = {"prop_a_feel", "Prop_A_Feel", "prop-a-feel", " Prop_A_Feel ", "PROP-A-FEEL"}
    assert any(len(_names(data) & spellings) >= 2 for data in files), "spelling twins of one word"
    assert any(_is_whole_and_broken_mix(data) for data in files), "a whole and a broken copy of one word"


def test_the_generator_makes_words_the_engine_has_a_built_in_half_life_for_with_only_broken_copies() -> None:
    """The repair's fallback for such a word is the engine's table, not the default: without them it's never tested."""
    files = [_random_vocabulary(random.Random(seed)) for seed in range(_PURE_CASES)]
    table_words = {vocabulary.canonical_name(n) for n in _TEXT_NAMES} & set(dev_constants.EMOTION_HALF_LIFE_DAYS)
    assert table_words
    assert any(
        _entries_of(data, word) and not any(_meets_contract(e) for e in _entries_of(data, word))
        for data in files
        for word in table_words
    )


def test_the_generator_makes_words_with_two_whole_copies_beside_a_broken_one() -> None:
    """The repair keeps a word's whole copies and drops only its broken ones: this is the file that tells them apart."""
    files = [_random_vocabulary(random.Random(seed)) for seed in range(_PURE_CASES)]
    assert any(
        sum(_meets_contract(e) for e in _entries_of(data, word)) >= 2
        and not all(_meets_contract(e) for e in _entries_of(data, word))
        for data in files
        for word in _text_words(data)
    )


def _names(data: dict) -> set[str]:
    return {e["name"] for e in data["emotions"] if isinstance(e, dict) and isinstance(e.get("name"), str)}


def _is_whole_and_broken_mix(data: dict) -> bool:
    by_word: dict[str, list[bool]] = {}
    for entry in data["emotions"]:
        word = persona_loader._entry_canonical_name(entry)
        if word is not None:
            by_word.setdefault(word, []).append(persona_loader._is_sound(entry))
    return any(True in sound and False in sound for sound in by_word.values())

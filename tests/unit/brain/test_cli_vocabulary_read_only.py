"""#801 step 5: a `nell` command never heals or writes her vocabulary file; it loads it read-only (spec section 5, no-bridge paragraph)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from brain.emotion import vocabulary, vocabulary_access
from brain.memory.store import Memory, MemoryStore

REPO = Path(__file__).resolve().parents[3]


def test_no_command_loads_her_vocabulary_with_a_store() -> None:
    """A load that has a store runs the memory heal and the corrupt-file rebuild, which write her file."""
    tree = ast.parse((REPO / "brain" / "cli.py").read_text(encoding="utf-8"))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", "")) in {"load_persona_vocabulary", "load_persona_vocabulary_with_anomaly"}
        and any(kw.arg == "store" for kw in node.keywords)
    ]
    assert offenders == [], f"brain/cli.py loads her vocabulary with a store at lines {offenders}"


def test_the_commands_load_through_the_read_only_helper() -> None:
    text = (REPO / "brain" / "cli.py").read_text(encoding="utf-8")
    assert text.count("vocabulary_access.load_read_only(persona_dir)") == 7


@pytest.fixture
def persona_with_a_stray_name(tmp_path: Path) -> Path:
    store = MemoryStore(tmp_path / "memories.db")
    store.create(
        Memory.create_new(content="x", memory_type="conversation", domain="us", emotions={"stray_zz_word": 4.0}), check_names=False
    )
    store.close()
    return tmp_path


def test_the_read_only_load_writes_no_file_even_when_a_memory_names_a_feeling_her_vocabulary_lacks(persona_with_a_stray_name: Path) -> None:
    before = sorted(p.name for p in persona_with_a_stray_name.iterdir())
    vocabulary_access.load_read_only(persona_with_a_stray_name)
    assert sorted(p.name for p in persona_with_a_stray_name.iterdir()) == before
    assert not (persona_with_a_stray_name / "emotion_vocabulary.json").exists()
    assert vocabulary.get("stray_zz_word") is None


def test_the_read_only_load_registers_what_her_file_holds_and_leaves_a_damaged_file_as_it_is(tmp_path: Path) -> None:
    good = tmp_path / "good"
    good.mkdir()
    (good / "emotion_vocabulary.json").write_text(
        json.dumps({"version": 1, "emotions": [{"name": "Quiet Dread Zz", "description": "d", "category": "persona_extension", "decay_half_life_days": 9.0}]}),
        encoding="utf-8",
    )
    vocabulary_access.load_read_only(good)
    found = vocabulary.get("quiet_dread_zz")
    assert found is not None and found.name == "Quiet Dread Zz"

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "emotion_vocabulary.json").write_text("{not json", encoding="utf-8")
    vocabulary_access.load_read_only(bad)  # warned, not raised
    assert (bad / "emotion_vocabulary.json").read_text(encoding="utf-8") == "{not json"
    assert sorted(p.name for p in bad.iterdir()) == ["emotion_vocabulary.json"]  # not quarantined, not reset, no backup

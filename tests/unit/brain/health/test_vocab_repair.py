"""Tests for brain.health.vocab_repair — one-time vocab stub decay fix.

TDD: one test at a time.
"""

from __future__ import annotations

import json
from pathlib import Path

from brain.health.reconstruct import PLACEHOLDER_DESCRIPTION
from brain.memory.store import Memory, MemoryStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "memories.db"), integrity_check=False)


def _write_vocab(tmp_path: Path, entries: list[dict]) -> Path:
    """Write a well-formed emotion_vocabulary.json to tmp_path."""
    path = tmp_path / "emotion_vocabulary.json"
    data = {"version": 1, "emotions": entries}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _stub_entry(name: str) -> dict:
    """An entry with the OLD bad shape: placeholder desc + 1.0 half-life."""
    return {
        "name": name,
        "description": PLACEHOLDER_DESCRIPTION,
        "category": "persona_extension",
        "decay_half_life_days": 1.0,
        "intensity_clamp": 10.0,
    }


def _proper_entry(name: str, desc: str = "a real description") -> dict:
    """An entry with a real description and proper half-life."""
    return {
        "name": name,
        "description": desc,
        "category": "persona_extension",
        "decay_half_life_days": 45.0,
        "intensity_clamp": 10.0,
    }


# ---------------------------------------------------------------------------
# should_run_vocab_repair
# ---------------------------------------------------------------------------

def test_should_run_false_when_no_vocab_file(tmp_path: Path) -> None:
    """No vocab file → nothing to repair → should_run returns False."""
    from brain.health.vocab_repair import should_run_vocab_repair

    assert should_run_vocab_repair(tmp_path) is False


def test_should_run_true_when_stubs_present_no_state(tmp_path: Path) -> None:
    """Vocab with stub entries and no state file → should_run returns True."""
    from brain.health.vocab_repair import should_run_vocab_repair

    _write_vocab(tmp_path, [_stub_entry("body_grief"), _proper_entry("love")])
    assert should_run_vocab_repair(tmp_path) is True


def test_should_run_false_when_state_complete(tmp_path: Path) -> None:
    """Complete state file → should_run returns False regardless of vocab."""
    from brain.health.vocab_repair import should_run_vocab_repair

    _write_vocab(tmp_path, [_stub_entry("body_grief")])
    state = {
        "status": "complete",
        "repaired": 1,
        "described": 0,
        "completed_at": "2026-06-10T00:00:00Z",
    }
    (tmp_path / "vocab_repair_state.json").write_text(json.dumps(state), encoding="utf-8")
    assert should_run_vocab_repair(tmp_path) is False


def test_should_run_false_when_no_stubs(tmp_path: Path) -> None:
    """Vocab with zero stubs → should_run returns False."""
    from brain.health.vocab_repair import should_run_vocab_repair

    _write_vocab(tmp_path, [_proper_entry("love"), _proper_entry("joy")])
    assert should_run_vocab_repair(tmp_path) is False


# ---------------------------------------------------------------------------
# run_vocab_repair — Step 1 (half-life bump)
# ---------------------------------------------------------------------------

def test_repair_bumps_only_stub_entries(tmp_path: Path) -> None:
    """Stub entries (placeholder desc + 1.0 half-life) → bumped to 14.0.
    Proper entry (real desc, 45.0) → untouched.
    State file written with status=='complete', repaired==2.
    """
    from brain.health.vocab_repair import run_vocab_repair

    _write_vocab(tmp_path, [
        _stub_entry("body_grief"),
        _stub_entry("creative_hunger"),
        _proper_entry("love"),
    ])

    store = _make_store(tmp_path)
    try:
        report = run_vocab_repair(tmp_path, store=store, provider=None)
    finally:
        store.close()

    assert report.repaired == 2
    assert report.status == "complete"

    data = json.loads((tmp_path / "emotion_vocabulary.json").read_text(encoding="utf-8"))
    by_name = {e["name"]: e for e in data["emotions"]}

    assert by_name["body_grief"]["decay_half_life_days"] == 14.0
    assert by_name["creative_hunger"]["decay_half_life_days"] == 14.0
    assert by_name["love"]["decay_half_life_days"] == 45.0
    assert by_name["love"]["description"] == "a real description"

    state = json.loads((tmp_path / "vocab_repair_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["repaired"] == 2


def test_repair_idempotent(tmp_path: Path) -> None:
    """Complete state file → run_vocab_repair returns early without touching the vocab file."""
    from brain.health.vocab_repair import run_vocab_repair, should_run_vocab_repair

    _write_vocab(tmp_path, [_stub_entry("body_grief")])
    state = {
        "status": "complete",
        "repaired": 1,
        "described": 0,
        "completed_at": "2026-06-10T00:00:00Z",
    }
    state_path = tmp_path / "vocab_repair_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    orig_vocab_mtime = (tmp_path / "emotion_vocabulary.json").stat().st_mtime

    assert should_run_vocab_repair(tmp_path) is False

    store = _make_store(tmp_path)
    try:
        report = run_vocab_repair(tmp_path, store=store, provider=None)
    finally:
        store.close()

    assert report.status == "complete"
    assert report.repaired == 1  # from state file, not re-run

    new_mtime = (tmp_path / "emotion_vocabulary.json").stat().st_mtime
    assert new_mtime == orig_vocab_mtime


# ---------------------------------------------------------------------------
# Step 2 — fail-soft
# ---------------------------------------------------------------------------

class _FakeProvider:
    """Minimal provider stub that records calls and returns a canned response."""

    def __init__(self, response: str | None = None, raises: bool = False):
        self.calls: list[dict] = []
        self._response = response
        self._raises = raises

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        persona_dir: Path | None = None,
    ) -> str:
        self.calls.append({"prompt": prompt, "system": system})
        if self._raises:
            raise RuntimeError("simulated provider failure")
        if self._response is not None:
            return self._response
        return "{}"


def test_repair_descriptions_fail_soft(tmp_path: Path) -> None:
    """Provider raises → placeholders kept, half-life still bumped, status complete, described==0."""
    from brain.health.vocab_repair import run_vocab_repair

    _write_vocab(tmp_path, [_stub_entry("body_grief"), _stub_entry("creative_hunger")])
    provider = _FakeProvider(raises=True)

    store = _make_store(tmp_path)
    try:
        report = run_vocab_repair(tmp_path, store=store, provider=provider)
    finally:
        store.close()

    assert report.repaired == 2
    assert report.described == 0
    assert report.status == "complete"

    data = json.loads((tmp_path / "emotion_vocabulary.json").read_text(encoding="utf-8"))
    by_name = {e["name"]: e for e in data["emotions"]}
    assert by_name["body_grief"]["decay_half_life_days"] == 14.0
    assert by_name["creative_hunger"]["decay_half_life_days"] == 14.0
    assert by_name["body_grief"]["description"] == PLACEHOLDER_DESCRIPTION
    assert by_name["creative_hunger"]["description"] == PLACEHOLDER_DESCRIPTION

    state = json.loads((tmp_path / "vocab_repair_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["described"] == 0


def test_repair_derives_descriptions(tmp_path: Path) -> None:
    """Provider returns valid JSON → placeholder descriptions are replaced.
    Proper entries are left untouched. described count == number of stubs replaced.
    """
    from brain.health.vocab_repair import run_vocab_repair

    stub_names = ["body_grief", "creative_hunger"]
    _write_vocab(tmp_path, [
        _stub_entry("body_grief"),
        _stub_entry("creative_hunger"),
        _proper_entry("love"),
    ])

    canned_json = json.dumps({
        "body_grief": "the physical weight of grief",
        "creative_hunger": "deep drive to make something new",
    })
    provider = _FakeProvider(response=canned_json)

    store = _make_store(tmp_path)
    for name in stub_names:
        store.create(Memory.create_new(
            content=f"memory referencing {name}",
            memory_type="conversation",
            domain="us",
            emotions={name: 5.0},
        ))
    try:
        report = run_vocab_repair(tmp_path, store=store, provider=provider)
    finally:
        store.close()

    assert report.repaired == 2
    assert report.described == 2
    assert report.status == "complete"
    assert len(provider.calls) >= 1

    data = json.loads((tmp_path / "emotion_vocabulary.json").read_text(encoding="utf-8"))
    by_name = {e["name"]: e for e in data["emotions"]}

    assert by_name["body_grief"]["description"] == "the physical weight of grief"
    assert by_name["creative_hunger"]["description"] == "deep drive to make something new"
    assert by_name["body_grief"]["decay_half_life_days"] == 14.0
    assert by_name["love"]["description"] == "a real description"
    assert by_name["love"]["decay_half_life_days"] == 45.0


class _NoPersonaDirProvider:
    """Provider whose generate signature does NOT accept persona_dir.

    Used to verify F1: vocab_repair falls back gracefully via TypeError retry
    instead of crashing when a non-Cli provider is passed.
    """

    def __init__(self, response: str) -> None:
        self.calls: list[str] = []
        self._response = response

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append(prompt)
        return self._response


def test_repair_provider_without_persona_dir_kwarg(tmp_path: Path) -> None:
    """A provider whose generate() lacks persona_dir → TypeError retry → descriptions derived."""
    from brain.health.vocab_repair import run_vocab_repair

    canned = json.dumps({"body_grief": "physical weight of loss"})
    provider = _NoPersonaDirProvider(response=canned)

    _write_vocab(tmp_path, [_stub_entry("body_grief")])

    store = _make_store(tmp_path)
    try:
        report = run_vocab_repair(tmp_path, store=store, provider=provider)
    finally:
        store.close()

    assert report.repaired == 1
    assert report.described == 1
    assert len(provider.calls) == 1  # exactly one generate call via the fallback

    data = json.loads((tmp_path / "emotion_vocabulary.json").read_text(encoding="utf-8"))
    by_name = {e["name"]: e for e in data["emotions"]}
    assert by_name["body_grief"]["description"] == "physical weight of loss"


def test_repair_no_stubs_noop(tmp_path: Path) -> None:
    """Vocab with zero stubs → state written complete, repaired==0, no LLM call."""
    from brain.health.vocab_repair import run_vocab_repair

    _write_vocab(tmp_path, [_proper_entry("love"), _proper_entry("joy")])
    provider = _FakeProvider()

    store = _make_store(tmp_path)
    try:
        report = run_vocab_repair(tmp_path, store=store, provider=provider)
    finally:
        store.close()

    assert report.repaired == 0
    assert report.status == "complete"
    assert provider.calls == [], "no LLM call when there are no stubs"

    state = json.loads((tmp_path / "vocab_repair_state.json").read_text(encoding="utf-8"))
    assert state["repaired"] == 0

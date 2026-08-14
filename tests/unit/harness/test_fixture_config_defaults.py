"""Relax persona-config forcing — token-free conformance for the config-defaults change.

Covers criteria C1, C3–C7 (1.5-criteria.md). Zero model tokens: real `sandbox()`, real
`build_persona`, real brain `write_persona_config` + `resolve_notes_folder`, real `warnings`
capture — no `claude` subprocess, no live provider. Fake-HOME pattern mirrors
`test_editable_paths._seed_fake_cred` (which patches `sandbox._documents_dir` — REQUIRED so the
notes-scan resolves the fake Documents, not the real `~/Documents`; see C3/MINOR-1).

C2 (existing isolation + F5 suites stay green) and C8/C9 (ruff, brain-untouched) are verified by
running the suite + git status, not by a test here.
"""

from __future__ import annotations

import importlib
import json
import warnings
from pathlib import Path

import pytest

from tests.harness import (
    DEFAULT_MODELS,
    PERSONA_NAME,
    SYNTHETIC_USER,
    PersonaSpec,
    SandboxLeak,
    build_persona,
    sandbox,
)

# The submodule object (NOT the re-exported `sandbox` FUNCTION that shadows it on the package).
sandbox_mod = importlib.import_module("tests.harness.sandbox")

# The brain-default persona-config keys, for the untouched-key preservation checks (C6/C7).
_DEFAULT_CONFIG_KEYS = frozenset(
    {
        "provider",
        "searcher",
        "mcp_audit_log_level",
        "user_name",
        "model",
        "last_opened_at",
        "user_pronouns",
        "notes_enabled",
        "notes_folder",
        "kindled_link_enabled",
        "kindled_relay_url",
    }
)


def _seed_fake_cred(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point HOME at a tmp dir with a fake ~/.claude/.credentials.json (never the real one) and
    point `sandbox._documents_dir` at <fake-home>/Documents so the notes shallow-scan and any notes
    folder share one Documents dir. Returns the fake home."""
    fake_home = tmp_path / "fake-home"
    (fake_home / ".claude").mkdir(parents=True)
    (fake_home / ".claude" / ".credentials.json").write_text('{"fake": true}')
    (fake_home / "Documents").mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(sandbox_mod, "_documents_dir", lambda: fake_home / "Documents")
    return fake_home


def _read_cfg(persona_dir: Path) -> dict:
    return json.loads((persona_dir / "persona_config.json").read_text())


# --- C1: default persona config byte-identical ----------------------------------------------------


def test_default_config_byte_identical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C1: with `PersonaSpec()` defaults, the emitted config equals today's default — the two
    externally-facing values are OFF and the canonical routing fields are unchanged. Pins the full
    parsed dict against the expected default (a drift in any field fails the assert)."""
    _seed_fake_cred(monkeypatch, tmp_path)
    with sandbox() as sb:
        live = build_persona(PersonaSpec(), sb)
        cfg = _read_cfg(live.persona_dir)

    expected = {
        "provider": "claude-cli",
        "searcher": "ddgs",
        "mcp_audit_log_level": "redacted",
        "user_name": SYNTHETIC_USER,
        "model": DEFAULT_MODELS.canary,
        "last_opened_at": None,
        "user_pronouns": None,
        "notes_enabled": False,
        "notes_folder": None,
        "kindled_link_enabled": False,
        "kindled_relay_url": None,
    }
    assert cfg == expected, f"default config drifted: {cfg!r}"


# --- C3: author-enabled notes STILL trips SandboxLeak (the backstop) -------------------------------


def test_notes_folder_appearing_still_trips_leak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C3 (MINOR-1): with notes ON and NO editable_path covering the notes folder, a `Canary Notes`
    folder appearing under Documents (as a real notes write would produce) STILL raises SandboxLeak.

    Asserts the `_documents_dir` patch is active FIRST, so this can never silently scan the real
    `~/Documents` and false-pass. The build itself only writes config (no folder), so the test
    creates the folder inside the run to exercise the `_shallow_notes_fingerprint` leak path — the
    same mechanism a live notes write triggers (no live notes engine in a token-free test)."""
    fake_home = _seed_fake_cred(monkeypatch, tmp_path)
    # MINOR-1 guard: the notes scan must resolve the FAKE Documents, not the real one.
    assert sandbox_mod._documents_dir() == fake_home / "Documents"

    with pytest.raises(SandboxLeak):
        with sandbox(live_check="off") as sb:  # no editable_paths → notes folder is NOT excluded
            build_persona(PersonaSpec(notes_enabled=True), sb)
            # A Canary Notes folder appearing under Documents is the leak signal.
            (fake_home / "Documents" / f"{PERSONA_NAME} Notes").mkdir()


# --- C4 / C5: relay URL emits the loud warning; None emits none ------------------------------------


def test_relay_url_emits_loud_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C4: building a persona with a relay URL set emits a RuntimeWarning naming the URL and flagging
    the network phone-home."""
    _seed_fake_cred(monkeypatch, tmp_path)
    url = "https://relay.example.test/mailbox"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with sandbox() as sb:
            build_persona(PersonaSpec(kindled_relay_url=url), sb)
    relay_warnings = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning) and "kindled_relay_url" in str(w.message)
    ]
    assert relay_warnings, "expected a loud relay RuntimeWarning"
    assert any(url in str(w.message) for w in relay_warnings), "warning must name the relay URL"
    assert any("PHONE HOME" in str(w.message) for w in relay_warnings), "must flag the network risk"


def test_default_emits_no_relay_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C5 (C4's can-fail negative): a default build emits NO relay warning."""
    _seed_fake_cred(monkeypatch, tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with sandbox() as sb:
            build_persona(PersonaSpec(), sb)
    assert not any(
        "kindled_relay_url" in str(w.message) for w in caught
    ), "default build must not emit the relay warning"


# --- C6: relay round-trips + all other keys survive -----------------------------------------------


def test_relay_url_round_trips_and_preserves_other_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C6 (+ MINOR-2/CH8-1): a set relay URL lands in the config with `kindled_link_enabled=True`,
    AND every untouched key still equals the default persona's value (the read-modify-write drops
    no key)."""
    _seed_fake_cred(monkeypatch, tmp_path)
    url = "https://relay.example.test/mailbox"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with sandbox() as sb:
            default_cfg = _read_cfg(build_persona(PersonaSpec(), sb).persona_dir)
        with sandbox() as sb:
            relay_cfg = _read_cfg(build_persona(PersonaSpec(kindled_relay_url=url), sb).persona_dir)

    assert relay_cfg["kindled_relay_url"] == url
    assert relay_cfg["kindled_link_enabled"] is True
    # No key added or dropped:
    assert set(relay_cfg) == _DEFAULT_CONFIG_KEYS == set(default_cfg)
    # Every UNTOUCHED key equals the default:
    for key in _DEFAULT_CONFIG_KEYS - {"kindled_relay_url", "kindled_link_enabled"}:
        assert relay_cfg[key] == default_cfg[key], f"override branch perturbed untouched key {key!r}"


# --- C7: notes round-trips + folder + all other keys survive --------------------------------------


def test_notes_enabled_round_trips_and_preserves_other_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C7 (+ MINOR-2/CH8-1): `notes_enabled=True` lands `notes_enabled=True` + a non-None
    `notes_folder`, AND every untouched key still equals the default."""
    _seed_fake_cred(monkeypatch, tmp_path)
    with sandbox() as sb:
        default_cfg = _read_cfg(build_persona(PersonaSpec(), sb).persona_dir)
    with sandbox() as sb:
        notes_cfg = _read_cfg(build_persona(PersonaSpec(notes_enabled=True), sb).persona_dir)

    assert notes_cfg["notes_enabled"] is True
    assert isinstance(notes_cfg["notes_folder"], str) and notes_cfg["notes_folder"]
    assert PERSONA_NAME in notes_cfg["notes_folder"]  # <Documents>/Canary Notes
    assert set(notes_cfg) == _DEFAULT_CONFIG_KEYS == set(default_cfg)
    for key in _DEFAULT_CONFIG_KEYS - {"notes_enabled", "notes_folder"}:
        assert notes_cfg[key] == default_cfg[key], f"override branch perturbed untouched key {key!r}"

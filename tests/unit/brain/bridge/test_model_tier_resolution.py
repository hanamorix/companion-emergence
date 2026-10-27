"""What the job table does at run time (#529: C3, C5, C13, C27-C32, C35, C38, C40, C43).

The oracle is the argv of the ``claude`` process the REAL provider code builds (the subprocess call is the one seam,
``capture_claude_argv``), so a model chosen anywhere but the tunables accessor cannot pass.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from brain import tunables
from brain.bridge import model_error
from brain.bridge.chat import ChatMessage
from brain.bridge.model_tier import (
    CLAUDE_TIERS,
    OPUS_CHOICE,
    TIER_ATTUNEMENT_DETECTOR,
    TIER_BACKGROUND_CLASSIFIER,
    TIER_BACKGROUND_GENERATIVE,
    TIER_BACKGROUND_HOUSEKEEPING,
    TIER_COMPACTION,
    TIER_INTERACTIVE_CHAT,
    TIER_SELF_MODEL_ARTICULATE,
    configured_job_models,
    model_for_tier,
    set_job_models,
    shipped_default,
)
from brain.bridge.persona_state import build_persona_state
from brain.bridge.provider import ClaudeCliProvider
from brain.utils import log_throttle

_REPO_ROOT = Path(__file__).resolve().parents[4]

_DEFAULTS = {
    TIER_INTERACTIVE_CHAT: "claude-sonnet-5",
    TIER_BACKGROUND_GENERATIVE: "claude-sonnet-5",
    TIER_BACKGROUND_CLASSIFIER: "haiku",
    TIER_BACKGROUND_HOUSEKEEPING: "haiku",
    TIER_COMPACTION: "haiku",
    TIER_SELF_MODEL_ARTICULATE: "haiku",
    TIER_ATTUNEMENT_DETECTOR: "claude-haiku-4-5",
}

# `--model` values captured on origin/main (4d5a9381) with the same capture (a fake `claude` answering a clean frame),
# recorded once for C27: the four Haiku jobs must run exactly what they ran there (the detector: see below). The chat and
# background-generative jobs ran the alias "sonnet" there and run the pinned Sonnet id now (D1, on purpose).
_ORIGIN_MAIN_UNCHANGED = {
    TIER_BACKGROUND_CLASSIFIER: "haiku",
    TIER_BACKGROUND_HOUSEKEEPING: "haiku",
    TIER_COMPACTION: "haiku",
    TIER_SELF_MODEL_ARTICULATE: "haiku",
}
# The detector ran the dated snapshot id there; the owner replaced it with the dateless `claude-haiku-4-5` (2026-10-21,
# C27 amended): still Haiku 4.5, a different spelling, on purpose.
_ORIGIN_MAIN_DETECTOR = "claude-haiku-4-5-20251001"

_KEYS = {tier: "model." + tier.replace("-", "_") for tier in CLAUDE_TIERS}

_MSGS = [ChatMessage(role="user", content="hi")]


@pytest.fixture(autouse=True)
def _fresh_throttle():
    log_throttle._last.clear()
    yield
    log_throttle._last.clear()


@pytest.fixture
def persona_dir(tmp_path) -> Path:
    d = tmp_path / "persona"
    d.mkdir()
    (d / "persona_config.json").write_text('{"provider": "claude-cli"}')
    return d


def _call(provider: ClaudeCliProvider, path: str, persona_dir: Path) -> None:
    if path == "generate":
        provider.generate("hi")
    elif path == "probe":
        provider.probe("hi")
    elif path == "chat":
        provider.chat(_MSGS)
    elif path == "chat_stream":
        list(provider.chat_stream(_MSGS))
    elif path == "chat_mcp":
        provider.chat(_MSGS, tools=[{"name": "t"}], options={"persona_dir": str(persona_dir)})
    else:  # pragma: no cover
        raise AssertionError(path)


_PATHS = ["generate", "probe", "chat", "chat_stream", "chat_mcp"]


# ---------------------------------------------------------------------------
# C3 and C27: argv per job and path, live, no restart
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _PATHS)
def test_every_job_spawns_its_default_and_one_override_line_moves_only_that_job(
    path, persona_dir, capture_claude_argv
) -> None:
    providers = {tier: ClaudeCliProvider(tier) for tier in CLAUDE_TIERS}
    for tier, provider in providers.items():
        _call(provider, path, persona_dir)
        assert capture_claude_argv.last_model == _DEFAULTS[tier] == model_for_tier(tier)
    # ONE override line, no restart, the SAME provider objects
    tunables.set_values({_KEYS[TIER_BACKGROUND_GENERATIVE]: "claude-sonnet-6"})
    for tier, provider in providers.items():
        _call(provider, path, persona_dir)
        expected = "claude-sonnet-6" if tier == TIER_BACKGROUND_GENERATIVE else _DEFAULTS[tier]
        assert capture_claude_argv.last_model == expected
    tunables.set_values({_KEYS[TIER_BACKGROUND_GENERATIVE]: None})  # deleting the line restores the default
    _call(providers[TIER_BACKGROUND_GENERATIVE], path, persona_dir)
    assert capture_claude_argv.last_model == "claude-sonnet-5"


def test_a_provider_holds_a_job_name_not_a_model() -> None:
    provider = ClaudeCliProvider(TIER_COMPACTION)
    assert vars(provider)["_tier"] == TIER_COMPACTION
    assert provider.name() == "claude-cli:haiku"
    tunables.set_values({_KEYS[TIER_COMPACTION]: "claude-haiku-4-5"})
    assert provider.name() == "claude-cli:claude-haiku-4-5"


def test_the_save_time_probe_pins_the_id_it_checks(capture_claude_argv) -> None:
    ClaudeCliProvider(TIER_INTERACTIVE_CHAT, probe_model="claude-opus-9").probe("hi")
    assert capture_claude_argv.last_model == "claude-opus-9"


def test_the_detector_runs_the_dateless_haiku_4_5_id_not_the_origin_main_snapshot(
    persona_dir, capture_claude_argv
) -> None:
    for path in _PATHS:
        _call(ClaudeCliProvider(TIER_ATTUNEMENT_DETECTOR), path, persona_dir)
        assert capture_claude_argv.last_model == "claude-haiku-4-5" != _ORIGIN_MAIN_DETECTOR, path


def test_haiku_jobs_run_what_they_ran_on_origin_main(persona_dir, capture_claude_argv) -> None:
    """C27: argv equality with the recorded origin/main capture (the four Haiku jobs), for every path that builds argv."""
    for tier, was in _ORIGIN_MAIN_UNCHANGED.items():
        for path in _PATHS:
            _call(ClaudeCliProvider(tier), path, persona_dir)
            assert capture_claude_argv.last_model == was, (tier, path)


def test_the_argv_comparison_can_fail(persona_dir, capture_claude_argv, model_override) -> None:
    """The C27 oracle is shown able to fail: change a Haiku job's model and the equality breaks."""
    model_override(TIER_COMPACTION, "claude-haiku-4-5")
    ClaudeCliProvider(TIER_COMPACTION).generate("hi")
    assert capture_claude_argv.last_model != _ORIGIN_MAIN_UNCHANGED[TIER_COMPACTION]


def test_a_saved_sonnet_opus_or_full_id_keeps_running_that_model_after_the_move(
    persona_dir, capture_claude_argv, model_override
) -> None:
    """C27: sonnet -> no line, still Sonnet (the pinned id); opus -> the Opus id; a typed id -> that id."""
    model_override(TIER_INTERACTIVE_CHAT, "sonnet")
    ClaudeCliProvider(TIER_INTERACTIVE_CHAT).chat(_MSGS)
    assert capture_claude_argv.last_model == "claude-sonnet-5"
    assert tunables.get_raw(_KEYS[TIER_INTERACTIVE_CHAT]) is None
    model_override(TIER_INTERACTIVE_CHAT, "opus")
    ClaudeCliProvider(TIER_INTERACTIVE_CHAT).chat(_MSGS)
    assert capture_claude_argv.last_model == OPUS_CHOICE
    model_override(TIER_INTERACTIVE_CHAT, "claude-sonnet-4-6")
    ClaudeCliProvider(TIER_INTERACTIVE_CHAT).chat(_MSGS)
    assert capture_claude_argv.last_model == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# C13: no derivation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chat_id", ["claude-sonnet-5", OPUS_CHOICE])
def test_background_jobs_never_derive_a_model_from_the_chat_line(chat_id, persona_dir, capture_claude_argv) -> None:
    """C13: with the chat line set to a full id, each background job runs ITS OWN line; the id a derivation would
    have produced (claude-haiku-5, which the CLI rejects) never reaches argv."""
    set_job_models({TIER_INTERACTIVE_CHAT: chat_id})
    for tier in CLAUDE_TIERS:
        if tier == TIER_INTERACTIVE_CHAT:
            continue
        ClaudeCliProvider(tier).generate("hi")
        assert capture_claude_argv.last_model == _DEFAULTS[tier]
    assert "claude-haiku-5" not in capture_claude_argv.models


def test_state_and_config_carry_no_derivation_keys(persona_dir) -> None:
    state = build_persona_state(persona_dir)
    for gone in ("tier_fallbacks", "tier_unconfirmed", "derived_unavailable", "tier_models"):
        assert gone not in state and gone not in state["connection"]
    assert "derived_unavailable" not in (persona_dir / "persona_config.json").read_text()


# ---------------------------------------------------------------------------
# C5: a bad value, and an id Claude rejects
# ---------------------------------------------------------------------------

_BAD_VALUES = [123, ["x"], "banana", "claude-" + "x" * 70, "claude-x\n", ""]


@pytest.mark.parametrize("tier", CLAUDE_TIERS)
def test_a_malformed_override_runs_the_default_warns_once_per_value_and_leaves_the_file(tier, caplog) -> None:
    key = _KEYS[tier]
    path = tunables._file_path()
    for bad in _BAD_VALUES:
        tunables._warned_keys.clear()  # the wrong-type warning is once per key per process
        tunables.set_values({key: bad})
        before = path.read_bytes()
        log_throttle._last.clear()
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            results = {model_for_tier(tier) for _ in range(25)}
        assert results == {_DEFAULTS[tier]}, bad
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1, (bad, [r.getMessage() for r in warnings])
        assert path.read_bytes() == before  # the bad text stays in the file untouched (#446)
        tunables._reset_values_cache()
        tunables._warned_keys.clear()


def test_a_bad_value_never_reaches_argv(persona_dir, capture_claude_argv) -> None:
    tunables.set_values({_KEYS[TIER_INTERACTIVE_CHAT]: "banana"})
    ClaudeCliProvider(TIER_INTERACTIVE_CHAT).generate("hi")
    assert capture_claude_argv.last_model == "claude-sonnet-5"


def test_an_unknown_id_fails_once_then_falls_back_and_keeps_the_notice(persona_dir, capture_claude_argv) -> None:
    """C5 (unknown-id half) and the residual of spec s2.4(f): exactly ONE failing call per process per rejected id
    happens before the fallback engages (that call records the rejection), the file is never touched, and the notice
    stays."""
    bad = "claude-sonet-5"
    capture_claude_argv.reject(bad)
    tunables.set_values({_KEYS[TIER_INTERACTIVE_CHAT]: bad})
    before = tunables._file_path().read_bytes()
    provider = ClaudeCliProvider(TIER_INTERACTIVE_CHAT)
    with pytest.raises(Exception):  # noqa: B017 - the CLI's rejection surfaces as the provider's failure
        provider.generate("tick")
    provider.generate("tick")
    assert capture_claude_argv.models == [bad, "claude-sonnet-5"]
    assert [r["model"] for r in model_error.state()] == [bad]
    assert [r["model"] for r in build_persona_state(persona_dir)["model_rejections"]] == [bad]
    assert tunables._file_path().read_bytes() == before


@pytest.mark.parametrize("tier", CLAUDE_TIERS)
def test_every_job_falls_back_to_its_own_default_after_a_rejection(tier, persona_dir, capture_claude_argv) -> None:
    """C28. Limits the spec states, kept here so a later reader does not remove them as defects (C40): (1) the
    fallback cannot help when the rejected id IS the shipped default (see the C29 test): nothing to fall back to;
    (2) exactly one failing call per process per rejected id happens before the fallback engages."""
    bad = "claude-nonexistent-9"
    capture_claude_argv.reject(bad)
    tunables.set_values({_KEYS[tier]: bad})
    provider = ClaudeCliProvider(tier)
    with pytest.raises(Exception):  # noqa: B017
        provider.generate("tick")
    provider.generate("tick")
    assert capture_claude_argv.models == [bad, _DEFAULTS[tier]]


def test_a_rejected_shipped_default_has_no_fallback(persona_dir, capture_claude_argv) -> None:
    """C29 and spec s2.4(f)(1), the stated limit: an old `claude` that does not know the shipped default gives the
    job nothing to fall back to, and a hidden alias substitute would be a second chooser. The job keeps carrying the
    default, one record exists after many calls, and the notice is the only surface. Do not 'fix' this limit."""
    default = _DEFAULTS[TIER_INTERACTIVE_CHAT]
    capture_claude_argv.reject(default)
    provider = ClaudeCliProvider(TIER_INTERACTIVE_CHAT)
    for _ in range(5):
        with pytest.raises(Exception):  # noqa: B017
            provider.generate("tick")
    assert capture_claude_argv.models == [default] * 5
    assert model_for_tier(TIER_INTERACTIVE_CHAT) == default
    records = model_error.state()
    assert [r["model"] for r in records] == [default] and records[0]["count"] == 5
    state = build_persona_state(persona_dir)
    assert [r["model"] for r in state["model_rejections"]] == [default]
    assert state["model_fallbacks"] == []


# ---------------------------------------------------------------------------
# C30: every writer clears the record of the id it writes or removes
# ---------------------------------------------------------------------------

_X = "claude-sonnet-4-9"


def _reject_x_and_fall_back(capture, tier=TIER_INTERACTIVE_CHAT) -> None:
    capture.reject(_X)
    tunables.set_values({_KEYS[tier]: _X})
    p = ClaudeCliProvider(tier)
    with pytest.raises(Exception):  # noqa: B017
        p.generate("tick")
    assert model_for_tier(tier) == _DEFAULTS[tier]  # fallback active
    assert model_error.is_rejected(_X)


def test_set_job_models_clears_the_record_of_a_written_id(capture_claude_argv) -> None:
    """C30 (c, d, e share this path): re-pick X after it was rejected and the job runs X again, the record is gone."""
    _reject_x_and_fall_back(capture_claude_argv)
    capture_claude_argv.accept(_X)
    set_job_models({TIER_INTERACTIVE_CHAT: _X})
    assert not model_error.is_rejected(_X)
    assert model_for_tier(TIER_INTERACTIVE_CHAT) == _X


def test_picking_the_default_word_clears_the_record_of_the_id_it_replaces(capture_claude_argv) -> None:
    """C30 (f): the word that means the job's default writes the default id over the rejected one, and the replaced id's
    record goes with it."""
    _reject_x_and_fall_back(capture_claude_argv)
    set_job_models({TIER_INTERACTIVE_CHAT: "sonnet"})
    assert tunables.get_raw(_KEYS[TIER_INTERACTIVE_CHAT]) == _DEFAULTS[TIER_INTERACTIVE_CHAT]
    assert not model_error.is_rejected(_X)
    assert model_error.state() == []


def test_the_reset_path_clears_the_record_of_the_id_it_replaces(capture_claude_argv) -> None:
    """C30 (f), rev4: `clear-tier` resets the job to its default (the entry stays) and forgets the replaced id."""
    _reject_x_and_fall_back(capture_claude_argv)
    set_job_models({TIER_INTERACTIVE_CHAT: None})
    assert tunables.get_raw(_KEYS[TIER_INTERACTIVE_CHAT]) == _DEFAULTS[TIER_INTERACTIVE_CHAT]
    assert not model_error.is_rejected(_X)
    assert model_error.state() == []


@pytest.mark.parametrize("outcome", ["ok", "inconclusive"])
def test_the_picker_save_clears_the_record_after_a_conclusive_or_an_inconclusive_probe(
    outcome, capture_claude_argv, monkeypatch
) -> None:
    """C30 (a, b)."""
    import asyncio

    from brain.bridge import server

    _reject_x_and_fall_back(capture_claude_argv)
    capture_claude_argv.accept(_X)
    monkeypatch.setattr(server, "_check_model_blocking", lambda m, t=None: (outcome, "timeout" if outcome != "ok" else ""))
    status, body = asyncio.run(server._apply_chat_model(_X, check=True, lock=asyncio.Lock()))
    assert status == 200 and body["model"] == _X
    assert not model_error.is_rejected(_X)
    assert model_for_tier(TIER_INTERACTIVE_CHAT) == _X


# ---------------------------------------------------------------------------
# C31: an edit by ANOTHER process ends the fallback with no restart
# ---------------------------------------------------------------------------

_EDITOR = """
import sys
sys.path.insert(0, {root!r})
from brain.bridge.model_tier import set_job_models
set_job_models({{"background-classifier": sys.argv[1]}})
"""


def _bump_mtime_past_the_record() -> None:
    """Make the file strictly NEWER than the rejection record yet older than the next failing call (halfway)."""
    import time

    rec = model_error._rejected[_X]["last_epoch"]
    mid = rec + (time.time() - rec) / 2
    os.utime(tunables._file_path(), (mid, mid))


@pytest.mark.parametrize("edit", ["nell_model_set_style_process", "hand_edit_of_any_key", "another_bridge_boot"])
def test_a_write_by_another_process_ends_the_fallback_and_a_still_bad_id_falls_back_again(
    edit, tmp_path, capture_claude_argv
) -> None:
    _reject_x_and_fall_back(capture_claude_argv)
    if edit == "nell_model_set_style_process":
        code = _EDITOR.format(root=str(_REPO_ROOT))
        subprocess.run(
            [sys.executable, "-c", code, "haiku"], check=True, env={**os.environ, "KINDLED_HOME": str(tunables._file_path().parent)}
        )
    elif edit == "hand_edit_of_any_key":
        path = tunables._file_path()
        path.write_text(path.read_text(encoding="utf-8") + '"some.other.key" = 1\n', encoding="utf-8")
    else:  # another persona's bridge boot rewrites the whole file
        tunables.sync_file()
    _bump_mtime_past_the_record()
    provider = ClaudeCliProvider(TIER_INTERACTIVE_CHAT)
    # no restart, no state reset: the id is tried again
    assert model_for_tier(TIER_INTERACTIVE_CHAT) == _X
    with pytest.raises(Exception):  # noqa: B017
        provider.generate("tick")
    assert capture_claude_argv.last_model == _X
    # still rejected: the failing call re-created the record and the fallback re-engages
    assert model_error.is_rejected(_X)
    assert model_for_tier(TIER_INTERACTIVE_CHAT) == _DEFAULTS[TIER_INTERACTIVE_CHAT]
    assert [r["model"] for r in model_error.state()] == [_X]  # the notice never disappeared


def test_a_record_is_trusted_only_while_it_is_newer_than_the_file(capture_claude_argv) -> None:
    """Equal timestamps (forced with utime): the record is NOT newer than the file, so it is untrusted and the id is
    tried again; one tick older than the record and it is trusted."""
    _reject_x_and_fall_back(capture_claude_argv)
    rec = model_error._rejected[_X]["last_epoch"]
    os.utime(tunables._file_path(), (rec, rec))
    assert model_for_tier(TIER_INTERACTIVE_CHAT) == _X
    os.utime(tunables._file_path(), (rec - 1, rec - 1))
    assert model_for_tier(TIER_INTERACTIVE_CHAT) == _DEFAULTS[TIER_INTERACTIVE_CHAT]


# ---------------------------------------------------------------------------
# C43: an id the user moved away from is un-ignored when it is configured again
# ---------------------------------------------------------------------------


def test_an_id_moved_away_from_and_configured_again_by_an_external_write_still_gets_its_rejection_noticed(
    capture_claude_argv,
) -> None:
    """The #440 failure: `switch(X, Y)` ignores X; a later external write sets X again; the CLI rejects X. Without
    the `activate` call in model_for_tier the failure is dropped silently (no record, no fallback, no notice)."""
    model_error.switch(_X, "claude-sonnet-5")  # the picker moved the chat model X -> Y: X is now ignored
    capture_claude_argv.reject(_X)
    tunables.set_values({_KEYS[TIER_INTERACTIVE_CHAT]: _X})  # an external write (CLI / hand edit) sets X again
    provider = ClaudeCliProvider(TIER_INTERACTIVE_CHAT)
    with pytest.raises(Exception):  # noqa: B017
        provider.generate("tick")
    assert capture_claude_argv.last_model == _X
    assert model_error.is_rejected(_X)  # recorded, not dropped as an ignored id
    provider.generate("tick")
    assert capture_claude_argv.models == [_X, "claude-sonnet-5"]  # the fallback engaged on the call after
    assert [r["model"] for r in model_error.state()] == [_X]


# ---------------------------------------------------------------------------
# C32 and C35: what /persona/state says
# ---------------------------------------------------------------------------


def test_while_a_fallback_is_active_the_notice_names_the_configured_id(persona_dir, capture_claude_argv, caplog) -> None:
    capture_claude_argv.reject(_X)
    tunables.set_values({_KEYS[TIER_INTERACTIVE_CHAT]: _X, _KEYS[TIER_COMPACTION]: _X})
    with caplog.at_level(logging.WARNING):
        with pytest.raises(Exception):  # noqa: B017
            ClaudeCliProvider(TIER_INTERACTIVE_CHAT).generate("tick")
    state = build_persona_state(persona_dir)
    assert [r["model"] for r in state["model_rejections"]] == [_X]  # the CONFIGURED id stays in the notice
    fallbacks = {f["job"]: f for f in state["model_fallbacks"]}
    assert fallbacks[TIER_INTERACTIVE_CHAT] == {"job": TIER_INTERACTIVE_CHAT, "configured": _X, "using": "claude-sonnet-5"}
    assert fallbacks[TIER_COMPACTION]["using"] == "haiku"
    assert state["connection"]["model"] == "claude-sonnet-5"  # the effective id feeds the connection line
    assert state["connection"]["model_choice"] == "sonnet"
    texts = [r.getMessage() for r in caplog.records if "rejected the model" in r.getMessage()]
    assert texts and all("—" not in t for t in texts)
    assert all("shipped default" in t for t in texts)
    assert all("keep failing until the model is changed in the connection panel" not in t for t in texts)


def test_the_warning_for_a_rejected_shipped_default_says_the_job_keeps_failing(persona_dir, capture_claude_argv, caplog) -> None:
    default = _DEFAULTS[TIER_INTERACTIVE_CHAT]
    capture_claude_argv.reject(default)
    with caplog.at_level(logging.WARNING), pytest.raises(Exception):  # noqa: B017
        ClaudeCliProvider(TIER_INTERACTIVE_CHAT).generate("tick")
    text = next(r.getMessage() for r in caplog.records if "rejected the model" in r.getMessage())
    assert "keep failing until the Claude CLI is updated or the model setting is changed" in text and "—" not in text


def test_model_unconfirmed_is_keyed_on_the_chat_id_as_configured(persona_dir, capture_claude_argv) -> None:
    capture_claude_argv.reject(_X)
    tunables.set_values({_KEYS[TIER_INTERACTIVE_CHAT]: _X})
    model_error.note_unconfirmed(_X)
    with pytest.raises(Exception):  # noqa: B017
        ClaudeCliProvider(TIER_INTERACTIVE_CHAT).generate("tick")
    assert build_persona_state(persona_dir)["model_unconfirmed"] == [_X]


def test_model_unsupported_reports_the_text_in_the_file(persona_dir) -> None:
    tunables.set_values({_KEYS[TIER_INTERACTIVE_CHAT]: "banana\x07"})
    assert build_persona_state(persona_dir)["model_unsupported"] == "banana"
    assert model_for_tier(TIER_INTERACTIVE_CHAT) == "claude-sonnet-5"


def test_a_non_claude_persona_reports_no_model(tmp_path) -> None:
    """C35: an ollama persona's own model is not a Claude id, the picker has nothing to show, and the unsupported
    line (a Claude setting) is not reported for it either."""
    d = tmp_path / "ollama_persona"
    d.mkdir()
    (d / "persona_config.json").write_text('{"provider": "ollama"}')
    tunables.set_values({_KEYS[TIER_INTERACTIVE_CHAT]: "banana"})
    state = build_persona_state(d)
    assert state["connection"]["provider"] == "ollama"
    assert state["connection"]["model"] is None and state["connection"]["model_choice"] is None
    assert state["model_unsupported"] is None and state["model_unconfirmed"] == []


def test_the_chat_choice_follows_the_effective_id(persona_dir, model_override) -> None:
    assert build_persona_state(persona_dir)["connection"]["model_choice"] == "sonnet"
    model_override(TIER_INTERACTIVE_CHAT, "opus")
    conn = build_persona_state(persona_dir)["connection"]
    assert conn["model"] == OPUS_CHOICE and conn["model_choice"] == "opus"
    model_override(TIER_INTERACTIVE_CHAT, "claude-sonnet-4-6")
    conn = build_persona_state(persona_dir)["connection"]
    assert conn["model"] == "claude-sonnet-4-6" and conn["model_choice"] is None  # custom: no radio highlighted
    model_override(TIER_INTERACTIVE_CHAT, "sonnet")  # the removed override still highlights Sonnet
    assert build_persona_state(persona_dir)["connection"]["model_choice"] == "sonnet"


# ---------------------------------------------------------------------------
# C38: no import cycle, in the import orders that differ
# ---------------------------------------------------------------------------

# Every fresh interpreter imports all of these; only the module imported FIRST varies.
_MODULES = [
    "brain.bridge.model_error",
    "brain.bridge.model_tier",
    "brain.bridge.provider",
    "brain.bridge.persona_state",
    "brain.tunables",
]
# Three first imports are enough, read off the import graph. Importing any `brain.bridge.X` first runs
# `brain/bridge/__init__.py` first, which imports provider, which loads tunables, model_error and model_tier in one fixed
# order, so model_error-, model_tier- and provider-first are the same load sequence (the cycle between model_error and
# model_tier is broken by the function-scoped `from brain.bridge import model_error` in model_tier.py `model_for_tier` and
# `set_job_models`), and persona_state-first is that sequence plus its own imports (nothing in the chain imports it).
# tunables-first is the one other order (tunables imports nothing from brain.bridge). So there are two distinct orders
# today; the third entry is persona_state, the one module the chain does not load. The premise is checked, not assumed:
# the bridge entries first import the bare package and assert that the chain was loaded by it; if that fails the five
# orders are different processes again and the dropped entries (model_tier, provider) must be restored.
_FIRST = ["brain.bridge.model_error", "brain.bridge.persona_state", "brain.tunables"]
_PREMISE = (
    "import sys\n"
    "import brain.bridge\n"
    "_missing = [m for m in ('brain.bridge.provider', 'brain.bridge.model_error', 'brain.bridge.model_tier', "
    "'brain.tunables') if m not in sys.modules]\n"
    "if _missing:\n"
    "    raise SystemExit(f'importing brain.bridge no longer loads {_missing}, so the import orders are different "
    "again: restore the first-import entries model_tier and provider')\n"
)


def _import_script(first: str) -> str:
    """The program a fresh interpreter runs: (for a bridge module, check the premise,) import ``first``, then every other
    module of ``_MODULES``."""
    premise = _PREMISE if first.startswith("brain.bridge.") else ""
    others = [m for m in _MODULES if m != first]
    return premise + f"import {first}\n" + "".join(f"import {m}\n" for m in others) + "print('ok')\n"


@pytest.mark.parametrize("first", _FIRST)
def test_every_import_order_succeeds_in_a_fresh_interpreter(first, tmp_path) -> None:
    out = subprocess.run(
        [sys.executable, "-c", _import_script(first)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT), "KINDLED_HOME": str(tmp_path)},
    )
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr[-600:]


def test_configured_ids_ignore_the_rejection_state(capture_claude_argv) -> None:
    capture_claude_argv.reject(_X)
    tunables.set_values({_KEYS[TIER_COMPACTION]: _X})
    with pytest.raises(Exception):  # noqa: B017
        ClaudeCliProvider(TIER_COMPACTION).generate("tick")
    assert configured_job_models()[TIER_COMPACTION] == _X
    assert model_for_tier(TIER_COMPACTION) == shipped_default(TIER_COMPACTION)

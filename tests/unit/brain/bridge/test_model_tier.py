"""The job table and the picker vocabulary of brain/bridge/model_tier.py (#529: C1, C2, C4, C34, C35, C39, C41).

One tunables line per Claude job decides its model; ``model_for_tier`` reads it on every call. The behaviours that need
a spawned ``claude`` (argv per job, live edits, fallback after a rejection) are in test_model_tier_resolution.py; the
guard that nothing outside this module chooses a model is the model case of the shared-settings guard
(tests/unit/test_shared_settings_guard.py).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from brain import tunables
from brain.bridge import model_tier
from brain.bridge.model_tier import (
    CLAUDE_TIERS,
    LOCAL_MODEL_IDS,
    MODEL_EMBEDDING,
    MODEL_EMBEDDING_DIM,
    MODEL_RELEVANCE_JUDGE,
    MODEL_RERANKER,
    OPUS_CHOICE,
    TIER_ATTUNEMENT_DETECTOR,
    TIER_BACKGROUND_CLASSIFIER,
    TIER_BACKGROUND_GENERATIVE,
    TIER_BACKGROUND_HOUSEKEEPING,
    TIER_COMPACTION,
    TIER_EMBEDDING,
    TIER_INTERACTIVE_CHAT,
    TIER_RELEVANCE_JUDGE,
    TIER_RERANKER,
    TIER_SELF_MODEL_ARTICULATE,
    build_tier_provider,
    choice_for,
    configured_job_models,
    model_for_tier,
    model_label_for_provider,
    resolve_choice,
    set_job_models,
    shipped_default,
)
from brain.bridge.provider import ClaudeCliProvider
from tests.source_tree import brain_sources, tree_sources

_REPO_ROOT = Path(__file__).resolve().parents[4]

_EXPECTED_JOBS = {
    TIER_INTERACTIVE_CHAT: ("model.interactive_chat", "claude-sonnet-5"),
    TIER_BACKGROUND_GENERATIVE: ("model.background_generative", "claude-sonnet-5"),
    TIER_BACKGROUND_CLASSIFIER: ("model.background_classifier", "haiku"),
    TIER_BACKGROUND_HOUSEKEEPING: ("model.background_housekeeping", "haiku"),
    TIER_COMPACTION: ("model.compaction", "haiku"),
    TIER_SELF_MODEL_ARTICULATE: ("model.self_model_articulate", "haiku"),
    TIER_ATTUNEMENT_DETECTOR: ("model.attunement_detector", "claude-haiku-4-5"),
}


def _claude_cli_persona(tmp_path: Path) -> Path:
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "persona_config.json").write_text('{"provider": "claude-cli"}')
    return persona_dir


# ---------------------------------------------------------------------------
# C1: one commented line per job, the shipped defaults, one extra literal
# ---------------------------------------------------------------------------


def test_the_seven_jobs_are_registered_with_their_defaults() -> None:
    """A fresh process that imports model_tier registers exactly these seven model keys with these defaults.

    A subprocess, not the live registry: the registry is process-global and other tests clear it
    (``tunables._reset_for_tests``), so reading it here depended on which tests ran earlier in the same process (the shipped-runtime
    CI job runs a subset in which a clearing test comes first).
    """
    code = (
        "import json, brain.tunables as t; import brain.bridge.model_tier; "
        "print(json.dumps({k: v for k, v in t._registry.items() if k.startswith('model.')}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=_REPO_ROOT, capture_output=True, text=True, check=True, timeout=120
    ).stdout
    assert json.loads(out.strip().splitlines()[-1]) == dict(_EXPECTED_JOBS.values())
    assert set(CLAUDE_TIERS) == set(_EXPECTED_JOBS)
    for tier, (_key, default) in _EXPECTED_JOBS.items():
        assert shipped_default(tier) == default
        assert model_for_tier(tier) == default


def test_every_job_registration_carries_a_description_saying_what_the_job_is() -> None:
    """C1 (rev4): each `tunables.register("model.<job>", <default>, description=...)` has a description of real length that
    says what the job is and holds no model id (the reviewer judges the wording, C1b); the registry snapshot has the 7 keys,
    their 7 defaults and 7 non-empty descriptions; and OPUS_CHOICE is the one other model literal, also commented."""
    import ast

    source = (_REPO_ROOT / "brain" / "bridge" / "model_tier.py").read_text(encoding="utf-8")
    seen: dict[str, str] = {}
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and str(node.args[0].value).startswith("model.")
        ):
            desc = next((k.value for k in node.keywords if k.arg == "description"), None)
            assert isinstance(desc, ast.Constant), f"{node.args[0].value}: needs a description= argument"
            seen[node.args[0].value] = desc.value
    assert set(seen) == {key for key, _ in _EXPECTED_JOBS.values()}
    for key, desc in seen.items():
        assert len(desc.split()) >= 6 and not re.search(r"claude-|haiku\b", desc), f"{key}: {desc!r}"
        assert tunables._descriptions[key] == desc  # stored beside the registry
        assert tunables._registry[key] == dict(_EXPECTED_JOBS.values())[key]
    opus_lines = [ln for ln in source.splitlines() if ln.startswith("OPUS_CHOICE")]
    assert len(opus_lines) == 1 and "#" in source[source.index(opus_lines[0]) - 400 : source.index(opus_lines[0])]
    assert OPUS_CHOICE == "claude-opus-5"
    # the detector's rationale stays a code comment beside its registration (it is not description text)
    assert "newest 4.5 snapshot" in source


# ---------------------------------------------------------------------------
# C2: the old machinery is gone
# ---------------------------------------------------------------------------

_REMOVED_EVERYWHERE = (
    "MODEL_LITTLE MODEL_MEDIUM MODEL_BIGGEST TIER_MODEL TIER_DEV_CLI NAMED_CHOICES _ATTUNEMENT_DETECTOR_MODEL "
    "DEFAULT_MODEL COMPACTION_MODEL SELF_MODEL_MODEL _default_model_for ROUTED_TIER_NAMES ROUTED_CLAUDE_TIERS "
    "_FAMILIES tier_family derive_model derived_tier_model resolve_tier_model candidate_tier_models unchecked_tiers "
    "tier_fallbacks build_interactive_chat_provider _resolve_provider_kind _load_config model_override "
    "model_unsupported_display set_model _warn_bad_once _warn_unknown_model_once _apply_recheck_outcomes "
    "_RECHECK_STAMP _recheck_recent _stamp_recheck _recheck_needed _recheck_derived_unavailable "
    "derived_recheck_task _probe_models hot_swap before_write note_tier_unconfirmed clear_tier_unconfirmed "
    "tier_unconfirmed _tier_unconfirmed _model_set_via_bridge _bridge_running _BridgeStartedMidCommandError "
    "_bridge_text _BRIDGE_ANSWER_MAX_BYTES _model_cfg_path DEFAULT_INIT_MODEL DEFAULT_WIZARD_MODEL "
    "TierFallbackInfo tierFallbackMessage rejected_models"
).split()
# The retired per-persona keys: only the pass-through field and the migration may name them.
_LEGACY_KEYS = ("tier_models", "derived_unavailable")
_LEGACY_OWNERS = ("brain/persona_config.py", "brain/model_migration.py")


def _source_files():
    """``(path relative to the repository, text)`` of the product source: brain, the app, scripts (read once per process)."""
    for root, suffixes in (
        ("brain", (".py",)),
        ("app/src", (".ts", ".tsx")),
        ("app/src-tauri/src", (".rs",)),
        ("scripts", (".py", ".sh")),
    ):
        yield from tree_sources(root, suffixes, ("node_modules",)).items()


# Tests that name removed identifiers ON PURPOSE (to assert they are gone, or to feed the guard a violating fixture).
_TESTS_NAMING_REMOVED_NAMES = (
    "tests/unit/brain/bridge/test_model_tier.py",
    "tests/unit/brain/bridge/test_model_tier_resolution.py",
    "tests/shared_settings.py",
    "tests/unit/test_shared_settings_guard.py",
)
# In tests, `model_override` is the live conftest fixture's name; only the removed KEYWORD form is a leftover.
_REMOVED_KWARG = re.compile(r"(?<![\w])model_override\s*=(?!=)")


def _test_files():
    for path in sorted((_REPO_ROOT / "tests").rglob("*.py")):
        if path.relative_to(_REPO_ROOT).as_posix() not in _TESTS_NAMING_REMOVED_NAMES:
            yield path


@pytest.mark.slow(reason='1.4 s of CPU in its call phase on a 4-core dev VM, 1.4 s wall (serial run for #540, J12)')
def test_removed_names_exist_nowhere_in_code_or_comments() -> None:
    """C2 and C20: every removed name has zero hits, code AND comments/docstrings, in brain, the app, scripts and tests."""
    names = [n for n in _REMOVED_EVERYWHERE if n != "model_override"]
    pattern = re.compile(r"(?<![\w])(" + "|".join(map(re.escape, _REMOVED_EVERYWHERE)) + r")(?![\w])")
    test_pattern = re.compile(r"(?<![\w])(" + "|".join(map(re.escape, names)) + r")(?![\w])")
    legacy = re.compile(r"(?<![\w])(" + "|".join(_LEGACY_KEYS) + r")(?![\w])")
    hits: list[str] = []
    for rel, text in _source_files():
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{rel}:{lineno}: {line.strip()[:90]}")
            if legacy.search(line) and rel not in _LEGACY_OWNERS:
                hits.append(f"{rel}:{lineno}: {line.strip()[:90]}")
    for path in _test_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if test_pattern.search(line) or _REMOVED_KWARG.search(line):
                hits.append(f"{rel}:{lineno}: {line.strip()[:90]}")
    assert hits == [], "removed names still present:\n" + "\n".join(hits)


def test_the_tests_scan_catches_a_removed_keyword_and_a_removed_name_in_a_comment() -> None:
    """The M1 class (a test calling a removed parameter) must be visible to the scan."""
    assert _REMOVED_KWARG.search('get_provider("claude-cli", model_override="haiku")')
    assert not _REMOVED_KWARG.search("def test_x(persona_dir, model_override) -> None:")
    assert not _REMOVED_KWARG.search("model_override(TIER_COMPACTION, 'x')")


# ---------------------------------------------------------------------------
# model_for_tier and the non-Claude tiers
# ---------------------------------------------------------------------------


def test_model_for_tier_resolves_every_declared_tier() -> None:
    for tier in (*CLAUDE_TIERS, TIER_EMBEDDING, TIER_RERANKER, TIER_RELEVANCE_JUDGE):
        assert isinstance(model_for_tier(tier), str)


def test_model_for_tier_unknown_tier_raises_keyerror() -> None:
    """Fail loud on a typo'd tier name, never silently default."""
    with pytest.raises(KeyError):
        model_for_tier("not-a-real-tier")


def test_non_claude_tiers_keep_their_constants() -> None:
    assert model_for_tier(TIER_EMBEDDING) == MODEL_EMBEDDING == "intfloat/multilingual-e5-large"
    assert model_for_tier(TIER_RERANKER) == MODEL_RERANKER
    assert model_for_tier(TIER_RELEVANCE_JUDGE) == MODEL_RELEVANCE_JUDGE == "BAAI/bge-reranker-v2-m3"
    assert MODEL_EMBEDDING_DIM == 1024
    assert set(LOCAL_MODEL_IDS) == {TIER_EMBEDDING, TIER_RERANKER, TIER_RELEVANCE_JUDGE}


def test_local_tiers_are_repointable_in_this_module_only(monkeypatch) -> None:
    """The embedder, reranker and judge ids are plain constants resolved by the same accessor (O1)."""
    monkeypatch.setitem(LOCAL_MODEL_IDS, TIER_EMBEDDING, "some/other-embedding-model")
    assert model_for_tier(TIER_EMBEDDING) == "some/other-embedding-model"
    assert model_for_tier(TIER_BACKGROUND_CLASSIFIER) == "haiku"


def test_the_detector_stays_on_haiku_4_5_by_its_dateless_id_not_the_haiku_alias() -> None:
    """Owner ruling 2026-10-21: the dateless id (the CLI accepts it and it follows the newest 4.5 snapshot) replaces the
    dated snapshot id; the `haiku` alias would move the detector off 4.5 when a newer Haiku ships."""
    from brain.bridge.model_tier import is_valid_model

    assert model_for_tier(TIER_ATTUNEMENT_DETECTOR) == "claude-haiku-4-5"
    assert is_valid_model("claude-haiku-4-5")


# ---------------------------------------------------------------------------
# C4: pinning is one line, and the function repeats no literal
# ---------------------------------------------------------------------------


def test_editing_the_job_table_default_reaches_the_argv_with_no_second_edit(monkeypatch, capture_claude_argv) -> None:
    """C4: the table entry is the only holder of the default; patching it moves what the provider spawns."""
    key, _ = model_tier._CLAUDE_JOBS[TIER_INTERACTIVE_CHAT]
    monkeypatch.setitem(model_tier._CLAUDE_JOBS, TIER_INTERACTIVE_CHAT, (key, "claude-sonnet-6"))
    assert model_for_tier(TIER_INTERACTIVE_CHAT) == "claude-sonnet-6"
    ClaudeCliProvider(TIER_INTERACTIVE_CHAT).generate("hi")
    assert capture_claude_argv.last_model == "claude-sonnet-6"
    assert model_for_tier(TIER_BACKGROUND_GENERATIVE) == "claude-sonnet-5"  # one line per job, no size family


# ---------------------------------------------------------------------------
# C34: the picker vocabulary and the write rule
# ---------------------------------------------------------------------------


def test_the_three_words_resolve_through_the_job_table() -> None:
    assert resolve_choice("sonnet") == shipped_default(TIER_INTERACTIVE_CHAT) == "claude-sonnet-5"
    assert resolve_choice("opus") == OPUS_CHOICE
    assert resolve_choice("haiku") == "haiku"
    assert resolve_choice("claude-haiku-4-5") == "claude-haiku-4-5"
    assert resolve_choice("banana") is None
    assert resolve_choice("claude-") is None


def test_choice_for_is_the_exact_reverse_and_none_for_a_custom_id() -> None:
    assert choice_for("claude-sonnet-5") == "sonnet"
    assert choice_for(OPUS_CHOICE) == "opus"
    assert choice_for("haiku") == "haiku"
    assert choice_for("claude-sonnet-4-6") is None  # a custom id highlights no radio
    assert choice_for(None) is None


def test_per_job_equality_of_the_words() -> None:
    """`sonnet` equals the default of chat and background-generative but not of the Haiku jobs; `haiku` equals the default of
    the four Haiku jobs, not of chat (so `set-tier <haiku job> haiku` writes nothing new and `... sonnet` writes the id)."""
    chat, gen = shipped_default(TIER_INTERACTIVE_CHAT), shipped_default(TIER_BACKGROUND_GENERATIVE)
    assert resolve_choice("sonnet") == chat == gen
    for tier in (TIER_BACKGROUND_CLASSIFIER, TIER_BACKGROUND_HOUSEKEEPING, TIER_COMPACTION, TIER_SELF_MODEL_ARTICULATE):
        assert resolve_choice("sonnet") != shipped_default(tier)
        assert resolve_choice("haiku") == shipped_default(tier)
    assert resolve_choice("haiku") != chat


def test_the_sonnet_word_must_stay_a_sonnet_model() -> None:
    """If the chat default ever moves to a non-Sonnet id, the picker's Sonnet radio would silently follow it to that
    model. Fail loudly so the picker words are edited on purpose (spec s3.2)."""
    assert "sonnet" in shipped_default(TIER_INTERACTIVE_CHAT), (
        "The chat job's shipped default is no longer a Sonnet id, but the picker's 'Sonnet' word resolves through "
        "it: edit the picker words (ModelPicker.tsx, resolve_choice) deliberately before changing it."
    )


def test_no_table_is_keyed_by_size() -> None:
    assert not any(n.startswith("MODEL_") and n.lower() in ("model_little", "model_medium") for n in dir(model_tier))


def test_the_write_rule_through_set_job_models() -> None:
    """A pick WRITES its id as the entry's value (nothing is removed); a value equal to the current one writes nothing; `None`
    resets the job to its shipped default and keeps the entry."""
    chat_key = "model.interactive_chat"
    assert set_job_models({TIER_INTERACTIVE_CHAT: "sonnet"}) == {}  # already the default: nothing written
    assert not tunables._file_path().exists()
    assert set_job_models({TIER_INTERACTIVE_CHAT: "opus"}) == {TIER_INTERACTIVE_CHAT: OPUS_CHOICE}
    assert tunables.get_raw(chat_key) == OPUS_CHOICE
    assert set_job_models({TIER_INTERACTIVE_CHAT: "sonnet"}) == {TIER_INTERACTIVE_CHAT: "claude-sonnet-5"}
    assert tunables.get_raw(chat_key) == "claude-sonnet-5"  # the entry holds the id the word means
    assert set_job_models({TIER_INTERACTIVE_CHAT: "claude-sonnet-5"}) == {}  # a full id equal to the current value: nothing
    assert set_job_models({TIER_INTERACTIVE_CHAT: "claude-sonnet-4-6"}) == {TIER_INTERACTIVE_CHAT: "claude-sonnet-4-6"}
    assert set_job_models({TIER_INTERACTIVE_CHAT: "haiku"}) == {TIER_INTERACTIVE_CHAT: "haiku"}
    assert set_job_models({TIER_BACKGROUND_CLASSIFIER: "haiku"}) == {}  # that job's value already is haiku
    assert set_job_models({TIER_BACKGROUND_CLASSIFIER: "sonnet"}) == {TIER_BACKGROUND_CLASSIFIER: "claude-sonnet-5"}
    assert set_job_models({TIER_BACKGROUND_GENERATIVE: "sonnet"}) == {}
    assert set_job_models({TIER_INTERACTIVE_CHAT: None}) == {TIER_INTERACTIVE_CHAT: "claude-sonnet-5"}  # reset
    assert tunables.get_raw(chat_key) == "claude-sonnet-5"
    assert set_job_models({TIER_INTERACTIVE_CHAT: None}) == {}  # already reset


def test_set_job_models_rejects_an_unusable_value_and_an_unknown_job() -> None:
    with pytest.raises(ValueError):
        set_job_models({TIER_INTERACTIVE_CHAT: "banana"})
    with pytest.raises(KeyError):
        set_job_models({"not-a-job": "haiku"})
    assert tunables.get_raw("model.interactive_chat") is None


def test_set_job_models_only_if_default_leaves_a_changed_job_alone() -> None:
    set_job_models({TIER_COMPACTION: "claude-haiku-4-5"})
    written = set_job_models({TIER_COMPACTION: "claude-haiku-4-9", TIER_BACKGROUND_CLASSIFIER: "claude-haiku-4-8"}, only_if_default=True)
    assert written == {TIER_BACKGROUND_CLASSIFIER: "claude-haiku-4-8"}
    assert tunables.get_raw("model.compaction") == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# C48, C50: a value equal to the default follows a default bump; a value the user changed does not (owner E2, E7)
# ---------------------------------------------------------------------------


def _bump_chat_default(monkeypatch, new: str) -> None:
    """Move the chat job's shipped default to ``new`` the way a new version would: the job table and the registry, then the
    boot pass. (A running process does not follow a code change; the next boot's `sync_file` carries it to the file.)"""
    key, _old = model_tier._CLAUDE_JOBS[TIER_INTERACTIVE_CHAT]
    monkeypatch.setitem(model_tier._CLAUDE_JOBS, TIER_INTERACTIVE_CHAT, (key, new))
    monkeypatch.setitem(tunables._registry, key, new)
    tunables.sync_file()


def test_a_value_equal_to_the_default_follows_a_bump_however_it_got_there(monkeypatch, capture_claude_argv) -> None:
    """C48 (i) picked `sonnet`, (ii) typed the default id verbatim, (iii) tried another model then set the default id back.

    Owner E7, verbatim: "only reason it'd be a typed replacement for the default is if someone tried a different one and
    went back to default. thus, default holds". No special case for typed ids: do not "fix" (ii) and (iii).
    """
    ways = {
        "picked the word": lambda: set_job_models({TIER_INTERACTIVE_CHAT: "sonnet"}),
        "typed the default id": lambda: set_job_models({TIER_INTERACTIVE_CHAT: "claude-sonnet-5"}),
        "went back to the default": lambda: (
            set_job_models({TIER_INTERACTIVE_CHAT: "claude-opus-4-9"}),
            set_job_models({TIER_INTERACTIVE_CHAT: "claude-sonnet-5"}),
        ),
    }
    for name, act in ways.items():
        tunables._file_path().unlink(missing_ok=True)
        tunables._reset_values_cache()
        tunables.sync_file()  # a file as an earlier boot left it, recording default "claude-sonnet-5"
        act()
        _bump_chat_default(monkeypatch, "claude-sonnet-6")
        ClaudeCliProvider(TIER_INTERACTIVE_CHAT).generate("hi")
        assert capture_claude_argv.last_model == "claude-sonnet-6", name
        _bump_chat_default(monkeypatch, "claude-sonnet-5")  # back, for the next way


def test_a_value_the_user_changed_keeps_after_the_same_bump(monkeypatch, capture_claude_argv) -> None:
    """C48 (iv) and C50: the second user changed the value to a different id and keeps it."""
    tunables.sync_file()
    set_job_models({TIER_INTERACTIVE_CHAT: "claude-sonnet-4-6"})
    _bump_chat_default(monkeypatch, "claude-sonnet-6")
    ClaudeCliProvider(TIER_INTERACTIVE_CHAT).generate("hi")
    assert capture_claude_argv.last_model == "claude-sonnet-4-6"


def test_the_bump_is_not_seen_by_a_running_process_until_the_boot_pass(monkeypatch, capture_claude_argv) -> None:
    """C4 (rev4): a running process does not follow a code change; the file's value is in force until `sync_file` runs."""
    tunables.sync_file()
    key, _old = model_tier._CLAUDE_JOBS[TIER_INTERACTIVE_CHAT]
    monkeypatch.setitem(model_tier._CLAUDE_JOBS, TIER_INTERACTIVE_CHAT, (key, "claude-sonnet-6"))
    monkeypatch.setitem(tunables._registry, key, "claude-sonnet-6")
    ClaudeCliProvider(TIER_INTERACTIVE_CHAT).generate("hi")
    assert capture_claude_argv.last_model == "claude-sonnet-5"  # the file still says the old default
    tunables.sync_file()
    ClaudeCliProvider(TIER_INTERACTIVE_CHAT).generate("hi")
    assert capture_claude_argv.last_model == "claude-sonnet-6"


# ---------------------------------------------------------------------------
# build_tier_provider, labels, C35
# ---------------------------------------------------------------------------


def test_build_tier_provider_builds_the_claude_provider_for_the_job(tmp_path, capture_claude_argv) -> None:
    persona_dir = _claude_cli_persona(tmp_path)
    for tier, (_key, default) in _EXPECTED_JOBS.items():
        provider = build_tier_provider(persona_dir, tier)
        assert isinstance(provider, ClaudeCliProvider)
        provider.generate("hi")
        assert capture_claude_argv.last_model == default


def test_build_tier_provider_defaults_the_kind_when_there_is_no_persona_config(tmp_path) -> None:
    persona_dir = tmp_path / "no_config_persona"
    persona_dir.mkdir()
    assert isinstance(build_tier_provider(persona_dir, TIER_BACKGROUND_HOUSEKEEPING), ClaudeCliProvider)


def test_build_tier_provider_reads_the_kind_strictly(tmp_path, monkeypatch) -> None:
    """A persona's provider KIND is never guessed from an unreadable file."""
    persona_dir = _claude_cli_persona(tmp_path)
    from brain import persona_config

    def boom(path):
        raise OSError("transient")

    monkeypatch.setattr(persona_config.PersonaConfig, "load", classmethod(lambda cls, path: boom(path)))
    with pytest.raises(OSError):
        build_tier_provider(persona_dir, TIER_BACKGROUND_CLASSIFIER)


def test_a_non_claude_provider_ignores_the_tier(tmp_path) -> None:
    """C35: get_provider("ollama" | "fake", tier=...) does not look the tier up (the Claude jobs' ids mean nothing there)."""
    from brain.bridge.provider import FakeProvider, OllamaProvider, get_provider

    assert isinstance(get_provider("fake", tier=TIER_BACKGROUND_GENERATIVE), FakeProvider)
    assert isinstance(get_provider("ollama", tier=TIER_BACKGROUND_GENERATIVE), OllamaProvider)
    with pytest.raises(ValueError):
        get_provider("nope")
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    (persona_dir / "persona_config.json").write_text('{"provider": "ollama"}')
    assert isinstance(build_tier_provider(persona_dir, TIER_INTERACTIVE_CHAT), OllamaProvider)


def test_model_label_prefers_the_providers_live_id(tmp_path, model_override) -> None:
    persona_dir = _claude_cli_persona(tmp_path)
    provider = build_tier_provider(persona_dir, TIER_BACKGROUND_CLASSIFIER)
    assert model_label_for_provider(provider, TIER_BACKGROUND_CLASSIFIER) == "haiku"
    model_override(TIER_BACKGROUND_CLASSIFIER, "claude-haiku-4-5")
    assert model_label_for_provider(provider, TIER_BACKGROUND_CLASSIFIER) == "claude-haiku-4-5"  # live, not construction time


def test_model_label_falls_back_for_a_provider_with_no_model_attr() -> None:
    class _NoModelAttrProvider:
        pass

    assert model_label_for_provider(_NoModelAttrProvider(), TIER_BACKGROUND_GENERATIVE) == "claude-sonnet-5"


def test_configured_job_models_lists_the_seven_jobs(model_override) -> None:
    assert configured_job_models() == {tier: default for tier, (_k, default) in _EXPECTED_JOBS.items()}
    model_override(TIER_COMPACTION, "claude-haiku-4-5")
    assert configured_job_models()[TIER_COMPACTION] == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# C39 / G5: writers know no key names
# ---------------------------------------------------------------------------


def test_only_set_job_models_writes_model_keys() -> None:
    """C39: the endpoint, the CLI and the migration write through set_job_models; no key literal exists elsewhere."""
    import ast

    offenders = []
    for full, text in brain_sources().items():
        rel = full.removeprefix("brain/")
        if rel in ("tunables.py", "bridge/model_tier.py") or "set_values(" not in text:
            continue
        # tunables_migration.py folds an old file of arbitrary keys and names none of them (key-agnostic): its ONE function
        # `_fold` is the other caller (the guard's G5 allows exactly that function, not the file).
        callers = {
            fn.name
            for fn in ast.walk(ast.parse(text))
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            for call in ast.walk(fn)
            if isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "set_values"
        }
        if not (rel == "tunables_migration.py" and callers == {"_fold"}):
            offenders.append(rel)
    assert offenders == [], f"set_values called outside tunables.py / model_tier.py / tunables_migration.py::_fold: {offenders}"


# ---------------------------------------------------------------------------
# C6: the model grammar is the same in Python and TypeScript
# ---------------------------------------------------------------------------

_FULL_IDS = [
    "claude-sonnet-5",
    "claude-cli",
    "claude-opus-4-8",
    "claude-3-5-haiku-20241022",
    "claude-sonnet-4.5",
    "claude-" + "a" * 57,  # exactly 64 characters: the longest accepted id
]
_BAD_IDS = [
    "gpt4",
    "Claude-Sonnet-5",
    "claude-",
    "claude-sonnet-5\n",
    " claude-sonnet-5",
    "claude-sonnet-5 ",
    "claude_sonnet_5",
    "claude-sonnet 5",
    "claude-sonnet-5;ls",
    "claude-Sonnet",
    "sonnet[1m]",
    "opusplan",
    "claude-sonnet-5[1m]",
    "claude-" + "a" * 58,  # 65 characters: one over the cap
    "claude-" + "a" * 5000,
    "",
]


def _ts_array(source: str, name: str) -> list[str]:
    """The string items of ``const <name> = [ ... ];`` in a TS test file (literals and ``"x" + "y".repeat(n)``)."""
    import json

    body = re.search(rf"const {name} = \[(.*?)\n\];", source, re.S)
    assert body, f"{name} not found"
    items: list[str] = []
    for line in body.group(1).splitlines():
        m = re.match(r'\s*"((?:[^"\\]|\\.)*)"(?:\s*\+\s*"((?:[^"\\]|\\.)*)"\.repeat\((\d+)\))?,', line)
        if m:
            text = json.loads(f'"{m.group(1)}"')
            if m.group(2) is not None:
                text += json.loads(f'"{m.group(2)}"') * int(m.group(3))
            items.append(text)
    return items


def test_the_grammar_accepts_and_rejects_the_shared_table() -> None:
    from brain.bridge.model_tier import is_valid_model

    for word in ("sonnet", "opus", "haiku"):
        assert is_valid_model(word)
    for good in _FULL_IDS:
        assert is_valid_model(good), good
    for bad in _BAD_IDS:
        assert not is_valid_model(bad), bad
    assert not is_valid_model(None) and not is_valid_model(5) and not is_valid_model(["sonnet"])


def test_the_typescript_test_holds_the_same_table() -> None:
    """Change one side's patterns or table and the other side's test file stops matching: this fails (the TS test
    asserts the same strings with the same grammar)."""
    source = (_REPO_ROOT / "app" / "src" / "lib" / "modelId.test.ts").read_text(encoding="utf-8")
    assert _ts_array(source, "FULL_IDS") == _FULL_IDS
    assert _ts_array(source, "BAD_IDS") == _BAD_IDS
    ts = (_REPO_ROOT / "app" / "src" / "lib" / "modelId.ts").read_text(encoding="utf-8")
    assert model_tier.MODEL_ID_PATTERN.pattern in ts  # claude-[a-z0-9.-]+ is the pattern in both
    assert f"MODEL_ID_MAX_LENGTH = {model_tier.MODEL_ID_MAX_LENGTH}" in ts


def test_the_unsupported_display_is_cut_and_cleaned() -> None:
    from brain.bridge.model_tier import MODEL_ID_MAX_LENGTH, unsupported_display

    assert unsupported_display("x" * 200) == "x" * MODEL_ID_MAX_LENGTH + "…"
    assert unsupported_display("a\x1b[31m\nb") == "a[31mb"
    assert unsupported_display("​\x07") == "(non-printing characters)"
    assert unsupported_display("") == "(empty)"  # an empty text is not "non-printing characters"

"""The job table and the picker vocabulary of brain/bridge/model_tier.py (#529: C1, C2, C4, C34, C35, C39, C41).

One tunables line per Claude job decides its model; ``model_for_tier`` reads it on every call. The behaviours that need
a spawned ``claude`` (argv per job, live edits, fallback after a rejection) are in test_model_tier_resolution.py; the
guard that nothing outside this module chooses a model is in test_model_guard.py.
"""

from __future__ import annotations

import io
import re
import tokenize
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

_REPO_ROOT = Path(__file__).resolve().parents[4]

_EXPECTED_JOBS = {
    TIER_INTERACTIVE_CHAT: ("model.interactive_chat", "claude-sonnet-5"),
    TIER_BACKGROUND_GENERATIVE: ("model.background_generative", "claude-sonnet-5"),
    TIER_BACKGROUND_CLASSIFIER: ("model.background_classifier", "haiku"),
    TIER_BACKGROUND_HOUSEKEEPING: ("model.background_housekeeping", "haiku"),
    TIER_COMPACTION: ("model.compaction", "haiku"),
    TIER_SELF_MODEL_ARTICULATE: ("model.self_model_articulate", "haiku"),
    TIER_ATTUNEMENT_DETECTOR: ("model.attunement_detector", "claude-haiku-4-5-20251001"),
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
    """The registry after importing model_tier holds exactly these seven model keys with these defaults."""
    registered = {k: v for k, v in tunables._registry.items() if k.startswith("model.")}
    assert registered == dict(_EXPECTED_JOBS.values())
    assert set(CLAUDE_TIERS) == set(_EXPECTED_JOBS)
    for tier, (_key, default) in _EXPECTED_JOBS.items():
        assert shipped_default(tier) == default
        assert model_for_tier(tier) == default


def test_every_job_line_carries_a_comment_saying_what_the_job_is() -> None:
    """C1: each `tunables.register("model.<job>", ...)` line has a `#` comment of real length (the reviewer judges the
    wording against the table in the spec, C1b); and OPUS_CHOICE is the one other model literal, also commented."""
    source = (_REPO_ROOT / "brain" / "bridge" / "model_tier.py").read_text(encoding="utf-8")
    seen: dict[str, str] = {}
    for line in source.splitlines():
        m = re.search(r'tunables\.register\("(model\.[a-z_]+)"', line)
        if m:
            comments = [t.string for t in tokenize.generate_tokens(io.StringIO(line + "\n").readline) if t.type == tokenize.COMMENT]
            seen[m.group(1)] = comments[0] if comments else ""
    assert set(seen) == {key for key, _ in _EXPECTED_JOBS.values()}
    for key, comment in seen.items():
        assert len(comment.split()) >= 6, f"{key}: the job's line needs a comment saying what the job does, got {comment!r}"
    opus_lines = [ln for ln in source.splitlines() if ln.startswith("OPUS_CHOICE")]
    assert len(opus_lines) == 1 and "#" in source[source.index(opus_lines[0]) - 400 : source.index(opus_lines[0])]
    assert OPUS_CHOICE == "claude-opus-5"


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
    for root, suffixes in (
        (_REPO_ROOT / "brain", (".py",)),
        (_REPO_ROOT / "app" / "src", (".ts", ".tsx")),
        (_REPO_ROOT / "app" / "src-tauri" / "src", (".rs",)),
        (_REPO_ROOT / "scripts", (".py", ".sh")),
    ):
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in suffixes and "node_modules" not in path.parts:
                yield path


# Tests that name removed identifiers ON PURPOSE (to assert they are gone, or to feed the guard a violating fixture).
_TESTS_NAMING_REMOVED_NAMES = (
    "tests/unit/brain/bridge/test_model_tier.py",
    "tests/unit/brain/bridge/test_model_tier_resolution.py",
    "tests/unit/brain/bridge/test_model_guard.py",
)
# In tests, `model_override` is the live conftest fixture's name; only the removed KEYWORD form is a leftover.
_REMOVED_KWARG = re.compile(r"(?<![\w])model_override\s*=(?!=)")


def _test_files():
    for path in sorted((_REPO_ROOT / "tests").rglob("*.py")):
        if path.relative_to(_REPO_ROOT).as_posix() not in _TESTS_NAMING_REMOVED_NAMES:
            yield path


def test_removed_names_exist_nowhere_in_code_or_comments() -> None:
    """C2 and C20: every removed name has zero hits, code AND comments/docstrings, in brain, the app, scripts and tests."""
    names = [n for n in _REMOVED_EVERYWHERE if n != "model_override"]
    pattern = re.compile(r"(?<![\w])(" + "|".join(map(re.escape, _REMOVED_EVERYWHERE)) + r")(?![\w])")
    test_pattern = re.compile(r"(?<![\w])(" + "|".join(map(re.escape, names)) + r")(?![\w])")
    legacy = re.compile(r"(?<![\w])(" + "|".join(_LEGACY_KEYS) + r")(?![\w])")
    hits: list[str] = []
    for path in _source_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
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


def test_removed_attributes_do_not_exist() -> None:
    from brain import persona_config
    from brain.bridge import persona_state, provider
    from brain.chat import compaction
    from brain.self_model import articulate

    for mod, names in (
        (model_tier, ("MODEL_LITTLE", "MODEL_MEDIUM", "MODEL_BIGGEST", "TIER_MODEL", "TIER_DEV_CLI", "NAMED_CHOICES")),
        (persona_config, ("DEFAULT_MODEL", "ROUTED_TIER_NAMES", "is_valid_model", "KNOWN_MODELS")),
        (compaction, ("COMPACTION_MODEL",)),
        (articulate, ("SELF_MODEL_MODEL",)),
        (persona_state, ("_default_model_for",)),
        (provider, ("DEFAULT_MODEL",)),
    ):
        for name in names:
            assert not hasattr(mod, name), f"{mod.__name__}.{name} should not exist"


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


def test_the_detector_keeps_its_pinned_snapshot_not_an_alias() -> None:
    assert model_for_tier(TIER_ATTUNEMENT_DETECTOR) == "claude-haiku-4-5-20251001"


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


def test_per_job_equality_drives_the_removal_rule() -> None:
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
    """A word that means the job's default removes the line; opus and a full id write; haiku writes only where it is
    not the default; a full id is written even when it equals the default."""
    chat_key = "model.interactive_chat"
    assert set_job_models({TIER_INTERACTIVE_CHAT: "sonnet"}) == {}  # nothing to remove, nothing written
    assert tunables.get_override(chat_key) is None
    assert set_job_models({TIER_INTERACTIVE_CHAT: "opus"}) == {TIER_INTERACTIVE_CHAT: OPUS_CHOICE}
    assert tunables.get_override(chat_key) == OPUS_CHOICE
    assert set_job_models({TIER_INTERACTIVE_CHAT: "sonnet"}) == {TIER_INTERACTIVE_CHAT: None}  # removal
    assert tunables.get_override(chat_key) is None
    assert set_job_models({TIER_INTERACTIVE_CHAT: "claude-sonnet-5"}) == {TIER_INTERACTIVE_CHAT: "claude-sonnet-5"}
    assert tunables.get_override(chat_key) == "claude-sonnet-5"  # a full id is written verbatim, default-equal or not
    assert set_job_models({TIER_INTERACTIVE_CHAT: "haiku"}) == {TIER_INTERACTIVE_CHAT: "haiku"}
    assert set_job_models({TIER_BACKGROUND_CLASSIFIER: "haiku"}) == {}  # haiku is that job's default: no line
    assert set_job_models({TIER_BACKGROUND_CLASSIFIER: "sonnet"}) == {TIER_BACKGROUND_CLASSIFIER: "claude-sonnet-5"}
    assert set_job_models({TIER_BACKGROUND_GENERATIVE: "sonnet"}) == {}  # equals that job's default: removal of nothing
    assert set_job_models({TIER_INTERACTIVE_CHAT: None}) == {TIER_INTERACTIVE_CHAT: None}


def test_set_job_models_rejects_an_unusable_value_and_an_unknown_job() -> None:
    with pytest.raises(ValueError):
        set_job_models({TIER_INTERACTIVE_CHAT: "banana"})
    with pytest.raises(KeyError):
        set_job_models({"not-a-job": "haiku"})
    assert tunables.get_override("model.interactive_chat") is None


# ---------------------------------------------------------------------------
# C41: the point of the removal rule, end to end
# ---------------------------------------------------------------------------


def test_a_user_who_picked_the_word_follows_a_default_bump_and_one_who_typed_the_id_does_not(
    monkeypatch, capture_claude_argv
) -> None:
    key, _ = model_tier._CLAUDE_JOBS[TIER_INTERACTIVE_CHAT]
    # user A picked the word, user B typed the old id verbatim
    set_job_models({TIER_INTERACTIVE_CHAT: "sonnet"})
    assert tunables.get_override(key) is None
    monkeypatch.setitem(model_tier._CLAUDE_JOBS, TIER_INTERACTIVE_CHAT, (key, "claude-sonnet-6"))
    ClaudeCliProvider(TIER_INTERACTIVE_CHAT).generate("hi")
    assert capture_claude_argv.last_model == "claude-sonnet-6"  # A follows the bump with no other change
    monkeypatch.setitem(model_tier._CLAUDE_JOBS, TIER_INTERACTIVE_CHAT, (key, "claude-sonnet-5"))  # the old pin again
    set_job_models({TIER_INTERACTIVE_CHAT: "claude-sonnet-5"})
    monkeypatch.setitem(model_tier._CLAUDE_JOBS, TIER_INTERACTIVE_CHAT, (key, "claude-sonnet-6"))
    ClaudeCliProvider(TIER_INTERACTIVE_CHAT).generate("hi")
    assert capture_claude_argv.last_model == "claude-sonnet-5"  # B keeps the id they typed


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
    offenders = []
    for path in sorted((_REPO_ROOT / "brain").rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT / "brain").as_posix()
        text = path.read_text(encoding="utf-8")
        if rel not in ("tunables.py", "bridge/model_tier.py") and "set_overrides(" in text:
            offenders.append(rel)
    assert offenders == [], f"set_overrides called outside tunables.py / model_tier.py: {offenders}"


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

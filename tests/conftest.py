"""Shared pytest fixtures and configuration for companion-emergence tests."""

from __future__ import annotations

import functools
import inspect
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

import pytest

from brain.bridge import cli_throttle, model_error, provider_auth
from brain.chat import pass2_queue
from brain.memory import pending

pytest_plugins = ["pytester"]


@functools.cache
def _symlinks_available() -> bool:
    """Probe once per session whether this host lets us create a symlink (#262).

    Non-elevated Windows without Developer Mode raises ``OSError: [WinError 1314]``; GitHub's
    ``windows-latest`` runner is elevated, so CI never sees it — a contributor's stock checkout does.
    """
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "target"
        target.mkdir()
        try:
            os.symlink(target, Path(tmp) / "link", target_is_directory=True)
        except (OSError, NotImplementedError):
            return False
    return True


@pytest.fixture
def requires_symlinks() -> None:
    """Skip (not fail) a test whose setup creates a symlink, where the host denies it (#262)."""
    if not _symlinks_available():
        pytest.skip("os.symlink needs an elevated shell or Developer Mode on Windows")


# The REAL per-user directories, captured at import (before any test patches the platform directories), for the #488 isolation test.
from brain import paths as _paths_module  # noqa: E402

# The XDG cache variable as the run started (None when unset), for the #488 isolation test: the fixture must leave it alone.
REAL_XDG_CACHE_HOME = os.environ.get("XDG_CACHE_HOME")

REAL_PLATFORM_DIRS = {
    name: Path(getattr(_paths_module._dirs, name)).resolve()
    for name in ("user_data_path", "user_cache_path", "user_log_path", "user_state_path", "user_config_path")
}


def _isolated_dir(root: Path, dirs: object, sub: str) -> str:
    """One directory under the per-test root, laid out like the real thing: per app name (the harness keeps a SIBLING app dir,
    "companion-emergence-harness", outside the guarded roots) and per app author where the platform puts it in the path (Windows,
    macOS: an explicit author string, or on Windows no author meaning the app name), so ``PlatformDirs(app)`` and the engine's
    ``PlatformDirs(app, author)`` stay different directories, which a harness guard test depends on."""
    appname = getattr(dirs, "appname", None) or "app"
    author = getattr(dirs, "appauthor", None)
    if not (isinstance(author, str) and author):
        author = appname if (author is None and os.name == "nt") else None
    return str(root / author / appname / sub) if author else str(root / appname / sub)


# The platformdirs directory properties as the platform defines them, captured before #488's fixture redirects them.
_REAL_DIR_PROPS = {
    prop: inspect.getattr_static(type(_paths_module._dirs), prop)
    for prop in ("user_data_dir", "user_log_dir", "user_state_dir", "user_config_dir")
}


@pytest.fixture
def real_platform_dirs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """For a test that compares the home computed from an environment with platformdirs' own answer (#456 S15, the
    pre-PR bridge proof): put the platform's real properties back over #488's redirect. For comparing paths ONLY.

    On Linux the XDG variables #488 set still point into the test's tmp root; on macOS and Windows the restored
    properties resolve to the REAL user directories, so the fixture fails the test if any exposed directory was
    created, or gained or lost a top-level entry, during the test (tests/real_dirs_guard.py; names only, nothing read;
    a developer's running bridge writing to its existing files there does not trip it)."""
    from platformdirs import PlatformDirs

    from brain import paths
    from tests.real_dirs_guard import assert_untouched, fingerprint

    for cls in {type(paths._dirs), PlatformDirs}:
        for prop, value in _REAL_DIR_PROPS.items():
            monkeypatch.setattr(cls, prop, value)
    exposed = [Path(getattr(paths._dirs, prop)) for prop in _REAL_DIR_PROPS]
    before = fingerprint(exposed)
    yield
    assert_untouched(before, fingerprint(exposed))


@pytest.fixture(autouse=True)
def _isolate_user_data_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """#488: no test may write into the real per-user data directory.

    A test that builds the bridge app (its lifespan syncs the settings file) or calls ``get_home()`` with neither
    ``KINDLED_HOME`` nor ``NELLBRAIN_HOME`` set resolved to the platformdirs default, the REAL ``~/.local/share/companion-emergence``
    of whoever ran the tests, so a run could change real settings and a later test could read what an earlier one left. Every test
    now gets its own empty root as that default. The directory properties are replaced on the platform's ``PlatformDirs`` class
    (every OS), so ``brain.paths._dirs`` and any fresh ``PlatformDirs(...)`` (the update overlay hook builds its own, and a parity
    test holds the two together) agree and both land in the tmp root; the XDG data and state variables point there too (so a
    subprocess a test starts, on Linux and macOS, lands there as well). A test that sets ``KINDLED_HOME`` / ``NELLBRAIN_HOME`` still wins, as before, and the env-var resolution order is
    unchanged.

    The CACHE directory and ``XDG_CACHE_HOME`` are deliberately NOT redirected: the cache is where downloaded models live, not user
    state. platformdirs reads ``XDG_CACHE_HOME`` on Linux and macOS and huggingface_hub derives ``HF_HOME`` from it on every OS, so
    moving it made a subprocess a test starts (the judge release RSS job's C3a) find no cached judge, go to the network and fail.

    The patches are held by this fixture's OWN ``MonkeyPatch``, not the test's: a test that calls ``monkeypatch.undo()`` reverts
    only what it patched itself. With the test's instance, an ``undo()`` put the platform default back and the next
    ``set_job_models`` wrote a model line into the real install's settings file (#530).
    """
    from platformdirs import PlatformDirs

    from brain import paths

    root = tmp_path_factory.mktemp("userdata")
    monkeypatch = pytest.MonkeyPatch()
    # Two classes when something (the update overlay hook drops platformdirs from sys.modules) made a new one after brain.paths
    # built its instance: patch the one `brain.paths._dirs` is an instance of AND the one `from platformdirs import PlatformDirs` gives now.
    for cls in {type(paths._dirs), PlatformDirs}:
        for prop, sub in (
            ("user_data_dir", "data"),
            ("user_log_dir", "log"),
            ("user_state_dir", "state"),
            ("user_config_dir", "config"),
        ):
            monkeypatch.setattr(cls, prop, property(lambda self, _s=sub: _isolated_dir(root, self, _s)))
    for var, sub in (
        ("XDG_DATA_HOME", "xdg-data"),
        ("XDG_STATE_HOME", "xdg-state"),
    ):
        monkeypatch.setenv(var, str(root / sub))
    try:
        yield
    finally:
        monkeypatch.undo()


@pytest.fixture(autouse=True)
def _reset_cli_throttle() -> Iterator[None]:
    """Reset cli_throttle global state before each test.

    background_slot() reads process-global monotonic timestamps. Without
    this reset, a test that calls mark_interactive_active() contaminates
    subsequent tests in the same process — causing background-engine calls
    to be gated when the test expects them to fire.
    """
    cli_throttle.reset()
    provider_auth.reset()  # #246: the auth-expiry hooks live inside the CLI detail helpers
    model_error.reset()  # #440: process-wide rejected-model records must not leak between tests
    # #456: a bridge app's stop seals the process-wide task registry; a later test in this process starts fresh.
    from brain.bridge import shutdown_progress

    shutdown_progress.reset_for_tests()
    yield
    cli_throttle.reset()
    provider_auth.reset()
    model_error.reset()


@pytest.fixture(autouse=True)
def _no_cwd_queue_leak() -> Iterator[None]:
    """Error a test that grows ``pending_candidates.jsonl`` in the cwd it started in (#375).

    An in-memory MemoryStore used to report the working directory as its persona dir, so 17
    tests appended gated candidates to the repo root on every run, invisibly (it is gitignored)
    and with later tests able to read the debris back. The path is fixed at setup, so a test
    that chdirs elsewhere still has its original cwd checked; a file that already exists is
    fine as long as it does not grow.
    """
    queue = Path.cwd() / pending._QUEUE_FILENAME

    def _size() -> int | None:
        return queue.stat().st_size if queue.exists() else None

    before = _size()
    yield
    after = _size()
    assert after == before, (
        f"this test wrote to {queue} ({before} -> {after} bytes): a store with no persona "
        "directory must not enqueue; give it a file-backed store (tmp_path / 'memories.db')"
    )


@pytest.fixture(autouse=True)
def _pin_cli_tools_flag() -> Iterator[None]:
    """Pin "this claude CLI supports --tools" to True for every test (#329).

    Otherwise the first argv build in a test would spawn a real `claude --help`
    probe, or consume a test's own `subprocess.run` mock. Assigned directly,
    not via monkeypatch, so a test's `monkeypatch.undo()` can't put it back to
    the unknown state mid-test. Probe tests reset it themselves.
    """
    from brain.bridge import provider

    def _pin() -> None:
        provider._TOOLS_FLAG_SUPPORTED = True
        provider._tools_probe_retry_at = 0.0
        provider._tools_warned = False
        provider._tools_window_warned_at = None

    _pin()
    yield
    _pin()


@pytest.fixture(autouse=True)
def _inhibit_bridge_background_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep build_app's lifespan from starting the supervisor + migration threads.

    Every bridge endpoint test enters build_app's lifespan; the real supervisor
    thread it started ran a startup catch-up compaction that rolled over the
    months-old session the test had just seeded and deleted its buffer, racing
    the test's own request (hunts/bridge-order-pollution-flakes/diagnosis.md —
    #155, #161, and the c8 / WinError-32 CI reds). Mirrors _reset_pass2_queue:
    tests that genuinely exercise those threads pass background_threads=True.
    """
    from brain.bridge import server

    # raising=False: a bisect onto a pre-flag commit must not error every test at setup.
    monkeypatch.setattr(server, "_background_threads_inhibited", True, raising=False)


@pytest.fixture(autouse=True)
def _no_real_service_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """#456 (D20): `supervisor start` and the kill helper ask whether the persona's OS service is installed, which reads
    the real user's unit / plist / task. No test may route a start or a kill through a real service manager: every test
    sees "not installed" unless it patches this itself (the service tests do, with spies)."""
    monkeypatch.setattr("brain.bridge.kill.service_installed", lambda persona, platform=None: False, raising=False)


@pytest.fixture(autouse=True)
def _reset_pass2_queue() -> Iterator[None]:
    """Reset pass2_queue global state before and after each test.

    Mirrors _reset_cli_throttle.  Without this, test-only side effects
    registered for queued items would leak into subsequent tests in the same
    process. (There is no worker thread to inhibit any more: ram-spike-fix
    INC-9 moved the bridge-side drain into the supervisor's central cadence
    function; tests drive drain_pending()/drain_all_locked() directly.)
    """
    pass2_queue.reset()
    yield
    pass2_queue.reset()


@pytest.fixture(autouse=True)
def _fake_embedding_provider_by_default(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force brain.memory.embeddings.build_embedding_provider() to the
    deterministic, offline FakeEmbeddingProvider for the whole suite by
    default.

    build_embedding_provider() is the PRODUCTION default (FastEmbedProvider —
    a real local ONNX model, downloaded once over the network into a shared
    cache dir). Every production call site that embeds anything (recall's
    query embed, dedupe's candidate embed, embed-on-write) goes through this
    one function (Stage 1 of the local semantic-retrieval build; the old
    build_embedding_cache()/EmbeddingCache layer in front of it is gone as
    of F1 #259 increment 8), so ANY test that exercises those
    code paths — even indirectly, via a background thread the test itself
    never awaits — would otherwise attempt a real model download: slow,
    network-dependent, and (seen while landing this fixture) capable of
    retrying for minutes past the test's own teardown in a now-deleted tmp
    dir. A test that genuinely needs the real provider opts out with
    `@pytest.mark.requires_network`.
    """
    if "requires_network" in request.keywords:
        return
    from brain.memory import embeddings

    monkeypatch.setattr(
        embeddings, "build_embedding_provider", lambda: embeddings.FakeEmbeddingProvider(dim=256)
    )


@pytest.fixture(autouse=True)
def _reset_embedding_provider_cache() -> Iterator[None]:
    """Reset embeddings.build_embedding_provider()'s process-level provider
    cache before and after each test.

    Most tests never touch this cache at all — the fake-provider override
    above replaces build_embedding_provider() wholesale (cache included), so
    the fake path never reads or writes it. But a couple of tests in
    test_embeddings.py import `build_embedding_provider` by NAME and call the
    real function directly to exercise its own model_tier wiring, which
    bypasses that monkeypatch entirely (the import binds the original
    function object before any fixture runs). Without this reset, whichever
    such test ran first would cache a provider that a later one — expecting
    to build its own, under its own patched tmp_path / stubbed TextEmbedding
    — would get served back instead.
    """
    from brain.memory import embeddings

    embeddings._reset_embedding_provider_cache()
    yield
    embeddings._reset_embedding_provider_cache()


@pytest.fixture(autouse=True)
def _reset_embedding_matrix_cache() -> Iterator[None]:
    """Reset embedding_matrix.build_embedding_matrix()'s process-level matrix
    cache before and after each test — mirrors `_reset_embedding_provider_cache`
    above for the same reason: the cache is process-global and keyed by
    `str(db_path)`, so without a reset a matrix built (and possibly warmed)
    by one test against a given path could leak into a later test that
    happens to reuse that path, or hold a stale reference across tests that
    each expect a fresh singleton for their own tmp_path db."""
    from brain.memory import embedding_matrix

    embedding_matrix._reset_embedding_matrix_cache()
    yield
    embedding_matrix._reset_embedding_matrix_cache()


@pytest.fixture(autouse=True)
def _reset_embedding_backfill_batch_cache() -> Iterator[None]:
    """Reset embedding_backfill's measured/derived batch-size cache before
    and after each test — mirrors `_reset_embedding_provider_cache` above:
    the cache is process-global and keyed by model_id, measured once per
    process (F1 #259 increment 3), so without a reset a batch size measured
    (and possibly deliberately controlled/monkeypatched) by one test could
    leak into a later test expecting its own fresh measurement."""
    from brain.memory import embedding_backfill

    embedding_backfill._reset_batch_size_cache()
    yield
    embedding_backfill._reset_batch_size_cache()


@pytest.fixture(autouse=True)
def _fake_reranker_provider_by_default(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force brain.memory.reranker.build_reranker_provider() to the
    deterministic, offline FakeRerankerProvider for the whole suite by
    default (#231 offline-test discipline, mirrors
    `_fake_embedding_provider_by_default` above).

    build_reranker_provider() is the PRODUCTION default (CrossEncoderProvider
    — a real local ONNX cross-encoder model, downloaded once over the
    network into a shared cache dir). Any test that exercises
    run_semantic_recall / search_memories(mode="semantic") — even
    indirectly — would otherwise attempt a real model download. A test that
    genuinely needs the real provider opts out with
    `@pytest.mark.requires_network`.

    FakeRerankerProvider defaults every UNSCRIPTED document to a score far
    below any plausible calibrated floor (see that class's docstring) — so a
    test that never scripts reranker scores gets the same "semantic
    inconclusive -> lexical fallback" behavior it would have gotten from an
    empty/orthogonal cosine result pre-#231, rather than an arbitrary
    reranker score accidentally clearing the floor. Tests that need a
    CONCLUSIVE reranked result construct their own
    `FakeRerankerProvider(scores={...})` and monkeypatch this function
    directly, mirroring how `_ScriptedProvider` overrides the embedding
    fixture above for the same reason.

    F2b (#276 §2/§4): every rerank() call now also scores the fixed
    `reranker.ANCHOR_POOL` documents alongside the real candidates (once
    `width` is wide enough to reserve `k >= K_MIN` anchors — see
    `normalize_against_anchors`), and the floor gate compares the
    NORMALIZED score (`raw - median(anchor_scores)`), not the raw one. If
    the DEFAULT fake left `ANCHOR_POOL` unscripted too, it would collide
    with the SAME `_DEFAULT_UNSCORED` sentinel every other unscripted
    document already falls back to — collapsing `median(anchor_scores)` to
    exactly that sentinel, and so `normalized = _DEFAULT_UNSCORED -
    _DEFAULT_UNSCORED == 0.0` for EVERY unscripted real candidate, silently
    breaking the "unscripted -> far below floor" invariant the paragraph
    above promises (0.0 sits comfortably ABOVE a near-zero floor, not "far
    below" it). Explicitly scripting the anchors to a fixed neutral value
    (0.0) — while leaving every OTHER unscripted document at
    `_DEFAULT_UNSCORED` via `default=` — restores the invariant exactly:
    an unscripted real candidate normalizes to `_DEFAULT_UNSCORED - 0.0 ==
    _DEFAULT_UNSCORED`, unchanged from pre-F2b, comfortably below any
    plausible floor again. A test that scripts its OWN reranker scores
    (the common case, via its own `FakeRerankerProvider(scores={...})`)
    is unaffected — this only shapes the SUITE-WIDE DEFAULT a test gets
    when it never overrides `build_reranker_provider` at all.

    Also patches `reranker._bootstrap_reranker_provider` (F2a inc8, #250 §7
    UPDATED — the bootstrap-floor ruling) the same way: that function is the
    OTHER production entry point that constructs a real `CrossEncoderProvider`
    (deliberately independent of `build_reranker_provider` itself — see its
    own docstring on why), used by `floor_calibration.get_bootstrap_floor`
    whenever `store.get_reranker_floor` is asked about a model_id with no
    persisted row yet. Without this, ANY test whose store has no persisted
    floor row (the common case for a fresh in-memory/tmp_path store) would
    attempt a REAL fastembed model load the first time `get_reranker_floor`
    is called — even tests that never intentionally touch the reranker at
    all.

    UNLIKE `build_reranker_provider`'s fake, this one is scripted (not left
    fully unscripted): the bootstrap fits a THRESHOLD from whatever scores
    its provider returns for the bundled `_FP16_GATE_PAIRS[:6]` pairs, so an
    UNSCRIPTED default-everywhere provider would fit a floor of roughly
    `_DEFAULT_UNSCORED - 1.0` (every pair ties at the same sentinel score,
    and `fit_threshold_fbeta` picks the threshold just below it) — only
    ONE unit below `FakeRerankerProvider`'s own `_DEFAULT_UNSCORED`, not
    "far below" it. That would silently break the "unscripted reranker
    score never clears an unscripted floor -> INCONCLUSIVE" invariant every
    other test in this suite relies on for its OWN default (an unscripted
    -1000.0 candidate score would clear a -1001.0 bootstrap floor). Scripted
    here to a realistic, well-SEPARATED pair of scores instead (relevant
    pairs high, irrelevant pairs low) so the default bootstrap floor lands
    near 0.0 — comfortably above `_DEFAULT_UNSCORED`, restoring that
    invariant for every test that never scripts its OWN bootstrap provider.
    """
    if "requires_network" in request.keywords:
        return
    from brain.memory import reranker

    # Both consuming call sites (brain.memory.semantic_recall,
    # brain.tools.impls.search_memories) import the module itself
    # (`from brain.memory import reranker as reranker_mod`) and call
    # `reranker_mod.build_reranker_provider()` — a dynamic attribute lookup
    # at call time, exactly like every embedding-provider call site's
    # identical dynamic lookup on `embeddings.build_embedding_provider` — so
    # patching this ONE module attribute is sufficient to intercept every
    # call site.
    # F2b (#276 §2/§4): script ANCHOR_POOL to a fixed neutral value (0.0) so
    # the default fake's anchor median never collapses onto the SAME
    # `_DEFAULT_UNSCORED` sentinel an unscripted real candidate also falls
    # back to — see this fixture's own docstring for why that collision
    # would otherwise zero out the normalization offset for every test that
    # never scripts its own reranker.
    default_anchor_scores = dict.fromkeys(reranker.ANCHOR_POOL, 0.0)
    monkeypatch.setattr(
        reranker,
        "build_reranker_provider",
        lambda *, store=None: reranker.FakeRerankerProvider(scores=default_anchor_scores),
    )
    default_bootstrap_scores = {
        doc: (5.0 if i < 3 else -5.0) for i, (_query, doc) in enumerate(reranker._FP16_GATE_PAIRS[:6])
    }
    monkeypatch.setattr(
        reranker,
        "_bootstrap_reranker_provider",
        lambda model_id: reranker.FakeRerankerProvider(scores=default_bootstrap_scores),
    )


@pytest.fixture(autouse=True)
def _reset_reranker_provider_cache() -> Iterator[None]:
    """Reset reranker.build_reranker_provider()'s process-level provider
    cache, the per-message width fit's rerank cost model (running sums and
    warm-up state per reranker model id; name-recall fix R1, plan P-24 —
    replaces the removed hourly warm-latency / warm-memory caches), and
    floor_calibration's bootstrap-floor cache (F2a inc8, #250 §7 UPDATED)
    before and after each test — mirrors `_reset_embedding_provider_cache`
    above for the same reason (a test that calls the REAL `build_reranker_
    provider()` / `get_reranker_floor()` directly must not read or leak a
    provider/bootstrap/measurement a prior/later test's call happened to
    cache)."""
    from brain.memory import floor_calibration, reranker

    reranker._reset_reranker_provider_cache()
    reranker._reset_rerank_cost_model()
    floor_calibration._reset_bootstrap_floor_cache()
    floor_calibration._reset_cosine_bootstrap_floor_cache()
    yield
    reranker._reset_reranker_provider_cache()
    reranker._reset_rerank_cost_model()
    floor_calibration._reset_bootstrap_floor_cache()
    floor_calibration._reset_cosine_bootstrap_floor_cache()


@pytest.fixture(autouse=True)
def _paragraph_time_model_reset(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Name-recall fix R6 (plan P-24): reset the per-message budget's running
    averages (`T_m`, `T_p`) before and after each test, and freeze the budget
    clock (`semantic_recall._clock`) at 0 so every test sees `T_m = T_p = 0`
    and the whole rerank budget, exactly as before R6, unless it scripts the
    clock itself (the per-paragraph budget tests do)."""
    from brain.memory import semantic_recall

    semantic_recall._reset_paragraph_time_model()
    monkeypatch.setattr(semantic_recall, "_clock", lambda: 0.0)
    yield
    semantic_recall._reset_paragraph_time_model()


@pytest.fixture(autouse=True)
def _keyword_budget_reset(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Long-paste increment (S103, plan P-3): reset the keyword extra-words cost
    series and re-arm its first-sample discard before and after each test, and
    freeze the budget clock (`keyword_budget._clock`) at 0, so an unscripted
    test sees `left` = the whole budget and a seed rate of 0 (every extra word
    fits), exactly as before S103, unless it scripts the clock itself."""
    from brain.memory import keyword_budget

    keyword_budget._reset_keyword_cost()
    monkeypatch.setattr(keyword_budget, "_clock", lambda: 0.0)
    yield
    keyword_budget._reset_keyword_cost()


@pytest.fixture(autouse=True)
def _tunables_values_cache_reset() -> Iterator[None]:
    """#462: forget the cached tunables values before and after each test.

    `tunables._load_values_locked` caches the parsed values (a content failure too) keyed on the file's stat signature.
    Only the cache is cleared: the registry that modules fill at import time is left alone.
    """
    from brain import tunables

    tunables._reset_values_cache()
    yield
    tunables._reset_values_cache()


_REAL_TUNABLE_KEYS: frozenset[str] | None = None


def _real_tunable_keys() -> frozenset[str]:
    """Every literal key a ``tunables.register`` call in brain/ declares (an AST scan, once per session)."""
    global _REAL_TUNABLE_KEYS
    if _REAL_TUNABLE_KEYS is None:
        from tests.tunables_file import registered_literal_keys

        keys = registered_literal_keys(Path(__file__).resolve().parents[1])
        _REAL_TUNABLE_KEYS = frozenset(keys)
    return _REAL_TUNABLE_KEYS


@pytest.fixture(autouse=True)
def _tunables_registry_restore() -> Iterator[None]:
    """A test may clear the tunables registry (``_reset_for_tests``) or register throwaway keys. The registry that modules fill
    at import time must survive for the tests after it, so restore it: the state before the test, plus any real brain key the
    test's own imports registered (a module imported for the first time inside a test stays imported)."""
    from brain import tunables

    with tunables._lock:
        saved = (dict(tunables._registry), dict(tunables._descriptions))
    yield
    real = _real_tunable_keys()
    with tunables._lock:
        kept = {k: v for k, v in tunables._registry.items() if k in real}
        kept_desc = {k: v for k, v in tunables._descriptions.items() if k in real}
        tunables._registry.clear()
        tunables._registry.update(saved[0])
        tunables._registry.update(kept)
        tunables._descriptions.clear()
        tunables._descriptions.update(saved[1])
        tunables._descriptions.update(kept_desc)


@pytest.fixture(autouse=True)
def _cost_series_state_reset(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Cost-series increment (S146, S159, S163, S164): empty the in-process
    sample buffer and the warn-once flags before and after each test, and give
    every test a fixed machine id so no test writes `machine_id` into a real
    home directory. The `machine_id` tests clear `_machine_override` themselves
    and point `KINDLED_HOME` at a tmp path."""
    from brain.memory import cost_series

    cost_series._reset_pending()
    cost_series._reset_machine_id()
    monkeypatch.setattr(cost_series, "_machine_override", "test-machine")
    yield
    cost_series._reset_pending()
    cost_series._reset_machine_id()


@pytest.fixture(autouse=True)
def _floor_bootstrap_background_off_by_default(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Keep the floor bootstraps' background starters (name-recall fix S91/S92:
    the process-start thread `start_background`, the first-need rerank request
    and the on-each-message retry) from spawning threads in unrelated tests: a
    recall without a rerank floor or a `respond()` after a failed bootstrap
    would otherwise leak a daemon thread that computes (and caches) a floor
    after this test's teardown reset. The bookkeeping (need flag, failed
    record) still happens; tests of the starters themselves flip
    `floor_startup._background_inhibited` back. Also clears the in-flight slots
    before and after each test."""
    from brain.memory import floor_startup

    monkeypatch.setattr(floor_startup, "_background_inhibited", True)
    monkeypatch.setattr(floor_startup, "_marker_provider", floor_startup._no_marker)  # noqa: SLF001
    monkeypatch.setattr(floor_startup, "_ram_figure_persona_dir", None)  # noqa: SLF001
    monkeypatch.setattr(floor_startup, "_ram_figure_requested", False)  # noqa: SLF001
    floor_startup._inflight.clear()  # noqa: SLF001
    yield
    floor_startup._inflight.clear()  # noqa: SLF001


@pytest.fixture(autouse=True)
def _ram_figure_not_required_by_default(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """S200: a recall on a machine with no usable memory model takes the cosine path
    until the lull job has measured one. The suite's recall tests build reranks from
    fakes with no seeded figure, so the requirement is off by default; the S200
    tests set `reranker._require_ram_figure` True themselves. Also clears the
    need flags and attempt stamps before and after each test."""
    from brain.memory import peak_rss, reranker

    monkeypatch.setattr(reranker, "_require_ram_figure", False)
    # S253: the free-memory check before the model load reads the host's memory and the model cache:
    # off by default so no test depends on either; its own tests restore the real function.
    monkeypatch.setattr(reranker, "model_load_blocked_by_memory", lambda: False)
    # S255: the memory fit's safety factor is 1.0 in the suite so the many tests that pin exact
    # arithmetic (a headroom equal to a prediction) keep their meaning; the tests of the factor itself
    # (test_ram_fit.py, test_ram_ladder.py, test_not_loaded_batch_check.py) set the shipped value explicitly.
    from brain import dev_constants

    monkeypatch.setattr(dev_constants, "RAM_FIT_SAFETY_FACTOR", 1.0)
    # S220: the per-call peak reader is process-wide (on Linux it resets the kernel's peak
    # counter; elsewhere, S248, it samples the RSS in a thread): off by default so the suite's
    # fake rerankers neither touch /proc nor start a sampler nor record memory samples; the
    # reader's own tests and the S200/S220 gate tests turn it on.
    monkeypatch.setattr(peak_rss, "_disabled", True)
    peak_rss._reset_for_tests()  # noqa: SLF001
    reranker._ram_figure_needed.clear()  # noqa: SLF001
    reranker._ram_figure_attempt.clear()  # noqa: SLF001
    yield
    reranker._ram_figure_needed.clear()  # noqa: SLF001
    reranker._ram_figure_attempt.clear()  # noqa: SLF001


@pytest.fixture(autouse=True)
def _cosine_floor_never_clears_by_default(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default the COSINE bootstrap floor (name-recall fix R2, plan P-24) to a
    value no cosine similarity can reach, mirroring
    `_fake_reranker_provider_by_default`'s "unscripted scores never clear the
    floor" default.

    With the no-rerank path, a test whose store has fewer than 5 embedded
    candidates (or a reranker that fails) now reaches the cosine gate instead
    of returning `None`. Left to the real bootstrap, the offline fake
    embedder's arbitrary cosines would fit an arbitrary floor and some
    candidates could clear it by accident; pinning the floor above 1.0 keeps
    every such test on its old keyword-only outcome. A test that wants the
    cosine path to pass either persists a row (`store.write_cosine_floor`) or
    monkeypatches `floor_calibration.get_cosine_bootstrap_floor` itself; a
    test of the real bootstrap calls the function object it imported at
    module load (this patch replaces only the module attribute).
    """
    if "requires_network" in request.keywords:
        return
    from brain.memory import floor_calibration

    monkeypatch.setattr(
        floor_calibration,
        "get_cosine_bootstrap_floor",
        lambda embedder_model_id: {
            "embedder_model_id": embedder_model_id,
            "floor": 2.0,
            "raw_fit_floor": 2.0,
            "sample_pairs": 0,
            "is_cold_start": True,
            "updated_at": None,
        },
    )


@pytest.fixture(autouse=True)
def _reset_known_names_cache() -> Iterator[None]:
    """Drop brain.memory.known_names's process-level per-path list cache before
    and after each test (name-recall fix N1, P-24): a test that reads the list
    through the cache must not see one a prior/later test cached, and a stale
    entry could otherwise survive a tmp_path reuse of the same signature."""
    from brain.memory import known_names

    known_names._reset_cache()
    yield
    known_names._reset_cache()


@pytest.fixture(autouse=True)
def _fake_relevance_judge_provider_by_default(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force brain.memory.relevance_judge.build_judge_provider() to the
    deterministic, offline FakeRelevanceJudgeProvider for the whole suite by
    default (F2a #250 inc6, mirrors `_fake_reranker_provider_by_default`
    above).

    build_judge_provider() is the PRODUCTION default (TorchCrossEncoderJudge
    — a real local torch/sentence-transformers cross-encoder, downloaded
    once over the network into a shared cache dir). Any test that exercises
    the daily calibration tick's judge-labeling pass — even indirectly via
    `_run_calibration_tick` — would otherwise attempt a real model download
    and a real torch import. A test that genuinely needs the real judge
    opts out with `@pytest.mark.requires_network`.

    Every candidate the FakeRelevanceJudgeProvider default scores lands far
    below any plausible ambiguous band (its unscripted-pair default score,
    see that class's docstring) — mirroring FakeRerankerProvider's "never
    accidentally clears the floor" posture, this default never accidentally
    routes an unscripted pair to a stubbed Haiku call either. Tests that
    need a specific label/ambiguous-band outcome construct their own
    `FakeRelevanceJudgeProvider(scores={...})` and pass it directly to
    `label_calibration_sample`/`_run_calibration_tick` (the `judge=`
    parameter), mirroring how tests construct their own
    `FakeRerankerProvider(scores={...})` for the reranker.
    """
    if "requires_network" in request.keywords:
        return
    from brain.memory import relevance_judge

    monkeypatch.setattr(
        relevance_judge,
        "build_judge_provider",
        # Same parameters as the real `build_judge_provider(full_model_dir=None)`
        # (F2c inc7; a signature mismatch would raise inside the stub and be
        # swallowed by label_calibration_sample's fault isolation — V3 pins it).
        lambda full_model_dir=None: relevance_judge.FakeRelevanceJudgeProvider(),
    )


@pytest.fixture(autouse=True)
def _reset_judge_provider_cache() -> Iterator[None]:
    """Reset relevance_judge.build_judge_provider()'s process-level provider
    cache before and after each test — mirrors `_reset_reranker_provider_cache`
    above for the same reason (a test that calls the REAL
    `build_judge_provider()` directly must not read or leak a provider a
    prior/later test's call happened to cache)."""
    from brain.memory import relevance_judge

    relevance_judge._reset_judge_provider_cache()
    yield
    relevance_judge._reset_judge_provider_cache()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Walk upward from this file to find the repo root (pyproject.toml).

    Replaces brittle `Path(__file__).parents[N]` patterns in tests that
    need absolute paths to checked-in resources.
    """
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not find pyproject.toml above {here}")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove companion-emergence-relevant env vars for isolation.

    Used by tests for brain.paths and brain.config (Tasks 2+).
    Each key here corresponds to an env var the framework reads at runtime.
    """
    for key in [
        "NELLBRAIN_HOME",
        "NELL_IPC_JID",
        "BRIDGE_BIND",
        "PROVIDER",
        "MODEL",
    ]:
        monkeypatch.delenv(key, raising=False)
    yield


_WEDGED_HOLDER = """
import sys
sys.path.insert(0, {root!r})
from pathlib import Path
from brain.utils.file_lock import file_lock
with file_lock(Path(sys.argv[1])):
    print("READY", flush=True)
    sys.stdin.read()
"""


@pytest.fixture
def wedged_lock(repo_root: Path) -> Callable[[Path], AbstractContextManager[None]]:
    """Hold ``file_lock(path)`` in ANOTHER process for the length of a ``with`` block (a wedged lock holder).

    Used by the lock-timeout tests: the holder never releases until the block ends, so a bounded wait must give up
    and a caller must not write.
    """

    @contextmanager
    def _hold(path: Path) -> Iterator[None]:
        code = _WEDGED_HOLDER.format(root=str(repo_root))
        proc = subprocess.Popen(
            [sys.executable, "-c", code, str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert proc.stdout is not None
            assert proc.stdout.readline().strip() == "READY"
            yield
        finally:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=20)

    return _hold


_OK_FRAME = (
    '{"type": "result", "subtype": "success", "is_error": false, "result": "OK", "duration_ms": 1, "num_turns": 1}\n'
)


def _rejection_frame(model: str) -> str:
    import json

    return (
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": True,
                "api_error_status": 404,
                "result": f"There's an issue with the selected model ({model}). It may not exist or you may not have "
                "access to it.",
            }
        )
        + "\n"
    )


class ClaudeArgvCapture:
    """Every ``claude`` argv the provider spawned, and the answer each one got (#529 test instrument).

    ``reject(id)`` makes the CLI answer that model id with its real rejection frame (exit 1, status 404, the
    "issue with the selected model" sentence); every other id gets a clean frame. The seam is the subprocess call the
    provider tests already patch, so the argv is built by the real provider code.
    """

    def __init__(self) -> None:
        self.argvs: list[list[str]] = []
        self.rejected: set[str] = set()

    def reject(self, *models: str) -> None:
        self.rejected.update(models)

    def accept(self, *models: str) -> None:
        self.rejected.difference_update(models)

    @staticmethod
    def model_of(argv: list[str]) -> str:
        return argv[argv.index("--model") + 1]

    @property
    def models(self) -> list[str]:
        """The ``--model`` value of every spawn, in order."""
        return [self.model_of(a) for a in self.argvs]

    @property
    def last_model(self) -> str:
        return self.models[-1]

    def _answer(self, cmd: list[str]) -> tuple[int, str]:
        self.argvs.append(list(cmd))
        model = self.model_of(cmd)
        if model in self.rejected:
            return 1, _rejection_frame(model)
        return 0, _OK_FRAME


@pytest.fixture
def capture_claude_argv(monkeypatch: pytest.MonkeyPatch) -> ClaudeArgvCapture:
    """Patch the provider's subprocess seam (run and Popen) and record every argv (see ``ClaudeArgvCapture``)."""
    import io

    from brain.bridge import provider as provider_mod

    cap = ClaudeArgvCapture()
    real_run, real_popen = subprocess.run, subprocess.Popen  # `subprocess` is one module: only `claude` is faked

    def fake_run(cmd, *args, **kwargs):
        if not (isinstance(cmd, list) and cmd and cmd[0] == "claude"):
            return real_run(cmd, *args, **kwargs)
        rc, out = cap._answer(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=rc, stdout=out, stderr="")

    class FakePopen(real_popen):  # type: ignore[misc, valid-type]
        def __new__(cls, cmd, *args, **kwargs):
            if not (isinstance(cmd, list) and cmd and cmd[0] == "claude"):
                return real_popen(cmd, *args, **kwargs)
            return super().__new__(cls)

        def __init__(self, cmd, *args, **kwargs):
            self._rc, out = cap._answer(cmd)
            self.stdin = io.StringIO()
            self.stdout = iter(out.splitlines(keepends=True))
            self.stderr = io.StringIO("")
            self.pid = 4242

        def wait(self, timeout=None):
            return self._rc

        def poll(self):
            return self._rc

        def terminate(self):
            pass

        def kill(self):
            pass

    monkeypatch.setattr(provider_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(provider_mod.subprocess, "Popen", FakePopen)
    return cap


@pytest.fixture
def model_override() -> Callable[[str, str | None], None]:
    """Set (or with ``None`` clear) one job's model in the sandbox home's tunables file, through the one writer.

    ``model_override(tier, value)`` is ``set_job_models({tier: value})`` under the per-test home, so the write rule
    applies (a word that means the job's default removes the line, a full id is written as given).
    """
    from brain.bridge.model_tier import set_job_models

    def _set(tier: str, value: str | None) -> None:
        set_job_models({tier: value})

    return _set

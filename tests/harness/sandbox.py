"""The safety core — a context manager that confines a live run to a temp sandbox.

**This is the #1 requirement.** The harness runs on developer laptops where a REAL companion lives
(sometimes a Mac with no VM). Nothing a run does may touch or corrupt anything outside its temp
sandbox. Every run passes through :func:`sandbox`.

What it does (see the module ``README.md`` for the guarantees):

1. Create a fresh ``mkdtemp(prefix="ce-harness-")`` root + a ``claude-config`` subdir.
2. Redirect env — set ``KINDLED_HOME=<root>`` and ``CLAUDE_CONFIG_DIR=<root>/claude-config``, and
   UNSET ``NELLBRAIN_HOME``. These are the REAL engine mechanisms:
   ``brain/paths.py:58-82`` routes all persona state to ``KINDLED_HOME``; the provider respects an
   upstream ``CLAUDE_CONFIG_DIR`` (``brain/bridge/provider.py:174``) so the CLI subprocess reads our
   seeded config, not the user's real ``~/.claude``.
3. Auth-only seed — copy ONLY ``~/.claude/.credentials.json`` into the sandbox config dir. Never
   ``CLAUDE.md`` / ``settings*`` / ``skills`` / ``plugins``. On Mac the credential may live in the
   Keychain; a fresh ``CLAUDE_CONFIG_DIR`` still authenticates via Keychain, so the branch is
   recorded, not hard-failed.
4. Fingerprint the real guarded roots (broad, platformdirs-derived) BEFORE the run. For ``~/.claude``
   the orchestrator's own claude-code session-runtime logs are pruned (F4; see
   ``_claude_session_log_excludes``) — the sandboxed subject cannot reach those (its CLI runs under
   the tempdir ``CLAUDE_CONFIG_DIR``), so pruning them drops only the live driving session's noise.
5. ``yield`` a :class:`SandboxHandle`.
6. AFTER the run, re-fingerprint; if any guarded root changed, raise :class:`SandboxLeak`.
7. Restore env (in a ``finally``, even on ``SandboxLeak``) and ``rmtree`` the root (unless ``keep``).

**Concurrency:** ``sandbox()`` mutates process-global ``os.environ`` — it is NOT thread-safe or
safely nestable and assumes serial use within a process. pytest-xdist workers are separate
processes, so parallel CI is fine.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import warnings
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import PlatformDirs

_APP = "companion-emergence"

# Files small + critical enough to content-hash (defends the same-size+same-mtime overwrite blind
# spot): the auth/config dotfiles directly under ~/.claude. Cheap — a handful of small files.
_HASH_MAX_BYTES = 1 << 20  # 1 MiB — never hash a large file (keep the fingerprint cheap)


class SandboxLeak(RuntimeError):  # noqa: N818 — owner-specified public API name (spec/criteria)
    """Raised when a guarded real-home root was mutated during a sandboxed run."""


class EditablePathRefused(RuntimeError):  # noqa: N818 — owner-specified public API name (F5)
    """Raised when a declared ``editable_paths`` entry fails the sandbox-extension collision-guard.

    F5 lets a test author name real, OUTSIDE-sandbox paths that become PART OF the sandbox (excluded
    from the leak fingerprint AND writable by the sandboxed persona via the Bob-confirms mechanism).
    This is a deliberate, opt-in hole in the #1 isolation guarantee, so the collision-guard is
    STRICT (sentinel-mandatory): a named path is accepted ONLY if it does not exist yet, OR it is a
    directory carrying the distinctive :data:`HARNESS_EDITABLE_SENTINEL` file. An existing folder
    WITHOUT the sentinel (empty OR populated), a regular file, or an unreadable path is REFUSED — so
    the allowlist can never be pointed at a real companion's real ``~/Documents/<Name> Notes`` (a
    real companion never drops the harness sentinel into its notes folder). Distinct from
    :class:`SandboxLeak` (NOT a subclass).
    """


# The distinctive marker a test author must place inside a PRE-EXISTING editable path so the
# collision-guard accepts it. A real companion's real notes folder never contains this file, so its
# presence is proof the folder is test-owned. A non-existent editable path needs no sentinel (nothing
# to collide with) — the run/test creates it.
HARNESS_EDITABLE_SENTINEL = ".ce-harness-editable"


class LiveServiceDetected(RuntimeError):  # noqa: N818 — owner-specified public API name (Phase 2)
    """Raised (default policy) when a live companion bridge is detected BEFORE a run starts.

    Distinct from :class:`SandboxLeak` (NOT a subclass), so a caller catching ``SandboxLeak`` for a
    real containment failure does not accidentally swallow this, and vice-versa. The pre-check that
    raises it detects a running bridge UP FRONT so the run fails with an actionable "quit your
    companion first" message instead of dying later with a misleading *spurious* ``SandboxLeak``
    (its concurrent writes tripping the post-run fingerprint).
    """


# live_check policy values (Phase 2).
LIVE_CHECK_RAISE = "raise"
LIVE_CHECK_WARN = "warn"
LIVE_CHECK_OFF = "off"
_LIVE_CHECK_POLICIES = (LIVE_CHECK_RAISE, LIVE_CHECK_WARN, LIVE_CHECK_OFF)

# The double-fingerprint probe's default wait window (seconds). Small — the probe is opt-in and the
# pidfile scan carries default detection, so this only costs when a caller passes probe=True.
_PROBE_WAIT_S = 0.4


@dataclass
class SandboxHandle:
    """Handle yielded to the run: the sandbox root + the env it installed."""

    root: Path
    env: dict[str, str]
    claude_config_dir: Path
    auth_source: str  # "credentials-file" | "keychain-or-inherited"
    guard_roots: list[Path] = field(default_factory=list)

    @property
    def personas_dir(self) -> Path:
        return self.root / "personas"

    def persona_dir(self, name: str) -> Path:
        return self.personas_dir / name


def _guarded_roots(extra: list[Path] | None = None) -> list[Path]:
    """The set of real-home roots we fingerprint. BROAD + platformdirs-DERIVED (stage-3 R1).

    Deriving from ``platformdirs`` (not a hand-list) means the whole data/cache/state/log/config
    family for the app is covered — including the cache and state dirs a bare ``user_data_path``
    would miss — and any future engine dir under the same family is covered automatically. We
    fingerprint the roots themselves; a write to any child changes the recursive fingerprint.

    NOTE: ``~/Documents`` is deliberately NOT here — a full recursive walk of a real (often
    iCloud-synced) Documents dir is slow AND a concurrent sync mtime bump would raise a SPURIOUS
    ``SandboxLeak``. Notes are force-disabled per persona anyway, so only the specific
    ``<Persona> Notes`` folder is checked, shallow + non-recursive (see :func:`_notes_roots`).
    """
    dirs = PlatformDirs(_APP)
    home = Path.home()
    roots: list[Path] = [
        Path(dirs.user_data_path),
        Path(dirs.user_cache_path),
        Path(dirs.user_state_path),
        Path(dirs.user_log_path),
        Path(dirs.user_config_path),
        home / ".claude",                 # the real claude config (sandbox config is under tempdir)
        home / ".config" / "systemd" / "user",   # Linux autostart
        home / "Library" / "LaunchAgents",        # Mac autostart
        home / "Library" / "Application Support" / _APP,  # Mac data (redundant w/ platformdirs; safe)
    ]
    # Stage-3 F1: PlatformDirs(_APP) above drops the appauthor the ENGINE uses
    # (brain/paths.py: PlatformDirs(appname=..., appauthor="hanamorix")), so on Windows/macOS it
    # points at a DIFFERENT dir than the real bridge writes. Add the engine's appauthor-correct
    # DEFAULT home so the leak fingerprint covers the dir the real bridge actually uses. NOTE: we use
    # the engine's un-overridden ``_dirs.user_data_path`` — NOT ``get_home()`` — because inside a
    # sandbox ``KINDLED_HOME`` makes ``get_home()`` return the SANDBOX root, and fingerprinting the
    # sandbox's own root would (correctly-written) in-sandbox persona files trip a false SandboxLeak.
    # We want the REAL user's home regardless of the sandbox override. Import-only use of brain.
    try:
        from brain.paths import _dirs as _brain_dirs

        roots.append(Path(_brain_dirs.user_data_path))
    except Exception:  # noqa: BLE001 — never let a resolver hiccup break the guard-root set
        pass
    if extra:
        roots.extend(Path(p) for p in extra)
    # De-dup while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for r in roots:
        rr = r.resolve()
        if rr not in seen:
            seen.add(rr)
            unique.append(rr)
    return unique


# --- claude-code ORCHESTRATOR session-runtime logs (F4) ------------------------------------------
# In the Agent-Bob orchestration model the process DRIVING the harness is itself a live claude-code
# session, and it (plus any spawned subagent) writes its normal session/subagent runtime logs under
# the developer's REAL ~/.claude DURING the run. Those writes are the ORCHESTRATOR's own logging —
# never companion/Canary state — so a recursive fingerprint of ~/.claude trips a GUARANTEED SPURIOUS
# SandboxLeak on every orchestrator-driven run (F4). We prune these session-log subtrees/files from
# the ~/.claude fingerprint.
#
# WHY THIS OPENS NO ESCAPE HOLE: the sandboxed subject's own `claude` CLI subprocesses inherit
# CLAUDE_CONFIG_DIR=<tempdir>/claude-config (set into os.environ at `sandbox()` startup) because
# brain/bridge/provider.py:_subprocess_env RESPECTS an already-set upstream CLAUDE_CONFIG_DIR
# ("if 'CLAUDE_CONFIG_DIR' not in env"). So the SUBJECT's projects/todos/shell-snapshots/... land in
# the tempdir, NOT the real ~/.claude — only the ORCHESTRATOR (running under the real, unset-by-us
# config dir) writes the real paths below. Pruning them drops only orchestrator noise.
#
# FAIL-CLOSED ALLOWLIST, not a denylist: only the NAMED entries below are excluded; everything else
# under ~/.claude stays guarded (.credentials.json, settings*, CLAUDE.md, skills/, plugins/, hooks/,
# cache/, and any FUTURE/unknown dir). A future claude-code release that adds a new session-log dir
# would re-trip a SAFE spurious leak (fail-closed) until that name is added here — never a silent
# hole. The set is complete for the claude-code version observed on the dev boxes (see the run's
# 8-harness.md live-session churn reconciliation); it is a manual/advisory completeness check
# because it can only be observed with a REAL live claude session, not a fake-HOME unit test.
#
# PER-DIR SAFETY (AF2, 2026-07-12): every entry below is excluded on ONE mechanical argument, not a
# per-entry judgement about "companion vs not": the sandboxed Canary's `claude` CLI runs under the
# tempdir CLAUDE_CONFIG_DIR (provider.py:174 / _subprocess_env respects the upstream value we set),
# so the CANARY CANNOT WRITE ANY REAL ~/.claude SUBDIR — its projects/todos/file-history/... all land
# in the tempdir. Therefore EVERY real ~/.claude write during a run is the ORCHESTRATOR's own claude
# session, and excluding an orchestrator session-runtime dir hides NO Canary leak. The per-entry
# comments below name WHICH orchestrator-runtime artifact each is (why it churns every session), but
# the safety rests on the shared confinement mechanism, not on any entry being intrinsically benign.
# This is pinned by test_af2_canary_claude_writes_confined_to_tempdir + the provider-guard test.
_CLAUDE_SESSION_LOG_DIRS = (
    "projects",         # orchestrator session + subagent transcripts (the core F4 case)
    "todos",            # orchestrator per-session todo state
    "shell-snapshots",  # orchestrator per-shell env snapshots
    "statsig",          # orchestrator feature-flag / telemetry gate cache
    "sessions",         # orchestrator session runtime state
    "session-env",      # orchestrator per-session env dirs
    "telemetry",        # orchestrator usage telemetry
    "backups",          # orchestrator's rotated ~/.claude.json snapshots (churn every session)
    "paste-cache",      # orchestrator transient paste cache
    "downloads",        # orchestrator CLI download cache
    # file-history: the CLI's OWN edit-history/backup log of files THE ORCHESTRATOR'S claude session
    # edited this run — on a real dev box those can be real project files. It is excluded NOT because
    # its contents are "not companion files" (they may be real files) but because the CANARY cannot
    # write it: the Canary's CLI file-history lands in the tempdir CLAUDE_CONFIG_DIR, never real
    # ~/.claude/file-history. So every entry here is the orchestrator's own edit log → excluding it
    # hides no Canary leak. (This is the entry the AF2 audit flagged as asserted-not-proven; the
    # proof is the confinement mechanism above, test-backed.)
    "file-history",
    "tasks",            # orchestrator per-session task/subagent bookkeeping
    "plans",            # orchestrator plan-mode scratch
    "ide",              # orchestrator IDE lock/handshake files (per session)
)
_CLAUDE_SESSION_LOG_FILES = (
    "history.jsonl",              # orchestrator CLI command history
    ".last-cleanup",             # orchestrator housekeeping marker
    ".last-update-result.json",  # orchestrator updater result marker
    "mcp-needs-auth-cache.json",  # orchestrator transient MCP-auth hint cache
)


def _claude_session_log_excludes() -> list[Path]:
    """Session-runtime log subtrees + files under ~/.claude to prune from the leak fingerprint (F4).

    Orchestrator's OWN claude-code logs — pruning them stops the guaranteed-spurious SandboxLeak on
    every orchestrator-driven run WITHOUT opening a hole (the sandboxed subject writes only into the
    tempdir; see the module comment above and provider.py:_subprocess_env). Derived by NAME as
    ``Path.home()/".claude"/<name>`` so an absent entry is a harmless no-op (the prune simply never
    matches) → correct on Linux and macOS alike. Fail-closed allowlist: only these names are
    excluded; every other ~/.claude path stays guarded.
    """
    base = Path.home() / ".claude"
    return [base / name for name in (*_CLAUDE_SESSION_LOG_DIRS, *_CLAUDE_SESSION_LOG_FILES)]


def _documents_dir() -> Path:
    """The user's Documents dir, resolved via the SAME resolver the engine uses for the real notes
    folder (``platformdirs.user_documents_dir`` — see ``brain/notes/config.py:resolve_notes_folder``).

    F5/F-6: deriving the shallow-scan Documents dir from ``platformdirs`` (which honors XDG
    ``user-dirs.dirs`` / ``XDG_DOCUMENTS_DIR``) keeps it consistent with where the REAL notes folder
    lands, so an editable path under a *relocated* Documents is correctly matched by the
    ``exclude_names`` parent-check instead of tripping a spurious (fail-safe) leak. Falls back to
    ``~/Documents`` if the resolver hiccups. Returned ABSOLUTE (not a ``~``-string)."""
    try:
        from platformdirs import user_documents_dir

        return Path(user_documents_dir())
    except Exception:  # noqa: BLE001 — never let a resolver hiccup break the guard
        return Path.home() / "Documents"


def _notes_roots() -> list[Path]:
    """Notes folders to check SHALLOW + non-recursive (A1: never recursively walk Documents).

    Returns the absolute Documents dir (``_documents_dir()`` — a real path, NOT a ``~``-string; F-5).
    Notes are force-disabled per persona, so any appearance/growth of a ``*Notes`` folder directly
    under Documents is itself the leak signal — a shallow listing catches it without an expensive
    recursive walk of an iCloud-synced Documents.
    """
    return [_documents_dir()]


def _content_hash(f: Path) -> str | None:
    """sha256 of a small file's bytes, or None if too large / unreadable."""
    try:
        if f.stat().st_size > _HASH_MAX_BYTES:
            return None
        return hashlib.sha256(f.read_bytes()).hexdigest()
    except OSError:
        return None


def _hash_critical(root: Path) -> bool:
    """Whether to content-hash files under this root. True for ~/.claude (small auth/config
    dotfiles) — closes the same-size+same-mtime overwrite blind spot cheaply."""
    return root.resolve() == (Path.home() / ".claude").resolve()


def _fingerprint(
    root: Path, exclude: Iterable[Path] | None = None, *, hash_content: bool = False
) -> dict:
    """Recursive {relpath -> (size, mtime_ns[, sha256])} for a root. Missing root -> empty map.

    ``exclude`` is a set of paths to prune — each may name a **directory** (its whole subtree is
    skipped) or an individual **file** (that file is skipped). Used so ``~/.claude`` fingerprinting
    ignores (a) the sandbox's own claude-config subtree if it ever nested under it — defensive, it
    does not — and (b) the orchestrator's own claude-code session-runtime logs (F4; see
    :func:`_claude_session_log_excludes`). A single-subtree caller passes a one-element iterable.
    Paths are compared **resolved on both sides** — deliberately more symlink-robust than an
    unresolved equality; the equivalence at the one pre-existing (canonical, non-symlinked) call
    site is unchanged. ``hash_content`` adds a content sha256 for small files so a
    same-size+same-mtime in-place overwrite is still detected (M1).
    """
    fp: dict = {}
    if not root.exists():
        return fp
    excludes = {e.resolve() for e in exclude} if exclude else set()
    # Cheap pre-filter for the per-FILE exclusion: only the basenames of excluded paths can match a
    # file, so we skip the expensive f.resolve() syscall for every file whose name isn't even a
    # candidate. A name collision (a kept file sharing a basename with an excluded path elsewhere) is
    # then disambiguated by the resolve() confirm — so behavior is identical, minus the syscall on the
    # (overwhelming) majority of files. Empty when ``exclude`` is None.
    exclude_names = {ex.name for ex in excludes}
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        rp = dp.resolve()
        # Prune a whole DIRECTORY subtree named in ``excludes`` (dir at, or under, an excluded path).
        if any(rp == ex or ex in rp.parents for ex in excludes):
            dirnames[:] = []
            continue
        for name in filenames:
            f = dp / name
            # Skip an individual excluded FILE (e.g. a top-level ~/.claude/history.jsonl, seen on the
            # first os.walk iteration where dirpath == root and no dir-prune applies). Name-match
            # first (cheap), resolve() only to confirm a candidate (MINOR-2: no per-file syscall).
            if name in exclude_names and f.resolve() in excludes:
                continue
            try:
                st = f.stat()
            except OSError:
                continue
            entry: tuple = (st.st_size, st.st_mtime_ns)
            if hash_content:
                entry = (st.st_size, st.st_mtime_ns, _content_hash(f))
            fp[str(f.relative_to(root))] = entry
    return fp


def _shallow_notes_fingerprint(docs: Path, exclude_names: set[str] | None = None) -> dict:
    """Shallow, non-recursive listing of ``*Notes`` entries directly under Documents (A1).

    Records each top-level ``*Notes`` entry's (size, mtime_ns). No recursion into the (possibly huge,
    iCloud-synced) tree — a new/changed persona-Notes folder appearing is the leak.

    ``exclude_names`` (F5): basenames of declared ``editable_paths`` whose resolved parent IS this
    ``docs`` dir — those ``*Notes`` entries are SKIPPED so a declared editable notes folder does not
    trip a leak. Exact by basename equality: a SIBLING ``*Notes`` folder's basename is NOT in the set,
    so it still trips (the exclusion is not one path broader). Empty/None ⇒ byte-identical to before.
    """
    fp: dict = {}
    if not docs.exists():
        return fp
    try:
        entries = list(os.scandir(docs))
    except OSError:
        return fp
    skip = exclude_names or set()
    for e in entries:
        if e.name.endswith("Notes") and e.name not in skip:
            try:
                st = e.stat()
            except OSError:
                continue
            fp[e.name] = (st.st_size, st.st_mtime_ns)
    return fp


def _validate_editable_paths(paths: Iterable[Path] | None) -> list[Path]:
    """The F5 sandbox-extension collision-guard (sentinel-MANDATORY). Resolve + refuse-unless-safe.

    Accept a named path ONLY if it (a) does NOT exist yet (the run/test creates it as a directory),
    OR (b) exists as a DIRECTORY carrying the :data:`HARNESS_EDITABLE_SENTINEL` file directly inside
    it. REFUSE (:class:`EditablePathRefused`) an existing directory WITHOUT the sentinel (empty OR
    populated), an existing regular file / non-directory, or an unreadable path.

    Why sentinel-mandatory (owner decision, stage-3 MAJOR-1): the fixed persona name "Canary" means
    the default editable target ``~/Documents/Canary Notes`` is the EXACT path a real Canary-named
    companion uses. A real companion never drops the harness sentinel into its notes folder, so
    requiring the sentinel (or non-existence) is a bright line no real-data folder — even an empty
    one — crosses. Runs BEFORE any env/fingerprint mutation. Returns the resolved list (empty for
    ``None``/empty input ⇒ the whole feature is inert / default-OFF).
    """
    if not paths:
        return []
    out: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        try:
            exists = p.exists()
        except OSError as exc:
            raise EditablePathRefused(f"editable path {p} is unreadable: {exc}") from exc
        if not exists:
            out.append(p)  # (a) non-existent — nothing to collide with; created later
            continue
        if not p.is_dir():
            raise EditablePathRefused(
                f"editable path {p} exists but is not a directory — refused (must be a "
                "non-existent path or a sentinel-marked directory)"
            )
        if not (p / HARNESS_EDITABLE_SENTINEL).is_file():
            raise EditablePathRefused(
                f"editable path {p} is an existing directory without the harness sentinel "
                f"{HARNESS_EDITABLE_SENTINEL!r} — refused (it could be a real companion's real "
                "data; drop the sentinel file inside it to declare it test-owned)"
            )
        out.append(p)  # (b) sentinel-marked dir — provably test-owned
    return out


def _live_bridges() -> list[tuple[int, str]]:
    """Scan for RUNNING companion bridges (the load-bearing live-service detector, Phase 2).

    A live bridge writes ``bridge.json`` into its ``persona_dir`` carrying its ``pid``
    (``brain/bridge/state_file.py``). We glob ``<home>/personas/*/bridge.json`` where ``<home>`` is
    the engine's REAL resolver ``brain.paths.get_home()`` — NOT a re-derived ``PlatformDirs`` (which
    drops the ``appauthor`` the engine uses and so points at the WRONG dir on Windows/macOS,
    stage-3 F1). Liveness is decided by the REAL, side-effect-free ``state_file.pid_is_alive``.

    **Read-only, deliberately:** we parse ``bridge.json`` bytes directly and never call
    ``state_file.read()`` — ``read()`` heals-on-read (``attempt_heal`` can ``os.replace`` a ``.bak``
    over a corrupt primary), which would WRITE to the developer's real files. The cost is a known,
    accepted gap: a live bridge whose primary ``bridge.json`` is corrupt-but-``.bak``-recoverable is
    NOT detected here (backstopped by the post-run :class:`SandboxLeak`).

    Returns a list of ``(pid, persona_name)`` for each live bridge found (empty = none live).
    Tolerant: a missing home, a missing/corrupt/unreadable ``bridge.json`` is skipped, never raised.
    """
    import json

    from brain.bridge import state_file
    from brain.paths import get_home

    live: list[tuple[int, str]] = []
    try:
        personas = get_home() / "personas"
        candidates = list(personas.glob("*/bridge.json"))
    except OSError:
        return live
    for path in candidates:
        try:
            data = json.loads(path.read_bytes())
        except (OSError, ValueError):  # ValueError ⊇ json.JSONDecodeError
            continue
        if not isinstance(data, dict):
            continue
        pid = data.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            continue
        try:
            alive = state_file.pid_is_alive(pid)
        except OSError:
            continue
        if alive:
            persona = path.parent.name
            live.append((pid, persona))
    return live


def _probe_external_writer(snapshot_fn, wait_s: float) -> bool:
    """Double-fingerprint probe (OPT-IN, default OFF — stage-3 M1): snapshot the guarded roots,
    wait, snapshot again; return True if anything changed (an external writer is live).

    Complementary to :func:`_live_bridges` — catches a live writer with no discoverable pidfile.
    Off by default because the pidfile scan carries normal detection and the post-run
    :class:`SandboxLeak` already backstops the no-pidfile case; a probe-only hit is reported with a
    GENERIC message (it is NOT necessarily a companion bridge).
    """
    import time

    before = snapshot_fn()
    if wait_s > 0:
        time.sleep(wait_s)
    after = snapshot_fn()
    return before != after


def _run_live_check(
    policy: str, snapshot_fn, *, probe: bool, probe_wait: float
) -> str | None:
    """Run the live-service pre-check and dispatch by ``policy`` (stage-3 M1/L4).

    ``"off"`` → skip entirely (no scan, no probe, no cost) and return ``None``.
    ``"raise"`` → raise :class:`LiveServiceDetected` if a live service is found.
    ``"warn"`` → emit a ``RuntimeWarning`` and return the message (so a later :class:`SandboxLeak`
    can be annotated), continuing the run.

    An invalid ``policy`` raises ``ValueError`` (fail-fast on a typo, never silent-off).
    """
    if policy not in _LIVE_CHECK_POLICIES:
        raise ValueError(
            f"live_check must be one of {_LIVE_CHECK_POLICIES!r}, got {policy!r}"
        )
    if policy == LIVE_CHECK_OFF:
        return None

    parts: list[str] = []
    bridges = _live_bridges()
    if bridges:
        listed = ", ".join(f"pid {pid} (persona {name})" for pid, name in bridges)
        parts.append(
            "a live companion service was detected at pre-check: " + listed
        )
    if probe and _probe_external_writer(snapshot_fn, probe_wait):
        # A probe-only hit is NOT necessarily a companion bridge — keep the message GENERIC (M1/L4).
        parts.append(
            "an external process mutated a guarded root during the pre-check probe window"
        )
    if not parts:
        return None

    msg = (
        "; ".join(parts)
        + ". Quit your companion bridge (and any launchd/systemd/task-scheduler service) "
        "before running the harness, then retry."
    )
    if policy == LIVE_CHECK_RAISE:
        raise LiveServiceDetected(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)
    return msg


def _seed_auth(claude_config_dir: Path) -> str:
    """Auth-only seed: copy ONLY ~/.claude/.credentials.json. Return the auth source.

    On Mac the credential can live in the Keychain rather than a file — a fresh
    ``CLAUDE_CONFIG_DIR`` still authenticates via Keychain, so we record the branch instead of
    hard-failing. NEVER copy CLAUDE.md / settings* / skills / plugins.
    """
    cred = Path.home() / ".claude" / ".credentials.json"
    if cred.is_file():
        claude_config_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cred, claude_config_dir / ".credentials.json")
        return "credentials-file"
    claude_config_dir.mkdir(parents=True, exist_ok=True)
    # No cred file. On a Mac the credential may be in the Keychain (a fresh CLAUDE_CONFIG_DIR still
    # auths via Keychain) — plausible only on Darwin. Elsewhere this is genuinely unauthenticated,
    # which would fail deep in a live bridge run; record it clearly + warn now (A2) rather than let
    # it surface as an opaque provider failure. (Unit tests seed a fake cred, so never hit this.)
    import sys

    if sys.platform == "darwin":
        return "keychain-or-inherited"
    warnings.warn(
        "sandbox: no ~/.claude/.credentials.json found and not on macOS — a live run "
        "will be UNAUTHENTICATED and fail at the provider. Authenticate the claude CLI first.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "unauthenticated"


@contextmanager
def sandbox(
    *,
    keep: bool = False,
    extra_guard_roots: list[Path] | None = None,
    editable_paths: list[Path] | None = None,
    live_check: str = LIVE_CHECK_RAISE,
    probe: bool = False,
    probe_wait: float = _PROBE_WAIT_S,
) -> Iterator[SandboxHandle]:
    """Confine a live run to a fresh temp sandbox; assert no guarded root was mutated.

    Args:
        keep: leave the tempdir on disk after exit (post-mortem). Default: ``rmtree``.
        extra_guard_roots: additional roots to fingerprint. Used by the isolation test's negative
            control to prove the leak oracle fires on a real mutation.
        editable_paths: the F5 SANDBOX-BOUNDARY EXTENSION (default ``None`` ⇒ inert / byte-identical
            to today). Real, OUTSIDE-sandbox paths the author declares to be PART OF the sandbox:
            each is (a) EXCLUDED from the leak fingerprint — a write to it (or under it) does NOT
            raise :class:`SandboxLeak`, while every other real-home mutation still does — and (b)
            writable by the sandboxed persona via the Bob-confirms mechanism (see
            ``bob.Bob.confirm_writes``). A **deliberate, opt-in hole** in the #1 isolation guarantee:
            each path passes the STRICT sentinel-mandatory collision-guard (:func:`_validate_editable_paths`
            → :class:`EditablePathRefused` if unsafe) and the run declares the writable paths LOUDLY
            (a ``RuntimeWarning`` at start).
        live_check: live-companion-service pre-check policy (Phase 2):
            ``"raise"`` (default) → raise :class:`LiveServiceDetected` up front if a live bridge is
            found; ``"warn"`` → warn + continue (and annotate any later :class:`SandboxLeak`);
            ``"off"`` → skip the pre-check entirely (for CI, where no live bridge exists). An
            invalid value raises ``ValueError``.
        probe: opt-in double-fingerprint probe (default ``False``) — a complementary net for a live
            external writer with no discoverable ``bridge.json`` (a probe-only hit gets a GENERIC
            message, not a companion-specific one).
        probe_wait: the probe's wait window in seconds (only used when ``probe=True``).

    Yields a :class:`SandboxHandle`. Raises :class:`SandboxLeak` if any guarded root changed,
    :class:`LiveServiceDetected` (before yielding) if the pre-check finds a live service, or
    :class:`EditablePathRefused` (before yielding) if an ``editable_paths`` entry is unsafe.
    """
    root = Path(tempfile.mkdtemp(prefix="ce-harness-"))
    claude_config_dir = root / "claude-config"
    claude_config_dir.mkdir(parents=True, exist_ok=True)

    # Save prior env so we can restore it exactly (including "was unset").
    _saved: dict[str, str | None] = {
        k: os.environ.get(k) for k in ("KINDLED_HOME", "CLAUDE_CONFIG_DIR", "NELLBRAIN_HOME")
    }

    def _restore_env() -> None:
        for k, v in _saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    try:
        os.environ["KINDLED_HOME"] = str(root)
        os.environ["CLAUDE_CONFIG_DIR"] = str(claude_config_dir)
        os.environ.pop("NELLBRAIN_HOME", None)  # a stray value would win the fallback (paths.py:60)

        # F5 sandbox-extension: validate the declared editable paths (sentinel-mandatory
        # collision-guard) BEFORE any fingerprinting. A refusal raises here inside the outer try, so
        # _restore_env + rmtree in the finally still run (no stale KINDLED_HOME). Empty ⇒ inert.
        editable = _validate_editable_paths(editable_paths)
        if editable:
            # LOUD: declare exactly which real outside-sandbox paths are writable this session.
            warnings.warn(
                "sandbox: F5 editable_paths ACTIVE — these REAL outside-sandbox paths are part of "
                "the sandbox this run (excluded from the leak guard AND writable): "
                + ", ".join(str(p) for p in editable),
                RuntimeWarning,
                stacklevel=3,
            )

        auth_source = _seed_auth(claude_config_dir)
        guard_roots = _guarded_roots(extra_guard_roots)
        claude_root = (Path.home() / ".claude").resolve()
        # INVARIANT (MINOR-1): the `gr == claude_root` gate below identifies the real ~/.claude root
        # by RESOLVED-path equality — it works only because `_guarded_roots` resolves every entry
        # (sandbox.py: `rr = r.resolve()`) and `claude_root` is resolved just above. If a future edit
        # stored an unresolved root, the gate would silently fail to match and the F4 session-log
        # exclusion would not apply (fail-SAFE: a spurious leak, not a hole) — so we assert it here.
        assert all(gr == gr.resolve() for gr in guard_roots), "guard_roots must be resolved"
        notes_roots = _notes_roots()
        # For the real ~/.claude root, prune (a) the sandbox's own claude-config subtree (defensive)
        # AND (b) the orchestrator's own claude-code session-runtime logs (F4). Any OTHER guarded
        # root gets no exclusions.
        claude_excludes = [claude_config_dir, *_claude_session_log_excludes()]

        # F5: per-guard-root editable subtree exclusions + per-notes-root editable basenames. Both
        # are exact: an editable path is excluded from a root only if it is that root or under it
        # (`root in e.parents`); an editable notes basename is skipped only if its resolved parent IS
        # that notes root. A sibling / any other path is untouched → still trips (not one broader).
        def _editable_excludes_for(root: Path) -> list[Path]:
            return [e for e in editable if e == root or root in e.parents]

        def _editable_notes_names_for(notes_root: Path) -> set[str]:
            nr = notes_root.resolve()
            return {e.name for e in editable if e.parent == nr}

        def _snapshot() -> dict:
            snap = {}
            for gr in guard_roots:
                ex: list[Path] = list(_editable_excludes_for(gr))
                if gr == claude_root:
                    ex.extend(claude_excludes)
                snap[str(gr)] = _fingerprint(
                    gr,
                    exclude=ex or None,
                    hash_content=_hash_critical(gr),
                )
            for nr in notes_roots:
                snap[f"notes:{nr}"] = _shallow_notes_fingerprint(
                    nr, exclude_names=_editable_notes_names_for(nr) or None
                )
            return snap

        # Live-service pre-check (Phase 2). Runs BEFORE the "before" fingerprint and before yield —
        # so a running bridge fails fast with an actionable message instead of a misleading
        # post-run SandboxLeak. On "raise" this throws here (inside the outer try, so _restore_env
        # + rmtree in the finally still run — no stale KINDLED_HOME leaks; P14). "warn" returns a
        # message we use to annotate a later SandboxLeak (P5); "off" is a no-op.
        live_note = _run_live_check(
            live_check, _snapshot, probe=probe, probe_wait=probe_wait
        )

        before = _snapshot()

        handle = SandboxHandle(
            root=root,
            env=dict(os.environ),
            claude_config_dir=claude_config_dir,
            auth_source=auth_source,
            guard_roots=guard_roots,
        )
        try:
            yield handle
        finally:
            after = _snapshot()
            changed = [g for g in before if before[g] != after.get(g)]
            if changed:
                msg = (
                    "guarded real-home root(s) mutated during a sandboxed run: "
                    + ", ".join(changed)
                )
                if live_note is not None:
                    # warn-mode pre-check saw a live service — attribute the leak correctly (P5).
                    msg += (
                        " — NOTE: " + live_note + " This leak is probably that service's "
                        "concurrent writes, not a sandbox escape."
                    )
                raise SandboxLeak(msg)
    finally:
        _restore_env()
        if not keep:
            shutil.rmtree(root, ignore_errors=True)

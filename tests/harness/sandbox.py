"""The safety core — a context manager that confines a behavioral run to a temp sandbox.

**This is the #1 requirement.** The harness runs on developer laptops where a REAL companion lives
(a Mac with no VM, in Hana's case). Nothing a run does may touch or corrupt anything outside its
temp sandbox. Every run passes through :func:`sandbox`.

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
4. Fingerprint the real guarded roots (broad, platformdirs-derived) BEFORE the run.
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
from collections.abc import Iterator
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


def _notes_roots() -> list[Path]:
    """Notes folders to check SHALLOW + non-recursive (A1: never recursively walk ~/Documents).

    Notes are force-disabled per persona, so any appearance/growth of a ``* Notes`` folder under
    ~/Documents is itself the leak signal — a shallow listing catches it without an expensive
    recursive walk of an iCloud-synced Documents.
    """
    docs = Path.home() / "Documents"
    return [docs]


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


def _fingerprint(root: Path, exclude: Path | None = None, *, hash_content: bool = False) -> dict:
    """Recursive {relpath -> (size, mtime_ns[, sha256])} for a root. Missing root -> empty map.

    ``exclude`` (resolved) prunes a subtree — used so ``~/.claude`` fingerprinting ignores the
    sandbox's own claude-config if it ever nested under it (it does not, but be defensive).
    ``hash_content`` adds a content sha256 for small files so a same-size+same-mtime in-place
    overwrite is still detected (M1).
    """
    fp: dict = {}
    if not root.exists():
        return fp
    excl = exclude.resolve() if exclude else None
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        if excl is not None and (dp == excl or excl in dp.resolve().parents):
            dirnames[:] = []
            continue
        for name in filenames:
            f = dp / name
            try:
                st = f.stat()
            except OSError:
                continue
            entry: tuple = (st.st_size, st.st_mtime_ns)
            if hash_content:
                entry = (st.st_size, st.st_mtime_ns, _content_hash(f))
            fp[str(f.relative_to(root))] = entry
    return fp


def _shallow_notes_fingerprint(docs: Path) -> dict:
    """Shallow, non-recursive listing of ``* Notes`` entries directly under ~/Documents (A1).

    Records each top-level ``* Notes`` entry's (size, mtime_ns). No recursion into the (possibly
    huge, iCloud-synced) tree — a new/changed persona-Notes folder appearing is the leak.
    """
    fp: dict = {}
    if not docs.exists():
        return fp
    try:
        entries = list(os.scandir(docs))
    except OSError:
        return fp
    for e in entries:
        if e.name.endswith("Notes"):
            try:
                st = e.stat()
            except OSError:
                continue
            fp[e.name] = (st.st_size, st.st_mtime_ns)
    return fp


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
        "sandbox: no ~/.claude/.credentials.json found and not on macOS — a live behavioral run "
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
    live_check: str = LIVE_CHECK_RAISE,
    probe: bool = False,
    probe_wait: float = _PROBE_WAIT_S,
) -> Iterator[SandboxHandle]:
    """Confine a behavioral run to a fresh temp sandbox; assert no guarded root was mutated.

    Args:
        keep: leave the tempdir on disk after exit (post-mortem). Default: ``rmtree``.
        extra_guard_roots: additional roots to fingerprint. Used by the isolation test's negative
            control to prove the leak oracle fires on a real mutation.
        live_check: live-companion-service pre-check policy (Phase 2):
            ``"raise"`` (default) → raise :class:`LiveServiceDetected` up front if a live bridge is
            found; ``"warn"`` → warn + continue (and annotate any later :class:`SandboxLeak`);
            ``"off"`` → skip the pre-check entirely (for CI, where no live bridge exists). An
            invalid value raises ``ValueError``.
        probe: opt-in double-fingerprint probe (default ``False``) — a complementary net for a live
            external writer with no discoverable ``bridge.json`` (a probe-only hit gets a GENERIC
            message, not a companion-specific one).
        probe_wait: the probe's wait window in seconds (only used when ``probe=True``).

    Yields a :class:`SandboxHandle`. Raises :class:`SandboxLeak` if any guarded root changed, or
    :class:`LiveServiceDetected` (before yielding) if the pre-check finds a live service.
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

        auth_source = _seed_auth(claude_config_dir)
        guard_roots = _guarded_roots(extra_guard_roots)
        claude_root = (Path.home() / ".claude").resolve()
        notes_roots = _notes_roots()

        def _snapshot() -> dict:
            snap = {
                str(gr): _fingerprint(
                    gr,
                    exclude=claude_config_dir if gr == claude_root else None,
                    hash_content=_hash_critical(gr),
                )
                for gr in guard_roots
            }
            for nr in notes_roots:
                snap[f"notes:{nr}"] = _shallow_notes_fingerprint(nr)
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

"""LLM provider abstraction — ABC + concrete providers + factory.

Two call shapes coexist here:

  generate(prompt, *, system) -> str
      Simple single-turn text generation.  All existing engines (dream,
      heartbeat, reflex, research, growth) use this surface.  Do not change it.

  chat(messages, *, tools, options) -> ChatResponse
      Multi-turn structured chat with optional tool-calling.  Used by the
      chat engine, brain-tools, and the bridge daemon (SP-3, SP-4, SP-6, SP-7).

Both shapes live on LLMProvider so callers can swap providers without caring
which shape they need.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import queue as _queue_mod
import re
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from brain import tunables
from brain.bridge import cli_throttle, model_error, provider_auth, shutdown_progress
from brain.bridge.chat import (
    ChatMessage,
    ChatResponse,
    ChatStreamEvent,
    StreamDone,
    StreamError,
    TextDelta,
    ToolCall,
)
from brain.bridge.model_tier import TIER_INTERACTIVE_CHAT, model_for_tier
from brain.bridge.usage_log import log_usage
from brain.utils.time import format_local, parse_iso_utc

logger = logging.getLogger(__name__)

# Windows: a console child (the `claude` CLI) launched from a windowless bridge
# (pythonw / Task Scheduler, since v0.0.33) gets a fresh visible console unless
# told otherwise — a "claude" window flashes on every provider call and can
# steal keyboard focus. CREATE_NO_WINDOW suppresses it; the flag exists only on
# Windows, so off-platform this is 0 (a no-op). stdout/stderr capture is
# unaffected. Passed as creationflags= on every `claude` spawn below.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# #456 (D23): every `claude` child starts in its own session on POSIX, so a terminal Ctrl-C on `nell supervisor run`
# reaches the bridge (which stops gracefully) and not its in-flight calls. Windows children are already detached from
# the console (CREATE_NO_WINDOW); subprocess ignores the flag there.
_OWN_SESSION = os.name != "nt"

# ---------------------------------------------------------------------------
# Per-turn budget backstop (Task 6 / A)
# ---------------------------------------------------------------------------

# Generous catastrophe ceiling — must never bite a normal reply.
# Sized to let even a heavy multi-tool Opus turn finish; only catches
# runaway 11-28-tool file-crawl loops that would exhaust the subscription.
# Matched by substring of the model id so short aliases ("sonnet") and
# fully-qualified ids ("claude-sonnet-4-6") both resolve.
_BUDGET_BY_TIER: tuple[tuple[str, float], ...] = (
    ("haiku", tunables.register(
        "provider.max_turn_budget_usd.haiku", 0.30,
        description="Safety cap on what one chat turn on a Haiku model may cost, in US dollars.",
    )),
    ("sonnet", tunables.register(
        "provider.max_turn_budget_usd.sonnet", 0.75,
        description="Safety cap on what one chat turn on a Sonnet model may cost, in US dollars.",
    )),
    ("opus", tunables.register(
        "provider.max_turn_budget_usd.opus", 3.00,
        description="Safety cap on what one chat turn on an Opus model may cost, in US dollars.",
    )),
)
_BUDGET_FALLBACK: float = tunables.register(
    "provider.max_turn_budget_usd.fallback", 1.50,
    description="Safety cap on what one chat turn on any other model may cost, in US dollars.",
)

# What json.loads raises on a line it cannot take: JSONDecodeError and, on a hostile line, a plain ValueError
# (a 5000-digit run) or RecursionError (nesting beyond the recursion limit). JSONDecodeError is a ValueError.
_JSON_LINE_ERRORS: tuple[type[BaseException], ...] = (ValueError, RecursionError)


def _text_only(value: object) -> str:
    """The value if it is a string, else "". A JSON `null` (or a number) in a text field is not text: `str(None)`
    would put the literal word "None" into a reply (#467)."""
    return value if isinstance(value, str) else ""

# The result-frame subtype the CLI reports when --max-budget-usd stops a turn (exit code 1).
_BUDGET_SUBTYPE: str = "error_max_budget_usd"

# Graceful message when the CLI hits the per-turn ceiling mid-reply.
_BUDGET_EXCEEDED_MSG: str = (
    "I reached this turn's work limit before I finished — "
    "ask me to continue and I'll pick up where I left off."
)


def _MAX_TURN_BUDGET_USD(model: str) -> float:  # noqa: N802  (module-level constant function)
    """Return the per-turn catastrophe budget for *model*.

    This is NOT a normal budget — it must never fire on a legitimate reply.
    It exists solely to cap runaway file-tool loops (11-28 calls/reply) that
    would exhaust the subscription in one turn.

    Matched by substring of the lowercased model id so both short aliases
    ('sonnet', 'opus') and fully-qualified ids ('claude-sonnet-4-6') work.
    """
    low = model.lower()
    for tier, usd in _BUDGET_BY_TIER:
        if tier in low:
            return tunables.get_tunable(f"provider.max_turn_budget_usd.{tier}", usd)
    return tunables.get_tunable("provider.max_turn_budget_usd.fallback", _BUDGET_FALLBACK)


def _log_stream_timeout(persona_dir: Path | None, payload: dict[str, Any]) -> None:
    """Record a stream idle-timeout for later diagnosis (spec
    2026-06-01-stream-timeout-instrumentation). logger.warning always; best-effort
    append to <persona_dir>/stream_timeouts.jsonl when persona_dir is known.
    Never raises — runs inside stream teardown."""
    logger.warning("stream idle timeout: %s", payload)
    if persona_dir is None:
        return
    try:
        path = persona_dir / "stream_timeouts.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass

_DEFAULT_TIMEOUT_SECONDS = 300


def _brain_claude_config_dir() -> str | None:
    """Return a dedicated, brain-owned ``CLAUDE_CONFIG_DIR`` to isolate Nell's
    ``claude`` spawns from the user's *global* interactive config — or ``None``
    to leave the default behaviour untouched.

    Why: the ``claude`` CLI injects the user's global config into every ``-p``
    invocation as context (the superpowers ``using-superpowers`` block, the
    agent-types list, the enabled-skills catalogue, and the *user* memory
    ``<config>/CLAUDE.md``). None of that is Nell's persona or her
    conversation, but she receives it every turn and sometimes narrates it. It
    can't be flag-stripped without breaking her MCP tools or auth. A dedicated
    ``CLAUDE_CONFIG_DIR`` with its own one-time ``claude auth login`` is the one
    clean escape for that leg (``--mcp-config`` is unaffected by the config dir,
    so tool-calling is preserved). It does NOT strip cwd-keyed *project* memory
    (``<cwd>/.claude/CLAUDE.md`` and every ancestor's ``CLAUDE.md`` — which for a
    cwd under ``$HOME`` is the owner's global file again), cwd-keyed auto-memory,
    or project-scoped plugins; :func:`_claude_work_dir` handles that leg (#122).
    ``HOME`` is not a lever and redirecting it hides the macOS Keychain. See the
    dated row in ``docs/cli-provider-capabilities.md``.

    **Safety spine:** used ONLY when ``<KINDLED_HOME>/claude-config`` exists AND
    carries the ``.brain-authed`` marker (written by the setup helper after it
    confirms ``claude auth status`` reports logged-in). Absent the marker this
    returns ``None`` → the spawn inherits the current default config, so a
    not-yet-set-up or removed brain login can never break chat/tools; it just
    keeps the (noisy) status quo. Fail-soft on any path error.
    """
    try:
        from brain.paths import get_home

        cfg = get_home() / "claude-config"
        if (cfg / ".brain-authed").is_file():
            return str(cfg)
    except Exception:  # noqa: BLE001 — never let config isolation break a spawn
        return None
    return None


# #122: the claude CLI loads ``<cwd>/.claude/CLAUDE.md`` and every ancestor's ``CLAUDE.md`` as
# *project* memory, so a spawn whose cwd sits under ``$HOME`` (launchd/systemd run the bridge from
# there) receives the owner's global ``~/.claude/CLAUDE.md`` even with the dedicated config dir.
# Spawn from a directory whose resolved ancestor chain carries none of these markers instead.
_CWD_MEMORY_MARKERS = (".claude", "CLAUDE.md", "CLAUDE.local.md", ".git")
_CLAUDE_WORK_DIRNAME = "companion-emergence-claude-work"
_claude_work_dir_warned = False


def _has_memory_ancestor(d: Path) -> bool:
    for anc in (d, *d.parents):
        for m in _CWD_MEMORY_MARKERS:
            if (anc / m).exists():
                return True
    return False


def _claude_work_dir_candidates() -> list[Path]:
    from brain.paths import get_home

    out = [Path(tempfile.gettempdir()) / _CLAUDE_WORK_DIRNAME]
    try:
        out.append(get_home() / "claude-work")
    except Exception:  # noqa: BLE001
        pass
    return out


def _claude_work_dir() -> str | None:
    """A memory-free working directory for ``claude`` spawns, or ``None`` (status quo).

    Candidates, in order: ``<tempdir>/companion-emergence-claude-work`` (outside ``$HOME`` on
    macOS/Linux), then ``<KINDLED_HOME>/claude-work`` — a formality on any machine with Claude
    Code installed (``~/.claude`` is its ancestor), kept for tempdir-unwritable hosts. A candidate
    wins if its resolved ancestor chain has no ``.claude``/``CLAUDE.md``/``CLAUDE.local.md``/
    ``.git``, it can be created, and (POSIX) it is owned by this uid (shared ``/tmp``). No
    candidate → one WARNING per process and ``None``. Windows: both candidates sit under the
    profile, so a profile-level ``.claude\\CLAUDE.md`` keeps the status quo — see #252.
    Fail-soft: never raises.
    """
    global _claude_work_dir_warned
    tried: list[str] = []
    try:
        for cand in _claude_work_dir_candidates():
            try:
                try:
                    d = cand.resolve()
                except OSError:
                    d = cand
                tried.append(str(d))
                if _has_memory_ancestor(d):  # may raise PermissionError on a hardened parent
                    continue
                d.mkdir(parents=True, exist_ok=True, mode=0o700)
                if os.name == "posix" and hasattr(os, "getuid") and d.stat().st_uid != os.getuid():
                    continue  # foreign-owned dir in a shared /tmp
                return str(d)
            except OSError:
                continue  # this candidate is unusable; try the next
    except Exception:  # noqa: BLE001 — never let cwd selection break a spawn
        logger.debug("claude work dir selection raised", exc_info=True)
    if not _claude_work_dir_warned:
        _claude_work_dir_warned = True
        extra = " (Windows: no CLAUDE.md-free location under the profile — see #252)" if sys.platform == "win32" else ""
        logger.warning(
            "no memory-free working directory for claude spawns; tried %s — the owner's global "
            "CLAUDE.md may reach the companion%s",
            tried,
            extra,
        )
    return None


def _claude_work_dir_reset_for_tests() -> None:
    global _claude_work_dir_warned
    _claude_work_dir_warned = False


def _subprocess_env() -> dict[str, str]:
    """Environment for every ``claude`` CLI spawn.

    Scrubs ``CAVEMAN_*`` vars and forces ``CAVEMAN_DEFAULT_MODE=off``. Without
    this, a bridge daemon that was ever launched from inside a Claude Code
    session scoped to a project with a coding-session caveman override (e.g.
    this repo's ``.claude/settings.local.json``) bakes that var into its own
    process env at spawn time, and every subsequent ``claude`` subprocess it
    forks inherits it — including calls made on Nell's behalf, injecting the
    caveman ruleset into her replies. See project memory for the incident.

    Also points ``CLAUDE_CONFIG_DIR`` at a dedicated brain-owned config dir when
    one has been set up + authed (see :func:`_brain_claude_config_dir`), to keep
    the user's global plugins/skills and *user* memory out of Nell's context
    (cwd-keyed project memory is handled by :func:`_claude_work_dir`, #122). If
    an explicit ``CLAUDE_CONFIG_DIR`` is already set upstream, it is respected.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("CAVEMAN_")}
    env["CAVEMAN_DEFAULT_MODE"] = "off"
    if "CLAUDE_CONFIG_DIR" not in env:
        brain_cfg = _brain_claude_config_dir()
        if brain_cfg is not None:
            env["CLAUDE_CONFIG_DIR"] = brain_cfg
    return env


_PROVIDER_CONTEXT_OPTION_KEYS = frozenset(
    # Internal context-routing keys threaded through the provider `options` dict by
    # engine.respond — NOT generation params. They must be stripped before options are
    # forwarded to a provider's sampler (e.g. Ollama gen-params), or they corrupt the
    # model call. Keep in sync with every context key engine.py sets on chat_options
    # (persona_dir + the Option-A/A+ trio + user_name, #227). See fork 7e7c7491.
    {"persona_dir", "volatile_suffix", "include_block_clock", "session_id", "user_name"}
)

# Per-event idle budget for chat_stream. If no stdout line arrives within
# this many seconds, the subprocess is treated as wedged: it gets terminated
# and the stream yields StreamError(stage="claude_cli_idle_timeout"). This
# is deliberately not bounded by total wall-clock duration — long Opus
# runs can take time. Changeable in tunables.toml (#55).
_STREAM_PER_EVENT_IDLE_SECONDS: float = tunables.register(
    "provider.stream_per_event_idle_seconds", 60.0,
    description="Seconds of silence from Claude in the middle of a reply before the reply is treated as stuck.",
)

# First-event budget. Wider than the per-event budget because Opus can
# take time before emitting any stream_event frames. Also bounds the
# initial subprocess spawn +
# system-prompt-file write + MCP server boot.
_STREAM_FIRST_EVENT_SECONDS: float = tunables.register(
    "provider.stream_first_event_seconds", 120.0,
    description="Seconds to wait for Claude to begin a reply before the reply is treated as stuck.",
)


def _stream_per_event_idle_seconds() -> float:
    return tunables.get_tunable(
        "provider.stream_per_event_idle_seconds", _STREAM_PER_EVENT_IDLE_SECONDS
    )


def _stream_first_event_seconds() -> float:
    return tunables.get_tunable(
        "provider.stream_first_event_seconds", _STREAM_FIRST_EVENT_SECONDS
    )

# A sentinel object pushed onto the stdout queue by the reader thread
# when the subprocess closes its stdout (normal exit, error, etc.). Lets
# the main generator distinguish "reader hit EOF" from a real line.
_STREAM_EOF = object()

# How long a streaming child gets, at each step, to exit once it has been told to stop or has finished
# (#482): to leave on its own after its last frame, to answer SIGTERM, to die after SIGKILL, and (a pipe
# closes when its writer exits) for the stdout reader and the stderr drain to reach EOF after that. One knob,
# because it is one question: how long may the CLI take to flush its last output and exit? No measurement
# stands behind 5 seconds: it is the bound this path already used for the reader join before the knob
# existed, and the steps run one after the other, so the worst case is a small multiple of it. A slow
# machine raises it in tunables.toml (`provider.child_exit_grace_seconds`).
_CHILD_EXIT_GRACE_SECONDS: float = tunables.register(
    "provider.child_exit_grace_seconds", 5.0,
    description="Seconds the Claude program gets to finish its output and exit before it is stopped.",
)

# stderr kept for the failure detail (the detail uses its first 200 characters; the rest of a full pipe is
# read and dropped so the child never blocks on a write): 64 KiB is Linux's default pipe capacity.
_STDERR_KEEP_CHARS = 65536
# One read of the child's stderr: a page, so a very long line without a newline is read in pieces.
_STDERR_READ_CHARS = 4096


def _child_exit_grace_seconds() -> float:
    return tunables.get_tunable("provider.child_exit_grace_seconds", _CHILD_EXIT_GRACE_SECONDS)


class _StderrDrain:
    """Read a child's stderr on a thread for the life of the process (#482).

    A pipe nothing reads fills (64 KiB on Linux) and blocks the child's next write, which stops its stdout.
    Keeps the first ``_STDERR_KEEP_CHARS`` characters for the failure detail and discards the rest.
    """

    def __init__(self, stream: Any) -> None:
        self._chunks: list[str] = []
        self._kept = 0
        self._thread = threading.Thread(target=self._run, args=(stream,), daemon=True)
        self._thread.start()

    def _run(self, stream: Any) -> None:
        try:
            while True:
                chunk = stream.readline(_STDERR_READ_CHARS)
                if not isinstance(chunk, str) or not chunk:
                    return
                if self._kept < _STDERR_KEEP_CHARS:
                    self._chunks.append(chunk)
                    self._kept += len(chunk)
        except (OSError, ValueError):
            return  # the pipe closed under the reader

    def text(self, wait_s: float | None = None) -> str:
        """What the child wrote to stderr; waits up to ``wait_s`` (default: the child exit grace) for the pipe to reach EOF (the child has exited)."""
        self._thread.join(timeout=_child_exit_grace_seconds() if wait_s is None else wait_s)
        return "".join(self._chunks)


def _stop_child(proc: Any, *, finished: bool) -> None:
    """Leave no child behind: a finished one is given a moment to exit on its own, then terminate, wait, kill, wait (#482).

    Every exit of the streaming generator (result frame, idle timeout, error, generator close) ends here so the
    child is reaped and one that ignores SIGTERM does not keep running. Each wait is bounded; a child that
    survives SIGKILL is left to the OS and logged. On Windows ``terminate()`` is already a hard kill.
    """
    grace = _child_exit_grace_seconds()
    try:
        if proc.poll() is None and finished:
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        logger.warning("claude child %s did not exit after SIGKILL", getattr(proc, "pid", "?"))
    except OSError:
        pass  # already gone

# ---------------------------------------------------------------------------
# Lean CLI invocation — strip built-in tool definitions (~14K tok/call saved)
# ---------------------------------------------------------------------------

# Claude Code loads its built-in tools into every -p invocation, and
# --dangerously-skip-permissions rides every call, so whatever is loaded is
# callable. --allowedTools is a permission list, not an exclusive one.
#
# Two layers (#329):
#   1. --tools=<list> — the CLI's EXCLUSIVE built-in allowlist. Chat gets
#      WebSearch,WebFetch (#71: web access is a real capability); background
#      generate() gets none. Only applied when the CLI supports it
#      (_cli_supports_tools_flag).
#   2. --disallowedTools (this list) — a second layer, and the whole posture on
#      a CLI without --tools. POLICY, not dead weight: 87bfc692 once claimed "she
#      can't call them anyway", which was wrong, and swept WebFetch/WebSearch out
#      on that premise (#71).
# --strict-mcp-config then limits MCP servers to our own --mcp-config (none for
# background calls), which also keeps the account's claude.ai connectors out.
#
# Spiked, not remembered: docs/cli-provider-capabilities.md carries the dated
# rows (2026-07-14 --allowedTools; 2026-09-30 --tools). Re-spike on a CLI bump.
#
# Keep Bash/Edit/Write/Task blocked — a companion has no business with a shell.
# Do NOT re-add WebFetch/WebSearch; test_web_tools_stay_callable_at_chat_time
# guards that. Do NOT add ToolSearch either (#525, test_toolsearch_not_disallowed).
_BUILTIN_TOOLS_DISALLOWED: tuple[str, ...] = (
    "Bash",
    "Read",
    "Edit",
    "Write",
    "Glob",
    "Grep",
    "Task",
    "TodoWrite",
    "NotebookEdit",
    "BashOutput",
    "KillShell",
)


# Built-ins a chat turn may use (#71); background calls get none.
# ToolSearch is listed for #525: on claude 2.1.303 the CLI defers tool schemas
# (sends names, loads a schema on demand) only when ToolSearch is in the tool
# set. Without it every chat request carried all ~32 schemas up front. With it,
# every brain tool except ALWAYS_LOADED_TOOLS (brain/mcp_server/tools.py) and
# both web tools are deferred. Ledger: docs/cli-provider-capabilities.md.
_CHAT_BUILTIN_TOOLS: tuple[str, ...] = ("WebSearch", "WebFetch", "ToolSearch")
_NO_BUILTIN_TOOLS: tuple[str, ...] = ()

# Whether this machine's claude CLI has --tools. None = not yet known.
_TOOLS_FLAG_SUPPORTED: bool | None = None
_tools_probe_retry_at: float = 0.0
_tools_warned: bool = False
_tools_window_warned_at: float | None = None
_tools_flag_lock = threading.Lock()
_monotonic = time.monotonic  # module alias so tests can stub the clock without patching `time`

_TOOLS_PROBE_ARGV: tuple[str, ...] = ("claude", "--help")
# Parse-only: an empty stdin makes the CLI stop before any model call (rc 1 either way).
# The print flag deliberately doesn't follow the program name directly, so the tests'
# spawn-site scan (program name then print flag) doesn't count a probe as a model call.
_TOOLS_CONFIRM_ARGV: tuple[str, ...] = ("claude", "--tools=", "-p")
_TOOLS_PROBE_TIMEOUT_S = 10
_TOOLS_PROBE_RETRY_S = 60.0
# The option line in the help's own shapes: `  --tools <`, `  -t, --tools <`,
# `  --x, --tools <`, `  --tools [..]` — never a description that merely mentions it
# (2.1.284's --restricted text wraps onto a line starting "--tools names them").
_TOOLS_OPTION_RE = re.compile(
    r"^ {2}(?:-\w, )?(?:--[\w-]+, )*--tools\b(?:, --[\w-]+)*[ =<\[]", re.M
)
# Proof the output really is the help text in the format we parse.
_HELP_SENTINEL_RE = re.compile(r"^ {2}--output-format <", re.M)
# The option itself, never --allowedTools / --disallowedTools / --tools-foo.
_TOOLS_REJECTED_RE = re.compile(r"(?<![\w-])--tools\b(?!-)")


def _run_tools_probe(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    shutdown_progress.abandon_gate()  # #456 (D17): nothing new starts once the hard exit has begun
    try:
        return _run_tools_probe_spawn(argv)
    finally:
        shutdown_progress.abandon_gate()


def _run_tools_probe_spawn(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        capture_output=True,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_subprocess_env(),
        cwd=_claude_work_dir(),
        timeout=_TOOLS_PROBE_TIMEOUT_S,
        check=False,
        creationflags=_NO_WINDOW,
        start_new_session=_OWN_SESSION,  # #456 D23: a terminal Ctrl-C reaches the bridge only
    )


def _probe_tools_flag() -> bool | None:
    """Ask the CLI whether it has --tools. None = couldn't tell (don't cache).

    `--help` is fast, but the CLI hides options it accepts (--system-prompt-file
    isn't listed), so "absent" is confirmed by a parse-only `--tools=` run:
    "Input must be provided" means the option parsed; "unknown option" means it
    didn't. Anything unexpected is indeterminate, never "unsupported".
    """
    try:
        help_run = _run_tools_probe(_TOOLS_PROBE_ARGV)
        if help_run.returncode != 0 or not _HELP_SENTINEL_RE.search(help_run.stdout or ""):
            return None
        if _TOOLS_OPTION_RE.search(help_run.stdout):
            return True
        confirm = _run_tools_probe(_TOOLS_CONFIRM_ARGV)
    except Exception:  # noqa: BLE001 — a probe must never break a call
        return None
    err = confirm.stderr or ""
    if "input must be provided" in err.lower():
        return True
    if "unknown option" in err.lower() and _TOOLS_REJECTED_RE.search(err):
        return False
    return None


def _cli_supports_tools_flag() -> bool:
    """True when the claude CLI accepts --tools (see _BUILTIN_TOOLS_DISALLOWED).

    Probed once per process and cached once the answer is definitive. A probe
    that couldn't tell is not cached: calls fall back until a retry 60 s later.
    """
    global _TOOLS_FLAG_SUPPORTED, _tools_probe_retry_at, _tools_warned, _tools_window_warned_at
    supported = _TOOLS_FLAG_SUPPORTED
    if supported is not None:
        return supported
    with _tools_flag_lock:
        if _TOOLS_FLAG_SUPPORTED is not None:
            return _TOOLS_FLAG_SUPPORTED
        now = _monotonic()
        if now < _tools_probe_retry_at:
            return False
        result = _probe_tools_flag()
        if result is None:
            _tools_probe_retry_at = now + _TOOLS_PROBE_RETRY_S
            if (
                _tools_window_warned_at is None
                or now - _tools_window_warned_at >= _TOOLS_PROBE_RETRY_S
            ):
                _tools_window_warned_at = now
                logger.warning(
                    "could not tell whether this claude CLI supports --tools; built-in tools "
                    "are only partly restricted until a retry in %.0fs",
                    _TOOLS_PROBE_RETRY_S,
                )
            return False
        _TOOLS_FLAG_SUPPORTED = result
        if not result and not _tools_warned:
            _tools_warned = True
            logger.warning(
                "this claude CLI has no --tools option; built-in tools are only partly "
                "restricted (disallow list only) — update Claude Code"
            )
        return result


def _reset_tools_flag_cache() -> None:
    """Test hook: forget the probe result."""
    global _TOOLS_FLAG_SUPPORTED, _tools_probe_retry_at, _tools_warned, _tools_window_warned_at
    with _tools_flag_lock:
        _TOOLS_FLAG_SUPPORTED = None
        _tools_probe_retry_at = 0.0
        _tools_warned = False
        _tools_window_warned_at = None


def _note_tools_flag_rejected(stderr: str) -> None:
    """Fall back for this process if the CLI rejected --tools (#329, owner ruling).

    Callers pass stderr only from a failure that happened before the CLI ran
    anything (empty stdout / no frame): that's where a parse rejection lands,
    and nothing a model, tool or web page wrote can be in it. Never raises.
    """
    global _TOOLS_FLAG_SUPPORTED
    if _TOOLS_FLAG_SUPPORTED is not True or not _TOOLS_REJECTED_RE.search(stderr or ""):
        return
    with _tools_flag_lock:
        if _TOOLS_FLAG_SUPPORTED is not True:
            return
        _TOOLS_FLAG_SUPPORTED = False
    logger.warning("claude rejected --tools; falling back for this process (disallow list only)")


def _apply_lean_flags(cmd: list[str], *, builtins: tuple[str, ...] = _NO_BUILTIN_TOOLS) -> None:
    """Apply the tool posture (#329): the exclusive --tools=<builtins> allowlist
    (when the CLI supports it), the disallow list, and --strict-mcp-config.

    Defaults to NO built-ins, so a new call site fails closed; chat sites pass
    _CHAT_BUILTIN_TOOLS. The equals form keeps an empty list as one non-empty
    argv element ("--tools="), never a bare "" that Windows quoting could drop.
    """
    if _cli_supports_tools_flag():
        cmd.append(f"--tools={','.join(builtins)}")
    cmd.extend(["--disallowedTools", *_BUILTIN_TOOLS_DISALLOWED])
    cmd.append("--strict-mcp-config")


def brain_tools_mcp_entry(
    persona_dir: Path,
    *,
    request_id: str | None = None,
    session_id: str | None = None,
) -> dict:
    """The ``mcpServers["brain-tools"]`` entry the claude CLI uses to spawn the MCP child.

    Single source of truth for the child spawn (issue #138). ``-P`` (PYTHONSAFEPATH) drops the
    implicit launch-cwd entry from the child's ``sys.path``, so a foreign top-level ``brain/`` in
    the launch directory cannot shadow the installed/venv ``brain`` (right interpreter, wrong
    code). ``-P`` must precede ``-m``; it removes only the unsafe ``sys.path[0]`` (cwd / script
    dir), never ``site-packages``, so the venv/editable ``brain`` still resolves. requires-python
    >= 3.12 guarantees ``-P`` (added 3.11).

    The child inherits the parent ``claude`` process's cwd, which since #122 is the memory-free
    directory from :func:`_claude_work_dir` (or the bridge cwd when none qualifies); either way
    ``-P`` drops the cwd entry from ``sys.path`` unconditionally, so a foreign top-level ``brain/``
    in any cwd can never shadow the installed package.

    Keeping this the ONE definition lets the harness roster preflight
    (``tests.harness.roster_preflight``) reproduce the child argv/config by construction.

    ``request_id``/``session_id`` cross the process boundary via env vars, the only value
    transport available to a spawned subprocess (a live object cannot cross it) — mirrors the
    ``NELL_MCP_AUDIT_REQUEST_ID`` precedent, consumed by ``brain/mcp_server/audit.py``.
    ``session_id`` (``NELL_MCP_SESSION_ID``, consumed by ``brain/mcp_server/tools.py``) is what
    lets a ``_PROVIDER_TOOLS`` tool (currently only ``compact_history``, #80) resolve which
    session's buffer to act on when invoked via MCP. Built additively so either, both, or
    neither can be set without one clobbering the other, and ``"env"`` is omitted entirely (not
    left as an empty dict) when neither is given — preserves the pre-#80 entry shape for callers
    that check ``"env" not in entry``.
    """
    entry: dict = {
        "command": sys.executable,
        "args": [
            "-P",
            "-m",
            "brain.mcp_server",
            "--persona-dir",
            str(persona_dir),
            # The CLI registers the tools under this entry's "brain-tools" key (#417).
            "--tool-prefix",
            ClaudeCliProvider.tool_name_prefix,
        ],
    }
    env: dict[str, str] = {}
    if request_id is not None:
        env["NELL_MCP_AUDIT_REQUEST_ID"] = request_id
    if session_id is not None:
        env["NELL_MCP_SESSION_ID"] = session_id
    if env:
        entry["env"] = env
    return entry


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ProviderError(RuntimeError):
    """Raised when a provider call fails with structured context.

    Attributes
    ----------
    stage:
        Short machine-readable tag identifying where the failure occurred
        (e.g. "ollama_http", "ollama_parse", "claude_cli_timeout").
    detail:
        Human-readable explanation with whatever context is available.
    """

    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(f"[{stage}] {detail}")
        self.stage = stage
        self.detail = detail


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract LLM provider.  Subclasses implement both call shapes."""

    # What this backend calls a brain tool: prepended to the bare name wherever text the model
    # reads names one (brain.tools.naming, #417). Empty where tools are passed in by name.
    tool_name_prefix: str = ""

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Return the LLM's completion for the given prompt.

        Unchanged from Week-1 contract.  Engines (dream, heartbeat, reflex,
        research, growth) call this — do not touch it.
        """

    def complete(self, prompt: str) -> str:
        """Compatibility shim — the initiate pipeline calls ``.complete()``.

        Delegates to :meth:`generate` with no system prompt. Concrete
        providers do not need to override this; the default implementation
        works for every provider that satisfies the ``generate`` contract.
        Engines that need a system prompt should call ``generate`` directly.
        """
        return self.generate(prompt, system=None)

    @abstractmethod
    def name(self) -> str:
        """Return a short provider name (e.g. 'fake', 'claude-cli:sonnet')."""

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Multi-turn structured chat with optional tool-calls.

        Parameters
        ----------
        messages:
            Conversation history in chronological order.
        tools:
            Optional list of tool schemas (raw JSON-schema dicts in the shape
            OpenAI / Ollama / Claude all consume).  None means no tool-calling.
        options:
            Provider-specific generation options (temperature, top_p, etc.).
        """

    def healthy(self) -> bool:
        """Quick liveness check.  Default: assume healthy.

        Providers that can check (e.g. Ollama /api/tags) override this.
        ClaudeCliProvider and FakeProvider default to True.
        """
        return True


# ---------------------------------------------------------------------------
# FakeProvider — deterministic, zero network
# ---------------------------------------------------------------------------


class FakeProvider(LLMProvider):
    """Deterministic hash-based echo provider for tests — zero network calls."""

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        seed_input = (system or "").encode("utf-8") + b"||" + prompt.encode("utf-8")
        h = hashlib.sha256(seed_input).hexdigest()[:16]
        return f"DREAM: test dream {h} — an associative thread"

    def name(self) -> str:
        return "fake"

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Return a deterministic ChatResponse.

        tool_calls is always empty — FakeProvider never synthesises tool calls.
        The content hash is derived from the serialised messages so tests can
        assert determinism without caring about the exact value.
        """
        seed = json.dumps([m.to_dict() for m in messages], sort_keys=True).encode()
        h = hashlib.sha256(seed).hexdigest()[:16]
        return ChatResponse(
            content=f"FAKE_CHAT: response {h}",
            tool_calls=(),
            raw=None,
        )


# ---------------------------------------------------------------------------
# ClaudeCliProvider — shells out to `claude` CLI
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _system_prompt_tempfile(text: str | None) -> Iterator[str | None]:
    """Yield a tempfile path containing ``text``, cleaned up on exit.

    Yields ``None`` when ``text`` is ``None`` so callers can write::

        with _system_prompt_tempfile(system_prompt) as sp_path:
            if sp_path is not None:
                cmd.extend(["--system-prompt-file", sp_path])
            subprocess.run(cmd, ...)

    Why a file: Windows ``CreateProcess`` caps the entire command line at
    32,767 chars (``WinError 206``). voice.md alone runs ~15 KB; pushing it
    through ``--system-prompt`` on argv crosses the limit on long sessions.
    ``--system-prompt-file`` keeps the prompt off argv. macOS/Linux
    ``ARG_MAX`` is 256 KB–2 MB so the same problem looms there too, just
    with a larger margin.

    The file is created with ``NamedTemporaryFile(delete=False)`` (mode
    0600 on POSIX, user-only ACL on Windows) and unlinked in ``finally``
    so it survives a subprocess failure path the same as a success.
    """
    if text is None:
        yield None
        return
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    )
    try:
        tmp.write(text)
        tmp.close()
        yield tmp.name
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# Longest stdout line the assistant-frame scan will parse (a hostile or runaway line is skipped, not parsed).
_ASSISTANT_SCAN_MAX_LINE = 64 * 1024
_ERROR_CODE_SHAPE = re.compile(r"[a-z_]{1,40}")


def _argv_model(cmd: list[str]) -> str | None:
    """The model the subprocess was actually run with: the value after ``--model`` in its argv (#448).

    Read from the argv, never looked up again: a settings change can land while a call runs, and the rejection
    must name the model that was rejected.
    """
    try:
        return cmd[cmd.index("--model") + 1]
    except (ValueError, IndexError):
        return None


def _assistant_error_code(stdout: str) -> str | None:
    """The top-level ``error`` code of the LAST assistant frame in an NDJSON stdout that has one (#448).

    ``model_not_found`` sits on that frame, not on the result frame. Never raises: stdout can hold lines
    that are hostile to ``json.loads`` (a 5000-digit run raises ValueError, deep nesting RecursionError), and
    this runs inside the failure path, where an escape would replace the real CLI failure with a crash.
    """
    found: str | None = None
    try:
        for raw in (stdout or "").split("\n"):
            if not raw or len(raw) > _ASSISTANT_SCAN_MAX_LINE:
                continue
            try:
                obj = json.loads(raw)
            except _JSON_LINE_ERRORS:
                continue
            if not isinstance(obj, dict) or obj.get("type") != "assistant":
                continue
            code = obj.get("error")
            if isinstance(code, str) and _ERROR_CODE_SHAPE.fullmatch(code):
                found = code
    except Exception:  # noqa: BLE001 - the failure path must never raise from a scan
        return found
    return found


def _failure_tokens(status: object, error_code: str | None) -> str:
    """The structured signals the model-rejection detector reads, as a prefix of the failure text (#448).

    Only the two signals it reads are emitted (status 404 and the ``model_not_found`` code), so no other
    failure text changes: string classifiers elsewhere (for example ``"429"`` / ``"rate"`` in
    ``brain.initiate.review``) see exactly what they saw before.
    """
    out = ""
    if str(status) == "404":
        out += "api_error_status=404; "
    if error_code == "model_not_found":
        out += "error=model_not_found; "
    return out


def _with_model_token(detail: str, model: str | None) -> str:
    """``model=<id>; `` in front of a detail classified as a model rejection: the id that was executed."""
    from brain.bridge.model_tier import is_valid_model

    if model and is_valid_model(model) and model_error.is_model_rejection(detail):
        return f"model={model}; {detail}"
    return detail


def _is_over_budget_frame(frame: dict) -> bool:
    """True for the result frame the CLI emits when --max-budget-usd stops a turn.

    The one definition: `subtype == "error_max_budget_usd"`, or an `is_error` frame whose `errors`
    say "maximum budget". Total: a malformed `errors` (None, a number, a string, a dict) counts as no
    errors, so a frame can never raise here.
    """
    errors = frame.get("errors")
    return frame.get("subtype") == _BUDGET_SUBTYPE or bool(
        frame.get("is_error")
        and any("maximum budget" in str(e).lower() for e in (errors if isinstance(errors, list) else []))
    )


def _cli_error_detail(
    payload: dict, *, error_code: str | None = None, model: str | None = None, stdout: str | None = None
) -> str | None:
    """Detail for a result frame that reports failure, or None if it is fine.

    The CLI can exit 0 and report the failure *inside* the frame (#92) — e.g.
    ``{"is_error": true, "result": "Not logged in · Please run /login"}``. The
    non-zero-exit guard never sees those, so without this check the error
    string was assigned straight to ChatResponse.content and the user read a
    system error as something their companion said.

    The over-budget frame is excluded deliberately: it also carries
    ``is_error``, but it is a stop, not a failure, and has its own graceful
    path at each call site.

    #448: a frame with ``api_error_status`` 404 (and, from the streaming loop, the assistant frame's
    ``model_not_found`` code via ``error_code``) gets those signals as a prefix; a detail classified as a model
    rejection also gets ``model=<id the subprocess ran with>``. Frames with any other status are unchanged.
    ``stdout`` (blocking paths) is scanned for the assistant frame's code, only when the frame is a failure.
    """
    if not payload.get("is_error"):
        provider_auth.note_cli_success()  # #246: a clean frame is the recovery signal
        return None
    if _is_over_budget_frame(payload):
        return None
    detail = str(payload.get("result") or "").strip()
    detail = detail[:200] if detail else f"is_error with no result text: {payload!r}"[:200]
    if error_code is None and stdout:
        error_code = _assistant_error_code(stdout)
    detail = _with_model_token(_failure_tokens(payload.get("api_error_status"), error_code) + detail, model)
    # #246: every is_error frame on every path passes through here — the one place
    # an auth-expired text can be classified without enumerating raise sites.
    # ponytail: side-effect in a formatter, chosen so no site can be missed (#119 lesson).
    provider_auth.note_cli_failure(detail)
    return detail


def _claude_failure_detail(result: subprocess.CompletedProcess[str], *, model: str | None = None) -> str:
    """Return a useful error detail for failed Claude CLI subprocesses.

    Claude Code sometimes exits non-zero with an empty stderr but a structured
    JSON stdout payload, for example subscription quota responses:
    ``{"is_error": true, "api_error_status": 429, "result": "..."}``.
    If we only log stderr, background heartbeat failures become opaque
    ``exit 1:`` lines. Prefer the structured result when present, then fall
    back to stderr/stdout snippets.
    """
    detail = _with_model_token(_claude_failure_detail_text(result), model)
    provider_auth.note_cli_failure(detail)  # #246: every exit!=0 text passes through here
    if not (result.stdout or "").strip():  # #329: a pre-execution failure, e.g. a rejected --tools
        _note_tools_flag_rejected(result.stderr or "")
    return detail


def _last_result_line(stdout: str) -> dict[str, Any] | None:
    """The last `{"type": "result"}` line of an NDJSON stdout, else None (#330).

    Splits on "\n" only: `splitlines()` would also break a frame on a raw
    U+2028/U+2029/U+0085 inside a JSON string.
    """
    for raw in reversed(stdout.split("\n")):
        try:
            obj = json.loads(raw)
        except _JSON_LINE_ERRORS:
            continue
        if isinstance(obj, dict) and obj.get("type") == "result":
            return obj
    return None


def _claude_failure_detail_text(result: subprocess.CompletedProcess[str]) -> str:
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    if stdout:
        try:
            payload = json.loads(stdout)
        except _JSON_LINE_ERRORS:
            payload = _last_result_line(stdout)  # #330: stream-json NDJSON stdout
        if isinstance(payload, dict):
            parts: list[str] = []
            if payload.get("api_error_status") is not None:
                parts.append(f"api_error_status={payload['api_error_status']}")
            if _assistant_error_code(stdout) == "model_not_found":  # #448: lives on the assistant frame
                parts.append("error=model_not_found")
            if payload.get("is_error") is not None:
                parts.append(f"is_error={payload['is_error']}")
            if payload.get("result"):
                parts.append(str(payload["result"]))
            if parts:
                if stderr:
                    parts.append(f"stderr={stderr}")
                return "; ".join(parts)
        if not stderr:
            return f"stdout={_stdout_shape(stdout) or stdout[:500]}"
    return stderr


class ClaudeCliProvider(LLMProvider):
    """Shells out to `claude -p <prompt> --output-format json`.

    Uses Hana's Claude Code subscription — no per-token API billing.
    Per the feedback memory: this is the default Claude path for
    companion-emergence and Hana's other projects.

    Chat surface note
    -----------------
    The Claude CLI (as of 2026-04) exposes only text-based input:
      - ``-p / --print`` for the user turn
      - ``--system-prompt`` for the system message
      - ``--input-format`` supports "text" (default) or "stream-json" but NOT
        a structured messages-array format.

    Therefore ClaudeCliProvider.chat() flattens the messages array into the
    text surface:
      - The first "system" message → ``--system-prompt``.
      - A single remaining user message is passed verbatim via ``-p``.
      - Multi-turn history is told as one line per message (``<time> Hana: ...``),
        not a ``User:`` / ``Assistant:`` script. That script made Claude more
        likely to continue the transcript and leak labels into Nell's reply, so
        each line leads with its local time and the companion is ``you``.

    Tool-calling via --mcp-config
    -----------------------------
    When ``tools`` is provided, the provider writes a temp mcp.json pointing
    at ``brain.mcp_server`` and passes ``--mcp-config <path>`` to the claude
    CLI.  Tool calls happen inside the claude subprocess; the provider returns
    the final assistant text directly — ``tool_calls`` on the ChatResponse is
    always empty.  Requires ``options["persona_dir"]`` so the spawned server
    knows which persona directory to load.

    When ``tools`` is None, the text ``-p`` path applies. Multi-turn history
    is told as narrative lines to avoid role-label leakage.
    """

    # The CLI registers the brain tools under the "brain-tools" MCP server key.
    tool_name_prefix = "mcp__brain-tools__"

    def __init__(
        self,
        tier: str = TIER_INTERACTIVE_CHAT,
        *,
        probe_model: str | None = None,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """``tier`` is the JOB this provider does (a name from ``brain.bridge.model_tier``, not a model id): the
        model is looked up from the user's tunables at every call, so a settings change reaches a provider that is
        already built. ``probe_model`` pins one id for the save-time probe of a newly chosen model only."""
        self._tier = tier
        self._probe_model = probe_model
        self._timeout = timeout_seconds

    def _cli_model(self) -> str:
        """The id passed to ``claude --model`` for the next call: the probe's pin, else the job's model right now."""
        return self._probe_model if self._probe_model is not None else model_for_tier(self._tier)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        persona_dir: Path | None = None,
    ) -> str:
        # #456 (D17): once the hard exit has begun, block (no spawn, no answer from a killed call).
        shutdown_progress.abandon_gate()
        # #456: a stopping bridge starts no new background call ("not now"; the caller keeps its work).
        if cli_throttle.shutdown_requested():
            raise cli_throttle.ShutdownDeferred("bridge is stopping")
        # #246: every background LLM call funnels through generate() (engines call it;
        # initiate's complete() delegates to it). While the login is expired, defer all
        # but one probe per interval so the autonomic layer stops spawning failing CLIs.
        if provider_auth.should_skip_background():
            raise provider_auth.ProviderAuthDeferred("provider auth expired — deferred")
        # Prompt piped via stdin; system prompt via --system-prompt-file.
        # Keeps both heavy payloads off argv (Windows CreateProcess 32,767-char
        # limit — WinError 206). See _system_prompt_tempfile for context.
        model = self._cli_model()
        cmd = [
            "claude",
            "-p",
            "--dangerously-skip-permissions",
            "--output-format",
            "json",
            "--model",
            model,
        ]
        _apply_lean_flags(cmd)  # #329: background calls get no tools at all
        with _system_prompt_tempfile(system) as sp_path:
            if sp_path is not None:
                cmd.extend(["--system-prompt-file", sp_path])

            try:
                try:
                    result = subprocess.run(
                        cmd,
                        input=prompt,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=self._timeout,
                        env=_subprocess_env(),
                        cwd=_claude_work_dir(),  # #122: memory-free working directory
                        check=False,
                        creationflags=_NO_WINDOW,
                        start_new_session=_OWN_SESSION,  # #456 D23: a terminal Ctrl-C reaches the bridge only
                    )
                finally:
                    shutdown_progress.abandon_gate()
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"ClaudeCliProvider: subprocess timed out after {self._timeout}s"
                ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"ClaudeCliProvider failed (exit {result.returncode}): "
                f"{_claude_failure_detail(result, model=_argv_model(cmd))}"
            )

        try:
            payload = json.loads(result.stdout)
            if not isinstance(payload, dict):
                raise ValueError("not a result object")
            log_usage(persona_dir, call_type="generate", model=model, frame=payload)
            # #119: the CLI can exit 0 and report the failure inside the frame,
            # so the returncode guard above never sees it. Without this the error
            # string was returned as the generation itself — and this is the
            # Haiku path, so the attunement detector then parsed "Not logged in
            # · Please run /login" as if it were its classification JSON.
            cli_err = _cli_error_detail(payload, model=_argv_model(cmd))
            if cli_err is not None:
                raise RuntimeError(f"ClaudeCliProvider: {cli_err}")
            return _text_only(payload["result"])
        except (*_JSON_LINE_ERRORS, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"ClaudeCliProvider: unexpected output format: {result.stdout[:200]!r}"
            ) from exc

    def probe(self, prompt: str) -> str:
        """One tiny tools-less call whose only job is to find out whether the model is usable (#448).

        Runs ``--output-format stream-json --verbose``, the output chat uses, because only that output carries
        the ``model_not_found`` code (on the assistant frame; the plain ``json`` output has just the status and
        the sentence). Parsed by the same functions the blocking chat path uses (``_run_blocking_cli`` /
        ``_cli_error_detail``), so a rejection is classified from the code. Writes no usage-log row, gives the
        model no tools, and honours the provider's timeout. Raises ``ProviderAuthDeferred`` while the login is
        expired (as ``generate`` does) and ``ProviderError`` for any failure, its text carrying the signals.
        """
        shutdown_progress.abandon_gate()  # #456 (D17)
        if cli_throttle.shutdown_requested():  # #456: no new background call while the bridge stops
            raise cli_throttle.ShutdownDeferred("bridge is stopping")
        if provider_auth.should_skip_background():
            raise provider_auth.ProviderAuthDeferred("provider auth expired — deferred")
        model = self._cli_model()
        cmd = [
            "claude",
            "-p",
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            model,
        ]
        _apply_lean_flags(cmd)  # #329: no tools at all
        result, payload = self._run_blocking_cli(cmd, prompt, {})
        if payload is None:
            raise ProviderError(
                "claude_cli_parse",
                f"unexpected output format: {_stdout_shape(result.stdout) or 'no result frame'}",
            )
        cli_err = _cli_error_detail(payload, model=_argv_model(cmd), stdout=result.stdout)
        if cli_err is not None:
            raise ProviderError("claude_cli_error", cli_err)
        return _text_only(payload.get("result"))  # #467: a null or non-string result is not text

    def name(self) -> str:
        return f"claude-cli:{self._cli_model()}"

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Flatten messages into Claude CLI; route through MCP when tools given.

        When ``tools`` is None:
          - Sends the current user turn, plus multi-turn narrative context when
            needed, via ``-p``.
          - Returns ChatResponse(content=<text>, tool_calls=()).

        When ``tools`` is provided:
          - Requires ``options["persona_dir"]`` to point the MCP server at the
            active persona.
          - Writes a temp mcp.json, calls claude with --mcp-config, returns
            the final assistant text. Tool calls happen inside the claude
            subprocess; tool_calls on the response is always empty.

        Raises
        ------
        ProviderError("claude_cli_timeout", ...)
        ProviderError("claude_cli_exit", ...)
        ProviderError("claude_cli_parse", ...)
        ProviderError("mcp_unavailable", ...)
            When tools is non-None and options["persona_dir"] is missing,
            or when the mcp SDK is not importable.
        ProviderError("claude_cli_setup", ...)
            When the temp config file write fails.
        """
        system_prompt: str | None = None
        conversation_messages: list[ChatMessage] = []
        for msg in messages:
            if msg.role == "system" and system_prompt is None:
                # content_text() flattens both legacy str content and the
                # tuple[ContentBlock, ...] multimodal shape uniformly.
                system_prompt = msg.content_text()
            else:
                conversation_messages.append(msg)

        # Single assembly path: the image transport is gone. Shared images are
        # surfaced to the model as a file path in the user text (engine step 5)
        # and read back via the read_file MCP tool, so no message carries image
        # pixels here — every turn flows through the text/MCP path below.
        volatile_suffix = (options or {}).get("volatile_suffix")
        include_block_clock = (options or {}).get("include_block_clock", True)
        flat_prompt = _format_claude_print_prompt(
            conversation_messages,
            volatile_suffix=volatile_suffix,
            include_block_clock=include_block_clock,
            user_name=(options or {}).get("user_name"),
        )
        _maybe_log_cache_debug(
            options, call_type="chat", system_prompt=system_prompt, volatile_suffix=volatile_suffix
        )

        if tools:
            persona_dir_str = (options or {}).get("persona_dir")
            if not persona_dir_str:
                raise ProviderError(
                    "mcp_unavailable",
                    "tool-calling via MCP requires options['persona_dir']",
                )
            return self._chat_with_mcp_tools(
                flat_prompt=flat_prompt,
                system_prompt=system_prompt,
                persona_dir=Path(persona_dir_str),
                session_id=(options or {}).get("session_id"),
            )

        # Text path for turns without image blocks. Multi-turn history is
        # told as narrative lines instead of a script with canonical
        # User:/Assistant: labels, because those labels can leak into replies.
        # Prompt → stdin, system_prompt → tempfile + --system-prompt-file
        # keeps both off argv (Windows CreateProcess 32,767-char limit).
        # #356: with a persona dir the CLI runs `stream-json --verbose`, so its built-in
        # tool calls (WebSearch, WebFetch) are visible in the output and can be audited.
        # Without one - run_tool_loop's forced final pass, which passes no persona_dir
        # on purpose - nothing could be audited, so the argv stays `json`.
        persona_dir_opt = (options or {}).get("persona_dir")
        audit_dir = Path(persona_dir_opt) if persona_dir_opt else None
        cmd = [
            "claude",
            "-p",
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json" if audit_dir else "json",
        ]
        if audit_dir:
            cmd.append("--verbose")
        model = self._cli_model()
        cmd.extend(["--model", model])
        cmd.extend(["--max-budget-usd", str(_MAX_TURN_BUDGET_USD(model))])
        _apply_lean_flags(cmd, builtins=_CHAT_BUILTIN_TOOLS)
        builtin_calls: dict[str, dict[str, Any]] = {}
        try:
            with _system_prompt_tempfile(system_prompt) as sp_path:
                if sp_path is not None:
                    cmd.extend(["--system-prompt-file", sp_path])

                result, payload = self._run_blocking_cli(cmd, flat_prompt, builtin_calls)

            try:
                if payload is None:
                    raise ValueError("no result frame")
                if _is_over_budget_frame(payload):
                    partial = _text_only(payload.get("result")).strip()
                    text = (
                        f"{partial}\n\n{_BUDGET_EXCEEDED_MSG}" if partial else _BUDGET_EXCEEDED_MSG
                    )
                    # #399: the over-budget turn is the costliest one; log it as the streaming path does.
                    log_usage(audit_dir, call_type="chat", model=model, frame=payload)
                    return ChatResponse(content=text, tool_calls=(), raw=payload)
                cli_err = _cli_error_detail(payload, model=_argv_model(cmd), stdout=result.stdout)
                if cli_err is not None:
                    raise ProviderError("claude_cli_error", cli_err)
                content = _text_only(payload["result"])
            except (ValueError, KeyError, TypeError) as exc:
                raise ProviderError(
                    "claude_cli_parse",
                    "unexpected output format: "
                    f"{_stdout_shape(result.stdout) or repr(result.stdout[:200])}",
                ) from exc

            log_usage(
                audit_dir,
                call_type="chat",
                model=model,
                frame=payload,
            )

            return ChatResponse(
                content=_truncate_at_role_leak(content),
                tool_calls=(),
                raw=None,
            )
        finally:
            _flush_builtin_audit(audit_dir, builtin_calls)

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> Iterator[ChatStreamEvent]:
        """Yield ChatStreamEvent values as the claude subprocess streams.

        Per-event idle timeout: _stream_per_event_idle_seconds() (default 60s,
        changeable via tunables key provider.stream_per_event_idle_seconds).
        First-event timeout: _stream_first_event_seconds() (default 120s,
        key provider.stream_first_event_seconds).
        Both can be monkeypatched in tests for fast verification.

        Yields:
          TextDelta    — one per streamed text chunk
          StreamDone   — when the result frame arrives (or EOF)
          StreamError  — on timeout, non-zero exit, or spawn failure

        Note: there is no image transport — shared images are surfaced as a
        file path in the user text and read back via the read_file MCP tool,
        so every turn is a text/MCP turn on this path.
        """
        system_prompt: str | None = None
        conversation_messages: list[ChatMessage] = []
        for msg in messages:
            if msg.role == "system" and system_prompt is None:
                system_prompt = msg.content_text()
            else:
                conversation_messages.append(msg)

        _pd = (options or {}).get("persona_dir")
        log_persona_dir: Path | None = Path(_pd) if _pd else None

        volatile_suffix = (options or {}).get("volatile_suffix")
        include_block_clock = (options or {}).get("include_block_clock", True)
        flat_prompt = _format_claude_print_prompt(
            conversation_messages,
            volatile_suffix=volatile_suffix,
            include_block_clock=include_block_clock,
            user_name=(options or {}).get("user_name"),
        )
        _maybe_log_cache_debug(
            options, call_type="chat_stream", system_prompt=system_prompt, volatile_suffix=volatile_suffix
        )

        model = self._cli_model()
        cmd = [
            "claude",
            "-p",
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            model,
        ]
        cmd.extend(["--max-budget-usd", str(_MAX_TURN_BUDGET_USD(model))])

        # MCP tools path — write a temp config and extend argv.
        tmp_mcp_path: str | None = None
        if tools:
            persona_dir_str = (options or {}).get("persona_dir")
            if not persona_dir_str:
                yield StreamError(
                    stage="mcp_unavailable",
                    detail="tool-calling via MCP requires options['persona_dir']",
                )
                return
            persona_dir_path = Path(persona_dir_str)
            try:
                from brain.tools import NELL_TOOL_NAMES  # local import avoids circular
            except ImportError as exc:
                yield StreamError(
                    stage="mcp_unavailable",
                    detail=f"brain.tools not importable: {exc}",
                )
                return
            allowed_mcp = [f"mcp__brain-tools__{n}" for n in NELL_TOOL_NAMES]
            # #80: this is the config the LIVE WS chat path actually uses (see
            # brain/bridge/server.py's _StreamingProxy, which calls chat_stream()
            # in preference to chat() whenever the real provider defines it) —
            # session_id must be threaded here too, not only in
            # _chat_with_mcp_tools, or a Kindled-invoked compact_history call
            # still can't resolve its session on the path real traffic runs.
            config = {
                "mcpServers": {
                    "brain-tools": brain_tools_mcp_entry(
                        persona_dir_path, session_id=(options or {}).get("session_id")
                    ),
                }
            }
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    delete=False,
                    encoding="utf-8",
                ) as tmp:
                    json.dump(config, tmp)
                    tmp_mcp_path = tmp.name
            except OSError as exc:
                yield StreamError(
                    stage="claude_cli_setup",
                    detail=f"failed to write temp mcp.json: {exc}",
                )
                return
            cmd.extend(["--mcp-config", tmp_mcp_path])
            cmd.extend(["--allowedTools", *allowed_mcp])
        # Always, not only with tools: a tools-less turn must not get every built-in (#329).
        _apply_lean_flags(cmd, builtins=_CHAT_BUILTIN_TOOLS)

        try:
            yield from self._run_chat_stream(
                cmd=cmd,
                flat_prompt=flat_prompt,
                system_prompt=system_prompt,
                persona_dir=log_persona_dir,
            )
        finally:
            if tmp_mcp_path is not None:
                try:
                    os.unlink(tmp_mcp_path)
                except OSError:
                    pass

    def _run_chat_stream(
        self,
        cmd: list[str],
        flat_prompt: str,
        system_prompt: str | None,
        persona_dir: Path | None = None,
    ) -> Iterator[ChatStreamEvent]:
        """Core Popen + queue-based streaming loop. The usage log names the model ``cmd`` was built with.

        Separated from chat_stream() so the generator's finally-block can
        clean up the MCP tempfile after the inner generator is exhausted
        (you can't have both yield and return in the same try/finally when
        the outer try manages non-generator resources).
        """
        q: _queue_mod.Queue[str | object] = _queue_mod.Queue()

        with _system_prompt_tempfile(system_prompt) as sp_path:
            if sp_path is not None:
                cmd = list(cmd) + ["--system-prompt-file", sp_path]

            shutdown_progress.abandon_gate()  # #456 (D17)
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=_subprocess_env(),
                    cwd=_claude_work_dir(),  # #122: memory-free working directory
                    creationflags=_NO_WINDOW,
                    start_new_session=_OWN_SESSION,  # #456 D23: a terminal Ctrl-C reaches the bridge only
                )
            except OSError as exc:
                yield StreamError(stage="claude_cli_spawn", detail=str(exc))
                return

            # Reader thread: drains stdout onto the queue; signals EOF with sentinel.
            def _reader(stdout=proc.stdout, out_q=q) -> None:  # type: ignore[assignment]
                try:
                    for line in stdout:
                        out_q.put(line)
                finally:
                    out_q.put(_STREAM_EOF)

            reader = threading.Thread(target=_reader, daemon=True)
            reader.start()
            # #482: stderr is read while the process runs, so a chatty CLI cannot fill the pipe and stall stdout.
            stderr_drain = _StderrDrain(proc.stderr) if proc.stderr is not None else None

            # Send prompt via stdin then close so the subprocess can start.
            try:
                assert proc.stdin is not None
                proc.stdin.write(flat_prompt)
                proc.stdin.close()
            except OSError:
                pass

            delta_chunks: list[str] = []
            assistant_snapshot: str | None = None
            assistant_error: str | None = None  # #448: last assistant frame's error code (model_not_found)
            done_emitted = False
            timeout = _stream_first_event_seconds()

            # Idle-timeout instrumentation (spec 2026-06-01). Recomputed per
            # frame so they reflect the last frame seen when a gap trips the
            # watchdog. Initialised for the cold-start case (timeout on first get).
            start = time.monotonic()
            last_frame_type: str | None = None
            tool_in_flight = False
            tool_name: str | None = None
            frames_seen = 0
            # #273: the CLI's built-in tool calls (WebSearch, WebFetch, ...),
            # keyed by tool_use id, written to the audit log once the stream ends.
            builtin_calls: dict[str, dict[str, Any]] = {}

            try:
                while True:
                    try:
                        item = q.get(timeout=timeout)
                    except _queue_mod.Empty:
                        shutdown_progress.abandon_gate()
                        proc.terminate()
                        _log_stream_timeout(
                            persona_dir,
                            {
                                "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                                "budget_s": timeout,
                                "last_frame_type": last_frame_type,
                                "tool_in_flight": tool_in_flight,
                                "tool_name": tool_name,
                                "frames_seen": frames_seen,
                                "elapsed_s": round(time.monotonic() - start, 3),
                            },
                        )
                        yield StreamError(
                            stage="claude_cli_idle_timeout",
                            detail=(
                                f"no stdout line after {timeout:.1f}s "
                                f"(last_frame={last_frame_type}, tool={tool_name})"
                            ),
                        )
                        return

                    shutdown_progress.abandon_gate()  # #456 (D17): no frame of a killed call is acted on
                    if item is _STREAM_EOF:
                        break

                    timeout = _stream_per_event_idle_seconds()

                    line = str(item).strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except _JSON_LINE_ERRORS:
                        continue
                    if not isinstance(obj, dict):
                        continue

                    t = obj.get("type")
                    last_frame_type = t
                    frames_seen += 1
                    tool_in_flight = False
                    tool_name = None
                    if t == "stream_event":
                        # A frame whose nested value is not an object carries no text: skip it, as the line-level
                        # non-object skip above does (#467). Only a string is text.
                        event = obj.get("event")
                        if isinstance(event, dict) and event.get("type") == "content_block_delta":
                            delta = event.get("delta")
                            if isinstance(delta, dict) and delta.get("type") == "text_delta":
                                text = delta.get("text")
                                if isinstance(text, str):
                                    delta_chunks.append(text)
                                    yield TextDelta(text=text)
                    elif t == "assistant":
                        _code = obj.get("error")
                        if isinstance(_code, str) and _ERROR_CODE_SHAPE.fullmatch(_code):
                            assistant_error = _code
                        # Snapshot frame — fallback content when result frame absent.
                        # None, not [], for a message that is not an object: a skipped frame must change nothing
                        # (an empty list would reset `assistant_snapshot` below). Same shape as the blocking scanner.
                        message = obj.get("message", {})
                        msg_content = message.get("content", []) if isinstance(message, dict) else None
                        _collect_builtin_calls(msg_content, builtin_calls)
                        if isinstance(msg_content, list):
                            for blk in msg_content:
                                if isinstance(blk, dict) and blk.get("type") == "tool_use":
                                    tool_in_flight = True
                                    tool_name = str(blk.get("name") or "") or None
                                    break
                            # Only string text counts: a text block whose `text` is null, a number or absent would
                            # make the join raise TypeError (same class as the frames above; `_tool_result_text` does
                            # this too). The snapshot is the fallback reply, so a frame whose text is empty ONLY
                            # because its text blocks were malformed leaves it as it was, like a malformed message;
                            # a genuinely valid message with no text blocks still replaces it.
                            parts: list[str] = []
                            dropped_malformed = False
                            for blk in msg_content:
                                if isinstance(blk, dict) and blk.get("type") == "text":
                                    if isinstance(blk.get("text"), str):
                                        parts.append(blk["text"])
                                    else:
                                        dropped_malformed = True
                            joined = "".join(parts)
                            if joined or not dropped_malformed:
                                assistant_snapshot = joined
                    elif t == "user":
                        _collect_builtin_results(obj, builtin_calls)
                    elif t == "result":
                        result_text = _text_only(obj.get("result", ""))
                        metadata: dict[str, Any] = {
                            k: obj[k]
                            for k in ("duration_ms", "num_turns", "stop_reason", "is_error")
                            if k in obj
                        }
                        log_usage(persona_dir, call_type="chat", model=_argv_model(cmd) or "", frame=obj)
                        # Over-budget catastrophe backstop: the CLI hit --max-budget-usd.
                        # Flush any partial text already streamed + append the cut-off note.
                        if _is_over_budget_frame(obj):
                            partial = "".join(delta_chunks).strip() or result_text.strip()
                            cutoff = (
                                f"{partial}\n\n{_BUDGET_EXCEEDED_MSG}"
                                if partial
                                else _BUDGET_EXCEEDED_MSG
                            )
                            yield StreamDone(content=cutoff, metadata=metadata)
                        else:
                            # #92: an exit-0 frame can still report a failure —
                            # surface it as an error rather than streaming the
                            # CLI's error text as her reply.
                            cli_err = _cli_error_detail(obj, error_code=assistant_error, model=_argv_model(cmd))
                            if cli_err is not None:
                                yield StreamError(stage="claude_cli_error", detail=cli_err)
                            else:
                                # #146: an exit-0 result frame can carry empty
                                # text. Fall back to what already streamed, as
                                # the EOF branch does, instead of a blank reply.
                                content = (
                                    result_text
                                    or "".join(delta_chunks)
                                    or (assistant_snapshot or "")
                                )
                                yield StreamDone(content=content, metadata=metadata)
                        done_emitted = True
                        # The result frame is terminal — stop here. Looping back
                        # to q.get would re-arm the idle-timeout watchdog and, if
                        # the subprocess is slow to close stdout (EOF), fire a
                        # spurious StreamError on top of the already-successful
                        # reply (bug-hunt finding #2). The reader thread still
                        # drains + reaps in the finally (reader.join).
                        break

                # EOF: emit terminal event if result frame never arrived.
                if not done_emitted:
                    try:
                        rc = proc.wait(timeout=_child_exit_grace_seconds())
                    except subprocess.TimeoutExpired:
                        # stdout closed but the process did not exit (#482): do not hang here; the finally reaps it.
                        yield StreamError(
                            stage="claude_cli_exit",
                            detail="the claude process did not exit after its output ended",
                        )
                        return
                    if rc != 0:
                        stderr_text = stderr_drain.text() if stderr_drain is not None else ""
                        # #246: the one exit!=0 site that bypasses _claude_failure_detail.
                        provider_auth.note_cli_failure(stderr_text)
                        if frames_seen == 0:  # #329: failed before any frame, e.g. rejected --tools
                            _note_tools_flag_rejected(stderr_text)
                        yield StreamError(
                            stage="claude_cli_exit",
                            detail=f"exit {rc}: {stderr_text[:200]}",
                        )
                    else:
                        content = "".join(delta_chunks) or (assistant_snapshot or "")
                        yield StreamDone(content=content, metadata={})

            except GeneratorExit:
                proc.terminate()
                raise
            finally:
                # Every exit reaps the child (#482); a finished one (result frame) gets a moment to leave on its own.
                _stop_child(proc, finished=done_emitted)
                reader.join(timeout=_child_exit_grace_seconds())
                # One flush for every exit: on success the proxy exhausts this
                # generator before it reads the audit log; on a StreamError it
                # raises out of its for-loop and CPython closes the generator
                # while the exception unwinds, so the rows are on disk first.
                _flush_builtin_audit(persona_dir, builtin_calls)

    def _run_blocking_cli(
        self, cmd: list[str], flat_prompt: str, builtin_calls: dict[str, dict[str, Any]]
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None]:
        """#456 (D17): no spawn once the hard exit has begun, and no answer from a call it killed."""
        shutdown_progress.abandon_gate()
        try:
            return self._run_blocking_cli_inner(cmd, flat_prompt, builtin_calls)
        finally:
            shutdown_progress.abandon_gate()

    def _run_blocking_cli_inner(
        self, cmd: list[str], flat_prompt: str, builtin_calls: dict[str, dict[str, Any]]
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None]:
        """Run `claude` to completion for the two blocking `chat()` branches.

        Scans stdout for the CLI's built-in tool calls (#273/#330) BEFORE any raise, so
        the caller's `finally` can flush them after an exit != 0 and, for a timeout
        (#380), from the partial stdout, which is all there is. Returns the process and
        the terminal `result` frame (None if the stream had none).
        """
        try:
            result = subprocess.run(
                cmd,
                input=flat_prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout,
                env=_subprocess_env(),
                cwd=_claude_work_dir(),  # #122: memory-free working directory
                check=False,
                creationflags=_NO_WINDOW,
                start_new_session=_OWN_SESSION,  # #456 D23: a terminal Ctrl-C reaches the bridge only
            )
        except subprocess.TimeoutExpired as exc:
            try:
                _scan_blocking_stream_stdout(_partial_stdout_text(exc.stdout), builtin_calls)
            except Exception as scan_exc:  # noqa: BLE001 — audit must never mask the timeout
                logger.warning("built-in tool audit scan of a timed-out turn failed: %s", scan_exc)
            raise ProviderError(
                "claude_cli_timeout",
                f"subprocess timed out after {self._timeout}s",
            ) from exc

        # Scan before the exit-code check so a non-zero exit still flushes the
        # built-in calls it saw.
        payload = _scan_blocking_stream_stdout(result.stdout, builtin_calls)

        # #399: the real CLI exits 1 on a --max-budget-usd overrun (both output formats), with a
        # result frame saying so. That is the graceful path's input, not a failure: the callers
        # turn the frame into the budget reply. Keyed on the subtype alone - every real frame
        # carries it - so no other non-zero exit is swallowed.
        over_budget = payload is not None and payload.get("subtype") == _BUDGET_SUBTYPE
        if result.returncode != 0 and not over_budget:
            raise ProviderError(
                "claude_cli_exit",
                f"exit {result.returncode}: {_claude_failure_detail(result, model=_argv_model(cmd))}",
            )
        return result, payload

    def _chat_with_mcp_tools(
        self,
        flat_prompt: str,
        system_prompt: str | None,
        persona_dir: Path,
        session_id: str | None = None,
    ) -> ChatResponse:
        """Tool-calling path: claude with --mcp-config pointing at brain.mcp_server.

        The mcp SDK is only imported here — keeps the legacy text path
        usable on systems without the SDK installed.

        Telemetry: snapshots the persona's audit log size before invoking
        the claude subprocess; reads any newly-appended lines afterward
        and surfaces them as ChatResponse.dispatched_invocations. Per-
        session /chat is serialized via in_flight_locks so no other
        writer interleaves into the snapshot window. Tools dispatched
        here have ALREADY run inside the subprocess — run_tool_loop
        must not re-dispatch them.

        ``session_id``, when given, is threaded into the spawned MCP
        subprocess's env (NELL_MCP_SESSION_ID) so a ``_PROVIDER_TOOLS`` tool
        (currently only ``compact_history``, #80) can resolve it on that side
        of the process boundary — see ``brain_tools_mcp_entry``.
        """
        try:
            import mcp  # noqa: F401
        except ImportError as exc:
            raise ProviderError(
                "mcp_unavailable",
                "the 'mcp' SDK is required for the Claude tool-calling path. "
                "pip install 'mcp>=1.0.0,<2.0.0'",
            ) from exc

        request_id = uuid.uuid4().hex
        config = {
            "mcpServers": {
                "brain-tools": brain_tools_mcp_entry(
                    persona_dir, request_id=request_id, session_id=session_id
                ),
            }
        }

        # Snapshot audit log size BEFORE subprocess so we can read only
        # the lines this subprocess appends.
        audit_log_path = persona_dir / "tool_invocations.log.jsonl"
        try:
            audit_offset_before = audit_log_path.stat().st_size
        except FileNotFoundError:
            audit_offset_before = 0

        tmp_path: str | None = None
        # #330: the CLI's built-in tool calls, written to the audit log in the
        # `finally` below (one flush for every exit that got as far as stdout).
        builtin_calls: dict[str, dict[str, Any]] = {}
        try:
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    delete=False,
                    encoding="utf-8",
                ) as tmp:
                    json.dump(config, tmp)
                    tmp_path = tmp.name
            except OSError as exc:
                raise ProviderError(
                    "claude_cli_setup",
                    f"failed to write temp mcp.json: {exc}",
                ) from exc

            # Build the list of allowed MCP tool names for --allowedTools.
            # Claude CLI blocks MCP tool calls in non-interactive (-p) mode
            # unless each tool is explicitly pre-approved.  The MCP server
            # name in mcp.json is "brain-tools", so Claude registers tools
            # as "mcp__brain-tools__<name>". We allow all registered
            # brain-tools so the LLM can call them without a permission
            # prompt.
            from brain.tools import NELL_TOOL_NAMES  # local import — avoids circular

            allowed_mcp = [f"mcp__brain-tools__{n}" for n in NELL_TOOL_NAMES]
            # flat_prompt → stdin, system_prompt → tempfile + --system-prompt-file.
            # Off-argv on both keeps the cmd under Windows' 32,767-char limit
            # even after a long session has grown voice.md + buffer into the
            # tens of KB. This is the exact code path that surfaced WinError
            # 206 in the field.
            model = self._cli_model()
            cmd = [
                "claude",
                "-p",
                "--dangerously-skip-permissions",
                "--output-format",
                "stream-json",
                "--verbose",
                "--model",
                model,
            ]
            cmd.extend(["--mcp-config", tmp_path])
            cmd.extend(["--allowedTools", *allowed_mcp])
            _apply_lean_flags(cmd, builtins=_CHAT_BUILTIN_TOOLS)
            cmd.extend(["--max-budget-usd", str(_MAX_TURN_BUDGET_USD(model))])

            with _system_prompt_tempfile(system_prompt) as sp_path:
                if sp_path is not None:
                    cmd.extend(["--system-prompt-file", sp_path])
                result, payload = self._run_blocking_cli(cmd, flat_prompt, builtin_calls)

            try:
                if payload is None:
                    raise ValueError("no result frame")
                if _is_over_budget_frame(payload):
                    partial = _text_only(payload.get("result")).strip()
                    text = (
                        f"{partial}\n\n{_BUDGET_EXCEEDED_MSG}" if partial else _BUDGET_EXCEEDED_MSG
                    )
                    # #399: the over-budget turn is the costliest one; log it as the streaming path does.
                    log_usage(persona_dir, call_type="chat", model=model, frame=payload)
                    # The tools that ran before the stop are real (the budget is the backstop for
                    # runaway tool loops): keep their records, as the success path does.
                    return ChatResponse(
                        content=text,
                        tool_calls=(),
                        dispatched_invocations=tuple(
                            _read_audit_lines_since(
                                audit_log_path, audit_offset_before, request_id=request_id
                            )
                        ),
                        raw=payload,
                    )
                cli_err = _cli_error_detail(payload, model=_argv_model(cmd), stdout=result.stdout)
                if cli_err is not None:
                    raise ProviderError("claude_cli_error", cli_err)
                content = _text_only(payload["result"])
            except (ValueError, KeyError, TypeError) as exc:
                raise ProviderError(
                    "claude_cli_parse",
                    "unexpected output format: "
                    f"{_stdout_shape(result.stdout) or repr(result.stdout[:200])}",
                ) from exc

            # Log per-call token usage (incl. cache_creation/cache_read) like the
            # other CLI paths — this is the production tool-bearing chat path, so
            # without this chat_usage.jsonl misses the bulk of real chat turns.
            log_usage(
                persona_dir,
                call_type="chat",
                model=model,
                frame=payload,
            )

            dispatched = _read_audit_lines_since(
                audit_log_path,
                audit_offset_before,
                request_id=request_id,
            )
            return ChatResponse(
                content=_truncate_at_role_leak(content),
                tool_calls=(),
                dispatched_invocations=tuple(dispatched),
                raw=None,
            )
        finally:
            # After the `dispatched` read above, so this call's own built-in rows
            # can never become its dispatched_invocations (audit-only, #273).
            _flush_builtin_audit(persona_dir, builtin_calls)
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


_CLI_BUILTIN_ORIGIN = "cli_builtin"
_BRAIN_TOOLS_PREFIX = ClaudeCliProvider.tool_name_prefix


def _collect_builtin_calls(content: Any, calls: dict[str, dict[str, Any]]) -> None:
    """Record each CLI built-in `tool_use` block (#273).

    A brain-tools call (`mcp__brain-tools__*`) is skipped: the MCP child logs
    it. Anything else ran inside the `claude` subprocess and is invisible to
    the MCP child: WebSearch, WebFetch, ToolSearch and whatever other built-in
    the CLI exposes (owner ruling: every built-in, no allowlist). With
    --strict-mcp-config only brain-tools is loaded, so a non-brain MCP tool
    here would also be labelled a built-in. The first block for an id wins.
    """
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        tool_id, name = block.get("id"), block.get("name")
        if not isinstance(tool_id, str) or not isinstance(name, str) or not name:
            continue
        if name.startswith(_BRAIN_TOOLS_PREFIX) or tool_id in calls:
            continue
        tool_input = block.get("input")
        calls[tool_id] = {
            "name": name,
            "arguments": tool_input if isinstance(tool_input, dict) else {},
            "result": "",
            "is_error": False,
            "answered": False,
        }


def _collect_builtin_results(frame: dict[str, Any], calls: dict[str, dict[str, Any]]) -> None:
    """Attach a `user` frame's `tool_result` blocks to their recorded calls."""
    message = frame.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        tool_use_id = block.get("tool_use_id")
        call = calls.get(tool_use_id) if isinstance(tool_use_id, str) else None
        if call is None:
            continue
        call["result"] = _tool_result_text(block.get("content"))
        call["is_error"] = bool(block.get("is_error"))
        call["answered"] = True


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b["text"]
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
            else json.dumps(b, default=str, ensure_ascii=False)
            for b in content
        )
    return json.dumps(content, default=str, ensure_ascii=False)


def _partial_stdout_text(raw: bytes | str | None) -> str:
    """`TimeoutExpired.stdout` as text (#380).

    Even in text mode it is `bytes` on POSIX (the partial read taken before the kill),
    `str` on Windows (re-read after the kill), and `None` when nothing arrived. Total:
    never raises.
    """
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw if isinstance(raw, str) else ""


# The CLI's own `type`/`subtype` words. 40 covers its longest (`error_max_structured_output_retries`,
# 35, in 2.1.300) with margin; a cap picked against the subtypes first seen turned
# `error_during_execution` (22) into `?`. The charset, not the length, is what keeps text out.
_FRAME_KIND_RE = re.compile(r"[A-Za-z0-9_.-]{1,40}")
_SHAPE_MAX_KINDS = 8


def _frame_kind_part(value: Any) -> str:
    """A frame's `type` or `subtype` as it may be printed: only the CLI's own short
    vocabulary (<= 40 chars of [A-Za-z0-9_.-]); anything else is the fixed `?`."""
    return value if isinstance(value, str) and _FRAME_KIND_RE.fullmatch(value) else "?"


def _stdout_shape(stdout: str | None) -> str | None:
    """What a stream-json stdout held, as frame kinds and counts, never its text (#358).

    Under `--verbose` the head of stdout is the CLI's configured hook output, so an
    error detail built from "the first N characters" would carry whatever the user's
    hooks print into the bridge logs. None when there is no frame, so callers keep
    their raw snippet for plain-text and untyped-JSON stdout.
    """
    counts: dict[str, int] = {}
    last = ""
    other = 0
    for raw in (stdout or "").split("\n"):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except Exception:  # noqa: BLE001 — an error path must not raise (RecursionError on deep nesting)
            obj = None
        if isinstance(obj, dict) and "type" in obj:
            last = _frame_kind_part(obj["type"])
            if obj.get("subtype"):
                last = f"{last}/{_frame_kind_part(obj['subtype'])}"
            counts[last] = counts.get(last, 0) + 1
        else:
            other += 1
    if not counts:
        return None
    listed = list(counts.items())
    kinds = ", ".join(f"{kind} x{n}" for kind, n in listed[:_SHAPE_MAX_KINDS])
    if len(listed) > _SHAPE_MAX_KINDS:
        kinds += f", +{len(listed) - _SHAPE_MAX_KINDS} more"
    tail = f"; {other} other line(s)" if other else ""
    return f"{sum(counts.values())} frame(s), last={last}: {kinds}{tail}"


def _scan_blocking_stream_stdout(
    stdout: str | None, calls: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """Scan a blocking `claude -p --output-format stream-json --verbose` stdout (#330).

    Feeds `assistant`/`user` frames to the #273 built-in collectors and returns
    the terminal `result` frame (the same object `--output-format json` printed),
    or None. Lines split on "\n" only: `str.splitlines()` also breaks on
    U+2028/U+2029/U+0085, which JSON serialisers emit raw inside strings.
    """
    result: dict[str, Any] | None = None
    for raw in (stdout or "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except _JSON_LINE_ERRORS:
            continue
        if not isinstance(obj, dict):
            continue
        kind = obj.get("type")
        if kind == "assistant":
            message = obj.get("message")
            _collect_builtin_calls(message.get("content") if isinstance(message, dict) else None, calls)
        elif kind == "user":
            _collect_builtin_results(obj, calls)
        elif kind == "result":
            result = obj
    if result is None:
        # ponytail: single-object shape (what `--output-format json` printed). Kept so
        # stubs and any CLI that ignores the format flag still parse; delete once migrated.
        # A typed non-result frame (e.g. a lone `init`) must NOT become the payload:
        # _cli_error_detail would call provider_auth.note_cli_success() on it.
        try:
            whole = json.loads(stdout or "")
        except _JSON_LINE_ERRORS:
            return None
        if isinstance(whole, dict) and whole.get("type") in (None, "result"):
            return whole
    return result


def _flush_builtin_audit(persona_dir: Path | None, calls: dict[str, dict[str, Any]]) -> None:
    """Write the turn's built-in calls to the audit log, marked `cli_builtin`.

    Runs once, after the stream. By then this turn's MCP child has written its
    own rows (it logs before returning each result), so the two writers don't
    interleave on the success path. Rows are stamped at flush time and follow
    the turn's MCP rows. Observability only: never raises.
    """
    if persona_dir is None or not calls:
        return
    try:
        from brain.mcp_server import audit

        for call in calls.values():
            if not call["answered"]:
                outcome, error = "error", "no result before the stream ended"
            elif call["is_error"]:
                outcome, error = "error", "tool reported an error"
            else:
                outcome, error = "ok", None
            audit.log_invocation(
                persona_dir,
                name=call["name"],
                arguments=call["arguments"],
                # JSON so redacted mode can hide it: a search result starts with the query.
                result_summary=json.dumps({"result": call["result"]}, ensure_ascii=False),
                error=error,
                outcome=outcome,
                origin=_CLI_BUILTIN_ORIGIN,
            )
    except Exception as exc:  # noqa: BLE001 — audit must never break a turn
        logger.warning("built-in tool audit write failed: %s", exc)


def _read_audit_lines_since(
    audit_log_path: Path,
    offset: int,
    *,
    request_id: str | None = None,
) -> list[dict[str, Any]]:
    """Read audit log lines newly appended after `offset` bytes.

    Returns a list of invocation records in the engine's tool_invocations
    shape: ``{name, arguments, result_summary, error?, outcome?}``. Malformed
    lines are skipped silently — telemetry should never break a chat response.
    """
    try:
        # Rotation guard (bug #5): audit._rotate_if_needed renames the log to
        # `.1` and starts a fresh (empty) file once it exceeds ~1MB. If that
        # happens mid-turn, the current file is SMALLER than the pre-turn
        # `offset`, so a plain seek(offset) lands past EOF and drops the whole
        # turn's invocations. Detect the shrink and stitch: the `.1` tail from
        # `offset` (entries written before the rotation) + the entire new file
        # (entries written after). Single-rotation recovery — a turn writing
        # >1MB of audit (multiple rotations) is pathological and out of scope.
        try:
            current_size = audit_log_path.stat().st_size
        except OSError:
            current_size = 0
        if current_size < offset:
            new_bytes = b""
            backup = audit_log_path.with_name(f"{audit_log_path.name}.1")
            try:
                with backup.open("rb") as fh:
                    fh.seek(min(offset, backup.stat().st_size))
                    new_bytes += fh.read()
            except (FileNotFoundError, OSError):
                pass
            try:
                with audit_log_path.open("rb") as fh:
                    new_bytes += fh.read()
            except FileNotFoundError:
                pass
        else:
            with audit_log_path.open("rb") as fh:
                fh.seek(offset)
                new_bytes = fh.read()
    except FileNotFoundError:
        return []
    except OSError as exc:
        logger.warning("MCP audit telemetry read failed for %s: %s", audit_log_path, exc)
        return []

    records: list[dict[str, Any]] = []
    malformed = 0
    for raw_line in new_bytes.splitlines():
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except _JSON_LINE_ERRORS:
            malformed += 1
            continue
        if not isinstance(entry, dict):
            malformed += 1
            continue
        if entry.get("origin") == _CLI_BUILTIN_ORIGIN:
            # #273: audit-only. The CLI already ran these; they are not the
            # turn's tool invocations.
            continue
        if request_id is not None:
            entry_request_id = entry.get("request_id")
            # Records produced by current code carry request_id and can be
            # strictly correlated. Legacy no-id records are still admitted so
            # old tests/logs remain readable, but records from another current
            # request are filtered out.
            if entry_request_id not in (None, request_id):
                continue
        record: dict[str, Any] = {
            "name": entry.get("name", "?"),
            "arguments": entry.get("arguments", {}),
            "result_summary": entry.get("result_summary", ""),
        }
        if entry.get("error"):
            record["error"] = entry["error"]
        if entry.get("outcome"):
            record["outcome"] = entry["outcome"]
        if entry.get("monologue_text"):
            record["monologue_text"] = entry["monologue_text"]
        # Content-addressed image path (a hash, never base64) — surfaced so the
        # engine can persist it into the durable buffer for memory binding.
        if entry.get("stored_image_path"):
            record["stored_image_path"] = entry["stored_image_path"]
        records.append(record)
    if malformed:
        logger.warning(
            "MCP audit telemetry skipped %d malformed line(s) from %s",
            malformed,
            audit_log_path,
        )
    return records


# ---------------------------------------------------------------------------
# ClaudeCliProvider — prompt/context formatting helpers
# ---------------------------------------------------------------------------


# Lines beginning with a known role label followed by ': ' indicate the
# model has stopped replying and started scripting the next turn — a
# common failure on creative scenes where the model treats its reply
# as multi-turn fiction. We truncate at the first such marker.
#
# Conservative: only matches at the start of a line (after \n) and only
# the canonical role-label set. Lowercase variants included because the
# model occasionally emits 'user:' (Nell's voice is lowercase-leaning).
# We do NOT include persona-specific names like 'Hana:' / 'Nell:' here
# because Hana legitimately quotes those in fiction.
_ROLE_LEAK_PATTERN = re.compile(
    r"\n(?:User|user|Human|human|Assistant|assistant):\s",
)


def _truncate_at_role_leak(text: str) -> str:
    """Strip everything from the first role-label line onward.

    Idempotent and safe: when no leak is present, returns ``text``
    unchanged. When the model overruns into a hypothetical multi-turn
    script, returns just the prefix up to (but not including) the
    overrun. The trailing newline before the cut is preserved so the
    reply still ends on a clean line.
    """
    match = _ROLE_LEAK_PATTERN.search(text)
    if match is None:
        return text
    return text[: match.start()].rstrip()


def _maybe_log_cache_debug(
    options: dict[str, Any] | None,
    *,
    call_type: str,
    system_prompt: str | None,
    volatile_suffix: str | None,
) -> None:
    """Append one cache-debug row per chat call when ``NELL_CACHE_DEBUG=1``.

    Instrumentation for the prompt-caching work: the live, in-situ probe of whether
    the ``--system-prompt-file`` content is byte-identical across consecutive
    same-session turns (count distinct ``system_sha256``). Off by default — no
    production cost — and fully fail-soft: a debug-log failure must never affect a
    chat turn. Writes ``cache_debug.jsonl`` next to the persona's other logs.

    ``volatile_suffix`` is recorded for the later Option A+ work (volatile pushed to
    the stdin tail); it is ``None`` until that lands, so ``volatile_present`` reads
    False today. See ``docs/guarded-change/prompt-caching-adopt/``.

    Enabled by EITHER ``NELL_CACHE_DEBUG=1`` in the environment OR a marker file
    ``<persona_dir>/cache_debug.on``. The file trigger exists because env vars
    don't reliably survive a GUI launch (e.g. nellface starting the bridge), so
    ``touch <persona_dir>/cache_debug.on`` turns logging on regardless of how the
    app was started; delete it to turn off.
    """
    pd = (options or {}).get("persona_dir")
    if not pd:
        return
    enabled = os.environ.get("NELL_CACHE_DEBUG") == "1"
    if not enabled:
        try:
            enabled = (Path(pd) / "cache_debug.on").exists()
        except Exception:  # noqa: BLE001
            enabled = False
    if not enabled:
        return
    try:
        sys_text = system_prompt or ""
        sys_bytes = sys_text.encode("utf-8")
        vol_bytes = (volatile_suffix or "").encode("utf-8")
        row = {
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "session_id": (options or {}).get("session_id"),
            "call_type": call_type,
            "system_sha256": hashlib.sha256(sys_bytes).hexdigest(),
            "system_char_len": len(sys_text),
            "system_tok_est": len(sys_bytes) // 4,
            "volatile_present": bool(volatile_suffix),
            "volatile_tok_est": len(vol_bytes) // 4,
        }
        path = Path(pd) / "cache_debug.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — debug logging must never break a turn
        pass


def _format_claude_print_prompt(
    messages: list[ChatMessage],
    *,
    volatile_suffix: str | None = None,
    include_block_clock: bool = True,
    user_name: str | None = None,
) -> str:
    """Format messages for ``claude -p`` without transcript role labels.

    A single text turn stays verbatim to preserve the simple generation shape.
    Multi-turn history is told as one line per message (``<time> Hana: ...``),
    one per message. The oldest shape, a ``User:`` / ``Assistant:`` script, primed
    Claude to continue that transcript and leaked role labels into Nell's visible
    reply; the JSONL shape that replaced it was sometimes completed onto a reply
    as ``", "ts": "..."}`` (#227). ``user_name`` is the configured name of the
    person Nell is talking to (``User`` when absent).

    ``volatile_suffix`` (Option A+) is the per-turn ambient context chunk. When
    provided it is appended AFTER the rendered context block in BOTH branches —
    so it is never dropped on a session's first turn (which hits the single-
    message branch) — placing every changing byte at the very end of the stdin
    prompt. ``include_block_clock=False`` drops the relocated clock line from the
    top of the context block (the suffix carries it instead). Both default to the
    pre-change behaviour, so non-chat callers are unaffected.
    """
    if not messages:
        return ""
    if len(messages) == 1:
        base = messages[0].content_text()
    else:
        base = _format_claude_context_block(
            messages,
            includes_latest_user=True,
            include_block_clock=include_block_clock,
            user_name=user_name,
        )
    if volatile_suffix:
        return f"{base}\n\n{volatile_suffix}"
    return base


def _format_claude_context_block(
    messages: list[ChatMessage],
    *,
    includes_latest_user: bool,
    now: datetime | None = None,
    include_block_clock: bool = True,
    user_name: str | None = None,
) -> str:
    """Return the narrative context block for Claude CLI prompt/system text.

    One physical line per message (see ``_claude_context_lines``). The wording
    intentionally avoids the canonical ``User:`` and ``Assistant:`` delimiters,
    and each message's text is a JSON-string literal, so embedded newlines or
    user-supplied delimiter-looking text stay inside the line instead of
    creating new transcript lines.

    ``now`` is an optional test seam — when None the real UTC clock is used
    internally, then converted to local wall-clock time for display (issue
    #217: the companion reads its own local time here, not UTC).

    ``include_block_clock`` (default True, the pre-change behaviour) controls the
    block-level ``Current time:`` anchor + its elapsed-time explainer at the top
    of the block. The chat path passes False because that anchor moves to the
    Option A+ volatile suffix (it was the one per-call-changing byte sitting
    above the whole transcript, busting the history prefix); the image path and
    any other caller keep it. Per-message times are unaffected by this toggle
    either way, but are independently converted to local time for display in
    ``_claude_context_lines`` (stored/passed-through values stay UTC; only the
    rendered line is localized).
    """
    if includes_latest_user:
        instruction = (
            "Answer the final human entry directly in the companion's voice. "
            "Do not prefix the reply with any speaker name, role label, JSON, "
            "or transcript marker."
        )
    else:
        instruction = (
            "Use this only as context for the next human message. Do not "
            "prefix the reply with any speaker name, role label, JSON, or "
            "transcript marker."
        )

    lines = [
        "What has been said so far is below, one line per message, with the local time "
        "it was said where that is known. It is context, not a transcript to continue.",
    ]
    if include_block_clock:
        # tz-local-display: block-level "Current time:" anchor (image path etc.)
        now_iso = format_local(now or datetime.now(UTC))
        lines.append(f"Current time: {now_iso}.")
        lines.append(
            "Each line's time (when present) is the wall-clock time of that message; "
            "use it to compute how much time has passed."
        )
    lines.append(instruction)
    lines.append("")
    lines.extend(_claude_context_lines(messages, user_name=user_name))
    return "\n".join(lines)


_NARRATIVE_NAME_MAX = 40
_NARRATIVE_LINE_BREAKS = {"\u2028": "\\u2028", "\u2029": "\\u2029", "\u0085": "\\u0085"}
# Every character that can end a line for ``str.splitlines()`` or a terminal, plus the other C0 controls and DEL.
# Tab is left alone: it does not start a new line and escaping it only costs readability.
_NARRATIVE_ESCAPE = re.compile("[\x00-\x08\x0a-\x1f\x7f\x85\u2028\u2029]")
_NARRATIVE_NAMED_ESCAPES = {"\n": "\\n", "\r": "\\r"}


def _one_line(text: str) -> str:
    """Escape the Unicode line breaks ``json.dumps(ensure_ascii=False)`` leaves raw."""
    for ch, escaped in _NARRATIVE_LINE_BREAKS.items():
        text = text.replace(ch, escaped)
    return text


def _narrative_literal(text: str) -> str:
    """``text`` as a JSON string literal that can never span two physical lines."""
    return _one_line(json.dumps(text, ensure_ascii=False))


def _narrative_text(text: str) -> str:
    """``text`` with every line break and control character escaped, so it can never span two physical lines.

    Quotes and backslashes are left as written. # ponytail: that makes the escape lossy (a typed ``\\n`` and a
    real line break read the same); the point is that a message cannot forge a line, and the model does not decode
    it. Escape the backslash too if anything ever needs to parse the line back.
    """
    return _NARRATIVE_ESCAPE.sub(
        lambda m: _NARRATIVE_NAMED_ESCAPES.get(m.group(), f"\\u{ord(m.group()):04x}"), text
    )


def _narrative_name(raw: Any) -> str:
    """A speaker name safe to put in a line: no control or line-break characters, bounded.

    Whitespace of any kind becomes one space first, so a tab inside a name keeps two
    words; the control, format and separator characters that remain are dropped.
    May return "".
    """
    if not isinstance(raw, str):
        return ""
    spaced = "".join(" " if ch.isspace() else ch for ch in raw)
    kept = "".join(
        ch
        for ch in spaced
        if not unicodedata.category(ch).startswith("C")
        and unicodedata.category(ch) not in ("Zl", "Zp")
    )
    return " ".join(kept.split())[:_NARRATIVE_NAME_MAX].strip()


def _narrative_time(ts: str | None) -> str:
    """Local display time for a stored ``ts``, or "" when absent or unparseable.

    The raw string is never echoed: an unparseable value could carry a line break.
    """
    if not ts or not isinstance(ts, str):
        return ""
    try:
        # tz-local-display: per-message time in the narrative chat context. `local_display` is total and would return the
        # raw text on failure; this caller never echoes an unparseable value, so it renders directly and returns "" itself.
        return format_local(parse_iso_utc(ts))
    except Exception:  # noqa: BLE001 — unparseable, or no local equivalent (year 1 in a zone behind UTC)
        return ""


def _claude_context_lines(messages: list[ChatMessage], *, user_name: str | None = None) -> Iterator[str]:
    """Yield one line per chat turn: ``<time> <who>: <text>`` (#227).

    The time is the stored UTC ``ts`` rendered as local wall-clock time (issue #217);
    ``msg.ts`` itself is untouched and a missing or unparseable ``ts`` drops the clause.
    ``<who>`` is the configured user name (``User`` without one, so it cannot be mistaken
    for a person called "Human") for ``user`` messages and ``you`` for the companion's own.
    Line breaks inside the text are escaped, so nothing in a message, a name, an id or a
    ``ts`` can add, remove or reorder lines.
    """
    name = _narrative_name(user_name) or "User"
    for msg in messages:
        when = _narrative_time(msg.ts)
        text = _narrative_text(msg.content_text())
        if msg.role == "user":
            body = f"{name}: {text}"
        elif msg.role == "assistant":
            body = f"you: {text}"
            for tc in msg.tool_calls:
                args = _one_line(json.dumps(tc.arguments, ensure_ascii=False, default=str))
                body += (
                    f"; it called {_narrative_name(tc.name) or 'a tool'} with {args}"
                    f" (call id {_narrative_literal(tc.id)})"
                )
        elif msg.role == "tool":
            what = (
                f"The result of tool call {_narrative_literal(msg.tool_call_id)} came back"
                if msg.tool_call_id
                else "The result of a tool call came back"
            )
            body = f"{what}: {text}"
        else:
            yield f"A {_narrative_name(msg.role) or 'system'} note: {text}"
            continue
        yield f"{when} {body}" if when else body


# ---------------------------------------------------------------------------
# OllamaProvider — httpx-based, full tool-call support
# ---------------------------------------------------------------------------


class OllamaProvider(LLMProvider):
    """Local Ollama integration over httpx.

    Full port of OG NellBrain's OllamaProvider (nell_bridge_providers.py).

    Default model: huihui_ai/qwen2.5-abliterated:7b — uncensored Qwen2.5
    abliterated variant.  Same base architecture as Nell's nell-dpo fine-tune,
    light enough for most local hardware (~4.7GB), works well with the brain's
    emotional + creative tone.  Users can override via OllamaProvider(model="…").

    Streaming
    ---------
    ``chat_stream()`` POSTs with ``stream=True`` and yields each content
    chunk as Ollama emits it. The bridge WS handler (or any other
    streaming consumer) can pipe these straight to the client without
    waiting for the full reply. Tool-calling is intentionally not on
    the streaming path — pass ``tools=`` through :meth:`chat` instead,
    which returns a structured ChatResponse with parsed tool_calls.
    """

    def __init__(
        self,
        model: str = "huihui_ai/qwen2.5-abliterated:7b",
        host: str = "http://localhost:11434",
        timeout: float = 300.0,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout

    def name(self) -> str:
        return f"ollama:{self._model}"

    def healthy(self) -> bool:
        """GET /api/tags — True if Ollama is reachable and responds 200."""
        try:
            r = httpx.get(f"{self._host}/api/tags", timeout=5.0)
            return r.status_code == 200
        except httpx.RequestError:
            return False

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """POST to /api/chat and parse the structured response.

        Raises
        ------
        ProviderError("ollama_http", ...)
            HTTP error response from Ollama (4xx / 5xx).
        ProviderError("ollama_request", ...)
            Network-level failure (connection refused, DNS, etc.).
        ProviderError("ollama_parse", ...)
            Response body is not valid JSON or missing expected fields.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if options:
            generation_options = {
                key: value
                for key, value in options.items()
                if key not in _PROVIDER_CONTEXT_OPTION_KEYS
            }
            if generation_options:
                payload["options"] = generation_options

        url = f"{self._host}/api/chat"
        try:
            resp = httpx.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                "ollama_http",
                f"{exc.response.status_code}: {exc.response.text[:200]}",
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderError("ollama_request", str(exc)) from exc

        try:
            data = resp.json()
        except _JSON_LINE_ERRORS as exc:
            raise ProviderError("ollama_parse", f"invalid json: {exc}") from exc

        # A body or a nested value of the wrong shape is a parse error, not an AttributeError/TypeError out of
        # chat() (#468). A falsy `message` (null, "", 0, [], false) stays "an empty message", as it always was.
        if not isinstance(data, dict):
            raise ProviderError("ollama_parse", f"response is not a JSON object: {type(data).__name__}")
        msg = data.get("message") or {}
        if not isinstance(msg, dict):
            raise ProviderError("ollama_parse", f"message is not an object: {type(msg).__name__}")
        content = msg.get("content", "") or ""
        if not isinstance(content, str):
            raise ProviderError("ollama_parse", f"message.content is not a string: {type(content).__name__}")
        raw_tool_calls = msg.get("tool_calls") or []
        if not isinstance(raw_tool_calls, list):
            raise ProviderError(
                "ollama_parse", f"message.tool_calls is not a list: {type(raw_tool_calls).__name__}"
            )

        parsed_tool_calls: list[ToolCall] = []
        for tc_dict in raw_tool_calls:
            try:
                parsed_tool_calls.append(ToolCall.from_provider_dict(tc_dict))
            except _JSON_LINE_ERRORS as exc:
                raise ProviderError(
                    "ollama_parse",
                    f"malformed tool_call in response: {str(exc)[:200]}",
                ) from exc

        return ChatResponse(
            content=_truncate_at_role_leak(content),
            tool_calls=tuple(parsed_tool_calls),
            raw=data,
        )

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        options: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Stream content chunks from /api/chat as Ollama emits them.

        Each yielded value is a str fragment of the assistant's reply.
        The iterator exhausts when Ollama signals ``done=True``. Joining
        all yielded chunks reproduces the same content :meth:`chat`
        returns for the same prompt.

        Tool-calling is not supported here — Ollama's tool_calls arrive
        on the final stream frame, which complicates the perceived-typing
        UX this method exists to power. Callers needing tools should use
        :meth:`chat` and word-chunk the result client-side, or call
        :meth:`chat` first then stream a follow-up turn.

        Parameters
        ----------
        messages:
            Conversation history in chronological order.
        options:
            Provider-specific generation options. Reserved keys
            (``persona_dir``) are stripped before forwarding.

        Yields
        ------
        str
            Each content chunk Ollama emits. Empty chunks are skipped, and so is
            anything that is not a usable frame: a line that is valid JSON but not
            an object, and a frame whose ``message`` is not an object or whose
            ``content`` is not a string contribute no chunk (a ``done`` on such a
            frame still ends the stream). Unlike :meth:`chat`, which rejects a
            reply of the wrong shape with ``ollama_parse``, the stream keeps going.

        Raises
        ------
        ProviderError("ollama_http", ...)
            HTTP error response from Ollama (4xx / 5xx).
        ProviderError("ollama_request", ...)
            Network-level failure (connection refused, DNS, etc.).
        ProviderError("ollama_parse", ...)
            A streamed line could not be JSON-decoded (including a line nested
            past the recursion limit or a digit run past the integer limit).
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
        }
        if options:
            generation_options = {
                key: value
                for key, value in options.items()
                if key not in _PROVIDER_CONTEXT_OPTION_KEYS
            }
            if generation_options:
                payload["options"] = generation_options

        url = f"{self._host}/api/chat"
        try:
            with httpx.stream("POST", url, json=payload, timeout=self._timeout) as resp:
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    body = exc.response.read().decode("utf-8", errors="replace")
                    raise ProviderError(
                        "ollama_http",
                        f"{exc.response.status_code}: {body[:200]}",
                    ) from exc
                for raw_line in resp.iter_lines():
                    line = (
                        raw_line.strip()
                        if isinstance(raw_line, str)
                        else raw_line.decode("utf-8", errors="replace").strip()
                    )
                    if not line:
                        continue
                    try:
                        frame = json.loads(line)
                    except _JSON_LINE_ERRORS as exc:
                        raise ProviderError(
                            "ollama_parse",
                            f"invalid streaming line json: {exc}",
                        ) from exc
                    if not isinstance(frame, dict):
                        continue  # valid JSON that is not an object carries no frame (#468)
                    msg = frame.get("message") or {}
                    chunk = msg.get("content") if isinstance(msg, dict) else None
                    if chunk and isinstance(chunk, str):
                        yield chunk
                    if frame.get("done"):
                        return
        except httpx.RequestError as exc:
            raise ProviderError("ollama_request", str(exc)) from exc

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Single-turn text generation.

        Implemented by delegating to chat() so all request/response logic
        lives in one place.  This is the first working Ollama generate() path
        in companion-emergence — previously this raised NotImplementedError.
        """
        shutdown_progress.abandon_gate()  # #456 (D17)
        if cli_throttle.shutdown_requested():  # #456: no new background call while the bridge stops
            raise cli_throttle.ShutdownDeferred("bridge is stopping")
        messages: list[ChatMessage] = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        messages.append(ChatMessage(role="user", content=prompt))
        response = self.chat(messages)
        return response.content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_provider(name: str, *, tier: str = TIER_INTERACTIVE_CHAT) -> LLMProvider:
    """Resolve a provider identifier to an instance.

    Parameters
    ----------
    name:
        Provider name: "claude-cli", "ollama", or "fake".
    tier:
        The JOB the provider will do (a ``brain.bridge.model_tier`` name). The "claude-cli" provider looks its model
        up from the user's tunables for that job at every call. Ignored for "ollama" and "fake", which run their own
        model.

    Raises ValueError on unknown name.
    """
    if name == "fake":
        return FakeProvider()
    if name == "claude-cli":
        return ClaudeCliProvider(tier)
    if name == "ollama":
        return OllamaProvider()
    raise ValueError(f"Unknown provider: {name!r}")

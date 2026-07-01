"""Caveman-mode env leak guard.

The ``claude`` CLI subprocess spawned by ``ClaudeCliProvider`` inherits the
parent process's environment by default. If the bridge daemon was ever
started from inside a Claude Code session scoped to a project with a
``CAVEMAN_DEFAULT_MODE`` override in its ``.claude/settings.local.json``
(e.g. this repo, which sets it to "full" for interactive coding sessions),
that var gets baked into the daemon's own environment at spawn time and
propagates to every subsequent ``claude`` subprocess it forks — including
calls made on behalf of Nell's own chat, injecting the caveman SessionStart/
UserPromptSubmit ruleset into her replies to Hana.

Fix: ``_subprocess_env()`` builds the env passed to every ``claude`` spawn,
scrubbing any ``CAVEMAN_*`` var and forcing ``CAVEMAN_DEFAULT_MODE=off``
explicitly — regardless of what the daemon's own process happens to have
inherited.
"""

import ast
from pathlib import Path

_PROVIDER = Path(__file__).resolve().parents[2] / "brain" / "bridge" / "provider.py"


def _spawn_calls(tree: ast.AST) -> list[ast.Call]:
    out: list[ast.Call] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr in ("run", "Popen")
        ):
            out.append(node)
    return out


def _passes_scrubbed_env(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "env":
            # Accept `env=_subprocess_env()` directly, or `env=env_overrides`
            # where env_overrides is built from _subprocess_env() — the AST
            # guard only checks the spawn call itself; the env_overrides
            # construction is covered by a separate unit test.
            if isinstance(kw.value, ast.Call) and isinstance(kw.value.func, ast.Name):
                return kw.value.func.id == "_subprocess_env"
            if isinstance(kw.value, ast.Name):
                return kw.value.id == "env_overrides"
    return False


def test_subprocess_env_helper_defined() -> None:
    src = _PROVIDER.read_text(encoding="utf-8")
    assert "def _subprocess_env(" in src


def test_every_claude_spawn_passes_scrubbed_env() -> None:
    tree = ast.parse(_PROVIDER.read_text(encoding="utf-8"))
    calls = _spawn_calls(tree)
    assert calls, "expected subprocess.run/Popen launches in provider.py"
    bad = [c.lineno for c in calls if not _passes_scrubbed_env(c)]
    assert not bad, f"claude spawn(s) missing scrubbed env= at lines {bad}"


def test_subprocess_env_forces_caveman_off(monkeypatch) -> None:
    monkeypatch.setenv("CAVEMAN_DEFAULT_MODE", "full")
    monkeypatch.setenv("CAVEMAN_DEBUG", "1")

    from brain.bridge.provider import _subprocess_env

    env = _subprocess_env()

    assert env["CAVEMAN_DEFAULT_MODE"] == "off"
    assert "CAVEMAN_DEBUG" not in env


def test_subprocess_env_preserves_other_vars(monkeypatch) -> None:
    monkeypatch.setenv("SOME_UNRELATED_VAR", "keep-me")

    from brain.bridge.provider import _subprocess_env

    env = _subprocess_env()

    assert env["SOME_UNRELATED_VAR"] == "keep-me"

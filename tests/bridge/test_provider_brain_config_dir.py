"""Clean-config isolation for the brain's `claude` spawns.

The `claude` CLI injects the user's *global* interactive config into every
`-p` invocation as context: the superpowers `using-superpowers` block, the
agent-types list, the enabled-skills catalogue, and the entire
`~/.claude/CLAUDE.md`. None of that is part of Nell's persona or her
conversation, but she receives it every turn and sometimes narrates it
("skill-loading noise mid-scene"). It also costs cache/tokens.

It cannot be stripped by CLI flags without breaking either her MCP tools
(`--safe-mode`, `--disable-slash-commands`) or auth (`--bare`, isolated
config dir with no login). The one clean escape is a DEDICATED, brain-owned
`CLAUDE_CONFIG_DIR` that has its own one-time `claude auth login` and no
plugins/hooks/CLAUDE.md.

Safety spine: the brain dir is used ONLY when it exists AND carries the
`.brain-authed` marker (written by the setup helper after it confirms
`claude auth status` reports logged-in). Absent the marker, `_subprocess_env`
does NOT set `CLAUDE_CONFIG_DIR`, so every spawn falls back to exactly the
current default behaviour — a missing or not-yet-set-up brain login can never
break Nell's chat or tool-calling, it just leaves the (noisy) status quo.
"""

def test_no_brain_config_dir_leaves_env_unset(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

    from brain.bridge.provider import _subprocess_env

    env = _subprocess_env()

    # Fail-safe: no brain config dir set up → do not steer CLAUDE_CONFIG_DIR.
    assert "CLAUDE_CONFIG_DIR" not in env


def test_brain_config_dir_without_marker_is_ignored(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    # dir exists but was never confirmed-authed → must NOT be used.
    (tmp_path / "claude-config").mkdir()

    from brain.bridge.provider import _subprocess_env

    env = _subprocess_env()

    assert "CLAUDE_CONFIG_DIR" not in env


def test_brain_config_dir_with_marker_is_used(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    cfg = tmp_path / "claude-config"
    cfg.mkdir()
    (cfg / ".brain-authed").write_text("ok", encoding="utf-8")

    from brain.bridge.provider import _subprocess_env

    env = _subprocess_env()

    assert env["CLAUDE_CONFIG_DIR"] == str(cfg)


def test_explicit_env_claude_config_dir_still_wins(monkeypatch, tmp_path) -> None:
    # If something upstream already pinned CLAUDE_CONFIG_DIR, don't override it.
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/some/explicit/dir")
    cfg = tmp_path / "claude-config"
    cfg.mkdir()
    (cfg / ".brain-authed").write_text("ok", encoding="utf-8")

    from brain.bridge.provider import _subprocess_env

    env = _subprocess_env()

    assert env["CLAUDE_CONFIG_DIR"] == "/some/explicit/dir"

"""Guard: `nell init` must never block on a prompt when stdin is closed.

When the Tauri app spawns `nell init` with stdin redirected to NUL/dev-null,
every `input()` raises EOFError. _prompt/_prompt_choice must absorb that and
fall back to defaults so init completes non-interactively. This is the bug
behind the Windows init_timeout report (2026-05-24): init reached the model
prompt and, on Windows, a live console stdin made it hang. The model prompt
now returns blank on EOF, which leaves the install-wide model setting as it is.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from brain import cli, tunables


def test_init_handler_non_interactive_under_eof(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    # Any prompt reached must EOF, never block.
    monkeypatch.setattr("builtins.input", lambda *a, **k: (_ for _ in ()).throw(EOFError()))

    args = argparse.Namespace(
        persona="testbot",
        user_name="Ada",
        migrate_from=None,
        voice_template="default",
        force=False,
        model=None,  # force the model prompt path
        fresh=True,
    )
    rc = cli._init_handler(args)
    assert rc == 0

    assert tunables.get_override("model.interactive_chat") is None  # the prompt EOF'd to "keep the current setting"
    assert not (tmp_path / "tunables.json").exists()  # and wrote nothing

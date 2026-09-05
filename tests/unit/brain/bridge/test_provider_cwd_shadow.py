"""#138 — the brain-tools MCP child must launch with -P so a foreign cwd `brain/` can't shadow it.

Two things proven here, offline:

* **C1** — all three provider spawn sites build the child config from ONE shared helper
  (`brain_tools_mcp_entry`), whose args carry `-P` before `-m brain.mcp_server`, and no site
  bypasses it (the literal ``"brain.mcp_server"`` appears exactly once in provider.py).
* **C2** — `-P` actually prevents a cwd-shadow, and the test discriminates: from a cwd holding a
  fake top-level `brain/`, a child launched WITHOUT `-P` resolves the fake, WITH `-P` resolves the
  real installed brain.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import brain.bridge.provider as provider_mod
from brain.bridge.provider import brain_tools_mcp_entry

PROVIDER_SRC = Path(provider_mod.__file__)


# --- C1: one shared definition, -P present, no bypass ---------------------------------------------


def test_entry_without_request_id_has_no_env_and_safe_path(tmp_path: Path) -> None:
    entry = brain_tools_mcp_entry(tmp_path / "persona")
    assert entry["command"] == sys.executable
    assert entry["args"] == ["-P", "-m", "brain.mcp_server", "--persona-dir", str(tmp_path / "persona")]
    assert "env" not in entry  # matches legacy site A exactly


def test_entry_with_request_id_carries_audit_env_and_safe_path(tmp_path: Path) -> None:
    entry = brain_tools_mcp_entry(tmp_path / "persona", request_id="abc123")
    assert entry["env"] == {"NELL_MCP_AUDIT_REQUEST_ID": "abc123"}
    assert entry["args"][:3] == ["-P", "-m", "brain.mcp_server"]


def test_entry_with_session_id_only_carries_session_env(tmp_path: Path) -> None:
    """#80: session_id crosses the process boundary the same way request_id
    does (its own env key, NELL_MCP_SESSION_ID)."""
    entry = brain_tools_mcp_entry(tmp_path / "persona", session_id="sess-42")
    assert entry["env"] == {"NELL_MCP_SESSION_ID": "sess-42"}


def test_entry_with_both_request_id_and_session_id_carries_both_env_keys(tmp_path: Path) -> None:
    """#80: request_id and session_id are additive — neither clobbers the other."""
    entry = brain_tools_mcp_entry(tmp_path / "persona", request_id="abc123", session_id="sess-42")
    assert entry["env"] == {
        "NELL_MCP_AUDIT_REQUEST_ID": "abc123",
        "NELL_MCP_SESSION_ID": "sess-42",
    }


def test_safe_path_precedes_the_m_flag(tmp_path: Path) -> None:
    args = brain_tools_mcp_entry(tmp_path / "p")["args"]
    assert args.index("-P") < args.index("-m"), "-P must precede -m or CPython ignores it"


def test_exactly_one_brain_mcp_server_literal_no_site_bypasses_helper() -> None:
    """No spawn site may inline its own config: the double-quoted literal lives only in the helper."""
    src = PROVIDER_SRC.read_text(encoding="utf-8")
    assert src.count('"brain.mcp_server"') == 1, (
        "expected exactly one occurrence (inside brain_tools_mcp_entry); a second means a site "
        "bypassed the shared helper and may lack -P"
    )


def test_count_invariant_oracle_can_fail() -> None:
    """Self-test (ST1.5f): the count check fires on a known-violating input."""
    two_site = 'a = "brain.mcp_server"\nb = "brain.mcp_server"\n'
    assert two_site.count('"brain.mcp_server"') == 2  # the oracle discriminates


# --- C2: -P prevents the cwd-shadow, and the control DOES shadow ----------------------------------

_FIND_BRAIN = (
    "import importlib.util as u, sys; "
    "s = u.find_spec('brain'); "
    "sys.stdout.write(s.origin if s and s.origin else 'NONE')"
)


def _resolve_brain_from(cwd: Path, *, safe_path: bool) -> str:
    argv = [sys.executable]
    if safe_path:
        argv.append("-P")
    argv += ["-c", _FIND_BRAIN]
    proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_safe_path_prevents_cwd_shadow(tmp_path: Path) -> None:
    shadow = tmp_path / "shadow"
    (shadow / "brain").mkdir(parents=True)
    (shadow / "brain" / "__init__.py").write_text("# fake shadow brain\n", encoding="utf-8")

    without_p = _resolve_brain_from(shadow, safe_path=False)
    with_p = _resolve_brain_from(shadow, safe_path=True)

    # Discrimination (oracle shown able to fail): the control MUST resolve the fake cwd brain,
    # else the test never exercises the shadow.
    assert str(shadow) in without_p, (
        f"control did not shadow — test is not exercising the bug (got {without_p})"
    )
    # The fix: -P drops the cwd entry, so brain resolves from the installed/venv location.
    assert str(shadow) not in with_p, f"-P failed to prevent the cwd shadow (got {with_p})"
    # str(Path) is OS-native — on Windows this string uses backslashes, so a
    # POSIX literal can never match. as_posix() normalises; it is a no-op off Windows.
    assert Path(with_p).as_posix().endswith("brain/__init__.py"), with_p

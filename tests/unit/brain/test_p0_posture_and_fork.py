"""P0 image-tool-route — posture preservation (C4) + no-image-fork (C6).

C4: the MCP-only disallowed-builtins posture must be untouched — built-in
``Read`` stays disallowed (the image must arrive via the MCP ImageContent
block, never a re-enabled built-in file tool).

C6: the image/text builder fork is gone — none of the removed transport
symbols remain in non-test source, and the engine no longer forks on images.
"""

from __future__ import annotations

from pathlib import Path

import brain

_BRAIN_ROOT = Path(brain.__file__).parent


# ── C4 — MCP-only posture preserved ────────────────────────────────────────


def test_read_stays_in_disallowed_builtins() -> None:
    from brain.bridge.provider import _BUILTIN_TOOLS_DISALLOWED

    assert "Read" in _BUILTIN_TOOLS_DISALLOWED
    # Shown-able-to-fail: the assertion below fails if Read is removed from the
    # tuple (verify by a temporary local mutation of _BUILTIN_TOOLS_DISALLOWED).
    assert isinstance(_BUILTIN_TOOLS_DISALLOWED, tuple)


def test_apply_lean_flags_emits_read_in_disallowed_set() -> None:
    from brain.bridge.provider import _apply_lean_flags

    cmd: list[str] = []
    _apply_lean_flags(cmd)
    assert "--disallowedTools" in cmd
    idx = cmd.index("--disallowedTools")
    disallowed = cmd[idx + 1 :]
    assert "Read" in disallowed
    assert "--strict-mcp-config" in cmd


# ── C6 — no image/text builder fork remains ────────────────────────────────

# The transport symbols removed by this change. None may appear in brain/
# (non-test) source; if any does, the fork the volatile-drop bug lived in is
# still reachable.
_REMOVED_TRANSPORT_SYMBOLS = (
    "_chat_with_images",
    "_message_has_image",
    "_build_stream_json_user_message",
)


def _brain_py_files() -> list[Path]:
    return [p for p in _BRAIN_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


def test_removed_transport_symbols_absent_from_source() -> None:
    offenders: dict[str, list[str]] = {}
    for path in _brain_py_files():
        text = path.read_text(encoding="utf-8")
        for sym in _REMOVED_TRANSPORT_SYMBOLS:
            if sym in text:
                offenders.setdefault(str(path.relative_to(_BRAIN_ROOT)), []).append(sym)
    assert not offenders, f"removed transport symbols still present: {offenders}"


def test_engine_has_no_image_builder_fork() -> None:
    """The engine no longer branches on per-turn images or builds ImageBlocks —
    a single assembly path serves both file-send and text turns."""
    engine_src = (_BRAIN_ROOT / "chat" / "engine.py").read_text(encoding="utf-8")
    # No ImageBlock construction anywhere (the transport's content type).
    assert "ImageBlock" not in engine_src
    # No per-turn image branch (the fork the volatile-drop bug lived in).
    assert "if image_shas" not in engine_src
    assert "image_shas=" not in engine_src
    # build_system_message (the old image-fork's unsplit builder) is no longer
    # imported/used by the engine — every turn takes the static+volatile split.
    # (build_static_system_message is the split head and does NOT match this.)
    assert "build_system_message(" not in engine_src
    assert "from brain.chat.prompt import (\n    build_static_system_message,\n" in engine_src

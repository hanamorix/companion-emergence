"""Tests for brain/tools/impls/compact_history.py — #80.

compact_history builds its own cost-pinned provider internally via
build_compaction_provider(persona_dir), like every other compaction caller,
rather than accepting one injected — a live provider object cannot cross the
MCP dispatch path's process boundary, which is what made this tool
unreachable via MCP (issue #80).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.tools.impls.compact_history import compact_history


def test_compact_history_uses_build_compaction_provider(tmp_path: Path, monkeypatch) -> None:
    """C3: compact_history's sole path to a provider is
    build_compaction_provider(persona_dir) — asserted by observing the patched
    stub is actually invoked with persona_dir, and that its return value is
    what reaches compact_conversation (via a FakeProvider-like stub that
    records whether .generate()/.chat() was ever asked for anything, proving
    it's genuinely wired in rather than merely constructed and discarded)."""
    calls: list[Path] = []

    class _StubProvider:
        def generate(self, prompt: str, *, system: str | None = None) -> str:  # pragma: no cover
            raise AssertionError("generate() should not be called — no aged turns to fold")

    def _fake_build_compaction_provider(persona_dir):
        calls.append(Path(persona_dir))
        return _StubProvider()

    monkeypatch.setattr(
        "brain.chat.compaction.build_compaction_provider",
        _fake_build_compaction_provider,
    )

    # No signature parameter for `provider` exists any more — passing one
    # would be a TypeError, proving there is no other path to a provider.
    with pytest.raises(TypeError):
        compact_history(1.0, persona_dir=tmp_path, session_id="sess-1", provider=_StubProvider())

    result = compact_history(1.0, persona_dir=tmp_path, session_id="sess-1")

    assert calls == [tmp_path]
    assert isinstance(result, dict)
    assert "compacted" in result
    assert "reason" in result

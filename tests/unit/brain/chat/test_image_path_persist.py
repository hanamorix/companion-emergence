"""image-path-persist — the read_file image handle persists into the durable
buffer so a normal memory can bind to the image by content hash.

Criteria under test (see changes/image-path-persist/1.5-criteria.md):
  C1  — durable buffer carries images/<sha>.<ext> after a read_file image read
  C2  — the path is present in the memory-formation input (extract.format_transcript)
  C5  — the "image" record is NOT replayed to the model (direct path)
  C5b — the "image" record is NOT fed to the compaction summariser (fold path),
        while still archived and length-target-neutral
  C7  — the persisted line has no em-dash / LLM-tell
  C10 — the content-addressed path survives the REAL audit write -> read round
        trip across audit modes (the security-sensitive cli channel)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.bridge.chat import ChatResponse
from brain.bridge.provider import FakeProvider
from brain.chat.engine import _buffer_turns_to_messages, respond
from brain.chat.session import reset_registry
from brain.ingest.buffer import read_session
from brain.ingest.extract import format_transcript
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

_SHA = "a" * 64
_REL = f"images/{_SHA}.png"
_IMAGE_LINE_RE = r"images/[0-9a-f]{64}\.(png|jpg|webp|gif)"


@pytest.fixture(autouse=True)
def _reset_sessions():
    reset_registry()
    yield
    reset_registry()


@pytest.fixture()
def persona_dir(tmp_path: Path) -> Path:
    d = tmp_path / "personas" / "canary"
    d.mkdir(parents=True)
    (d / "persona_config.json").write_text(
        json.dumps({"provider": "fake", "searcher": "noop"}), encoding="utf-8"
    )
    return d


class _ScriptedImageProvider(FakeProvider):
    """A FakeProvider that reports a read_file image invocation carrying a
    content-addressed stored_image_path — mirroring what the cli-path MCP audit
    surfaces into ChatResponse.dispatched_invocations. Deterministic, no I/O."""

    def __init__(self, rel_paths: list[str]):
        self._rel_paths = rel_paths

    def chat(self, messages, *, tools=None, options=None) -> ChatResponse:  # type: ignore[override]
        disp = tuple(
            {
                "name": "read_file",
                "arguments": {"path": "/somewhere/user_gave.png"},
                "result_summary": "image/png 64B",
                "stored_image_path": rel,
            }
            for rel in self._rel_paths
        )
        return ChatResponse(content="i see it.", tool_calls=(), dispatched_invocations=disp)


def _drive(persona_dir: Path, rel_paths: list[str]):
    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    try:
        result = respond(
            persona_dir,
            "look at this",
            store=store,
            hebbian=hebbian,
            provider=_ScriptedImageProvider(rel_paths),
            voice_md_override="# Canary\n\nhello.",
        )
        return result, read_session(persona_dir, result.session_id)
    finally:
        store.close()
        hebbian.close()


# ── C1 / C7 — the path lands in the durable buffer, with no claudism ──────────


def test_c1_image_path_persists_in_durable_buffer(persona_dir: Path):
    import re

    _result, turns = _drive(persona_dir, [_REL])
    image_rows = [t for t in turns if t.get("speaker") == "image"]
    assert len(image_rows) == 1
    assert re.search(_IMAGE_LINE_RE, image_rows[0]["text"])
    assert _REL in image_rows[0]["text"]
    # ordering: user -> image -> assistant
    speakers = [t.get("speaker") for t in turns]
    assert speakers == ["user", "image", "assistant"]


def test_c7_persisted_image_line_has_no_claudism(persona_dir: Path):
    _result, turns = _drive(persona_dir, [_REL])
    line = next(t["text"] for t in turns if t.get("speaker") == "image")
    assert "—" not in line  # em-dash
    for tell in ("delve", "tapestry"):
        assert tell not in line.lower()


def test_c1_multi_image_dedup_and_order(persona_dir: Path):
    rel_b = f"images/{'b' * 64}.jpg"
    # two distinct images + a duplicate of the first
    _result, turns = _drive(persona_dir, [_REL, rel_b, _REL])
    image_rows = [t["text"] for t in turns if t.get("speaker") == "image"]
    assert len(image_rows) == 2  # deduped
    assert _REL in image_rows[0] and rel_b in image_rows[1]  # first-seen order


# ── C2 — available where memory formation reads it ────────────────────────────


def test_c2_path_present_in_extraction_input(persona_dir: Path):
    _result, turns = _drive(persona_dir, [_REL])
    transcript = format_transcript(turns)
    assert _REL in transcript


# ── C5 — not replayed to the model (direct path) ──────────────────────────────


def test_c5_image_record_not_replayed_to_model(persona_dir: Path):
    base = [
        {"speaker": "user", "text": "hi", "ts": "2026-08-18T00:00:00Z"},
        {"speaker": "assistant", "text": "hello", "ts": "2026-08-18T00:00:01Z"},
    ]
    with_image = base[:1] + [
        {"speaker": "image", "text": f"[image opened, stored at {_REL}]", "ts": "2026-08-18T00:00:00Z"}
    ] + base[1:]
    msgs_base = _buffer_turns_to_messages(persona_dir, base)
    msgs_img = _buffer_turns_to_messages(persona_dir, with_image)
    assert [(m.role, m.content_text()) for m in msgs_base] == [
        (m.role, m.content_text()) for m in msgs_img
    ]
    assert all(_REL not in m.content_text() for m in msgs_img)


# ── C5b — not fed to the compaction summariser; still archived ────────────────


def test_c5b_image_row_excluded_from_compaction_render(persona_dir: Path):
    from brain.chat.compaction import _render_transcript

    turns = [
        {"speaker": "user", "text": "hi"},
        {"speaker": "image", "text": f"[image opened, stored at {_REL}]"},
        {"speaker": "assistant", "text": "hello"},
    ]
    rendered = _render_transcript(turns, "Canary")
    assert _REL not in rendered
    assert "image:" not in rendered
    # the real conversational turns still render
    assert "hi" in rendered and "hello" in rendered


def test_c5b_render_would_leak_without_the_guard():
    """Shows the C5b oracle can fail: a NON-image speaker with the same text
    would be rendered — proving the guard (skip speaker=='image') is what keeps
    the path out, not the text being special."""
    from brain.chat.compaction import _render_transcript

    leaky = [{"speaker": "user", "text": f"[image opened, stored at {_REL}]"}]
    assert _REL in _render_transcript(leaky, "Canary")


# ── C10 — real audit write -> read round trip across audit modes ──────────────


@pytest.mark.parametrize("mode", ["full", "redacted", "metadata"])
def test_c10_stored_image_path_survives_audit_round_trip(tmp_path: Path, mode: str):
    from brain.bridge.provider import _read_audit_lines_since
    from brain.mcp_server.audit import log_invocation

    persona = tmp_path / "canary"
    persona.mkdir()
    (persona / "persona_config.json").write_text(
        json.dumps({"mcp_audit_log_level": mode}), encoding="utf-8"
    )
    log_path = persona / "tool_invocations.log.jsonl"
    offset_before = log_path.stat().st_size if log_path.exists() else 0

    log_invocation(
        persona,
        name="read_file",
        arguments={"path": "/somewhere/user_gave.png"},
        result_summary="image/png 64B",
        stored_image_path=_REL,
    )
    records = _read_audit_lines_since(log_path, offset_before)
    assert len(records) == 1
    assert records[0]["stored_image_path"] == _REL
    # never base64
    assert _REL == records[0]["stored_image_path"]
    assert "data_b64" not in log_path.read_text()


def test_c10_audit_off_writes_nothing(tmp_path: Path):
    from brain.bridge.provider import _read_audit_lines_since
    from brain.mcp_server.audit import log_invocation

    persona = tmp_path / "canary"
    persona.mkdir()
    (persona / "persona_config.json").write_text(
        json.dumps({"mcp_audit_log_level": "off"}), encoding="utf-8"
    )
    log_invocation(
        persona,
        name="read_file",
        arguments={"path": "/x.png"},
        result_summary="image/png 64B",
        stored_image_path=_REL,
    )
    log_path = persona / "tool_invocations.log.jsonl"
    # off mode writes no record at all -> feature degrades (documented, acceptable)
    records = _read_audit_lines_since(log_path, 0) if log_path.exists() else []
    assert records == []

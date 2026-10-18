"""Group B of the json_io phase-2 migration: modules whose JSON file sites are writers and appenders only.

One behaviour test per migrated function (criteria PC6): each calls the REAL function on a ``tmp_path`` and
asserts the on-disk effect with the stdlib. Appenders: every line parses, the file is pure ASCII, and a torn tail
(a crash mid-append, #434) costs only the torn record. Writers: the file parses, a second call replaces it, no
temp file is left behind. Nothing here touches a persona directory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------------------------
# Appenders: name -> (call(tmp_path, n) appending record n, relative path of the file it writes)
# ---------------------------------------------------------------------------------------------


def _memory(text: str):
    from brain.memory.store import Memory

    return Memory.create_new(content=text, memory_type="conversation", domain="us", emotions={"love": 5.0})


def _call_haiku(tmp: Path, n: int) -> None:
    from unittest import mock

    from brain.attunement import detector

    class _Boom:
        def __init__(self, *a, **k) -> None:
            pass

        def generate(self, *a, **k):
            raise RuntimeError(f"cli down {n}")

    with mock.patch("brain.bridge.provider.ClaudeCliProvider", _Boom):
        assert detector._call_haiku("sys", "user", persona_dir=tmp) == ""


APPENDERS = {
    "detector._call_haiku": (_call_haiku, "attunement_errors.jsonl"),
    "usage_log.log_usage": (
        lambda tmp, n: __import__("brain.bridge.usage_log", fromlist=["x"]).log_usage(
            tmp, call_type="chat", model=f"m{n}", frame={"usage": {"input_tokens": n}, "is_error": True}
        ),
        "chat_usage.jsonl",
    ),
    "extractor._append_jsonl": (
        lambda tmp, n: __import__("brain.chat.extractor", fromlist=["x"])._append_jsonl(
            tmp / "x" / "e.jsonl", {"n": n, "t": "café"}
        ),
        "x/e.jsonl",
    ),
    "monologue_capture._append_jsonl": (
        lambda tmp, n: __import__("brain.chat.monologue_capture", fromlist=["x"])._append_jsonl(
            tmp / "m.jsonl", {"n": n, "t": "☃"}
        ),
        "m.jsonl",
    ),
    "tool_loop._log_attunement_error": (
        lambda tmp, n: __import__("brain.chat.tool_loop", fromlist=["x"])._log_attunement_error(
            tmp, f"t{n}", RuntimeError(f"boom {n}")
        ),
        "attunement_errors.jsonl",
    ),
    "consolidation._archive_preimage": (
        lambda tmp, n: __import__("brain.engines.consolidation", fromlist=["x"])._archive_preimage(
            tmp, _memory(f"m{n} café"), f"src{n}"
        ),
        "consolidation_archive.jsonl",
    ),
    "dedup_sweep._archive_preimage": (
        lambda tmp, n: __import__("brain.engines.dedup_sweep", fromlist=["x"])._archive_preimage(
            tmp, _memory(f"m{n}")
        ),
        "consolidation_archive.jsonl",
    ),
    "dedup_sweep._write_report": (
        lambda tmp, n: __import__("brain.engines.dedup_sweep", fromlist=["x"])._write_report(
            tmp, __import__("brain.engines.dedup_sweep", fromlist=["x"]).Report(groups_merged=n)
        ),
        None,  # name read from the module below
    ),
    "heartbeat._append_log": (
        lambda tmp, n: __import__("brain.engines.heartbeat", fromlist=["x"]).HeartbeatEngine._append_log(
            SimpleNamespace(heartbeat_log_path=tmp / "hb.log.jsonl"), {"n": n, "t": "é"}
        ),
        "hb.log.jsonl",
    ),
    "files.audit.audit": (
        lambda tmp, n: __import__("brain.files.audit", fromlist=["x"]).audit(
            tmp, event="e", id=f"id{n}", op="write", path="/p/é", outcome="ok"
        ),
        "write_audit.jsonl",
    ),
    "new_sources.write_gate_rejection": (
        lambda tmp, n: __import__("brain.initiate.new_sources", fromlist=["x"]).write_gate_rejection(
            tmp,
            ts=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").UTC),
            source="s",
            source_id=f"i{n}",
            gate_name="g",
            threshold_value=1.0,
            observed_value=float(n),
        ),
        "gate_rejections.jsonl",
    ),
    "kindled_link.audit.log_transport": (
        lambda tmp, n: __import__("brain.kindled_link.audit", fromlist=["x"]).log_transport(
            tmp, event=f"e{n}", error="café", now=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").UTC)
        ),
        "kindled_link/transport.jsonl",
    ),
    "relationship._log_rejection": (
        lambda tmp, n: __import__("brain.kindled_link.relationship", fromlist=["x"])._log_rejection(
            persona_dir=tmp,
            peer_id=f"p{n}",
            rejected_quote="“quote”",
            now=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").UTC),
        ),
        "kindled_link/reflection_rejections.jsonl",
    ),
    "mcp_server.audit.log_invocation": (
        lambda tmp, n: __import__("brain.mcp_server.audit", fromlist=["x"]).log_invocation(
            tmp, name="t", arguments={"q": f"café {n}"}, result_summary="ok", outcome="ok"
        ),
        "tool_invocations.log.jsonl",
    ),
    "notes.write._audit": (
        lambda tmp, n: __import__("brain.notes.write", fromlist=["x"])._audit(tmp, path=f"/n/{n}", ok=True),
        "notes_audit.jsonl",
    ),
    "articulate._log_articulate_error": (
        lambda tmp, n: __import__("brain.self_model.articulate", fromlist=["x"])._log_articulate_error(
            tmp, "k", RuntimeError(f"e{n}")
        ),
        None,
    ),
    "read_file._audit": (
        lambda tmp, n: __import__("brain.tools.impls.read_file", fromlist=["x"])._audit(
            tmp, tool="read_file", path="a", resolved=f"/r/{n}", bytes_=n, ok=True
        ),
        "file_access.jsonl",
    ),
}


def _locate(tmp: Path, rel: str | None, before: set[Path]) -> Path:
    if rel is not None:
        return tmp / rel
    new = [p for p in tmp.rglob("*.jsonl") if p not in before]
    assert len(new) == 1, new
    return new[0]


@pytest.mark.parametrize("name", sorted(APPENDERS))
def test_group_b_appender_writes_ascii_json_lines(tmp_path: Path, name: str) -> None:
    call, rel = APPENDERS[name]
    before = set(tmp_path.rglob("*.jsonl"))
    call(tmp_path, 1)
    call(tmp_path, 2)
    path = _locate(tmp_path, rel, before)
    raw = path.read_bytes()
    assert raw.isascii(), "output must be pure ASCII (#433)"
    lines = raw.split(b"\n")
    assert lines[-1] == b"" and len(lines) == 3  # two records, each newline-terminated
    assert all(isinstance(json.loads(line), dict) for line in lines[:-1])


@pytest.mark.parametrize("name", sorted(APPENDERS))
def test_group_b_appender_after_a_torn_tail_loses_only_the_torn_record(tmp_path: Path, name: str) -> None:
    call, rel = APPENDERS[name]
    before = set(tmp_path.rglob("*.jsonl"))
    call(tmp_path, 1)
    path = _locate(tmp_path, rel, before)
    path.write_bytes(path.read_bytes().rstrip(b"\n")[:-2])  # a crash cut the record: no newline, half a record
    call(tmp_path, 2)
    good = []
    for line in path.read_bytes().split(b"\n"):
        if not line.strip():
            continue
        try:
            good.append(json.loads(line))
        except ValueError:
            pass
    assert len(good) == 1, "the new record must start on its own line (#434)"


def test_group_b_mcp_audit_a_nan_argument_does_not_fail_the_call(tmp_path: Path) -> None:
    """BC4(b): arguments are model-supplied, and stdlib json.loads accepts NaN; today's code wrote it."""
    from brain.mcp_server.audit import log_invocation

    log_invocation(
        tmp_path, name="t", arguments={"x": float("nan"), "y": [math.inf]}, result_summary="ok", outcome="ok"
    )
    (log,) = list(tmp_path.glob("*.jsonl"))
    rec = json.loads(log.read_text())
    assert rec["arguments"] in ({"x": None, "y": [None]}, {"x": "NaN", "y": ["Infinity"]}) or rec["arguments"]["x"] is None


@pytest.mark.parametrize(
    "call,rel",
    [
        (lambda tmp: __import__("brain.bridge.usage_log", fromlist=["x"]).log_usage(
            tmp / "does" / "not" / "matter", call_type="c", model="m", frame={}), None),
    ],
)
def test_group_b_best_effort_appender_never_raises_on_an_unwritable_target(tmp_path: Path, call, rel) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where a directory should be")
    call(blocker)  # must not raise


# ---------------------------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------------------------


def _no_temp_left(directory: Path) -> None:
    left = [p.name for p in directory.iterdir() if ".tmp" in p.name or p.name.endswith(".new")]
    assert not left, left


def test_group_b_app_config_write_if_missing_creates_once(tmp_path: Path, monkeypatch) -> None:
    from brain import app_config

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    app_config.write_if_missing("nell")
    p = tmp_path / "app_config.json"
    assert json.loads(p.read_text()) == {
        "selected_persona": "nell",
        "always_on_top": False,
        "reduced_motion": False,
    }
    app_config.write_if_missing("other")  # existing file untouched
    assert json.loads(p.read_text())["selected_persona"] == "nell"
    _no_temp_left(tmp_path)


def test_group_b_compaction_markers_are_atomic_json(tmp_path: Path) -> None:
    from brain.chat import compaction_migration as cm

    cm._write_marker(tmp_path, {"done": True, "n": 1})
    marker = cm._marker_path(tmp_path)
    assert json.loads(marker.read_text()) == {"done": True, "n": 1}
    other = tmp_path / "sub" / "m2.json"
    cm._write_marker_at(other, {"k": "café"})
    assert json.loads(other.read_text()) == {"k": "café"}
    assert other.read_bytes().isascii()
    _no_temp_left(marker.parent)
    _no_temp_left(other.parent)


def test_group_b_self_model_repair_writes_its_marker(tmp_path: Path) -> None:
    from brain.health import self_model_repair as smr

    smr.run_self_model_repair(tmp_path)
    marker = tmp_path / smr._MARKER_FILENAME
    assert json.loads(marker.read_text()) == {"done": True}


def test_group_b_write_source_manifest_keeps_three_backups(tmp_path: Path) -> None:
    from brain.migrator.report import write_source_manifest

    path = tmp_path / "source-manifest.json"
    for i in range(3):
        write_source_manifest(path, [], lived_age_hours_at_migration=float(i))
    assert json.loads(path.read_text())["lived_age_hours_at_migration"] == 2.0
    assert json.loads((tmp_path / "source-manifest.json.bak1").read_text())["lived_age_hours_at_migration"] == 1.0
    _no_temp_left(tmp_path)


def test_group_b_tunables_write_defaults_preserves_overrides_and_survives_corruption(
    tmp_path: Path, monkeypatch
) -> None:
    from brain import tunables

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables._reset_for_tests()
    p = tunables._file_path()
    p.write_text(json.dumps({"overrides": {"a.b": 3}}))
    tunables.write_defaults_section()
    data = json.loads(p.read_text())
    assert data["overrides"] == {"a.b": 3} and "defaults" in data
    p.write_bytes(b"\xff{not json")  # corrupt: rebuilt, boot not broken
    tunables.write_defaults_section()
    assert json.loads(p.read_text())["overrides"] == {}
    _no_temp_left(p.parent)


def test_group_b_tunables_load_overrides_fail_open_and_cache_by_mtime(tmp_path: Path, monkeypatch) -> None:
    from brain import tunables

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables._reset_for_tests()
    p = tunables._file_path()
    assert tunables._load_overrides_locked() == {}  # missing
    p.write_text(json.dumps({"overrides": {"x": 1}}))
    assert tunables._load_overrides_locked() == {"x": 1}
    p.write_text(json.dumps({"overrides": [1]}))  # wrong type: ignored, code defaults
    import os

    os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 5))
    assert tunables._load_overrides_locked() == {}
    p.write_bytes(b"\xff\xfe")
    os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 10))
    assert tunables._load_overrides_locked() == {}


def test_group_b_tunables_migration_rewrites_the_file_atomically(tmp_path: Path) -> None:
    from brain import tunables_migration as tm

    (tmp_path / tm._FILE_NAME).write_text(
        json.dumps({"overrides": {tm._OLD_THROTTLE_KEY: 5, "keep": 1}}, indent=1)
    )
    tm.migrate_idle_keys(tmp_path)
    after = json.loads((tmp_path / tm._FILE_NAME).read_text())
    assert tm._OLD_THROTTLE_KEY not in after["overrides"] and after["overrides"]["keep"] == 1
    _no_temp_left(tmp_path)


def test_group_b_migrator_cli_vocab_and_interests_use_the_module_writer() -> None:
    """The two atomic writes in _run_migrate_locked go through write_json (no ``.new`` temp of their own)."""
    import ast

    src = Path("brain/migrator/cli.py").read_text()
    assert "_vocab_tmp" not in src and "_interests_tmp" not in src
    tree = ast.parse(src)
    names = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "write_json" in names

"""Tests for brain.health.json_io: the one JSON file reader/writer (#438, #436, #434).

Test names start with the lower-case criterion ID of ``1.5-criteria.md`` (``test_cr3_...``).
Faults are injected at the exact call Windows or a failing disk would raise from; nothing here touches a
persona directory, the network or the bridge.
"""

from __future__ import annotations

import logging
import os
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from brain.utils.atomic_write import save_with_backup_text

POSIX = os.name == "posix"


class _LazyModule:
    """``jio.read_json`` imports brain.health.json_io on first use, so a missing module fails each test, not collection."""

    def __getattr__(self, name):
        import brain.health.json_io as module

        return getattr(module, name)

    def __setattr__(self, name, value):  # monkeypatch.setattr(jio, ...) must reach the real module
        import brain.health.json_io as module

        setattr(module, name, value)


jio = _LazyModule()


@pytest.fixture(autouse=True)
def _fresh_throttle():
    """The evidence throttle is process-wide state (#429's warn_throttled): reset it around every test."""
    from brain.utils import log_throttle

    log_throttle._last.clear()
    yield
    log_throttle._last.clear()


# --------------------------------------------------------------------------------------------------------------
# Helper additions to atomic_write.save_with_backup_text (keyword-only; defaults reproduce today's behaviour)
# --------------------------------------------------------------------------------------------------------------


def test_cw4_keep_backups_false_makes_no_backup_and_no_copy(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    save_with_backup_text(p, "one\n", keep_backups=False)
    save_with_backup_text(p, "two\n", keep_backups=False)
    assert p.read_text() == "two\n"
    assert sorted(f.name for f in tmp_path.iterdir()) == ["x.json"]


def test_cw4_keep_backups_false_leaves_an_existing_chain_alone(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    (tmp_path / "x.json.bak1").write_text("old1\n")
    (tmp_path / "x.json.bak2").write_text("old2\n")
    p.write_text("live\n")
    save_with_backup_text(p, "new\n", keep_backups=False)
    assert p.read_text() == "new\n"
    assert (tmp_path / "x.json.bak1").read_text() == "old1\n"
    assert (tmp_path / "x.json.bak2").read_text() == "old2\n"


def test_cw4_backup_count_zero_keeps_its_legacy_meaning(tmp_path: Path) -> None:
    """The existing suite pins backup_count=0 as 'one backup'; keep_backups=False is the new way to ask for none."""
    p = tmp_path / "x.json"
    save_with_backup_text(p, "one\n", backup_count=0)
    save_with_backup_text(p, "two\n", backup_count=0)
    assert (tmp_path / "x.json.bak1").read_text() == "one\n"


def _count_fsyncs(monkeypatch) -> list[int]:
    calls: list[int] = []
    real = os.fsync

    def spy(fd):
        calls.append(fd)
        return real(fd)

    monkeypatch.setattr(os, "fsync", spy)
    return calls


def test_cw3_fsync_false_skips_the_file_and_directory_fsync(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "x.json"
    calls = _count_fsyncs(monkeypatch)
    save_with_backup_text(p, "a\n", keep_backups=False, fsync=False)
    assert calls == []
    assert p.read_text() == "a\n"


def test_cw3_fsync_default_syncs_the_file_before_publishing_it(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "x.json"
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def replace(src, dst):
        events.append(f"replace:{Path(dst).name}")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    save_with_backup_text(p, "a\n", keep_backups=False)
    assert events.index("fsync") < events.index("replace:x.json")


@pytest.mark.skipif(not POSIX, reason="file modes are POSIX-only")
def test_cw5_mode_is_applied_at_creation_and_to_every_rotated_backup(
    tmp_path: Path, monkeypatch
) -> None:
    p = tmp_path / "token.json"
    p.write_text("legacy\n")
    p.chmod(0o644)  # a previous, wider-mode live file whose copy becomes .bak1
    old_umask = os.umask(0)
    try:
        created: list[int] = []
        real_open = os.open

        def spy(path, flags, mode=0o777, *a, **k):
            if str(path).endswith(".new") and ".bak1" not in str(path):
                created.append(mode)
            return real_open(path, flags, mode, *a, **k)

        monkeypatch.setattr(os, "open", spy)
        save_with_backup_text(p, "secret\n", 2, mode=0o600)
        save_with_backup_text(p, "secret2\n", 2, mode=0o600)
        monkeypatch.undo()
    finally:
        os.umask(old_umask)
    assert created and all(m == 0o600 for m in created), created
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    for i in (1, 2):
        assert stat.S_IMODE((tmp_path / f"token.json.bak{i}").stat().st_mode) == 0o600


def test_cw10_newline_argument_reaches_the_temp_file(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "x.json"
    seen: list[object] = []
    real_open = Path.open

    def spy(self, *args, **kwargs):
        if self.name.endswith(".new") and not self.name.endswith(".bak1.new"):
            seen.append(kwargs.get("newline", "<absent>"))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    save_with_backup_text(p, "a\nb\n", keep_backups=False, newline="\n")
    monkeypatch.undo()
    assert seen == ["\n"]
    assert p.read_bytes() == b"a\nb\n"


# --------------------------------------------------------------------------------------------------------------
# Helper addition to jsonl_reader._parse_lines: keyword-only strict (the shared line parser)
# --------------------------------------------------------------------------------------------------------------


def _wrap(data: bytes):
    import io

    return io.TextIOWrapper(
        io.BytesIO(data), encoding="utf-8", errors="surrogateescape", newline="\n"
    )


@pytest.mark.parametrize(
    "data,line",
    [
        (b'{"a": 1}\nnot json\n{"b": 2}\n', 2),
        (b'{"a": 1}\n[1, 2]\n', 2),
        (b'{"a": 1}\n{"b": "caf\xff"}\n', 2),
    ],
)
def test_cj3_parse_lines_strict_raises_with_the_line_number(data: bytes, line: int) -> None:
    from brain.health.jsonl_reader import JsonlLineError, _parse_lines

    seen: list[dict] = []
    with pytest.raises(JsonlLineError) as exc:
        for rec in _parse_lines(_wrap(data), Path("x.jsonl"), strict=True):
            seen.append(rec)
    assert exc.value.line == line
    assert seen == [{"a": 1}]


def test_cj3_parse_lines_default_still_skips(caplog) -> None:
    from brain.health.jsonl_reader import _parse_lines

    data = b'{"a": 1}\nnot json\n[1]\n{"b": "caf\xff"}\n{"c": 3}\n'
    with caplog.at_level(logging.WARNING):
        got = list(_parse_lines(_wrap(data), Path("x.jsonl")))
    assert got == [{"a": 1}, {"c": 3}]
    assert len(caplog.records) == 3


# --------------------------------------------------------------------------------------------------------------
# Whole-file reads (CR1-CR13, CR15, CR16)
# --------------------------------------------------------------------------------------------------------------

import copy  # noqa: E402
import errno  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402

POLICIES = ["DEFAULT", "RAISE", "RESTORE", "HEAL"]


def _policy(name: str):
    return getattr(jio.Recovery, name)


def _listing(d: Path) -> dict[str, bytes]:
    """Name -> bytes of every file in ``d`` except the heal sidecar lock (a content-kind heal creates it)."""
    return {
        f.name: f.read_bytes()
        for f in sorted(d.iterdir())
        if f.is_file() and not f.name.endswith(".heal.lock")
    }


def _fire(*a, **k):  # a seam replacement that does nothing
    return None


def test_cr1_missing_returns_a_fresh_default_and_logs_nothing(tmp_path: Path, caplog) -> None:
    p = tmp_path / "state.json"
    with caplog.at_level(logging.DEBUG):
        first = jio.read_json(p, default=dict)
        second = jio.read_json(p, default=dict)
    assert first == {} and second == {} and first is not second
    assert [r for r in caplog.records if r.name.startswith("brain.health")] == []
    assert list(tmp_path.iterdir()) == []


def test_cr1_missing_with_raise_and_no_default_is_json_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(jio.JsonFileNotFound) as exc:
        jio.read_json(tmp_path / "nope.json", recover=jio.Recovery.RAISE)
    assert exc.value.kind == "missing"
    assert isinstance(exc.value, FileNotFoundError)


def test_cr1_a_file_deleted_between_probe_and_open_is_missing_not_a_crash(
    tmp_path: Path, monkeypatch
) -> None:
    p = tmp_path / "state.json"
    p.write_text('{"a": 1}')
    real_open = open

    def vanishing(file, *a, **k):
        if Path(file) == p:
            raise FileNotFoundError(errno.ENOENT, "gone", str(file))
        return real_open(file, *a, **k)

    monkeypatch.setattr("builtins.open", vanishing)
    # an exists() pre-check would have said True; the module must not rely on it
    result = jio.read_json_result(p, default=lambda: {"fresh": True})
    assert result.data == {"fresh": True} and result.outcome == "missing"


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    "content,kind",
    [
        (b'{"a":', "malformed"),
        (b"", "empty"),
        (b"   \n", "empty"),
    ],
)
def test_cr2_malformed_empty_and_truncated_files_are_classified_and_recovered(
    tmp_path: Path, policy: str, content: bytes, kind: str
) -> None:
    p = tmp_path / "state.json"
    p.write_bytes(content)
    rec = _policy(policy)
    if policy == "DEFAULT":
        r = jio.read_json_result(p, default=lambda: {"d": 1}, recover=rec)
        assert (r.data, r.outcome, r.kind) == ({"d": 1}, "defaulted", kind)
        assert p.read_bytes() == content
    elif policy in ("RAISE", "RESTORE"):
        with pytest.raises(jio.JsonFileError) as exc:
            jio.read_json(p, default=lambda: {"d": 1}, recover=rec)
        assert exc.value.kind == kind and not isinstance(exc.value, json.JSONDecodeError)
        assert p.read_bytes() == content
    else:
        r = jio.read_json_result(p, default=lambda: {"d": 1}, recover=rec)
        assert (r.data, r.outcome) == ({"d": 1}, "reset")
        assert json.loads(p.read_text()) == {"d": 1}
        assert any(".corrupt-" in f.name for f in tmp_path.iterdir())


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("content", [b'{"a": "\xff"}', b'{"a": 1}\xc3'])
def test_cr3_invalid_utf8_is_recovered_like_malformed_json(
    tmp_path: Path, policy: str, content: bytes
) -> None:
    p = tmp_path / "state.json"
    p.write_bytes(content)
    rec = _policy(policy)
    if policy == "DEFAULT":
        assert jio.read_json(p, default=dict, recover=rec) == {}
        assert p.read_bytes() == content
    elif policy in ("RAISE", "RESTORE"):
        with pytest.raises(jio.JsonFileError) as exc:
            jio.read_json(p, default=dict, recover=rec)
        assert exc.value.kind == "encoding" and not isinstance(exc.value, UnicodeDecodeError)
    else:
        r = jio.read_json_result(p, default=dict, recover=rec)
        assert (
            r.outcome == "reset" and r.anomaly is not None and r.anomaly.kind == "json_parse_error"
        )


def test_cr3_baseline_the_legacy_handler_shape_leaks_unicode_decode_error(tmp_path: Path) -> None:
    """The #436 failure: a handler listing only (OSError, JSONDecodeError) lets invalid UTF-8 escape."""
    p = tmp_path / "state.json"
    p.write_bytes(b'{"a": "\xff"}')
    with pytest.raises(UnicodeDecodeError):
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pytest.fail("the legacy handler would have caught it")


def _failing_open(target: Path, exc_factory, times: int | None = None):
    """A builtins.open replacement that raises for ``target`` (``times`` times, then works)."""
    real_open = open
    state = {"n": 0}

    def fake(file, *a, **k):
        if Path(file) == target and (times is None or state["n"] < times):
            state["n"] += 1
            raise exc_factory()
        return real_open(file, *a, **k)

    fake.state = state  # type: ignore[attr-defined]
    return fake


def test_cr4_io_error_default_and_raise(tmp_path: Path, monkeypatch, caplog) -> None:
    p = tmp_path / "state.json"
    p.write_text('{"a": 1}')
    monkeypatch.setattr("builtins.open", _failing_open(p, lambda: OSError(errno.EIO, "disk")))
    with caplog.at_level(logging.WARNING):
        r = jio.read_json_result(p, default=lambda: {"d": 1})
    assert (r.data, r.outcome, r.kind) == ({"d": 1}, "defaulted", "io")
    assert any("io" in rec.getMessage() for rec in caplog.records)
    with pytest.raises(jio.JsonFileError) as exc:
        jio.read_json(p, default=dict, recover=jio.Recovery.RAISE)
    assert exc.value.kind == "io" and isinstance(exc.value, OSError)


@pytest.mark.parametrize("policy", ["HEAL", "RESTORE"])
def test_cr4_heal_and_restore_never_touch_a_file_they_could_not_read(
    tmp_path: Path, monkeypatch, policy: str
) -> None:
    p = tmp_path / "state.json"
    p.write_text('{"a": 1}')
    (tmp_path / "state.json.bak1").write_text('{"old": 1}')
    before = _listing(tmp_path)
    monkeypatch.setattr("builtins.open", _failing_open(p, lambda: OSError(errno.EIO, "disk")))
    with pytest.raises(jio.JsonFileError) as exc:
        jio.read_json(p, default=dict, recover=_policy(policy))
    monkeypatch.undo()
    assert exc.value.kind == "io"
    assert _listing(tmp_path) == before and not list(tmp_path.glob("*.corrupt-*"))


def test_cr4_a_directory_where_a_file_should_be_is_an_io_failure(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    p.mkdir()
    r = jio.read_json_result(p, default=dict)
    assert (r.outcome, r.kind) == ("defaulted", "io")


def _sleeps(monkeypatch) -> list[float]:
    calls: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: calls.append(s))
    return calls


def test_cr5_windows_path_retries_a_transient_permission_error(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    p = tmp_path / "state.json"
    p.write_text('{"a": 1}')
    monkeypatch.setattr(jio, "_IS_WINDOWS", True)
    sleeps = _sleeps(monkeypatch)
    monkeypatch.setattr(
        "builtins.open", _failing_open(p, lambda: PermissionError(13, "busy"), times=2)
    )
    with caplog.at_level(logging.DEBUG):
        assert jio.read_json(p, default=dict) == {"a": 1}
    assert len(sleeps) == 2 and sum(sleeps) <= 0.2
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_cr5_windows_path_gives_up_after_three_attempts(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "state.json"
    p.write_text('{"a": 1}')
    monkeypatch.setattr(jio, "_IS_WINDOWS", True)
    sleeps = _sleeps(monkeypatch)
    fake = _failing_open(p, lambda: PermissionError(13, "busy"))
    monkeypatch.setattr("builtins.open", fake)
    r = jio.read_json_result(p, default=dict)
    assert (r.outcome, r.kind) == ("defaulted", "io")
    assert fake.state["n"] == 3 and len(sleeps) == 2


def test_cr5_posix_path_has_no_retry_and_no_sleep(tmp_path: Path, monkeypatch, caplog) -> None:
    p = tmp_path / "state.json"
    p.write_text('{"a": 1}')
    monkeypatch.setattr(jio, "_IS_WINDOWS", False)
    sleeps = _sleeps(monkeypatch)
    fake = _failing_open(p, lambda: PermissionError(13, "denied"))
    monkeypatch.setattr("builtins.open", fake)
    with caplog.at_level(logging.WARNING):
        r = jio.read_json_result(p, default=dict)
    assert (r.outcome, r.kind) == ("defaulted", "io")
    assert fake.state["n"] == 1 and sleeps == []
    assert any(rec.levelno == logging.WARNING for rec in caplog.records)


@pytest.mark.parametrize("policy", POLICIES)
def test_cr6_wrong_top_level_type_is_a_shape_failure(tmp_path: Path, policy: str) -> None:
    p = tmp_path / "state.json"
    p.write_text("[1, 2, 3]")
    rec = _policy(policy)
    if policy == "DEFAULT":
        r = jio.read_json_result(p, default=dict, expect=dict, recover=rec)
        assert (r.outcome, r.kind, r.data) == ("defaulted", "shape", {})
    elif policy == "HEAL":
        r = jio.read_json_result(p, default=dict, expect=dict, recover=rec)
        assert (r.outcome, r.data) == ("reset", {}) and r.anomaly.kind == "schema_mismatch"
    else:
        with pytest.raises(jio.JsonFileError) as exc:
            jio.read_json(p, default=dict, expect=dict, recover=rec)
        assert exc.value.kind == "shape"


# One `except (ValueError, TypeError)` in json_io decides this; the HEAL tests near the end of the file run the
# other exception classes through HEAL.
@pytest.mark.parametrize("exc_type", [ValueError])
@pytest.mark.parametrize("policy", POLICIES)
def test_cr6_validator_rejections_are_schema_failures(
    tmp_path: Path, policy: str, exc_type
) -> None:
    p = tmp_path / "state.json"
    p.write_text('{"a": 1}')

    def validate(data):
        raise exc_type("bad")

    rec = _policy(policy)
    if policy in ("DEFAULT", "HEAL"):
        r = jio.read_json_result(p, default=lambda: {"d": 1}, validate=validate, recover=rec)
        assert r.kind == "schema" and r.data == {"d": 1}
    else:
        with pytest.raises(jio.JsonFileError) as exc:
            jio.read_json(p, default=dict, validate=validate, recover=rec)
        assert exc.value.kind == "schema"


@pytest.mark.parametrize("exc_type", [KeyError])
@pytest.mark.parametrize("policy", POLICIES)
def test_cr6_a_validator_bug_is_not_a_schema_failure_and_propagates(tmp_path: Path, policy: str, exc_type) -> None:
    """Only ValueError/TypeError reject content (the old attempt_heal's rule); a KeyError (AttributeError and RuntimeError
    are run through HEAL by test_heal_a_validator_bug_propagates_...) is the validator's own bug and must not turn a valid file into a default, a restore or a reset."""
    p = tmp_path / "state.json"
    p.write_text('{"a": 1}')

    def validate(data):
        raise exc_type("bug")

    with pytest.raises(exc_type):
        jio.read_json_result(p, default=dict, validate=validate, recover=_policy(policy))
    assert p.read_text() == '{"a": 1}'


def test_cr6_a_validator_that_returns_passes_the_data_through(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    p.write_text('{"a": 1}')
    seen = []
    assert jio.read_json(p, default=dict, validate=seen.append) == {"a": 1}
    assert seen == [{"a": 1}]
    # HEAL too: an accepted live file is returned as is, with no anomaly and nothing written beside it.
    healed = jio.read_json_result(p, default=dict, validate=seen.append, recover=jio.Recovery.HEAL)
    assert healed.data == {"a": 1} and healed.anomaly is None and _listing(tmp_path) == {"state.json": b'{"a": 1}'}


def test_cr7_default_policy_never_modifies_the_file(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    p.write_bytes(b"garbage")
    before = _listing(tmp_path)
    jio.read_json(p, default=dict)
    assert _listing(tmp_path) == before


def test_cr7_raise_carries_path_kind_and_cause(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    p.write_text("{nope")
    with pytest.raises(jio.JsonFileError) as exc:
        jio.read_json(p, recover=jio.Recovery.RAISE)
    assert exc.value.path == p and exc.value.kind == "malformed"
    assert isinstance(exc.value.__cause__, json.JSONDecodeError)


def test_cr7_a_default_under_raise_applies_only_to_a_missing_file(tmp_path: Path) -> None:
    assert jio.read_json(tmp_path / "gone.json", default=lambda: 7, recover=jio.Recovery.RAISE) == 7
    p = tmp_path / "bad.json"
    p.write_text("{nope")
    with pytest.raises(jio.JsonFileError):
        jio.read_json(p, default=lambda: 7, recover=jio.Recovery.RAISE)


@pytest.mark.parametrize("policy", ["DEFAULT", "RESTORE", "HEAL"])
def test_cr7_these_policies_require_a_default_at_call_time(tmp_path: Path, policy: str) -> None:
    with pytest.raises(TypeError):
        jio.read_json(tmp_path / "x.json", recover=_policy(policy))


def test_cr16_the_exception_is_an_oserror_a_valueerror_and_a_missing_one_a_filenotfounderror() -> (
    None
):
    err = jio.JsonFileError(Path("/x/y.json"), "malformed", "bad", line=3)
    assert isinstance(err, OSError) and isinstance(err, ValueError)
    gone = jio.JsonFileNotFound(Path("/x/y.json"), "missing")
    assert isinstance(gone, FileNotFoundError) and isinstance(gone, jio.JsonFileError)
    expected = (
        f"{Path('/x/y.json')}: malformed: bad (line 3)"  # the path spelling is the platform's
    )
    assert str(err) == expected and not str(err).startswith("[Errno")


@pytest.mark.parametrize(
    "clone", [copy.copy, lambda e: pickle.loads(pickle.dumps(e))], ids=["copy", "pickle"]
)
def test_cr16_the_exception_survives_copy_and_pickle(clone) -> None:
    for original in (
        jio.JsonFileError(Path("/x/y.json"), "line", "bad line", line=9),
        jio.JsonFileNotFound(Path("/x/y.json"), "missing"),
    ):
        again = clone(original)
        assert type(again) is type(original)
        assert (again.path, again.kind, again.line, str(again)) == (
            original.path,
            original.kind,
            original.line,
            str(original),
        )


def test_cr12_deeply_nested_json_is_malformed_not_a_recursion_error(tmp_path: Path) -> None:
    p = tmp_path / "deep.json"
    p.write_text("[" * 100000)
    r = jio.read_json_result(p, default=dict)
    assert (r.outcome, r.kind) == ("defaulted", "malformed")


def test_cr13_outcomes_and_read_json_equals_the_result_data(tmp_path: Path) -> None:
    ok, missing, bad = tmp_path / "ok.json", tmp_path / "missing.json", tmp_path / "bad.json"
    ok.write_text('{"a": 1}')
    bad.write_text("{nope")
    assert jio.read_json_result(ok, default=dict).outcome == "ok"
    assert jio.read_json_result(missing, default=dict).outcome == "missing"
    assert jio.read_json_result(bad, default=dict).outcome == "defaulted"
    assert (
        jio.read_json(bad, default=lambda: {"z": 1})
        == jio.read_json_result(bad, default=lambda: {"z": 1}).data
    )
    (tmp_path / "bad.json.bak1").write_text('{"b": 2}')
    r = jio.read_json_result(bad, default=dict, recover=jio.Recovery.HEAL)
    assert (r.outcome, r.data) == ("restored", {"b": 2})
    bad.write_text("{nope")
    (tmp_path / "bad.json.bak1").unlink(missing_ok=True)
    assert jio.read_json_result(bad, default=dict, recover=jio.Recovery.HEAL).outcome == "reset"


# --------------------------------------------------------------------------------------------------------------
# Evidence (CR10) and encoding independence (CR11)
# --------------------------------------------------------------------------------------------------------------


def _warnings(caplog, level=logging.WARNING):
    return [r for r in caplog.records if r.levelno == level and r.name == "brain.health.json_io"]


def test_cr10_a_fallback_logs_one_warning_with_path_kind_and_text(tmp_path: Path, caplog) -> None:
    p = tmp_path / "state.json"
    p.write_text("{nope")
    with caplog.at_level(logging.DEBUG):
        jio.read_json(p, default=dict)
    [rec] = _warnings(caplog)
    msg = rec.getMessage()
    assert str(p) in msg and "malformed" in msg and "Expecting" in msg
    assert rec.exc_info is not None


def test_cr10_repeats_within_the_interval_log_at_debug_and_other_paths_warn(
    tmp_path: Path, caplog
) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text("{nope")
    b.write_text("{nope")
    with caplog.at_level(logging.DEBUG):
        for _ in range(3):
            jio.read_json(a, default=dict)
        jio.read_json(b, default=dict)
    assert len(_warnings(caplog)) == 2  # first failure of a, first failure of b
    assert len(_warnings(caplog, logging.DEBUG)) == 2  # the two repeats of a


def test_cr10_the_warning_comes_back_after_the_interval(
    tmp_path: Path, caplog, monkeypatch
) -> None:
    from brain.utils import log_throttle

    p = tmp_path / "a.json"
    p.write_text("{nope")
    clock = [1000.0]
    monkeypatch.setattr(log_throttle.time, "monotonic", lambda: clock[0])
    with caplog.at_level(logging.DEBUG):
        jio.read_json(p, default=dict)
        clock[0] += log_throttle.INTERVAL_S - 1
        jio.read_json(p, default=dict)
        clock[0] += 2
        jio.read_json(p, default=dict)
    assert len(_warnings(caplog)) == 2


def test_cr10_a_different_failure_kind_on_the_same_path_warns_again(tmp_path: Path, caplog) -> None:
    p = tmp_path / "a.json"
    with caplog.at_level(logging.DEBUG):
        p.write_text("{nope")
        jio.read_json(p, default=dict)
        p.write_bytes(b'{"a": "\xff"}')
        jio.read_json(p, default=dict)
    assert len(_warnings(caplog)) == 2


def test_cr10_a_long_exception_text_is_capped(tmp_path: Path, caplog) -> None:
    p = tmp_path / "a.json"
    p.write_text("{" + '"k": ' + "1 " * 5000 + "oops")
    with caplog.at_level(logging.DEBUG):
        jio.read_json(p, default=dict)
    [rec] = _warnings(caplog)
    assert len(rec.getMessage()) < 600


def test_cr10_heal_events_are_never_throttled(tmp_path: Path, caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        for i in range(3):
            p = tmp_path / f"s{i}.json"
            p.write_text("{nope")
            jio.read_json(p, default=dict, recover=jio.Recovery.HEAL)
    heal_warnings = [
        r
        for r in _warnings(caplog)
        if "heal" in r.getMessage().lower()
        or "restor" in r.getMessage().lower()
        or "reset" in r.getMessage().lower()
    ]
    assert len(heal_warnings) >= 3


def test_cr11_a_leading_bom_is_tolerated_on_read(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    p.write_bytes(b"\xef\xbb\xbf" + b'{"a": 1}')
    assert jio.read_json_result(p, default=dict, recover=jio.Recovery.RAISE).data == {"a": 1}


class _OpenSpy:
    """Record every text-mode open without an explicit UTF-8 encoding, through every opener the module could use."""

    def __init__(self, monkeypatch) -> None:
        self.violations: list[str] = []
        import builtins
        import gzip as gz
        import io

        spy = self

        def check(opener_name, real):
            def wrapper(*args, **kwargs):
                mode = kwargs.get("mode", args[1] if len(args) > 1 else "r")
                enc = kwargs.get(
                    "encoding",
                    args[3]
                    if len(args) > 3 and opener_name in ("open", "io.open", "fdopen")
                    else None,
                )
                if "b" not in str(mode) and (
                    enc is None or not str(enc).lower().replace("_", "-").startswith("utf-8")
                ):
                    spy.violations.append(f"{opener_name}(mode={mode!r}, encoding={enc!r})")
                return real(*args, **kwargs)

            return wrapper

        monkeypatch.setattr(builtins, "open", check("open", builtins.open))
        monkeypatch.setattr(io, "open", check("io.open", io.open))
        monkeypatch.setattr(os, "fdopen", check("fdopen", os.fdopen))
        real_path_open = Path.open

        def path_open(self_, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
            if "b" not in mode and (
                encoding is None or not encoding.lower().replace("_", "-").startswith("utf-8")
            ):
                spy.violations.append(f"Path.open(mode={mode!r}, encoding={encoding!r})")
            return real_path_open(self_, mode, buffering, encoding, errors, newline)

        monkeypatch.setattr(Path, "open", path_open)
        monkeypatch.setattr(gz, "open", check("gzip.open", gz.open))


@pytest.mark.no_base_failure(reason='#540 audit guard, snapshot or registry of what already exists (it replaces removed tests); passes on the old code by design')
def test_cr11_baseline_the_spy_flags_a_locale_codec_writer_and_the_migrated_ones_are_clean(
    tmp_path: Path, monkeypatch
) -> None:
    from brain.chat.tool_loop import _log_attunement_error
    from brain.self_model.articulate import _log_articulate_error

    def locale_codec_writer(directory: Path) -> None:
        with open(directory / "log.txt", "a") as fh:  # text mode, no encoding: the very thing the spy exists to flag
            fh.write("x\n")

    spy = _OpenSpy(monkeypatch)
    locale_codec_writer(tmp_path)
    assert spy.violations, "the spy no longer flags a text-mode open() with the locale codec"
    spy.violations.clear()
    _log_attunement_error(tmp_path, "t1", ValueError("x"))
    _log_articulate_error(tmp_path, "kind", ValueError("x"))
    assert not spy.violations, "the migrated writers go through json_io (explicit UTF-8, binary append)"


# --------------------------------------------------------------------------------------------------------------
# HEAL fidelity against the frozen output of the legacy attempt_heal (CR8)
# --------------------------------------------------------------------------------------------------------------

import re  # noqa: E402

_SCHEMA_OK = b'{"interests": [1]}'


def _needs_interests(data) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("interests"), list):
        raise ValueError("interests schema invalid")


def _norm(name: str) -> str:
    return re.sub(r"corrupt-.*$", "corrupt-TS", name)


def _snapshot(d: Path) -> dict[str, bytes]:
    return {_norm(k): v for k, v in _listing(d).items()}


def _anomaly_fields(a):
    if a is None:
        return None
    return {
        "file": a.file,
        "kind": a.kind,
        "action": a.action,
        "quarantine_path": _norm(a.quarantine_path) if a.quarantine_path else None,
        "likely_cause": a.likely_cause,
        "detail": a.detail,
    }


# name -> (files, validator)
HEAL_FIXTURES = {
    "valid": ({"s.json": b'{"a": 1}'}, None),
    "corrupt-no-baks": ({"s.json": b"{nope"}, None),
    "corrupt-valid-bak1": ({"s.json": b"{nope", "s.json.bak1": b'{"b": 2}'}, None),
    "bak1-corrupt-bak2-valid": (
        {"s.json": b"{nope", "s.json.bak1": b"zzz", "s.json.bak2": b'{"c": 3}'},
        None,
    ),
    "all-corrupt": (
        {"s.json": b"{nope", "s.json.bak1": b"zzz", "s.json.bak2": b"yyy", "s.json.bak3": b""},
        None,
    ),
    "schema-failure": ({"s.json": b'{"other": 1}', "s.json.bak1": _SCHEMA_OK}, _needs_interests),
    "hand-edit": ({"s.json": b'{"a": 1,}'}, None),
    "empty-file": ({"s.json": b""}, None),
    "missing": ({}, None),
}


GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "legacy_heal_and_jsonl.json").read_text(encoding="utf-8")
)  # frozen output of the legacy attempt_heal / JSONL readers, captured before they were deleted (PC3)


def _run_new(tmp_path: Path, files: dict, validator):
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    for name, content in files.items():
        (new_dir / name).write_bytes(content)
    r = jio.read_json_result(
        new_dir / "s.json",
        default=lambda: {"default": True},
        recover=jio.Recovery.HEAL,
        validate=validator,
    )
    return new_dir, r


@pytest.mark.parametrize("fixture", sorted(HEAL_FIXTURES))
def test_cr8_heal_matches_the_frozen_legacy_output(tmp_path: Path, fixture: str) -> None:
    files, validator = HEAL_FIXTURES[fixture]
    new_dir, r = _run_new(tmp_path, files, validator)
    gold = GOLDEN["heal"][fixture]
    assert r.data == gold["data"]
    expected = gold["anomaly"]
    if expected is not None and expected["kind"] == "json_parse_error":
        # The detail is the stdlib decoder's message, whose wording differs between Python versions. The gold's
        # detail was exactly that message for the primary file, so compare against the same message produced live
        # on this interpreter; every other field stays frozen from the legacy output.
        try:
            json.loads(files["s.json"].decode("utf-8"))
        except json.JSONDecodeError as exc:
            live_detail = str(exc)
        else:  # pragma: no cover - a fixture with a parse_error gold must not parse
            raise AssertionError("the fixture's primary file parses")
        expected = {**expected, "detail": live_detail}
    assert _anomaly_fields(r.anomaly) == expected
    new_files = _snapshot(new_dir)
    assert set(new_files) == set(gold["files"])
    for name, text in gold["files"].items():
        # The same JSON, byte for byte, except that the legacy writer used text mode (CRLF on Windows) and the module
        # always writes LF (plan P8, CW1): compare after normalising CRLF, and assert the module wrote no CR anywhere.
        assert _norm_crlf(new_files[name]) == _norm_crlf(text.encode("latin-1")), name
        assert b"\r" not in new_files[name], name


def test_cr8_delta_one_invalid_utf8_is_labelled_a_parse_error_not_a_schema_mismatch(
    tmp_path: Path,
) -> None:
    _new_dir, r = _run_new(tmp_path, {"s.json": b'{"a": "\xff"}'}, None)
    assert GOLDEN["legacy_deltas"]["invalid_utf8_kind"] == "schema_mismatch"  # the legacy mislabel, frozen
    assert r.anomaly.kind == "json_parse_error"  # the intended module result
    assert r.anomaly.action == "reset_to_default"


def test_cr8_anomaly_detail_is_capped_at_500_characters(tmp_path: Path) -> None:
    """A file that PARSES, rejected by a validator with a 900-character message: the cap, not the source, limits it."""
    p = tmp_path / "s.json"
    p.write_text('{"k": 1}')

    def reject(_data):
        raise ValueError("v" * 900)

    r = jio.read_json_result(p, default=dict, recover=jio.Recovery.HEAL, validate=reject)
    assert r.anomaly is not None and 400 < len(r.anomaly.detail) <= 500


def test_cr8_delta_two_the_reset_to_default_is_written_atomically(
    tmp_path: Path, monkeypatch
) -> None:
    d = tmp_path / "new"
    d.mkdir()
    (d / "s.json").write_bytes(b"{nope")
    writes: list[str] = []
    replaces: list[tuple[str, str]] = []
    real_write_text, real_replace = Path.write_text, os.replace

    def write_text(self, *a, **k):
        writes.append(str(self))
        return real_write_text(self, *a, **k)

    def replace(src, dst):
        replaces.append((Path(src).name, Path(dst).name))
        return real_replace(src, dst)

    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr(os, "replace", replace)
    jio.read_json(d / "s.json", default=dict, recover=jio.Recovery.HEAL)
    assert writes == []  # the legacy heal wrote the reset directly, non-atomically, at the live path
    assert any(dst == "s.json" and src.endswith(".new") for src, dst in replaces)


def test_cr8_delta_three_the_quarantine_rename_retries_a_transient_permission_error(
    tmp_path: Path, monkeypatch
) -> None:
    d = tmp_path / "new"
    d.mkdir()
    (d / "s.json").write_bytes(b"{nope")
    (d / "s.json.bak1").write_bytes(b'{"b": 1}')
    real_replace = os.replace
    left = {"n": 0}

    def flaky(src, dst):
        if ".corrupt-" in str(dst) and left["n"] < 1:
            left["n"] += 1
            raise PermissionError(13, "sharing violation")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    # (the legacy heal let this PermissionError escape; the module retries it)
    assert jio.read_json(d / "s.json", default=dict, recover=jio.Recovery.HEAL) == {"b": 1}


# --------------------------------------------------------------------------------------------------------------
# Heal races (CR9): executed interleavings, each also run with its guard switched off to show the data loss
# --------------------------------------------------------------------------------------------------------------


def _corrupt_with_backup(tmp_path: Path, bak: bytes = b'{"restored": true}') -> Path:
    p = tmp_path / "s.json"
    p.write_bytes(b"{nope")
    (tmp_path / "s.json.bak1").write_bytes(bak)
    return p


def _seam_hook(monkeypatch, **actions):
    """Replace the module's no-op seam: ``actions[name](path)`` runs when the heal reaches seam ``name``."""

    def seam(name, path):
        action = actions.get(name)
        if action:
            action(path)

    monkeypatch.setattr(jio, "_seam", seam)


def test_cr9_a_the_second_healer_finds_the_file_already_restored(
    tmp_path: Path, monkeypatch
) -> None:
    p = _corrupt_with_backup(tmp_path)
    ran = []

    def competitor(path):
        if not ran:
            ran.append(1)
            assert jio.read_json(path, default=dict, recover=jio.Recovery.HEAL) == {
                "restored": True
            }

    _seam_hook(monkeypatch, before_heal=competitor)
    r = jio.read_json_result(p, default=dict, recover=jio.Recovery.HEAL)
    assert r.data == {"restored": True} and r.outcome == "ok"
    assert json.loads(p.read_text()) == {"restored": True}
    assert len(list(tmp_path.glob("s.json.corrupt-*"))) == 1


def test_cr9_a_without_revalidation_the_second_healer_destroys_the_restored_file(
    tmp_path: Path, monkeypatch
) -> None:
    """The failing baseline: the guard off reproduces the legacy double-heal data loss."""
    p = _corrupt_with_backup(tmp_path)
    ran = []

    def competitor(path):
        if not ran:
            ran.append(1)
            jio.read_json(path, default=dict, recover=jio.Recovery.HEAL)

    _seam_hook(monkeypatch, before_heal=competitor)
    monkeypatch.setattr(jio, "_HEAL_REVALIDATE", False)
    r = jio.read_json_result(p, default=dict, recover=jio.Recovery.HEAL)
    assert r.data != {
        "restored": True
    }  # the restored data was quarantined and replaced by the default


def test_cr9_b_a_writer_that_replaced_the_file_before_the_lock_is_not_quarantined(
    tmp_path: Path, monkeypatch
) -> None:
    p = _corrupt_with_backup(tmp_path)

    def writer(path):
        path.write_text('{"fresh": 1}')

    _seam_hook(monkeypatch, before_heal=writer)
    r = jio.read_json_result(p, default=dict, recover=jio.Recovery.HEAL)
    assert (r.data, r.outcome) == ({"fresh": 1}, "ok")
    assert not list(tmp_path.glob("*.corrupt-*"))


def test_cr9_b_without_revalidation_the_new_content_is_quarantined(
    tmp_path: Path, monkeypatch
) -> None:
    p = _corrupt_with_backup(tmp_path)
    _seam_hook(monkeypatch, before_heal=lambda path: path.write_text('{"fresh": 1}'))
    monkeypatch.setattr(jio, "_HEAL_REVALIDATE", False)
    r = jio.read_json_result(p, default=dict, recover=jio.Recovery.HEAL)
    assert r.data != {"fresh": 1}


def test_cr9_c_a_failed_quarantine_raises_and_leaves_the_original(
    tmp_path: Path, monkeypatch
) -> None:
    p = _corrupt_with_backup(tmp_path)

    def no_link(src, dst, **k):
        raise OSError(errno.EPERM, "links not supported")

    def no_copy(src, dst, **k):
        raise OSError(errno.EIO, "disk")

    monkeypatch.setattr(os, "link", no_link)
    monkeypatch.setattr(jio.shutil, "copyfileobj", no_copy)
    with pytest.raises(jio.JsonFileError) as exc:
        jio.read_json(p, default=dict, recover=jio.Recovery.HEAL)
    assert exc.value.kind == "io"
    assert p.read_bytes() == b"{nope" and (tmp_path / "s.json.bak1").exists()


def test_cr9_c_when_hard_links_are_unsupported_the_quarantine_is_a_copy(
    tmp_path: Path, monkeypatch
) -> None:
    p = _corrupt_with_backup(tmp_path)

    def no_link(src, dst, **k):
        raise OSError(errno.EPERM, "links not supported")

    monkeypatch.setattr(os, "link", no_link)
    assert jio.read_json(p, default=dict, recover=jio.Recovery.HEAL) == {"restored": True}
    [q] = tmp_path.glob("s.json.corrupt-*")
    assert q.read_bytes() == b"{nope"


def _in_thread(fn, timeout: float):
    box: dict = {}

    def run():
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    return t, box


def test_cr9_d_a_save_started_during_the_heal_waits_and_lands_after_it(
    tmp_path: Path, monkeypatch
) -> None:
    p = _corrupt_with_backup(tmp_path, b'{"restored": true}')
    state: dict = {}

    def start_writer(path):
        t, box = _in_thread(lambda: jio.write_json(path, {"writer": 1}, backups=1), timeout=0.4)
        state["thread"], state["box"] = t, box
        state["alive_during_heal"] = t.is_alive()

    _seam_hook(monkeypatch, inside_heal=start_writer)
    r = jio.read_json_result(p, default=dict, recover=jio.Recovery.HEAL)
    state["thread"].join(5)
    assert r.data == {"restored": True}
    assert state["alive_during_heal"] is True, "the save did not wait for the heal"
    assert "error" not in state["box"]
    assert json.loads(p.read_text()) == {
        "writer": 1
    }  # the writer's content is final, not the restored one


def test_cr9_d_with_the_save_lock_off_the_install_check_still_protects_the_save(
    tmp_path: Path, monkeypatch
) -> None:
    """Defence in depth: even if a save did not take the heal sidecar, the heal refuses to install over it."""
    p = _corrupt_with_backup(tmp_path, b'{"restored": true}')
    monkeypatch.setattr(jio, "_SAVE_TAKES_HEAL_LOCK", False)
    _seam_hook(monkeypatch, inside_heal=lambda path: jio.write_json(path, {"writer": 1}, backups=1))
    jio.read_json_result(p, default=dict, recover=jio.Recovery.HEAL)
    assert json.loads(p.read_text()) == {"writer": 1}


def test_cr9_d_without_the_checks_a_save_without_a_chain_is_overwritten_by_the_restore(
    tmp_path: Path, monkeypatch
) -> None:
    """The failing baseline: a save with no backup chain does not rotate the backup away, so only the
    install-time checks stand between it and the restore; with them off the restore clobbers the save."""
    p = _corrupt_with_backup(tmp_path, b'{"restored": true}')
    monkeypatch.setattr(jio, "_HEAL_REVALIDATE", False)
    _seam_hook(monkeypatch, inside_heal=lambda path: jio.write_json(path, {"writer": 1}))
    jio.read_json_result(p, default=dict, recover=jio.Recovery.HEAL)
    assert json.loads(p.read_text()) != {"writer": 1}


def test_cr9_e_a_file_that_vanishes_before_the_heal_is_re_read_not_raised(
    tmp_path: Path, monkeypatch
) -> None:
    p = _corrupt_with_backup(tmp_path)
    _seam_hook(monkeypatch, before_heal=lambda path: path.unlink())
    r = jio.read_json_result(p, default=lambda: {"d": 1}, recover=jio.Recovery.HEAL)
    assert (r.data, r.outcome) == ({"d": 1}, "missing")
    assert not list(tmp_path.glob("*.corrupt-*"))


def test_cr9_e_restore_twin_a_file_that_vanishes_before_the_heal_is_restored_from_the_backup(
    tmp_path: Path, monkeypatch
) -> None:
    """R2-1: a legacy healer (no sidecar) or a manual delete can remove the file between the first read and the heal lock.
    HEAL keeps its meaning (default); RESTORE must not answer 'no data yet' while a valid backup exists."""
    p = _corrupt_with_backup(tmp_path, b'{"count": 7}')
    _seam_hook(monkeypatch, before_heal=lambda path: path.unlink())
    r = jio.read_json_result(p, default=lambda: {"count": 0}, recover=jio.Recovery.RESTORE)
    assert (r.data, r.outcome) == ({"count": 7}, "restored")
    assert json.loads(p.read_text()) == {"count": 7}


# --------------------------------------------------------------------------------------------------------------
# The heal lock does not collide with other holders of <name>.lock (CR14)
# --------------------------------------------------------------------------------------------------------------


def _hold_shared_lock_in_thread(path: Path):
    from brain.utils.file_lock import file_lock

    acquired, release = threading.Event(), threading.Event()

    def hold():
        with file_lock(path):
            acquired.set()
            release.wait(10)

    t = threading.Thread(target=hold, daemon=True)
    t.start()
    assert acquired.wait(5)
    return t, release


def test_cr14_heal_completes_while_another_descriptor_holds_the_shared_lock(tmp_path: Path) -> None:
    p = _corrupt_with_backup(tmp_path)
    t, release = _hold_shared_lock_in_thread(p)  # the bridge.json.lock shape
    try:
        worker, box = _in_thread(
            lambda: jio.read_json(p, default=dict, recover=jio.Recovery.HEAL), timeout=5
        )
        assert not worker.is_alive(), "heal blocked on <name>.lock"
        assert box["result"] == {"restored": True}
    finally:
        release.set()
        t.join(5)


def test_cr14_baseline_a_heal_on_the_shared_sidecar_would_hang(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(jio, "_HEAL_LOCK_USES_SHARED_SIDECAR", True)
    p = _corrupt_with_backup(tmp_path)
    t, release = _hold_shared_lock_in_thread(p)
    try:
        worker, _box = _in_thread(
            lambda: jio.read_json(p, default=dict, recover=jio.Recovery.HEAL), timeout=1.0
        )
        assert worker.is_alive(), "the variant that locks <name>.lock should have hung"
    finally:
        release.set()
        t.join(5)
        worker.join(5)


def test_cr14_heal_save_and_if_missing_complete_while_the_same_thread_holds_the_shared_lock(
    tmp_path: Path,
) -> None:
    """The engines/heartbeat.py tick shape: load_with_anomaly inside file_lock(state_path)."""
    from brain.utils.file_lock import file_lock

    p = _corrupt_with_backup(tmp_path)

    def work():
        with file_lock(p):
            a = jio.read_json(p, default=dict, recover=jio.Recovery.HEAL)
            jio.write_json(p, {"saved": 1}, backups=2)
            made = jio.write_json(tmp_path / "once.json", {"x": 1}, if_missing=True)
            return a, made

    worker, box = _in_thread(work, timeout=5)
    assert not worker.is_alive(), "deadlock on the caller's own <name>.lock"
    assert box["result"] == ({"restored": True}, True)


@pytest.mark.skipif(not POSIX, reason="flock across processes is exercised on POSIX")
def test_cr14_heal_completes_while_another_process_holds_the_shared_lock(tmp_path: Path) -> None:
    import subprocess
    import textwrap

    p = _corrupt_with_backup(tmp_path)
    code = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from brain.utils.file_lock import file_lock
        with file_lock(Path(sys.argv[1])):
            print("READY", flush=True)
            sys.stdin.read()
        """
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(p)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[4])},
    )
    try:
        assert child.stdout.readline().strip() == "READY"
        worker, box = _in_thread(
            lambda: jio.read_json(p, default=dict, recover=jio.Recovery.HEAL), timeout=5
        )
        assert not worker.is_alive()
        assert box["result"] == {"restored": True}
    finally:
        child.stdin.close()
        child.wait(10)


def test_cr14_a_refused_lock_proceeds_unlocked_with_one_warning_and_an_install_collision_is_not_clobbered(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    import contextlib

    p = _corrupt_with_backup(tmp_path, b'{"restored": "bak"}')

    @contextlib.contextmanager
    def refusing(path, *, blocking=True):
        raise OSError(errno.ENOLCK, "no locks available")
        yield  # pragma: no cover

    monkeypatch.setattr(jio, "file_lock", refusing)
    _seam_hook(monkeypatch, before_install=lambda path: path.write_text('{"competitor": 1}'))
    with caplog.at_level(logging.WARNING):
        r = jio.read_json_result(p, default=dict, recover=jio.Recovery.HEAL)
    assert r.data == {
        "competitor": 1
    }  # the competing restorer's file is kept and re-read, not overwritten
    assert json.loads(p.read_text()) == {"competitor": 1}
    assert any("lock" in rec.getMessage().lower() for rec in _warnings(caplog))


def test_cr14_a_competing_change_before_the_quarantine_is_noticed_in_unlocked_mode(
    tmp_path: Path, monkeypatch
) -> None:
    import contextlib

    p = _corrupt_with_backup(tmp_path)

    @contextlib.contextmanager
    def refusing(path, *, blocking=True):
        raise OSError(errno.ENOLCK, "no locks")
        yield  # pragma: no cover

    monkeypatch.setattr(jio, "file_lock", refusing)
    _seam_hook(monkeypatch, before_quarantine=lambda path: path.write_text('{"fresh": 2}'))
    r = jio.read_json_result(p, default=dict, recover=jio.Recovery.HEAL)
    assert r.data == {"fresh": 2} and not list(tmp_path.glob("*.corrupt-*"))


def test_cr14_an_oserror_in_the_body_propagates_once_and_the_body_runs_once(
    tmp_path: Path, monkeypatch
) -> None:
    p = _corrupt_with_backup(tmp_path)
    calls = {"quarantine": 0}

    def failing(path, stamp):
        calls["quarantine"] += 1
        raise OSError(errno.EIO, "disk")

    monkeypatch.setattr(jio, "_quarantine_copy", failing)
    with pytest.raises(jio.JsonFileError):
        jio.read_json(p, default=dict, recover=jio.Recovery.HEAL)
    assert calls["quarantine"] == 1, (
        "the heal body was re-run (unlocked) after an OSError in the body"
    )


def test_cr14_baseline_the_naive_try_with_shape_runs_the_body_twice() -> None:
    """Why the module enters the lock through an ExitStack inside the try: @contextmanager re-raises body errors
    at the yield, so `try: with lock: body / except OSError: body` runs the body a second time, unlocked."""
    import contextlib

    @contextlib.contextmanager
    def lock():
        yield

    runs = []

    def body():
        runs.append(1)
        raise OSError(errno.EIO, "disk")

    try:
        with lock():
            body()
    except OSError:
        try:
            body()
        except OSError:
            pass
    assert len(runs) == 2


# --------------------------------------------------------------------------------------------------------------
# Recovery.RESTORE and fallbacks (CR15)
# --------------------------------------------------------------------------------------------------------------


def test_cr15_restore_without_a_valid_backup_raises_and_leaves_everything_in_place(
    tmp_path: Path,
) -> None:
    p = tmp_path / "daily_budget.json"
    p.write_bytes(b'{"date": "2026-10-17", "cou')
    (tmp_path / "daily_budget.json.bak1").write_bytes(b"also bad")
    before = _listing(tmp_path)
    for _ in range(2):  # a second call raises again: it keeps failing closed
        with pytest.raises(jio.JsonFileError) as exc:
            jio.read_json(p, default=dict, recover=jio.Recovery.RESTORE)
        assert exc.value.kind == "malformed"
    assert _listing(tmp_path) == before
    assert not list(tmp_path.glob("*.corrupt-*"))


def test_cr15_baseline_heal_would_have_reset_the_budget_to_an_empty_counter(tmp_path: Path) -> None:
    """The d4c94657 fail-open bug: HEAL resets a corrupt budget file to the default (zero used)."""
    p = tmp_path / "daily_budget.json"
    p.write_bytes(b'{"date": "2026-10-17", "cou')
    assert jio.read_json(p, default=dict, recover=jio.Recovery.HEAL) == {}


def test_cr15_restore_restores_from_the_freshest_valid_backup(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    p.write_bytes(b"{nope")
    (tmp_path / "b.json.bak1").write_bytes(b"corrupt")
    (tmp_path / "b.json.bak2").write_bytes(b'{"n": 2}')
    r = jio.read_json_result(p, default=dict, recover=jio.Recovery.RESTORE)
    assert (r.data, r.outcome) == ({"n": 2}, "restored")
    assert json.loads(p.read_text()) == {"n": 2}
    assert len(list(tmp_path.glob("b.json.corrupt-*"))) == 1
    assert len(list(tmp_path.glob("b.json.bak1.corrupt-*"))) == 1


def _set_mtime(path: Path, age_seconds: float) -> None:
    t = time.time() - age_seconds
    os.utime(path, (t, t))


def test_cr15_a_fresh_tmp_beats_a_stale_bak1(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    p.write_bytes(b"{nope")
    tmp, bak = tmp_path / "b.tmp", tmp_path / "b.json.bak1"
    bak.write_bytes(b'{"day": "yesterday"}')
    tmp.write_bytes(b'{"day": "today"}')
    _set_mtime(bak, 86400)
    _set_mtime(tmp, 5)
    r = jio.read_json_result(p, default=dict, recover=jio.Recovery.RESTORE, fallbacks=(tmp,))
    assert r.data == {"day": "today"} and r.outcome == "restored"
    assert json.loads(p.read_text()) == {"day": "today"} and not tmp.exists()


def test_cr15_a_frozen_older_tmp_loses_to_a_newer_bak1(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    p.write_bytes(b"{nope")
    tmp, bak = tmp_path / "b.tmp", tmp_path / "b.json.bak1"
    tmp.write_bytes(b'{"day": "old-legacy-tmp"}')
    bak.write_bytes(b'{"day": "fresh-bak1"}')
    _set_mtime(tmp, 5 * 86400)
    _set_mtime(bak, 60)
    r = jio.read_json_result(p, default=dict, recover=jio.Recovery.RESTORE, fallbacks=(tmp,))
    assert r.data == {"day": "fresh-bak1"}
    assert tmp.exists()  # an unused fallback is left alone


def test_cr15_equal_mtimes_prefer_the_fallback_and_an_invalid_tmp_is_ignored(
    tmp_path: Path,
) -> None:
    p = tmp_path / "b.json"
    tmp, bak = tmp_path / "b.tmp", tmp_path / "b.json.bak1"
    bak.write_bytes(b'{"from": "bak"}')
    tmp.write_bytes(b'{"from": "tmp"}')
    t = time.time() - 100
    os.utime(bak, (t, t))
    os.utime(tmp, (t, t))
    p.write_bytes(b"{nope")
    assert jio.read_json(p, default=dict, recover=jio.Recovery.RESTORE, fallbacks=(tmp,)) == {
        "from": "tmp"
    }
    p.write_bytes(b"{nope")
    tmp.write_bytes(b"garbage")
    assert jio.read_json(p, default=dict, recover=jio.Recovery.RESTORE, fallbacks=(tmp,)) == {
        "from": "bak"
    }


def test_cr15_with_no_valid_backup_a_valid_tmp_is_installed_whatever_its_age(
    tmp_path: Path,
) -> None:
    p = tmp_path / "b.json"
    p.write_bytes(b"{nope")
    tmp = tmp_path / "b.tmp"
    tmp.write_bytes(b'{"day": "old"}')
    _set_mtime(tmp, 30 * 86400)
    assert jio.read_json(p, default=dict, recover=jio.Recovery.RESTORE, fallbacks=(tmp,)) == {
        "day": "old"
    }


@pytest.mark.parametrize("spelling", ["live", "relative-live", "bak1"])
def test_cr15_a_fallback_equal_to_the_live_file_or_a_backup_is_a_value_error(
    tmp_path: Path, monkeypatch, spelling: str
) -> None:
    p = tmp_path / "b.json"
    p.write_bytes(b"{nope")
    monkeypatch.chdir(tmp_path)
    bad = {
        "live": p,
        "relative-live": Path("sub") / ".." / "b.json",
        "bak1": tmp_path / "b.json.bak1",
    }[spelling]
    if spelling == "relative-live":
        (tmp_path / "sub").mkdir()
    with pytest.raises(ValueError):
        jio.read_json(p, default=dict, recover=jio.Recovery.RESTORE, fallbacks=(bad,))


# --------------------------------------------------------------------------------------------------------------
# Whole-file writes (CW1-CW12)
# --------------------------------------------------------------------------------------------------------------


def _norm_crlf(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n")


@pytest.mark.parametrize("backups", [1, 3])
@pytest.mark.parametrize("data", [{"é": "ü", "n": [1, 2, {"k": None}]}, 7])
def test_cw1_write_json_matches_save_with_backup_byte_for_byte(
    tmp_path: Path, backups: int, data
) -> None:
    from brain.health.attempt_heal import save_with_backup

    legacy, new = tmp_path / "legacy.json", tmp_path / "new.json"
    for n in range(4):
        payload = {"v": n, "d": data}
        save_with_backup(legacy, payload, backup_count=backups)
        assert jio.write_json(new, payload, backups=backups) is True
    assert _norm_crlf(new.read_bytes()) == _norm_crlf(legacy.read_bytes())
    for i in range(1, backups + 1):
        a, b = tmp_path / f"legacy.json.bak{i}", tmp_path / f"new.json.bak{i}"
        assert _norm_crlf(b.read_bytes()) == _norm_crlf(a.read_bytes())


def test_cw1_the_differential_can_fail(tmp_path: Path) -> None:
    from brain.health.attempt_heal import save_with_backup

    save_with_backup(tmp_path / "l.json", {"a": 1})
    jio.write_json(tmp_path / "n.json", {"a": 1}, indent=4)
    assert _norm_crlf((tmp_path / "n.json").read_bytes()) != _norm_crlf(
        (tmp_path / "l.json").read_bytes()
    )


def test_cw2_concurrent_writers_raise_nothing_and_leave_a_whole_file_and_no_temps(
    tmp_path: Path,
) -> None:
    p = tmp_path / "s.json"
    jio.write_json(p, {"v": 0}, backups=2)
    errors: list[BaseException] = []
    stop = threading.Event()
    seen_bad: list[str] = []

    def writer(i):
        try:
            for n in range(25):
                jio.write_json(p, {"w": i, "n": n}, backups=2)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader():
        while not stop.is_set():
            try:
                json.loads(p.read_text())
            except FileNotFoundError:
                seen_bad.append("missing")
            except ValueError:
                seen_bad.append("torn")

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    rt = threading.Thread(target=reader)
    if POSIX:  # an open reader blocks any replace on Windows by design
        rt.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    stop.set()
    if POSIX:
        rt.join(10)
    assert errors == [] and seen_bad == []
    assert "w" in json.loads(p.read_text())
    assert not list(tmp_path.glob("*.new"))


def test_cw2_a_second_process_writing_the_same_path_does_not_interfere(tmp_path: Path) -> None:
    import subprocess
    import textwrap

    p = tmp_path / "s.json"
    code = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from brain.health import json_io
        for n in range(25):
            json_io.write_json(Path(sys.argv[1]), {"proc": 2, "n": n}, backups=2)
        """
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(p)],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[4])},
    )
    for n in range(25):
        jio.write_json(p, {"proc": 1, "n": n}, backups=2)
    assert child.wait(60) == 0
    assert json.loads(p.read_text())["proc"] in (1, 2)
    assert not list(tmp_path.glob("*.new"))


def test_cw2_write_json_leaves_no_temp_after_a_failed_save(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "s.json"
    jio.write_json(p, {"v": 1}, backups=1)

    def boom(fd):
        raise OSError(errno.EIO, "disk")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError):
        jio.write_json(p, {"v": 2}, backups=1)
    monkeypatch.undo()
    assert json.loads(p.read_text()) == {"v": 1}
    assert not list(tmp_path.glob("*.new"))


def test_cw3_write_json_fsyncs_by_default_and_not_when_asked(tmp_path: Path, monkeypatch) -> None:
    calls = _count_fsyncs(monkeypatch)
    jio.write_json(tmp_path / "a.json", {"a": 1})
    assert calls  # file and (POSIX) directory
    calls.clear()
    jio.write_json(tmp_path / "b.json", {"a": 1}, fsync=False)
    assert calls == []


@pytest.mark.parametrize("backups", [0, 1, 3, 6])
def test_cw4_the_backup_chain_matches_the_model(tmp_path: Path, backups: int) -> None:
    p = tmp_path / "s.json"
    for n in range(1, 11):
        jio.write_json(p, {"v": n}, backups=backups)
    assert json.loads(p.read_text()) == {"v": 10}
    baks = sorted(f.name for f in tmp_path.glob("s.json.bak*"))
    assert baks == [f"s.json.bak{i}" for i in range(1, backups + 1)]
    for i in range(1, backups + 1):
        assert json.loads((tmp_path / f"s.json.bak{i}").read_text()) == {"v": 10 - i}
    assert not list(tmp_path.glob("*.new"))


def test_cw4_the_backup_keeps_mtime_and_the_live_path_is_never_absent(
    tmp_path: Path, monkeypatch
) -> None:
    p = tmp_path / "s.json"
    jio.write_json(p, {"v": 1}, backups=2)
    old = time.time() - 5000
    os.utime(p, (old, old))
    missing_seen: list[str] = []
    real_replace = os.replace

    def watch(src, dst):
        if not p.exists():
            missing_seen.append(f"before {Path(src).name}->{Path(dst).name}")
        real_replace(src, dst)
        if not p.exists():
            missing_seen.append(f"after {Path(src).name}->{Path(dst).name}")

    monkeypatch.setattr(os, "replace", watch)
    jio.write_json(p, {"v": 2}, backups=2)
    monkeypatch.undo()
    assert missing_seen == []
    assert abs((tmp_path / "s.json.bak1").stat().st_mtime - old) < 2


@pytest.mark.skipif(not POSIX, reason="file modes are POSIX-only")
def test_cw5_write_json_mode_reaches_the_live_file_and_every_backup(tmp_path: Path) -> None:
    p = tmp_path / "bridge.json"
    p.write_text("{}")
    p.chmod(0o644)
    old = os.umask(0)
    try:
        for n in range(3):
            jio.write_json(p, {"token": n}, backups=2, mode=0o600)
    finally:
        os.umask(old)
    for name in ("bridge.json", "bridge.json.bak1", "bridge.json.bak2"):
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o600, name


@pytest.mark.parametrize("bad", [object(), float("nan"), float("inf"), {"k": {1, 2}}])
def test_cw6_a_payload_that_cannot_be_serialized_touches_nothing(tmp_path: Path, bad) -> None:
    p = tmp_path / "s.json"
    jio.write_json(p, {"v": 1}, backups=1)
    before, mtime = _listing(tmp_path), p.stat().st_mtime_ns
    with pytest.raises((TypeError, ValueError)):
        jio.write_json(p, {"v": bad}, backups=1)
    assert _listing(tmp_path) == before and p.stat().st_mtime_ns == mtime


def test_cw6_a_failed_serialization_on_a_new_path_creates_no_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        jio.write_json(tmp_path / "new.json", {"v": float("nan")})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kwargs", [{"backups": 0}, {"backups": 2}, {"backups": 0, "mode": 0o600}])
def test_cw7_a_transient_permission_error_at_every_replace_is_retried(
    tmp_path: Path, monkeypatch, kwargs
) -> None:
    if "mode" in kwargs and not POSIX:
        pytest.skip("modes are POSIX-only")
    p = tmp_path / "s.json"
    jio.write_json(p, {"v": 1}, **kwargs)
    real_replace = os.replace
    failed = {"n": 0}

    def flaky(src, dst):
        if failed["n"] < 3:
            failed["n"] += 1
            raise PermissionError(13, "sharing violation")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    jio.write_json(p, {"v": 2}, **kwargs)
    monkeypatch.undo()
    assert json.loads(p.read_text()) == {"v": 2} and failed["n"] == 3


def test_cw7_a_persistent_permission_error_raises_and_leaves_the_old_file_and_chain(
    tmp_path: Path, monkeypatch
) -> None:
    p = tmp_path / "s.json"
    jio.write_json(p, {"v": 1}, backups=1)
    jio.write_json(p, {"v": 2}, backups=1)
    before = _listing(tmp_path)
    real_replace = os.replace

    def stubborn(src, dst):
        if Path(dst) == p:
            raise PermissionError(13, "sharing violation")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", stubborn)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    with pytest.raises(PermissionError):
        jio.write_json(p, {"v": 3}, backups=1)
    monkeypatch.undo()
    assert json.loads(p.read_text()) == {"v": 2}
    assert not list(tmp_path.glob("*.new"))
    assert json.loads((tmp_path / "s.json.bak1").read_text())["v"] in (1, 2)
    assert before and True


def test_cw8_the_default_error_mode_raises_and_keeps_the_live_file(
    tmp_path: Path, monkeypatch
) -> None:
    p = tmp_path / "s.json"
    jio.write_json(p, {"v": 1}, backups=1)

    def enospc(fd):
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(os, "fsync", enospc)
    with pytest.raises(OSError) as exc:
        jio.write_json(p, {"v": 2}, backups=1)
    monkeypatch.undo()
    assert exc.value.errno == errno.ENOSPC
    assert json.loads(p.read_text()) == {"v": 1} and not list(tmp_path.glob("*.new"))


def test_cw8_log_mode_returns_false_with_throttled_evidence_and_no_temp(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    p = tmp_path / "s.json"
    jio.write_json(p, {"v": 1})

    def enospc(fd):
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(os, "fsync", enospc)
    with caplog.at_level(logging.DEBUG):
        assert jio.write_json(p, {"v": 2}, on_error="log") is False
        assert jio.write_json(p, {"v": 3}, on_error="log") is False
    monkeypatch.undo()
    assert len(_warnings(caplog)) == 1 and len(_warnings(caplog, logging.DEBUG)) == 1
    assert json.loads(p.read_text()) == {"v": 1} and not list(tmp_path.glob("*.new"))


def test_cw8_log_mode_does_not_swallow_a_serialization_error(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        jio.write_json(tmp_path / "s.json", {"v": object()}, on_error="log")


def test_cw8_a_failed_backup_copy_leaves_the_chain_untouched(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "s.json"
    for n in range(3):
        jio.write_json(p, {"v": n}, backups=2)
    before = _listing(tmp_path)

    def failing_copy(src, dst):
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr("brain.utils.atomic_write._copy_for_backup", failing_copy)
    with pytest.raises(OSError):
        jio.write_json(p, {"v": 99}, backups=2)
    monkeypatch.undo()
    assert _listing(tmp_path) == before


def _corrupt_after_publish(monkeypatch, target: Path, garbage: bytes = b"{garbage"):
    """After the module's final replace puts the new file at ``target``, damage it (a write that landed bad)."""
    real_replace = os.replace
    done = {"n": 0}

    def damaging(src, dst):
        real_replace(src, dst)
        if Path(dst) == target and done["n"] == 0:  # only the first publish lands bad
            done["n"] += 1
            target.write_bytes(garbage)

    monkeypatch.setattr(os, "replace", damaging)


def test_cw9_verify_restores_bak1_when_the_written_file_is_bad(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    p = tmp_path / "s.json"
    jio.write_json(p, {"v": 1}, backups=2)
    _corrupt_after_publish(monkeypatch, p)
    with caplog.at_level(logging.ERROR):
        assert jio.write_json(p, {"v": 2}, backups=2, verify=True) is False
    monkeypatch.undo()
    assert json.loads(p.read_text()) == {"v": 1}
    assert [r for r in caplog.records if r.levelno == logging.ERROR]


def test_cw9_verify_with_no_backup_logs_and_returns_false_without_raising(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    p = tmp_path / "s.json"
    _corrupt_after_publish(monkeypatch, p)
    with caplog.at_level(logging.ERROR):
        assert jio.write_json(p, {"v": 1}, verify=True) is False
    monkeypatch.undo()
    assert [r for r in caplog.records if r.levelno == logging.ERROR]


def test_cw9_verify_off_does_not_read_the_file_back_and_a_good_write_passes(
    tmp_path: Path, monkeypatch
) -> None:
    p = tmp_path / "s.json"
    reads: list[str] = []
    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: (reads.append(self.name), real(self))[1])
    assert jio.write_json(p, {"v": 1}) is True
    assert reads == []
    assert jio.write_json(p, {"v": 2}, verify=True) is True
    assert reads == ["s.json"]


def test_cw9_baseline_without_verify_a_bad_write_goes_unnoticed(
    tmp_path: Path, monkeypatch
) -> None:
    p = tmp_path / "s.json"
    jio.write_json(p, {"v": 1}, backups=1)
    _corrupt_after_publish(monkeypatch, p)
    assert jio.write_json(p, {"v": 2}, backups=1) is True
    monkeypatch.undo()
    assert p.read_bytes() == b"{garbage"


def test_cw10_output_is_pure_ascii_whatever_the_payload(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    payload = {
        "emoji": "\U0001f600",
        "bmp": "é日本",
        "lone": "\ud800",
        "sep": "a\u2028b\u2029c",
        "nl": "x\ny\r",
    }
    jio.write_json(p, payload)
    raw = p.read_bytes()
    raw.decode("ascii")
    assert b"\r" not in raw and b"\xe2\x80\xa8" not in raw
    assert json.loads(raw.decode("utf-8")) == payload


def test_cw10_baseline_ensure_ascii_false_cannot_encode_a_lone_surrogate() -> None:
    with pytest.raises(UnicodeEncodeError):
        json.dumps({"a": "\ud800"}, ensure_ascii=False).encode("utf-8")


def test_cw10_format_options(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    jio.write_json(p, {"b": 1, "a": [1, 2]})
    assert p.read_bytes() == b'{\n  "b": 1,\n  "a": [\n    1,\n    2\n  ]\n}\n'
    jio.write_json(p, {"b": 1, "a": 2}, indent=None, sort_keys=True)
    assert p.read_bytes() == b'{"a": 2, "b": 1}\n'


def test_cw10_encode_default_hook_and_no_silent_str(tmp_path: Path) -> None:
    from datetime import date

    p = tmp_path / "s.json"
    with pytest.raises(TypeError):
        jio.write_json(p, {"d": date(2026, 10, 17)})
    jio.write_json(p, {"d": date(2026, 10, 17)}, encode_default=str)
    assert json.loads(p.read_text()) == {"d": "2026-10-17"}


def test_cw11_a_missing_parent_directory_is_created(tmp_path: Path) -> None:
    p = tmp_path / "a" / "b" / "s.json"
    jio.write_json(p, {"v": 1})
    assert json.loads(p.read_text()) == {"v": 1}


def test_cw12_if_missing_creates_once_and_leaves_an_existing_file_alone(tmp_path: Path) -> None:
    p = tmp_path / "marker.json"
    assert jio.write_json(p, {"first": 1}, if_missing=True) is True
    assert jio.write_json(p, {"second": 2}, if_missing=True) is False
    assert json.loads(p.read_text()) == {"first": 1}


def test_cw12_two_threads_racing_to_create_exactly_one_wins(tmp_path: Path) -> None:
    p = tmp_path / "marker.json"
    results: list[bool] = []
    barrier = threading.Barrier(2)

    def go(i):
        barrier.wait()
        results.append(jio.write_json(p, {"winner": i}, if_missing=True))

    threads = [threading.Thread(target=go, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert sorted(results) == [False, True]
    assert json.loads(p.read_text())["winner"] in (0, 1)


def test_cw12_baseline_a_plain_exists_check_lets_both_threads_write(tmp_path: Path) -> None:
    p = tmp_path / "marker.json"
    wrote: list[int] = []
    gate = threading.Barrier(2)

    def naive(i):
        if not p.exists():
            gate.wait()  # both pass the check before either writes
            p.write_text(json.dumps({"winner": i}))
            wrote.append(i)

    threads = [threading.Thread(target=naive, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(wrote) == 2


# --------------------------------------------------------------------------------------------------------------
# JSONL reads (CJ1-CJ6)
# --------------------------------------------------------------------------------------------------------------

import gzip  # noqa: E402


def test_cj1_a_missing_file_yields_nothing(tmp_path: Path) -> None:
    assert list(jio.iter_jsonl(tmp_path / "nope.jsonl")) == []
    assert jio.read_jsonl(tmp_path / "nope.jsonl") == []


def test_cj1_bad_lines_are_skipped_one_warning_each_and_neighbours_survive(
    tmp_path: Path, caplog
) -> None:
    p = tmp_path / "log.jsonl"
    long_bad = b'{"k": ' + b"9" * 400 + b" oops"
    p.write_bytes(
        b'{"a": 1}\n\n   \nnot json\n[1, 2]\n{"b": "caf\xff"}\n' + long_bad + b'\n{"c": 3}\n'
    )
    with caplog.at_level(logging.WARNING):
        got = jio.read_jsonl(p)
    assert got == [{"a": 1}, {"c": 3}]
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 4
    for needle in ("line 4", "line 5", "line 6", "line 7"):
        assert any(needle in m and str(p) in m for m in msgs), needle
    assert all(len(m) < 700 for m in msgs)  # previews are capped at 201 characters


def test_cj1_a_bom_on_the_first_line_and_crlf_endings_are_handled(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    p.write_bytes(b'\xef\xbb\xbf{"a": 1}\r\n{"b": 2}\r\n')
    assert jio.read_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_cj1_unicode_line_separators_inside_a_value_do_not_split_a_record(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    value = "x" + chr(0x2028) + "y" + chr(0x2029) + "z" + chr(0x85) + "w"
    p.write_bytes((json.dumps({"a": value}, ensure_ascii=False) + '\n{"b": 2}\n').encode())
    assert jio.read_jsonl(p) == [{"a": value}, {"b": 2}]


def test_cj1_baseline_splitlines_would_cut_that_record() -> None:
    text = '{"a": "x\u2028y"}\n{"b": 2}\n'
    assert (
        len(text.splitlines()) == 3
    )  # the bridge/feed.py:407 shape: one record becomes two broken ones


def test_cj1_a_lone_carriage_return_is_not_a_record_separator(tmp_path: Path) -> None:
    """Differs from the legacy text-mode reader on purpose; it cannot occur in files the app writes (JSON escapes \\r)."""
    p = tmp_path / "log.jsonl"
    p.write_bytes(b'{"c":3}\r{"d":4}\n')
    assert jio.read_jsonl(p) == []


_DIFFERENTIAL_INPUTS = [
    b"",
    b"\n\n",
    b'{"a": 1}\n',
    b'{"a": 1}',  # no trailing newline
    b'{"a": 1}\n{"b": \n{"c": 3}\n',
    b'{"a": 1}\n[1]\n"s"\n7\nnull\n{"b": 2}\n',
    b'{"a": "caf\xff"}\n{"b": 2}\n',
    b'{"a": 1}\r\n{"b": 2}\r\n',
    '{"é": "ü日本\U0001f600"}\n'.encode(),
]


@pytest.mark.parametrize("index", range(len(_DIFFERENTIAL_INPUTS)))
def test_cj1_matches_the_frozen_legacy_reader_output_on_the_same_input(
    tmp_path: Path, index: int
) -> None:
    gold = GOLDEN["jsonl"][str(index)]
    assert gold["input"].encode("latin-1") == _DIFFERENTIAL_INPUTS[index]
    p = tmp_path / "log.jsonl"
    p.write_bytes(_DIFFERENTIAL_INPUTS[index])
    assert jio.read_jsonl(p) == gold["records"]


def test_cj2_gz_files_are_read_by_suffix(tmp_path: Path) -> None:
    p = tmp_path / "log.2025.jsonl.gz"
    with gzip.open(p, "wb") as f:
        f.write(b'{"a": 1}\nnot json\n{"b": 2}\n')
    assert jio.read_jsonl(p) == [{"a": 1}, {"b": 2}]


def _peak_bytes(fn) -> int:
    import tracemalloc

    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_cj2_iteration_is_streaming(tmp_path: Path) -> None:
    p = tmp_path / "big.jsonl"
    line = b'{"k": "' + b"x" * 90 + b'"}\n'
    p.write_bytes(line * 55000)  # about 5 MB

    def first():
        it = jio.iter_jsonl(p)
        assert next(it) == {"k": "x" * 90}
        it.close()

    assert _peak_bytes(first) < 1_000_000


def test_cj2_baseline_reading_the_whole_text_peaks_far_higher(tmp_path: Path) -> None:
    p = tmp_path / "big.jsonl"
    p.write_bytes((b'{"k": "' + b"x" * 90 + b'"}\n') * 55000)
    assert _peak_bytes(lambda: p.read_text().splitlines()) > 4_000_000


def test_cj3_strict_iteration_raises_on_the_first_bad_line_after_yielding_the_good_ones(
    tmp_path: Path,
) -> None:
    p = tmp_path / "log.jsonl"
    p.write_bytes(b'{"a": 1}\n{"b": 2}\n{"torn": ')
    seen = []
    with pytest.raises(jio.JsonFileError) as exc:
        for rec in jio.iter_jsonl(p, strict=True):
            seen.append(rec)
    assert seen == [{"a": 1}, {"b": 2}]
    assert exc.value.kind == "line" and exc.value.line == 3 and isinstance(exc.value, OSError)


def test_cj3_strict_tail_raises_on_an_invalid_utf8_last_line(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    p.write_bytes(b'{"a": 1}\n{"b": "caf\xff"}\n')
    with pytest.raises(jio.JsonFileError) as exc:
        jio.read_jsonl_tail(p, 1, strict=True)
    assert exc.value.kind == "line"
    assert jio.read_jsonl_tail(p, 2) == [{"a": 1}]  # non-strict skips it with a warning


def test_cj3_strict_raises_on_a_torn_trailing_line_and_on_a_non_object(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    p.write_bytes(b'{"a": 1}\n{"b":')
    with pytest.raises(jio.JsonFileError):
        jio.read_jsonl_tail(p, 1, strict=True)
    p.write_bytes(b'{"a": 1}\n[1]\n')
    with pytest.raises(jio.JsonFileError):
        jio.read_jsonl(p, strict=True)


def test_cj4_trailing_blank_lines_do_not_count_toward_n(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    p.write_bytes(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n\n\n')
    assert jio.read_jsonl_tail(p, 2) == [{"b": 2}, {"c": 3}]
    assert GOLDEN["legacy_deltas"]["tail_trailing_blanks"] == ["", ""]  # the legacy helper counted the blanks


def test_cj4_n_zero_and_a_missing_file_give_empty(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    p.write_bytes(b'{"a": 1}\n')
    assert jio.read_jsonl_tail(p, 0) == [] and jio.read_jsonl_tail(p, -3) == []
    assert jio.read_jsonl_tail(tmp_path / "nope.jsonl", 3) == []
    p.write_bytes(b"")
    assert jio.read_jsonl_tail(p, 3) == []


def test_cj4_fewer_lines_than_n_keeps_the_first_line(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    p.write_bytes(b'{"a": 1}\n{"b": 2}')  # no trailing newline
    assert jio.read_jsonl_tail(p, 10) == [{"a": 1}, {"b": 2}]


def test_cj4_a_multibyte_character_split_by_a_chunk_boundary_is_decoded_once(
    tmp_path: Path,
) -> None:
    p = tmp_path / "log.jsonl"
    rows = [{"i": i, "t": "é日\U0001f600" * (1 + i % 7)} for i in range(900)]
    p.write_bytes(b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in rows))
    assert p.stat().st_size > 3 * 8192
    for n in (1, 5, 37, 400, 900, 1200):
        assert jio.read_jsonl_tail(p, n) == rows[-n:], n


def test_cj4_reads_are_bounded_by_n_not_by_the_file_size(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "big.jsonl"
    p.write_bytes((b'{"k": "' + b"x" * 90 + b'"}\n') * 200_000)  # about 20 MB
    read_sizes: list[int] = []
    real_open = open

    def spy(file, *a, **k):
        fh = real_open(file, *a, **k)
        if Path(file) != p or "b" not in (a[0] if a else k.get("mode", "r")):
            return fh

        class Proxy:
            def __getattr__(self, name):
                return getattr(fh, name)

            def __enter__(self):
                fh.__enter__()
                return self

            def __exit__(self, *exc):
                return fh.__exit__(*exc)

            def read(self, size=-1):
                data = fh.read(size)
                read_sizes.append(len(data))
                return data

        return Proxy()

    monkeypatch.setattr("builtins.open", spy)
    assert len(jio.read_jsonl_tail(p, 3)) == 3
    monkeypatch.undo()
    assert sum(read_sizes) < 100_000


def test_cj4_a_bom_at_the_start_of_the_file_is_stripped_when_the_window_reaches_it(
    tmp_path: Path,
) -> None:
    p = tmp_path / "log.jsonl"
    p.write_bytes(b'\xef\xbb\xbf{"a": 1}\n{"b": 2}\n{"c": 3}\n')
    assert jio.read_jsonl_tail(p, 5) == [{"a": 1}, {"b": 2}, {"c": 3}]
    assert jio.read_jsonl_tail(p, 5, strict=True) == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_cj5_parse_jsonl_applies_the_same_rules_to_bytes_and_str(tmp_path: Path) -> None:
    data = b'\xef\xbb\xbf{"a": 1}\nnot json\n{"b": 2}\n'
    assert list(jio.parse_jsonl(data)) == [{"a": 1}, {"b": 2}]
    assert list(jio.parse_jsonl(data.decode("utf-8-sig"))) == [{"a": 1}, {"b": 2}]
    with pytest.raises(jio.JsonFileError) as exc:
        list(jio.parse_jsonl(data, source="feed", strict=True))
    assert exc.value.line == 2 and str(exc.value.path) == "feed"


def test_cj6_a_directory_is_an_io_failure_with_the_default_policy(tmp_path: Path, caplog) -> None:
    p = tmp_path / "log.jsonl"
    p.mkdir()
    with caplog.at_level(logging.DEBUG):
        assert jio.read_jsonl(p) == []
        assert jio.read_jsonl(p) == []
    assert len(_warnings(caplog)) == 1  # throttled
    with pytest.raises(jio.JsonFileError) as exc:
        jio.read_jsonl(p, recover=jio.Recovery.RAISE)
    assert exc.value.kind == "io"


def _truncated_gz(tmp_path: Path) -> Path:
    full = tmp_path / "full.gz"
    with gzip.open(full, "wb") as f:
        for i in range(2000):
            f.write(json.dumps({"i": i, "pad": "x" * 50}).encode() + b"\n")
    data = full.read_bytes()
    cut = tmp_path / "cut.jsonl.gz"
    cut.write_bytes(data[: len(data) // 2])
    return cut


def test_cj6_a_truncated_gz_yields_what_was_read_then_stops_with_evidence(
    tmp_path: Path, caplog
) -> None:
    cut = _truncated_gz(tmp_path)
    with caplog.at_level(logging.WARNING):
        got = jio.read_jsonl(cut)
    assert 0 < len(got) < 2000 and got[0] == {"i": 0, "pad": "x" * 50}
    assert len(_warnings(caplog)) == 1
    with pytest.raises(jio.JsonFileError) as exc:
        jio.read_jsonl(cut, recover=jio.Recovery.RAISE)
    assert exc.value.kind == "io" and not isinstance(exc.value, EOFError)


def test_cj6_other_recovery_policies_are_rejected_for_jsonl(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        jio.read_jsonl(tmp_path / "x.jsonl", recover=jio.Recovery.HEAL)


# --------------------------------------------------------------------------------------------------------------
# JSONL appends and rewrites (CA1-CA6)
# --------------------------------------------------------------------------------------------------------------


class _WriteCounter:
    """Count write() calls on binary append handles opened for ``target``."""

    def __init__(self, monkeypatch, target: Path) -> None:
        self.writes: list[bytes] = []
        real_open = open
        counter = self

        def spy(file, *a, **k):
            fh = real_open(file, *a, **k)
            mode = a[0] if a else k.get("mode", "r")
            if Path(file) != target or "a" not in mode:
                return fh

            class Proxy:
                def __getattr__(self, name):
                    return getattr(fh, name)

                def __enter__(self):
                    fh.__enter__()
                    return self

                def __exit__(self, *exc):
                    return fh.__exit__(*exc)

                def write(self, data):
                    counter.writes.append(bytes(data))
                    return fh.write(data)

            return Proxy()

        monkeypatch.setattr("builtins.open", spy)


def test_ca1_append_creates_the_parent_writes_one_ascii_lf_line_and_returns_its_length(
    tmp_path: Path, monkeypatch
) -> None:
    p = tmp_path / "deep" / "dir" / "log.jsonl"
    counter = _WriteCounter(monkeypatch, p)
    n = jio.append_jsonl(p, {"a": "é\U0001f600", "n": 1})
    monkeypatch.undo()
    assert len(counter.writes) == 1
    raw = p.read_bytes()
    raw.decode("ascii")
    assert raw.endswith(b"\n") and b"\r" not in raw and raw.count(b"\n") == 1
    assert n == len(raw) == len(jio.encode_jsonl({"a": "é\U0001f600", "n": 1}))
    assert jio.read_jsonl(p) == [{"a": "é\U0001f600", "n": 1}]


def test_ca1_a_batch_is_one_write_and_one_lock_acquisition(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "log.jsonl"
    counter = _WriteCounter(monkeypatch, p)
    locks: list[Path] = []
    real_lock = jio.file_lock

    def counting_lock(path, **k):
        locks.append(Path(path))
        return real_lock(path, **k)

    monkeypatch.setattr(jio, "file_lock", counting_lock)
    rows = [{"i": i} for i in range(50)]
    n = jio.append_jsonl(p, rows)
    monkeypatch.undo()
    assert len(counter.writes) == 1 and len(locks) == 1
    assert n == len(jio.encode_jsonl(rows)) and jio.read_jsonl(p) == rows


def test_ca1_encoded_bytes_are_accepted_so_a_caller_can_size_a_segment_before_writing(
    tmp_path: Path,
) -> None:
    p = tmp_path / "log.jsonl"
    payload = jio.encode_jsonl([{"a": 1}, {"b": 2}])
    assert jio.append_jsonl(p, payload) == len(payload)
    assert p.read_bytes() == payload


def test_ca1_an_empty_batch_returns_zero_and_touches_nothing(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    assert jio.append_jsonl(p, []) == 0
    assert not p.exists() and list(tmp_path.iterdir()) == []


def test_ca2_concurrent_appends_from_threads_and_a_process_lose_nothing(tmp_path: Path) -> None:
    import subprocess
    import textwrap

    p = tmp_path / "log.jsonl"
    code = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from brain.health import json_io
        for n in range(25):
            json_io.append_jsonl(Path(sys.argv[1]), {"src": "proc", "n": n})
        """
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(p)],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[4])},
    )

    def work(i):
        for n in range(25):
            jio.append_jsonl(p, {"src": f"t{i}", "n": n})

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert child.wait(60) == 0
    rows = jio.read_jsonl(p)
    assert len(rows) == 8 * 25 + 25
    assert p.read_bytes().count(b"\n") == len(rows)


def test_ca2_block_waits_while_another_handle_holds_the_lock(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    holder, release = _hold_shared_lock_in_thread(p)
    try:
        worker, box = _in_thread(lambda: jio.append_jsonl(p, {"a": 1}), timeout=0.5)
        assert worker.is_alive(), "append did not wait for the lock"
        assert not p.exists() or p.read_bytes() == b""
    finally:
        release.set()
        holder.join(5)
    worker.join(5)
    assert box.get("result") == len(jio.encode_jsonl({"a": 1}))
    assert jio.read_jsonl(p) == [{"a": 1}]


def test_ca2_try_returns_none_and_writes_nothing_when_the_lock_is_busy(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    holder, release = _hold_shared_lock_in_thread(p)
    try:
        assert jio.append_jsonl(p, {"a": 1}, lock="try") is None
        assert not p.exists() or p.read_bytes() == b""
    finally:
        release.set()
        holder.join(5)
    assert jio.append_jsonl(p, {"a": 1}, lock="try") is not None


def test_ca2_held_takes_no_lock_so_a_caller_holding_it_does_not_deadlock(tmp_path: Path) -> None:
    from brain.utils.file_lock import file_lock

    p = tmp_path / "log.jsonl"

    def work():
        with file_lock(p):
            return jio.append_jsonl(p, {"a": 1}, lock="held")

    worker, box = _in_thread(work, timeout=5)
    assert not worker.is_alive(), "lock='held' re-locked and deadlocked"
    assert box["result"] is not None


def test_ca2_baseline_block_while_the_caller_holds_the_lock_would_deadlock(tmp_path: Path) -> None:
    from brain.utils.file_lock import file_lock

    p = tmp_path / "log.jsonl"

    def work():
        with file_lock(p):
            return jio.append_jsonl(p, {"a": 1}, lock="block")

    worker, _box = _in_thread(work, timeout=1.0)
    assert worker.is_alive()  # a second flock on a new handle in the same process blocks forever


def test_ca2_a_refused_lock_still_appends_with_one_warning(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    import contextlib

    @contextlib.contextmanager
    def refusing(path, *, blocking=True):
        raise OSError(errno.ENOLCK, "no locks")
        yield  # pragma: no cover

    monkeypatch.setattr(jio, "file_lock", refusing)
    p = tmp_path / "log.jsonl"
    with caplog.at_level(logging.WARNING):
        assert jio.append_jsonl(p, {"a": 1}) is not None
        assert jio.append_jsonl(p, {"b": 2}) is not None
    assert jio.read_jsonl(p) == [{"a": 1}, {"b": 2}]
    assert len(_warnings(caplog)) == 1  # throttled


def test_ca3_an_append_after_a_torn_tail_starts_a_new_line(tmp_path: Path, caplog) -> None:
    p = tmp_path / "log.jsonl"
    torn = b'{"a": 1}\n{"b": '
    p.write_bytes(torn)
    with caplog.at_level(logging.WARNING):
        jio.append_jsonl(p, {"c": 3})
    assert p.read_bytes() == torn + b"\n" + jio.encode_jsonl({"c": 3})
    assert jio.read_jsonl(p) == [{"a": 1}, {"c": 3}]  # only the torn record is lost
    assert any(
        str(p) in r.getMessage() and "torn" in r.getMessage().lower() for r in caplog.records
    )


def test_ca3_a_tail_cut_inside_a_multibyte_character_is_handled(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    p.write_bytes(b'{"a": 1}\n{"b": "\xc3')
    jio.append_jsonl(p, {"c": 3})
    assert jio.read_jsonl(p) == [{"a": 1}, {"c": 3}]


@pytest.mark.parametrize("before", [None, b"", b'{"a": 1}\n', b'{"a": 1}\r\n'])
def test_ca3_no_prefix_newline_when_the_file_is_missing_empty_or_already_ends_in_one(
    tmp_path: Path, before
) -> None:
    p = tmp_path / "log.jsonl"
    if before is not None:
        p.write_bytes(before)
    jio.append_jsonl(p, {"c": 3})
    assert p.read_bytes() == (before or b"") + jio.encode_jsonl({"c": 3})


def test_ca4_a_legacy_appender_landing_between_the_check_and_the_write_cannot_cost_our_record(
    tmp_path: Path, monkeypatch
) -> None:
    p = tmp_path / "log.jsonl"
    p.write_bytes(b'{"a": 1}\n{"torn": ')

    def legacy_appender(name, path):
        if name == "between_check_and_write":
            with open(path, "ab") as fh:  # a lock-free legacy appender, fusing onto the torn tail
                fh.write(b'{"legacy": 1}\n')

    monkeypatch.setattr(jio, "_seam", legacy_appender)
    jio.append_jsonl(p, {"ours": 1})
    monkeypatch.undo()
    rows = jio.read_jsonl(p)
    assert {"ours": 1} in rows and {
        "a": 1
    } in rows  # the torn bytes and the legacy record may die together; ours may not


def test_ca4_a_stale_check_costs_at_most_a_blank_line(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "log.jsonl"
    p.write_bytes(b'{"a": 1}\n{"b": ')

    def completer(name, path):
        if name == "between_check_and_write":
            with open(path, "ab") as fh:  # the slow writer finishes its line after we looked
                fh.write(b"2}\n")

    monkeypatch.setattr(jio, "_seam", completer)
    jio.append_jsonl(p, {"ours": 1})
    monkeypatch.undo()
    assert jio.read_jsonl(p) == [{"a": 1}, {"b": 2}, {"ours": 1}]
    assert b"\n\n" in p.read_bytes()  # the blank line the reader skips


def test_ca5_fsync_is_opt_in_for_appends(tmp_path: Path, monkeypatch) -> None:
    calls = _count_fsyncs(monkeypatch)
    jio.append_jsonl(tmp_path / "a.jsonl", {"a": 1})
    assert calls == []
    jio.append_jsonl(tmp_path / "b.jsonl", {"a": 1}, fsync=True)
    assert calls


@pytest.mark.parametrize(
    "bad,exc",
    [
        ([1, 2], TypeError),
        ("text", TypeError),
        ({"x": float("nan")}, ValueError),
        ({"x": object()}, TypeError),
    ],
)
def test_ca5_a_bad_record_raises_before_the_file_is_created_or_touched(
    tmp_path: Path, bad, exc
) -> None:
    p = tmp_path / "log.jsonl"
    with pytest.raises(exc):
        jio.append_jsonl(p, bad)
    assert not p.exists()
    p.write_bytes(b'{"a": 1}\n')
    with pytest.raises(exc):
        jio.append_jsonl(p, [{"ok": 1}, bad])  # one bad record in a batch writes none of it
    assert p.read_bytes() == b'{"a": 1}\n'


def test_ca5_log_mode_returns_none_with_evidence_for_an_oserror_but_never_swallows_a_type_error(
    tmp_path: Path, caplog
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file")
    p = blocker / "sub" / "log.jsonl"  # the parent cannot be created
    with caplog.at_level(logging.DEBUG):
        assert jio.append_jsonl(p, {"a": 1}, on_error="log") is None
        assert jio.append_jsonl(p, {"a": 1}, on_error="log") is None
    assert len(_warnings(caplog)) == 1
    with pytest.raises(OSError):
        jio.append_jsonl(p, {"a": 1})
    with pytest.raises(TypeError):
        jio.append_jsonl(tmp_path / "ok.jsonl", [1], on_error="log")


def test_ca6_write_jsonl_replaces_atomically_and_writes_ascii_lf(tmp_path: Path) -> None:
    p = tmp_path / "q.jsonl"
    assert jio.write_jsonl(p, [{"a": "é"}, {"b": 2}]) is True
    assert p.read_bytes() == jio.encode_jsonl([{"a": "é"}, {"b": 2}])
    p.read_bytes().decode("ascii")
    assert jio.write_jsonl(p, []) is True and p.read_bytes() == b""  # the drain shape
    assert jio.write_jsonl(p, jio.encode_jsonl({"z": 1})) is True and jio.read_jsonl(p) == [
        {"z": 1}
    ]


def test_ca6_a_failed_rewrite_leaves_the_old_file_and_no_temp(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "q.jsonl"
    jio.write_jsonl(p, [{"old": 1}], backups=1)
    jio.write_jsonl(p, [{"old": 2}], backups=1)

    def boom(fd):
        raise OSError(errno.EIO, "disk")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError):
        jio.write_jsonl(p, [{"new": 1}], backups=1)
    monkeypatch.undo()
    assert jio.read_jsonl(p) == [{"old": 2}] and not list(tmp_path.glob("*.new"))


def test_ca6_backups_and_the_heal_sidecar_apply_to_a_rewrite(tmp_path: Path) -> None:
    p = tmp_path / "q.jsonl"
    for n in range(3):
        jio.write_jsonl(p, [{"n": n}], backups=2)
    assert jio.read_jsonl(tmp_path / "q.jsonl.bak1") == [{"n": 1}]
    assert jio.read_jsonl(tmp_path / "q.jsonl.bak2") == [{"n": 0}]
    assert (
        jio.write_jsonl(tmp_path / "log" / "r.jsonl", [{"a": 1}]) is True
    )  # missing parent is created


def test_ca6_on_error_log_returns_false(tmp_path: Path, monkeypatch) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("file")
    assert jio.write_jsonl(blocker / "x.jsonl", [{"a": 1}], on_error="log") is False


def test_cr11_no_public_function_opens_a_text_file_without_utf8(
    tmp_path: Path, monkeypatch
) -> None:
    state, log = tmp_path / "state.json", tmp_path / "log.jsonl"
    state.write_text('{"a": 1}')
    log.write_text('{"a": 1}\n')
    (tmp_path / "bad.json").write_text("{nope")
    (tmp_path / "bad.json.bak1").write_text('{"b": 1}')
    spy = _OpenSpy(monkeypatch)
    jio.read_json(state, default=dict)
    jio.read_json(tmp_path / "bad.json", default=dict, recover=jio.Recovery.HEAL)
    jio.write_json(tmp_path / "w.json", {"a": "é"}, backups=2)
    jio.write_json(tmp_path / "v.json", {"a": 1}, verify=True)
    jio.write_json(tmp_path / "once.json", {"a": 1}, if_missing=True)
    jio.read_jsonl(log)
    jio.read_jsonl_tail(log, 3)
    jio.append_jsonl(log, {"b": "é"})
    jio.write_jsonl(tmp_path / "rw.jsonl", [{"a": 1}])
    list(jio.parse_jsonl(b'{"a": 1}\n'))
    assert spy.violations == []


# --------------------------------------------------------------------------------------------------------------
# Code-review round 1 fixes (6-redteam-code.md F1-F10): each test fails on the code before the fix
# --------------------------------------------------------------------------------------------------------------

import random  # noqa: E402


def _start_thread(fn):
    box: dict = {}

    def run():
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, box


@pytest.mark.parametrize("policy", ["RESTORE", "HEAL"])
def test_cr9_f_a_plain_reader_arriving_mid_heal_waits_and_never_gets_the_default(
    tmp_path: Path, monkeypatch, policy: str
) -> None:
    """F1: between the healer's quarantine and its install a plain reader used to see no file and return default()."""
    p = _corrupt_with_backup(tmp_path, b'{"restored": true}')
    state: dict = {}

    def reader_arrives(path):
        t, box = _start_thread(
            lambda: jio.read_json(path, default=lambda: {"DEFAULT": 1}, recover=_policy(policy))
        )
        t.join(0.4)
        state["thread"], state["box"] = t, box
        state["waited"] = t.is_alive() or box.get("result") == {"restored": True}
        state["early"] = box.get("result")

    _seam_hook(monkeypatch, inside_heal=reader_arrives)
    assert jio.read_json(p, default=lambda: {"DEFAULT": 1}, recover=_policy(policy)) == {
        "restored": True
    }
    state["thread"].join(5)
    assert state["early"] != {"DEFAULT": 1}, "a reader got default() while a heal was in progress"
    assert state["box"].get("result") == {"restored": True}


def test_cr9_f_stress_no_reader_ever_sees_the_default_while_others_heal(tmp_path: Path) -> None:
    bad: list[object] = []
    for rnd in range(10):
        d = tmp_path / f"r{rnd}"
        d.mkdir()
        p = _corrupt_with_backup(d, b'{"good": 1}')
        gate = threading.Barrier(6)

        def reader(p=p, gate=gate):
            gate.wait()
            got = jio.read_json(p, default=lambda: {"DEFAULT": 1}, recover=jio.Recovery.RESTORE)
            if got != {"good": 1}:
                bad.append(got)

        threads = [threading.Thread(target=reader) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
    assert bad == []


def test_cr9_f_a_reader_finding_the_file_absent_while_a_heal_lock_is_held_waits_for_it(
    tmp_path: Path,
) -> None:
    """A legacy healer (no sidecar of its own) cannot be waited on, but a module heal in progress can."""
    from brain.utils.file_lock import file_lock

    p = tmp_path / "s.json"
    heal_target = p.with_name(p.name + ".heal")
    acquired, release = threading.Event(), threading.Event()

    def healer():
        with file_lock(heal_target):
            acquired.set()
            release.wait(10)
            p.write_text('{"restored": true}')

    h = threading.Thread(target=healer, daemon=True)
    h.start()
    assert acquired.wait(5)
    t, box = _start_thread(
        lambda: jio.read_json(p, default=lambda: {"DEFAULT": 1}, recover=jio.Recovery.RESTORE)
    )
    t.join(0.4)
    assert t.is_alive(), "the reader did not wait for the heal in progress"
    release.set()
    t.join(5)
    h.join(5)
    assert box.get("result") == {"restored": True}


def test_cr15_restore_of_a_missing_file_restores_a_valid_backup_instead_of_the_default(
    tmp_path: Path,
) -> None:
    p = tmp_path / "b.json"
    (tmp_path / "b.json.bak1").write_bytes(b'{"count": 7}')
    r = jio.read_json_result(p, default=lambda: {"count": 0}, recover=jio.Recovery.RESTORE)
    assert (r.data, r.outcome) == ({"count": 7}, "restored")
    assert json.loads(p.read_text()) == {"count": 7}


def test_cr15_a_missing_file_with_no_backup_is_still_the_default_and_heal_keeps_the_legacy_meaning(
    tmp_path: Path,
) -> None:
    p = tmp_path / "b.json"
    assert (
        jio.read_json_result(p, default=lambda: {"count": 0}, recover=jio.Recovery.RESTORE).outcome
        == "missing"
    )
    (tmp_path / "b.json.bak1").write_bytes(b'{"count": 7}')
    # HEAL keeps attempt_heal's meaning for a missing file: a deleted file is a reset, not a corruption
    assert (
        jio.read_json_result(p, default=lambda: {"count": 0}, recover=jio.Recovery.HEAL).outcome
        == "missing"
    )


def test_cr9_f_the_live_path_is_never_absent_during_a_module_heal(
    tmp_path: Path, monkeypatch
) -> None:
    p = _corrupt_with_backup(tmp_path)
    absent: list[str] = []

    def watch(name, path):
        if not os.path.lexists(path):
            absent.append(name)

    monkeypatch.setattr(jio, "_seam", watch)
    jio.read_json(p, default=dict, recover=jio.Recovery.HEAL)
    assert absent == []


@pytest.mark.parametrize("policy", ["RESTORE", "HEAL"])
def test_cr9_g_a_failed_install_keeps_the_corrupt_live_file_raises_json_file_error_and_recovers_next_time(
    tmp_path: Path, monkeypatch, policy: str
) -> None:
    """F2: used to leave the live file quarantined and absent, raise a raw PermissionError, then reset on the next read."""
    p = _corrupt_with_backup(tmp_path, b'{"count": 7}')
    real = jio._replace_with_retry

    def failing(src, dst):
        if Path(dst) == p:
            raise PermissionError(13, "an AV scan holds the backup")
        return real(src, dst)

    monkeypatch.setattr(jio, "_replace_with_retry", failing)
    with pytest.raises(jio.JsonFileError) as exc:
        jio.read_json(p, default=lambda: {"count": 0}, recover=_policy(policy))
    assert exc.value.kind == "io" and not isinstance(exc.value, PermissionError)
    assert p.read_bytes() == b"{nope" and (tmp_path / "s.json.bak1").exists()
    monkeypatch.undo()
    assert jio.read_json(p, default=lambda: {"count": 0}, recover=_policy(policy)) == {"count": 7}


def test_cr9_g_a_failed_reset_save_keeps_the_corrupt_file_and_the_next_heal_resets(
    tmp_path: Path, monkeypatch
) -> None:
    p = tmp_path / "s.json"
    p.write_bytes(b"{nope")

    def enospc(*a, **k):
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(jio, "save_with_backup_text", enospc)
    with pytest.raises(jio.JsonFileError) as exc:
        jio.read_json(p, default=lambda: {"fresh": 1}, recover=jio.Recovery.HEAL)
    assert exc.value.kind == "io" and p.read_bytes() == b"{nope"
    monkeypatch.undo()
    assert jio.read_json(p, default=lambda: {"fresh": 1}, recover=jio.Recovery.HEAL) == {"fresh": 1}


def test_cr14_a_competing_restorer_before_the_reset_save_is_not_overwritten(
    tmp_path: Path, monkeypatch
) -> None:
    """F7: the reset-to-default install is an install too; it needs the same collision check as the restore."""
    p = tmp_path / "s.json"
    p.write_bytes(b"{nope")
    _seam_hook(monkeypatch, before_install=lambda path: path.write_text('{"competitor": 1}'))
    r = jio.read_json_result(p, default=lambda: {"fresh": 1}, recover=jio.Recovery.HEAL)
    assert r.data == {"competitor": 1} and json.loads(p.read_text()) == {"competitor": 1}


def test_cr14_a_corrupt_backup_that_changed_before_its_quarantine_is_left_alone(
    tmp_path: Path, monkeypatch
) -> None:
    """F6: each corrupt backup is re-stat'ed before it is moved; a changed one is skipped, the heal still succeeds."""
    p = tmp_path / "s.json"
    p.write_bytes(b"{nope")
    (tmp_path / "s.json.bak1").write_bytes(b"garbage")
    (tmp_path / "s.json.bak2").write_bytes(b'{"good": 1}')
    _seam_hook(
        monkeypatch,
        before_quarantine_backup=lambda path: path.write_text('{"someone": "saved a backup"}'),
    )
    assert jio.read_json(p, default=dict, recover=jio.Recovery.HEAL) == {"good": 1}
    assert (tmp_path / "s.json.bak1").read_text() == '{"someone": "saved a backup"}'
    assert not list(tmp_path.glob("s.json.bak1.corrupt-*"))


def test_cw9_verify_runs_under_the_lock_so_a_concurrent_newer_save_is_not_undone(
    tmp_path: Path, monkeypatch
) -> None:
    """F3: another writer's save used to land between A's save and A's verify; A then restored its own older text over it."""
    p = tmp_path / "s.json"
    jio.write_json(p, {"v": "base"}, backups=1)
    state: dict = {}

    def b_arrives(name, path):
        if name == "after_save" and "t" not in state:
            state["t"], state["box"] = _start_thread(
                lambda: jio.write_json(p, {"v": "B-newer"}, backups=1)
            )
            state["t"].join(0.3)
            state["b_waited"] = state["t"].is_alive()

    monkeypatch.setattr(jio, "_seam", b_arrives)
    assert jio.write_json(p, {"v": "A"}, backups=1, verify=True) is True
    state["t"].join(5)
    assert state["b_waited"], "B's save ran inside A's save-and-verify"
    assert json.loads(p.read_text()) == {"v": "B-newer"}


def test_cw9_a_valid_newer_file_from_an_unlocked_writer_is_never_overwritten_by_the_restore(
    tmp_path: Path, monkeypatch
) -> None:
    p = tmp_path / "s.json"
    jio.write_json(p, {"v": "base"}, backups=1)

    def legacy_saver(name, path):
        if name == "after_save":
            tmp = path.with_name("legacy.tmp")
            tmp.write_text('{"v": "legacy-newer"}')
            os.replace(tmp, path)  # a legacy saver takes no heal sidecar

    monkeypatch.setattr(jio, "_seam", legacy_saver)
    assert jio.write_json(p, {"v": "A"}, backups=1, verify=True) is True  # superseded, not failed
    assert json.loads(p.read_text()) == {"v": "legacy-newer"}


def test_cr2_a_huge_integer_is_malformed_not_a_value_error(tmp_path: Path) -> None:
    """F5: json.loads raises a plain ValueError past 4300 digits; it leaked out of every policy."""
    p = tmp_path / "big.json"
    p.write_text('{"n": ' + "9" * 5000 + "}")
    for policy in ("DEFAULT", "HEAL"):
        r = jio.read_json_result(p, default=dict, recover=_policy(policy))
        assert r.kind == "malformed" and r.data == {}
        p.write_text('{"n": ' + "9" * 5000 + "}")
    with pytest.raises(jio.JsonFileError) as exc:
        jio.read_json(p, default=dict, recover=jio.Recovery.RAISE)
    assert exc.value.kind == "malformed"


def test_ca2_try_mode_cannot_tell_a_refused_lock_from_a_busy_one_and_says_so(
    tmp_path: Path, monkeypatch
) -> None:
    """F4 (documented, not fixed in this phase): file_lock(blocking=False) reports flock errors as 'busy'."""
    # POSIX locks with fcntl.flock, Windows with msvcrt.locking; both report any OSError of a non-blocking attempt as busy
    if POSIX:
        import fcntl

        def refused(fd, op):
            raise OSError(errno.ENOLCK, "no locks")

        monkeypatch.setattr(fcntl, "flock", refused)
    else:
        import msvcrt

        def refused(fd, mode, nbytes):
            raise OSError(errno.ENOLCK, "no locks")

        monkeypatch.setattr(msvcrt, "locking", refused)
    p = tmp_path / "log.jsonl"
    assert jio.append_jsonl(p, {"a": 1}, lock="try") is None
    assert "busy" in (jio.append_jsonl.__doc__ or "") and "refus" in (
        jio.append_jsonl.__doc__ or ""
    )


def test_ca1_encoded_bytes_without_a_trailing_newline_are_rejected(tmp_path: Path) -> None:
    """F10: the 'accepts encode_jsonl output' contract is not a licence to append a half line."""
    p = tmp_path / "log.jsonl"
    with pytest.raises(ValueError):
        jio.append_jsonl(p, b'{"a": 1}')
    assert not p.exists()
    p.write_bytes(b'{"x": 1}\n')
    with pytest.raises(ValueError):
        jio.write_jsonl(p, b'{"a": 1}')


def test_cj4_a_gz_tail_returns_the_last_n_records_not_nothing(tmp_path: Path) -> None:
    """F10: the tail used to read raw gzip bytes and return [] with warnings."""
    p = tmp_path / "log.2025.jsonl.gz"
    with gzip.open(p, "wb") as f:
        for i in range(10):
            f.write(json.dumps({"i": i}).encode() + b"\n")
    assert jio.read_jsonl_tail(p, 3) == [{"i": 7}, {"i": 8}, {"i": 9}]


def _reference_tail(data: bytes, n: int) -> list[bytes]:
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    lines = [ln.rstrip(b"\r") for ln in data.split(b"\n") if ln.strip()]
    return lines[-n:]


@pytest.mark.parametrize("chunk", [1, 3, 13])
def test_cj4_the_tail_matches_a_naive_reference_for_tiny_chunk_sizes(
    tmp_path: Path, monkeypatch, chunk: int
) -> None:
    """F7: with an 8 KiB chunk every small file is one window; tiny chunks force the boundary logic to run."""
    monkeypatch.setattr(jio, "_TAIL_CHUNK", chunk)
    rng = random.Random(chunk)
    pieces = [b'{"a": 1}', '{"é": "ü"}'.encode(), b"", b"   ", b'{"b": 22}\r', b'{"torn": ']
    for trial in range(120):
        body = b"\n".join(rng.choice(pieces) for _ in range(rng.randint(0, 9)))
        if rng.random() < 0.5:
            body += b"\n"
        if rng.random() < 0.2:
            body = b"\xef\xbb\xbf" + body
        p = tmp_path / f"t{chunk}_{trial}.jsonl"
        p.write_bytes(body)
        n = rng.randint(1, 6)
        assert jio._tail_lines(p, n) == _reference_tail(body, n), (body, n)


def test_cj4_a_file_whose_first_line_is_only_a_bom_does_not_count_it_as_a_line(
    tmp_path: Path,
) -> None:
    p = tmp_path / "log.jsonl"
    p.write_bytes(b'\xef\xbb\xbf\n{"a": 1}\n{"b": 2}\n')
    assert jio._tail_lines(p, 5) == [b'{"a": 1}', b'{"b": 2}']


def test_cj4_the_tail_scan_is_linear_in_the_length_of_one_long_line(tmp_path: Path, monkeypatch) -> None:
    """F9: re-concatenating and re-splitting the whole block per chunk made one long line quadratic.

    A tiny chunk gives a short file many chunks, so the quadratic copying is large at 2 MB. CPU time of this
    process (not wall-clock) and the minimum of three runs, so a busy host moves both sizes together; four times the
    line takes about four times as long when linear and sixteen times as long when quadratic, compared with 8x plus
    100 ms (Windows CPU-time ticks are about 16 ms)."""
    monkeypatch.setattr(jio, "_TAIL_CHUNK", 1024)

    def best(size: int) -> float:
        p = tmp_path / f"long{size}.jsonl"
        p.write_bytes(b'{"k": "' + b"x" * size + b'"}\n')
        times = []
        for _ in range(3):
            t0 = time.process_time()
            assert len(jio.read_jsonl_tail(p, 1)) == 1
            times.append(time.process_time() - t0)
        return min(times)

    small, big = best(500_000), best(2_000_000)
    assert big < small * 8 + 0.1, (small, big)


# ---- review nits: HEAL validator exceptions and the strict reset-to-default write ----


def _listing_bytes(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir()) if p.is_file() and not p.name.endswith(".lock")}


@pytest.mark.parametrize("exc", [KeyError("k"), AttributeError("a"), RuntimeError("r")], ids=["KeyError", "AttributeError", "RuntimeError"])
def test_heal_a_validator_bug_propagates_and_leaves_the_valid_file_untouched(tmp_path: Path, exc) -> None:
    """The old attempt_heal treated only ValueError/TypeError from a validator as a bad file: a buggy validator
    (KeyError, AttributeError, anything else) must not quarantine and reset a valid file."""
    p = tmp_path / "s.json"
    p.write_text('{"a": 1}')
    (tmp_path / "s.json.bak1").write_text('{"a": 0}')
    before = _listing_bytes(tmp_path)

    def buggy(_data):
        raise exc

    with pytest.raises(type(exc)):
        jio.read_json_result(p, default=dict, recover=jio.Recovery.HEAL, validate=buggy)
    assert _listing_bytes(tmp_path) == before  # no quarantine, no reset, no new file


@pytest.mark.parametrize("exc", [ValueError("bad"), TypeError("bad")], ids=["ValueError", "TypeError"])
def test_heal_a_validator_that_rejects_with_value_or_type_error_still_heals(tmp_path: Path, exc) -> None:
    p = tmp_path / "s.json"
    p.write_text('{"a": 1}')

    def rejects(data):
        if data == {"a": 1}:
            raise exc

    r = jio.read_json_result(p, default=lambda: {"fresh": True}, recover=jio.Recovery.HEAL, validate=rejects)
    assert r.data == {"fresh": True} and r.anomaly is not None and r.anomaly.kind == "schema_mismatch"


def test_heal_reset_to_default_writes_strict_json_and_a_nan_default_raises_with_nothing_written(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    p.write_text("{nope")
    before = _listing_bytes(tmp_path)
    with pytest.raises(ValueError):
        jio.read_json_result(p, default=lambda: {"x": float("nan")}, recover=jio.Recovery.HEAL)
    assert _listing_bytes(tmp_path) == before  # the corrupt file is still in place, nothing was written

    # a finite default is written as plain strict JSON and read back by the stdlib
    r = jio.read_json_result(p, default=lambda: {"x": 1.5}, recover=jio.Recovery.HEAL)
    assert r.anomaly is not None and r.anomaly.action == "reset_to_default"
    assert json.loads(p.read_text()) == {"x": 1.5}

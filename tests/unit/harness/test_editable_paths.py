"""F5 sandbox-boundary extension — editable_paths + collision-guard + Bob-confirms (token-free).

Covers criteria C1–C9 + C11 (1.5-criteria.md). Zero model tokens: real `sandbox()`, real
`_validate_editable_paths`, real brain pending/commit APIs (seeded via `brain.files.pending.create`),
real `warnings` capture — no `claude` subprocess, no live provider. Fake-HOME pattern mirrors
`test_sandbox_isolation.py`. Each leak criterion pairs an oracle-can-fail demo (the un-excluded /
default-OFF write DOES trip), so a clean excluded result is trustworthy.
"""

from __future__ import annotations

import importlib
import warnings
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.harness import (
    HARNESS_EDITABLE_SENTINEL,
    DumbBob,
    EditablePathRefused,
    SandboxLeak,
    sandbox,
)
from tests.harness.bob import _commit_editable_pending

# The submodule object (NOT the re-exported `sandbox` FUNCTION that shadows it on the package).
sandbox_mod = importlib.import_module("tests.harness.sandbox")


def _seed_fake_cred(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point HOME at a tmp dir with a fake ~/.claude/.credentials.json (never the real one). Returns
    the fake home. Also point platformdirs' Documents resolver at <fake-home>/Documents so the notes
    shallow-scan and any editable notes folder share one Documents dir."""
    fake_home = tmp_path / "fake-home"
    (fake_home / ".claude").mkdir(parents=True)
    (fake_home / ".claude" / ".credentials.json").write_text('{"fake": true}')
    (fake_home / "Documents").mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    # Keep the notes-scan Documents dir == Path.home()/Documents for the fake home (F-6 resolver).
    monkeypatch.setattr(sandbox_mod, "_documents_dir", lambda: fake_home / "Documents")
    return fake_home


def _mark(d: Path) -> Path:
    """Create a directory carrying the harness sentinel (so the collision-guard accepts it)."""
    d.mkdir(parents=True, exist_ok=True)
    (d / HARNESS_EDITABLE_SENTINEL).write_text("test-owned")
    return d


# --- C3/C4: collision-guard (sentinel-mandatory) --------------------------------------------------


def test_c3_guard_refuses_unmarked_existing_dir_empty_or_populated(tmp_path: Path) -> None:
    """C3: an existing directory WITHOUT the sentinel is REFUSED — empty OR populated (MAJOR-1 fix).
    Empty-real `~/Documents/Canary Notes` must not be accepted."""
    empty = tmp_path / "Canary Notes"
    empty.mkdir()
    with pytest.raises(EditablePathRefused):
        with sandbox(editable_paths=[empty], live_check="off"):
            pass

    populated = tmp_path / "Real Notes"
    populated.mkdir()
    (populated / "diary.md").write_text("real user data")
    with pytest.raises(EditablePathRefused):
        with sandbox(editable_paths=[populated], live_check="off"):
            pass


def test_c3_guard_refuses_existing_regular_file(tmp_path: Path) -> None:
    """C3/CH8-a: an editable path that exists as a regular file (not a directory) is REFUSED."""
    f = tmp_path / "notes.txt"
    f.write_text("x")
    with pytest.raises(EditablePathRefused):
        with sandbox(editable_paths=[f], live_check="off"):
            pass


def test_c4_guard_accepts_nonexistent_and_sentinel_marked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C4: the guard ACCEPTS only (a) a non-existent path and (b) a sentinel-marked directory."""
    _seed_fake_cred(monkeypatch, tmp_path)
    nonexistent = tmp_path / "fake-home" / "Documents" / "Canary Notes"  # not created
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with sandbox(editable_paths=[nonexistent], live_check="off") as sb:
            assert sb.root.exists()

    marked = _mark(tmp_path / "fake-home" / "Documents" / "Marked Notes")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with sandbox(editable_paths=[marked], live_check="off") as sb:
            assert sb.root.exists()


# --- C1: leak-exclusion is exact (named path does NOT trip) ---------------------------------------


def test_c1_write_to_editable_notes_folder_does_not_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C1: a real write to a declared editable `*Notes` folder under Documents does NOT raise
    SandboxLeak; a CHILD write under it also does not. Oracle-can-fail: the SAME write DOES trip when
    the folder is NOT declared editable (test_c5_default_off_...)."""
    fake_home = _seed_fake_cred(monkeypatch, tmp_path)
    editable = _mark(fake_home / "Documents" / "Canary Notes")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with sandbox(editable_paths=[editable], live_check="off") as sb:
            _ = sb.root
            (editable / "note.md").write_text("she wrote this")          # direct child
            (editable / "sub").mkdir()
            (editable / "sub" / "deep.md").write_text("nested child")     # deeper child
    # Clean exit — no SandboxLeak for the declared editable folder or its children.


def test_c1_write_to_editable_path_under_guarded_root_does_not_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C1 (recursive-root mechanism): an editable path UNDER a fingerprinted guarded root is excluded
    from the recursive `_fingerprint` walk — a write there does not trip. Uses extra_guard_roots to
    make a controllable guarded root, with the editable path nested inside it."""
    _seed_fake_cred(monkeypatch, tmp_path)
    guarded = tmp_path / "guarded"
    guarded.mkdir()
    (guarded / "pre.txt").write_text("pre")
    editable = _mark(guarded / "editable-sub")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with sandbox(
            extra_guard_roots=[guarded], editable_paths=[editable], live_check="off"
        ) as sb:
            _ = sb.root
            (editable / "landed.txt").write_text("under a guarded root but excluded")
    # Clean exit — the editable subtree was pruned from the guarded-root fingerprint.


# --- C2: not one path broader (just-outside STILL trips) ------------------------------------------


def test_c2_sibling_notes_folder_still_trips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C2: with `Canary Notes` declared editable, a write to a SIBLING `*Notes` folder STILL trips —
    the exclusion is exactly the named basename, not the whole `*Notes` family."""
    fake_home = _seed_fake_cred(monkeypatch, tmp_path)
    editable = _mark(fake_home / "Documents" / "Canary Notes")
    with pytest.raises(SandboxLeak):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with sandbox(editable_paths=[editable], live_check="off") as sb:
                _ = sb.root
                sibling = fake_home / "Documents" / "Other Notes"
                sibling.mkdir()  # a NEW sibling *Notes folder appearing is the leak


def test_c2_other_guarded_root_still_trips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C2: an editable notes folder does not weaken the rest of the guard — a write to ~/.claude STILL
    trips SandboxLeak."""
    fake_home = _seed_fake_cred(monkeypatch, tmp_path)
    editable = _mark(fake_home / "Documents" / "Canary Notes")
    with pytest.raises(SandboxLeak):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with sandbox(editable_paths=[editable], live_check="off") as sb:
                _ = sb.root
                (fake_home / ".claude" / "settings.json").write_text("leaked config")


# --- C5: default-OFF is byte-identical + fully guarded --------------------------------------------


def test_c5_default_off_notes_write_still_trips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C5 + C1 oracle-can-fail: with NO editable_paths, a write to `~/Documents/Canary Notes` STILL
    raises SandboxLeak (the exact write C1 shows is silenced when the folder IS declared)."""
    fake_home = _seed_fake_cred(monkeypatch, tmp_path)
    with pytest.raises(SandboxLeak):
        with sandbox(live_check="off") as sb:  # no editable_paths → default-OFF
            _ = sb.root
            notes = fake_home / "Documents" / "Canary Notes"
            notes.mkdir()


def test_c5_default_off_emits_no_editable_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C5/C6-negative: default-OFF emits NO editable-paths warning (so C6's positive can fail)."""
    _seed_fake_cred(monkeypatch, tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with sandbox(live_check="off") as sb:
            _ = sb.root
    assert not any(
        "editable_paths ACTIVE" in str(w.message) for w in caught
    ), "default-OFF must not emit the editable-paths declaration"


# --- C6: loud declaration -------------------------------------------------------------------------


def test_c6_loud_declaration_names_each_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C6: with editable_paths set, sandbox start emits a RuntimeWarning naming each real path."""
    fake_home = _seed_fake_cred(monkeypatch, tmp_path)
    editable = _mark(fake_home / "Documents" / "Canary Notes")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with sandbox(editable_paths=[editable], live_check="off") as sb:
            _ = sb.root
    decls = [w for w in caught if "editable_paths ACTIVE" in str(w.message)]
    assert decls, "expected a loud editable-paths declaration"
    assert any(str(editable) in str(w.message) for w in decls), "declaration must name the path"
    assert all(issubclass(w.category, RuntimeWarning) for w in decls)


# --- C7: Bob-confirms lands a real write via UNMODIFIED brain, scoped to editable paths -----------


def _propose_pending(persona_dir: Path, target: Path, content: str, op: str = "create") -> str:
    """Seed a PENDING write via the REAL brain pending store (as notes-off propose_write would)."""
    from brain.files import pending

    return pending.create(
        persona_dir,
        op=op,
        resolved_path=str(target),
        content=content,
        now=datetime.now(UTC),
    )


def test_c7_bob_confirms_in_scope_skips_out_of_scope_and_append_missing(
    tmp_path: Path,
) -> None:
    """C7: DumbBob.confirm_writes (the Bob-protocol METHOD) commits an in-scope create, SKIPS an
    out-of-scope write, and does NOT land an in-scope append-to-missing (brain refuses it). Records
    are seeded with DISTINCT ids (vary target+content so the sha256(resolved_path+now+content[:64])
    rid does not collapse — S-6/CH8-c)."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    editable = _mark(tmp_path / "Canary Notes")
    in_scope_target = editable / "landed.md"
    out_target = tmp_path / "elsewhere" / "escape.md"
    (tmp_path / "elsewhere").mkdir()
    append_missing_target = editable / "not-yet.md"  # append to a file that doesn't exist

    rid_in = _propose_pending(persona_dir, in_scope_target, "IN-SCOPE content", op="create")
    rid_out = _propose_pending(persona_dir, out_target, "OUT-OF-SCOPE content", op="create")
    rid_append = _propose_pending(
        persona_dir, append_missing_target, "APPEND to missing", op="append"
    )
    assert len({rid_in, rid_out, rid_append}) == 3, "seeded rids must be distinct (S-6)"

    bob = DumbBob("claude", "any-mood")
    committed = bob.confirm_writes(persona_dir, [editable])

    # in-scope create landed:
    assert rid_in in committed
    assert in_scope_target.is_file()
    assert in_scope_target.read_text() == "IN-SCOPE content"

    # out-of-scope skipped: no file, still pending:
    assert rid_out not in committed
    assert not out_target.exists()

    # append-to-missing: not committed (brain refuses append to a nonexistent target), no file:
    assert rid_append not in committed
    assert not append_missing_target.exists()

    # real pending-record statuses reflect brain's outcomes:
    from brain.files import pending

    assert pending.get(persona_dir, rid_in)["status"] == "committed"
    assert pending.get(persona_dir, rid_out)["status"] == "pending"
    assert pending.get(persona_dir, rid_append)["status"] == "refused"


def test_c7_confirm_writes_empty_editable_is_noop(tmp_path: Path) -> None:
    """C7 edge: no editable paths → nothing committed (defensive)."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _propose_pending(persona_dir, tmp_path / "x.md", "content")
    assert _commit_editable_pending(persona_dir, []) == []


# --- C11: nested (non-direct-child) editable path + symlinked-Documents exactness -----------------


def test_c11_nested_editable_notes_folder_does_not_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C11/CH8-b: a nested editable path (NOT a direct child of Documents) is handled — the shallow
    notes scan only lists direct children, so a write to `<Documents>/sub/Canary Notes` neither
    trips nor needs the notes exclusion. No SandboxLeak."""
    fake_home = _seed_fake_cred(monkeypatch, tmp_path)
    nested = _mark(fake_home / "Documents" / "sub" / "Canary Notes")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with sandbox(editable_paths=[nested], live_check="off") as sb:
            _ = sb.root
            (nested / "note.md").write_text("nested write")
    # Clean exit — nested editable write does not trip.


def test_c11_symlinked_documents_editable_still_excluded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """C11 (MINOR-4): when Documents is reached through a symlink, an editable notes folder is STILL
    correctly excluded (parent match resolves both sides). Skipped where symlinks are unavailable."""
    fake_home = _seed_fake_cred(monkeypatch, tmp_path)
    real_docs = fake_home / "Documents"
    editable = _mark(real_docs / "Canary Notes")
    # Point the notes-scan Documents dir at a SYMLINK to the real Documents.
    link = tmp_path / "docs-link"
    try:
        link.symlink_to(real_docs, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    monkeypatch.setattr(sandbox_mod, "_documents_dir", lambda: link)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with sandbox(editable_paths=[editable], live_check="off") as sb:
            _ = sb.root
            (editable / "note.md").write_text("via symlinked docs")
    # Clean exit — resolve-both-sides parent match kept the exclusion exact through the symlink.

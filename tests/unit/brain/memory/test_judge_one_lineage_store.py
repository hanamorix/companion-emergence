"""F2c inc7 — the ONE per-persona tuned-judge store (plain checkpoints only).

Criteria (changes/f2c-inc7-one-lineage/1.5-criteria.md): S1 (one store, one
pointer, no per-tier sub-store), S3 (delete-after-swap retry path:
`reap_unreferenced`), S5 (fail-soft resolve; legacy adapter dirs never served).
The adapter-era names that must stay removed are listed in
tests/unit/test_removed_names_stay_removed.py.
Torch-free: filesystem only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.memory import judge_lora, judge_selftune


def _place(root: Path, *, adapter_layout: bool = False, point: bool = True) -> Path:
    d = judge_lora.staged_adapter_path(root)
    d.mkdir(parents=True)
    if adapter_layout:
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    else:
        (d / "model.safetensors.txt").write_text("x", encoding="utf-8")
    if point:
        judge_lora.swap_champion_pointer(root, d)
    return d


def _dirs(root: Path) -> set[str]:
    return {e.name for e in root.iterdir() if e.is_dir()} if root.is_dir() else set()


# ---------------------------------------------------------------------------
# S1 / S5 — resolve_current_checkpoint
# ---------------------------------------------------------------------------


def test_resolve_current_checkpoint_absent_dangling_adapter_and_plain(tmp_path: Path) -> None:
    root = judge_lora.champion_dir(tmp_path)
    assert judge_lora.resolve_current_checkpoint(tmp_path) is None  # absent -> base

    plain = _place(root)
    assert judge_lora.resolve_current_checkpoint(tmp_path) == plain

    legacy = _place(root, adapter_layout=True)
    assert judge_lora.resolve_current_checkpoint(tmp_path) is None, "an adapter dir is never served"
    assert legacy.exists()

    (root / "current").write_text("adapter-does-not-exist", encoding="utf-8")
    assert judge_lora.resolve_current_checkpoint(tmp_path) is None  # dangling -> base


def test_store_root_is_the_single_per_persona_root(tmp_path: Path) -> None:
    assert judge_lora.champion_dir(tmp_path) == tmp_path / "models" / "relevance_judge"


# ---------------------------------------------------------------------------
# S3 — reap_unreferenced (the Q1 retry path) + helpers
# ---------------------------------------------------------------------------


def test_reap_keeps_only_the_pointer_named_dir(tmp_path: Path) -> None:
    root = judge_lora.champion_dir(tmp_path)
    leftover = _place(root, point=False)
    served = _place(root)
    (root / "full").mkdir()  # a legacy per-tier sub-root: not prefixed, never touched
    # a pre-#465 leftover temp name (the pointer is now written through the shared writer, as `current.t<pid>-<hex>.new`);
    # either shape is a non-adapter entry that the reap must leave alone
    (root / ".current.tmp-x").write_text("tmp", encoding="utf-8")
    judge_lora.reap_unreferenced(root)
    assert not leftover.exists()
    assert served.exists() and (root / "full").is_dir() and (root / ".current.tmp-x").exists()
    assert judge_lora.resolve_current_checkpoint(tmp_path) == served


def test_reap_keeps_a_pointer_named_dir_even_if_it_is_an_adapter(tmp_path: Path) -> None:
    root = judge_lora.champion_dir(tmp_path)
    legacy = _place(root, adapter_layout=True)
    other = _place(root, point=False)
    judge_lora.reap_unreferenced(root)
    assert legacy.exists() and not other.exists()


def test_reap_with_no_pointer_removes_every_stored_dir(tmp_path: Path) -> None:
    root = judge_lora.champion_dir(tmp_path)
    a = _place(root, point=False)
    b = _place(root, point=False)
    judge_lora.reap_unreferenced(root)
    assert not a.exists() and not b.exists()


def test_reap_skips_when_the_pointer_is_empty(tmp_path: Path) -> None:
    root = judge_lora.champion_dir(tmp_path)
    d = _place(root)
    (root / "current").write_text("", encoding="utf-8")
    judge_lora.reap_unreferenced(root)
    assert d.exists()


def test_reap_skips_when_the_pointer_cannot_be_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """S3d: a pointer read error must never turn into 'reap everything'."""
    root = judge_lora.champion_dir(tmp_path)
    served = _place(root)
    leftover = _place(root, point=False)
    judge_lora.swap_champion_pointer(root, served)
    real_read_text = Path.read_text

    def read_text(self, *a, **k):
        if self.name == "current":
            raise PermissionError("locked")
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", read_text)
    judge_lora.reap_unreferenced(root)
    assert served.exists() and leftover.exists()


def test_reap_never_raises_on_a_missing_root(tmp_path: Path) -> None:
    judge_lora.reap_unreferenced(tmp_path / "nope")


def test_refused_delete_is_logged_not_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = judge_lora.champion_dir(tmp_path)
    leftover = _place(root, point=False)
    _place(root)

    def refuse(path, *a, **k):
        raise PermissionError("[WinError 32] in use")

    monkeypatch.setattr(judge_lora.shutil, "rmtree", refuse)
    judge_lora.reap_unreferenced(root)  # must not raise
    judge_lora.discard_stored_dir(leftover)  # must not raise
    assert leftover.exists()


def test_read_pointer_name_and_discard(tmp_path: Path) -> None:
    root = judge_lora.champion_dir(tmp_path)
    assert judge_lora.read_pointer_name(root) is None
    d = _place(root)
    assert judge_lora.read_pointer_name(root) == d.name
    judge_lora.discard_stored_dir(d)
    assert not d.exists()
    judge_lora.discard_stored_dir(d)  # idempotent


def test_keep_after_accept_keeps_only_the_new_checkpoint() -> None:
    """Ruling Q1: the previous checkpoint is deleted in the same tick."""
    assert judge_selftune._keep_after_accept("adapter-new") == ["adapter-new"]


def test_training_start_is_the_current_model_else_base(tmp_path: Path) -> None:
    from brain.bridge.model_tier import MODEL_RELEVANCE_JUDGE

    assert judge_selftune._training_start(None) == MODEL_RELEVANCE_JUDGE
    assert judge_selftune._training_start(tmp_path / "adapter-x") == str(tmp_path / "adapter-x")

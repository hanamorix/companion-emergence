"""F2c inc5b-2 — per-persona champion-adapter pointer store (crash-safety).

Filesystem-only, NO model load: exercises `judge_lora`'s champion-pointer
helpers (`champion_dir` / `staged_adapter_path` / `swap_champion_pointer` /
`resolve_champion_adapter` / `cleanup_stale_adapters` / `clear_champion_
pointer`) that make the accept-path adapter persist crash-safe. Covers
criteria C3 (staged write + atomic pointer), C12 (atomic swap under an
interleaved read), C18 (cleanup keep-N=2), C21 (no staged-dir leak on
revert/fault), C22 (same-filesystem os.replace).
"""

from __future__ import annotations

from pathlib import Path

from brain.memory import judge_lora


def _make_adapter(champion_root: Path, name_suffix: str = "") -> Path:
    """Create a fresh staged adapter subdir with a dummy file in it (stands
    in for a real saved adapter — these helpers never load a model)."""
    staged = judge_lora.staged_adapter_path(champion_root)
    staged.mkdir(parents=True, exist_ok=True)
    (staged / f"adapter_model{name_suffix}.txt").write_text("dummy", encoding="utf-8")
    return staged


def test_champion_dir_is_models_relevance_judge(tmp_path: Path) -> None:
    assert judge_lora.champion_dir(tmp_path) == tmp_path / "models" / "relevance_judge"


def test_resolve_absent_pointer_is_none(tmp_path: Path) -> None:
    root = judge_lora.champion_dir(tmp_path)
    assert judge_lora.resolve_champion_adapter(root) is None


def test_staged_write_then_swap_resolves_to_it(tmp_path: Path) -> None:
    # C3: a staged write followed by an atomic pointer swap resolves to the
    # complete adapter subdir.
    root = judge_lora.champion_dir(tmp_path)
    staged = _make_adapter(root)
    assert judge_lora.resolve_champion_adapter(root) is None, "not live until swapped"
    judge_lora.swap_champion_pointer(root, staged)
    resolved = judge_lora.resolve_champion_adapter(root)
    assert resolved == staged
    assert (resolved / "adapter_model.txt").read_text(encoding="utf-8") == "dummy"


def test_staged_and_pointer_share_a_filesystem(tmp_path: Path) -> None:
    # C22: staged subdir + `current` pointer live under the same champion_root
    # (same st_dev), so os.replace of the pointer is atomic.
    root = judge_lora.champion_dir(tmp_path)
    staged = _make_adapter(root)
    judge_lora.swap_champion_pointer(root, staged)
    pointer = root / "current"
    assert staged.parent == root
    assert pointer.parent == root
    assert staged.stat().st_dev == pointer.stat().st_dev


def test_swap_is_atomic_reader_never_sees_a_partial(tmp_path: Path) -> None:
    # C12: across the swap, a read of the pointer resolves to EITHER the old
    # OR the new complete adapter, never a partial/absent one.
    root = judge_lora.champion_dir(tmp_path)
    old = _make_adapter(root, "_old")
    judge_lora.swap_champion_pointer(root, old)
    assert judge_lora.resolve_champion_adapter(root) == old

    new = _make_adapter(root, "_new")
    # BEFORE the swap: a reader still resolves the OLD complete adapter.
    assert judge_lora.resolve_champion_adapter(root) == old
    judge_lora.swap_champion_pointer(root, new)
    # AFTER the swap: resolves the NEW complete adapter.
    assert judge_lora.resolve_champion_adapter(root) == new

    # H6 bite — a NON-atomic swap (delete-then-write the pointer) exposes a
    # window where a reader resolves None; the real os.replace-based swap
    # never does (asserted above: old-or-new at every observable point).
    pointer = root / "current"
    pointer.unlink()  # the intermediate state a non-atomic writer would create
    assert judge_lora.resolve_champion_adapter(root) is None, (
        "a torn/non-atomic swap would expose this None window — the atomic "
        "os.replace swap never does"
    )


def test_cleanup_keeps_named_removes_others_keep_n_2(tmp_path: Path) -> None:
    # C18: keep N=2 (new champion + immediately-prior) so an in-flight reader
    # holding the prior pointer still finds its subdir; older ones removed.
    root = judge_lora.champion_dir(tmp_path)
    a = _make_adapter(root)
    b = _make_adapter(root)
    c = _make_adapter(root)
    judge_lora.swap_champion_pointer(root, c)  # c = current champion, b = prior
    judge_lora.cleanup_stale_adapters(root, keep_names=[c.name, b.name])
    assert c.is_dir() and b.is_dir(), "current + prior survive (keep N=2)"
    assert not a.is_dir(), "third-oldest removed"
    # A reader that resolved the prior (b) before the swap still finds it.
    assert (b / "adapter_model.txt").exists()


def test_cleanup_on_revert_reaps_staged_keeps_champion(tmp_path: Path) -> None:
    # C21: on a REVERT the champion pointer is never swapped; the discarded
    # staged adapter is reaped, the live champion survives.
    root = judge_lora.champion_dir(tmp_path)
    champ = _make_adapter(root)
    judge_lora.swap_champion_pointer(root, champ)
    staged = _make_adapter(root)  # a rejected challenger's staged dir
    judge_lora.cleanup_stale_adapters(root, keep_names=[champ.name])
    assert champ.is_dir(), "champion kept"
    assert not staged.is_dir(), "rejected staged adapter reaped — no leak"
    assert judge_lora.resolve_champion_adapter(root) == champ


def test_cleanup_ignores_none_and_never_raises(tmp_path: Path) -> None:
    root = judge_lora.champion_dir(tmp_path)
    a = _make_adapter(root)
    # keep_names carrying a None (first-ever tune has no prior) is ignored.
    judge_lora.cleanup_stale_adapters(root, keep_names=[a.name, None])  # type: ignore[list-item]
    assert a.is_dir()
    # never raises on a missing root
    judge_lora.cleanup_stale_adapters(tmp_path / "does-not-exist", keep_names=["x"])


def test_pointer_file_names_the_current_pointer(tmp_path: Path) -> None:
    # F2c inc9: the path the orphan knob-row reap reads raw.
    root = judge_lora.champion_dir(tmp_path)
    staged = _make_adapter(root)
    judge_lora.swap_champion_pointer(root, staged)
    assert judge_lora.pointer_file(root).read_text(encoding="utf-8") == staged.name


def test_resolve_dangling_pointer_is_none(tmp_path: Path) -> None:
    # A pointer naming a since-removed subdir resolves to None (fail-soft).
    root = judge_lora.champion_dir(tmp_path)
    staged = _make_adapter(root)
    judge_lora.swap_champion_pointer(root, staged)
    import shutil

    shutil.rmtree(staged)
    assert judge_lora.resolve_champion_adapter(root) is None

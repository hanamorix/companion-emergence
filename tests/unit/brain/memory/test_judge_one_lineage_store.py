"""F2c inc7 — the ONE per-persona tuned-judge store (plain checkpoints only)
and the source sweeps that pin the removal of the adapter-era / per-tier code.

Criteria (changes/f2c-inc7-one-lineage/1.5-criteria.md): S1 (one store, one
pointer, no per-tier sub-store), S3 (delete-after-swap retry path:
`reap_unreferenced`), S5 (fail-soft resolve; legacy adapter dirs never served),
M4 (no adapter save/reload path remains), H3 (repo-wide removed-symbol sweep).
Torch-free: filesystem + source inspection only.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from brain.memory import judge_lora, judge_selftune

REPO = Path(__file__).resolve().parents[4]
BRAIN = REPO / "brain"
TESTS = REPO / "tests"


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
    assert not hasattr(judge_lora, "full_champion_dir")
    assert not hasattr(judge_lora, "resolve_serving_tuned_judge")


# ---------------------------------------------------------------------------
# S3 — reap_unreferenced (the Q1 retry path) + helpers
# ---------------------------------------------------------------------------


def test_reap_keeps_only_the_pointer_named_dir(tmp_path: Path) -> None:
    root = judge_lora.champion_dir(tmp_path)
    leftover = _place(root, point=False)
    served = _place(root)
    (root / "full").mkdir()  # a legacy per-tier sub-root: not prefixed, never touched
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
    assert judge_selftune._keep_after_accept("adapter-new", "adapter-old") == ["adapter-new"]
    assert judge_selftune._keep_after_accept("adapter-new", None) == ["adapter-new"]


def test_training_start_is_the_current_model_else_base(tmp_path: Path) -> None:
    from brain.bridge.model_tier import MODEL_RELEVANCE_JUDGE

    assert judge_selftune._training_start(None) == MODEL_RELEVANCE_JUDGE
    assert judge_selftune._training_start(tmp_path / "adapter-x") == str(tmp_path / "adapter-x")


# ---------------------------------------------------------------------------
# Source sweeps (M4, S1, H3). Each sweep has a self-test on a scratch source so
# a "no hits" result is shown able to fail (ST1.5f).
# ---------------------------------------------------------------------------

_ADAPTER_IO_ATTRS = {"load_adapter", "load_lora_scorer"}
_REMOVED = ("full_champion_dir", "resolve_serving_tuned_judge", "LoraAdapterJudge", "load_lora_scorer", "save_adapter_dir")


def _adapter_io_calls(source: str) -> list[str]:
    """Calls that save or reload an adapter: `PeftModel.from_pretrained(...)`,
    `<x>.load_adapter(...)`, `load_lora_scorer(...)`."""
    hits: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute):
            if f.attr in _ADAPTER_IO_ATTRS:
                hits.append(f.attr)
            if f.attr == "from_pretrained" and isinstance(f.value, ast.Name) and f.value.id == "PeftModel":
                hits.append("PeftModel.from_pretrained")
        elif isinstance(f, ast.Name) and f.id in _ADAPTER_IO_ATTRS:
            hits.append(f.id)
    return hits


def _removed_symbol_uses(source: str) -> list[str]:
    """AST-level USES of a removed symbol (names, attributes, imports, keyword
    args) — comments and string literals do not count."""
    hits: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and node.id in _REMOVED:
            hits.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in _REMOVED:
            hits.append(node.attr)
        elif isinstance(node, ast.alias) and node.name in _REMOVED:
            hits.append(node.name)
        elif isinstance(node, ast.keyword) and node.arg in ("save_adapter_dir",):
            hits.append(node.arg)
        elif isinstance(node, ast.Call):
            fname = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if fname in ("build_judge_provider", "label_calibration_sample"):
                hits += [f"{fname}(adapter_dir=)" for kw in node.keywords if kw.arg == "adapter_dir"]
    return hits


def _py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if ".venv" not in p.parts and "__pycache__" not in p.parts]


def test_sweeps_fire_on_a_known_bad_source() -> None:
    bad = (
        "from peft import PeftModel\n"
        "from brain.memory.judge_lora import load_lora_scorer\n"
        "m = PeftModel.from_pretrained(b, d)\n"
        "model.load_adapter(x)\n"
        "judge_lora.full_champion_dir(p)\n"
        "build_judge_provider(adapter_dir='a')\n"
        "f(save_adapter_dir='s')\n"
    )
    assert set(_adapter_io_calls(bad)) == {"PeftModel.from_pretrained", "load_adapter"}
    uses = _removed_symbol_uses(bad)
    assert {"load_lora_scorer", "full_champion_dir", "build_judge_provider(adapter_dir=)", "save_adapter_dir"} <= set(uses)


def test_no_adapter_save_or_reload_path_remains_in_brain() -> None:
    """M4 (bite: at d25e1002 `load_lora_scorer` / `PeftModel.from_pretrained`
    are present in brain/memory/judge_lora.py)."""
    hits = {str(p.relative_to(REPO)): _adapter_io_calls(p.read_text(encoding="utf-8")) for p in _py_files(BRAIN)}
    assert {k: v for k, v in hits.items() if v} == {}
    src = inspect.getsource(judge_lora.build_lora_retrain_fn)
    assert "get_peft_model(" in src and "merge_and_unload()" in src


def test_no_removed_symbol_is_used_anywhere_in_brain_or_tests() -> None:
    """H3 + S1 (bite: at d25e1002 brain/ and tests/ use every one of them)."""
    offenders: dict[str, list[str]] = {}
    for p in _py_files(BRAIN) + _py_files(TESTS):
        uses = _removed_symbol_uses(p.read_text(encoding="utf-8"))
        if uses and p.resolve() != Path(__file__).resolve():
            offenders[str(p.relative_to(REPO))] = uses
    assert offenders == {}
    brain_text = "\n".join(p.read_text(encoding="utf-8") for p in _py_files(BRAIN))
    for name in (*_REMOVED, "sibling_root"):
        assert name not in brain_text, f"{name} still referenced in brain/"


def test_tick_and_serve_resolve_through_the_one_store() -> None:
    """S1 positive assertion."""
    from brain.bridge import supervisor

    assert "resolve_current_checkpoint(" in inspect.getsource(judge_selftune._run_weight_retrain)
    assert "resolve_current_checkpoint(" in inspect.getsource(supervisor._run_calibration_tick)

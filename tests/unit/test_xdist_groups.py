"""The xdist groups (#540): modules that share one real resource must stay in one group.

CI runs the suite with ``pytest -n <workers> --dist loadgroup``; tests carrying the same
``pytest.mark.xdist_group(name)`` then run one after another on one worker. Serially the mark does
nothing, so a deleted or misspelt mark would only show up as a flaky parallel run. This pins which
file belongs to which group and why.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# file -> (group, what the files share)
MODULE_GROUPS = {
    "tests/integration/test_systemd_install.py": ("systemd", "one fixed unit name in the one real user manager"),
    "tests/unit/brain/memory/test_embedding_real_model.py": ("hf-cache", "the shared Hugging Face cache"),
    "tests/unit/brain/memory/test_embedding_tokens_real.py": ("hf-cache", "the shared Hugging Face cache"),
    "tests/unit/brain/memory/test_relevance_judge_real_model.py": ("hf-cache", "the shared Hugging Face cache"),
    "tests/unit/brain/memory/test_reranker_real_model.py": ("hf-cache", "the shared Hugging Face cache"),
    "tests/unit/brain/memory/test_reranker_tokenizer_real.py": ("hf-cache", "the shared Hugging Face cache"),
    "tests/unit/brain/memory/test_judge_full_ft.py": ("torch-training", "CPU-heavy torch training"),
    "tests/unit/brain/memory/test_judge_lora.py": ("torch-training", "CPU-heavy torch training"),
    "tests/unit/brain/test_tunables_descriptions.py": ("tunables-probe", "one module-scoped fresh-interpreter probe"),
    "tests/unit/harness/test_dropin_integration.py": ("network-install", "a network install"),
}
# The torch-training modules also run on one torch thread (the `one_torch_thread` fixture in tests/conftest.py).
ONE_TORCH_THREAD = (
    "tests/unit/brain/memory/test_judge_full_ft.py",
    "tests/unit/brain/memory/test_judge_lora.py",
)
# file -> (test function, group)
FUNCTION_GROUPS = {
    "tests/unit/brain/bridge/test_c15c_stress.py": (
        "test_c15c_heartbeat_and_gated_items_60s_with_concurrent_writes_no_lock_errors",
        "stress",
    ),
}


def _is_group_mark(node: ast.AST, group: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "xdist_group"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == group
    )


def _module_marks(tree: ast.Module) -> list[ast.AST]:
    marks: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            marks.extend(ast.walk(node.value))
    return marks


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
@pytest.mark.parametrize("rel", sorted(MODULE_GROUPS))
def test_a_module_that_shares_a_resource_is_in_its_xdist_group(rel: str) -> None:
    group, shared = MODULE_GROUPS[rel]
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
    assert any(_is_group_mark(n, group) for n in _module_marks(tree)), (
        f"{rel} must carry pytestmark = pytest.mark.xdist_group({group!r}): it uses {shared}"
    )


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
@pytest.mark.parametrize("rel", sorted(FUNCTION_GROUPS))
def test_a_function_that_needs_its_own_group_carries_the_mark(rel: str) -> None:
    name, group = FUNCTION_GROUPS[rel]
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert funcs, f"{name} no longer exists in {rel}; update FUNCTION_GROUPS"
    assert any(_is_group_mark(d, group) for f in funcs for d in f.decorator_list), (
        f"{name} must carry @pytest.mark.xdist_group({group!r})"
    )


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
@pytest.mark.parametrize("rel", ONE_TORCH_THREAD)
def test_a_torch_training_module_runs_on_one_torch_thread(rel: str) -> None:
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
    uses = [
        n for n in _module_marks(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "usefixtures"
        and any(isinstance(a, ast.Constant) and a.value == "one_torch_thread" for a in n.args)
    ]
    assert uses, f"{rel} must carry pytestmark = pytest.mark.usefixtures('one_torch_thread')"

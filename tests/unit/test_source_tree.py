"""``tests/source_tree.py``: the process-wide copy of the product source that the whole-tree scanners read (#540)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests import source_tree

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_brain_sources_holds_exactly_the_python_files_of_brain() -> None:
    expected = {
        p.relative_to(REPO).as_posix()
        for p in (REPO / "brain").rglob("*.py")
        if "__pycache__" not in p.parts
    }
    sources = source_tree.brain_sources()
    assert set(sources) == expected
    assert len(sources) > 300, "a scan that finds almost nothing would pass every scanner"
    assert "brain/tunables.py" in sources and all(k.startswith("brain/") for k in sources)
    assert sources["brain/tunables.py"] == (REPO / "brain" / "tunables.py").read_text(encoding="utf-8")


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_the_entry_points_share_one_cached_copy_and_it_is_read_only() -> None:
    assert source_tree.tree_sources("brain") is source_tree.brain_sources()
    assert source_tree.tree_asts("brain") is source_tree.brain_trees()
    assert source_tree.tree_sources("brain/") is source_tree.brain_sources()  # another spelling of the same root
    assert source_tree.tree_asts("./brain") is source_tree.brain_trees()
    with pytest.raises(TypeError):
        source_tree.brain_sources()["brain/x.py"] = ""  # type: ignore[index]
    with pytest.raises(TypeError):
        source_tree.brain_trees()["brain/x.py"] = ast.parse("")  # type: ignore[index]


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_the_trees_are_the_parse_of_the_same_text() -> None:
    sources, trees = source_tree.brain_sources(), source_tree.brain_trees()
    assert list(trees) == list(sources)
    rel = "brain/tunables.py"
    assert ast.dump(trees[rel]) == ast.dump(ast.parse(sources[rel]))


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_suffixes_and_exclusions_select_files_by_the_relative_path() -> None:
    both = source_tree.tree_sources("brain", (".py", ".toml"))
    assert any(k.endswith(".toml") for k in both), "brain/ ships a .toml file (prompt_strings.toml)"
    assert set(source_tree.brain_sources()) < set(both)
    kept = source_tree.tree_sources("brain", (".py",), ("brain/memory/",))
    assert kept and not any(k.startswith("brain/memory/") for k in kept)
    assert set(kept) == {k for k in source_tree.brain_sources() if not k.startswith("brain/memory/")}


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_a_root_that_does_not_exist_is_empty_not_an_error() -> None:
    assert dict(source_tree.tree_sources("no/such/dir")) == {}

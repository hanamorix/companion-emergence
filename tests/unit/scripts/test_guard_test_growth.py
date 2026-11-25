"""The test-growth guard CLI (#746), run end to end on small throw-away git repositories.

Every repository is built and every ``main`` call is made in a module fixture, so the tests only look at the stored
return codes and output. The main repository's head commit adds a scanner test, a 2x3 grid and two near-identical
tests; two more repositories cover a deleted test file and a head test file that does not parse.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("guard_test_growth", REPO / "scripts" / "guard_test_growth.py")
guard = importlib.util.module_from_spec(_spec)
sys.modules["guard_test_growth"] = guard
_spec.loader.exec_module(guard)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the guard uses git worktrees and POSIX paths")

_MINI_PYPROJECT = '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'

_NEW_TESTS = textwrap.dedent(
    """\
    from pathlib import Path

    import pytest


    @pytest.mark.parametrize("path", sorted(Path(__file__).parent.glob("*.py")))
    def test_every_file_in_the_folder_exists(path):
        assert path.exists()


    @pytest.mark.parametrize("left", [1, 2])
    @pytest.mark.parametrize("right", [1, 2, 3])
    def test_a_grid_of_sums(left, right):
        assert left + right > 0


    def test_alpha_totals_the_values():
        values = [1, 2, 3, 4, 5, 6]
        total = sum(values)
        doubled = [value * 2 for value in values]
        assert total == 21
        assert doubled[0] == 2
        assert len(doubled) == 6


    def test_beta_totals_the_values():
        numbers = [1, 2, 3, 4, 5, 6]
        total = sum(numbers)
        doubled = [number * 2 for number in numbers]
        assert total == 21
        assert doubled[0] == 2
        assert len(doubled) == 6
    """
)


_OLD_TESTS = "def test_one():\n    assert True\n\n\ndef test_two():\n    assert True\n"


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.email=t@example.test", "-c", "user.name=t", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return done.stdout.strip()


def _base_repo(root: Path, name: str, extra: dict[str, str]) -> tuple[Path, str]:
    repo = root / name
    (repo / "tests").mkdir(parents=True)
    (repo / "brain").mkdir()
    (repo / "pyproject.toml").write_text(_MINI_PYPROJECT, encoding="utf-8")
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    marker_source = (REPO / "tests" / "no_base_failure.py").read_text(encoding="utf-8")
    (repo / "tests" / "no_base_failure.py").write_text(marker_source, encoding="utf-8")
    (repo / "brain" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "brain" / "x.py").write_text("VALUE = 1\n", encoding="utf-8")
    for rel, text in extra.items():
        (repo / rel).write_text(text, encoding="utf-8")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def _commit_head(repo: Path) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "head")
    return _git(repo, "rev-parse", "HEAD")


@dataclass
class Outcome:
    code: int
    out: str


def _invoke(repo: Path, base: str, head: str, *extra: str) -> Outcome:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = guard.main(["--repo", str(repo), "--base", base, "--head", head, *extra])
    return Outcome(code, buffer.getvalue())


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Outcome]:
    root = tmp_path_factory.mktemp("guard")
    results: dict[str, Outcome] = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("GITHUB_STEP_SUMMARY", raising=False)

        repo, base = _base_repo(root, "growth", {})
        (repo / "tests" / "test_new.py").write_text(_NEW_TESTS, encoding="utf-8")
        (repo / "brain" / "x.py").write_text("VALUE = 2\n", encoding="utf-8")
        head = _commit_head(repo)
        results["growth"] = _invoke(repo, base, head)
        results["growth_enforced"] = _invoke(repo, base, head, "--enforce")

        repo, base = _base_repo(root, "deleted", {"tests/test_old.py": _OLD_TESTS})
        _git(repo, "rm", "-q", "tests/test_old.py")
        results["deleted"] = _invoke(repo, base, _commit_head(repo))

        repo, base = _base_repo(root, "broken", {})
        (repo / "tests" / "test_broken.py").write_text("def test_x(:\n    pass\n", encoding="utf-8")
        results["broken"] = _invoke(repo, base, _commit_head(repo))
    return results


def test_the_guard_passes_without_enforce(runs: dict[str, Outcome]) -> None:
    assert runs["growth"].code == 0, runs["growth"].out


def test_the_totals_row_counts_the_added_ids(runs: dict[str, Outcome]) -> None:
    assert "| **Total** | 0 → 4 | 0 → 11 | +11 | −0 |" in runs["growth"].out


def test_the_scanner_the_grid_and_the_duplicate_pair_are_named(runs: dict[str, Outcome]) -> None:
    out = runs["growth"].out
    assert "`tests/test_new.py::test_every_file_in_the_folder_exists` [scanner]" in out
    assert "`tests/test_new.py::test_a_grid_of_sums` [cross-product]" in out
    assert "tests/test_new.py::test_alpha_totals_the_values" in out
    assert "tests/test_new.py::test_beta_totals_the_values" in out


def test_enforce_fails_on_the_flagged_tests(runs: dict[str, Outcome]) -> None:
    assert runs["growth_enforced"].code == 1


def test_a_deleted_test_file_is_counted_with_no_head_ids(runs: dict[str, Outcome]) -> None:
    assert runs["deleted"].code == 0, runs["deleted"].out
    assert "| tests/test_old.py | 2 → 0 | 2 → 0 | +0 | −2 |" in runs["deleted"].out


def test_a_head_test_file_that_does_not_parse_cannot_be_judged(runs: dict[str, Outcome]) -> None:
    assert runs["broken"].code == 1
    assert "can't judge" in runs["broken"].out

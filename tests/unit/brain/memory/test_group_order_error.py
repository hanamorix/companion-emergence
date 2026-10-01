"""A mis-edited result-order list stops the bridge at startup with an error that names
the bad entry, the file and the line number (name-recall fix, S143; criteria B1a-B1e).

The mechanism is run for real: a subprocess imports the modules with the list patched
(an import-time failure is what the bridge sees), and the location function is run on
temp files whose lines are known. Synthetic data only.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from brain import dev_constants
from brain.memory import semantic_recall as sr

_REPO = Path(__file__).resolve().parents[4]
_GOOD = list(dev_constants.RECALL_RESULT_GROUP_ORDER)


def _write_constants(path: Path, entries: list[object], *, pad: int = 3) -> int:
    """A one-element-per-line file; returns the 1-based line of the FIRST element."""
    lines = ['"""header"""'] + ["# pad"] * pad
    lines.append("RECALL_RESULT_GROUP_ORDER: tuple[object, ...] = (")
    first = len(lines) + 1
    lines += [f"    {e!r}," for e in entries]
    lines.append(")")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return first


def test_the_default_list_is_accepted() -> None:
    assert sr._group_order_problem(_GOOD) is None  # noqa: SLF001
    assert sr.result_group_order() == tuple(_GOOD)


@pytest.mark.parametrize(
    ("entries", "index", "needle"),
    [
        (lambda g: g[:1] + ["reranked_genuin"] + g[2:], 1, "'reranked_genuin'"),
        (lambda g: g[:3] + [g[0]] + g[4:], 3, "listed twice"),
        (lambda g: g[:2] + [7] + g[3:], 2, "7"),
    ],
)
def test_the_problem_is_the_bad_element_and_its_position(entries, index, needle) -> None:
    problem = sr._group_order_problem(entries(_GOOD))  # noqa: SLF001
    assert problem is not None
    assert problem[0] == index and needle in problem[1]
    assert f"entry {index + 1} of 6" in problem[1]


def test_a_missing_group_is_named() -> None:
    index, why = sr._group_order_problem(_GOOD[:5])  # noqa: SLF001
    assert index is None and "'keyword_family'" in why


@pytest.mark.parametrize("pad", [0, 3, 11])
def test_the_line_is_the_bad_elements_line_in_a_multi_line_file(tmp_path: Path, pad: int) -> None:
    """B1b: not line 1, not the assignment's line, not a constant."""
    entries = ["reranked_genuine", "reranked_genuin", "cosine_genuine", "keyword_genuine", "cosine_family", "keyword_family"]
    first = _write_constants(tmp_path / "dc.py", entries, pad=pad)
    for index in range(6):
        where, line = sr._dev_constants_location(index, tmp_path / "dc.py")  # noqa: SLF001
        assert where == str(tmp_path / "dc.py")
        assert line == first + index
    assert first - 1 != first  # the assignment line differs from every element line


def test_a_missing_group_has_the_assignments_line(tmp_path: Path) -> None:
    first = _write_constants(tmp_path / "dc.py", _GOOD[:5], pad=4)
    _where, line = sr._dev_constants_location(None, tmp_path / "dc.py")  # noqa: SLF001
    assert line == first - 1


def test_an_unreadable_or_unparseable_file_has_no_line(tmp_path: Path) -> None:
    assert sr._dev_constants_location(0, tmp_path / "missing.py")[1] is None  # noqa: SLF001
    (tmp_path / "bad.py").write_text("RECALL_RESULT_GROUP_ORDER = (\n", encoding="utf-8")
    assert sr._dev_constants_location(0, tmp_path / "bad.py")[1] is None  # noqa: SLF001
    (tmp_path / "other.py").write_text("X = 1\n", encoding="utf-8")
    assert sr._dev_constants_location(0, tmp_path / "other.py")[1] is None  # noqa: SLF001


def test_the_error_carries_entry_file_and_line_through_the_real_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1b2: the glue passes the right index; the file is read from `dev_constants.__file__`."""
    entries = list(_GOOD)
    entries[4] = "cosine_famly"
    first = _write_constants(tmp_path / "dc.py", entries, pad=6)
    monkeypatch.setattr(dev_constants, "__file__", str(tmp_path / "dc.py"))
    monkeypatch.setattr(dev_constants, "RECALL_RESULT_GROUP_ORDER", tuple(entries))
    with pytest.raises(ValueError) as exc:
        sr.result_group_order()
    text = str(exc.value)
    assert f"{tmp_path / 'dc.py'}:{first + 4}:" in text
    assert "'cosine_famly'" in text and "entry 5 of 6" in text


def test_the_error_says_the_line_is_unknown_when_the_source_is_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1e."""
    monkeypatch.setattr(dev_constants, "__file__", str(tmp_path / "gone.py"))
    monkeypatch.setattr(dev_constants, "RECALL_RESULT_GROUP_ORDER", tuple(_GOOD[:5]))
    with pytest.raises(ValueError) as exc:
        sr.result_group_order()
    text = str(exc.value)
    assert str(tmp_path / "gone.py") in text and "line unknown" in text and "'keyword_family'" in text


def _real_assignment_line() -> int:
    path = Path(dev_constants.__file__)
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.startswith("RECALL_RESULT_GROUP_ORDER"):
            return number
    raise AssertionError("assignment not found")


def _import_failure(patch: str, module: str) -> subprocess.CompletedProcess:
    code = f"import brain.dev_constants as d\n{patch}\nimport {module}\n"
    return subprocess.run(
        [sys.executable, "-c", code], cwd=_REPO, capture_output=True, text=True, timeout=300
    )


@pytest.mark.parametrize("module", ["brain.memory.semantic_recall", "brain.bridge.server"])
def test_a_mis_edited_list_stops_the_import_with_entry_file_and_line(module: str) -> None:
    """B1a/B1c on the real file: the bridge's startup import raises the error."""
    run = _import_failure(
        "d.RECALL_RESULT_GROUP_ORDER = tuple('reranked_genuin' if g == 'reranked_genuine' else g"
        " for g in d.RECALL_RESULT_GROUP_ORDER)",
        module,
    )
    assert run.returncode != 0
    real = Path(dev_constants.__file__)
    assert f"{real}:{_real_assignment_line()}:" in run.stderr
    assert "'reranked_genuin'" in run.stderr and "entry 1 of 6" in run.stderr


def test_a_missing_group_stops_the_import_naming_the_group() -> None:
    run = _import_failure(
        "d.RECALL_RESULT_GROUP_ORDER = d.RECALL_RESULT_GROUP_ORDER[:5]", "brain.memory.semantic_recall"
    )
    assert run.returncode != 0
    assert f"{Path(dev_constants.__file__)}:{_real_assignment_line()}:" in run.stderr
    assert "'keyword_family'" in run.stderr

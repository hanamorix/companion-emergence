"""#797: every citation of the body-state spec names a file that is in the repo.

The spec was cited from nine files under a private, gitignored folder, so it was never committed and was lost.
A citation has to name a tracked file, and every section number the code cites has to be a heading in that file.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_CITATION = re.compile(r"docs/[\w./-]*?body-state-design\.md")


def _citations() -> dict[str, list[str]]:
    """{cited path: [file:line, ...]} over brain/ and tests/."""
    found: dict[str, list[str]] = {}
    for root in ("brain", "tests"):
        for path in sorted((REPO / root).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "body-state-design" not in text:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                for match in _CITATION.finditer(line):
                    found.setdefault(match.group(0), []).append(f"{path.relative_to(REPO).as_posix()}:{lineno}")
    return found


@pytest.fixture(scope="module")
def citations() -> dict[str, list[str]]:
    """Read once in setup: the scan reads every file under brain/ and tests/ (the call-phase CPU budget, #787)."""
    return _citations()


def test_every_body_state_spec_citation_names_one_tracked_file(citations: dict[str, list[str]]) -> None:
    sites = sum(len(where) for where in citations.values())
    assert sites >= 9, f"expected the nine citations of the body-state spec, found {sites}: {citations}"
    assert len(citations) == 1, f"the body-state spec is cited under more than one path: {citations}"
    [(cited, where)] = citations.items()
    spec = REPO / cited
    assert spec.is_file(), f"{cited} is cited at {where} but does not exist"
    ignored = subprocess.run(["git", "check-ignore", "-q", cited], cwd=REPO, check=False).returncode == 0
    assert not ignored, f"{cited} is gitignored, so it is never committed (the cause of #797)"


# The sections the code cites, with the path or as a bare "§N" beside the body code (prompt.py, get_body_state.py, the tests).
_CITED_SECTIONS = ("2.1", "3.1", "3.2", "3.3", "4", "5.1", "7", "7.3")


def test_every_cited_section_is_a_heading_in_the_spec(citations: dict[str, list[str]]) -> None:
    [cited] = citations
    text = (REPO / cited).read_text(encoding="utf-8")
    headings = {m.group(1) for m in re.finditer(r"^#+\s*(\d+(?:\.\d+)?)\.?\s", text, flags=re.MULTILINE)}
    missing = [section for section in _CITED_SECTIONS if section not in headings]
    assert not missing, f"sections cited in the code have no heading in {cited}: {missing}"

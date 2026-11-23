"""#797: no tracked file cites a document under `docs/superpowers/`.

That folder is private on purpose: `.gitignore` ignores it and the pre-push leak guard (`hooks/denied-paths.txt`) refuses
it. Every spec, plan and audit written there was therefore never committed, and the ones the code cited are lost. A
citation has to name something a reader of this repository can open: a committed doc (`docs/source-spec/`), an issue, or
the module itself.

The files allowed to name the folder are the ones that keep it private, and the one script that writes its report there.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_PRIVATE = b"docs/superpowers/"
_TEXT_SUFFIXES = {".py", ".md", ".ts", ".tsx", ".js", ".toml", ".txt", ".sh", ".json", ".yml", ".yaml", ".rs", ".cfg", ".ini"}
_ALLOWED = {
    ".gitignore",  # ignores the folder
    "hooks/denied-paths.txt",  # the leak guard's refusal of it
    "hooks/test_leak_guard.sh",  # the leak guard's own tests
    "tests/hooks/test_leak_guard.py",
    "scripts/stress_test_voice.py",  # writes its private report there by default
    "tests/unit/test_no_private_doc_citations.py",  # this file
}


def _private_citations() -> list[str]:
    """`file:line` of every line of a tracked text file that names the private folder, outside `_ALLOWED`."""
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, check=True, capture_output=True).stdout
    found: list[str] = []
    for raw in listed.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8")
        path = REPO / rel
        if rel in _ALLOWED or (path.suffix not in _TEXT_SUFFIXES and path.name != ".gitignore"):
            continue
        try:
            data = path.read_bytes()
        except FileNotFoundError:  # deleted in the working tree, not yet committed
            continue
        if _PRIVATE not in data:
            continue
        for lineno, line in enumerate(data.splitlines(), start=1):
            if _PRIVATE in line:
                found.append(f"{rel}:{lineno}")
    return found


@pytest.fixture(scope="module")
def private_citations() -> list[str]:
    """Scanned once in setup: it reads every tracked text file, which belongs to no one test (the call-phase CPU budget, #787)."""
    return _private_citations()


def test_no_tracked_file_cites_a_document_in_the_private_folder(private_citations: list[str]) -> None:
    assert not private_citations, (
        f"{len(private_citations)} lines name a document under docs/superpowers/, which is never committed (#797). Point at "
        f"a committed doc (docs/source-spec/), an issue or the module instead: {private_citations}"
    )


# `docs/audits/` and `docs/guarded-change/` are gitignored and refused by the leak guard too, but some of their files were
# committed anyway and are cited correctly. So a citation under them is checked against what git tracks, not refused.
_PARTLY_PRIVATE = re.compile(rb"docs/(?:audits|guarded-change)/[A-Za-z0-9_./-]+")


def _untracked_citations() -> list[str]:
    """`file:line: path` for every path under the partly-private folders that names no tracked file or folder."""
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, check=True, capture_output=True).stdout
    tracked = {raw.decode("utf-8") for raw in listed.split(b"\0") if raw}
    found: list[str] = []
    for rel in sorted(tracked):
        path = REPO / rel
        if path.suffix not in _TEXT_SUFFIXES:  # no exemptions: the leak guard's files cite these folders by name only
            continue
        try:
            data = path.read_bytes()
        except FileNotFoundError:  # deleted in the working tree, not yet committed
            continue
        if b"docs/audits/" not in data and b"docs/guarded-change/" not in data:
            continue
        for lineno, line in enumerate(data.splitlines(), start=1):
            for match in _PARTLY_PRIVATE.finditer(line):
                cited = match.group(0).decode("utf-8").rstrip(".")
                folder = cited.rstrip("/") + "/"
                if cited in tracked or any(name.startswith(folder) for name in tracked):
                    continue
                found.append(f"{rel}:{lineno}: {cited}")
    return found


@pytest.fixture(scope="module")
def untracked_citations() -> list[str]:
    """Scanned once in setup, for the same reason as `private_citations`."""
    return _untracked_citations()


def test_every_citation_under_audits_or_guarded_change_names_a_tracked_file(untracked_citations: list[str]) -> None:
    assert not untracked_citations, (
        f"{len(untracked_citations)} citations under docs/audits/ or docs/guarded-change/ name a file that is not "
        f"committed (#797). Name the document and say it is private or lost instead: {untracked_citations}"
    )

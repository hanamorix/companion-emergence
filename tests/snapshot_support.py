"""Read and write a committed snapshot file for the tests that warn loudly when something changes (#540).

A snapshot is a JSON object of strings, kept in ``tests/snapshots/``. The tests that use it compare the current text with
the approved text and, on a difference, fail with a message that says how to approve the new text: a one-shot script
mode (``uv run python <test file> --approve``) that rewrites the file, so the change shows up in the diff of the commit
and is reviewed there. There is no environment variable to leave switched on.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_snapshot(path: Path) -> dict[str, str] | None:
    """The approved snapshot, or ``None`` when there is no file yet. A file that is not a JSON object of strings raises."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ValueError(f"{path} is not a JSON object of strings")
    return data


def write_snapshot(path: Path, data: dict[str, str]) -> None:
    """Write ``data`` as sorted, indented utf-8 JSON with ``\\n`` line ends on every platform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


# --- text sections: a snapshot of long texts that stays readable in a git diff ---------------------------------------------

_BEGIN = "@@@ begin "
_END = "@@@ end "
_LINE = "|"


def read_sections(path: Path) -> dict[str, str] | None:
    """The approved texts of a sections file, or ``None`` when there is no file. A malformed file raises ``ValueError``.

    The file is a free header, then sections ``@@@ begin <key>`` ... ``@@@ end <key>`` whose lines each start with ``|``;
    the text is those lines (the ``|`` removed) joined by ``\\n``, so a trailing newline is a last empty ``|`` line.
    """
    if not path.is_file():
        return None
    out: dict[str, str] = {}
    key: str | None = None
    lines: list[str] = []
    raw = path.read_text(encoding="utf-8-sig").split("\n")  # text mode already turns CRLF into LF
    if raw and raw[-1] == "":
        raw.pop()
    for n, line in enumerate(raw, 1):  # not str.splitlines(): that also splits on U+2028 and the like inside a text
        if key is None:
            if line.startswith(_BEGIN):
                key = line[len(_BEGIN) :]
                if key in out:
                    raise ValueError(f"{path}:{n}: duplicate section {key!r}")
                lines = []
            elif line.startswith(_END) or line.startswith(_LINE):
                raise ValueError(f"{path}:{n}: text outside a section")
            continue
        if line == _END + key:
            out[key] = "\n".join(lines)
            key = None
        elif line.startswith(_LINE):
            lines.append(line[len(_LINE) :])
        else:
            raise ValueError(f"{path}:{n}: a line of section {key!r} does not start with {_LINE!r}")
    if key is not None:
        raise ValueError(f"{path}: section {key!r} is not closed")
    return out


def unstorable(texts: dict[str, str]) -> list[str]:
    """Why ``write_sections`` would refuse some of ``texts``: what an editor or git would change (a carriage return, a
    space or tab at the end of a line) or a key it cannot hold. Empty when everything can be stored."""
    problems: list[str] = []
    for key, text in sorted(texts.items()):
        if "\n" in key or "\r" in key:
            problems.append(f"section {key!r}: a newline or carriage return in the key cannot be stored")
        if "\r" in text:
            problems.append(f"section {key!r}: the text holds a carriage return, which git may convert")
        problems += [
            f"section {key!r}: a line ends with a space or tab ({line[-20:]!r}), which editors strip"
            for line in text.split("\n")
            if line != line.rstrip(" \t")
        ]
    return problems


def write_sections(path: Path, texts: dict[str, str], header: str = "") -> None:
    """Write ``texts`` as sections (sorted by key) with ``\\n`` line ends; refuses what :func:`unstorable` names."""
    problems = unstorable(texts)
    if problems:
        raise ValueError("; ".join(problems))
    parts = [header.rstrip("\n") + "\n\n"] if header else []
    for key in sorted(texts):
        text = texts[key]
        parts.append(f"{_BEGIN}{key}\n" + "".join(f"{_LINE}{line}\n" for line in text.split("\n")) + f"{_END}{key}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8", newline="\n")

"""A loud diff when persona-facing wording changes (#540, J4).

The persona's voice template, the approved prompts, the cold-start strings, the name tools' texts and the recall
invitation are wording the owner approves. Persona wording is finalized late, so a test that pins each string byte for
byte breaks on every legitimate edit and says little. This test holds all of them in one committed file,
``tests/snapshots/persona_wording.txt``, compares the current text with it and fails with a diff of each text that
changed (a unified diff plus the words that changed, because the prompts are single long lines). A deliberate edit is
approved with one command, which rewrites the file so the new wording is part of the commit and is reviewed there:

    uv run python tests/unit/test_persona_wording_snapshot.py --approve

The em-dash and LLM-tell checks are separate and stay in ``test_name_extraction_placeholders.py``. The snapshot is a copy
of the current text that a human approved; it is not a clearance of the text for those tells.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / "tests" / "snapshots" / "persona_wording.txt"
APPROVE_COMMAND = "uv run python tests/unit/test_persona_wording_snapshot.py --approve"
HEADER = (
    "# Approved persona-facing wording (#540). Each section is the text exactly as brain/ holds it, every line\n"
    "# prefixed with '|'. Do not edit by hand: change the wording in brain/, run the test and approve with\n"
    "#   uv run python tests/unit/test_persona_wording_snapshot.py --approve\n"
    "# and review the diff of this file in the commit. It is a copy of the current text approved by a human,\n"
    "# not a clearance of the text for em-dashes or other LLM tells (the tests that check those are separate)."
)


def current_wording() -> dict[str, str]:
    """``{key: text}`` of every approved persona-facing string, read from the live code."""
    from brain import prompt_strings
    from brain.chat.prompt import _RECALL_SNIPPET_INVITATION
    from brain.chat.voice import DEFAULT_VOICE_TEMPLATE
    from brain.engines import consolidation
    from brain.felt_time.prompt import render_prompt_context
    from brain.felt_time.state import FeltTimeState
    from brain.initiate.reflection import _TASK_FRAME_TEMPLATE
    from brain.narrative_memory.prompt import render_current_arc_block
    from brain.tools import schemas

    # The module constants are the registry's values (wiring), so one key per prompt is enough.
    assert consolidation._REAPPRAISER_PROMPT == prompt_strings.get("engines.consolidation.reappraiser_prompt")
    assert consolidation._CLASSIFIER_PROMPT == prompt_strings.get("engines.consolidation.classifier_prompt")

    def entry(table: dict) -> str:
        return json.dumps(table, indent=2, sort_keys=True, ensure_ascii=False)

    with tempfile.TemporaryDirectory() as empty_persona:
        arcs_cold_start = render_current_arc_block(Path(empty_persona))
    return {
        "voice.DEFAULT_VOICE_TEMPLATE": DEFAULT_VOICE_TEMPLATE,
        "consolidation.reappraiser_prompt": consolidation._REAPPRAISER_PROMPT,
        "consolidation.classifier_prompt": consolidation._CLASSIFIER_PROMPT,
        "prompt.recall_snippet_invitation": _RECALL_SNIPPET_INVITATION,
        "felt_time.cold_start_block": render_prompt_context(FeltTimeState()),
        "arcs.cold_start_block": arcs_cold_start,
        "initiate.reflection_task_frame": _TASK_FRAME_TEMPLATE,
        "names.add_name.tool_name": schemas.ADD_NAME_TOOL_NAME,
        "names.remove_name.tool_name": schemas.REMOVE_NAME_TOOL_NAME,
        "names.add_name.description": schemas.ADD_NAME_TOOL_DESCRIPTION,
        "names.add_name.param": schemas.ADD_NAME_PARAM_DESCRIPTION,
        "names.remove_name.description": schemas.REMOVE_NAME_TOOL_DESCRIPTION,
        "names.remove_name.param": schemas.REMOVE_NAME_PARAM_DESCRIPTION,
        "names.add_name.rejected_reply": prompt_strings.get("tools.add_name.rejected_reply"),
        "names.add_name.too_long_reply": prompt_strings.get("tools.add_name.too_long_reply"),
        "names.schema.add_name": entry(schemas.SCHEMAS["add_name"]),
        "names.schema.remove_name": entry(schemas.SCHEMAS["remove_name"]),
        "names.schema.add_name.persona_built": entry(schemas.build_schemas("Canary")["add_name"]),
        "names.schema.remove_name.persona_built": entry(schemas.build_schemas("Canary")["remove_name"]),
    }


def _words(a: str, b: str, context: int = 4) -> str:
    """The words that changed, as ``[-old-]{+new+}`` with a few words of context (the prompts are single long lines)."""
    ta, tb = re.findall(r"\s+|\S+", a), re.findall(r"\s+|\S+", b)
    out: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        before = "".join(ta[max(0, i1 - context) : i1])
        after = "".join(ta[i2 : i2 + context])
        out.append(f"...{before}[-{''.join(ta[i1:i2])}-]{{+{''.join(tb[j1:j2])}+}}{after}...")
    return "\n    ".join(out)


def describe_change(key: str, approved: str, current: str) -> str:
    """The diff of one changed text: unified, then word level, and the first differing characters as ``repr``."""
    def lines(text: str) -> list[str]:
        out = text.splitlines(keepends=True)
        if out and not out[-1].endswith("\n"):
            out[-1] += "\n"  # otherwise difflib fuses a last line of the old text with the new one
        return out

    unified = "".join(difflib.unified_diff(lines(approved), lines(current), f"approved {key}", f"current {key}", n=1))
    first = len(os.path.commonprefix([approved, current]))
    where = (
        f"first difference at character {first}: approved {approved[max(0, first - 15) : first + 25]!r}, "
        f"current {current[max(0, first - 15) : first + 25]!r}"
    )
    return f"{key} changed:\n  {where}\n  words: {_words(approved, current)}\n{unified}"


def differences(approved: dict[str, str], current: dict[str, str]) -> list[str]:
    """One block per text that changed, is new or was removed."""
    blocks = [
        describe_change(key, approved[key], current[key])
        for key in sorted(approved.keys() & current.keys())
        if approved[key] != current[key]
    ]
    blocks += [f"{key} is new:\n{current[key]}" for key in sorted(current.keys() - approved.keys())]
    blocks += [f"{key} was removed (it was {approved[key]!r})" for key in sorted(approved.keys() - current.keys())]
    return blocks


def test_persona_wording_matches_the_approved_snapshot() -> None:
    from tests.snapshot_support import read_sections, unstorable

    current = current_wording()
    problems = unstorable(current)
    if problems:
        pytest.fail(
            "this wording cannot be kept in the snapshot, so it cannot be approved as it is (remove the trailing "
            "space or the carriage return from the text):\n  " + "\n  ".join(problems),
            pytrace=False,
        )
    try:
        approved = read_sections(SNAPSHOT)
    except ValueError as exc:
        pytest.fail(f"{exc}. Rewrite it with: {APPROVE_COMMAND}", pytrace=False)
    if approved is None:
        pytest.fail(f"there is no approved snapshot at {SNAPSHOT}. Create it with: {APPROVE_COMMAND}", pytrace=False)
    blocks = differences(approved, current)
    if blocks:
        pytest.fail(
            "persona-facing wording differs from tests/snapshots/persona_wording.txt:\n\n"
            + "\n\n".join(blocks)
            + f"\n\nIf this wording is intended, approve it and commit the changed snapshot: {APPROVE_COMMAND}",
            pytrace=False,
        )


if __name__ == "__main__":
    if sys.argv[1:] == ["--approve"]:
        sys.path.insert(0, str(REPO))
        from tests.snapshot_support import write_sections

        write_sections(SNAPSHOT, current_wording(), header=HEADER)
        print(f"rewrote {SNAPSHOT}: review it with git diff, then commit it")
    else:
        print(f"usage: {APPROVE_COMMAND}")
        raise SystemExit(2)

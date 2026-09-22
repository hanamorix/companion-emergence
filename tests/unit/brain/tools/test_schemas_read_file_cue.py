"""#269: the read_file tool description quotes the shared-file line the engine emits."""

from __future__ import annotations

from brain.tools.schemas import SCHEMAS


def test_read_file_description_quotes_current_shared_file_line() -> None:
    desc = SCHEMAS["read_file"]["description"]
    assert (
        '[the user shared a file "<filename>": <path>. '
        "Open it with your read_file tool to see what it says.]"
    ) in desc
    assert "[the user shared a file: <path>]" not in desc

"""#748: two more bold layouts of a research reply's EMOTIONS line are read.

#709 taught the parser ``EMOTIONS: **{"joy": 7}**``. When the bold opens right after the colon
(``EMOTIONS:**{...}**``) or wraps the whole line (``**EMOTIONS: {...}**``), the line pattern takes the
opening marker and the closing one was left on the value, so the JSON read failed and the session's
feelings read as none. Through ``extract_felt_lines``, which the live research fire calls.
"""
import pytest

from brain.engines.research_session import extract_felt_lines


@pytest.mark.parametrize(
    "line",
    [
        'EMOTIONS:**{"joy": 7}**',
        'EMOTIONS:__{"joy": 7}__',
        '**EMOTIONS: {"joy": 7}**',
        '__EMOTIONS: {"joy": 7}__',
    ],
)
def test_a_bold_emotions_line_in_the_other_layouts_is_read(line):
    assert extract_felt_lines(f"notes\n\n{line}")[1] == {"joy": 7.0}

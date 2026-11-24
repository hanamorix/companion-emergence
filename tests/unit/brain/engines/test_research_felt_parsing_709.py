"""#709: a research reply's felt lines read a bold value, and dropped names leave a trace.

A bold name was handled (``**ENGAGEMENT:** 7``) but a bold value was not, so
``ENGAGEMENT: **7**`` read as no rating. Through ``extract_felt_lines``, which the live research fire calls. (An
EMOTIONS object whose names were all unregistered used to come back ``{}`` here; the names are now passed on and the
memory write checks them against her vocabulary.)
"""
import pytest

from brain.engines.research_session import extract_felt_lines


@pytest.mark.parametrize("line", ["ENGAGEMENT: **7**", "ENGAGEMENT: __7__", "ENGAGEMENT: **7/10**", "ENGAGEMENT: **7**/10"])
def test_a_bold_rating_value_is_read(line):
    assert extract_felt_lines(f"notes\n\n{line}")[2] == 7.0


def test_a_bold_emotions_object_is_read():
    assert extract_felt_lines('notes\n\nEMOTIONS: **{"joy": 7}**')[1] == {"joy": 7.0}


@pytest.mark.parametrize(
    "obj, kept",
    [('{"zorble": 6}', {"zorble": 6.0}), ('{"joy": 7, "zorble": 6, "quux": 2}', {"joy": 7.0, "zorble": 6.0, "quux": 2.0})],
)
def test_names_are_passed_on_for_the_memory_write_to_check(obj, kept):
    """The registered-name filter is gone (#801 section 6): the memory write checks each name against her vocabulary."""
    assert extract_felt_lines(f"notes\n\nEMOTIONS: {obj}")[1] == kept

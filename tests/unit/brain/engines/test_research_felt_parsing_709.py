"""#709: a research reply's felt lines read a bold value, and dropped names leave a trace.

A bold name was handled (``**ENGAGEMENT:** 7``) but a bold value was not, so
``ENGAGEMENT: **7**`` read as no rating. An EMOTIONS object whose names were all
unregistered came back ``{}`` with nothing logged. Through ``extract_felt_lines``,
which the live research fire calls.
"""
import pytest

from brain.engines.research_session import extract_felt_lines


@pytest.mark.parametrize("line", ["ENGAGEMENT: **7**", "ENGAGEMENT: __7__", "ENGAGEMENT: **7/10**", "ENGAGEMENT: **7**/10"])
def test_a_bold_rating_value_is_read(line):
    assert extract_felt_lines(f"notes\n\n{line}")[2] == 7.0


def test_a_bold_emotions_object_is_read():
    assert extract_felt_lines('notes\n\nEMOTIONS: **{"joy": 7}**')[1] == {"joy": 7.0}


@pytest.mark.parametrize("obj, kept", [('{"zorble": 6}', {}), ('{"joy": 7, "zorble": 6, "quux": 2}', {"joy": 7.0})])
def test_unregistered_names_that_are_dropped_are_logged(caplog, obj, kept):
    import logging

    with caplog.at_level(logging.WARNING, logger="brain.engines.research_session"):
        assert extract_felt_lines(f"notes\n\nEMOTIONS: {obj}")[1] == kept
    dropped = [r.getMessage() for r in caplog.records if "not registered" in r.getMessage()]
    assert len(dropped) == 1 and "zorble" in dropped[0], caplog.records

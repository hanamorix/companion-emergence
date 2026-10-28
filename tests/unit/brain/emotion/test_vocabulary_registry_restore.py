"""#540: a test's change to the emotion vocabulary registry must not reach the next test.

``vocabulary.register`` (a persona's extension emotions) and the warn-once set of unregistered emotions are
process-global. Under pytest-xdist the order of tests in a worker is not the order in a file, so a count such as
"the baseline has 26 emotions" is only stable when every test restores both. The autouse fixture in tests/conftest.py
does it; these tests run two ordered tests in an inner session to prove it, with a control arm without the fixture.
"""

from __future__ import annotations

import pytest

_TESTS = '''
from brain.emotion import aggregate, vocabulary
from brain.emotion.vocabulary import Emotion

_REGISTRY_ID = id(vocabulary._REGISTRY)
_WARNED_ID = id(aggregate._warned_unregistered)


def test_a_registers_and_warns():
    vocabulary.register(
        Emotion(name="zz_leak", description="d", category="nell_specific", decay_half_life_days=None, intensity_clamp=10)
    )
    aggregate._warned_unregistered.add("zz_warned")
    assert vocabulary.get("zz_leak") is not None


def test_b_sees_none_of_it_and_the_same_objects():
    assert vocabulary.get("zz_leak") is None
    assert "zz_warned" not in aggregate._warned_unregistered
    assert id(vocabulary._REGISTRY) == _REGISTRY_ID
    assert id(aggregate._warned_unregistered) == _WARNED_ID
'''


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_a_registration_and_a_warn_once_entry_do_not_reach_the_next_test(pytester) -> None:
    pytester.makeconftest("from tests.conftest import _vocabulary_registry_restore  # noqa: F401\n")
    pytester.makepyfile(_TESTS)
    pytester.runpytest("-p", "no:cacheprovider", "-p", "no:randomly").assert_outcomes(passed=2)


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_without_the_fixture_the_registration_does_reach_the_next_test(pytester) -> None:
    """Control arm: the inner session's own fixture is what makes the test above pass."""
    pytester.makepyfile(_TESTS)
    # The outer session's autouse fixture restores the real registry after THIS test, whatever the inner run left.
    pytester.runpytest("-p", "no:cacheprovider", "-p", "no:randomly").assert_outcomes(passed=1, failed=1)

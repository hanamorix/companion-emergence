"""Shared fixtures for brain.initiate unit tests."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_events_publisher():
    """Reset the module-level events._publisher singleton between tests.

    `brain.bridge.events._publisher` is a process-global set by the bridge
    lifespan and by tests that install a live publisher to assert on emitted
    events. The bridge conftest has this autouse reset but is scoped to
    tests/bridge/; the initiate unit tests (test_review.py especially) set a
    live publisher and previously cleared it with inline try/finally pairs.
    This autouse fixture makes the reset uniform and removes that boilerplate,
    closing the cross-test pollution gap.
    """
    from brain.bridge import events

    events.set_publisher(None)
    yield
    events.set_publisher(None)

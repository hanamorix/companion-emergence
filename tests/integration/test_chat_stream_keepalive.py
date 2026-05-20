"""WS /stream emits reply_chunk frames during long generation.

Integration harness for the streaming keepalive path. The full test requires a
'fake_streaming' provider variant that emits deltas over ~90s — longer than
the 60s idle window used in the production idle-timer. This infrastructure
(bridge_subprocess with injectable provider) does not yet exist; the test body
is a TODO stub so the file participates in the test collection path.

Phase 5.5 shipped the _StreamingProxy that intercepts chat() and forwards
TextDelta.text to the WS in real time. The unit-level coverage lives in
tests/bridge/test_provider_stream_json.py and tests/bridge/test_endpoints.py
(test_stream_round_trip validates reply_chunk frames arrive and are reassembled
to the full content).
"""

import pytest


@pytest.mark.skip(
    reason="integration harness (bridge_subprocess + fake_streaming provider) "
    "not yet built — covered at unit level by test_stream_round_trip"
)
def test_reply_chunks_keep_idle_alive(tmp_path):
    """A 90-second generation streams enough deltas to outlast the 60s idle window."""
    pass

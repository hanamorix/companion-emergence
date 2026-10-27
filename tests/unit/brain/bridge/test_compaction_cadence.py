"""Tests for the compaction tick's signature.

  - _run_compaction_tick is importable and has the expected signature.
  (The persisted cadence is held by test_central_cadence.py and test_persisted_cadence.py; run_folded's default
  compaction interval is pinned in test_calibration_cadence.py beside the calibration one.)
"""
import inspect


def test_run_compaction_tick_importable_and_callable():
    from brain.bridge.supervisor import _run_compaction_tick

    sig = inspect.signature(_run_compaction_tick)
    params = list(sig.parameters)
    # Positional contract: (persona_dir, provider). The idle-gate added an optional
    # keyword-only ``is_session_busy`` (defaulted, so the tick stays importable and
    # callable without it — startup catch-up passes None).
    # INC-10 (ram-spike-fix) adds keyword-only `should_pause` (between-items
    # pause hook, default None => never pauses).
    assert params == ["persona_dir", "provider", "is_session_busy", "should_pause"], (
        f"expected (persona_dir, provider, *, is_session_busy, should_pause), got {params}"
    )
    busy = sig.parameters["is_session_busy"]
    assert busy.kind is inspect.Parameter.KEYWORD_ONLY
    assert busy.default is None

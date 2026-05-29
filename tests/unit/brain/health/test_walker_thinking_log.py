def test_thinking_log_in_optional_files():
    from brain.health.walker import _OPTIONAL_TEXT_FILES
    assert "thinking_log.jsonl" in _OPTIONAL_TEXT_FILES

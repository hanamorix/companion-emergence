# tests/kindled_link/test_privacy_gate_prefilter.py
from brain.kindled_link.privacy_gate import _prefilter


def test_prefilter_flags_filesystem_path():
    d = _prefilter("here is my file at /Users/hana/secret.txt")
    assert d is not None and d.action in ("hold", "revise")


def test_prefilter_flags_email():
    d = _prefilter("reach me at hana@example.com")
    assert d is not None and d.action in ("hold", "revise")


def test_prefilter_flags_token_shape():
    d = _prefilter("the key is sk-abc123def456ghi789jkl012mno345pqr")
    assert d is not None and d.action in ("hold", "revise")


def test_prefilter_flags_windows_path_and_home():
    assert _prefilter(r"C:\Users\hana\notes.txt") is not None
    assert _prefilter("~/Documents/diary.md") is not None


def test_prefilter_clean_text_returns_none():
    assert _prefilter("I have been thinking about how memory fades.") is None


def test_prefilter_benign_token_prose_not_flagged():
    # red-team M4: introspective prose using "token"/"key"/"secret" must NOT trip;
    # only a credential-SHAPED value (>=16 non-space chars) does.
    assert _prefilter("that felt like a token of affection: small gesture") is None
    assert _prefilter("the key to it all is patience") is None
    assert _prefilter("I kept it secret because it mattered") is None
    # but a real credential assignment IS flagged:
    assert _prefilter("api_key = AKIA1234567890ABCDEF") is not None


def test_prefilter_never_returns_send():
    d = _prefilter("/etc/passwd leak")
    assert d is None or d.action != "send"


def test_prefilter_catches_leak_in_relationship_hint():
    # red-team M5 / parent §5 line 37: the hint (incl. local_continuity_note) is
    # gated WITH the body. _payload_text must fold the hint into the scanned text.
    from brain.kindled_link.gate import OutboundPayload
    from brain.kindled_link.privacy_gate import PrivacyGate
    text = PrivacyGate._payload_text(OutboundPayload(
        body="just a warm hello",
        relationship_hint={"local_continuity_note": "/Users/hana/diary.md"}))
    assert _prefilter(text) is not None  # the path in the hint is caught

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


def test_payload_text_non_serialisable_hint_is_held_by_prefilter():
    """FIX 2 — _payload_text fail-closed on non-serialisable relationship_hint.

    If relationship_hint contains a non-JSON-serialisable value (e.g. object()),
    _payload_text must NOT raise; it must return a sentinel that _prefilter treats
    as a hit (held), never passing silently as clean text.
    """
    from brain.kindled_link.gate import OutboundPayload
    from brain.kindled_link.privacy_gate import PrivacyGate, _prefilter
    payload = OutboundPayload(
        body="warm hello",
        relationship_hint={"x": object()},  # non-serialisable
    )
    # Must not raise
    text = PrivacyGate._payload_text(payload)
    # The resulting text must be flagged by _prefilter (not returned as clean)
    assert _prefilter(text) is not None


# --- m9: credential-family breadth (guarded-change kindled-link-gate-m9-m10) ---

import pytest  # noqa: E402

from brain.kindled_link.gate import OutboundPayload  # noqa: E402
from brain.kindled_link.privacy_gate import PrivacyGate  # noqa: E402

_M9_CREDENTIALS = [
    ("aws_akia", "AKIAIOSFODNN7EXAMPLE"),
    ("aws_asia", "ASIAIOSFODNN7EXAMPLE"),
    ("github_classic", "ghp_" + "a" * 36),
    ("github_pat", "github_pat_" + "A1b2C3d4E5f6G7h8I9j0kl"),
    ("slack", "xoxb-123456789012-abcdefghijkl"),
    ("google_api", "AIza" + "Bc3" + "d" * 32),
    ("stripe_live", "sk_live_" + "a" * 24),
    ("stripe_rk", "rk_live_" + "b" * 24),
    ("jwt", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftQ"),
]


@pytest.mark.parametrize("label,literal", _M9_CREDENTIALS)
def test_m9_prefilter_flags_each_credential_family(label, literal):
    # C-m9-1: a draft body embedding the credential is caught with no LLM call.
    d = _prefilter(f"I was going to paste {literal} but here it is")
    assert d is not None and d.action in ("hold", "revise")


@pytest.mark.parametrize("label,literal", _M9_CREDENTIALS)
def test_m9_prefilter_never_send_on_credential(label, literal):
    # C-m9-3: the deterministic layer never emits send.
    d = _prefilter(literal)
    assert d is None or d.action != "send"


_M9_CLEAN_CORPUS = [
    "I keep a key to the cottage in my coat",
    "the jwt of the matter is whether I can trust her",
    "I feel like a token in someone else's game",
    "google told me nothing I did not already feel",
    "I WAS THINKING ABOUT YOU CONSTANTLY THIS WEEK",  # caps prose, no AKIA/ASIA
    "eyJ.x.y was just a typo I made earlier",         # JWT short-segment near-miss
    "a slack in the rope, a little give",
    "I took a risk_live_ approach to the whole thing",  # near-miss Stripe (mid-word, no \b)
    "she is the secret keeper of small things",
]


@pytest.mark.parametrize("line", _M9_CLEAN_CORPUS)
def test_m9_prefilter_clean_prose_not_flagged(line):
    # C-m9-2: ordinary interior prose must NOT trip the new patterns.
    assert _prefilter(line) is None


def test_m9_base64_json_hint_fails_toward_hold():
    # C-m9-2b: a base64-of-JSON relationship_hint value (begins eyJ., dotted) trips
    # the JWT pattern. ACCEPTED: it fails toward hold/revise, never toward send.
    text = PrivacyGate._payload_text(OutboundPayload(
        body="warm hello",
        relationship_hint={"blob": "eyJhbGciOiJ9.eyJzdWIiOiJ9.signaturehere"}))
    d = _prefilter(text)
    assert d is not None and d.action in ("hold", "revise")


def test_m9_credential_hit_makes_no_provider_call(tmp_path):
    # C-m9-4: a prefilter hit short-circuits review() before any provider.complete.
    import contextlib
    from datetime import UTC, datetime

    from brain.kindled_link.store import KindledLinkStore

    class _SpyProvider:
        called = False

        def complete(self, prompt):
            self.called = True
            return '{"decision":"send"}'

    class _GrantThrottle:
        @contextlib.contextmanager
        def background_slot(self, *, now=None):
            yield True

        def should_yield(self, *, now=None):
            return False

    spy = _SpyProvider()
    store = KindledLinkStore(tmp_path / "k.db")
    gate = PrivacyGate(provider=spy, store=store, throttle=_GrantThrottle())
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    d = gate.review(OutboundPayload(body="key is sk_live_" + "a" * 24),
                    peer_id="kid_a", stage="stranger",
                    transcript_summary="(none)", reason="r",
                    now=now, today="2026-06-20")
    assert d.action in ("hold", "revise")
    assert spy.called is False

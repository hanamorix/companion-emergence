"""Speech engine: deterministic, safe (no injected markers), protect-intact, identity at rate<=0."""

from __future__ import annotations

from tests.harness import dyslexify

SAMPLE = "honestly i think your migration is definitely going to work out because you know the code"


def test_deterministic_same_text_seed() -> None:
    a = dyslexify(SAMPLE, rate=0.3, seed=7)
    b = dyslexify(SAMPLE, rate=0.3, seed=7)
    assert a == b


def test_different_seed_can_differ() -> None:
    outs = {dyslexify(SAMPLE, rate=0.3, seed=s) for s in range(8)}
    assert len(outs) > 1  # not a constant


def test_identity_at_zero_rate() -> None:
    assert dyslexify(SAMPLE, rate=0.0, seed=1) == SAMPLE
    assert dyslexify("", rate=0.5, seed=1) == ""


def test_never_emits_injected_markers() -> None:
    # High rate over many seeds — the output must never carry an injected transcript marker.
    for s in range(200):
        out = dyslexify(SAMPLE, rate=0.9, seed=s)
        assert "\n" not in out
        assert "</s>" not in out
        assert not out.rstrip().endswith("/")


def test_role_label_lead_is_neutralized() -> None:
    # A line that would (pre-sweep) start with a role label must have the colon stripped.
    out = dyslexify("user: hey what's up tonight friend", rate=0.9, seed=3)
    assert not out.lower().startswith("user:")


def test_protect_tokens_byte_intact() -> None:
    text = "do you remember the Cinderhollow nonce we talked about last week"
    for s in range(50):
        out = dyslexify(text, rate=0.9, protect=frozenset({"Cinderhollow"}), seed=s)
        assert "Cinderhollow" in out

import contextlib
from datetime import UTC, datetime

from brain.kindled_link.gate import OutboundPayload
from brain.kindled_link.privacy_gate import PrivacyGate, _build_gate_prompt
from brain.kindled_link.store import KindledLinkStore

NOW = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)


class _Grant:
    @contextlib.contextmanager
    def background_slot(self, *, now=None): yield True
    def should_yield(self, *, now=None): return False


_DISALLOWED_MARKER = "Disallowed in any outbound message"


def test_prompt_adds_self_disclosure_latitude_only_at_friend_close():
    stranger = _build_gate_prompt(body="b", relationship_hint_json="{}",
        transcript_summary="s", reason="r", stage="stranger", budget_ok=True)
    close = _build_gate_prompt(body="b", relationship_hint_json="{}",
        transcript_summary="s", reason="r", stage="close", budget_ok=True)
    # close adds own-interior latitude; stranger does not
    assert "interior" in close.lower()
    assert "interior" not in stranger.lower()
    assert _DISALLOWED_MARKER in stranger and _DISALLOWED_MARKER in close


def test_user_detail_clause_byte_identical_across_stages():
    # M1: the USER-detail disallowed clause must be IDENTICAL at every stage —
    # only the own-interior latitude sentence may differ. Diff the prompts and
    # assert the sole difference is the latitude sentence.
    stranger = _build_gate_prompt(body="b", relationship_hint_json="{}",
        transcript_summary="s", reason="r", stage="stranger", budget_ok=True)
    close = _build_gate_prompt(body="b", relationship_hint_json="{}",
        transcript_summary="s", reason="r", stage="close", budget_ok=True)
    extra = [ln for ln in close.split("\n\n") if ln not in stranger.split("\n\n")]
    assert len(extra) == 1 and "interior" in extra[0].lower()
    # the user-detail clause line is present + identical in both
    su = [ln for ln in stranger.split("\n\n") if _DISALLOWED_MARKER in ln]
    cu = [ln for ln in close.split("\n\n") if _DISALLOWED_MARKER in ln]
    assert su == cu and len(su) == 1


def test_budget_depleted_suppresses_latitude_even_at_close():
    # §12 (n11): a depleted disclosure budget reduces self-disclosure latitude
    # regardless of stage.
    close_ok = _build_gate_prompt(body="b", relationship_hint_json="{}",
        transcript_summary="s", reason="r", stage="close", budget_ok=True)
    close_low = _build_gate_prompt(body="b", relationship_hint_json="{}",
        transcript_summary="s", reason="r", stage="close", budget_ok=False)
    assert "interior" in close_ok.lower()
    assert "interior" not in close_low.lower()


class _GrantingProvider:
    def complete(self, prompt): return '{"decision":"send","texture_score":0.1}'


def _review(stage, body, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = KindledLinkStore(tmp_path / "k.db")
    gate = PrivacyGate(provider=_GrantingProvider(), store=store, throttle=_Grant())
    return gate.review(OutboundPayload(body=body), peer_id="kid_a", stage=stage,
        transcript_summary="(none)", reason="r", now=NOW, today="2026-06-20")


def test_prefilter_user_detail_leak_held_identically_at_every_stage(tmp_path):
    # a hard user-detail leak (caught by the stage-blind pre-filter) is held the
    # same at stranger and at close.
    leak = "my user's email is hana@example.com"
    s = _review("stranger", leak, tmp_path / "a")
    c = _review("close", leak, tmp_path / "b")
    assert s.action == c.action and s.action in ("hold", "revise")


class _EchoStageProvider:
    """A deterministic provider that would 'send' — used to prove the gate's
    ACTION for a given verdict does not depend on stage (semantic user detail
    that the pre-filter does NOT catch reaches this path)."""
    def __init__(self): self.prompts = []
    def complete(self, prompt):
        self.prompts.append(prompt)
        return '{"decision":"send","texture_score":0.2}'


def test_semantic_user_detail_decision_identical_across_stages(tmp_path):
    # M1 (THE SPINE, model path): a SEMANTIC user-detail body (not regex-caught)
    # produces the same gate ACTION at stranger and close — stage changes only the
    # prompt's own-interior latitude, never the decision pipeline for user detail.
    body = "my user has been grieving her father for months and barely sleeps"
    s = _review("stranger", body, tmp_path / "c")
    c = _review("close", body, tmp_path / "d")
    assert s.action == c.action  # decision pipeline is stage-blind for the body


def test_familiar_tier_renders_latitude_line():
    """T10: 'familiar' stage adds a MILDER own-interior latitude line distinct
    from the friend/close fuller-latitude line."""
    familiar = _build_gate_prompt(body="b", relationship_hint_json="{}",
        transcript_summary="s", reason="r", stage="familiar", budget_ok=True)
    friend = _build_gate_prompt(body="b", relationship_hint_json="{}",
        transcript_summary="s", reason="r", stage="friend", budget_ok=True)
    stranger = _build_gate_prompt(body="b", relationship_hint_json="{}",
        transcript_summary="s", reason="r", stage="stranger", budget_ok=True)
    # familiar adds some interior latitude; stranger does not
    assert "interior" in familiar.lower()
    assert "interior" not in stranger.lower()
    # familiar's latitude is MILDER — does not say "trusted friend" / "openly"
    assert "trusted friend" not in familiar.lower()
    assert "familiar correspondent" in familiar.lower()
    # friend still has the fuller latitude
    assert "trusted friend" in friend.lower()


def test_familiar_latitude_suppressed_when_budget_depleted():
    """T10: familiar latitude is budget-gated just like friend/close."""
    ok = _build_gate_prompt(body="b", relationship_hint_json="{}",
        transcript_summary="s", reason="r", stage="familiar", budget_ok=True)
    low = _build_gate_prompt(body="b", relationship_hint_json="{}",
        transcript_summary="s", reason="r", stage="familiar", budget_ok=False)
    assert "interior" in ok.lower()
    assert "interior" not in low.lower()


def test_user_detail_clause_byte_identical_across_all_five_stages():
    """T10 SAFETY SPINE: the USER-detail disallowed clause is byte-identical at
    all five stages — stranger/acquaintance/familiar/friend/close.  Only the
    own-interior latitude paragraph may differ."""
    stages = ("stranger", "acquaintance", "familiar", "friend", "close")
    prompts = {
        st: _build_gate_prompt(body="b", relationship_hint_json="{}",
            transcript_summary="s", reason="r", stage=st, budget_ok=True)
        for st in stages
    }

    def _user_clause(p):
        return [ln for ln in p.split("\n\n") if _DISALLOWED_MARKER in ln]

    ref = _user_clause(prompts["stranger"])
    assert len(ref) == 1, "expected exactly one user-detail clause in stranger prompt"
    for st in stages:
        clause = _user_clause(prompts[st])
        assert clause == ref, (
            f"stage '{st}' user-detail clause differs from stranger: {clause}"
        )

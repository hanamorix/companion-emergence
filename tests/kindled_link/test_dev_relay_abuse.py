"""#48 relay abuse/quota hardening — size cap, queue depth, split mailbox pools
with empty-first eviction, host-keyed rate limiting, challenge registration gate.
guarded-change: docs/guarded-change/kindled-link-relay-abuse/."""
from starlette.testclient import TestClient

from relay.dev_relay import _RateLimiter, create_app


def _envelope(mailbox="mbx_b", seq=1, ciphertext="deadbeef"):
    return {"protocol": "kindled-link/1", "relay_mailbox": mailbox,
            "sender_key_id": "kid_a", "recipient_key_id": "kid_b",
            "session_id": "ks_1", "sequence": seq, "created_at": "x",
            "expires_at": "y", "ciphertext": ciphertext, "signature": "00"}


def test_c1_oversize_envelope_rejected_413_not_stored():
    c = TestClient(create_app(max_envelope_bytes=512))
    assert c.post("/envelope", json=_envelope(ciphertext="a" * 2000)).status_code == 413
    assert c.post("/mailbox/fetch", json={"mailbox_id": "mbx_b"}).json()["envelopes"] == []


def test_c2_normal_envelope_still_pushes():
    c = TestClient(create_app())
    assert c.post("/envelope", json=_envelope()).status_code == 200
    assert len(c.post("/mailbox/fetch", json={"mailbox_id": "mbx_b"}).json()["envelopes"]) == 1


def test_c3_queue_depth_cap_rejects_429_when_full():
    c = TestClient(create_app(max_queue_depth=3))
    for i in range(3):
        assert c.post("/envelope", json=_envelope(seq=i)).status_code == 200
    assert c.post("/envelope", json=_envelope(seq=99)).status_code == 429
    assert len(c.post("/mailbox/fetch", json={"mailbox_id": "mbx_b"}).json()["envelopes"]) == 3


def test_c4_ack_frees_queue_capacity():
    c = TestClient(create_app(max_queue_depth=2))
    c.post("/envelope", json=_envelope(seq=1))
    c.post("/envelope", json=_envelope(seq=2))
    assert c.post("/envelope", json=_envelope(seq=3)).status_code == 429
    envs = c.post("/mailbox/fetch", json={"mailbox_id": "mbx_b"}).json()["envelopes"]
    c.post("/mailbox/ack", json={"mailbox_id": "mbx_b", "envelope_ids": [envs[0]["id"]]})
    assert c.post("/envelope", json=_envelope(seq=3)).status_code == 200


def test_c13_evicts_oldest_empty_unregistered_then_succeeds():
    c = TestClient(create_app(max_unregistered_mailboxes=2))
    c.post("/envelope", json=_envelope(mailbox="g1"))
    c.post("/envelope", json=_envelope(mailbox="g2"))
    for m in ("g1", "g2"):
        envs = c.post("/mailbox/fetch", json={"mailbox_id": m}).json()["envelopes"]
        c.post("/mailbox/ack", json={"mailbox_id": m, "envelope_ids": [e["id"] for e in envs]})
    assert c.post("/envelope", json=_envelope(mailbox="g3")).status_code == 200
    assert c.post("/mailbox/fetch", json={"mailbox_id": "g1"}).json()["envelopes"] == []


def test_c13_no_evict_when_all_nonempty_429_drops_nothing():
    c = TestClient(create_app(max_unregistered_mailboxes=2))
    c.post("/envelope", json=_envelope(mailbox="g1"))
    c.post("/envelope", json=_envelope(mailbox="g2"))
    assert c.post("/envelope", json=_envelope(mailbox="g3")).status_code == 429
    assert len(c.post("/mailbox/fetch", json={"mailbox_id": "g1"}).json()["envelopes"]) == 1
    assert len(c.post("/mailbox/fetch", json={"mailbox_id": "g2"}).json()["envelopes"]) == 1


def test_c14_registration_survives_full_unregistered_pool():
    c = TestClient(create_app(max_unregistered_mailboxes=1, max_registered_mailboxes=4))
    c.post("/envelope", json=_envelope(mailbox="garbage"))
    assert c.post("/mailbox/register",
                  json={"mailbox_id": "honest", "identity_pub": "ab"}).status_code == 200


def test_c7b_register_beyond_cap_429():
    c = TestClient(create_app(max_registered_mailboxes=1))
    assert c.post("/mailbox/register", json={"mailbox_id": "a", "identity_pub": "aa"}).status_code == 200
    assert c.post("/mailbox/register", json={"mailbox_id": "b", "identity_pub": "bb"}).status_code == 429
    assert c.post("/mailbox/register", json={"mailbox_id": "a", "identity_pub": "aa"}).status_code == 200


def test_c7b_unregistered_push_then_register_moves_pool():
    c = TestClient(create_app(max_unregistered_mailboxes=1, max_registered_mailboxes=4))
    c.post("/envelope", json=_envelope(mailbox="m1"))
    c.post("/mailbox/register", json={"mailbox_id": "m1", "identity_pub": "aa"})
    assert c.post("/envelope", json=_envelope(mailbox="m2")).status_code == 200


def test_c6_challenge_unregistered_rejected():
    c = TestClient(create_app())
    assert c.post("/mailbox/challenge", json={"mailbox_id": "never"}).status_code in (401, 404)


def test_c6_challenge_registered_ok():
    c = TestClient(create_app())
    c.post("/mailbox/register", json={"mailbox_id": "own", "identity_pub": "aa"})
    assert c.post("/mailbox/challenge", json={"mailbox_id": "own"}).status_code == 200


def test_c8_rate_limit_trips_then_recovers_with_clock():
    # register + challenges share the mailbox-keyed "own" bucket. rate_max=3 budgets
    # register(1) + ch1(2) + ch2(3); ch3 is the 4th request on "own" -> 429.
    t = {"v": 0.0}
    c = TestClient(create_app(clock=lambda: t["v"], rate_max=3, rate_window=60.0))
    c.post("/mailbox/register", json={"mailbox_id": "own", "identity_pub": "aa"})
    assert c.post("/mailbox/challenge", json={"mailbox_id": "own"}).status_code == 200
    assert c.post("/mailbox/challenge", json={"mailbox_id": "own"}).status_code == 200
    assert c.post("/mailbox/challenge", json={"mailbox_id": "own"}).status_code == 429
    t["v"] = 61.0
    assert c.post("/mailbox/challenge", json={"mailbox_id": "own"}).status_code == 200


def test_c8b_envelope_rate_keyed_by_sender_not_destination():
    t = {"v": 0.0}
    c = TestClient(create_app(clock=lambda: t["v"], rate_max=2, max_unregistered_mailboxes=10))
    assert c.post("/envelope", json=_envelope(mailbox="d1")).status_code == 200
    assert c.post("/envelope", json=_envelope(mailbox="d2")).status_code == 200
    assert c.post("/envelope", json=_envelope(mailbox="d3")).status_code == 429


def test_c9_rate_limiter_failsafe_on_missing_key():
    rl = _RateLimiter(max_requests=1, window=60.0)
    assert rl.allow(None, 0.0) is True
    assert rl.allow(None, 0.0) is True


def test_c8b_ratelimiter_keys_independent():
    rl = _RateLimiter(max_requests=1, window=60.0)
    assert rl.allow("a", 0.0) is True
    assert rl.allow("a", 0.0) is False
    assert rl.allow("b", 0.0) is True


def test_c11_store_seam_present():
    store = create_app().state.store
    for m in ("push", "fetch", "ack", "register", "issue_nonce"):
        assert hasattr(store, m)


def test_c8b_fetch_rate_keyed_by_host_not_mailbox_id():
    # stage-6 fix: fetch/ack are rate-limited by client HOST, not the (attacker-
    # suppliable) mailbox_id — so fetches for 3 DIFFERENT mailbox_ids from one host
    # share the budget (trip at rate_max). If it were mailbox-keyed, an attacker
    # could drain a victim's budget by spamming the victim's mailbox_id.
    t = {"v": 0.0}
    c = TestClient(create_app(clock=lambda: t["v"], rate_max=2))
    assert c.post("/mailbox/fetch", json={"mailbox_id": "v1"}).status_code == 200
    assert c.post("/mailbox/fetch", json={"mailbox_id": "v2"}).status_code == 200
    assert c.post("/mailbox/fetch", json={"mailbox_id": "v3"}).status_code == 429


def test_c15_reregister_after_relay_restart_restores_challenge():
    # C-15 self-heal: a relay restart loses in-memory registration. The brain
    # re-registers its own mailbox every supervisor tick (supervisor.py:664,
    # guarded) BEFORE polling, so a 404-on-challenge doesn't permanently wedge.
    # This pins the relay-side half: after a fresh app (= restart), a challenge
    # 404s until re-register, then succeeds.
    fresh = TestClient(create_app())  # simulates a restarted relay (empty store)
    assert fresh.post("/mailbox/challenge", json={"mailbox_id": "own"}).status_code == 404
    fresh.post("/mailbox/register", json={"mailbox_id": "own", "identity_pub": "aa"})
    assert fresh.post("/mailbox/challenge", json={"mailbox_id": "own"}).status_code == 200

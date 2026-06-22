"""Supervisor kindled-link tick wiring (Phase 7a T7).

Organ DoD: the tick fires through the live supervisor path, fail-isolated,
OFF BY DEFAULT and SKIPPED when kindled_relay_url is None.
"""
from unittest.mock import MagicMock, patch

from brain.bridge import supervisor
from brain.persona_config import PersonaConfig

# ---------------------------------------------------------------------------
# Test 1 — early-return when kindled_relay_url is None
# ---------------------------------------------------------------------------

def test_maybe_run_kindled_link_tick_skips_when_no_relay(tmp_path, monkeypatch):
    """_maybe_run_kindled_link_tick returns without calling run_kindled_link_tick
    when kindled_relay_url is None (feature not wired)."""
    # Write a config with no relay url
    cfg_path = tmp_path / "persona_config.json"
    PersonaConfig(kindled_link_enabled=True, kindled_relay_url=None).save(cfg_path)

    tick_calls = []

    with patch("brain.kindled_link.tick.run_kindled_link_tick", side_effect=lambda *a, **kw: tick_calls.append(1)):
        supervisor._maybe_run_kindled_link_tick(tmp_path, provider=MagicMock())

    assert tick_calls == [], "run_kindled_link_tick must NOT be called when kindled_relay_url is None"


# ---------------------------------------------------------------------------
# Test 2 — relay configured: deps constructed + tick called once
# ---------------------------------------------------------------------------

def test_maybe_run_kindled_link_tick_calls_tick_when_relay_configured(tmp_path):
    """With kindled_relay_url set, _maybe_run_kindled_link_tick constructs deps
    and calls run_kindled_link_tick exactly once.  httpx and RelayClient.register
    are patched so no real network is needed."""
    cfg_path = tmp_path / "persona_config.json"
    PersonaConfig(
        kindled_link_enabled=True,
        kindled_relay_url="https://relay.example.com",
    ).save(cfg_path)

    tick_calls = []

    def _fake_tick(persona_dir, *, store, identity, relay_client, provider, config, now, **kw):
        tick_calls.append({
            "store": store,
            "identity": identity,
            "relay_client": relay_client,
            "config": config,
            "now": now,
        })

    with (
        patch("brain.kindled_link.tick.run_kindled_link_tick", side_effect=_fake_tick),
        patch("brain.kindled_link.relay_client.RelayClient.register"),
        patch("httpx.Client") as mock_http_cls,
    ):
        mock_http_cls.return_value.close = MagicMock()
        mock_http_cls.return_value.post = MagicMock(return_value=MagicMock(status_code=200))

        supervisor._maybe_run_kindled_link_tick(tmp_path, provider=MagicMock())

    assert len(tick_calls) == 1, f"run_kindled_link_tick must be called exactly once, got {len(tick_calls)}"
    call = tick_calls[0]
    assert call["store"] is not None
    assert call["identity"] is not None
    assert call["relay_client"] is not None
    assert call["config"].kindled_relay_url == "https://relay.example.com"
    assert call["now"] is not None


# ---------------------------------------------------------------------------
# Test 3 — relay register() error is caught; block does not raise
# ---------------------------------------------------------------------------

def test_maybe_run_kindled_link_tick_register_error_is_fault_isolated(tmp_path):
    """If relay.register() raises, _maybe_run_kindled_link_tick must NOT propagate
    the exception — it logs a warning and skips the tick this round."""
    cfg_path = tmp_path / "persona_config.json"
    PersonaConfig(
        kindled_link_enabled=True,
        kindled_relay_url="https://relay.example.com",
    ).save(cfg_path)

    tick_calls = []

    with (
        patch("brain.kindled_link.tick.run_kindled_link_tick",
              side_effect=lambda *a, **kw: tick_calls.append(1)),
        patch("brain.kindled_link.relay_client.RelayClient.register",
              side_effect=RuntimeError("relay down")),
        patch("httpx.Client") as mock_http_cls,
    ):
        mock_http_cls.return_value.close = MagicMock()

        # Must not raise
        supervisor._maybe_run_kindled_link_tick(tmp_path, provider=MagicMock())

    # Tick was skipped this round because register raised
    assert tick_calls == [], "run_kindled_link_tick must be skipped when register() raises"

"""Phase 6 conformance and security checks for kindled-link subsystem.

Tests the static and behavioural criteria from spec §8.1:
  1. No /kindled-link/ route accepts peer message text.
  2. holds_status projection never selects payload_json.
  3. Phase 6 adds no autonomous-send path.
  4. All POST /kindled-link/ routes are auth-gated.
"""

import pathlib

import pytest
from fastapi.testclient import TestClient

_ROOT = pathlib.Path(__file__).resolve().parents[2]  # tests/bridge -> tests -> companion-emergence


def test_no_kindled_link_route_accepts_peer_message_text():
    """Spec §8.1 criterion 1 (parent §15): spectators cannot type; no /kindled-link/
    route takes peer body for sending. Assert no 'message'/'compose'/'send' POST
    body key appears in the kindled-link route block."""
    src = (_ROOT / "brain" / "bridge" / "server.py").read_text(encoding="utf-8")
    # Extract the kindled-link route block (from the first /kindled-link comment
    # to the end of that function group).
    import re

    block = "".join(re.findall(r"/kindled-link[^\n]*\n(?:.*\n){0,50}", src))
    for forbidden in (
        "payload.get(\"message\")",
        "payload.get('message')",
        "compose",
        "send_to_peer",
    ):
        assert forbidden not in block, f"a /kindled-link/ route references {forbidden!r}"


def test_views_holds_never_selects_payload_json():
    """Spec §8.1 criterion 2: the holds projection SQL must not select
    payload_json — the load-bearing safety spine against draft-body leaks."""
    src = (_ROOT / "brain" / "kindled_link" / "views.py").read_text(encoding="utf-8")
    # Extract the holds_status function body, excluding the docstring.
    import re

    match = re.search(
        r"def holds_status\([^)]*\)[^:]*:(.*?)(?=\ndef |\Z)", src, re.DOTALL
    )
    assert match, "holds_status function not found"
    func_body = match.group(1)
    # Skip docstring to check only the actual code.
    docstring_end = func_body.find('"""')
    if docstring_end >= 0:
        docstring_end = func_body.find('"""', docstring_end + 3) + 3
        holds = func_body[docstring_end:]
    else:
        holds = func_body
    assert (
        "payload_json" not in holds
    ), "holds_status references payload_json (body-leak risk)"


def test_phase6_adds_no_autonomous_send_path():
    """Spec §8.1 criterion 3: views.py + the kindled-link endpoints must not call
    the session_engine send path (no process_outbound or send_fn)."""
    views = (_ROOT / "brain" / "kindled_link" / "views.py").read_text(encoding="utf-8")
    assert (
        "process_outbound" not in views and "send_fn" not in views
    ), "Phase 6 added autonomous-send (process_outbound/send_fn)"


# ── Auth gating for POST routes ───────────────────────────────────────────────


def _make_app(persona_dir, auth_token="secret-token"):
    """Build a test app with auth_token configured."""
    from brain.bridge.server import build_app

    return build_app(
        persona_dir=persona_dir, client_origin="tests", auth_token=auth_token
    )


_POST_ROUTES = [
    ("POST", "/kindled-link/invite"),
    ("POST", "/kindled-link/invite/accept"),
    ("POST", "/kindled-link/peers/kid_a/consent"),
]


@pytest.mark.parametrize("method,path", _POST_ROUTES)
def test_post_kindled_routes_reject_missing_auth(tmp_path, method, path):
    """Spec §8.1 criterion 4 / B2: All /kindled-link/ POST routes must be
    auth-gated (required bearer token). Request without auth → 401 or 403."""
    app = _make_app(tmp_path)
    client = TestClient(app)
    # Payload required by each endpoint (minimal valid shape).
    payload = {}
    if path == "/kindled-link/invite":
        payload = {"relay_url": "https://relay.example.com"}
    elif path == "/kindled-link/invite/accept":
        payload = {"invite": "dummy-invite-packet"}
    elif "/consent" in path:
        payload = {"action": "pause"}

    r = client.request(method, path, json=payload)
    assert r.status_code in (
        401,
        403,
    ), f"{method} {path} not auth-gated (got {r.status_code})"

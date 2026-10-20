"""#475: every loopback call to the persona's own bridge goes DIRECT, never through an HTTP proxy.

`urllib.request.urlopen` (and httpx, and a websockets client) honour HTTP_PROXY / HTTPS_PROXY / ALL_PROXY. On a machine with a proxy
configured, the `nell` commands that talk to the persona's bridge on 127.0.0.1 therefore sent their requests, bearer token
included, to the proxy instead of straight to the bridge. These tests stand up a REAL fake bridge and a REAL fake proxy on
loopback, set every proxy environment variable to the fake proxy, and assert the call reaches the bridge with its token while the
proxy sees no connection at all. A final scan keeps a new loopback call from being added the old way.

Synthetic data only.
"""

from __future__ import annotations

import ast
import json
import socket
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

TOKEN = "secret-token-475"
_PROXY_VARS = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "WS_PROXY", "ws_proxy"]
_BRIDGE_ANSWERS = {
    "/kindled-link/identity/rotate": {"new_key_id": "k2", "fingerprint_phrase": "red wolf"},
    "/kindled-link/my-code": {"code": "CODE-123", "fingerprint_phrase": "blue heron"},
    "/kindled-link/connect": {"peer_id": "peer-1", "consent_state": "pending", "relay_url": "https://relay.example"},
    "/kindled-link/self-test": {"ok": True, "stages": [{"name": "relay", "ok": True}], "relay_url": "https://relay.example"},
    "/supervisor/shutdown": {"ok": True},
    "/tools/search_memories": {"memories": [{"id": "m1"}], "mode": "lexical"},
}


class _Proxy(socketserver.ThreadingTCPServer):
    """Counts every TCP connection it receives; answers 502."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self):
        self.connections = 0
        self.lock = threading.Lock()
        super().__init__(("127.0.0.1", 0), _ProxyHandler)


class _ProxyHandler(socketserver.BaseRequestHandler):
    def handle(self):
        with self.server.lock:
            self.server.connections += 1
        try:
            self.request.recv(65536)
            self.request.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        except OSError:
            pass


class _Bridge(HTTPServer):
    def __init__(self, status: int = 200):
        self.status = status
        self.seen: list[dict] = []
        super().__init__(("127.0.0.1", 0), _BridgeHandler)


class _BridgeHandler(BaseHTTPRequestHandler):
    def _answer(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.seen.append(
            {"method": self.command, "path": self.path, "auth": self.headers.get("Authorization"), "body": body}
        )
        status = self.server.status
        payload = _BRIDGE_ANSWERS.get(self.path, {}) if status < 400 else {"detail": "bridge says no"}
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    do_GET = do_POST = _answer  # noqa: N815 — http.server's required method names

    def log_message(self, *args):  # silence
        pass


def _serve(server):
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return t


@pytest.fixture
def world(monkeypatch, tmp_path):
    """A fake bridge, a fake proxy, and every proxy variable pointing at the proxy."""
    proxy, bridge = _Proxy(), _Bridge()
    _serve(proxy)
    _serve(bridge)
    for var in _PROXY_VARS:
        monkeypatch.setenv(var, f"http://127.0.0.1:{proxy.server_address[1]}")
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    state = SimpleNamespace(port=bridge.server_address[1], auth_token=TOKEN, pid=1)
    yield SimpleNamespace(proxy=proxy, bridge=bridge, state=state, tmp=tmp_path)
    proxy.shutdown()
    bridge.shutdown()
    proxy.server_close()
    bridge.server_close()


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cli(monkeypatch, state):
    import brain.cli as cli

    monkeypatch.setattr(cli.state_file, "read", lambda persona_dir: state)
    return cli


_ARGS = {
    "_kindled_rotate_handler": {"persona": "canary"},
    "_kindled_my_code_handler": {"persona": "canary"},
    "_kindled_connect_handler": {"persona": "canary", "code": "CODE-XYZ"},
    "_kindled_self_test_handler": {"persona": "canary"},
}
_PATHS = {
    "_kindled_rotate_handler": "/kindled-link/identity/rotate",
    "_kindled_my_code_handler": "/kindled-link/my-code",
    "_kindled_connect_handler": "/kindled-link/connect",
    "_kindled_self_test_handler": "/kindled-link/self-test",
}


@pytest.mark.parametrize("handler", sorted(_ARGS))
def test_each_kindled_command_reaches_the_bridge_directly_with_its_token(world, monkeypatch, capsys, handler):
    cli = _cli(monkeypatch, world.state)

    rc = getattr(cli, handler)(SimpleNamespace(**_ARGS[handler]))

    assert rc == 0, capsys.readouterr()
    assert world.proxy.connections == 0  # the proxy was never contacted
    assert [(s["path"], s["auth"]) for s in world.bridge.seen] == [(_PATHS[handler], f"Bearer {TOKEN}")]


def test_the_connect_command_still_sends_its_code_as_json(world, monkeypatch, capsys):
    cli = _cli(monkeypatch, world.state)

    cli._kindled_connect_handler(SimpleNamespace(persona="canary", code="CODE-XYZ"))

    assert json.loads(world.bridge.seen[0]["body"]) == {"code": "CODE-XYZ"}


def test_the_output_of_the_kindled_commands_is_unchanged(world, monkeypatch, capsys):
    cli = _cli(monkeypatch, world.state)

    cli._kindled_rotate_handler(SimpleNamespace(persona="canary"))
    cli._kindled_my_code_handler(SimpleNamespace(persona="canary"))

    out = capsys.readouterr().out
    assert "Key rotated. New key ID: k2" in out and "New fingerprint phrase: red wolf" in out
    assert "CODE-123" in out and "Fingerprint (optional cross-check): blue heron" in out


def test_a_bridge_error_status_is_reported_as_before(world, monkeypatch, capsys):
    world.bridge.status = 500
    cli = _cli(monkeypatch, world.state)

    rc = cli._kindled_my_code_handler(SimpleNamespace(persona="canary"))

    err = capsys.readouterr().err
    assert rc == 1 and "error: bridge returned 500:" in err and "bridge says no" in err
    assert world.proxy.connections == 0


def test_an_unreachable_bridge_is_reported_as_before(world, monkeypatch, capsys):
    state = SimpleNamespace(port=_closed_port(), auth_token=TOKEN, pid=1)
    cli = _cli(monkeypatch, state)

    rc = cli._kindled_my_code_handler(SimpleNamespace(persona="canary"))

    assert rc == 1 and "error: could not reach bridge:" in capsys.readouterr().err
    assert world.proxy.connections == 0


def test_the_daemon_shutdown_request_goes_direct(world):
    from brain.bridge import daemon

    daemon._request_shutdown_via_http(world.state)

    assert world.proxy.connections == 0
    assert [(s["path"], s["auth"]) for s in world.bridge.seen] == [("/supervisor/shutdown", f"Bearer {TOKEN}")]


def test_the_mcp_search_call_goes_direct(world, monkeypatch):
    import brain.mcp_server.tools as tools

    monkeypatch.setattr(tools.state_file, "read", lambda persona_dir: world.state)

    result = tools._search_via_bridge(world.tmp, {"query": "x", "limit": 3})

    assert result == {"memories": [{"id": "m1"}], "mode": "lexical"}
    assert world.proxy.connections == 0
    assert world.bridge.seen[0]["auth"] == f"Bearer {TOKEN}"


def test_the_direct_helper_never_reads_proxy_variables(world):
    from brain.bridge.loopback import direct_request

    status, body = direct_request(
        "GET", f"http://127.0.0.1:{world.state.port}/kindled-link/my-code", headers={"Authorization": f"Bearer {TOKEN}"}
    )

    assert status == 200 and json.loads(body)["code"] == "CODE-123" and world.proxy.connections == 0


def test_the_direct_websocket_connect_never_uses_a_proxy(world):
    from websockets.sync.server import serve

    from brain.bridge.loopback import direct_ws_connect

    def echo(ws):
        ws.send("hello " + ws.recv())

    with serve(echo, "127.0.0.1", 0) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.socket.getsockname()[1]
        with direct_ws_connect(f"ws://127.0.0.1:{port}/events") as ws:
            ws.send("canary")
            assert ws.recv() == "hello canary"
        server.shutdown()

    assert world.proxy.connections == 0


# ---------------------------------------------------------------------------
# A scan, so a new loopback call cannot be added the old way
# ---------------------------------------------------------------------------

_BRAIN = Path(__file__).resolve().parents[4] / "brain"
_HTTPX_VERBS = {"get", "post", "put", "patch", "delete", "head", "request", "stream"}
_HTTPX_CLIENTS = {"Client", "AsyncClient"}
# provider.py talks to an Ollama host the user configures (possibly remote, possibly behind a proxy on purpose).
# supervisor.py builds the Kindled RELAY poll client (a remote service, no bridge token is sent): its missing trust_env=False is
# recorded as a found-not-fixed item in the bundle's RESUME-HERE (scope frozen), not accepted as right.
_NOT_LOOPBACK = {"bridge/provider.py"}
_NOT_LOOPBACK_CLIENT = {"bridge/supervisor.py"}
_WEBSOCKET_OK_MODULES = {"websockets.exceptions"}  # exception classes only


def _is_false(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def violations(source: str, rel: str) -> list[str]:
    """Every way the source could reach the bridge through the environment's proxy settings (code review 3b, S1)."""
    out: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        # urlopen in any spelling: urllib.request.urlopen(...), request.urlopen(...), a bare urlopen(...) after a from-import
        if isinstance(node, ast.Attribute) and node.attr == "urlopen":
            out.append(f"{rel}:{node.lineno} urlopen")
        elif isinstance(node, ast.Name) and node.id == "urlopen":
            out.append(f"{rel}:{node.lineno} urlopen")
        elif isinstance(node, ast.ImportFrom) and node.module == "urllib.request" and any(a.name == "urlopen" for a in node.names):
            out.append(f"{rel}:{node.lineno} from urllib.request import urlopen")
        # websockets clients: any import of the library (not its exception classes) outside the direct helper, aliased or not
        if rel != "bridge/loopback.py":
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("websockets") and node.module not in _WEBSOCKET_OK_MODULES:
                out.append(f"{rel}:{node.lineno} websockets import")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith("websockets") and a.name not in _WEBSOCKET_OK_MODULES:
                        out.append(f"{rel}:{node.lineno} websockets import")
        if rel in _NOT_LOOPBACK:
            continue
        # httpx: names imported straight from the library cannot be checked at the call, so the import is the violation
        if isinstance(node, ast.ImportFrom) and node.module == "httpx":
            for a in node.names:
                if a.name in _HTTPX_VERBS | _HTTPX_CLIENTS:
                    out.append(f"{rel}:{node.lineno} from httpx import {a.name}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "httpx" and (node.func.attr in _HTTPX_VERBS or node.func.attr in _HTTPX_CLIENTS):
                if node.func.attr in _HTTPX_CLIENTS and rel in _NOT_LOOPBACK_CLIENT:
                    continue
                kw = {k.arg: k.value for k in node.keywords}
                if not _is_false(kw.get("trust_env")):
                    out.append(f"{rel}:{node.lineno} httpx.{node.func.attr} without trust_env=False")
    return out


def _brain_violations() -> list[str]:
    found: list[str] = []
    for path in sorted(_BRAIN.rglob("*.py")):
        found += violations(path.read_text(encoding="utf-8"), path.relative_to(_BRAIN).as_posix())
    return found


def test_no_brain_code_reaches_the_bridge_through_the_environments_proxy_settings():
    assert _brain_violations() == []


def test_the_scan_catches_every_spelling_it_exists_for():
    cases = {
        "import urllib.request\nurllib.request.urlopen(u)": "urlopen",
        "from urllib import request\nrequest.urlopen(u)": "urlopen",
        "from urllib.request import urlopen\nurlopen(u)": "urlopen",
        "import httpx\nhttpx.post(u)": "without trust_env=False",
        "import httpx\nhttpx.Client(base_url=u)": "without trust_env=False",
        "import httpx\nhttpx.AsyncClient()": "without trust_env=False",
        "import httpx\nhttpx.Client(trust_env=True)": "without trust_env=False",
        "from httpx import post\npost(u)": "from httpx import post",
        "from httpx import Client": "from httpx import Client",
        "from websockets.sync.client import connect": "websockets import",
        "import websockets.sync.client as c\nc.connect(u)": "websockets import",
        "import websockets": "websockets import",
    }
    for source, expect in cases.items():
        assert any(expect in v for v in violations(source, "x/y.py")), source
    # and what is allowed stays allowed
    assert violations("import httpx\nhttpx.post(u, trust_env=False)\nhttpx.Client(trust_env=False)", "x/y.py") == []
    assert violations("from websockets.exceptions import InvalidStatus", "x/y.py") == []
    assert violations("import websockets", "bridge/loopback.py") == []
    assert violations("import httpx\nhttpx.get(u)", "bridge/provider.py") == []

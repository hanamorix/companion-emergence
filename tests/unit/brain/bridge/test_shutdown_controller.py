from __future__ import annotations

from brain.bridge.shutdown import BridgeShutdownController


class FakeServer:
    should_exit = False


def test_request_marks_bound_uvicorn_server_for_exit() -> None:
    server = FakeServer()
    controller = BridgeShutdownController()
    controller.bind_server(server)

    accepted = controller.request("manual_restart")

    assert accepted is True
    assert server.should_exit is True
    assert controller.requested is True
    assert controller.reason == "manual_restart"


def test_request_is_idempotent_and_keeps_first_reason() -> None:
    server = FakeServer()
    controller = BridgeShutdownController()
    controller.bind_server(server)

    assert controller.request("idle_timeout") is True
    assert controller.request("manual_restart") is True

    assert server.should_exit is True
    assert controller.reason == "idle_timeout"


def test_request_without_server_returns_false() -> None:
    controller = BridgeShutdownController()

    assert controller.request("manual_restart") is False
    assert controller.requested is False
    assert controller.reason is None

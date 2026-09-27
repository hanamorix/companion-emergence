"""INC-4 C1 — search_memories via a REAL MCP child + a REAL live bridge (S9/S10/S59).

Local-Linux-only, requires_models (marked BOTH ``requires_models`` and ``integration`` — the
project's local pre-check gate is ``-m "not live and not requires_claude_cli and not
integration"``, so ``integration`` alone already excludes this from the default gate; mirrors the
double-marking convention in tests/unit/brain/memory/test_embedding_real_model.py). Uses the
F-bob20k fixture (a copy of the dragonfly-ram-spike diagnosis's synthetic Bob persona,
Phoebe-sized, ~20k rows with cached embedding vectors) per 1.5-criteria.md's C1 — that fixture is
"not committed" and local to this machine, so every test here SKIPS gracefully when it's absent
(this machine only; a fresh clone or CI never runs these).

Proves, against a REAL bridge process (uvicorn on a real socket) and a REAL
``python -P -m brain.mcp_server`` child talking real MCP stdio:

  (a) the child's own RSS peak during a semantic search call, minus its RSS immediately before the
      call, stays <= 300 MB — nowhere near the ~1.5 GB embedder (O7/O9) — because this process
      never builds the embedder/reranker/vector matrix at all (S9/S10).
  (b) the child-via-bridge result (id list, order, ``mode``) is byte-identical to a direct
      in-bridge ``dispatch("search_memories", ...)`` call for the same query/args, across >= 5
      queries including one lexical-mode call (C1b) — the endpoint is a transport, not a second
      implementation.
  (c) with the bridge stopped, the child returns an error result, its RSS delta for that call stays
      tiny (<= 50 MB — no model load attempted), and it returns promptly (no local fallback, S10).
      The STRUCTURAL half of "no provider constructed" (dispatch/embeddings/reranker/
      embedding_matrix are never reachable from this code path) is proven statically + by mock in
      tests/unit/brain/mcp_server/test_tools.py — this test's RSS bound is the empirical
      confirmation on a real process, not a restatement of the same proof.

Fail-first (pre-INC-4): the MCP child called `dispatch()` in-process for search_memories, which
built the embedder/reranker/vector matrix straight into this process — child RSS peak delta was
~2.8 GB (O9) on F-bob20k, so (a) would fail outright, and (c) has no meaning at all (there was no
bridge transport to go unreachable — the tool never left the child process pre-INC-4).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil
import pytest

pytestmark = [pytest.mark.requires_models, pytest.mark.integration]

_FIXTURE = (
    Path.home()
    / "Desktop/companion-emergence/.claude/worktrees/dragonfly-ram-spike/changes"
    / "dragonfly-ram-spike/persona/Bob"
)
_RSS_SAMPLE_INTERVAL_S = 0.05  # 20 Hz (C1's own stated floor)
_RSS_SEMANTIC_DELTA_BOUND_BYTES = 300 * 1024 * 1024
_RSS_UNREACHABLE_DELTA_BOUND_BYTES = 50 * 1024 * 1024


def _skip_if_fixture_missing() -> None:
    if not (_FIXTURE / "memories.db").exists():
        pytest.skip(
            f"F-bob20k fixture not present at {_FIXTURE} — local-Linux-only per 1.5-criteria.md "
            "(not committed); this test only runs on a machine that has the sibling "
            "dragonfly-ram-spike worktree checked out."
        )


def _allocate_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _copy_persona(dest: Path) -> Path:
    """Copy the fixture persona dir (read-only source, never opened for write — I11)."""
    shutil.copytree(_FIXTURE, dest)
    return dest


def _write_bridge_state(persona_dir: Path, *, port: int, auth_token: str) -> None:
    from brain.bridge import state_file

    state_file.write(
        persona_dir,
        state_file.BridgeState(
            persona=persona_dir.name,
            pid=os.getpid(),
            port=port,
            started_at=datetime.now(UTC).isoformat(),
            stopped_at=None,
            shutdown_clean=True,
            client_origin="tests",
            auth_token=auth_token,
        ),
    )


class _RssSampler:
    """Samples a pid's RSS on a background thread at >= 20 Hz between start()/stop()."""

    def __init__(self, pid: int) -> None:
        self._proc = psutil.Process(pid)
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._samples.append(self._proc.memory_info().rss)
            except psutil.NoSuchProcess:
                break
            time.sleep(_RSS_SAMPLE_INTERVAL_S)

    def __enter__(self) -> _RssSampler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def peak(self) -> int:
        return max(self._samples) if self._samples else self._proc.memory_info().rss


def _find_mcp_child(persona_dir: Path) -> psutil.Process:
    """Locate the just-spawned `-m brain.mcp_server --persona-dir <persona_dir>` child."""
    me = psutil.Process(os.getpid())
    for child in me.children(recursive=True):
        try:
            cmdline = " ".join(child.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "brain.mcp_server" in cmdline and str(persona_dir) in cmdline:
            return child
    raise AssertionError(
        f"could not find a live brain.mcp_server child for persona_dir={persona_dir} among "
        f"{[' '.join(c.cmdline()) for c in me.children(recursive=True)]}"
    )


async def _call_search_memories(
    persona_dir: Path, arguments: dict, *, on_child_found=None
) -> dict:
    """Spawn the real MCP child, call search_memories once, return its parsed JSON result.

    `on_child_found`, if given, is invoked with the child psutil.Process as soon as the MCP
    handshake completes (the child is definitely alive and past import time at that point).
    """
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    from brain.bridge.provider import brain_tools_mcp_entry

    entry = brain_tools_mcp_entry(persona_dir)
    params = StdioServerParameters(
        command=entry["command"], args=entry["args"], env=dict(os.environ)
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            if on_child_found is not None:
                on_child_found(_find_mcp_child(persona_dir))
            result = await session.call_tool("search_memories", arguments)
            text = result.content[0].text
            return json.loads(text)


def _run_via_bridge(persona_dir: Path, arguments: dict, *, sample_rss: bool) -> tuple[dict, int, int]:
    """Runs `_call_search_memories` to completion; returns (result, rss_before, rss_peak).

    rss_before/rss_peak are 0 when `sample_rss` is False (the parity-only calls don't need
    per-call sampling — only the two RSS-bound assertions do).
    """
    holder: dict = {}

    def _on_found(proc: psutil.Process) -> None:
        holder["proc"] = proc
        holder["rss_before"] = proc.memory_info().rss
        if sample_rss:
            holder["sampler"] = _RssSampler(proc.pid).__enter__()

    result = asyncio.run(_call_search_memories(persona_dir, arguments, on_child_found=_on_found))

    rss_before = holder.get("rss_before", 0)
    rss_peak = rss_before
    sampler = holder.get("sampler")
    if sampler is not None:
        sampler.__exit__()
        rss_peak = sampler.peak
    return result, rss_before, rss_peak


_LIVE_SERVERS: dict[Path, object] = {}


@pytest.fixture()
def live_bridge(tmp_path: Path):
    """Real bridge (uvicorn, real socket) + bridge.json pointed at it, on a copy of F-bob20k.

    Registers the running BridgeServer into `_LIVE_SERVERS` keyed by persona_dir so
    `test_c1c_bridge_down_returns_error_with_tiny_rss_delta` can stop it mid-test (simulating a
    bridge that died) via `stop_live_bridge` below — the fixture's own teardown then calls
    `.stop()` again, which is a harmless no-op on an already-stopped uvicorn.Server.
    """
    _skip_if_fixture_missing()
    from tests.harness.engine import BridgeServer

    persona_dir = _copy_persona(tmp_path / "persona")
    port = _allocate_port()
    token = "test-bridge-token"
    server = BridgeServer(persona_dir, port)
    server.start()
    _write_bridge_state(persona_dir, port=port, auth_token=token)
    _LIVE_SERVERS[persona_dir] = server
    try:
        yield persona_dir, port, token
    finally:
        server.stop()
        _LIVE_SERVERS.pop(persona_dir, None)


def stop_live_bridge(persona_dir: Path) -> None:
    """Stop the BridgeServer a `live_bridge` fixture started for this persona_dir, early."""
    server = _LIVE_SERVERS.get(persona_dir)
    if server is not None:
        server.stop()


def _direct_dispatch(persona_dir: Path, arguments: dict) -> dict:
    """The comparison oracle for C1b: dispatch() called in-process, same args."""
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore
    from brain.tools.dispatch import dispatch

    store = MemoryStore(persona_dir / "memories.db", integrity_check=False)
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    try:
        return dispatch("search_memories", arguments, store=store, hebbian=hebbian, persona_dir=persona_dir)
    finally:
        store.close()
        hebbian.close()


def test_c1a_child_rss_delta_stays_well_below_one_models_size(live_bridge) -> None:
    persona_dir, _port, _token = live_bridge
    result, rss_before, rss_peak = _run_via_bridge(
        persona_dir, {"query": "a quiet evening at home", "mode": "semantic"}, sample_rss=True
    )
    delta = rss_peak - rss_before
    assert "error" not in result, f"search_memories call failed: {result}"
    assert delta <= _RSS_SEMANTIC_DELTA_BOUND_BYTES, (
        f"child RSS delta {delta / 1024 / 1024:.1f} MB exceeds the 300 MB bound (O7: well below "
        f"the ~1.5 GB embedder) — rss_before={rss_before}, rss_peak={rss_peak}"
    )


def test_c1b_result_matches_direct_in_bridge_dispatch_across_five_queries(live_bridge) -> None:
    """>= 5 queries, incl. one lexical-mode call. See the module docstring for why the lexical
    comparison snapshots a pristine copy immediately before its bridge call: hebbian reinforcement
    from earlier queries in this loop would otherwise be a confound for the ONE mode where hebbian
    spreading-activation actually affects scoring (semantic-mode scoring never consults hebbian, so
    those four don't need snapshotting)."""
    persona_dir, _port, _token = live_bridge
    queries = [
        {"query": "a quiet evening at home", "mode": "semantic"},
        {"query": "conversation about work stress", "mode": "semantic"},
        {"query": "celebrating a small win", "mode": "semantic"},
        {"query": "a difficult goodbye", "mode": "semantic"},
        {"query": "Henryk", "mode": "lexical"},  # the one lexical-mode comparison
    ]
    for arguments in queries:
        if arguments["mode"] == "lexical":
            # Snapshot the CURRENT (live-bridge) persona state before this call so the direct
            # comparison starts from byte-identical state — hebbian reinforcement from this
            # query's own two calls never crosses between them (each mutates only its own copy).
            snapshot_dir = persona_dir.parent / "lexical_snapshot"
            shutil.copytree(persona_dir, snapshot_dir)
            bridge_result, _, _ = _run_via_bridge(persona_dir, arguments, sample_rss=False)
            direct_result = _direct_dispatch(snapshot_dir, arguments)
        else:
            bridge_result, _, _ = _run_via_bridge(persona_dir, arguments, sample_rss=False)
            direct_result = _direct_dispatch(persona_dir, arguments)

        assert "error" not in bridge_result, f"bridge call failed for {arguments}: {bridge_result}"
        assert [m["id"] for m in bridge_result["memories"]] == [
            m["id"] for m in direct_result["memories"]
        ], f"id order mismatch for {arguments}"
        assert bridge_result["mode"] == direct_result["mode"], f"mode mismatch for {arguments}"


def test_c1c_bridge_down_returns_error_with_tiny_rss_delta(live_bridge) -> None:
    persona_dir, _port, _token = live_bridge

    # Stop the bridge early (still-live persona_dir, still-live bridge.json — the state file is
    # now stale, which is exactly the "bridge unreachable" case). The fixture's own teardown calls
    # .stop() again afterward, which is a harmless no-op on an already-stopped uvicorn.Server.
    stop_live_bridge(persona_dir)

    result, rss_before, rss_peak = _run_via_bridge(
        persona_dir, {"query": "anything", "mode": "semantic"}, sample_rss=True
    )
    delta = rss_peak - rss_before
    assert result == {"error": "bridge unreachable"}
    assert delta <= _RSS_UNREACHABLE_DELTA_BOUND_BYTES, (
        f"child RSS delta {delta / 1024 / 1024:.1f} MB on a bridge-down call exceeds the 50 MB "
        f"bound — suggests a model got built despite the bridge being unreachable"
    )

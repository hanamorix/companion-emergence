"""# Safety tier: 1

Time three old emotion code paths on a copy of a synthetic memories.db.

Usage:
    bash scripts/time_old_emotion_paths.sh [path to memories.db]
    uv run python scripts/time_old_emotion_paths.py [path to memories.db]

The path is optional and defaults to test-data/synthetic-db/memories.db under the repository root. The file is
copied into a fresh temporary directory (the copy is the persona directory) and three paths are timed on the copy:
get_emotional_state, one write_trace_memory, and the heartbeat's decay pass (HeartbeatEngine._apply_emotion_decay).
Each of the first two is one cold first call plus nine warm repeats; the decay pass is ten consecutive passes, the
first being the cold run, with the rows each pass changed. Warm timings are reported as minimum, median and maximum
in milliseconds. Nothing is asserted: there is no threshold and no pass or fail.

It refuses a path under the brain's home directory, aborts when the source has a non-empty -wal file, copies only
the main database file, and removes its temporary directory when it ends. It starts one process, `git rev-parse
HEAD`, calls no model and reads no persona data file other than the path given (the app code it imports may read
its own settings file from the home directory). The AVX2 field comes from the app's own probe, which reads
/proc/cpuinfo on Linux and answers "no" when that read fails; the script prints "unknown" on other systems.

A timing whose function is absent on the checked-out commit is skipped with a printed reason. On a failure it
prints one line, `time_old_emotion_paths: FAILED at <step>: <detail>`, to stderr, prints no timings and exits 1.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

WARM_REPEATS = 9
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = Path("test-data") / "synthetic-db" / "memories.db"


class StepError(Exception):
    """A step that cannot continue; carries the step name for the one-line abort message."""

    def __init__(self, step: str, detail: str) -> None:
        super().__init__(detail)
        self.step = step
        self.detail = detail


def _under_home(path: Path, home: Path) -> bool:
    return path == home or home in path.parents


def _step_source(arg: str | None) -> tuple[Path, str]:
    """Resolve and vet the source file; return it with the commit the repository is on."""
    from brain.paths import get_home

    given = Path(arg).expanduser() if arg else REPO_ROOT / DEFAULT_DB
    source = given.resolve()
    if not source.is_file():
        raise StepError("source", f"not a regular file: {source}")
    home = get_home()
    if _under_home(source, home):
        raise StepError("source", f"refusing a path under the brain's home directory: {source}")
    wal = source.with_name(source.name + "-wal")
    if wal.is_file() and wal.stat().st_size > 0:
        raise StepError("source", f"{wal} has content: close the database that owns it first")
    done = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    commit = done.stdout.strip()
    if done.returncode != 0 or not commit:
        raise StepError("source", f"git rev-parse HEAD failed: {done.stderr.strip() or 'no output'}")
    return source, commit


def _step_copy(source: Path, scratch: Path) -> str:
    """Copy the main database file only; return the source's sha256."""
    from brain.utils.atomic_write import save_bytes_atomic

    try:
        data = source.read_bytes()
        save_bytes_atomic(scratch / "memories.db", data)
    except OSError as exc:
        raise StepError("copy", f"{type(exc).__name__}: {exc}") from exc
    return hashlib.sha256(data).hexdigest()


def _step_open(scratch: Path):
    """Open the copy as the app does (default integrity check, schema and migrations)."""
    from brain.memory.store import MemoryStore

    try:
        return MemoryStore(scratch / "memories.db")
    except Exception as exc:  # noqa: BLE001 - any open or migration failure becomes this one named abort
        raise StepError("open", f"{type(exc).__name__}: {exc}") from exc


def _step_vocabulary(scratch: Path, store) -> tuple[int, int]:
    """Register the copy's emotion names through the app's own vocabulary load; run once per process."""
    from brain.emotion import vocabulary
    from brain.emotion.persona_loader import load_persona_vocabulary

    try:
        registered = load_persona_vocabulary(scratch / "emotion_vocabulary.json", store=store)
        registry_size = len(vocabulary.list_all())
    except Exception as exc:  # noqa: BLE001 - reported as the named step
        raise StepError("vocabulary", f"{type(exc).__name__}: {exc}") from exc
    return registered, registry_size


def _ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.1f} ms"


def _timing_lines(label: str, cold_label: str, durations: list[float]) -> list[str]:
    warm = durations[1:]
    return [
        f"{label}",
        f"  cold ({cold_label}): {_ms(durations[0])}",
        f"  warm x{len(warm)}: min {_ms(min(warm))}, median {_ms(statistics.median(warm))}, "
        f"max {_ms(max(warm))}",
    ]


def _time_calls(call: Callable[[], object], n: int, before_each: Callable[[], object] | None = None) -> list[float]:
    durations = []
    for _ in range(n):
        if before_each is not None:
            before_each()
        started = time.perf_counter()
        call()
        durations.append(time.perf_counter() - started)
    return durations


def _time_state(scratch: Path, store) -> list[str]:
    try:
        from brain.tools.impls.get_emotional_state import get_emotional_state
    except (ImportError, AttributeError) as exc:
        return [f"state: skipped, get_emotional_state is not available on this commit ({exc})"]
    from brain.memory.hebbian import HebbianMatrix

    hebbian = HebbianMatrix(":memory:")
    try:
        try:
            durations = _time_calls(
                lambda: get_emotional_state(store=store, hebbian=hebbian, persona_dir=scratch),
                1 + WARM_REPEATS,
            )
        except Exception as exc:  # noqa: BLE001 - reported as the named step
            raise StepError("state", f"{type(exc).__name__}: {exc}") from exc
    finally:
        hebbian.close()
    return _timing_lines(
        "get_emotional_state (rows changed: 0)",
        "process-cold, OS page cache warm (fresh copy)",
        durations,
    )


def _time_trace(scratch: Path, store) -> list[str]:
    try:
        from brain.monologue.trace import write_trace_memory
    except (ImportError, AttributeError) as exc:
        return [f"trace: skipped, write_trace_memory is not available on this commit ({exc})"]
    from brain.health.json_io import write_jsonl

    traces = [m for m in store.list_active() if m.memory_type == "monologue_trace"]
    if not traces:
        raise StepError("trace", "the database holds no active monologue_trace rows to take a text from")
    middle = statistics.median_low(len(m.content) for m in traces)
    text = next(m.content for m in traces if len(m.content) == middle)
    queue = scratch / "pending_candidates.jsonl"
    try:
        durations = _time_calls(
            lambda: write_trace_memory(store, text),
            1 + WARM_REPEATS,
            before_each=lambda: write_jsonl(queue, []),
        )
    except Exception as exc:  # noqa: BLE001 - reported as the named step
        raise StepError("trace", f"{type(exc).__name__}: {exc}") from exc
    return _timing_lines(
        f"write_trace_memory, one trace of {middle} characters (rows changed: 1)",
        "process-cold, OS page cache warm (fresh copy)",
        durations,
    )


def _time_decay(scratch: Path, store) -> list[str]:
    from brain.engines.heartbeat import HeartbeatEngine, HeartbeatState

    if getattr(HeartbeatEngine, "_apply_emotion_decay", None) is None:
        return ["decay: skipped, HeartbeatEngine._apply_emotion_decay is not on this commit"]
    from brain.bridge import supervisor
    from brain.bridge.provider import FakeProvider
    from brain.memory.hebbian import HebbianMatrix

    interval = inspect.signature(supervisor.run_folded).parameters["heartbeat_interval_s"].default
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or interval <= 0:
        raise StepError("decay", f"run_folded's heartbeat_interval_s default is not a positive number: {interval!r}")
    hebbian = HebbianMatrix(":memory:")
    try:
        try:
            engine = HeartbeatEngine(
                store=store,
                hebbian=hebbian,
                provider=FakeProvider(),
                state_path=scratch / "heartbeat_state.json",
                config_path=scratch / "heartbeat_config.json",
                dream_log_path=scratch / "dreams.log.jsonl",
                heartbeat_log_path=scratch / "heartbeats.log.jsonl",
                persona_name="Timing",
                persona_system_prompt="You are Timing.",
            )
            state = HeartbeatState.fresh("timing")
            changed: list[int] = []

            def one_pass() -> None:
                changed.append(
                    engine._apply_emotion_decay(
                        float(interval), tick_at=datetime.now(UTC), state=state, dry_run=False
                    )
                )

            durations = _time_calls(one_pass, 1 + WARM_REPEATS)
        except Exception as exc:  # noqa: BLE001 - reported as the named step
            raise StepError("decay", f"{type(exc).__name__}: {exc}") from exc
    finally:
        hebbian.close()
    lines = _timing_lines(
        f"HeartbeatEngine._apply_emotion_decay, {1 + WARM_REPEATS} consecutive passes of {interval:g} s",
        "first pass, the warm-up",
        durations,
    )
    lines.append("  rows changed per pass: " + ", ".join(str(n) for n in changed))
    return lines


def _host_line() -> str:
    cpu = platform.processor() or platform.machine() or "unknown"
    avx2 = "unknown"
    ram = "unknown"
    try:
        if sys.platform == "linux":
            from brain.memory.reranker import _detect_avx2

            avx2 = "yes" if _detect_avx2() else "no"
        from brain.memory.judge_selftune import _read_total_ram_bytes

        total = _read_total_ram_bytes()
        if total is not None:
            ram = f"{total / 1024**3:.1f} GiB"
    except ImportError:
        pass
    return (
        f"host: {platform.platform()}; cpu {cpu}; avx2={avx2}; cores {os.cpu_count() or 'unknown'}; ram {ram}"
    )


def run(arg: str | None) -> list[str]:
    source, commit = _step_source(arg)
    scratch = Path(tempfile.mkdtemp(prefix="a1r-timing-")).resolve()
    store = None
    try:
        from brain.paths import get_home

        if _under_home(scratch, get_home()):
            raise StepError("copy", f"the temporary directory is under the brain's home directory: {scratch}")
        digest = _step_copy(source, scratch)
        store = _step_open(scratch)
        registered, registry_size = _step_vocabulary(scratch, store)
        active = len(store.list_active())
        out = [
            _host_line(),
            f"commit: {commit}",
            f"database: sha256 {digest}",
            f"emotion names registered by the first vocabulary load: {registered}; registry size: {registry_size}",
            "  (a name with no stored half-life gets the placeholder half-life in brain.health.reconstruct, "
            "14.0 days, which sets how many rows the decay pass rewrites)",
            f"active rows: {active}",
            "",
        ]
        out += _time_state(scratch, store) + [""]
        out += _time_trace(scratch, store) + [""]
        out += _time_decay(scratch, store)
        return out
    finally:
        if store is not None:
            store.close()
        try:
            shutil.rmtree(scratch)
        except OSError as exc:
            _CLEANUP_ERRORS.append(f"time_old_emotion_paths: could not remove {scratch}: {exc}")


_CLEANUP_ERRORS: list[str] = []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "db",
        nargs="?",
        default=None,
        help=f"path to a memories.db (default: {DEFAULT_DB.as_posix()} under the repository root)",
    )
    args = parser.parse_args(argv)
    try:
        lines = run(args.db)
    except StepError as exc:
        print(f"time_old_emotion_paths: FAILED at {exc.step}: {exc.detail}", file=sys.stderr)
        for message in _CLEANUP_ERRORS:
            print(message, file=sys.stderr)
        return 1
    print("\n".join(lines))
    for message in _CLEANUP_ERRORS:
        print(message, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

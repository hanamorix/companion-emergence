"""S253: free memory must be at least a bound before the reranker model is loaded (the RAM lull job and
the floor job alike) and before the ladder's first rung. The bound is the model file's size for a load,
and the kept fit's one-pair prediction for the full-length pair (else the file size) for the first rung.
Below it nothing loads or runs and the next lull retries. Real `run_ram_figure`, `measure_ram_figure`,
`run_rerank_floor` and the real file-size reader over a temp hub-layout cache; scripted provider, patched
free-memory reading. No model."""

from __future__ import annotations

import inspect
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain import dev_constants
from brain.memory import cost_series, floor_calibration, floor_startup, ram_fit
from brain.memory import reranker as reranker_mod
from brain.memory.store import MemoryStore
from tests.unit.brain.memory._ram_support import FULL, ScriptedLadderProvider

_MODEL = "scripted-ladder"
_FLOOR_MODEL = "rr"
_REAL_BLOCKED = reranker_mod.model_load_blocked_by_memory  # the suite default replaces it with `lambda: False`
_REAL_FILE_BYTES = reranker_mod._runtime_model_file_bytes  # noqa: SLF001
_FACTOR = dev_constants.RAM_FIT_SAFETY_FACTOR  # the shipped S255 value, read at import (the suite default is 1.0)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_mod, "model_load_blocked_by_memory", _REAL_BLOCKED)
    monkeypatch.setattr(reranker_mod, "_require_ram_figure", True)
    monkeypatch.setattr(reranker_mod.peak_rss, "available", lambda: True)
    monkeypatch.setattr(reranker_mod, "resolve_reranker_model_id", lambda: _MODEL)
    monkeypatch.setattr(reranker_mod, "_full_rerank_budget", lambda: 2.0)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: None)


@pytest.fixture
def store(tmp_path: Path):
    s = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    yield s
    s.close()


def _headroom(monkeypatch, value, reads: list[int] | None = None) -> None:
    def read():
        if reads is not None:
            reads.append(1)
        return value() if callable(value) else value

    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", read)


def _file_size(monkeypatch, size, calls: list[str] | None = None) -> None:
    def read(model_id: str):
        if calls is not None:
            calls.append(model_id)
        return size

    monkeypatch.setattr(reranker_mod, "_runtime_model_file_bytes", read)


def _kept_fit(store, constant: float) -> ram_fit.RamFit:
    """A kept fit that predicts exactly `constant` bytes for ONE pair of FULL tokens and grows with the shape
    (b = constant / FULL per padded token), so a wrong pair count or length in the call changes the bound."""
    fit = ram_fit.RamFit(a=0.0, b=constant / FULL, c=0.0, shapes=4, samples=4, max_pairs=1, max_length=FULL)
    assert fit.predict(1, FULL) == constant and fit.predict(2, FULL) != constant and fit.predict(1, FULL - 1) != constant
    reranker_mod._ram_fit_kept[(cost_series._store_key(store), _MODEL)] = fit  # noqa: SLF001
    return fit


def _kept_fit_bytes(constant: float) -> ram_fit.RamFit:
    """A fit predicting `constant` bytes for one pair of FULL tokens (not registered anywhere)."""
    return ram_fit.RamFit(a=constant, b=0.0, c=0.0, shapes=4, samples=4, max_pairs=1, max_length=FULL)


def _s253_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "(S253)" in r.getMessage()]


# ---- the file-size reader (real temp hub-layout cache) ----


def _hub_file(root: Path, repo: str, name: str, size: int) -> None:
    folder = root / ("models--" + repo.replace("/", "--"))
    (folder / "refs").mkdir(parents=True, exist_ok=True)
    (folder / "refs" / "main").write_text("abc123")
    path = folder / "snapshots" / "abc123" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def _fp16_id() -> str:
    from brain.bridge.model_tier import MODEL_RERANKER, MODEL_RERANKER_FP16

    reranker_mod._register_fp16_reranker_model(MODEL_RERANKER_FP16, MODEL_RERANKER)  # noqa: SLF001
    return MODEL_RERANKER_FP16


def test_a_cached_file_reads_its_size_on_disk_and_the_registered_file_not_the_fp32_one(tmp_path) -> None:
    from brain.bridge.model_tier import MODEL_RERANKER

    fp16 = _fp16_id()
    _hub_file(tmp_path, MODEL_RERANKER, "onnx/model.onnx", 99)  # the fp32 export: a different file
    _hub_file(tmp_path, MODEL_RERANKER, "onnx/model_fp16.onnx", 1234)
    assert reranker_mod._reranker_model_file_bytes(fp16, tmp_path) == 1234.0  # noqa: SLF001
    assert reranker_mod._reranker_model_file_bytes(MODEL_RERANKER, tmp_path) == 99.0  # noqa: SLF001


def test_a_file_not_cached_reads_the_registry_declared_size_in_bytes(tmp_path) -> None:
    from brain.bridge.model_tier import MODEL_RERANKER

    fp16 = _fp16_id()
    declared = reranker_mod._registry_entry(fp16)["size_in_GB"]  # noqa: SLF001
    assert declared == 0.56
    assert reranker_mod._reranker_model_file_bytes(fp16, tmp_path) == declared * 1e9  # noqa: SLF001 - an empty cache
    _hub_file(tmp_path, MODEL_RERANKER, "onnx/model.onnx", 99)  # only the fp32 file is cached: the fp16 one is fetched
    assert reranker_mod._reranker_model_file_bytes(fp16, tmp_path) == declared * 1e9  # noqa: SLF001


def test_a_size_that_is_not_a_positive_finite_number_is_no_size(tmp_path, monkeypatch) -> None:
    entry = {"model": "m", "sources": {"hf": "org/name"}, "model_file": "m.onnx", "size_in_GB": 0.0}
    monkeypatch.setattr(reranker_mod, "_registry_entry", lambda model_id: dict(entry))
    assert reranker_mod._reranker_model_file_bytes("m", tmp_path) is None  # noqa: SLF001 - declared zero, not cached
    for bad in (-1.0, math.nan, math.inf, "big", None, True):
        entry["size_in_GB"] = bad
        assert reranker_mod._reranker_model_file_bytes("m", tmp_path) is None, bad  # noqa: SLF001
    _hub_file(tmp_path, "org/name", "m.onnx", 0)  # a zero-byte cached file is no size either: the declared one stands in
    entry["size_in_GB"] = 0.5
    assert reranker_mod._reranker_model_file_bytes("m", tmp_path) == 0.5e9  # noqa: SLF001
    entry["size_in_GB"] = 0.0
    assert reranker_mod._reranker_model_file_bytes("m", tmp_path) is None  # noqa: SLF001
    _hub_file(tmp_path, "org/name", "m.onnx", 7)
    assert reranker_mod._reranker_model_file_bytes("m", tmp_path) == 7.0  # noqa: SLF001


def test_a_model_the_registry_does_not_know_or_a_cache_error_has_no_size_and_warns_once(tmp_path, monkeypatch, caplog) -> None:
    assert reranker_mod._reranker_model_file_bytes("nobody/unknown-reranker", tmp_path) is None  # noqa: SLF001
    import huggingface_hub

    def _boom(*args, **kwargs):
        raise OSError("cache unreadable")

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _boom)
    with caplog.at_level("WARNING"):
        for _ in range(3):
            assert reranker_mod._reranker_model_file_bytes(_fp16_id(), tmp_path) is None  # noqa: SLF001
    assert len([r for r in caplog.records if "model file's size" in r.getMessage()]) == 1


def test_the_runtime_reader_registers_the_pinned_fp16_model_before_it_reads(tmp_path, monkeypatch) -> None:
    """The check can run before the provider is ever built; without the registration the registry does not know
    the id, the size reads as undetermined and a fresh install would never download its model. fastembed's
    registry is process-wide and idempotent, so the entry is REMOVED first (and restored after)."""
    from fastembed.rerank.cross_encoder.custom_text_cross_encoder import CustomTextCrossEncoder

    from brain.bridge.model_tier import MODEL_RERANKER_FP16

    custom = CustomTextCrossEncoder.SUPPORTED_MODELS
    saved = list(custom)
    custom[:] = [m for m in custom if m.model != MODEL_RERANKER_FP16]
    try:
        assert reranker_mod._registry_entry(MODEL_RERANKER_FP16) is None  # noqa: SLF001 - the precondition
        monkeypatch.setattr("brain.paths.get_cache_dir", lambda: tmp_path)
        assert _REAL_FILE_BYTES(MODEL_RERANKER_FP16) == 0.56e9
    finally:
        custom[:] = saved


def test_the_runtime_reader_reads_the_shared_cache_dir_and_never_raises(tmp_path, monkeypatch, caplog) -> None:
    from brain.bridge.model_tier import MODEL_RERANKER

    fp16 = _fp16_id()
    _hub_file(tmp_path, MODEL_RERANKER, "onnx/model_fp16.onnx", 4321)
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: tmp_path)
    assert _REAL_FILE_BYTES(fp16) == 4321.0

    def _boom(*args, **kwargs):
        raise RuntimeError("registration exploded")

    monkeypatch.setattr(reranker_mod, "_register_fp16_reranker_model", _boom)
    with caplog.at_level(logging.WARNING, logger=reranker_mod.log.name):
        assert _REAL_FILE_BYTES(fp16) is None


# ---- the shared check: boundary, unreadable memory, undetermined bound, no lock held ----


@pytest.mark.parametrize(("headroom", "blocked"), [(999.0, True), (1000.0, False), (1001.0, False)])
def test_free_memory_must_be_at_least_the_bound_and_equality_passes(monkeypatch, headroom, blocked) -> None:
    _headroom(monkeypatch, headroom)
    assert reranker_mod._memory_floor_blocks("c", "w", lambda: 1000.0) is blocked  # noqa: SLF001


def test_free_memory_of_zero_never_passes_even_against_a_bound_of_zero(monkeypatch) -> None:
    _headroom(monkeypatch, 0.0)
    assert reranker_mod._memory_floor_blocks("c", "w", lambda: 0.0) is True  # noqa: SLF001
    _headroom(monkeypatch, 1.0)
    assert reranker_mod._memory_floor_blocks("c", "w", lambda: 0.0) is False  # noqa: SLF001


def test_unreadable_free_memory_blocks_and_the_bound_is_not_even_computed(monkeypatch, caplog) -> None:
    """S261 (was: blocks nothing, S241/S245): free memory that cannot be read at all takes the cheaper path."""
    computed: list[int] = []
    _headroom(monkeypatch, None)
    with caplog.at_level(logging.INFO, logger=reranker_mod.log.name):
        for _ in range(3):
            assert reranker_mod._memory_floor_blocks("c", "w", lambda: computed.append(1) or None) is True  # noqa: SLF001
    assert computed == []
    assert len([r for r in caplog.records if "(S261)" in r.getMessage()]) == 1, "logged once"
    _headroom(monkeypatch, 1000.0)
    assert reranker_mod._memory_floor_blocks("c", "w", lambda: 1.0) is False  # noqa: SLF001
    _headroom(monkeypatch, None)
    with caplog.at_level(logging.INFO, logger=reranker_mod.log.name):
        assert reranker_mod._memory_floor_blocks("c", "w", lambda: 1.0) is True  # noqa: SLF001
    assert len([r for r in caplog.records if "(S261)" in r.getMessage()]) == 2, "re-armed once it passed"


def test_a_bound_that_cannot_be_determined_blocks_when_the_free_memory_is_readable(monkeypatch) -> None:
    _headroom(monkeypatch, 1e12)
    assert reranker_mod._memory_floor_blocks("c", "w", lambda: None) is True  # noqa: SLF001


def test_the_bound_is_computed_with_no_lock_held(monkeypatch) -> None:
    """`_cost_lock` is not re-entrant: a bound that takes it (the first rung's kept-fit read does) must not be called
    from inside the check's own log bookkeeping."""
    _headroom(monkeypatch, 1e12)

    def bound() -> float:
        assert reranker_mod._cost_lock.acquire(blocking=False), "the bound was computed with _cost_lock held"  # noqa: SLF001
        reranker_mod._cost_lock.release()  # noqa: SLF001
        return 1.0

    assert reranker_mod._memory_floor_blocks("c", "w", bound) is False  # noqa: SLF001
    _headroom(monkeypatch, 0.5)
    assert reranker_mod._memory_floor_blocks("c", "w", bound) is True  # noqa: SLF001


# ---- log once per condition ----


def test_a_blocked_condition_logs_once_until_it_passes_and_conditions_log_separately(monkeypatch, caplog) -> None:
    level = {"free": 10.0}
    _headroom(monkeypatch, lambda: level["free"])
    with caplog.at_level(logging.INFO, logger=reranker_mod.log.name):
        for _ in range(4):
            assert reranker_mod._memory_floor_blocks("load", "the model load", lambda: 100.0) is True  # noqa: SLF001
        assert len(_s253_records(caplog)) == 1
        for _ in range(3):
            assert reranker_mod._memory_floor_blocks("rung", "the first rung", lambda: 100.0) is True  # noqa: SLF001
        assert len(_s253_records(caplog)) == 2, "another condition logs on its own"
        for _ in range(3):  # a size that cannot be determined is a third condition
            assert reranker_mod._memory_floor_blocks("load", "the model load", lambda: None) is True  # noqa: SLF001
        assert len(_s253_records(caplog)) == 3
        level["free"] = 1000.0
        assert reranker_mod._memory_floor_blocks("load", "the model load", lambda: 100.0) is False  # noqa: SLF001
        level["free"] = 10.0
        assert reranker_mod._memory_floor_blocks("load", "the model load", lambda: 100.0) is True  # noqa: SLF001
        assert len(_s253_records(caplog)) == 4, "re-armed once the check passed"


def test_a_condition_that_passed_logs_again_when_it_recurs(monkeypatch, caplog) -> None:
    level = {"free": 10.0}
    _headroom(monkeypatch, lambda: level["free"])
    with caplog.at_level(logging.INFO, logger=reranker_mod.log.name):
        for free, blocked in ((10.0, True), (10.0, True), (1000.0, False), (10.0, True), (10.0, True)):
            level["free"] = free
            assert reranker_mod._memory_floor_blocks("rung", "the first rung", lambda: 100.0) is blocked  # noqa: SLF001
        assert len(_s253_records(caplog)) == 2


def test_alternating_kinds_of_one_condition_each_log_when_they_come_back(monkeypatch, caplog) -> None:
    _headroom(monkeypatch, 10.0)
    with caplog.at_level(logging.INFO, logger=reranker_mod.log.name):
        for needed in (lambda: 100.0, lambda: None, lambda: 100.0, lambda: None):
            assert reranker_mod._memory_floor_blocks("load", "the model load", needed) is True  # noqa: SLF001
        assert len(_s253_records(caplog)) == 4


def test_no_free_memory_is_a_block_of_its_own_and_looks_up_no_bound(monkeypatch, caplog) -> None:
    computed: list[int] = []
    _headroom(monkeypatch, -5.0)
    with caplog.at_level(logging.INFO, logger=reranker_mod.log.name):
        assert reranker_mod._memory_floor_blocks("c", "w", lambda: computed.append(1) or 1.0) is True  # noqa: SLF001
    assert computed == [] and len(_s253_records(caplog)) == 1


# ---- the load check (model_load_blocked_by_memory) ----


@pytest.mark.parametrize(("headroom", "blocked"), [(5000.0, True), (9999.0, True), (10000.0, False), (11000.0, False)])
def test_with_no_fit_the_load_needs_the_file_size_twice_the_file_and_the_first_rung_bound(monkeypatch, headroom, blocked) -> None:
    """S257: the order's own bound for a first rung with no fit is the file size again."""
    _headroom(monkeypatch, headroom)
    _file_size(monkeypatch, 5000.0)
    assert reranker_mod.model_load_blocked_by_memory() is blocked


@pytest.mark.parametrize(("delta", "blocked"), [(-1.0, True), (0.0, False), (1.0, False)])
def test_with_a_kept_fit_the_load_needs_the_file_size_plus_its_one_pair_prediction_times_the_factor(
    store, monkeypatch, delta, blocked
) -> None:
    monkeypatch.setattr(dev_constants, "RAM_FIT_SAFETY_FACTOR", _FACTOR)
    _kept_fit(store, 1000.0)
    _file_size(monkeypatch, 5000.0)
    _headroom(monkeypatch, 5000.0 + 1000.0 * _FACTOR + delta)
    assert reranker_mod.model_load_blocked_by_memory() is blocked


@pytest.mark.parametrize(("headroom", "blocked"), [(9999.0, True), (10000.0, False)])
def test_the_file_size_is_not_scaled_by_the_safety_factor_in_the_load_bound(monkeypatch, headroom, blocked) -> None:
    """S255 scales predictions only: with no fit the bound is exactly twice the file size at the shipped factor too."""
    monkeypatch.setattr(dev_constants, "RAM_FIT_SAFETY_FACTOR", _FACTOR)
    _headroom(monkeypatch, headroom)
    _file_size(monkeypatch, 5000.0)
    assert reranker_mod.model_load_blocked_by_memory() is blocked


def test_the_load_takes_the_largest_first_rung_among_the_fits_the_process_keeps_for_the_model(store, monkeypatch) -> None:
    _kept_fit(store, 1000.0)
    other = ram_fit.RamFit(a=0.0, b=3000.0 / 512, c=0.0, shapes=4, samples=4, max_pairs=1, max_length=512)
    reranker_mod._ram_fit_kept[("another-store", _MODEL)] = other  # noqa: SLF001 - one pair at ITS longest length: 3000
    reranker_mod._ram_fit_kept[("another-store", "another-model")] = _kept_fit_bytes(9e9)  # noqa: SLF001 - not this model
    _file_size(monkeypatch, 5000.0)
    _headroom(monkeypatch, 7999.0)
    assert reranker_mod.model_load_blocked_by_memory() is True
    _headroom(monkeypatch, 8000.0)
    assert reranker_mod.model_load_blocked_by_memory() is False


def test_the_load_bound_reads_the_file_size_once(store, monkeypatch) -> None:
    asked: list[str] = []
    _file_size(monkeypatch, 5000.0, asked)
    _headroom(monkeypatch, 1e12)
    assert reranker_mod._load_bound(_MODEL) == 10000.0  # noqa: SLF001
    assert asked == [_MODEL]
    _kept_fit(store, 1000.0)
    asked.clear()
    assert reranker_mod._load_bound(_MODEL) == 6000.0  # noqa: SLF001 - the suite factor is 1.0
    assert asked == [_MODEL]


def test_the_size_is_asked_for_the_runtime_model_id(monkeypatch) -> None:
    asked: list[str] = []
    _headroom(monkeypatch, 1e12)
    _file_size(monkeypatch, 5000.0, asked)
    assert reranker_mod.model_load_blocked_by_memory() is False
    assert asked == [_MODEL]


def test_a_load_with_an_undetermined_file_size_is_refused_and_so_is_one_whose_free_memory_cannot_be_read(monkeypatch) -> None:
    sizes: list[str] = []
    _file_size(monkeypatch, None, sizes)
    _headroom(monkeypatch, 1e12)
    assert reranker_mod.model_load_blocked_by_memory() is True
    _headroom(monkeypatch, None)
    sizes.clear()
    assert reranker_mod.model_load_blocked_by_memory() is True, "S261 (was: proceeds)"
    assert sizes == [], "no file lookup on a host whose memory cannot be read"


class _Resident(ScriptedLadderProvider):
    loaded = True

    def model_loaded(self) -> bool:
        return self.loaded


def test_a_provider_whose_model_is_loaded_is_not_asked_and_one_that_is_cached_but_not_loaded_is(monkeypatch) -> None:
    reads: list[int] = []
    _headroom(monkeypatch, 1.0, reads)
    _file_size(monkeypatch, 5000.0)
    provider = _Resident(model=_MODEL)
    reranker_mod._provider_cache[_MODEL] = provider  # noqa: SLF001
    assert reranker_mod.model_load_blocked_by_memory() is False
    assert reads == [], "a loaded model is out of the free memory already: nothing is read"
    provider.loaded = False  # constructed (lazy_load) but the session is not built yet
    assert reranker_mod.model_load_blocked_by_memory() is True
    reranker_mod._provider_cache.pop(_MODEL)  # noqa: SLF001
    assert reranker_mod.model_load_blocked_by_memory() is True, "no provider: asked"


def test_the_real_provider_reports_a_loaded_model_the_way_the_tokenizer_accessor_tests_it() -> None:
    provider = reranker_mod.CrossEncoderProvider.__new__(reranker_mod.CrossEncoderProvider)
    provider._model = SimpleNamespace(model=SimpleNamespace(model=None))  # noqa: SLF001 - lazy_load: no session yet
    assert provider.model_loaded() is False
    provider._model.model.model = object()  # noqa: SLF001
    assert provider.model_loaded() is True
    provider._model = SimpleNamespace()  # noqa: SLF001 - the structure is not what we expect: cannot tell, so not loaded
    assert provider.model_loaded() is False
    assert 'getattr(inner, "model", None) is None' in inspect.getsource(reranker_mod.CrossEncoderProvider._pair_tokenizer)  # noqa: SLF001
    assert 'getattr(self._model.model, "model", None) is not None' in inspect.getsource(
        reranker_mod.CrossEncoderProvider.model_loaded
    )
    assert reranker_mod.FakeRerankerProvider(scores={}).model_loaded() is True


def test_an_unexpected_failure_in_the_check_blocks_and_warns_once_and_never_raises(monkeypatch, caplog) -> None:
    _headroom(monkeypatch, 1e12)

    def _boom(model_id: str):
        raise RuntimeError("registry exploded")

    monkeypatch.setattr(reranker_mod, "_runtime_model_file_bytes", _boom)
    with caplog.at_level(logging.WARNING, logger=reranker_mod.log.name):
        for _ in range(3):
            assert reranker_mod.model_load_blocked_by_memory() is True
    assert len([r for r in caplog.records if "free-memory check before the model load failed" in r.getMessage()]) == 1


# ---- the RAM lull job ----


def _build_spy(monkeypatch, provider) -> list[int]:
    built: list[int] = []
    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kw: built.append(1) or provider)
    return built


def test_the_ram_job_builds_and_loads_nothing_below_the_file_size_and_retries_at_the_next_lull(store, monkeypatch) -> None:
    provider = ScriptedLadderProvider(model=_MODEL)
    built = _build_spy(monkeypatch, provider)
    _file_size(monkeypatch, 5000.0)
    _headroom(monkeypatch, 9999.0)  # S257: below the file size plus the first rung (the file size again, no fit)
    reranker_mod.note_ram_figure_needed(_MODEL)
    assert reranker_mod.run_ram_figure("m1", store) is False
    assert built == [] and provider.plain == [] and provider.timed == [] and provider.probes == 0
    assert not reranker_mod._ram_shapes.shape_groups(store, _MODEL)  # noqa: SLF001
    assert _MODEL in reranker_mod._ram_figure_needed, "the need flag is kept"  # noqa: SLF001
    assert reranker_mod.ram_figure_due("m1", lambda: _open(store)) is False, "once per chat activity marker"
    assert reranker_mod.ram_figure_due("m2", lambda: _open(store)) is True, "the next lull retries"
    _headroom(monkeypatch, 10000.0)  # exactly the bound
    reranker_mod.run_ram_figure("m2", store)
    assert built == [1]


def _open(store):
    from contextlib import nullcontext

    return nullcontext(store)


def test_the_ram_job_asks_nothing_of_a_loaded_provider_and_the_first_rung_still_applies(store, monkeypatch) -> None:
    provider = _Resident(model=_MODEL)
    reranker_mod._provider_cache[_MODEL] = provider  # noqa: SLF001
    built = _build_spy(monkeypatch, provider)
    _file_size(monkeypatch, 5000.0)
    _headroom(monkeypatch, 4999.0)  # below the file size: the load check would refuse, a loaded model skips it
    reranker_mod.run_ram_figure("m1", store)
    assert built == [1]
    assert provider.timed == [], "the first rung's own bound (the file size, no fit) still holds"


# ---- the floor job ----


def _floor_env(monkeypatch):
    monkeypatch.setattr(floor_startup, "reranker_model_id", lambda: _FLOOR_MODEL)
    built: list[int] = []
    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kw: built.append(1) or ScriptedLadderProvider())
    computed: list[str] = []
    monkeypatch.setattr(
        floor_calibration, "get_bootstrap_floor", lambda model_id: computed.append(model_id) or {"floor": 1.0}
    )
    return built, computed


def test_the_floor_job_builds_nothing_below_the_file_size_records_a_failed_attempt_and_retries(monkeypatch) -> None:
    built, computed = _floor_env(monkeypatch)
    _file_size(monkeypatch, 5000.0)
    _headroom(monkeypatch, 9999.0)  # S257: file size plus the first rung (the file size again, no fit)
    assert floor_startup.run_rerank_floor(activity_marker=1.0) is None
    assert built == [] and computed == []
    assert floor_calibration.bootstrap_failed("rerank", _FLOOR_MODEL)
    assert not floor_calibration.rerank_bootstrap_due(_FLOOR_MODEL, activity_marker=1.0), "same marker: not again"
    assert floor_calibration.rerank_bootstrap_due(_FLOOR_MODEL, activity_marker=2.0), "the next lull retries it"
    _headroom(monkeypatch, 10000.0)  # exactly the bound
    assert floor_startup.run_rerank_floor(activity_marker=2.0) == {"floor": 1.0}
    assert built == [1] and computed == [_FLOOR_MODEL]
    assert not floor_calibration.bootstrap_failed("rerank", _FLOOR_MODEL)


def test_the_floor_job_builds_nothing_when_free_memory_cannot_be_read(monkeypatch) -> None:
    """S261: no model load for the floor job either; a failed attempt, retried at the next lull."""
    built, computed = _floor_env(monkeypatch)
    _file_size(monkeypatch, 5000.0)
    _headroom(monkeypatch, None)
    assert floor_startup.run_rerank_floor(activity_marker=1.0) is None
    assert built == [] and computed == []
    assert floor_calibration.bootstrap_failed("rerank", _FLOOR_MODEL)


def test_the_floor_job_asks_nothing_of_a_loaded_provider(monkeypatch) -> None:
    built, computed = _floor_env(monkeypatch)
    _file_size(monkeypatch, 5000.0)
    _headroom(monkeypatch, 1.0)
    reranker_mod._provider_cache[_MODEL] = _Resident(model=_MODEL)  # noqa: SLF001
    assert floor_startup.run_rerank_floor(activity_marker=1.0) == {"floor": 1.0}
    assert built == [1] and computed == [_FLOOR_MODEL]


def test_the_floor_job_never_raises_when_the_check_itself_does(monkeypatch) -> None:
    built, computed = _floor_env(monkeypatch)

    def _boom():
        raise RuntimeError("check exploded")

    monkeypatch.setattr(reranker_mod, "model_load_blocked_by_memory", _boom)
    assert floor_startup.run_rerank_floor(activity_marker=1.0) is None
    assert built == [] and computed == []
    assert floor_calibration.bootstrap_failed("rerank", _FLOOR_MODEL)


# ---- the ladder's first rung ----


def _run_pass(store, monkeypatch) -> ScriptedLadderProvider:
    provider = ScriptedLadderProvider(model=_MODEL)
    reranker_mod.measure_ram_figure(provider, store)
    return provider


@pytest.mark.parametrize(("headroom", "ran"), [(4999.0, False), (5000.0, True), (9999.0, True)])
def test_the_first_rung_needs_the_file_size_when_no_fit_is_kept(store, monkeypatch, headroom, ran) -> None:
    _file_size(monkeypatch, 5000.0)
    _headroom(monkeypatch, headroom)
    provider = _run_pass(store, monkeypatch)
    assert (provider.timed[:1] == [(1, FULL)]) is ran
    if not ran:
        assert provider.timed == []
        assert not reranker_mod._ram_shapes.shape_groups(store, _MODEL), "nothing recorded"  # noqa: SLF001
        assert reranker_mod._window_ram_fit(store, _MODEL) is None  # noqa: SLF001 - no marker, no model


@pytest.mark.parametrize(("headroom", "ran"), [(999.0, False), (1000.0, True)])
def test_the_first_rung_needs_the_kept_fits_one_pair_prediction_not_the_file_size(store, monkeypatch, headroom, ran) -> None:
    _kept_fit(store, 1000.0)
    _file_size(monkeypatch, 1e12)  # a huge file size must not be used when a fit is kept
    _headroom(monkeypatch, headroom)
    provider = _run_pass(store, monkeypatch)
    assert (provider.timed[:1] == [(1, FULL)]) is ran


@pytest.mark.parametrize(("delta", "ran"), [(-1.0, False), (0.0, True), (1.0, True)])
def test_the_first_rung_needs_the_kept_fits_prediction_times_the_safety_factor(store, monkeypatch, delta, ran) -> None:
    """S255: the one-pair prediction is scaled; the file size (no fit kept) is a size and is not."""
    monkeypatch.setattr(dev_constants, "RAM_FIT_SAFETY_FACTOR", _FACTOR)
    _kept_fit(store, 1000.0)
    _file_size(monkeypatch, 1e12)
    _headroom(monkeypatch, 1000.0 * _FACTOR + delta)
    assert (_run_pass(store, monkeypatch).timed[:1] == [(1, FULL)]) is ran


def test_the_first_rung_does_not_scale_the_file_size_when_no_fit_is_kept(store, monkeypatch) -> None:
    monkeypatch.setattr(dev_constants, "RAM_FIT_SAFETY_FACTOR", _FACTOR)
    _file_size(monkeypatch, 5000.0)
    _headroom(monkeypatch, 5000.0)
    assert _run_pass(store, monkeypatch).timed[:1] == [(1, FULL)]


def test_the_first_rung_asks_for_the_runtime_model_id_and_only_when_no_fit_is_kept(store, monkeypatch) -> None:
    asked: list[str] = []
    _file_size(monkeypatch, 5000.0, asked)
    _headroom(monkeypatch, 1e12)
    _run_pass(store, monkeypatch)
    assert asked == [_MODEL]


def test_an_undetermined_file_size_stops_the_first_rung_only_where_no_fit_is_kept(store, monkeypatch) -> None:
    _file_size(monkeypatch, None)
    _headroom(monkeypatch, 1e12)
    assert _run_pass(store, monkeypatch).timed == []
    _kept_fit(store, 1000.0)
    assert _run_pass(store, monkeypatch).timed[:1] == [(1, FULL)]


def test_unreadable_free_memory_runs_no_rung_and_looks_up_no_size(store, monkeypatch) -> None:
    """S261 (was: runs the first rung without a size lookup)."""
    sizes: list[str] = []
    _file_size(monkeypatch, None, sizes)
    _headroom(monkeypatch, None)
    assert _run_pass(store, monkeypatch).timed == []
    assert sizes == []


def test_a_rung_that_a_measured_shape_bounds_does_not_use_the_first_rung_bound(store, monkeypatch) -> None:
    sizes: list[str] = []
    _file_size(monkeypatch, 1e12, sizes)  # would refuse everything if it were consulted
    reranker_mod._record_rerank_cost(  # noqa: SLF001 - the first rung is already measured
        _MODEL, store, FULL, 0.1, 1e6, pairs=1, longest=FULL, full_length=True
    )
    _headroom(monkeypatch, 5e6)
    provider = _run_pass(store, monkeypatch)
    assert sizes == [], "later rungs are bounded by the shapes measured (S245), not by the file size"
    assert provider.timed[:1] == [(1, 512)]


def test_a_host_between_the_file_size_and_twice_it_never_loads_the_model(store, monkeypatch) -> None:
    """S257 closes the R16 dead zone (owner question Q1): with no fit the load needs the file size plus the first
    rung's bound (the file size again), so a host with free memory in [F, 2F) builds and loads nothing and holds no
    unused model; every lull retries and the cosine path serves."""
    provider = ScriptedLadderProvider(model=_MODEL)
    built = _build_spy(monkeypatch, provider)
    _file_size(monkeypatch, 5000.0)
    _headroom(monkeypatch, 7500.0)
    reranker_mod.note_ram_figure_needed(_MODEL)
    assert reranker_mod.run_ram_figure("m1", store) is False
    assert built == [] and provider.plain == [] and provider.timed == []
    assert _MODEL in reranker_mod._ram_figure_needed  # noqa: SLF001
    assert reranker_mod.ram_figure_due("m2", lambda: _open(store)) is True, "retried at the next lull"
    assert reranker_mod.run_ram_figure("m2", store) is False and built == []

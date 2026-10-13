"""S259: while the reranker model is not loaded yet, the reply-path batch check (`_batch_fits_ram_bound`) counts the
model load too: before any tokenizing the free memory must cover the S257 load bound (file size plus the first
rung's bound), and once the batch's size is known the file size plus its predicted growth (times the S255 factor)
must fit the free memory read before the load; otherwise nothing is tokenized (so nothing is loaded) and the turn
takes the cosine path ("sizes" hand-off). A loaded model adds nothing. Scripted providers whose model loads at the
first tokenizer use, the real `CrossEncoderProvider` structure over a fake session; patched free-memory and file-size
readings. No model."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain import dev_constants
from brain.memory import cost_series, ram_fit
from brain.memory import reranker as reranker_mod
from brain.memory.store import MemoryStore
from tests.unit.brain.memory._ram_support import FULL, ScriptedLadderProvider

_MODEL = "scripted-ladder"
_FILE = 5000.0
_PER_TOKEN = 1000.0  # the kept fit: 1000 bytes per padded token, no intercept
_FIRST_RUNG = _PER_TOKEN * FULL  # one pair at the full length
_LOAD_BOUND = _FILE + _FIRST_RUNG
_FACTOR = dev_constants.RAM_FIT_SAFETY_FACTOR  # the shipped S255 value, read at import (the suite default is 1.0)
_DOC = "x" * 400  # 100 scripted tokens with an empty query


class _Lazy(ScriptedLadderProvider):
    """The model loads at the first tokenizer use or rerank, like the real provider (lazy_load)."""

    def __init__(self, *, loaded: bool = False) -> None:
        super().__init__(model=_MODEL)
        self.loaded = loaded
        self.loads = 0

    def _load(self) -> None:
        if not self.loaded:
            self.loaded = True
            self.loads += 1

    def model_loaded(self) -> bool:
        return self.loaded

    def pair_token_lengths(self, query, documents):
        self._load()
        return super().pair_token_lengths(query, documents)

    def rerank(self, query, documents):
        self._load()
        return super().rerank(query, documents)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    monkeypatch.setattr(reranker_mod, "_require_ram_figure", True)
    monkeypatch.setattr(reranker_mod.peak_rss, "available", lambda: True)
    monkeypatch.setattr(reranker_mod, "resolve_reranker_model_id", lambda: _MODEL)
    asked: list[str] = []

    def size(model_id: str):
        asked.append(model_id)
        return _FILE

    monkeypatch.setattr(reranker_mod, "_runtime_model_file_bytes", size)
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: 1e15)
    return asked


@pytest.fixture
def store(tmp_path: Path):
    s = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    yield s
    s.close()


def _keep_fit(store=None) -> ram_fit.RamFit:
    fit = ram_fit.RamFit(a=0.0, b=_PER_TOKEN, c=0.0, shapes=5, samples=5, max_pairs=58, max_length=FULL)
    key = cost_series._store_key(store) if store is not None else "k"  # noqa: SLF001
    with reranker_mod._cost_lock:  # noqa: SLF001
        reranker_mod._ram_fit_kept[(key, _MODEL)] = fit  # noqa: SLF001
    return fit


def _headroom(monkeypatch, value) -> None:
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: value)


def _check(provider, docs=None) -> bool:
    return reranker_mod._batch_fits_ram_bound(provider, _MODEL, "", docs or [_DOC])  # noqa: SLF001


def test_below_the_load_bound_nothing_is_tokenized_and_so_nothing_is_loaded(monkeypatch, _env) -> None:
    _keep_fit()
    provider = _Lazy()
    _headroom(monkeypatch, _LOAD_BOUND - 1.0)
    assert _check(provider) is False
    assert provider.probes == 0 and provider.loads == 0 and provider.loaded is False
    assert _env == [_MODEL], "the size was asked for the model id, once"


def test_at_the_load_bound_a_one_pair_batch_loads_and_fits_because_its_growth_cannot_exceed_the_first_rung(monkeypatch) -> None:
    _keep_fit()
    provider = _Lazy()
    _headroom(monkeypatch, _LOAD_BOUND)
    assert _check(provider) is True
    assert provider.loads == 1 and provider.probes == 1


def test_the_file_size_is_added_to_the_batchs_growth_against_the_free_memory_read_before_the_load(monkeypatch) -> None:
    """A many-pair batch can need more than the first rung: it passes the load bound, loads, and is then judged on
    file size + growth. The free memory used is the reading taken BEFORE the load (the later reading would already
    lack the model)."""
    _keep_fit()
    docs = [_DOC] * 20  # 20 pairs x 100 tokens
    growth = 20 * 100 * _PER_TOKEN
    assert growth > _FIRST_RUNG
    readings = iter([_FILE + growth, 0.0, 0.0])  # a second reading would be worthless; only the first may be used
    monkeypatch.setattr(reranker_mod, "_available_ram_headroom_bytes", lambda: next(readings, 0.0))
    assert _check(_Lazy(), docs) is True
    _headroom(monkeypatch, _FILE + growth - 1.0)
    refused = _Lazy()
    assert _check(refused, docs) is False
    assert refused.loads == 1, "disclosed: the batch's size is known only after the tokenizer probe, which loads the model"


def test_a_loaded_model_adds_nothing_and_no_size_is_looked_up(monkeypatch, _env) -> None:
    _keep_fit()
    provider = _Lazy(loaded=True)
    growth = 100 * _PER_TOKEN
    _headroom(monkeypatch, growth)
    assert _check(provider) is True
    _headroom(monkeypatch, growth - 1.0)
    assert _check(provider) is False
    assert _env == []


def test_the_growth_is_scaled_by_the_safety_factor_and_the_file_size_is_not(monkeypatch) -> None:
    monkeypatch.setattr(dev_constants, "RAM_FIT_SAFETY_FACTOR", _FACTOR)
    _keep_fit()
    growth = 100 * _PER_TOKEN
    need = _FILE + growth * _FACTOR
    _headroom(monkeypatch, 1e12)  # (the first-rung part is scaled too: far below)
    assert _check(_Lazy()) is True
    _headroom(monkeypatch, max(need, _LOAD_BOUND * _FACTOR))
    assert _check(_Lazy()) is True
    loaded = _Lazy(loaded=True)
    _headroom(monkeypatch, growth * _FACTOR)
    assert _check(loaded) is True
    _headroom(monkeypatch, growth * _FACTOR - 1.0)
    assert _check(loaded) is False


def test_an_undetermined_file_size_refuses_without_tokenizing(monkeypatch) -> None:
    _keep_fit()
    monkeypatch.setattr(reranker_mod, "_runtime_model_file_bytes", lambda model_id: None)
    provider = _Lazy()
    assert _check(provider) is False
    assert provider.probes == 0 and provider.loads == 0


def test_unreadable_free_memory_or_no_applicable_bound_passes_without_a_size_lookup_or_tokenizing(monkeypatch, _env) -> None:
    _keep_fit()
    _headroom(monkeypatch, None)
    provider = _Lazy()
    assert _check(provider) is True
    assert provider.probes == 0 and _env == []
    _headroom(monkeypatch, 1.0)
    monkeypatch.setattr(reranker_mod.peak_rss, "available", lambda: False)
    assert _check(provider) is True
    assert provider.probes == 0 and _env == []


def test_a_process_with_no_memory_model_still_waits(monkeypatch) -> None:
    assert _check(_Lazy()) is False


def test_a_provider_that_cannot_say_whether_it_is_loaded_counts_as_not_loaded(monkeypatch) -> None:
    _keep_fit()

    class _Opaque(_Lazy):
        def model_loaded(self):
            raise RuntimeError("cannot tell")

    provider = _Opaque()
    _headroom(monkeypatch, _LOAD_BOUND - 1.0)
    assert _check(provider) is False and provider.probes == 0


def test_a_refusal_logs_once_per_condition_until_the_check_passes(monkeypatch, caplog) -> None:
    _keep_fit()
    with caplog.at_level(logging.INFO, logger=reranker_mod.log.name):
        _headroom(monkeypatch, 10.0)
        for _ in range(3):
            assert _check(_Lazy()) is False
        assert len([r for r in caplog.records if "(S259)" in r.getMessage()]) == 1
        _headroom(monkeypatch, 1e12)
        assert _check(_Lazy()) is True
        _headroom(monkeypatch, 10.0)
        assert _check(_Lazy()) is False
        assert len([r for r in caplog.records if "(S259)" in r.getMessage()]) == 2


# ---- the real provider's structure: a refusal never builds the session ----


class _FakeInner:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.loads = 0

    def load_onnx_model(self) -> None:
        self.loads += 1
        self.model = object()
        self.tokenizer = SimpleNamespace(
            encode_batch=lambda pairs: [SimpleNamespace(attention_mask=[1] * 100) for _ in pairs]
        )


def _real_provider() -> tuple[reranker_mod.CrossEncoderProvider, _FakeInner]:
    provider = reranker_mod.CrossEncoderProvider.__new__(reranker_mod.CrossEncoderProvider)
    inner = _FakeInner()
    provider._model = SimpleNamespace(model=inner)  # noqa: SLF001
    provider._model_id = _MODEL  # noqa: SLF001
    provider._rerank_lock = threading.Lock()  # noqa: SLF001
    return provider, inner


def test_the_real_providers_session_is_not_built_by_a_refused_check_and_is_built_once_by_a_passing_one(monkeypatch) -> None:
    _keep_fit()
    provider, inner = _real_provider()
    _headroom(monkeypatch, _LOAD_BOUND - 1.0)
    assert reranker_mod._batch_fits_ram_bound(provider, _MODEL, "", [_DOC]) is False  # noqa: SLF001
    assert inner.loads == 0 and provider.model_loaded() is False
    _headroom(monkeypatch, _LOAD_BOUND)
    assert reranker_mod._batch_fits_ram_bound(provider, _MODEL, "", [_DOC]) is True  # noqa: SLF001
    assert inner.loads == 1 and provider.model_loaded() is True
    _headroom(monkeypatch, 100 * _PER_TOKEN)  # loaded now: only the growth counts
    assert reranker_mod._batch_fits_ram_bound(provider, _MODEL, "", [_DOC]) is True  # noqa: SLF001
    assert inner.loads == 1


# ---- the warm-up gate and the two entry points ----

_QUERY = "which name did we settle on for the dog"


def _docs(n: int) -> list[str]:
    return [f"{i} " + "word " * 600 for i in range(n)]  # about 750 scripted tokens each


def test_a_refused_warm_up_of_an_unloaded_model_says_stop_and_marks_nothing(monkeypatch) -> None:
    _keep_fit()
    provider = _Lazy()
    _headroom(monkeypatch, _LOAD_BOUND - 1.0)
    assert reranker_mod._warm_up_once(provider, _QUERY, _docs(3), bounded=True) is False  # noqa: SLF001
    assert provider.plain == [] and provider.probes == 0 and provider.loads == 0
    assert _MODEL not in reranker_mod._warmed_model_ids  # noqa: SLF001


def test_a_refused_warm_up_of_a_loaded_model_goes_on_and_a_passing_one_warms_up(monkeypatch) -> None:
    _keep_fit()
    loaded = _Lazy(loaded=True)
    _headroom(monkeypatch, 1.0)
    assert reranker_mod._warm_up_once(loaded, _QUERY, _docs(3), bounded=True) is True  # noqa: SLF001
    assert loaded.plain == [] and _MODEL not in reranker_mod._warmed_model_ids  # noqa: SLF001
    _headroom(monkeypatch, 1e12)
    fresh = _Lazy()
    assert reranker_mod._warm_up_once(fresh, _QUERY, _docs(3), bounded=True) is True  # noqa: SLF001
    assert len(fresh.plain) == 2 and _MODEL in reranker_mod._warmed_model_ids  # noqa: SLF001


def test_the_ladders_unbounded_warm_up_and_a_warmed_model_are_never_stopped(monkeypatch) -> None:
    provider = _Lazy()
    _headroom(monkeypatch, 1.0)
    assert reranker_mod._warm_up_once(provider, "warm up", ["warm up"]) is True  # noqa: SLF001
    assert _MODEL in reranker_mod._warmed_model_ids  # noqa: SLF001
    assert reranker_mod._warm_up_once(_Lazy(), _QUERY, _docs(3), bounded=True) is True  # noqa: SLF001


def test_a_turn_whose_model_may_not_load_probes_no_sizes_and_takes_the_no_rerank_path(store, monkeypatch) -> None:
    _keep_fit(store)
    provider = _Lazy()
    docs = _docs(10)
    _headroom(monkeypatch, _LOAD_BOUND - 1.0)
    reranker_mod.warm_up_for_recall(provider, _QUERY, docs)  # still returns nothing
    sizes = reranker_mod.recall_pair_sizes(provider, _QUERY, docs)
    assert sizes.lengths is None and sizes.seconds is None
    out = reranker_mod.rerank_for_recall(provider, _QUERY, docs, store=store, budget_seconds=4.0, sizes=sizes)
    assert (out.reranked, out.hand_off, out.width, out.normalization) == (False, "sizes", 0, None)
    out = reranker_mod.rerank_for_recall(provider, _QUERY, docs, store=store, budget_seconds=4.0)
    assert (out.reranked, out.hand_off, out.width) == (False, "sizes", 0)
    assert provider.probes == 0 and provider.loads == 0 and provider.plain == [] and provider.timed == []
    assert reranker_mod._budget_misses.get(_MODEL) is None, "a memory refusal is not a time-stuck miss"  # noqa: SLF001
    _headroom(monkeypatch, 1e15)  # memory frees up: the next turn loads, warms up and probes
    sizes = reranker_mod.recall_pair_sizes(provider, _QUERY, docs)
    assert sizes.lengths is not None and len(sizes.lengths) == len(docs) + reranker_mod.P
    assert provider.loads == 1 and len(provider.plain) == 2


def test_a_turn_whose_model_is_loaded_still_probes_its_sizes_after_a_refused_warm_up(store, monkeypatch) -> None:
    _keep_fit(store)
    provider = _Lazy(loaded=True)
    docs = _docs(10)
    _headroom(monkeypatch, 1.0)
    sizes = reranker_mod.recall_pair_sizes(provider, _QUERY, docs)
    assert sizes.lengths is not None and len(sizes.lengths) == len(docs) + reranker_mod.P
    assert provider.probes == 2, "one inside the refused one-pair gate, one for the sizes"
    assert provider.plain == [], "no warm-up call ran"


def test_the_floor_calibrations_bundled_normalization_defers_before_loading_an_unloaded_model(monkeypatch) -> None:
    _keep_fit()
    bundle = [("which name did we settle on", "we settled on a name for the dog"), ("quiet evening", "the printer jammed again")]
    provider = _Lazy()
    _headroom(monkeypatch, _LOAD_BOUND - 1.0)
    with pytest.raises(reranker_mod.RerankDeferred):
        reranker_mod.normalize_bundled_pairs_against_anchors(provider, bundle)
    assert provider.probes == 0 and provider.loads == 0 and provider.plain == []
    _headroom(monkeypatch, 1e15)
    reranker_mod.normalize_bundled_pairs_against_anchors(provider, bundle)
    assert provider.loads == 1 and len(provider.plain) == len(bundle)

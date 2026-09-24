"""Embedding provider abstraction.

Provider interface: EmbeddingProvider ABC. Two concrete providers:
- FakeEmbeddingProvider: deterministic hash-based, zero network, used in tests.
- FastEmbedProvider: real local embeddings via `fastembed` (ONNX, no torch,
  no network at inference — the model file is downloaded once into the
  shared cache dir and used offline after). Production default.

`build_embedding_provider()` caches the constructed provider PROCESS-WIDE
(keyed by model_id, thread-safe) — see its own docstring — so the expensive
model/ONNX-session load happens once per process, not once per call.

F1 (#259) increment 8: the old SQLite content-hash cache (`EmbeddingCache`,
`embeddings.db`) that used to sit in front of this provider abstraction is
REMOVED — every memory's embedding now lives on its own `memories` row
(`embedding`/`embedding_model_id` columns), and the one remaining transient
use (the per-recall query embed) calls `build_embedding_provider().embed()`
directly. The content-hash keying it used (`hash_content`) is removed too —
nothing keys off content-hash anymore (memories are keyed by row id); the
one other consumer, clustering's content-hash-keyed `MemoryClusterStore`,
was dead since increment 4 and is removed in this same increment.
`cosine_similarity` is retained (still used by dedupe/semantic recall).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_DIM = 256


class EmbeddingProvider(ABC):
    """Abstract embedding provider. Subclasses implement `embed`, `embedding_dim`
    and `model_id`."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a 1-D numpy array of dimension `embedding_dim()`."""

    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the output dimension of vectors this provider produces."""

    @abstractmethod
    def model_id(self) -> str:
        """Return a stable identifier for the model producing these vectors.

        Stored alongside every embedded row (`memories.embedding_model_id`)
        so a provider swap is a targeted invalidation — a vector made by one
        model/dim is never read back as if it came from another. Two
        providers that produce incompatible vectors MUST return different
        ids (dimension alone is not a safe proxy: two different models can
        share a dimension).
        """


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic pseudo-random embedding provider for tests.

    Uses SHA-256 of the input text to seed a NumPy Generator, then produces
    a unit-norm vector. Same text always produces the same vector; different
    text produces different vectors. No network, no external dependencies.
    """

    def __init__(self, dim: int = _DEFAULT_DIM) -> None:
        self._dim = dim

    def embed(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(seed=seed)
        vec = rng.standard_normal(self._dim)
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            raise ValueError(f"FakeEmbeddingProvider produced a zero-norm vector (dim={self._dim})")
        return vec / norm

    def embedding_dim(self) -> int:
        return self._dim

    def model_id(self) -> str:
        # Dim-qualified so two Fake instances of different dims (seen across
        # the test suite) never collide on the same cache rows.
        return f"fake-{self._dim}"


def _materialize_symlinked_files(directory: Path) -> Path:
    """Return a directory holding REAL (non-symlinked) files for everything
    fastembed downloaded into `directory`, materializing them if needed.

    WHY (#259 F1 model-swap, onnxruntime external-data workaround):
    huggingface_hub's default cache layout (`snapshot_download`) stores the
    actual file content once under `<cache>/models--org--repo/blobs/<hash>`
    and creates the SNAPSHOT directory fastembed actually reads from
    (`directory` here) as a directory of SYMLINKS into that `blobs/` dir. For
    a model that ships "external data" weights — a `model.onnx` file that
    references a separate `model.onnx_data` shard, e.g.
    `intfloat/multilingual-e5-large` — onnxruntime>=1.24.1's security check
    resolves the external-data reference relative to the model file's
    directory and rejects it if the resolved (symlink-realpath) location
    "escapes" that directory. Under the blobs/snapshots layout it always
    does, because `model.onnx` and `model.onnx_data` are two INDEPENDENT
    symlinks that can resolve into different blob paths — loading raises
    "External data path escapes model directory" even though the files are
    entirely legitimate. Spike-confirmed fix: materialize the snapshot as
    real, non-symlinked files sitting together in one directory; onnxruntime
    then loads it cleanly.

    Fast path: if `directory` contains no symlinks (already-real files, e.g.
    a fresh download under `HF_HUB_DISABLE_SYMLINKS`, or materialized on a
    prior run), returns `directory` unchanged — this function costs nothing
    in the common case.

    Writes are hardlink-or-copy into a temp path then `os.replace` (atomic on
    POSIX) into place, so a second process/instance racing to materialize the
    same directory concurrently can't observe a half-written file.

    Scope: this only materializes symlinked FILES, not symlinked directories.
    HF's cache always symlinks individual files, never whole directories, so
    that is not a live gap for this workaround's use.
    """
    entries = [p for p in directory.rglob("*") if p.is_file()]
    if not any(p.is_symlink() for p in entries):
        return directory

    materialized = directory.parent / f"{directory.name}.materialized"
    for entry in entries:
        rel = entry.relative_to(directory)
        dest = materialized / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        real_src = entry.resolve()  # follow symlink(s) to the real blob
        if dest.exists() and not dest.is_symlink() and os.path.samefile(dest, real_src):
            continue  # already a hardlink to real_src, from this or a prior process run
        if dest.exists() and not dest.is_symlink():
            dest.unlink()  # stale real file (different inode): re-materialize, don't trust size
        tmp_dest = dest.with_name(dest.name + f".tmp{os.getpid()}")
        try:
            if tmp_dest.exists():
                tmp_dest.unlink()
            os.link(real_src, tmp_dest)  # same filesystem: instant, no extra disk
        except OSError:
            shutil.copy2(real_src, tmp_dest)  # cross-filesystem fallback
        os.replace(tmp_dest, dest)
    return materialized


def _materialize_fastembed_model_dir(text_embedding: Any) -> None:
    """After constructing a fastembed `TextEmbedding`, repoint its internal
    model directory at a REAL-file materialization if the download left
    symlinks in place (see `_materialize_symlinked_files` for the "why").

    Reaches into fastembed's undocumented internals (`TextEmbedding.model.
    _model_dir`) because fastembed exposes no public hook for this — defensive
    by design: if a fastembed upgrade renames/removes either attribute, this
    logs a loud warning and no-ops rather than crashing provider construction.
    """
    inner = getattr(text_embedding, "model", None)
    model_dir = getattr(inner, "_model_dir", None)
    if inner is None or model_dir is None:
        logger.warning(
            "FastEmbedProvider: could not locate fastembed's internal model "
            "directory to apply the onnxruntime external-data symlink "
            "workaround (TextEmbedding.model._model_dir is missing, likely "
            "renamed by a fastembed upgrade). A model with sharded "
            "external-data weights (e.g. intfloat/multilingual-e5-large) may "
            "fail to load with 'External data path escapes model directory' "
            "if the HuggingFace cache for it is symlinked."
        )
        return
    real_dir = _materialize_symlinked_files(Path(model_dir))
    if real_dir != Path(model_dir):
        inner._model_dir = real_dir  # noqa: SLF001 — the documented workaround target


class FastEmbedProvider(EmbeddingProvider):
    """Real local embedding provider via `fastembed` (ONNX runtime, no torch).

    Production default. Model id comes from `model_tier.py`
    (`model_for_tier(TIER_EMBEDDING)`), never hardcoded here — see that
    module's docstring for why every model selection routes through it.

    The model file is downloaded once (fastembed's own download-on-
    construction behavior — see below) into `cache_dir` and used fully
    offline after: no network call happens at embed() time once the file is
    cached. Construction DOES perform that (one-time) download eagerly —
    fastembed's `lazy_load` only defers building the ONNX inference session
    itself to the first `embed()` call, not the file download (confirmed
    against the installed fastembed's `OnnxTextEmbedding.__init__`, which
    calls `download_model()` unconditionally before its own `lazy_load`
    check). Every current call site constructs this provider off the message
    hot path already, so paying the download cost at construction time (when
    the file isn't yet cached) is acceptable; only the ONNX session build is
    deferred, keeping steady-state (already-cached) construction cheap.

    ONNXRUNTIME EXTERNAL-DATA SYMLINK WORKAROUND (#259 F1 model-swap): a
    model that ships sharded "external data" weights (a `model.onnx` that
    references a separate `model.onnx_data` shard — e.g.
    `intfloat/multilingual-e5-large`) fails to load under
    onnxruntime>=1.24.1's external-data security check via HuggingFace's
    default symlink cache layout: `model.onnx` and `model.onnx_data` are two
    INDEPENDENT symlinks that can resolve to different `blobs/` paths, which
    onnxruntime treats as the data "escaping" the model directory
    ("External data path escapes model directory"). The fix (spike-verified)
    is applied around the `TextEmbedding(...)` construction below — see the
    comments there and `_materialize_symlinked_files`.

    DIMENSION (#259 inc7 red-team F1 — the one-touch-swap fix): `dim` is a
    DECLARED sanity value (production passes `model_tier.MODEL_EMBEDDING_DIM`)
    used ONLY for the loud mismatch log below — it is never what
    `embedding_dim()` returns. The REAL dimension is established by probing
    the loaded model with one actual `embed()` call (see `embedding_dim()`),
    so a `MODEL_EMBEDDING` swap to a different-dim model works correctly even
    if `MODEL_EMBEDDING_DIM` is never updated to match. Before this fix,
    `embedding_dim()` just echoed `dim` back — a silent no-op check that made
    a stale constant indistinguishable from a correct one.
    """

    # Short, fixed text used to probe the model's real output dimension on
    # first use — never persisted, never cached, just measures `len(vector)`.
    _PROBE_TEXT = "probe"

    def __init__(self, model_id: str, cache_dir: str | Path, dim: int | None = None) -> None:
        # Imported lazily so importing this module never requires fastembed/
        # onnxruntime to be installed unless the real provider is actually
        # constructed (tests exclusively use FakeEmbeddingProvider).
        #
        # onnxruntime external-data symlink WORKAROUND (#259 F1 model-swap,
        # class docstring above has the full "why"): force huggingface_hub to
        # write REAL files instead of its default blobs+symlinks cache layout
        # for any download the `TextEmbedding(...)` construction below
        # triggers. Set BEFORE importing fastembed (which transitively
        # imports huggingface_hub) so a fresh download never creates the
        # symlink layout in the first place.
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        # Belt-and-suspenders: huggingface_hub actually reads this setting
        # from a MODULE ATTRIBUTE (`huggingface_hub.constants.
        # HF_HUB_DISABLE_SYMLINKS`), re-read from the module namespace at
        # every call site (confirmed against the installed huggingface_hub's
        # `file_download.py`) — NOT a value frozen once at import time — so
        # patching it directly here is effective regardless of whether
        # something else in this process already imported huggingface_hub
        # before the env var above was set (which would otherwise leave that
        # earlier `os.environ.get(...)` read stale/too-late).
        import huggingface_hub.constants as _hf_hub_constants
        from fastembed import TextEmbedding

        _hf_hub_constants.HF_HUB_DISABLE_SYMLINKS = True

        self._model_id = model_id
        # DECLARED dim (a sanity value, e.g. MODEL_EMBEDDING_DIM) — compared
        # against the REAL probed dim on first use, never returned directly.
        # None is valid (no declared value to check against; the real dim is
        # still established on first use).
        self._declared_dim = dim
        # The REAL dim, established lazily from an actual embed() call (see
        # `embed`/`embedding_dim`). None until the first embed happens.
        self._real_dim: int | None = None
        # lazy_load=True: defer building the ONNX inference SESSION to the
        # first embed() call rather than construction time (the model FILE
        # download itself still happens eagerly, right here, inside
        # TextEmbedding's own __init__ — see the class docstring's "The model
        # file is downloaded once" paragraph). Every current call site
        # constructs this off the message hot path already, so paying the
        # download cost at construction (when not yet cached) is acceptable;
        # deferring the session build keeps steady-state construction cheap.
        self._model = TextEmbedding(model_name=model_id, cache_dir=str(cache_dir), lazy_load=True)
        # Fallback for a cache directory populated by a PRIOR run (before
        # this workaround existed) that may still hold the old symlink
        # layout, or any other path that slips past the disable-symlinks
        # setting above: realpath-resolve and materialize any symlinks left
        # in the downloaded model directory into real files, then repoint the
        # model at that directory. No-op fast path when the directory already
        # holds only real files (the common case after the fix above, or for
        # a model with no external-data shards at all).
        _materialize_fastembed_model_dir(self._model)
        # A shared instance of this provider (see build_embedding_provider's
        # process-wide cache) can have .embed() called concurrently from two
        # threads — the turn thread (recall) and the supervisor's background
        # backfill thread. fastembed's own lazy first-load
        # (`OnnxTextModel._embed_documents`: `if not hasattr(self, "model")
        # or self.model is None: self.load_onnx_model()`) is an unguarded
        # check-then-act with no lock of its own, so two threads racing the
        # FIRST embed() call on one instance can both see `model is None` and
        # both call load_onnx_model() concurrently — a real data race on
        # `self.model`/`self.tokenizer`, not merely a wasted duplicate load.
        # ONNX Runtime's InferenceSession.Run() is documented thread-safe for
        # concurrent inference once a session exists, but that guarantee
        # doesn't cover this lazy-construction race, so the conservative
        # choice is to serialize the whole embed() call (construction +
        # inference) behind one instance lock rather than assume safety we
        # can't confirm for the part that actually races.
        self._embed_lock = threading.Lock()

    def embed(self, text: str) -> np.ndarray:
        # TextEmbedding.embed() takes an iterable and yields one vector per
        # input; we pass exactly one string and take the one result.
        with self._embed_lock:
            (vec,) = self._model.embed([text])
            arr = np.asarray(vec, dtype=np.float32)
            if self._real_dim is None:
                # First real embed this instance has ever performed — this is
                # the "first used" moment the real dim is established from,
                # and the ONE point a stale MODEL_EMBEDDING_DIM gets caught
                # loudly rather than silently (#259 inc7 red-team F1).
                self._real_dim = arr.shape[0]
                if self._declared_dim is not None and self._real_dim != self._declared_dim:
                    logger.error(
                        "FastEmbedProvider: model %s produced dim=%d but the "
                        "declared/sanity dim (model_tier.MODEL_EMBEDDING_DIM) "
                        "is %d — that constant is stale (likely a model swap "
                        "that didn't update it together). This is NOT fatal: "
                        "embed/decode/cluster all follow the REAL dim (%d), "
                        "not the constant. Update MODEL_EMBEDDING_DIM to %d "
                        "to clear this warning.",
                        self._model_id,
                        self._real_dim,
                        self._declared_dim,
                        self._real_dim,
                        self._real_dim,
                    )
        return arr

    def embedding_dim(self) -> int:
        """The REAL output dimension of the loaded model.

        Established by an actual `embed()` call — reused from the first one
        this instance has ever performed, or triggered here via a one-time
        probe embed if none has happened yet. Never the constructor's `dim`
        sanity value (#259 inc7 red-team F1): this is what makes the
        embedding dimension genuinely one-touch-swappable — every consumer
        that asks this provider for its dim gets the model's ACTUAL output
        size, so a `MODEL_EMBEDDING` swap to a different-dim model works
        without also having to edit `MODEL_EMBEDDING_DIM` anywhere else.
        """
        if self._real_dim is None:
            self.embed(self._PROBE_TEXT)
        assert self._real_dim is not None  # embed() always sets it
        return self._real_dim

    def model_id(self) -> str:
        return self._model_id


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two vectors. Range [-1, 1]."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# Process-wide provider cache keyed by model_id (see build_embedding_provider).
# Kept at module scope rather than a closure so `_reset_embedding_provider_cache`
# (test-only) can reach it, and so monkeypatching the *function* (the whole
# object, this cache included — see that fixture) fully controls behavior.
_provider_cache: dict[str, EmbeddingProvider] = {}
_provider_cache_lock = threading.Lock()


def build_embedding_provider() -> EmbeddingProvider:
    """The production embedding provider: FastEmbedProvider pinned to
    `model_tier.TIER_EMBEDDING`'s model id, caching the model file in the
    shared `get_cache_dir()` (one download across every persona on the box,
    per the build-plan recommendation — the model isn't persona-specific
    data, just a local asset).

    The model id comes from `model_tier.py`, never hardcoded here — same
    convention as every Claude tier in that module. `MODEL_EMBEDDING_DIM` is
    passed through too, but ONLY as `FastEmbedProvider`'s declared/sanity dim
    (#259 inc7 red-team F1) — the provider's own `embedding_dim()` derives
    the REAL dim from the loaded model, so this constant going stale after a
    `MODEL_EMBEDDING` swap degrades to a loud log from `FastEmbedProvider`,
    never a silent or load-bearing failure here.

    PROCESS-WIDE CACHING: constructing a FastEmbedProvider builds a real
    fastembed/ONNX inference session — a one-time ~300-450ms cost. Before
    this cache existed, `semantic_recall.run_semantic_recall` called
    `build_embedding_cache()` -> this function fresh on EVERY recall (i.e.
    every conversational turn), so every turn paid that cost again. Now the
    provider for a given model_id is built once per process and reused for
    every subsequent call with that same model_id — steady-state calls pay
    only the per-query embed (tens of ms), not another full reload. Keyed by
    model_id (not a bare singleton) because model_id is the one legitimate
    reason this should ever return a DIFFERENT provider (see model_tier.py);
    in practice there's a single model_id, but keying by it is the robust
    choice if that ever changes.

    THREAD SAFETY: the supervisor's idle embedding backfill runs on a
    background thread while turn-time recall runs on the request/turn
    thread, so two threads can race to build the first provider for a given
    model_id. Guarded with double-checked locking: an unlocked fast-path
    read handles the (overwhelmingly common) already-cached case with no
    lock at all; the lock is only acquired — then re-checked, in case another
    thread won the race while this one was waiting — the first time a given
    model_id actually needs constructing. So the lock is never held across
    concurrent reads of an already-built provider, only across the one real
    construction race. (`FastEmbedProvider.embed()` carries its own separate
    per-instance lock for concurrent `.embed()` calls — see that class.)

    TEST ISOLATION: this dict is process-global, so `tests/conftest.py`'s
    autouse `_reset_embedding_provider_cache` fixture clears it before and
    after every test — needed because a couple of tests in
    `test_embeddings.py` import `build_embedding_provider` by name and call
    the real function directly (bypassing the suite-wide fake-provider
    monkeypatch below, which only intercepts callers that look the function
    up via `brain.memory.embeddings.build_embedding_provider` at call time).
    Every other test goes through that monkeypatch instead:
    `monkeypatch.setattr(embeddings, "build_embedding_provider", lambda: ...)`
    replaces this ENTIRE function object — this cache included — so the fake
    path never reads or writes `_provider_cache` at all, and a cached REAL
    provider can never leak into a test expecting the fake (nor vice versa).
    """
    from brain.bridge.model_tier import MODEL_EMBEDDING_DIM, TIER_EMBEDDING, model_for_tier
    from brain.paths import get_cache_dir

    model_id = model_for_tier(TIER_EMBEDDING)

    provider = _provider_cache.get(model_id)
    if provider is not None:
        return provider

    with _provider_cache_lock:
        provider = _provider_cache.get(model_id)  # re-check: lost the race?
        if provider is not None:
            return provider
        provider = FastEmbedProvider(
            model_id=model_id,
            cache_dir=get_cache_dir(),
            dim=MODEL_EMBEDDING_DIM,
        )
        _provider_cache[model_id] = provider
        return provider


def _reset_embedding_provider_cache() -> None:
    """Test-only: clear the process-level cache `build_embedding_provider()`
    builds up.

    Wired into `tests/conftest.py`'s autouse `_reset_embedding_provider_cache`
    fixture (before AND after every test) so a test that calls the REAL
    `build_embedding_provider()` directly always gets an independently-built
    provider rather than one a prior/later test's call happened to cache
    (see that function's TEST ISOLATION note).
    """
    with _provider_cache_lock:
        _provider_cache.clear()


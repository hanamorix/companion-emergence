"""Real-model validation for ``brain.memory.embeddings`` — #259 F1 model-swap
(acceptance #11).

NETWORK-ENABLED: downloads and loads the actual
``intfloat/multilingual-e5-large`` ONNX model (~2.2GB, sharded external-data
weights — ``model.onnx`` + ``model.onnx_data`` — via fastembed) through the
SAME production path ``build_embedding_provider()`` uses. Every other test in
this package runs entirely OFFLINE via ``FakeEmbeddingProvider``; this file
is the one place that proves the REAL model actually loads and embeds under
our pinned ``onnxruntime==1.29.0`` — specifically, that the onnxruntime
external-data symlink workaround (``FastEmbedProvider.__init__`` /
``_materialize_symlinked_files`` / ``_materialize_fastembed_model_dir`` in
``brain/memory/embeddings.py``) actually prevents the "External data path
escapes model directory" failure onnxruntime's post-1.24.1 security check
raises against HuggingFace's default symlinked cache layout for a model that
ships external-data weights. A FAKE-provider test is blind to a regression in
that workaround (it never touches fastembed/onnxruntime at all) — this is the
load-bearing proof, at RUNTIME, not just unit-passing on the fake.

Marked BOTH ``@pytest.mark.requires_network`` (opts this test OUT of
``tests/conftest.py``'s autouse fake-embedding-provider fixture, so the REAL
``build_embedding_provider()`` wiring runs) AND ``@pytest.mark.integration``
(the marker actually deselected by this project's conventional local
pre-check gate, ``-m "not live and not requires_claude_cli and not
integration"`` — mirrors ``tests/unit/brain/memory/test_reranker_real_model.
py``, the sibling real-model test for the reranker). The real GitHub Actions
CI (``.github/workflows/test.yml``) runs bare ``uv run pytest -v --tb=short``
with no ``-m`` filter, so this test DOES run there (GitHub-hosted runners
have real network) — only the local pre-check convention needs the extra
marker to skip it.

One module fixture builds the provider once, so the e5 session (about 2.3 GB resident) loads once for
all five tests (it used to load once per test here and again in a second file for the token-count tests, which
also skipped unless an earlier test had left the model in the cache: they were order dependent). The token-count
tests (name-recall fix R1-recovery, criteria C17e and C18: the embedder's token counts, S108, and its one-time
load, S109) live in this module now. The model is downloaded when it is absent, whichever test runs first.

Run by hand with network enabled, e.g.:
    uv run pytest tests/unit/brain/memory/test_embedding_real_model.py \
        -m "requires_network or integration" -v -s
"""

from __future__ import annotations

import numpy as np
import pytest

from brain.bridge.model_tier import MODEL_EMBEDDING, MODEL_EMBEDDING_DIM
from brain.memory.embeddings import FastEmbedProvider, build_embedding_provider

# #540: a real-model download/load can legitimately outlast the default per-test timeout in pyproject.toml.
pytestmark = [
    pytest.mark.requires_network,
    pytest.mark.requires_models,
    pytest.mark.integration,
    pytest.mark.timeout(3600),
    pytest.mark.xdist_group("hf-cache"),
]


@pytest.fixture(scope="module")
def provider():
    """The real production provider, built once for the module. The autouse cache reset in tests/conftest.py drops
    the process-level provider after every test, so a test that called `build_embedding_provider()` itself loaded
    the 2.3 GB ONNX session again; the session loads lazily, on the first test that embeds."""
    return build_embedding_provider()


def test_real_embedding_model_loads_and_embeds_without_external_data_escape_error(provider) -> None:
    """The decisive proof for acceptance #11: constructing the REAL
    provider for the current `MODEL_EMBEDDING` (multilingual-e5-large) and
    calling `.embed()` on it must succeed and return a vector of the
    model's real dimension — WITHOUT onnxruntime raising "External data
    path escapes model directory". That error is exactly what the symlinked
    HuggingFace cache layout triggers for this model's sharded
    `model.onnx`/`model.onnx_data` weights when the materialize-real-files
    workaround is missing or broken; a green run of this test is the only
    thing in the suite that would catch a regression in that workaround."""
    assert isinstance(provider, FastEmbedProvider)
    assert provider.model_id() == MODEL_EMBEDDING == "intfloat/multilingual-e5-large", (
        "expected the pinned #259 F1 embedding model id — model_tier.py's "
        "MODEL_EMBEDDING changed out from under this test"
    )

    vec = provider.embed("hello, this is a real embedding model load test")

    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.shape == (MODEL_EMBEDDING_DIM,), (
        f"expected the real model's output to match the declared sanity "
        f"constant MODEL_EMBEDDING_DIM={MODEL_EMBEDDING_DIM}, got shape "
        f"{vec.shape} — if this is a genuine model dim change, update "
        f"MODEL_EMBEDDING_DIM in model_tier.py to match"
    )
    assert provider.embedding_dim() == MODEL_EMBEDDING_DIM
    assert np.linalg.norm(vec) > 0.0, "a real embedding should never be the zero vector"


def test_real_embedding_model_produces_distinguishable_vectors_for_different_text(provider) -> None:
    """Basic sanity beyond "it didn't crash": two unrelated sentences embed
    to different, non-degenerate vectors (catches a workaround that loads
    SOME file successfully but the wrong / a corrupted one)."""
    a = provider.embed("the cat sat on the mat")
    b = provider.embed("quantum entanglement in superconducting circuits")

    assert not np.array_equal(a, b)
    assert a.shape == b.shape == (MODEL_EMBEDDING_DIM,)


def test_real_embedding_model_batch_matches_single_embeds_within_1e_5(provider) -> None:
    """E1 (name-recall fix, spec §3): `embed_batch` returns, for every text,
    the vector `embed` returns for that text, within 1e-5 per component.
    Fastembed pads a batch to its longest input, so texts of very different
    lengths are the case that could diverge; two texts are the shortest and
    longest of the set on purpose. Embedder only (one model, run alone)."""
    texts = [
        "ok",
        "the cat sat on the mat",
        "Pretzel is a scruffy terrier mix, about thirty pounds. " * 12,
        "quantum entanglement in superconducting circuits",
    ]

    batch = provider.embed_batch(texts)
    singles = [provider.embed(t) for t in texts]

    assert len(batch) == len(texts)
    for text, b, s in zip(texts, batch, singles, strict=True):
        assert b.dtype == np.float32
        assert b.shape == (MODEL_EMBEDDING_DIM,)
        np.testing.assert_allclose(
            b, s, atol=1e-5, rtol=0, err_msg=f"batch != single for {text[:30]!r}"
        )


@pytest.mark.no_base_failure(reason='moved from the deleted tests/unit/brain/memory/test_embedding_tokens_real.py (#540, J9) to share one embedder load, with one added line that resets the session first; the id is new but the check is the same')
def test_c18_real_ensure_loaded_builds_the_session_once(provider) -> None:
    inner = provider._inner_model()
    inner.model = None  # not built yet, whichever test ran first: ensure_loaded must build it
    provider.ensure_loaded()
    session = inner.model
    assert session is not None
    provider.ensure_loaded()
    assert inner.model is session


@pytest.mark.no_base_failure(reason='moved unchanged from the deleted tests/unit/brain/memory/test_embedding_tokens_real.py (#540, J9) to share one embedder load; the id is new but the test is not')
def test_c17e_real_token_lengths_are_the_attention_mask_sums(provider) -> None:
    short = "a quiet evening"
    long_text = "the harbour kite drifted over the grey water " * 200  # far past 512 tokens
    mid = "Bob always starts his day with a strong cup of black coffee, then walks the dog."
    lengths, seconds = provider.token_lengths_timed([short, long_text, mid])
    assert seconds is not None and seconds >= 0.0

    tokenizer = provider._inner_model().tokenizer
    # Independent reference: each text encoded ALONE (no batch padding), so
    # its surviving tokens are every id it has.
    alone = [len(tokenizer.encode(t).ids) for t in (short, long_text, mid)]
    assert lengths == alone
    assert lengths[1] == tokenizer.truncation["max_length"], "the long text counts at its truncated length"
    batch = tokenizer.encode_batch([short, long_text])
    assert len(batch[0].ids) == len(batch[1].ids) > lengths[0], "batch padding is not counted"

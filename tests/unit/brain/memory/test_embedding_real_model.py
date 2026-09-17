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

Run by hand with network enabled, e.g.:
    uv run pytest tests/unit/brain/memory/test_embedding_real_model.py \
        -m "requires_network or integration" -v -s
"""

from __future__ import annotations

import numpy as np
import pytest

from brain.bridge.model_tier import MODEL_EMBEDDING, MODEL_EMBEDDING_DIM
from brain.memory.embeddings import build_embedding_provider

pytestmark = [pytest.mark.requires_network, pytest.mark.integration]


def test_real_embedding_model_loads_and_embeds_without_external_data_escape_error() -> None:
    """The decisive proof for acceptance #11: constructing the REAL
    provider for the current `MODEL_EMBEDDING` (multilingual-e5-large) and
    calling `.embed()` on it must succeed and return a vector of the
    model's real dimension — WITHOUT onnxruntime raising "External data
    path escapes model directory". That error is exactly what the symlinked
    HuggingFace cache layout triggers for this model's sharded
    `model.onnx`/`model.onnx_data` weights when the materialize-real-files
    workaround is missing or broken; a green run of this test is the only
    thing in the suite that would catch a regression in that workaround."""
    provider = build_embedding_provider()
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


def test_real_embedding_model_produces_distinguishable_vectors_for_different_text() -> None:
    """Basic sanity beyond "it didn't crash": two unrelated sentences embed
    to different, non-degenerate vectors (catches a workaround that loads
    SOME file successfully but the wrong / a corrupted one)."""
    provider = build_embedding_provider()

    a = provider.embed("the cat sat on the mat")
    b = provider.embed("quantum entanglement in superconducting circuits")

    assert not np.array_equal(a, b)
    assert a.shape == b.shape == (MODEL_EMBEDDING_DIM,)

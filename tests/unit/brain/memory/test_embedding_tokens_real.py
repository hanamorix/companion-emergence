"""Name-recall fix R1-recovery, criteria C17e and C18 (real provider): the
embedder's token counts (S108) and its one-time load (S109) on the cached
e5-large model.

Loads the real ONNX session (about 2.3 GB resident): run ALONE under a memory
cap (`systemd-run --user --scope -p MemoryMax=3500M -p MemorySwapMax=0`).
Skips unless the model files are already cached; never downloads
(`HF_HUB_OFFLINE`). Marked `requires_models` and `integration`, so the local
pre-check gate excludes it.
"""

from __future__ import annotations

import pytest

from brain.bridge.model_tier import MODEL_EMBEDDING
from brain.memory.embeddings import FastEmbedProvider
from brain.paths import get_cache_dir

pytestmark = [pytest.mark.requires_models, pytest.mark.integration]


@pytest.fixture(scope="module")
def provider():
    mp = pytest.MonkeyPatch()
    mp.setenv("HF_HUB_OFFLINE", "1")
    try:
        yield FastEmbedProvider(model_id=MODEL_EMBEDDING, cache_dir=get_cache_dir())
    except Exception as exc:  # noqa: BLE001 — not cached (offline) -> skip
        pytest.skip(f"embedding model not in the local cache: {exc}")
    finally:
        mp.undo()


def test_c18_real_ensure_loaded_builds_the_session_once(provider) -> None:
    inner = provider._inner_model()
    provider.ensure_loaded()
    session = inner.model
    assert session is not None
    provider.ensure_loaded()
    assert inner.model is session


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

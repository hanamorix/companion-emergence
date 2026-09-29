"""Name-recall fix R1, criterion C1c (real tokenizer half): surviving pair
lengths (S62) agree with the reranker's OWN tokenizer offsets.

Loads ONLY the cached reranker tokenizer files (tokenizer.json and its
configs, via fastembed's own `load_tokenizer`, the loader the reranker uses)
— no ONNX session, no model weights. Skips, without any network request,
unless those files are already in the local cache. Marked `requires_models`
+ `integration` so the local pre-check gate (`-m "not integration"`)
excludes it, like the other real-model tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.bridge.model_tier import TIER_RERANKER, model_for_tier
from brain.memory.reranker import surviving_pair_char_lengths
from brain.paths import get_cache_dir

pytestmark = [pytest.mark.requires_models, pytest.mark.integration]

_TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json"]


def _cached_tokenizer_dir() -> Path | None:
    """The reranker repo's cached snapshot directory holding the tokenizer
    files (the fp16 export shares the fp32 repo), or None when not cached.
    `local_files_only=True`: never touches the network."""
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            model_for_tier(TIER_RERANKER),
            cache_dir=str(get_cache_dir()),
            local_files_only=True,
            allow_patterns=_TOKENIZER_FILES,
        )
    except Exception:  # noqa: BLE001 — not cached (or unreadable) -> skip
        return None
    directory = Path(path)
    return directory if all((directory / name).exists() for name in _TOKENIZER_FILES) else None


def test_surviving_length_agrees_with_the_real_tokenizer_offsets() -> None:
    directory = _cached_tokenizer_dir()
    if directory is None:
        pytest.skip("reranker tokenizer files not in the local cache; skipping without a network request")
    from fastembed.common.preprocessor_utils import load_tokenizer

    tokenizer, _ = load_tokenizer(model_dir=directory)
    query = "what does Bob like to drink in the morning"
    long_doc = " ".join(f"word{i}" for i in range(1200))[:6000]
    short_doc = "Bob always starts his day with a strong cup of black coffee."
    assert len(long_doc) == 6000

    encodings = tokenizer.encode_batch([(query, long_doc), (query, short_doc)])
    lengths = surviving_pair_char_lengths(query, [long_doc, short_doc], encodings)

    # Independent computation from the tokenizer's own offsets: the long
    # document is cut (the model maximum is reached), so its surviving
    # characters end at its last surviving token; the query is whole.
    long_enc = encodings[0]
    doc_ends = [end for (_s, end), seq in zip(long_enc.offsets, long_enc.sequence_ids, strict=True) if seq == 1]
    assert long_enc.overflowing, "fixture precondition: the 6,000-character document is truncated"
    expected_long = len(query) + max(doc_ends)
    assert max(doc_ends) < len(long_doc)
    assert lengths[0] == expected_long
    assert lengths[1] == len(query) + len(short_doc), "an untruncated pair counts at its full length"

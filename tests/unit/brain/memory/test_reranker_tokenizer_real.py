"""Name-recall fix R1, criterion C1c (real tokenizer half): surviving pair
lengths, in reranker tokens (S62, S75), agree with the reranker's OWN
tokenizer.

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
from brain.memory.reranker import surviving_pair_token_lengths
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


def _tokenizer():
    directory = _cached_tokenizer_dir()
    if directory is None:
        pytest.skip("reranker tokenizer files not in the local cache; skipping without a network request")
    from fastembed.common.preprocessor_utils import load_tokenizer

    tokenizer, _ = load_tokenizer(model_dir=directory)
    return tokenizer


def test_token_length_agrees_with_the_real_tokenizer_and_excludes_padding() -> None:
    tokenizer = _tokenizer()
    max_tokens = tokenizer.truncation["max_length"]
    query = "what does Bob like to drink in the morning"
    long_doc = " ".join(f"word{i}" for i in range(1200))[:6000]
    short_doc = "Bob always starts his day with a strong cup of black coffee."

    encodings = tokenizer.encode_batch([(query, long_doc), (query, short_doc)])
    lengths = surviving_pair_token_lengths(encodings)

    long_enc, short_enc = encodings
    assert long_enc.overflowing, "fixture precondition: the 6,000-character document is truncated"
    assert lengths[0] == max_tokens, "a truncated pair counts at the model maximum"
    assert lengths[1] < lengths[0]
    # The short pair is padded to the long pair's length inside this batch,
    # so its token ids are as long as the long pair's; only the mask tells
    # its real length, and a pair encoded alone must agree with it.
    assert len(short_enc.ids) == len(long_enc.ids)
    (alone,) = tokenizer.encode_batch([(query, short_doc)])
    assert lengths[1] == len(alone.ids), "padding is not counted: the same pair alone is unpadded"
    content = sum(1 for seq in alone.sequence_ids if seq is not None)
    assert lengths[1] > content, "special tokens are counted (the model runs them)"


def test_token_length_when_the_query_is_cut_too() -> None:
    """A very long query and a long document are both cut (the tokenizer's
    longest-first truncation): the pair still counts at the model maximum."""
    tokenizer = _tokenizer()
    query = " ".join(f"ask{i}" for i in range(1200))[:5600]
    doc = " ".join(f"note{i}" for i in range(1200))[:6000]

    (encoding,) = tokenizer.encode_batch([(query, doc)])
    (length,) = surviving_pair_token_lengths([encoding])

    ends = {0: 0, 1: 0}
    for (_s, end), seq in zip(encoding.offsets, encoding.sequence_ids, strict=True):
        if seq in ends:
            ends[seq] = max(ends[seq], end)
    assert ends[0] < len(query) and ends[1] < len(doc), "fixture precondition: both segments are cut"
    assert length == tokenizer.truncation["max_length"]


def test_cjk_and_emoji_cost_more_tokens_per_character_than_latin() -> None:
    """The misestimate S75 fixes, on the shipped tokenizer: the same number
    of characters is a very different number of tokens."""
    tokenizer = _tokenizer()
    query = "what does Bob like to drink"
    docs = ["coffee " * 60, "\u5496\u5561" * 210, "\U0001f600\U0001f389" * 210]
    assert len({len(d) for d in docs}) == 1, "fixture: the same character count"

    lengths = surviving_pair_token_lengths(tokenizer.encode_batch([(query, d) for d in docs]))
    latin, cjk, emoji = lengths

    assert cjk > 1.5 * latin, "a character count would size these alike"
    assert emoji > 2 * latin

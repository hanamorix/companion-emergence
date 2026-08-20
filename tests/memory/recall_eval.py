"""recall_eval.py — Tier-1 passive-recall offline eval harness (Stage 2 §C).

Reusable, token-free, no-dependency metrics harness for the recall-query
salience fix (`changes/recall-query-tier1/`). Drives the REAL
`brain.chat.prompt._build_recall_block` against a SYNTHETIC in-memory store
(user "Bob", persona label "Canary" — never Phoebe's persona) and computes
recall@k / MRR / nDCG@k (binary gain) for a labelled trigger set, demonstrating
buried-trigger recall reaching early-trigger parity and quantifying residual
under-recall depth.

Importable by `tests/memory/test_recall_query_tier1.py` (the gating pytest
module) so both share one implementation. Also runnable directly:

    uv run python tests/memory/recall_eval.py
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from brain.chat.prompt import _build_recall_block
from brain.memory.relevance import SNIPPET_COUNT
from brain.memory.store import Memory, MemoryStore

# recall@K cutoff — matches the production render limit (SNIPPET_COUNT), so
# the harness measures exactly what a real turn would surface.
K = SNIPPET_COUNT

# The real flagship trigger messages (diagnosis.md / repro.py). MSG_BURIED's
# salient tokens ("garbage", "treasure") sit at the tail of 6 other >=4-char
# filler words — the exact shape that broke the pre-fix positional cap.
MSG_EARLY = "garbage treasure"
MSG_BURIED = "alrighty logger is live. First a quick memory trigger: garbage treasure"

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

# Diverse background vocabulary — deliberately contains neither "treasure"
# nor "garbage" so it never competes for the target query, and none of the
# _RECALL_STOPWORDS-excluded content words ("issue", "first", "quick", etc.)
# so token selection for MSG_BURIED is not perturbed by the background.
_BACKGROUND_TOPICS = [
    "a quiet afternoon spent reorganizing the bookshelf by color",
    "planning a weekend hike along the river trail",
    "debugging a flaky network timeout in the staging cluster",
    "a recipe for lentil soup with roasted garlic",
    "watching an old film about a lighthouse keeper",
    "sketching notes for a short story about a train station",
    "comparing two espresso machines before buying one",
    "a conversation about learning to play the cello",
    "repainting the fence before the rain arrives",
    "a long email thread about quarterly budget planning",
    "practicing a new language with flashcards every evening",
    "assembling a bicycle from a mail-order kit",
    "a walk through the botanical garden in early spring",
    "notes from a lecture on tidal patterns",
    "organizing photographs from a trip to the coast",
    "a discussion about which houseplants tolerate low light",
    "fixing a squeaky door hinge in the hallway",
    "drafting an outline for a woodworking project",
    "a chat about favorite constellations visible in autumn",
    "reviewing a draft of a friend's wedding speech",
]

RARE_STORE_FRACTION = 0.03  # ~150 background docs; treasure is a small slice (high IDF)
COMMON_STORE_FRACTION = 0.28  # ~25-30% of the store (low IDF — the incident direction)


def seed_store(target_fraction: float, k: int = 5) -> tuple[MemoryStore, set[str]]:
    """Build a synthetic ':memory:' store with a tunable target-term prevalence.

    Seeds `k` TARGET memories whose content contains "garbage treasure" (the
    flagship trigger phrase), plus background memories sized so the targets
    occupy approximately `target_fraction` of the store — low fraction (~3%,
    RARE_STORE_FRACTION) gives the target term high corpus IDF; high fraction
    (~25-30%, COMMON_STORE_FRACTION) gives it LOW IDF, matching the
    documented incident direction (52/156 ~= 33% "treasure" occurrences).
    Also seeds a few distractors matching MSG_BURIED's filler words
    ("logger"/"memory"/"live") — what the pre-fix positional cap actually
    retrieved. Never Phoebe data; synthetic user "Bob", persona label
    "Canary" (labels only — MemoryStore itself is persona-agnostic).

    Returns (store, target_ids). `k` is kept <= K (the recall@k cutoff) so
    recall@k has no artificial ceiling below 1.0 (no saturation).
    """
    store = MemoryStore(":memory:")
    target_ids: set[str] = set()

    for i in range(k):
        m = Memory.create_new(
            content=f"garbage treasure find #{i}: a curbside treasure worth keeping.",
            memory_type="conversation",
            domain="us",
            importance=5.0,
        )
        store.create(m)
        target_ids.add(m.id)

    total = max(k + 1, round(k / target_fraction))
    background_count = total - k
    for i in range(background_count):
        topic = _BACKGROUND_TOPICS[i % len(_BACKGROUND_TOPICS)]
        store.create(
            Memory.create_new(
                content=f"{topic} (note {i})",
                memory_type="conversation",
                domain="us",
                importance=3.0,
            )
        )

    for txt in (
        "Roy set up a logger integrated into Claude Code to observe the memory system.",
        "The logger will go live first thing; a quick restart is needed.",
        "Notes on the memory retrieval system and its relevancy ranking.",
    ):
        store.create(
            Memory.create_new(
                content=txt, memory_type="conversation", domain="work", importance=5.0
            )
        )

    return store, target_ids


def _active_ids(block: str) -> list[str]:
    """Parse memory UUIDs from the ACTIVE section only of a rendered block.

    Path B (forgetting-aware, persona_dir set) renders an explicit
    '  active:' header followed by '    - <uuid>: "..."' lines, terminated by
    the next section header ('  softened...', '  lost...', '  not
    recognised...') or end of block. Path A (persona_dir=None, legacy) has no
    section headers at all — the whole block IS the active section, rendered
    as '- [importance N/10 ...] <uuid>: "..."'.
    """
    lines = block.splitlines()
    section_headers = (
        "active:",
        "softened (fading; original detail gone):",
        "lost (no longer in active memory):",
    )
    has_sections = any(
        ln.strip() in section_headers or ln.strip().startswith("not recognised")
        for ln in lines
    )
    ids: list[str] = []
    if not has_sections:
        for ln in lines:
            m = _UUID_RE.search(ln)
            if m:
                ids.append(m.group(0))
        return ids
    in_active = False
    for ln in lines:
        stripped = ln.strip()
        if stripped == "active:":
            in_active = True
            continue
        if stripped in section_headers[1:] or stripped.startswith("not recognised"):
            in_active = False
            continue
        if in_active:
            m = _UUID_RE.search(ln)
            if m:
                ids.append(m.group(0))
    return ids


def surfaced_ids(
    store: MemoryStore, msg: str, persona_dir: Path | None, limit: int = K
) -> list[str]:
    """Render `_build_recall_block` for `msg` and return the ACTIVE section's
    surfaced memory ids, best-match first (render order == rank order)."""
    block = _build_recall_block(store, msg, persona_dir=persona_dir, limit=limit)
    return _active_ids(block)


def not_recognised_tokens(block: str) -> list[str]:
    """Parse the tokens listed under the 'not recognised' section, if any."""
    if "not recognised" not in block:
        return []
    tail = block[block.index("not recognised") :]
    tokens: list[str] = []
    for ln in tail.splitlines()[1:]:
        stripped = ln.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            tokens.append(stripped[2:].strip())
        else:
            break
    return tokens


def metrics(surfaced: list[str], expected: set[str], k: int) -> dict[str, float]:
    """recall@k, MRR, nDCG@k (binary gain) of `surfaced` against `expected`."""
    topk = surfaced[:k]
    if not expected:
        return {"recall@k": 0.0, "mrr": 0.0, "ndcg@k": 0.0}

    hits_in_topk = sum(1 for mid in topk if mid in expected)
    recall = hits_in_topk / len(expected)

    mrr = 0.0
    for rank, mid in enumerate(topk, start=1):
        if mid in expected:
            mrr = 1.0 / rank
            break

    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, mid in enumerate(topk, start=1) if mid in expected
    )
    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {"recall@k": recall, "mrr": mrr, "ndcg@k": ndcg}


@contextmanager
def _fts_call_spy():
    """Context manager yielding a mutable counter of MemoryStore.search_fts_scored calls."""
    counter = {"n": 0}
    original = MemoryStore.search_fts_scored

    def spy(self, *args, **kwargs):
        counter["n"] += 1
        return original(self, *args, **kwargs)

    with patch.object(MemoryStore, "search_fts_scored", spy):
        yield counter


def fts_call_count(
    store: MemoryStore, msg: str, *, persona_dir: Path | None, limit: int = K
) -> int:
    """Count MemoryStore.search_fts_scored invocations made by ONE
    `_build_recall_block` call (C2 proxy — pre-fix this was up to
    `len(tokens)`; post-fix it must be exactly 1)."""
    with _fts_call_spy() as counter:
        _build_recall_block(store, msg, persona_dir=persona_dir, limit=limit)
    return counter["n"]


def _report_store(label: str, target_fraction: float, tmp_persona_dir: Path) -> None:
    store, target_ids = seed_store(target_fraction)
    try:
        early_surfaced = surfaced_ids(store, MSG_EARLY, tmp_persona_dir)
        buried_surfaced = surfaced_ids(store, MSG_BURIED, tmp_persona_dir)

        early_m = metrics(early_surfaced, target_ids, K)
        buried_m = metrics(buried_surfaced, target_ids, K)
        residual = early_m["recall@k"] - buried_m["recall@k"]

        print(f"=== {label} store (target_fraction={target_fraction:.2f}, k={len(target_ids)}) ===")
        print(f"  early  ({MSG_EARLY!r}): {early_m}")
        print(f"  buried ({MSG_BURIED!r}): {buried_m}")
        print(f"  buried-vs-early parity: buried >= 0.8*early? "
              f"{buried_m['recall@k'] >= 0.8 * early_m['recall@k']} "
              f"(buried={buried_m['recall@k']:.3f}, "
              f"0.8*early={0.8 * early_m['recall@k']:.3f})")
        print(f"  residual under-recall depth (early - buried): {residual:.3f}")
    finally:
        store.close()


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        pdir = Path(td)
        _report_store("rare-target", RARE_STORE_FRACTION, pdir)
        _report_store("common-target (incident)", COMMON_STORE_FRACTION, pdir)

        # C2 proxy: single-FTS-query-per-turn call count on both paths.
        store, _ = seed_store(RARE_STORE_FRACTION)
        try:
            path_a_calls = fts_call_count(store, MSG_BURIED, persona_dir=None)
            path_b_calls = fts_call_count(store, MSG_BURIED, persona_dir=pdir)
            print("=== search_fts_scored call count (C2 proxy) ===")
            print(f"  Path A (persona_dir=None): {path_a_calls} call(s)")
            print(f"  Path B (persona_dir set):  {path_b_calls} call(s)")
        finally:
            store.close()


if __name__ == "__main__":
    main()

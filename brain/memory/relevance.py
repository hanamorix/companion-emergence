"""relevance.py — cross-DB relevance ranking over the committed memory pool.

Blends four signals into a single 0..1-normalized score:

    score = W_MATCH·bm25 + W_IMP·importance + W_HEB·hebbian-activation + W_REC·recency

The blend spans two databases — ``memories.db`` (BM25 text-match, importance,
recency) and ``hebbian.db`` (spreading activation) — so it lives here rather
than in ``store.py`` (the pure ``memories.db`` layer, which has no handle on
``HebbianMatrix``). ``store.py`` owns only the FTS5 primitives.

P2 (memory relevance overhaul). The weight/decay consts are documented defaults;
tuning is deferred to #129. Also the single home for the snippet-then-read
render consts (imported by ``brain.chat.prompt`` and the search tools) so the
values live in one place.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from brain.dev_constants import MONOLOGUE_FAMILY_TYPES
from brain.memory.known_names import KnownNames, load_known_names, match_known_names
from brain.memory.store import FtsPhrases

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore

# --- ranking weights (applied to 0..1-normalized signals; tuning → #129) -----
# Reasoned ordering: BM25 text-match is the primary relevance signal; importance
# a strong secondary; hebbian "related-to-now" and recency are tie-breakers.
W_MATCH = 1.0
W_IMP = 0.5
W_HEB = 0.3
W_REC = 0.2

# recency decay: recency = 0.5 ** (age_days / RECENCY_HALFLIFE_DAYS)
RECENCY_HALFLIFE_DAYS = 30

# hebbian spreading-activation seed/traversal (spec §6.2). Seeded from this
# turn's top BM25 ids — the memories the current input is textually about.
HEB_SEED_COUNT = 5
HEB_DEPTH = 2
HEB_DECAY_PER_HOP = 0.5

# Candidate pool pulled from FTS before ranking — wider than the render limit so
# the ranker has something to reorder (and the recall paths something to
# partition genuine-first before cutting to their limit).
CANDIDATE_POOL = 50

# Flag: when False, falls back to recency-only `search_text` (score None).
RELEVANCE_RANKING_ENABLED = True

# --- snippet-then-read render consts (one home; imported by prompt + tools) ---
SNIPPET_COUNT = 8  # up from 5 — snippets are cheap
SNIPPET_MAX_CHARS = 140  # unchanged from the current recall truncation
SNIPPET_MIN_CHARS = 20  # floor so 20% of a tiny memory isn't a useless fragment
FULL_INJECT_IMPORTANCE = 9.0  # a genuinely important memory is never gated behind a read-call
FULL_INJECT_MAX = 3  # at most this many full-injects (bounds volatile-tail growth)
SNIPPET_MODE_ENABLED = True  # when False, falls back to the current full-body render


def snippet_length(body_len: int) -> int:
    """Proportional snippet length: ~20% of the body, floored and capped.

    ``min(max(SNIPPET_MIN_CHARS, body_len // 5), SNIPPET_MAX_CHARS)`` — keeps
    even short memories withholding ~80% of their content (so a follow-up
    ``read_full_memory`` still adds real text), while a long memory still
    caps at ``SNIPPET_MAX_CHARS``. A body at or below the floor shows in
    full rather than shrinking to a useless fragment.
    """
    return min(max(SNIPPET_MIN_CHARS, body_len // 5), SNIPPET_MAX_CHARS)

# The recall-path hebbian open degrades to None (w_heb=0) on any open/query error.
HEB_OPEN_FAILSOFT = True


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _created_ts(mem: Memory) -> float:
    try:
        return mem.created_at.timestamp()
    except Exception:  # noqa: BLE001 — defensive; tie-break only
        return 0.0


def rank_memories(
    store: MemoryStore,
    hebbian: HebbianMatrix | None,
    query: str | Sequence[str],
    *,
    limit: int,
    exclude_ids: Iterable[str] = frozenset(),
    active_only: bool = True,
    include_fading: bool = True,
    genuine_first: bool = False,
) -> list[tuple[Memory, float | None]]:
    """Rank committed memories by blended relevance — bump-free, best→worst.

    Returns ``(memory, blended_score)`` pairs. The score is ``None`` when
    ranking is disabled (an *unranked* sentinel — callers then fall back to
    ``(-importance, -ts)``; distinct from a real 0.0 blended score).

    ``rank_memories`` NEVER opens a ``HebbianMatrix``: ``hebbian`` is supplied by
    the caller (or ``None``, which zeroes the ``w_heb`` term). This makes the
    "opens N× per turn" defect structurally impossible.

    ``query`` is a raw string (the store drops tokens under 3 characters) or a
    token LIST, the recall selector's own output, of which the store admits
    every token (name-recall fix R4, spec §5). Both forms are one OR query.

    ``genuine_first`` (name-recall fix R4, spec §4, S16, Acceptance 8): the
    monologue family ranks after every genuine memory, in the candidate pool
    the FTS query hands the ranker AND in the final order, with no score
    multiplier. So a family flood can neither fill the pool nor fill the first
    ``limit`` results while a genuine match exists. Default False: unchanged.
    """
    exclude = frozenset(exclude_ids)

    if not RELEVANCE_RANKING_ENABLED:
        # Recency-only fallback. None signals "unranked". Exclusion is applied
        # BEFORE the final limit (mirroring the ranked path): fetch a wider pool,
        # drop excluded ids, THEN slice — so an excluded id inside the top-`limit`
        # matches backfills from the next candidate instead of shrinking the
        # result (stage-6 minor).
        fetch = (limit + len(exclude)) if limit is not None else None
        if isinstance(query, FtsPhrases):
            # Whole phrases: one substring search per phrase, newest first
            # across them (a joined string would only match all of them at once).
            found: dict[str, Memory] = {}
            for phrase in query:
                for m in store.search_text(
                    phrase,
                    active_only=active_only,
                    include_fading=include_fading,
                    bump=False,
                    limit=fetch,
                ):
                    found.setdefault(m.id, m)
            fallback = sorted(found.values(), key=_created_ts, reverse=True)
        else:
            fallback = store.search_text(
                query if isinstance(query, str) else " ".join(query),
                active_only=active_only,
                include_fading=include_fading,
                bump=False,
                limit=fetch,
            )
        kept = [(m, None) for m in fallback if m.id not in exclude]
        return kept[:limit] if limit is not None else kept

    scored = store.search_fts_scored(
        query,
        active_only=active_only,
        include_fading=include_fading,
        bump=False,
        limit=CANDIDATE_POOL,
        family_types=MONOLOGUE_FAMILY_TYPES if genuine_first else (),
    )
    if not scored:
        return []

    # Hebbian spreading activation seeded from this turn's top BM25 ids.
    activation: dict[str, float] = {}
    if hebbian is not None:
        seeds = [m.id for m, _ in scored[:HEB_SEED_COUNT]]
        try:
            activation = hebbian.spreading_activation(seeds, HEB_DEPTH, HEB_DECAY_PER_HOP)
        except Exception:  # noqa: BLE001 — hebbian is a tie-breaker, never fatal
            activation = {}

    # bm25 min-max over the pool, inverted so higher = better match.
    bm_values = [bm for _, bm in scored]
    bm_min, bm_max = min(bm_values), max(bm_values)
    bm_span = bm_max - bm_min

    now = datetime.now(UTC)
    ranked: list[tuple[Memory, float]] = []
    for mem, bm in scored:
        if mem.id in exclude:
            continue
        # Single-candidate set (bm_span == 0) → 1.0, no divide-by-zero.
        m_norm = 1.0 if bm_span == 0 else (bm_max - bm) / bm_span
        i_norm = _clamp01(mem.importance / 10.0)
        h_norm = _clamp01(activation.get(mem.id, 0.0))
        age_days = max(0.0, (now - mem.created_at).total_seconds() / 86400.0)
        r_norm = 0.5 ** (age_days / RECENCY_HALFLIFE_DAYS)
        score = W_MATCH * m_norm + W_IMP * i_norm + W_HEB * h_norm + W_REC * r_norm
        ranked.append((mem, score))

    # P3: filter superseded here  (no-op today — forward-compat seam, spec §6.8)

    if genuine_first:
        ranked.sort(
            key=lambda pair: (
                pair[0].memory_type in MONOLOGUE_FAMILY_TYPES,
                -pair[1],
                -_created_ts(pair[0]),
            )
        )
    else:
        ranked.sort(key=lambda pair: (-pair[1], -_created_ts(pair[0])))
    return [(mem, score) for mem, score in ranked[:limit]]


def names_in(persona_dir: Path | str | None, text: str) -> list[str]:
    """The known names that occur in ``text``: name protection's detection step.

    Name-recall fix R5 (spec §5, S27, S36, S47). The persona's known-names list
    (``brain.memory.known_names``, read once per process and re-read when the
    file changes) is matched against the RAW words of ``text``, lower-cased,
    BEFORE any stopword or length rule, as windows of consecutive words, so a
    listed 2-letter name, a listed multi-word name ("new york") and a listed
    name that is also a stopword are all found. Returns lower-cased names in
    order of occurrence, each once. ``[]`` when there is no persona directory
    (nothing to read), no list, or no match; fail-soft: any error means no
    protection for this call, never a failed recall.

    This is the single place where "which of her words count as a name" is
    decided for both passive recall and the search tool. A policy for a name
    that is also a stopword (an open question with the owner: a person called
    "Will" cannot be listed while the admission rule rejects stopword strings)
    belongs here, not in the two call sites.
    """
    if persona_dir is None or not text:
        return []
    try:
        return match_known_names(text, load_known_names(persona_dir))
    except Exception as exc:  # noqa: BLE001 - name protection must never break recall
        logger.warning("known names: lookup failed (%s); no name protection this call", exc)
        return []


def rank_name_hits(
    store: MemoryStore,
    hebbian: HebbianMatrix | None,
    names: Sequence[str],
    *,
    limit: int,
    exclude_ids: Iterable[str] = frozenset(),
) -> list[tuple[Memory, float | None]]:
    """The name query: ONE keyword query for the message's name words.

    Name-recall fix R5 (spec §5, S27, S35, S47). The same lexical ranker as
    every other keyword search (``rank_memories``: BM25 + importance + hebbian +
    recency), over the matched names sent as FTS PHRASES (each name is one
    quoted phrase, no length floor, ORed), active and fading memories, the
    monologue family after every genuine hit. The caller leads the keyword hits
    with them (`lead_with_names`, S89), because FTS5 cannot weight one term
    above another and a frequently mentioned name would otherwise be
    out-ranked by one rare word. It does NOT touch the lost-memory (graveyard) search: passive recall
    feeds that only the legacy capped token set until the owner rules on the
    graveyard widening (F11, plan P-14/P-25).

    ``names`` come from :func:`names_in`; none means no query (``[]``).
    """
    if not names:
        return []
    return rank_memories(
        store,
        hebbian,
        FtsPhrases(names),
        limit=limit,
        exclude_ids=exclude_ids,
        include_fading=True,
        genuine_first=True,
    )


def lead_with_names(
    name_hits: Sequence[Memory], general_hits: Sequence[Memory], names: Sequence[str]
) -> list[Memory]:
    """Order the keyword hits: the name query's lead, without letting name-only
    memories crowd out the ones that match the rest of the message too.

    Name-recall fix R5 follow-up (spec section 5, S89, derived from S27/S79/S40).
    Three groups, each in its own order, each memory once:

    1. general hits that also match a name (the memory matches the name AND the
       rest of the message), in the general search's order;
    2. name-only hits, in the name query's order;
    3. the remaining general hits, in the general search's order.

    So a name that matches a great many memories cannot push out a memory
    matching the name and the message's other words, while the name still
    outranks every general hit that does not match it. ``general_hits`` is the
    whole general keyword search in its final order (tier 1, then tier 2).
    "Also matches a name" is decided per general hit, not by the name query's
    ranked window (which a name with many memories would truncate): the hit is
    in the name query's result OR its text holds a matched name as consecutive
    words (the matcher's own rule). No names, or a name query with no hits (none
    matched, or it failed): ``general_hits`` unchanged.
    """
    if not names or not name_hits:
        return list(general_hits)
    known = KnownNames.from_names(names)
    name_ids = {m.id for m in name_hits}
    named = {
        m.id for m in general_hits if m.id in name_ids or match_known_names(m.content or "", known)
    }
    both = [m for m in general_hits if m.id in named]
    name_only = [m for m in name_hits if m.id not in named]
    rest = [m for m in general_hits if m.id not in named]
    seen: set[str] = set()
    out: list[Memory] = []
    for m in (*both, *name_only, *rest):
        if m.id not in seen:
            seen.add(m.id)
            out.append(m)
    return out

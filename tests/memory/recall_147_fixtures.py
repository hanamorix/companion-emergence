"""recall_147_fixtures.py: #147 control-recall fixtures (name-recall fix R4, C12).

#147 is the fear that removing the 10-token cap and admitting short tokens
(2-letter words, acronyms, digits) makes NORMAL-word recall worse: more OR
terms, more slot competition. These fixtures are the control: labelled queries
whose targets are reached through ordinary words only (3+ characters, no names
list), long enough (> 10 salient tokens) that the old cap binds, and carrying
the short tokens the admission change newly lets through.

Synthetic only: user "Bob", persona label "Canary", a seeded RNG, an in-memory
store. `build_control_set` is pure and deterministic, so the SAME set can be
measured on this build and on the base worktree (`origin/main` `ed9b83ef`),
which is how the committed floors in `test_recall_147_controls.py` were taken.
No import of anything added by R4: the module runs against the base code too.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from brain.chat.prompt import _build_recall_block
from brain.memory.semantic_recall import SemanticRecallResult
from brain.memory.store import Memory, MemoryStore

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

# Ordinary words (all 3+ characters, none a recall stopword, none a name).
NOUNS = [
    "kettle", "ladder", "window", "garden", "basket", "blanket", "candle", "mirror", "pillow", "carpet",
    "bottle", "hammer", "ribbon", "button", "camera", "engine", "bridge", "harbour", "market", "station",
    "library", "kitchen", "meadow", "orchard", "village", "puzzle", "letter", "parcel", "ticket", "journal",
    "compass", "lantern", "saddle", "anchor", "barrel", "bucket", "cabinet", "curtain", "drawer", "fabric",
]
ADJS = [
    "quiet", "narrow", "golden", "crooked", "gentle", "hollow", "rusty", "silver", "tidy", "wobbly",
    "ancient", "brisk", "clumsy", "dusty", "eager", "faded", "glossy", "humble", "jagged", "knotted",
]
VERBS = [
    "mended", "carried", "painted", "polished", "sorted", "folded", "measured", "repaired", "wrapped", "stacked",
    "cleaned", "counted", "packed", "traded", "borrowed", "lifted", "rolled", "tied", "swept", "watched",
]
PLACES = [
    "shed", "attic", "cellar", "porch", "pantry", "barn", "loft", "hallway", "studio", "workshop",
]
# Short tokens the admission change newly lets through: acronyms (2-5 uppercase
# letters), 2-letter words, digits. They occur in some background memories, so
# df > 0 (a df-0 token retrieves nothing).
SHORT = ["GPS", "CPU", "USB", "NASA", "FBI", "TV", "AI", "PC", "42", "7", "3D", "OK"]
# High-df filler that crowds the OR query once the cap is gone.
CROWD = [
    "thing", "time", "day", "morning", "week", "year", "people", "story", "part", "place",
    "work", "life", "world", "hand", "home", "night", "water", "room", "side", "point",
]
RARE = [
    "quokka", "zeppelin", "marmoset", "obelisk", "pangolin", "sextant", "trombone", "vermilion", "yodel", "zucchini",
    "abacus", "bagpipe", "cormorant", "dirigible", "ephemera", "flamingo", "gargoyle", "harpsichord", "iguana", "jubilee",
]


@dataclass(frozen=True)
class ControlQuery:
    text: str
    targets: frozenset[str]  # memory ids


def _sentence(
    rng: random.Random,
    *,
    extra: list[str] | None = None,
    verb: str | None = None,
    adj: str | None = None,
    noun: str | None = None,
    place: str | None = None,
) -> str:
    """One template for EVERY memory (background and target alike), so nothing
    but the chosen slot words tells a target apart from a distractor."""
    words = [
        "Bob", verb or rng.choice(VERBS), "the", adj or rng.choice(ADJS), noun or rng.choice(NOUNS),
        "with", "the", rng.choice(ADJS), rng.choice(NOUNS), "in", "the", place or rng.choice(PLACES),
        "and", "then", rng.choice(VERBS), "it", "for", rng.choice(CROWD), rng.choice(CROWD),
    ]
    if extra:
        words.extend(extra)
    return " ".join(words) + "."


def build_control_set(
    *,
    seed: int,
    n_queries: int,
    rare_target: bool = False,
    n_background: int = 500,
    n_short: int = 3,
) -> tuple[MemoryStore, list[ControlQuery]]:
    """Build a store and `n_queries` labelled control queries.

    Background: `n_background` sentences from one template over the ordinary-
    word slots, a fraction carrying the SHORT tokens. Target for query i: ONE
    extra memory of the same template with four query words in its slots
    (`rare_target`: ONE rare word, present in no other memory, and nothing else
    the query says). The query repeats those target words and adds `n_short`
    short tokens (0 isolates the cap removal from the short-token admission)
    and 9 (12 when `rare_target`) CROWD words, so it always has more than 10
    salient tokens when `n_short` is 3.
    """
    rng = random.Random(seed)
    store = MemoryStore(":memory:")
    for i in range(n_background):
        extra = [rng.choice(SHORT)] if i % 4 == 0 else None
        store.create(
            Memory.create_new(
                content=_sentence(rng, extra=extra),
                memory_type="conversation",
                domain="us",
                importance=float(rng.choice([3, 4, 5, 5, 6])),
            )
        )

    queries: list[ControlQuery] = []
    used_rare = list(RARE)
    rng.shuffle(used_rare)
    for i in range(n_queries):
        if rare_target:
            rare = used_rare[i % len(used_rare)] + ("" if i < len(used_rare) else f"x{i}")
            target_words = [rare]
            content = _sentence(rng, noun=rare)
        else:
            target_words = [rng.choice(NOUNS), rng.choice(ADJS), rng.choice(PLACES), rng.choice(VERBS)]
            content = _sentence(
                rng, noun=target_words[0], adj=target_words[1], place=target_words[2], verb=target_words[3]
            )
        target = Memory.create_new(
            content=content, memory_type="conversation", domain="us", importance=5.0
        )
        store.create(target)
        shorts = rng.sample(SHORT, 3)[:n_short]  # always draw 3: keeps the RNG stream identical
        crowd = rng.sample(CROWD, 12 if rare_target else 9)
        words = [*target_words, *shorts, *crowd]
        rng.shuffle(words)
        text = "Bob was asking me about " + ", ".join(words) + " from a while back"
        queries.append(ControlQuery(text=text, targets=frozenset({target.id})))
    return store, queries


def active_ids(block: str) -> list[str]:
    """Memory ids of the ACTIVE section of a rendered recall block, in order."""
    ids: list[str] = []
    in_active = False
    for line in block.splitlines():
        stripped = line.strip()
        if stripped == "active:":
            in_active = True
            continue
        if stripped.startswith(("softened", "lost", "not recognised")):
            in_active = False
            continue
        if in_active:
            m = _UUID_RE.search(line)
            if m:
                ids.append(m.group(0))
    return ids


def recall_at(
    store: MemoryStore,
    queries: list[ControlQuery],
    *,
    cutoff: int,
    persona_dir: Path,
) -> float:
    """Mean over queries of |targets in the first `cutoff` active ids| / |targets|,
    from the real `_build_recall_block(limit=cutoff)` (targets parsed from the
    active section)."""
    total = 0.0
    for q in queries:
        block = _build_recall_block(store, q.text, persona_dir=persona_dir, limit=cutoff)
        top = active_ids(block)[:cutoff]
        total += sum(1 for mid in top if mid in q.targets) / len(q.targets)
    return total / len(queries)


def add_semantic_decoys(store: MemoryStore, n: int = 2) -> list[Memory]:
    """`n` memories with no word in common with any control query, standing in
    for a conclusive semantic result (set (iv): the semantic-present turn type)."""
    decoys = []
    for i in range(n):
        m = Memory.create_new(
            content=f"Canary hummed an unrelated lullaby number {i} beside the sleeping moon",
            memory_type="conversation",
            domain="us",
            importance=5.0,
        )
        store.create(m)
        decoys.append(m)
    return decoys


def recall_at_with_semantic(
    store: MemoryStore,
    queries: list[ControlQuery],
    decoys: list[Memory],
    *,
    cutoff: int,
    persona_dir: Path,
) -> float:
    """Like `recall_at`, but `run_semantic_recall` returns a fixed conclusive
    result made of `decoys` on EVERY turn, so the block is a semantic-present
    turn (cutoff 9). The same fake semantic stage feeds both versions, which is
    what lets the base code (whose conclusive semantic result suppresses the
    keyword search) and this build be compared on the same turn."""
    result = SemanticRecallResult(
        full=list(decoys), snippet=[], scores={m.id: 5.0 for m in decoys}
    )
    total = 0.0
    with patch("brain.chat.prompt.run_semantic_recall", return_value=result):
        for q in queries:
            block = _build_recall_block(store, q.text, persona_dir=persona_dir, limit=cutoff)
            top = active_ids(block)[:cutoff]
            total += sum(1 for mid in top if mid in q.targets) / len(q.targets)
    return total / len(queries)

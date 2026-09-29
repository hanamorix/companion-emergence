"""Consolidation gate — the idle-tick two-pass memory consolidator.

TEMP (Root 2 stopgap — remove when the Phase 5 dream cycle lands to replace it).

Runs FIRST on the idle heartbeat tick (before reflex/dream/research). Drains the
pending-candidate queue and, per candidate, discards (reject), promotes
(``store.create`` into memories.db), or merges into an existing memory. Because a
rejected candidate was never a memories.db row, none of grief/forgetting/hebbian
applies to it — the "terminal fate of a rejected candidate" problem is dissolved
by the separate-queue architecture.

Concurrency: ``run_tick`` fires from unsynchronised threads (background supervisor,
session-close worker, CLI), so two gate runs can overlap. A non-blocking persona
gate lock serialises them — a contended run SKIPS (the next tick catches up), so
the one non-idempotent action (merge into a committed memory) is never double-folded.

Pass 2 uses a classifier: tests inject a stub; production builds a Haiku-backed one
from a `TIER_BACKGROUND_CLASSIFIER` provider the caller constructs (#154) — no longer
the shared heartbeat/chat provider. The classifier's *decision quality* is advisory
(magnitude/quality deferred to the monologue-volume tune); the gate MECHANISM
(dispatch on the verdict) is what the gating criteria verify.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np

from brain import prompt_strings
from brain.memory import embeddings as embeddings_mod
from brain.memory.embedding_matrix import build_embedding_matrix
from brain.memory.embeddings import cosine_similarity
from brain.memory.hebbian import HebbianMatrix
from brain.memory.known_names import admit_names
from brain.memory.pending import SALIENCE_ELIGIBLE_TYPES, PendingQueue
from brain.memory.semantic_recall import build_semantic_candidate_pool
from brain.memory.store import Memory, MemoryStore, clamp_importance
from brain.utils.file_lock import file_lock

logger = logging.getLogger(__name__)

# Text externalized to prompt_strings.toml [engines.consolidation] (issue #129 stage 2c).
_REAPPRAISER_PROMPT = prompt_strings.register("engines.consolidation.reappraiser_prompt")
_CLASSIFIER_PROMPT = prompt_strings.register("engines.consolidation.classifier_prompt")

_GATE_LOCK_FILENAME = "consolidation_gate"  # file_lock adds the .lock sidecar
_ARCHIVE_FILENAME = "consolidation_archive.jsonl"

ASSOC_WEIGHT = 5.0  # hebbian edge weight for corrections/continuations (tune via L-B)

VERDICTS = frozenset(
    {"duplicate", "merge", "distinct", "correction", "continuation", "new"}
)


@dataclass
class Decision:
    """A Pass-2 verdict for one candidate.

    verdict: one of VERDICTS.
    target_id: the existing memory id for merge/correction/continuation.
    merged_content: for a merge, the classifier's surgical-edit result; when
        absent the gate applies a conservative loss-free fold (append the new
        fact) so both facts survive.
    names: names / proper nouns the judge found in the CANDIDATE text (spec §5,
        "Names are added three ways", item 1). Written to the known-names list by
        `_dispatch` for every verdict except "duplicate" (on "merge" too, from the
        candidate text). Empty when the judge returned none or its `names` field
        was absent or malformed (never a reason to lose the verdict).
    """

    verdict: str
    target_id: str | None = None
    merged_content: str | None = None
    names: tuple[str, ...] = ()


Classifier = Callable[[Memory, list[Memory]], Decision]


class Reappraisal(NamedTuple):
    """What a Reappraiser returns for one memory: a fresh importance and the
    names / proper nouns it found in the memory's text (spec §5, item 2)."""

    importance: float
    names: tuple[str, ...] = ()


# P3 retention rework, Change 3: importance re-rating on recall, via the
# pending queue. A Reappraiser judges a fresh importance (and finds names) for
# an EXISTING memory (not a Pass-2 candidate decision) — same injectable-seam
# shape as Classifier, so tests pass a fake returning a known value. Since the
# name-recall fix it returns `Reappraisal(importance, names)`, no longer a bare
# number.
Reappraiser = Callable[[Memory], Reappraisal]


@dataclass
class ConsolidationResult:
    skipped: bool = False
    batch: int = 0
    exact_dropped: int = 0
    salience_dropped: int = 0
    duplicates: int = 0
    promoted: int = 0
    merged: int = 0
    corrections: int = 0
    continuations: int = 0
    deferred: int = 0
    reappraised: int = 0

    def as_log(self) -> dict:
        return {
            "gate_batch": self.batch,
            "exact_dropped": self.exact_dropped,
            "salience_dropped": self.salience_dropped,
            "duplicates": self.duplicates,
            "promoted": self.promoted,
            "merged": self.merged,
            "corrections": self.corrections,
            "continuations": self.continuations,
            "deferred": self.deferred,
            "reappraised": self.reappraised,
            "skipped": self.skipped,
        }


def _normalize(text: str) -> str:
    """Whitespace-collapse + casefold — the exact-repeat equivalence key."""
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def _promote_all_classifier(_cand: Memory, _context: list[Memory]) -> Decision:
    """Degraded default when no provider is available: promote everything.

    Used only where the gate is invoked without a provider AND without an
    injected classifier — it makes the gate a safe no-op consolidator (exact-dup
    + salience Pass-1 only). Production supplies the Haiku classifier; tests
    inject a stub. Logged so a silent degrade is visible.
    """
    return Decision("new")


def run_consolidation(
    store: MemoryStore,
    *,
    persona_dir: str | Path,
    classifier: Classifier | None = None,
    provider=None,
    hebbian: HebbianMatrix | None = None,
    salience_floor: float | None = None,
    reappraiser: Reappraiser | None = None,
) -> ConsolidationResult:
    """Drain the pending queue and consolidate, under a non-blocking gate lock.

    Returns a ConsolidationResult; ``skipped=True`` when a concurrent gate holds
    the lock. Does not raise on a bad candidate (skips it).

    reappraiser: Change 3's injectable importance re-appraiser. None (default)
        builds the Haiku-backed default when `provider` is given, else
        degrades to the no-op fallback (importance unchanged) — mirrors the
        classifier's degrade-to-`_promote_all_classifier` pattern.
    """
    persona_dir = Path(persona_dir)
    lock_path = persona_dir / _GATE_LOCK_FILENAME
    persona_dir.mkdir(parents=True, exist_ok=True)
    with file_lock(lock_path, blocking=False) as acquired:
        if not acquired:
            logger.debug("consolidation gate: lock contended — skipping this run")
            return ConsolidationResult(skipped=True)
        if classifier is None:
            if provider is not None:
                classifier = _make_haiku_classifier(provider)
            else:
                logger.warning(
                    "consolidation gate: no classifier and no provider — "
                    "degrading to promote-all (Pass-1 dedup only)"
                )
                classifier = _promote_all_classifier
        if reappraiser is None:
            reappraiser = _make_haiku_reappraiser(provider) if provider is not None else _noop_reappraiser
        result = _run_locked(store, persona_dir, classifier, hebbian, salience_floor, reappraiser)
    logger.info("consolidation gate run: %s", json.dumps(result.as_log()))
    return result


def _is_reappraise_item(entry: dict) -> bool:
    return isinstance(entry, dict) and entry.get("_route") == "reappraise_importance"


def _run_locked(
    store: MemoryStore,
    persona_dir: Path,
    classifier: Classifier,
    hebbian: HebbianMatrix | None,
    salience_floor: float | None,
    reappraiser: Reappraiser | None = None,
) -> ConsolidationResult:
    pending = PendingQueue(persona_dir)
    batch = pending.drain()

    # Change 3: split out existing-memory re-appraise items BEFORE the normal
    # Pass-1/Pass-2 candidate pipeline — they are not candidates (no dedup,
    # no promotion), just an in-place importance UPDATE on an already-committed
    # row. `result.batch` counts only true candidates, matching its pre-Change-3
    # meaning.
    reappraise_entries = [e for e in batch if _is_reappraise_item(e)]
    candidate_entries = [e for e in batch if not _is_reappraise_item(e)]

    result = ConsolidationResult(batch=len(candidate_entries))
    if reappraise_entries:
        _handle_reappraisals(
            store, persona_dir, reappraise_entries, reappraiser or _noop_reappraiser, result
        )
    if not candidate_entries:
        return result

    candidates: list[Memory] = []
    for entry in candidate_entries:
        try:
            candidates.append(Memory.from_dict(entry))
        except (KeyError, ValueError, TypeError):
            logger.warning("consolidation gate: dropping unparseable candidate")

    # --- Pass 1: exact-dup + (scoped) salience; cluster is implicit per-candidate.
    survivors: list[Memory] = []
    seen_norm: set[str] = set()
    for cand in candidates:
        norm = _normalize(cand.content)
        if norm and norm in seen_norm:  # within-batch exact repeat
            result.exact_dropped += 1
            continue
        if norm and _has_exact_existing(store, cand.content, norm):
            result.exact_dropped += 1
            seen_norm.add(norm)
            continue
        if (
            salience_floor is not None
            and cand.memory_type in SALIENCE_ELIGIBLE_TYPES
            and float(cand.importance) < salience_floor
        ):
            result.salience_dropped += 1
            continue
        if norm:
            seen_norm.add(norm)
        survivors.append(cand)

    # --- Pass 2: per-candidate decision against related existing memories.
    for cand in survivors:
        # F1 #259 F3: embed the candidate ONCE, here, before the judge runs
        # (one step earlier than the old "embed at promotion" timing) — the
        # vector feeds the cosine `_related_existing` retrieval below AND is
        # reused (never recomputed) at promotion in `_dispatch`. None on a
        # provider failure (fail-soft): `_related_existing` degrades to its
        # lexical fallback and `_dispatch` degrades to its own `embed_row`
        # compute, exactly matching pre-F3 behavior for that one candidate.
        embedded = _embed_candidate(cand.content)
        context = _related_existing(store, cand, embedded)
        try:
            decision = classifier(cand, context)
        except Exception:  # noqa: BLE001 — a classifier fault must not lose the batch
            logger.exception("consolidation gate: classifier raised; promoting candidate")
            decision = Decision("new")
        _dispatch(store, pending, hebbian, persona_dir, cand, decision, result, embedded)
    return result


def _embed_candidate(content: str) -> tuple[np.ndarray, str] | None:
    """Embed a Pass-2 candidate's content via the process-cached production
    provider (F1 #259 F3), returning `(vector, model_id)`.

    Looked up via the MODULE (not a bare imported name) so a test's
    monkeypatch on `embeddings_mod.build_embedding_provider` is honored —
    mirrors `MemoryStore.embed_row`'s / `dedupe.is_duplicate`'s identical
    dynamic lookup.

    Fail-soft: returns None on any provider/compute failure rather than
    raising, so one bad embed cannot lose the candidate or abort the drain
    tick. `_related_existing` treats None as "no cosine context available"
    (lexical fallback) and `_dispatch`'s promote branch treats it as "no
    precomputed vector" (falls back to its own `embed_row` compute, leaving
    the row NULL on a repeat failure for the idle backfill to pick up).
    """
    try:
        provider = embeddings_mod.build_embedding_provider()
        vec = provider.embed(content).astype("float32")
        return vec, provider.model_id()
    except Exception:  # noqa: BLE001 — degrade to lexical context + post-promote embed_row
        logger.warning(
            "consolidation gate: pre-judge candidate embed failed; falling back to "
            "lexical _related_existing context and a post-promote embed_row compute",
            exc_info=True,
        )
        return None


def _has_exact_existing(store: MemoryStore, content: str, norm: str) -> bool:
    """True if a memories.db row is exact-identical (normalized) to `content`.

    Non-recall-bumping (``bump=False``). Uses a content snippet as the LIKE
    probe, then confirms full normalized equality.
    """
    snippet = content.strip()[:200]
    if not snippet:
        return False
    try:
        hits = store.search_text(snippet, active_only=True, limit=20, bump=False)
    except ValueError:
        return False
    return any(_normalize(h.content) == norm for h in hits)


def _related_existing(
    store: MemoryStore,
    cand: Memory,
    embedded: tuple[np.ndarray, str] | None,
    *,
    limit: int = 8,
) -> list[Memory]:
    """Gather existing COMMITTED memories to surface to the Pass-2 Haiku
    judge as near-dup context (F1 #259 F3, spec §4b).

    COSINE top-k retrieval from the warm matrix when the candidate has a
    vector (`embedded` is not None, see `_embed_candidate`) AND the matrix
    actually holds at least one comparison vector — the same short-locked
    matrix reader `run_semantic_recall`/`build_semantic_candidate_pool` use,
    so this inherits their per-item short-lock discipline (no torn read;
    covered by the acceptance-#3 concurrency test). NO similarity
    threshold anywhere here (I3) — only the `limit` breadth, same knob the
    prior lexical probe had; the Haiku judge makes the keep/merge/reject
    call from whatever neighbors come back.

    Falls back to the PRIOR lexical token-overlap probe
    (`_related_existing_lexical`) when: the candidate could not be embedded
    (`embedded is None`), the warm matrix is cold / holds no active
    embedded rows yet (empty candidate pool — same "graceful warm-up"
    degrade `build_semantic_candidate_pool`'s other callers already have),
    or the cosine retrieval itself raises. Read-only, non-bumping — Pass-2
    context, not recall.
    """
    if embedded is not None:
        cand_vec, _model_id = embedded
        try:
            hits = _related_existing_cosine(store, cand_vec, limit=limit)
            if hits:
                return hits
        except Exception:  # noqa: BLE001 — degrade to lexical, never lose Pass-2 context
            logger.warning(
                "consolidation gate: cosine _related_existing failed; "
                "falling back to lexical token-overlap",
                exc_info=True,
            )
    return _related_existing_lexical(store, cand, limit=limit)


def _related_existing_cosine(
    store: MemoryStore, cand_vec: np.ndarray, *, limit: int
) -> list[Memory]:
    """Top-`limit` most cosine-similar active, embedded, committed memories
    to `cand_vec` — the same matrix-read + score-sort-slice shape
    `run_semantic_recall` uses (reusing `build_semantic_candidate_pool` for
    the matrix half, not reinventing it). Empty list when the matrix has no
    comparison vectors yet (cold-start / pre-backfill)."""
    matrix = build_embedding_matrix(store.db_path)
    pool = build_semantic_candidate_pool(store, matrix)
    if not pool:
        return []
    scored = [(mem, cosine_similarity(cand_vec, vec)) for mem, vec in pool.values()]
    scored.sort(key=lambda pair: -pair[1])
    return [mem for mem, _score in scored[:limit]]


def _related_existing_lexical(store: MemoryStore, cand: Memory, *, limit: int = 8) -> list[Memory]:
    """Gather existing memories sharing salient tokens with the candidate.

    The PRE-F3 cosine-free probe, kept as `_related_existing`'s fallback
    for a cold matrix / a failed candidate embed (F1 #259 F3, spec §4b).
    Read-only, non-bumping — Pass-2 context, not recall.
    """
    tokens = [w for w in re.findall(r"\w+", cand.content or "") if len(w) > 3][:6]
    seen: set[str] = set()
    out: list[Memory] = []
    for tok in tokens:
        try:
            hits = store.search_text(tok, active_only=True, limit=3, bump=False)
        except ValueError:
            continue
        for h in hits:
            if h.id not in seen:
                seen.add(h.id)
                out.append(h)
                if len(out) >= limit:
                    return out
    return out


def _noop_reappraiser(memory: Memory) -> Reappraisal:
    """No-provider fallback (Change 3 default when no provider is configured):
    importance unchanged, no names. There is NO mechanism-driven monotone climb —
    this is what dissolves the recall-frequency-ratchet risk at the root (STAGE-3
    CORRECTION finding #2)."""
    return Reappraisal(memory.importance)


def _parse_names(value: object, *, source: str) -> tuple[str, ...]:
    """Fail-soft reading of a model's `names` field: a list of strings.

    Absent (None) is normal and silent. Anything else that is not a list, and any
    list element that is not a non-blank string, is dropped with a log line;
    nothing here raises, so a malformed `names` can never cost the caller its
    verdict or its score. Order is kept; duplicates are the admission function's
    business.
    """
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        logger.warning(
            "consolidation %s: ignoring a non-list 'names' field (%s)", source, type(value).__name__
        )
        return ()
    kept: list[str] = []
    dropped = 0
    for item in value:
        if isinstance(item, str) and item.strip():
            kept.append(item.strip())
        else:
            dropped += 1
    if dropped:
        logger.warning(
            "consolidation %s: dropped %d malformed entries from 'names'", source, dropped
        )
    return tuple(kept)


def _admit_extracted_names(
    persona_dir: Path, names: object, *, list_source: str, label: str
) -> None:
    """Write model-extracted names to the known-names list through the one
    admission function (S70: stopword entries are rejected there). Fail-soft: a
    failure is logged and never propagates, so a names write can never lose the
    candidate or the importance update it rides on (P-17, P-18)."""
    try:
        cleaned = _parse_names(names, source=label)
        if cleaned:
            admit_names(persona_dir, cleaned, list_source)
    except Exception:  # noqa: BLE001 — names are a side channel, never a reason to lose the work
        logger.warning("consolidation %s: known-names write failed", label, exc_info=True)


def _coerce_score(value: object) -> float | None:
    """A finite number from a JSON value (int, float or numeric string), else None."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


_FIRST_NUMBER_RE = re.compile(r"-?\d+(\.\d+)?")


def _parse_reappraisal(raw: str, current: float) -> Reappraisal:
    """Read a re-appraiser reply: a JSON object with `importance` and `names`.

    JSON first (P-18). A reply with no parseable JSON object falls back to
    today's first-number regex with no names (the old "reply with ONLY the number"
    shape still works). A JSON object whose `importance` is missing or not a
    number leaves the importance unchanged (the regex is NOT used there: it would
    read a digit out of a name). Malformed `names` never changes the score.
    """
    data: object = None
    if re.search(r"\{.*\}", raw, re.DOTALL):
        try:
            data = json.loads(_extract_json(raw))
        except ValueError:
            data = None
    if isinstance(data, dict):
        score = _coerce_score(data.get("importance"))
        try:
            names = _parse_names(data.get("names"), source="reappraiser")
        except Exception:  # noqa: BLE001 — defensive: names never break the score
            names = ()
        return Reappraisal(current if score is None else score, names)
    match = _FIRST_NUMBER_RE.search(raw)
    if not match:
        return Reappraisal(current)
    return Reappraisal(float(match.group(0)))


def _make_haiku_reappraiser(provider) -> Reappraiser:
    """Build the Haiku-backed default importance re-appraiser (Change 3).

    Judges importance fresh from the memory's content in the moment it is
    re-appraised, not from how often it has been recalled: a still-relevant
    journal entry keeps or raises its importance, a past-due appointment
    (Roy's jury-duty case) drops. It reads the WHOLE memory (the old
    first-400-character cut is gone, spec §5) and also returns the names found in
    it. Output is clamped [0, 10] by the caller (`_handle_reappraisals`) via the
    shared `clamp_importance`. On any parse/provider failure the fallback is the
    memory's CURRENT importance with no names (a no-op) — never a crash, never a
    ratchet."""
    prompt = _REAPPRAISER_PROMPT

    def _reappraise(memory: Memory) -> Reappraisal:
        try:
            raw = provider.generate(memory.content, system=prompt)
            return _parse_reappraisal(raw, memory.importance)
        except Exception:  # noqa: BLE001 — a provider fault must not lose the row
            logger.warning("consolidation Haiku reappraise failed; importance unchanged")
            return Reappraisal(memory.importance)

    return _reappraise


def _handle_reappraisals(
    store: MemoryStore,
    persona_dir: Path,
    entries: list[dict],
    reappraiser: Reappraiser,
    result: ConsolidationResult,
) -> None:
    """Process Change-3 existing-memory re-appraise items: load the target
    row, compute a new importance (and names) via the injectable `reappraiser`,
    and UPDATE it in place (never an INSERT — C3.1).

    A row missing at the read (already gone before this tick) or hard-deleted
    between the read and the write (a concurrent forgetting LOSE, or a delete
    triggered mid-appraisal) is skipped: no resurrection, no crash (C3.2).
    The `store.update` call is wrapped in `try/except KeyError`, mirroring
    the merge-dispatch idiom above (`_dispatch`'s `verdict == "merge"` branch).

    The names the reappraiser found are written to the known-names list only
    after the importance update succeeded (P-18), fail-soft.
    """
    for entry in entries:
        memory_id = entry.get("memory_id")
        if not memory_id or not isinstance(memory_id, str):
            logger.warning("consolidation gate: dropping unparseable reappraise item")
            continue
        memory = store.get(memory_id, bump=False)
        if memory is None:
            continue  # already gone — no resurrection
        try:
            new_importance, names = reappraiser(memory)
        except Exception:  # noqa: BLE001 — a reappraiser fault must not lose the batch
            logger.exception("consolidation gate: reappraiser raised; skipping")
            continue
        try:
            store.update(memory_id, importance=clamp_importance(new_importance))
        except KeyError:
            # Deleted between the read above and this write — absorbed, not
            # raised (finding #3).
            continue
        result.reappraised += 1
        _admit_extracted_names(persona_dir, names, list_source="reappraiser", label="reappraiser")


def _archive_preimage(persona_dir: Path, target: Memory, source_id: str) -> None:
    """Append the pre-merge memory to the plain consolidation archive (lossless
    before lossy — NOT the grief graveyard)."""
    rec = {
        "archived_at": datetime.now(UTC).isoformat(),
        "reason": "consolidation_merge",
        "target": target.to_dict(),
        "merged_from_candidate": source_id,
    }
    path = persona_dir / _ARCHIVE_FILENAME
    with file_lock(path):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _dispatch(
    store: MemoryStore,
    pending: PendingQueue,
    hebbian: HebbianMatrix | None,
    persona_dir: Path,
    cand: Memory,
    decision: Decision,
    result: ConsolidationResult,
    embedded: tuple[np.ndarray, str] | None = None,
) -> None:
    verdict = decision.verdict if decision.verdict in VERDICTS else "new"

    if verdict == "duplicate":
        result.duplicates += 1  # discard: candidate was already removed by drain()
        return

    # Names the judge found in the candidate text: every verdict except
    # "duplicate", a deferred merge included (P-17; `INSERT OR IGNORE` makes the
    # repeat on the re-judged tick harmless). Fail-soft, before the verdict acts.
    _admit_extracted_names(persona_dir, decision.names, list_source="gate", label="gate judge")

    if verdict == "merge":
        target = store.get(decision.target_id) if decision.target_id else None
        if target is None or target.state == "fading":
            # Target missing or mid-fade: defer — re-enqueue for the next tick
            # rather than lose the candidate or clobber a concurrent fade.
            pending.enqueue(cand, source="gate-deferred")
            result.deferred += 1
            return
        _archive_preimage(persona_dir, target, cand.id)
        merged = decision.merged_content
        if not merged:
            merged = _fold(target.content, cand.content)
        try:
            store.update(target.id, content=merged)
        except KeyError:
            # Target hard-deleted between the read and the write (a concurrent
            # forgetting LOSE). Defer the candidate; the archive record is
            # harmless residue.
            pending.enqueue(cand, source="gate-deferred")
            result.deferred += 1
            return
        result.merged += 1
        return

    # promote (distinct / new / correction / continuation) → real memories.db row
    if verdict == "correction" and decision.target_id:
        cand.metadata = {**cand.metadata, "correction_of": decision.target_id}
    store.create(cand)
    # F1 #259 F3: embed-on-write at the pending-queue -> committed-memory
    # promotion — the steady-state embedding path (the idle backfill only
    # mops up rows this misses). The vector itself was already computed
    # PRE-judge (`_embed_candidate`, in `_run_locked`'s Pass-2 loop) so it
    # can feed the cosine `_related_existing` retrieval; REUSE it here via
    # `store.write_embedding` rather than recomputing (no double-embed).
    # When `embedded` is None (the pre-judge embed itself failed), fall back
    # to the original `store.embed_row` compute-and-persist path. Either
    # way: local try/except — a failed embed must never abort the drain
    # tick or lose the just-promoted candidate — leave the row's embedding
    # NULL and let the later idle backfill pick it up.
    try:
        if embedded is not None:
            cand_vec, model_id = embedded
            store.write_embedding(cand.id, cand_vec, model_id)
        else:
            store.embed_row(cand.id, cand.content)
    except Exception:  # noqa: BLE001 — degrade to backfill, never abort the drain
        logger.warning(
            "consolidation._dispatch: embed persist failed for promoted id=%s — "
            "leaving embedding NULL for the idle backfill",
            cand.id,
            exc_info=True,
        )
    if verdict in ("correction", "continuation") and decision.target_id and hebbian is not None:
        hebbian.set_edge_weight(cand.id, decision.target_id, ASSOC_WEIGHT)

    if verdict == "correction":
        result.corrections += 1
    elif verdict == "continuation":
        result.continuations += 1
    else:
        result.promoted += 1


def _fold(existing: str, addition: str) -> str:
    """Conservative loss-free fold: keep the existing memory and append the new
    fact if it is not already present. Not a rewrite — the real surgical edit is
    the classifier's ``merged_content``; this is the safe default."""
    if _normalize(addition) in _normalize(existing):
        return existing
    return f"{existing.rstrip()}\n{addition.strip()}"


def _make_haiku_classifier(provider) -> Classifier:
    """Build a Haiku-backed Pass-2 classifier from a generation provider.

    Decision QUALITY is advisory (C16, live-only) — the gate mechanism is what
    the gating criteria verify. On any parse/provider failure the candidate is
    promoted (fail-open toward keeping content).
    """
    prompt = _CLASSIFIER_PROMPT

    def _classify(cand: Memory, context: list[Memory]) -> Decision:
        ctx = "\n".join(f"- id={m.id}: {m.content[:200]}" for m in context) or "(none)"
        # The WHOLE candidate: the verdict and the names are both read from all of
        # it (the first-400-character cut is gone, spec §5 item 1, S51).
        user = f"CANDIDATE: {cand.content}\nEXISTING:\n{ctx}"
        try:
            raw = provider.generate(user, system=prompt)
            data = json.loads(_extract_json(raw))
            verdict = str(data.get("verdict", "new"))
            if verdict not in VERDICTS:
                verdict = "new"
            return Decision(
                verdict=verdict,
                target_id=data.get("target_id") or None,
                merged_content=data.get("merged_content") or None,
                names=_parse_names(data.get("names"), source="gate judge"),
            )
        except Exception:  # noqa: BLE001
            logger.warning("consolidation Haiku classify failed; promoting candidate")
            return Decision("new")

    return _classify


def _extract_json(text: str) -> str:
    """Pull the first {...} JSON object out of a model reply."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else "{}"

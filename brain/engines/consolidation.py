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
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np

from brain import prompt_strings
from brain.bridge.cli_throttle import ThrottleDeferred
from brain.health.json_io import append_jsonl
from brain.memory import embeddings as embeddings_mod
from brain.memory.embedding_matrix import build_embedding_matrix
from brain.memory.embeddings import cosine_similarity
from brain.memory.hebbian import HebbianMatrix
from brain.memory.known_names import (
    admit_names,
    listed_display_forms,
    load_known_names,
    match_known_names,
    normalize_name,
    remove_names,
)
from brain.memory.pending import SALIENCE_ELIGIBLE_TYPES, ClaimedBatch, PendingQueue
from brain.memory.semantic_recall import build_semantic_candidate_pool
from brain.memory.store import Memory, MemoryStore, clamp_importance
from brain.utils.failure_scope import ServiceCallFailed, settle_scope
from brain.utils.file_lock import file_lock

logger = logging.getLogger(__name__)

# Text externalized to prompt_strings.toml [engines.consolidation] (issue #129 stage 2c).
_REAPPRAISER_PROMPT = prompt_strings.register("engines.consolidation.reappraiser_prompt")
_CLASSIFIER_PROMPT = prompt_strings.register("engines.consolidation.classifier_prompt")
# The user-message templates (name-recall fix, names increment: S106 fences, S107,
# S136 one <existing> fence per EXISTING memory). Filled with str.format; memory
# text is an argument, so braces inside it are inert.
_CLASSIFIER_USER_TEMPLATE = prompt_strings.register(
    "engines.consolidation.classifier_user_template"
)
_CLASSIFIER_EXISTING_ITEM_TEMPLATE = prompt_strings.register(
    "engines.consolidation.classifier_existing_item_template"
)
_REAPPRAISER_USER_TEMPLATE = prompt_strings.register(
    "engines.consolidation.reappraiser_user_template"
)

_GATE_LOCK_FILENAME = "consolidation_gate"  # file_lock adds the .lock sidecar
_ARCHIVE_FILENAME = "consolidation_archive.jsonl"
_QUARANTINE_FILENAME = "consolidation_quarantine.jsonl"

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
        was absent or malformed (a malformed VALUE never costs the verdict; a reply
        that is not JSON at all still promotes the candidate, the classifier's
        fail-open-toward-keeping-content rule).
    """

    verdict: str
    target_id: str | None = None
    merged_content: str | None = None
    names: tuple[str, ...] = ()


Classifier = Callable[[Memory, list[Memory]], Decision]


class Reappraisal(NamedTuple):
    """What a Reappraiser returns for one memory: a fresh importance, the names /
    proper nouns it found in the memory's text (spec §5, item 2), and the listed
    names it was shown (KNOWN NAMES) that it judges are not names (``not_names``,
    S105, S107)."""

    importance: float
    names: tuple[str, ...] = ()
    not_names: tuple[str, ...] = ()


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
    failed: int = 0  # items whose work raised (requeued, quarantined or already saved), #459
    quarantined: int = 0  # of those, the ones set aside in the quarantine file

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
            "failed": self.failed,
            "quarantined": self.quarantined,
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
    the lock. Does not raise on a bad candidate: the batch is CLAIMED (kept in a
    processing file until each item is settled), and an item whose work raises is
    requeued (the failure depends on the store or filesystem; the pass then ends and
    the rest wait for the next run), quarantined in ``consolidation_quarantine.jsonl``
    (it would fail the same way again), or left alone (already saved). A deferred
    Claude call (expired login) means "not now": nothing is promoted unclassified, the
    item and the rest stay for the next run. A run killed midway is recovered by the
    next one. ``result.failed`` / ``result.quarantined`` count them.

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
            reappraiser = (
                _make_haiku_reappraiser(provider, persona_dir)
                if provider is not None
                else _noop_reappraiser
            )
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
    # Claimed, not drained: the batch stays in a processing file until each item is settled, so an exception or a kill
    # during the slow per-item work below cannot lose it (#459).
    claim = pending.claim()
    batch = claim.entries

    # Change 3: split out existing-memory re-appraise items BEFORE the normal
    # Pass-1/Pass-2 candidate pipeline — they are not candidates (no dedup,
    # no promotion), just an in-place importance UPDATE on an already-committed
    # row. `result.batch` counts only true candidates, matching its pre-Change-3
    # meaning.
    reappraise_entries = [e for e in batch if _is_reappraise_item(e)]
    candidate_entries = [e for e in batch if not _is_reappraise_item(e)]

    result = ConsolidationResult(batch=len(candidate_entries))
    if reappraise_entries and _handle_reappraisals(
        store, persona_dir, pending, claim, reappraise_entries, reappraiser or _noop_reappraiser, result
    ):
        claim.close()
        return result
    if not candidate_entries:
        claim.close()
        return result

    candidates: list[tuple[dict, Memory]] = []
    for entry in candidate_entries:
        try:
            candidates.append((entry, Memory.from_dict(entry)))
        except (KeyError, ValueError, TypeError):
            logger.warning("consolidation gate: dropping unparseable candidate")
            claim.settle(entry)  # a deliberate drop (#432), not a failure

    # --- Pass 1: exact-dup + (scoped) salience; cluster is implicit per-candidate.
    survivors: list[tuple[dict, Memory]] = []
    seen_norm: set[str] = set()
    ended = False
    for entry, cand in candidates:
        try:
            norm = _normalize(cand.content)
            if norm and norm in seen_norm:  # within-batch exact repeat
                result.exact_dropped += 1
                claim.settle(entry)
                continue
            if norm and _has_exact_existing(store, cand.content, norm):
                result.exact_dropped += 1
                seen_norm.add(norm)
                claim.settle(entry)
                continue
            if (
                salience_floor is not None
                and cand.memory_type in SALIENCE_ELIGIBLE_TYPES
                and float(cand.importance) < salience_floor
            ):
                result.salience_dropped += 1
                claim.settle(entry)
                continue
            if norm:
                seen_norm.add(norm)
            survivors.append((entry, cand))
        except Exception as exc:  # noqa: BLE001 — settled per item; never ends the loop with items undone
            if _settle_failed_item(store, pending, persona_dir, claim, entry, cand, "pass1", exc, result):
                ended = True
                break

    # --- Pass 2: per-candidate decision against related existing memories.
    for entry, cand in [] if ended else survivors:
        decision: Decision | None = None
        try:
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
            except ThrottleDeferred:
                raise  # "not now" (expired login, shutdown): never promote an unclassified candidate (#473)
            except Exception as call_exc:
                # The classifier call failed for ANY reason (the one call let through during an expired login, a CLI error, a
                # timeout): the candidate was not classified, so it is NOT promoted. It is kept for a later run and the pass
                # ends (one call per tick). A reply that arrives but cannot be read still promotes: see `_make_haiku_classifier`.
                raise ServiceCallFailed(f"classifier call failed: {type(call_exc).__name__}: {call_exc}") from call_exc
            decision = _clean_decision(decision)
            _dispatch(store, pending, hebbian, persona_dir, cand, decision, result, embedded)
            claim.settle(entry)
        except Exception as exc:  # noqa: BLE001 — settled per item; never ends the loop with items undone
            # The decision's target is named in the log so a link lost to a failed post-save step can be rebuilt.
            target = f" target_id={decision.target_id}" if decision is not None and decision.target_id else ""
            if _settle_failed_item(store, pending, persona_dir, claim, entry, cand, "pass2", exc, result, target):
                break
    claim.close()
    return result


def _clean_decision(decision: Decision) -> Decision:
    """The Decision with its model-derived ids and text checked before they reach SQLite.

    ``target_id`` and ``merged_content`` come from arbitrary model JSON (the Haiku classifier reads them with ``data.get``). A
    list or dict ``target_id``, or a non-string ``merged_content``, would be bound as an SQL parameter and raise, and one
    stochastic reply would then put a good candidate in quarantine. ``None`` already means "defer" (no target) or "fold"
    (no merged text), so a bad value becomes ``None``.
    """
    target = decision.target_id if isinstance(decision.target_id, str) and decision.target_id else None
    merged = decision.merged_content if isinstance(decision.merged_content, str) else None
    if target == decision.target_id and merged == decision.merged_content:
        return decision
    return replace(decision, target_id=target, merged_content=merged)


def _saved_state(store: MemoryStore, cand: Memory) -> tuple[str, Memory] | None:
    """Is ``cand`` already a row? ``("saved", row)``, ``("conflict", row)`` or None (no row, or the probe itself failed).

    Candidate ids are uuid4, so a row with the same id is a prior save of this candidate. It is SAVED when the row's
    normalized content equals the candidate's, equals the row's ``content_snapshot`` (a faded row keeps the original
    there) or contains it (a merged row, the ``_fold`` rule). A same-id row matching none of those is a CONFLICT.
    """
    try:
        row = store.get(cand.id, bump=False)
    except Exception as exc:  # noqa: BLE001 — a failing probe means "not saved"; the classification goes on
        logger.warning(
            "consolidation gate: could not check whether candidate id=%s was saved (%s: %s); treating it as not saved",
            cand.id, type(exc).__name__, exc,
        )
        return None
    if row is None:
        return None
    cand_norm = _normalize(cand.content)
    row_norm = _normalize(row.content)
    snapshot_norm = _normalize(row.content_snapshot or "")
    if cand_norm == row_norm or (snapshot_norm and cand_norm == snapshot_norm) or (cand_norm and cand_norm in row_norm):
        return "saved", row
    return "conflict", row


def _quarantine(
    persona_dir: Path, entry: dict, stage: str, exc: BaseException, existing: Memory | None = None
) -> None:
    """Append ``entry`` to the quarantine file with the error as evidence. Raises if the append fails."""
    record: dict = {
        "quarantined_at": datetime.now(UTC).isoformat(),
        "stage": stage,
        "error": f"{type(exc).__name__}: {str(exc)[:500]}",
        "entry": {k: v for k, v in entry.items() if k != "_claim_id"},
    }
    if existing is not None:
        record["existing_content"] = existing.content
    append_jsonl(persona_dir / _QUARANTINE_FILENAME, record, nonfinite="null")


def _item_label(entry: dict, cand: Memory | None) -> str:
    return f"id={cand.id} type={cand.memory_type}" if cand is not None else f"memory_id={entry.get('memory_id')}"


def _settle_failed_item(
    store: MemoryStore,
    pending: PendingQueue,
    persona_dir: Path,
    claim: ClaimedBatch,
    entry: dict,
    cand: Memory | None,
    stage: str,
    exc: Exception,
    result: ConsolidationResult,
    detail: str = "",
) -> bool:
    """Decide what happens to an item whose work raised. Returns True to END the pass. Total: never raises.

    If settling itself fails (the quarantine file cannot be written, the requeue fails, anything), the entry is left
    UNSETTLED, so it stays in the processing file and the next run retries it; one ERROR says so, and the pass ends.
    """
    try:
        return _settle_failed_item_inner(store, pending, persona_dir, claim, entry, cand, stage, exc, result, detail)
    except Exception:  # noqa: BLE001 — total by design: the loop must never end with items undone
        logger.exception(
            "consolidation gate: could not settle a failed item (%s) after %s: %s; leaving it for the next run",
            _item_label(entry, cand), type(exc).__name__, str(exc)[:200],
        )
        return True


def _settle_failed_item_inner(
    store: MemoryStore,
    pending: PendingQueue,
    persona_dir: Path,
    claim: ClaimedBatch,
    entry: dict,
    cand: Memory | None,
    stage: str,
    exc: Exception,
    result: ConsolidationResult,
    detail: str = "",
) -> bool:
    label = _item_label(entry, cand) + detail
    if isinstance(exc, ThrottleDeferred):
        # "not now": not a failure. The item and the rest of the batch stay in the processing file; the next run retries.
        logger.info("consolidation gate: Claude call deferred (%s); leaving %s and the rest for the next run", exc, label)
        return True
    result.failed += 1
    existing: Memory | None = None
    if cand is not None:
        probe = _saved_state(store, cand)
        if probe is not None and probe[0] == "saved":
            # A step AFTER the save failed (an edge weight, say): the candidate is a row, nothing is requeued or quarantined.
            logger.warning(
                "consolidation gate: %s was saved but a later step failed (%s: %s); not requeued", label, type(exc).__name__, exc
            )
            claim.settle(entry)
            return False
        if probe is not None:
            existing = probe[1]
    scope = "item" if existing is not None else settle_scope(exc, store)
    if scope == "item":
        _quarantine(persona_dir, entry, stage, exc, existing)
        logger.error(
            "consolidation gate: %s quarantined (%s: %s)", label, type(exc).__name__, str(exc)[:200], exc_info=exc
        )
        result.quarantined += 1
        claim.settle(entry)
        return False
    # environment: keep it for later and end the pass (the rest stay in the processing file).
    source = entry.get("_source") or "gate-retry"
    if cand is not None:
        pending.enqueue(cand, source=source)
    else:
        pending.enqueue_reappraisal(str(entry.get("memory_id")), source=source, include_held=False)
    logger.warning(
        "consolidation gate: %s requeued after an environment failure (%s: %s); ending this pass",
        label, type(exc).__name__, str(exc)[:200],
    )
    claim.settle(entry)
    return True


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


def _escape_tag(text: str, tag: str) -> str:
    """Escape every tag of the fence's name inside memory text (S106): opening,
    closing, self-closing, with attributes, any case, whitespace inside the
    brackets, terminated or not (``<`` becomes ``&lt;``), so memory text can
    neither close its fence nor open another. ``<memo>`` and ``<memoryx>`` are
    left alone. Look-alike characters (a zero-width space inside the tag, a
    fullwidth ``<``) are not tags and are not escaped: a stated limit."""
    # One whitespace run before the optional slash and one after it (never two
    # adjacent \s* runs: linear, not quadratic, on a long run of spaces).
    return re.sub(
        r"<(?=\s*(?:/\s*)?" + re.escape(tag) + r"\b)", "&lt;", text, flags=re.IGNORECASE
    )


def _escape_tags(text: str, tags: tuple[str, ...]) -> str:
    for tag in tags:
        text = _escape_tag(text, tag)
    return text


# Both of the judge's fence tags are escaped in the candidate AND in every
# EXISTING memory (S136), so a candidate cannot forge an EXISTING block and an
# existing memory cannot close its own fence or open a candidate.
_JUDGE_TAGS: tuple[str, ...] = ("candidate", "existing")


def _classifier_user_message(cand: Memory, context: list[Memory]) -> str:
    """The judge's user message: the whole candidate in its <candidate> fence, then
    each EXISTING memory (the base's first 200 characters) in its own <existing>
    fence after an ``- id=<id>:`` line (S136), or ``(none)``."""
    items = [
        _CLASSIFIER_EXISTING_ITEM_TEMPLATE.format(
            id=m.id, content=_escape_tags(m.content[:200], _JUDGE_TAGS)
        )
        for m in context
    ]
    return _CLASSIFIER_USER_TEMPLATE.format(
        candidate=_escape_tags(cand.content, _JUDGE_TAGS),
        existing="\n".join(items) or "(none)",
    )


def _reappraiser_user_message(memory: Memory, offered: list[str]) -> str:
    """The re-appraiser's user message: KNOWN NAMES (the listed names that occur in
    the memory, display forms, a JSON array) then the whole memory in its
    <memory> fence (S105, S106, S107)."""
    known = json.dumps([_escape_tag(d, "memory") for d in offered], ensure_ascii=False)
    return _REAPPRAISER_USER_TEMPLATE.format(
        known_names=known, memory=_escape_tag(memory.content, "memory")
    )


def _admit_extracted_names(
    persona_dir: Path, names: object, *, list_source: str, label: str
) -> None:
    """Write model-extracted names to the known-names list through the one
    admission function (S144, S152, S153: admitted as given, up to the length limit; the
    rejects list still refuses a removed name, S135). Fail-soft: a
    failure is logged and never propagates, so a names write can never lose the
    candidate or the importance update it rides on (P-17, P-18)."""
    try:
        cleaned = _parse_names(names, source=label)
        if cleaned:
            admit_names(persona_dir, cleaned, list_source)
    except Exception:  # noqa: BLE001 — names are a side channel, never a reason to lose the work
        logger.warning("consolidation %s: known-names write failed", label, exc_info=True)


_NUMBER_RE = re.compile(r"-?\d+(\.\d+)?")  # today's first-number regex, unchanged
_KEYED_IMPORTANCE_RE = re.compile(r'"importance"\s*:\s*"?(-?\d+(?:\.\d+)?)"?')


def _coerce_score(value: object) -> float | None:
    """A finite number from a JSON value (a number, or a plain decimal string), else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        if not _NUMBER_RE.fullmatch(value.strip()):
            return None
    elif not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _parse_reappraisal(raw: str, current: float) -> Reappraisal:
    """Read a re-appraiser reply: a JSON object with `importance` and `names`.

    JSON first (P-18). A reply with no `{` at all falls back to today's
    first-number regex with no names (the old "reply with ONLY the number" shape
    still works). A reply that has a `{` but is not parseable JSON (a comma too
    many, a second object, a cut-off reply) is read only through the keyed
    `"importance": <number>` pattern: the first-number regex would read a digit
    out of a name ("Apollo 13"). A JSON object whose `importance` is missing or
    not a number leaves the importance unchanged, for the same reason. Malformed
    `names` never change the score.
    """
    if "{" not in raw:
        match = _NUMBER_RE.search(raw)
        return Reappraisal(float(match.group(0)) if match else current)
    data: object = None
    if re.search(r"\{.*\}", raw, re.DOTALL):
        try:
            data = json.loads(_extract_json(raw))
        except ValueError:
            data = None
    if isinstance(data, dict):
        score = _coerce_score(data.get("importance"))
        names = _parse_names(data.get("names"), source="reappraiser")
        not_names = _parse_names(data.get("not_names"), source="reappraiser not_names")
        return Reappraisal(current if score is None else score, names, not_names)
    keyed = _KEYED_IMPORTANCE_RE.search(raw)
    return Reappraisal(float(keyed.group(1)) if keyed else current)


def _make_haiku_reappraiser(provider, persona_dir: Path | None = None) -> Reappraiser:
    """Build the Haiku-backed default importance re-appraiser (Change 3).

    Judges importance fresh from the memory's content in the moment it is
    re-appraised, not from how often it has been recalled: a still-relevant
    journal entry keeps or raises its importance, a past-due appointment
    (Roy's jury-duty case) drops. It reads the WHOLE memory (the old
    first-400-character cut is gone, spec §5) and also returns the names found in
    it. Output is clamped [0, 10] by the caller (`_handle_reappraisals`) via the
    shared `clamp_importance`. On any parse/provider failure the fallback is the
    memory's CURRENT importance with no names (a no-op) — never a crash, never a
    ratchet.

    The memory goes in its <memory> fence after a KNOWN NAMES line: the display
    forms of the persona's listed names that occur in it (``[]`` without a persona
    directory or on any failure), so it can name any that is not a name in
    ``not_names`` (S105, S107)."""
    prompt = _REAPPRAISER_PROMPT

    def _reappraise(memory: Memory) -> Reappraisal:
        try:
            offered = listed_display_forms(persona_dir, memory.content)
            raw = provider.generate(_reappraiser_user_message(memory, offered), system=prompt)
            return _parse_reappraisal(raw, memory.importance)
        except ThrottleDeferred:
            raise  # "not now" (expired login, shutdown): the item is retried, not consumed unchanged (#473)
        except Exception:  # noqa: BLE001 — a provider fault must not lose the row
            logger.warning("consolidation Haiku reappraise failed; importance unchanged")
            return Reappraisal(memory.importance)

    return _reappraise


def _handle_reappraisals(
    store: MemoryStore,
    persona_dir: Path,
    pending: PendingQueue,
    claim: ClaimedBatch,
    entries: list[dict],
    reappraiser: Reappraiser,
    result: ConsolidationResult,
) -> bool:
    """Process Change-3 existing-memory re-appraise items: load the target
    row, compute a new importance (and names) via the injectable `reappraiser`,
    and UPDATE it in place (never an INSERT — C3.1).

    A row missing at the read (already gone before this tick) or hard-deleted
    between the read and the write (a concurrent forgetting LOSE, or a delete
    triggered mid-appraisal) is skipped: no resurrection, no crash (C3.2).
    The `store.update` call is wrapped in `try/except KeyError`, mirroring
    the merge-dispatch idiom above (`_dispatch`'s `verdict == "merge"` branch).

    The names the reappraiser found are written to the known-names list only
    after the importance update succeeded (P-18), fail-soft. Before that, its
    ``not_names`` remove listed names (S105): only a name that is listed, occurs
    in this memory's text (the matcher, entry-side rules), and is not also in its
    ``names``; the write itself never removes a name the kindled added, and puts
    each removed name on the rejects list (S135).

    Each item runs in its own try (#459): a failure is settled by `_settle_failed_item`, and True is returned when the pass
    must end (an environment failure or a deferred call); the unsettled items stay in the processing file.
    """
    for entry in entries:
        try:
            memory_id = entry.get("memory_id")
            if not memory_id or not isinstance(memory_id, str):
                logger.warning("consolidation gate: dropping unparseable reappraise item")
                claim.settle(entry)
                continue
            memory = store.get(memory_id, bump=False)
            if memory is None:
                claim.settle(entry)
                continue  # already gone — no resurrection
            try:
                appraisal = reappraiser(memory)
                # Attribute access inside the same try: a reappraiser that returns a
                # bare number is skipped with a log line, as before.
                new_importance = appraisal.importance
                names = appraisal.names
                not_names = getattr(appraisal, "not_names", ())
            except ThrottleDeferred:
                raise  # "not now": the item is not consumed (#473)
            except Exception:  # noqa: BLE001 — a reappraiser fault must not lose the batch
                logger.exception("consolidation gate: reappraiser raised; skipping")
                claim.settle(entry)
                continue
            try:
                store.update(memory_id, importance=clamp_importance(new_importance))
            except KeyError:
                # Deleted between the read above and this write — absorbed, not
                # raised (finding #3).
                claim.settle(entry)
                continue
            result.reappraised += 1
            _remove_not_names(persona_dir, memory, names, not_names)
            _admit_extracted_names(persona_dir, names, list_source="reappraiser", label="reappraiser")
            claim.settle(entry)
        except Exception as exc:  # noqa: BLE001 — settled per item
            if _settle_failed_item(store, pending, persona_dir, claim, entry, None, "reappraise", exc, result):
                return True
    return False


def _remove_not_names(persona_dir: Path, memory: Memory, names: object, not_names: object) -> None:
    """The re-appraiser's removals (S105), fail-soft: listed names that occur in
    this memory and that the reply put under ``not_names`` but not under
    ``names``. ``remove_names`` refuses a name the kindled added."""
    try:
        wanted = {normalize_name(n) for n in _parse_names(not_names, source="reappraiser not_names")}
        kept = {normalize_name(n) for n in _parse_names(names, source="reappraiser")}
        wanted = {n for n in wanted - kept if n}
        if not wanted:
            return
        present = set(match_known_names(memory.content or "", load_known_names(persona_dir)))
        targets = sorted(wanted & present)
        if targets:
            remove_names(persona_dir, targets, "reappraiser")
    except Exception:  # noqa: BLE001 — a names write never costs the importance update
        logger.warning("consolidation reappraiser: known-names removal failed", exc_info=True)


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
    append_jsonl(path, rec, nonfinite="null")


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
        # The WHOLE candidate: the verdict and the names are both read from all of
        # it (the first-400-character cut is gone, spec §5 item 1, S51), in its
        # fence; each EXISTING memory in its own fence (S106, S136).
        user = _classifier_user_message(cand, context)
        # The CALL is outside the fail-open try: a call that fails (a deferral, a CLI error, a timeout) means the candidate
        # was not classified, so the exception goes to the gate, which keeps the candidate for a later run (#473).
        raw = provider.generate(user, system=prompt)
        try:
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
        except Exception:  # noqa: BLE001 — a reply that cannot be read keeps the candidate's content (fail-open)
            logger.warning("consolidation Haiku reply could not be read; promoting candidate")
            return Decision("new")

    return _classify


def _extract_json(text: str) -> str:
    """Pull the first {...} JSON object out of a model reply."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else "{}"

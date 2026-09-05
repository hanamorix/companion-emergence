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
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.pending import SALIENCE_ELIGIBLE_TYPES, PendingQueue
from brain.memory.store import Memory, MemoryStore
from brain.utils.file_lock import file_lock

logger = logging.getLogger(__name__)

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
    """

    verdict: str
    target_id: str | None = None
    merged_content: str | None = None


Classifier = Callable[[Memory, list[Memory]], Decision]


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
) -> ConsolidationResult:
    """Drain the pending queue and consolidate, under a non-blocking gate lock.

    Returns a ConsolidationResult; ``skipped=True`` when a concurrent gate holds
    the lock. Does not raise on a bad candidate (skips it).
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
        result = _run_locked(store, persona_dir, classifier, hebbian, salience_floor)
    logger.info("consolidation gate run: %s", json.dumps(result.as_log()))
    return result


def _run_locked(
    store: MemoryStore,
    persona_dir: Path,
    classifier: Classifier,
    hebbian: HebbianMatrix | None,
    salience_floor: float | None,
) -> ConsolidationResult:
    pending = PendingQueue(persona_dir)
    batch = pending.drain()
    result = ConsolidationResult(batch=len(batch))
    if not batch:
        return result

    candidates: list[Memory] = []
    for entry in batch:
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
        context = _related_existing(store, cand)
        try:
            decision = classifier(cand, context)
        except Exception:  # noqa: BLE001 — a classifier fault must not lose the batch
            logger.exception("consolidation gate: classifier raised; promoting candidate")
            decision = Decision("new")
        _dispatch(store, pending, hebbian, persona_dir, cand, decision, result)
    return result


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


def _related_existing(store: MemoryStore, cand: Memory, *, limit: int = 8) -> list[Memory]:
    """Gather existing memories sharing salient tokens with the candidate.

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
) -> None:
    verdict = decision.verdict if decision.verdict in VERDICTS else "new"

    if verdict == "duplicate":
        result.duplicates += 1  # discard: candidate was already removed by drain()
        return

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
    prompt = (
        "You consolidate a companion's auto-generated memory candidates. Given a "
        "CANDIDATE and EXISTING related memories, reply with ONE JSON object: "
        '{"verdict": one of ["duplicate","merge","distinct","correction",'
        '"continuation","new"], "target_id": <existing id or null>, '
        '"merged_content": <string or null>}. '
        "duplicate=already fully captured; merge=near-duplicate adding info "
        "(give target_id + a minimal surgical merged_content); correction/"
        "continuation=keep as its own memory linked to target_id; distinct/new="
        "keep fresh. Reply with JSON only."
    )

    def _classify(cand: Memory, context: list[Memory]) -> Decision:
        ctx = "\n".join(f"- id={m.id}: {m.content[:200]}" for m in context) or "(none)"
        user = f"CANDIDATE: {cand.content[:400]}\nEXISTING:\n{ctx}"
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
            )
        except Exception:  # noqa: BLE001
            logger.warning("consolidation Haiku classify failed; promoting candidate")
            return Decision("new")

    return _classify


def _extract_json(text: str) -> str:
    """Pull the first {...} JSON object out of a model reply."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else "{}"

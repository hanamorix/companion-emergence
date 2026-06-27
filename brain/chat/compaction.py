"""Conversation compaction — the one core that fades old history into a
persisted, archived summary block at the head of the buffer.

Callers drive it (see changes/timed-conversation-compaction/1-spec.md):
  * the Kindled ``compact_history`` tool  — fold_existing_summary=False (append)
  * the daily supervisor cadence          — fold_existing_summary=True  (fade)
  * the apply_budget backstop              — fold_existing_summary=True  (fade)
  * the startup backlog migration          — fold + max_compact_turns (bounded batches;
    see brain/chat/compaction_migration.py)

Design invariants this module upholds:
  * **Lossless before lossy.** Raw turns (and, when folding, the old summary)
    are written to the append-only archive and verified BEFORE the live buffer
    is rewritten — an archive failure leaves the buffer untouched (no data loss).
  * **Never drop the un-extracted.** Only raw turns at/before the ingest cursor
    (``ts <= cursor``) are removable, so a turn is never compacted away before it
    becomes a memory. A ``None`` cursor (nothing extracted yet) is a hard no-op.
  * **Stable prefix.** The summary block is a persisted record rendered by
    ``_buffer_turns_to_messages`` as a head system message; between compactions
    the buffer only grows at the tail, so the replayed prefix is byte-stable
    (the cache side effect).
  * **Idempotent.** No removable raw turns ⇒ hard no-op; the existing summary is
    never re-faded with no new input, regardless of the fold flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from brain.bridge.provider import LLMProvider
from brain.ingest.buffer import (
    acquire_compaction_lock,
    append_archive,
    read_cursor,
    read_session,
    release_compaction_lock,
    rewrite_session_atomic,
)

logger = logging.getLogger(__name__)

# Two prompts share one ethos: preserve substance, fade only the trivial, and do
# NOT over-compress. Both inject a dynamic ``{target_words}`` ≈ half the source
# length (the user-set ratio) so haiku stops crushing a 5.5k-word block into 300
# words. With new ≈ 0.5·(prior + batch) the folded head converges to ~one batch's
# worth of text — bounded, not growing.

# Compaction always summarises with this model regardless of the persona's chat
# model — a small, cheap model is plenty for memory folding and keeps the cost off
# the (larger) chat model. Change this one string to swap (e.g. "sonnet"/"opus");
# the future model-agnostic refactor replaces the whole seam below.
COMPACTION_MODEL = "haiku"


def build_compaction_provider(persona_dir):
    """The provider compaction should use — the persona's provider *kind* but
    forced to COMPACTION_MODEL. For a ``fake`` persona (tests) this resolves to a
    FakeProvider, so no real CLI is shelled. Production call sites pass the result
    into ``compact_conversation``; the core keeps its injected ``provider`` param so
    unit tests can still pass a deterministic stub directly."""
    from pathlib import Path

    from brain.bridge.provider import get_provider
    from brain.persona_config import DEFAULT_PROVIDER, PersonaConfig

    name = DEFAULT_PROVIDER
    cfg = Path(persona_dir) / "persona_config.json"
    if cfg.exists():
        name = PersonaConfig.load(cfg).provider
    return get_provider(name, persona_dir=Path(persona_dir), model_override=COMPACTION_MODEL)


# Voice + perspective WITHOUT importing voice.md (which can be longer than the text
# being summarised, and pulled the model toward stylistic reconstruction over
# fidelity). Instead the transcript's ``assistant`` turns are relabelled with the
# Kindled's name (``_render_transcript``), and the prompt tells the model to write
# from that individual's perspective and in the style of their own messages — the
# voice is self-derived from the content being summarised, and ACCURACY leads.
# ``user`` is left untouched (a deliberate stable placeholder for future work).
# ``{target_words}`` ≈ half the source keeps it from over-compressing.

# First-ever summary (or any no-prior summarise): condense raw turns only.
_SUMMARY_PROMPT = """Write {name}'s own first-person memory of the conversation below.
"I" is {name} — mirror the style, tone, and phrasing of the messages labelled
"{name}:". Refer to the other speaker (labelled "user") as "the user", or by their
actual name if the transcript gives one.
ACCURACY FIRST: record only what the transcript actually says. Do not invent, infer
beyond the text, or reverse who did what to whom — if the transcript says X, the
memory says X. Preserve names, decisions, emotional beats, unresolved threads,
ongoing projects, and concrete specifics (names, numbers, what was decided and why).
Drop only pleasantries, repetition, and formatting noise.
Length: aim for about {target_words} words — roughly {target_pct}% of the source.
That is the target: do not over-compress below it, and do not pad to reach it.
Output plain first-person prose ONLY: begin directly with the recollection. No title,
no name/description/metadata fields, no frontmatter, no headers, no lists, no
preamble, no closing sign-off.

CONVERSATION:
{transcript}

MEMORY:"""

# Fold: integrate new messages INTO the running memory, preserving a fading trace
# of everything already there (the fix for "no trace of the previous summary").
_FOLD_PROMPT = """Update {name}'s own running, first-person memory of a long, ongoing
conversation. Below is the EXISTING MEMORY (everything remembered so far) followed by
NEW MESSAGES not yet folded in. Produce an UPDATED MEMORY in the first person — "I" is
{name}; mirror the style, tone, and phrasing of the messages labelled "{name}:". Refer
to the other speaker (labelled "user") as "the user", or by their actual name if it
appears.

ACCURACY FIRST: record only what the sources actually say. Do not invent, infer beyond
the text, or reverse who did what to whom — if a source says X, the memory says X.

How to update:
- Carry the existing memory forward. Keep its names of people and places, decisions,
  emotional beats, unresolved threads, and ongoing projects — do not discard older
  material just because it is older. The newest messages may be richer in detail;
  older material should persist as a briefer but still-present trace, fading
  gradually rather than vanishing in one step.
- Integrate, don't staple: weave the new messages into the existing memory so the
  result reads as one continuous recollection, not two halves.
- Preserve concrete specifics: names, numbers, what was decided and why.
- Drop only pleasantries, repetition, and formatting noise.
- Length: aim for about {target_words} words — roughly {target_pct}% of the combined
  source below (existing memory + new messages). That is the target: do not
  over-compress below it, and do not pad to reach it.
- Output plain first-person prose ONLY: begin directly with the recollection. No
  title, no name/description/metadata fields, no frontmatter, no headers, no lists,
  no preamble, no closing sign-off.

EXISTING MEMORY:
{prior_summary}

NEW MESSAGES:
{transcript}

UPDATED MEMORY:"""

# The summary's target length as a fraction of the source being summarised. This is
# the HONEST target the prompt states (number + percent both derived from it), so it
# is model-agnostic: a model that follows instructions faithfully lands near this
# fraction rather than doubling/halving. Measured: haiku tracks the stated number
# best at low fractions (≈on-target at a quarter). Tune this one knob to taste.
_TARGET_FRACTION = 0.25
# Floor so a tiny batch can't request a degenerate ~0-word summary.
_MIN_TARGET_WORDS = 40


def _word_count(text: str) -> int:
    return len((text or "").split())


def _render_transcript(turns: list[dict], kindled_name: str) -> str:
    """Render raw turns as ``<speaker>: <text>`` lines, relabelling the Kindled's
    own turns (``speaker=="assistant"``) with ``kindled_name`` so the summariser can
    write from that individual's perspective and mirror their style. ``user`` (and
    any other speaker label) is left verbatim — a deliberate stable placeholder."""
    lines = []
    for r in turns:
        sp = r.get("speaker", "?")
        if sp == "assistant":
            sp = kindled_name
        lines.append(f"{sp}: {r.get('text', '')}")
    return "\n".join(lines)


@dataclass
class CompactionResult:
    """Outcome of one compact_conversation call."""

    compacted: bool          # did the buffer actually change?
    compacted_n: int         # raw turns moved to archive
    new_gen: int             # gen of the summary now at the head (0 if none)
    fell_soft: bool          # provider failed → deterministic note used
    reaped_stale_lock: bool  # a crashed predecessor's lock was reaped
    reason: str = ""         # why a no-op happened (for the log)


def _parse_ts(raw: object) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def _split_buffer(turns: list[dict]) -> tuple[dict | None, list[dict]]:
    """Return (existing_summary_row_or_None, raw_turns_in_order).

    The first ``speaker=="summary"`` row is the existing summary; raw turns are
    the user/assistant rows in file order. Defensive against a stray second
    summary row (ignored here; the assembler hoists/drops, and the writer keeps
    at most one)."""
    existing: dict | None = None
    raw: list[dict] = []
    for t in turns:
        if t.get("speaker") == "summary":
            if existing is None:
                existing = t
            continue
        # Any non-summary row is a real conversation turn (user/assistant or an
        # arbitrary name-speaker) — never dropped, so a rewrite can't lose data.
        raw.append(t)
    return existing, raw


def _turn_identity(t: dict) -> tuple:
    """A stable identity for a buffer turn, used to match archived turns against
    a re-read buffer (race-safe rewrite). (ts, speaker, text) is unique enough;
    a collision means two byte-identical turns in the same second, where keeping
    or dropping one is harmless."""
    return (t.get("ts"), t.get("speaker"), t.get("text"))


def _summary_text(row: dict | None) -> str:
    return (row or {}).get("text", "") or ""


def _summary_gen(row: dict | None) -> int:
    if not row:
        return 0
    try:
        return int(row.get("compaction", {}).get("gen", 0))
    except (AttributeError, TypeError, ValueError):
        return 0


def compact_conversation(
    persona_dir,
    session_id: str,
    *,
    older_than: timedelta,
    fold_existing_summary: bool,
    provider: LLMProvider,
    min_keep_tail: int = 40,
    max_compact_turns: int | None = None,
    now: datetime | None = None,
    lock_stale_s: float = 600.0,
) -> CompactionResult:
    """Fade in-range, already-extracted raw turns into the head summary block.

    See module docstring for the invariants. Returns a CompactionResult; a
    no-op (locked / None cursor / nothing aged) returns ``compacted=False`` with
    a ``reason``.
    """
    now = now or datetime.now(UTC)

    reaped = False
    if not acquire_compaction_lock(persona_dir, session_id, stale_s=lock_stale_s):
        return CompactionResult(False, 0, 0, False, False, reason="locked")
    # We acquired (possibly after reaping); we can't cheaply tell here whether a
    # reap happened, so surface it via the lock module if needed later. Keep
    # False — reaping is logged inside acquire on the rare path.
    try:
        turns = read_session(persona_dir, session_id)
        existing_summary, raw_turns = _split_buffer(turns)

        # --- Cursor guard: only compact what is provably extracted -------------
        cursor = read_cursor(persona_dir, session_id)
        if cursor is None:
            return CompactionResult(
                False, 0, _summary_gen(existing_summary), False, reaped,
                reason="cursor_none",
            )
        cursor_dt = _parse_ts(cursor)
        cutoff = now - older_than
        if cursor_dt is not None and cursor_dt < cutoff:
            cutoff = cursor_dt

        # Protect the most-recent min_keep_tail raw turns regardless of age.
        protected = set(range(max(0, len(raw_turns) - min_keep_tail), len(raw_turns)))
        removable: list[dict] = []
        retained: list[dict] = []
        for i, t in enumerate(raw_turns):
            ts = _parse_ts(t.get("ts"))
            if i not in protected and ts is not None and ts <= cutoff:
                removable.append(t)
            else:
                retained.append(t)

        if not removable:
            return CompactionResult(
                False, 0, _summary_gen(existing_summary), False, reaped,
                reason="nothing_aged",
            )

        # --- Batch cap (backlog migration) -------------------------------------
        # Optional: fold only the OLDEST ``max_compact_turns`` this pass; the
        # overflow stays live. ``removable`` is built in raw_turns file order
        # (chronological), so ``removable[:N]`` is the oldest N — repeated calls
        # drain oldest→newest with no gap or reorder. The overflow is simply not
        # placed in ``removable``/``archived_ids``, so the race-safe rewrite below
        # (which keeps every current raw turn whose identity is NOT archived)
        # retains it automatically — no extra bookkeeping needed. ``None`` (the
        # daily-tick + apply_budget backstop callers) preserves the prior
        # "compact everything aged" behavior unchanged.
        if max_compact_turns is not None and len(removable) > max_compact_turns:
            removable = removable[:max_compact_turns]

        # --- Summarize ---------------------------------------------------------
        # Perspective + voice are self-derived: relabel the Kindled's own turns
        # (assistant) with the persona name (from the persona dir), leave "user" as
        # is, and instruct the model to write from the Kindled's perspective in their
        # own style. No voice.md import — accuracy leads.
        from pathlib import Path
        persona_name = Path(persona_dir).name
        transcript = _render_transcript(removable, persona_name)
        removable_words = sum(_word_count(r.get("text", "")) for r in removable)
        prior_text = _summary_text(existing_summary)
        folding = fold_existing_summary and existing_summary is not None

        # Target ≈ _TARGET_FRACTION of the source being summarised (prior memory +
        # new turns when folding; the new turns alone otherwise). Number and percent
        # both derive from the one constant so they can't drift. Floored so a tiny
        # batch can't ask for a near-empty summary.
        source_words = removable_words + (_word_count(prior_text) if folding else 0)
        target_words = max(_MIN_TARGET_WORDS, int(source_words * _TARGET_FRACTION))
        target_pct = round(_TARGET_FRACTION * 100)

        if folding:
            prompt = _FOLD_PROMPT.format(
                name=persona_name, prior_summary=prior_text, transcript=transcript,
                target_words=target_words, target_pct=target_pct,
            )
        else:
            prompt = _SUMMARY_PROMPT.format(
                name=persona_name, transcript=transcript,
                target_words=target_words, target_pct=target_pct,
            )

        fell_soft = False
        try:
            new_part = provider.generate(prompt=prompt).strip()
        except Exception:
            logger.exception(
                "compaction: provider summarisation failed session=%s; falling back",
                session_id,
            )
            new_part = f"[truncated {len(removable)} earlier messages]"
            fell_soft = True

        if folding:
            # Fade: the integrated memory supersedes the old. On a provider failure,
            # PRESERVE the prior memory (don't let a hiccup wipe accumulated context)
            # and merely note the un-summarised batch.
            if fell_soft and prior_text:
                new_text = f"{prior_text}\n\n{new_part}"
            else:
                new_text = new_part
        elif existing_summary is None:
            # First-ever summary: the new text is the whole memory.
            new_text = new_part
        else:
            # Tool append (fold=False, prior exists): keep prior verbatim, append.
            new_text = f"{prior_text}\n\n{new_part}" if prior_text else new_part

        new_gen = _summary_gen(existing_summary) + 1
        covers_until = removable[-1].get("ts") or cutoff.isoformat()
        summary_row = {
            "session_id": session_id,
            "speaker": "summary",
            "text": new_text,
            "ts": now.isoformat(timespec="seconds"),
            "compaction": {
                "covers_until_ts": covers_until,
                "folded": bool(fold_existing_summary),
                "gen": new_gen,
            },
        }

        # --- Archive BEFORE mutating the live buffer (lossless-before-lossy) ----
        # Archive the removed raw turns and, when folding, the old summary being
        # superseded (so the provenance chain keeps every faded version).
        archive_records = list(removable)
        if fold_existing_summary and existing_summary is not None:
            archive_records.append(existing_summary)
        try:
            written = append_archive(persona_dir, session_id, archive_records)
            if written <= 0 and archive_records:
                raise OSError("archive append wrote zero bytes")
        except Exception:
            logger.exception(
                "compaction: archive write failed session=%s; buffer left untouched",
                session_id,
            )
            return CompactionResult(
                False, 0, _summary_gen(existing_summary), fell_soft, reaped,
                reason="archive_failed",
            )

        # --- Install [summary, *retained] atomically ---------------------------
        # Re-read the live buffer just before the rewrite and rebuild the retained
        # set from CURRENT turns minus the archived ones (by identity). The
        # summarize step above is a slow provider call (the claude CLI, seconds);
        # a concurrent chat turn for this session may have appended new turns via
        # ingest_turn during that window. Rewriting from the stale snapshot would
        # os.replace those appends away (lost-update). Reconstructing from the
        # re-read preserves them — shrinking the loss window from the whole
        # summarize to the µs between this re-read and os.replace. (`ingest_turn`
        # takes no compaction lock by design — appends must stay fast.)
        archived_ids: list[tuple] = [_turn_identity(t) for t in removable]
        current = read_session(persona_dir, session_id)
        _, current_raw = _split_buffer(current)
        retained_now: list[dict] = []
        for t in current_raw:
            tid = _turn_identity(t)
            if tid in archived_ids:
                archived_ids.remove(tid)  # consume once (multiset-safe vs dup turns)
            else:
                retained_now.append(t)
        rewrite_session_atomic(persona_dir, session_id, [summary_row, *retained_now])
        retained = retained_now
        logger.info(
            "compaction: session=%s gen=%d folded=%s compacted_n=%d fell_soft=%s",
            session_id, new_gen, fold_existing_summary, len(removable), fell_soft,
        )
        return CompactionResult(
            True, len(removable), new_gen, fell_soft, reaped, reason="ok"
        )
    finally:
        release_compaction_lock(persona_dir, session_id)

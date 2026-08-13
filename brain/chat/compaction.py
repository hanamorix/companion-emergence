"""Conversation compaction — the one core that fades old history into a
persisted, archived summary block at the head of the buffer.

Callers drive it (see docs/integration/2026-06-29-pr52-compaction-conflict-analysis.md):
  * the Kindled ``compact_history`` tool  — fold_existing_summary=False (append)
  * the daily supervisor cadence          — fold_existing_summary=True  (fade)
  * the apply_budget backstop              — fold_existing_summary=True  (fade)
  * the startup backlog migration          — fold, replayed in 24h time-increments
    (oldest cohort first; see brain/chat/compaction_migration.py)

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
import re
from dataclasses import dataclass, field
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

# ---------------------------------------------------------------------------
# Cascade compaction — 3-tier age-stratified summary (Phase 1)
# ---------------------------------------------------------------------------
#
# Age-band boundaries measured from ``now``. Material graduates raw→tier1→tier2→
# tier3 by TRUE content age (oldest covered edge), then STAYS in tier 3 (terminal,
# re-compacted forever so the oldest fades). See plan §1.3.
_AGE_24H = timedelta(hours=24)
_AGE_48H = timedelta(hours=48)
_AGE_72H = timedelta(hours=72)

# Per-tier soft compaction targets (fraction of that tier's source material).
_FRACTION_24H = 0.60
_FRACTION_48H = 0.40
_FRACTION_72H = 0.20

# Hard char caps that bound the cache-stable head. tier1 caps the "yesterday"
# band; tier3 is TERMINAL (re-compacted every cycle), so it carries its own hard
# ceiling ON TOP of the 20% compaction, else the persisting section + each
# graduated tier2 would accrete without bound. tier2 is transient (graduates each
# cycle) so it is bounded-by-input with no separate cap. See plan §0 (open-Q8).
# FOLLOW-UP(open-Q8): measure avg char length of a real 24h conversation on live
# data and re-tune _SECTION_24H_CHAR_CAP.
_SECTION_24H_CHAR_CAP = 12_000
_SECTION_72H_CHAR_CAP = int(0.20 * _SECTION_24H_CHAR_CAP)  # 2_400 at the default

# A legacy single-layer summary is accumulated (often months-old) history. It is
# read/migrated as TIER 3 with an explicit OLD-FLOOR covers_from_ts so the
# oldest-edge classifier keeps it terminal — never reclassified "yesterday" (#82).
# Set UNCONDITIONALLY, never falling back to covers_until_ts. See plan §1.1/§4.
_LEGACY_AGE_FLOOR = timedelta(hours=96)  # >72h with margin

# Weekly session-rollover (1c-B) tuning — consulted by the supervisor daily tick.
_WEEKLY_ROLLOVER_AGE = timedelta(days=7)
_ROLLOVER_QUIET_GAP = timedelta(minutes=30)

# Owner-specified human age-band labels (OWNER 2026-08-13). STATIC tier labels
# (NOT computed dates), so the render stays byte-stable between re-compactions.
_SECTION_ORDER = ("24h", "48h", "72h")
_SECTION_LABELS = {
    "24h": "yesterday",
    "48h": "day before yesterday",
    "72h": "a few days ago",
}

# --- Fold-output validation (#77) ------------------------------------------
# A refusal / policy completion / assistant-meta frame stored verbatim AS the
# memory is the corruption vector #77 names. These predicates reject it. They are
# deliberately conservative: a false-reject merely keeps last cycle's section
# (safe); a false-accept stores garbage as recalled past (the failure).
_REFUSAL_LEAD = re.compile(
    r"^\s*(i\s+won'?t\b|i\s+will\s+not\b|i\s+cannot\b|i\s+can'?t\b|i'?m\s+sorry\b"
    r"|i\s+am\s+sorry\b|sorry[,.]|i\s+refuse\b|i'?m\s+unable\b|i\s+am\s+unable\b"
    r"|as\s+an\s+ai\b|i'?m\s+not\s+able\b)",
    re.IGNORECASE,
)
_META_FRAME = re.compile(
    r"^\s*(here\s+is\b|here'?s\b|below\s+is\b|the\s+following\s+is\b"
    r"|summary\s*:|updated\s+memory\s*:|as\s+an\s+ai\b)",
    re.IGNORECASE,
)
_FIRST_PERSON = re.compile(
    r"\b(i|i'?m|i'?ve|i'?ll|i'?d|me|my|mine|myself|we|our|ours|us)\b",
    re.IGNORECASE,
)
# Below this word count a bare non-first-person output (e.g. a test stub token) is
# too short to be a refusal essay; only longer non-first-person prose is rejected.
_NONTRIVIAL_WORDS = 8


def _validate_fold_output(text: object) -> str | None:
    """Return the stripped fold output, or ``None`` when it must be rejected (#77).

    Rejects: empty/whitespace; a refusal / "I won't proceed"-style completion; an
    assistant-meta frame ("Here is the summary:" / "As an AI"); or a non-trivial
    output with no first-person pronoun (the persona fold is first-person by
    construction). A module-level, unit-testable predicate (criterion C4).
    """
    s = (text or "")
    s = s.strip() if isinstance(s, str) else ""
    if not s:
        return None
    if _META_FRAME.match(s):
        return None
    if _REFUSAL_LEAD.match(s):
        return None
    if len(s.split()) >= _NONTRIVIAL_WORDS and not _FIRST_PERSON.search(s):
        return None
    return s


def _generate_validated_fold(provider: LLMProvider, prompt: str) -> str | None:
    """Call the provider, validate (#77), retry ONCE on reject.

    Returns the validated text, or ``None`` when both attempts are rejected. A
    provider EXCEPTION propagates to the caller (which applies its own soft
    fallback) — only a rejected *output* returns None here.
    """
    out = _validate_fold_output(provider.generate(prompt=prompt))
    if out is not None:
        return out
    # One retry with the same prompt — a transient bad completion often clears.
    return _validate_fold_output(provider.generate(prompt=prompt))


# --- Section representation, render, tolerant reader (#82, C1/C5/C6) --------

def _iso_seconds(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _part_of_day(hour: int) -> str:
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def _coarse_stamp(raw: object) -> str | None:
    """A coarse, deterministic wall-clock anchor (``"Aug 10 evening"``) from a ts.

    Date + part-of-day only — no per-render clock read, no minute/second — so the
    render stays byte-stable between re-compactions (C6). ``%b %d`` avoids the
    platform-specific ``%-d`` (Windows CI). Returns None on an unparseable ts.
    """
    dt = _parse_ts(raw)
    if dt is None:
        return None
    return f"{dt.strftime('%b %d')} {_part_of_day(dt.hour)}"


def _coarse_span(covers_from_ts: object, covers_until_ts: object) -> str:
    """Render the covered ts range at coarse granularity (temporal markers #82)."""
    a = _coarse_stamp(covers_from_ts)
    b = _coarse_stamp(covers_until_ts)
    if a and b:
        return a if a == b else f"{a} – {b}"
    return a or b or ""


def _render_sections(sections: dict) -> str:
    """Deterministic render of the 3 sections — fixed tier1→tier2→tier3 order,
    owner human labels, coarse ts span; NO nonce/``now()`` (byte-stable, C6)."""
    parts: list[str] = []
    for key in _SECTION_ORDER:
        sec = sections.get(key)
        if not sec:
            continue
        text = (sec.get("text") or "").strip()
        if not text:
            continue
        label = _SECTION_LABELS[key]
        span = _coarse_span(sec.get("covers_from_ts"), sec.get("covers_until_ts"))
        header = f"[{label} — {span}]" if span else f"[{label}]"
        parts.append(f"{header} {text}")
    return "\n\n".join(parts)


def _read_sections(row: dict | None, now: datetime) -> dict:
    """Tolerant reader: return the row's ``compaction.sections`` dict, or map a
    legacy single-layer (section-less) row → TIER 3 with an OLD-FLOOR
    ``covers_from_ts`` (``now − _LEGACY_AGE_FLOOR``, set UNCONDITIONALLY) so the
    oldest-edge classifier keeps it terminal, never "yesterday" (#82). See §1.1."""
    if not row:
        return {}
    comp = row.get("compaction") or {}
    sections = comp.get("sections")
    if isinstance(sections, dict) and sections:
        out: dict = {}
        for key in _SECTION_ORDER:
            sec = sections.get(key)
            if sec and (sec.get("text") or "").strip():
                out[key] = dict(sec)
        if out:
            return out
    # Legacy flat row → tier 3 with the old-floor covers_from_ts.
    text = (row.get("text") or "").strip()
    if not text:
        return {}
    covers_until = comp.get("covers_until_ts") or row.get("ts") or _iso_seconds(now)
    return {
        "72h": {
            "text": text,
            "covers_from_ts": _iso_seconds(now - _LEGACY_AGE_FLOOR),
            "covers_until_ts": covers_until,
        }
    }


def _truncate_sentence(text: str, cap: int) -> str:
    """Truncate ``text`` to at most ``cap`` chars at a sentence boundary (fall back
    to a word boundary, then a hard cut). Used to enforce the tier hard caps."""
    if cap is None or len(text) <= cap:
        return text
    window = text[:cap]
    # Prefer the last sentence terminator inside the window.
    best = max(window.rfind(". "), window.rfind("! "), window.rfind("? "),
               window.rfind(".\n"), window.rfind("!\n"), window.rfind("?\n"))
    if best >= cap // 2:
        return window[: best + 1].rstrip()
    space = window.rfind(" ")
    if space >= cap // 2:
        return window[:space].rstrip()
    return window.rstrip()


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

        # Fold-output validation (#77): validate the provider output, retry once,
        # and on a double-reject (or a provider exception) fall back to a
        # deterministic note rather than storing a refusal / meta-frame verbatim
        # AS the memory. When folding, the prior memory is preserved below (a
        # provider hiccup never wipes accumulated context).
        fell_soft = False
        try:
            generated = _generate_validated_fold(provider, prompt)
        except Exception:
            logger.exception(
                "compaction: provider summarisation failed session=%s; falling back",
                session_id,
            )
            generated = None
        if generated is None:
            new_part = f"[truncated {len(removable)} earlier messages]"
            fell_soft = True
        else:
            new_part = generated

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


# ---------------------------------------------------------------------------
# Cascade pass — true age-gated graduation with a terminal, multi-input tier 3
# ---------------------------------------------------------------------------


@dataclass
class TierRecord:
    """Per-tier instrumentation for one cascade pass (C2/C4/C14 observability)."""

    tier: str                       # "24h" / "48h" / "72h"
    section_inputs: list[str]       # keys of existing sections folded into this tier
    raw_group_size: int             # count of raw turns in this tier's age group
    target_fraction: float
    fell_soft: bool                 # provider double-rejected → lossless-leaning join
    validated: bool                 # provider output accepted (not the fallback)


@dataclass
class CascadeResult:
    """Outcome of one cascade_conversation / emergency_fold_24h call."""

    compacted: bool
    compacted_n: int
    new_gen: int
    reason: str = ""
    tiers: dict[str, TierRecord] = field(default_factory=dict)


def _bucket_of_age(age: timedelta) -> str:
    """Age-band a duration by the oldest-edge classifier: ``24h`` if age < 48h,
    ``48h`` if < 72h, else terminal ``72h`` (plan §1.3 step 3). The boundary is
    strict ``<`` so a cohort sown at day 0 graduates tier1→tier2→tier3 on the daily
    passes at exactly day 1 / day 2 / day 3 (age 24h → tier1, 48h → tier2, 72h →
    tier3) — matching the criterion-C14 graduation cadence."""
    if age < _AGE_48H:
        return "24h"
    if age < _AGE_72H:
        return "48h"
    return "72h"


def _tier_span(section_inputs: list[dict], raw_turns: list[dict], *, fallback: datetime) -> tuple[str, str]:
    """(covers_from_ts, covers_until_ts) for a tier — oldest covered edge → newest,
    across its section inputs and its raw age-group turns."""
    froms: list[datetime] = []
    untils: list[datetime] = []
    for s in section_inputs:
        cf = _parse_ts(s.get("covers_from_ts"))
        cu = _parse_ts(s.get("covers_until_ts"))
        if cf:
            froms.append(cf)
        if cu:
            untils.append(cu)
    for r in raw_turns:
        ts = _parse_ts(r.get("ts"))
        if ts:
            froms.append(ts)
            untils.append(ts)
    covers_from = min(froms) if froms else fallback
    covers_until = max(untils) if untils else fallback
    return _iso_seconds(covers_from), _iso_seconds(covers_until)


def _fold_into_section(
    persona_name: str,
    section_inputs: list[dict],
    raw_turns: list[dict],
    fraction: float,
    cap: int | None,
    provider: LLMProvider,
    *,
    now: datetime,
) -> tuple[dict | None, bool, bool]:
    """Fold (section inputs + raw age-group) into one new tier section.

    The fold input is an ORDERED, lossless-leaning join — prior section prose
    (oldest-first) then the rendered raw turns — compacted to ``fraction`` and
    hard-capped by sentence-boundary truncation at ``cap``. On a #77 double-reject
    (or provider error) the fallback is the SAME lossless-leaning join truncated to
    the cap — never dropping either input (plan §1.3/§1.4 cascade fallback).

    Returns ``(section_dict | None, fell_soft, validated)``. ``None`` when the tier
    has no inputs at all (empty band).
    """
    # Join section inputs OLDEST-first (by covered edge) so the fade preferentially
    # compresses the oldest material (plan §1.3 lossless-leaning join).
    ordered_inputs = sorted(
        section_inputs,
        key=lambda s: (_parse_ts(s.get("covers_from_ts")) or datetime.max.replace(tzinfo=UTC)),
    )
    prior_texts = [(s.get("text") or "").strip() for s in ordered_inputs if (s.get("text") or "").strip()]
    prior_joined = "\n\n".join(prior_texts)
    raw_transcript = _render_transcript(raw_turns, persona_name)
    if not prior_joined and not raw_turns:
        return None, False, True

    source_words = _word_count(prior_joined) + sum(_word_count(r.get("text", "")) for r in raw_turns)
    target_words = max(_MIN_TARGET_WORDS, int(source_words * fraction))
    target_pct = round(fraction * 100)
    if prior_joined:
        prompt = _FOLD_PROMPT.format(
            name=persona_name, prior_summary=prior_joined, transcript=raw_transcript,
            target_words=target_words, target_pct=target_pct,
        )
    else:
        prompt = _SUMMARY_PROMPT.format(
            name=persona_name, transcript=raw_transcript,
            target_words=target_words, target_pct=target_pct,
        )

    validated = True
    fell_soft = False
    try:
        text = _generate_validated_fold(provider, prompt)
    except Exception:
        logger.exception("cascade: provider fold failed; using lossless-leaning fallback")
        text = None
    if text is None:
        # Double-reject / provider error → lossless-leaning safe join of the INPUTS,
        # truncated at a sentence boundary to the tier bound. NEVER drop source.
        validated = False
        fell_soft = True
        join = prior_joined
        if raw_transcript:
            join = f"{join}\n\n{raw_transcript}" if join else raw_transcript
        text = _truncate_sentence(join, cap or _SECTION_24H_CHAR_CAP)
    elif cap is not None and len(text) > cap:
        text = _truncate_sentence(text, cap)

    covers_from, covers_until = _tier_span(section_inputs, raw_turns, fallback=now)
    return (
        {"text": text, "covers_from_ts": covers_from, "covers_until_ts": covers_until},
        fell_soft,
        validated,
    )


def _install_cascade_row(
    persona_dir,
    session_id: str,
    *,
    new_sections: dict,
    existing_summary: dict | None,
    removable: list[dict],
    now: datetime,
) -> int | None:
    """Archive (removed raw + old summary) BEFORE rewrite, then install the new
    sectioned summary row + retained tail in ONE atomic rewrite. Returns the new
    gen, or ``None`` on an archive failure (buffer left untouched)."""
    from pathlib import Path  # noqa: F401 — parity with module's lazy-import style

    new_gen = _summary_gen(existing_summary) + 1
    all_untils = [_parse_ts(s.get("covers_until_ts")) for s in new_sections.values()]
    covers_until = _iso_seconds(max([u for u in all_untils if u], default=now))
    summary_row = {
        "session_id": session_id,
        "speaker": "summary",
        "ts": _iso_seconds(now),
        "text": _render_sections(new_sections),
        "compaction": {
            "gen": new_gen,
            "folded": True,
            "covers_until_ts": covers_until,
            "sections": new_sections,
        },
    }

    # Lossless-before-lossy: archive the removed raw turns and the superseded
    # summary (provenance) and verify the byte write before mutating the buffer.
    archive_records = list(removable)
    if existing_summary is not None:
        archive_records.append(existing_summary)
    if archive_records:
        try:
            written = append_archive(persona_dir, session_id, archive_records)
            if written <= 0:
                raise OSError("archive append wrote zero bytes")
        except Exception:
            logger.exception(
                "cascade: archive write failed session=%s; buffer left untouched", session_id
            )
            return None

    # Atomic single write: re-read the live buffer and subtract the archived raw by
    # identity so a concurrent append during the slow fold is not clobbered.
    archived_ids: list[tuple] = [_turn_identity(t) for t in removable]
    current = read_session(persona_dir, session_id)
    _, current_raw = _split_buffer(current)
    retained_now: list[dict] = []
    for t in current_raw:
        tid = _turn_identity(t)
        if tid in archived_ids:
            archived_ids.remove(tid)
        else:
            retained_now.append(t)
    rewrite_session_atomic(persona_dir, session_id, [summary_row, *retained_now])
    return new_gen


def cascade_conversation(
    persona_dir,
    session_id: str,
    *,
    provider: LLMProvider,
    now: datetime | None = None,
    min_keep_tail: int = 40,
    lock_stale_s: float = 600.0,
) -> CascadeResult:
    """One age-gated GRADUATION pass over the sectioned summary row (plan §1.3).

    Material graduates raw→tier1→tier2→tier3 by TRUE content age (oldest covered
    edge) and then stays in tier 3 (terminal, re-compacted forever). Never
    co-folds the prior tier1 section with fresh raw. All three tiers are computed
    from the pre-pass snapshot and installed in ONE atomic rewrite (C17).
    """
    from pathlib import Path

    now = now or datetime.now(UTC)
    if not acquire_compaction_lock(persona_dir, session_id, stale_s=lock_stale_s):
        return CascadeResult(False, 0, 0, reason="locked")
    try:
        turns = read_session(persona_dir, session_id)
        existing_summary, raw_turns = _split_buffer(turns)

        cursor = read_cursor(persona_dir, session_id)
        if cursor is None:
            return CascadeResult(False, 0, _summary_gen(existing_summary), reason="cursor_none")
        cursor_dt = _parse_ts(cursor)
        age_cutoff = now - _AGE_24H

        # Eligible raw = extracted (ts ≤ cursor), beyond the protected tail, aged ≥24h.
        protected = set(range(max(0, len(raw_turns) - min_keep_tail), len(raw_turns)))
        removable: list[dict] = []
        for i, t in enumerate(raw_turns):
            ts = _parse_ts(t.get("ts"))
            if i in protected or ts is None:
                continue
            if cursor_dt is not None and ts > cursor_dt:
                continue  # un-extracted → never compact (lossless-before-lossy)
            if ts <= age_cutoff:
                removable.append(t)

        # Age-partition eligible raw by true age.
        raw_groups: dict[str, list[dict]] = {"24h": [], "48h": [], "72h": []}
        for t in removable:
            ts = _parse_ts(t.get("ts"))
            raw_groups[_bucket_of_age(now - ts)].append(t)

        # Age-classify each existing section by its OLDEST covered edge.
        existing_sections = _read_sections(existing_summary, now)
        classified: dict[str, list[dict]] = {"24h": [], "48h": [], "72h": []}
        classified_keys: dict[str, list[str]] = {"24h": [], "48h": [], "72h": []}
        needs_graduate = False
        for key, sec in existing_sections.items():
            cf = _parse_ts(sec.get("covers_from_ts"))
            bucket = _bucket_of_age(now - cf) if cf else "72h"
            classified[bucket].append(sec)
            classified_keys[bucket].append(key)
            if bucket != key:
                needs_graduate = True

        # Idempotence: nothing newly crossed AND no section needs to graduate → no-op.
        if not removable and not needs_graduate:
            return CascadeResult(
                False, 0, _summary_gen(existing_summary), reason="nothing_aged"
            )

        # Assemble oldest-first (72 → 48 → 24). Tier 3 is a terminal, multi-input
        # fold: the persisting tier-3 section + the newly-graduated tier-2 section
        # (both classify into the 72h band) + any raw crossing 72h.
        persona_name = Path(persona_dir).name
        tiers_spec = [
            ("72h", _FRACTION_72H, _SECTION_72H_CHAR_CAP),
            ("48h", _FRACTION_48H, None),
            ("24h", _FRACTION_24H, _SECTION_24H_CHAR_CAP),
        ]
        new_sections: dict[str, dict] = {}
        records: dict[str, TierRecord] = {}
        for key, fraction, cap in tiers_spec:
            sec_inputs = classified[key]
            raw_group = raw_groups[key]
            section, fell_soft, validated = _fold_into_section(
                persona_name, sec_inputs, raw_group, fraction, cap, provider, now=now
            )
            records[key] = TierRecord(
                tier=key,
                section_inputs=list(classified_keys[key]),
                raw_group_size=len(raw_group),
                target_fraction=fraction,
                fell_soft=fell_soft,
                validated=validated,
            )
            if section is not None:
                new_sections[key] = section

        new_gen = _install_cascade_row(
            persona_dir, session_id,
            new_sections=new_sections, existing_summary=existing_summary,
            removable=removable, now=now,
        )
        if new_gen is None:
            return CascadeResult(
                False, 0, _summary_gen(existing_summary), reason="archive_failed"
            )
        logger.info(
            "cascade: session=%s gen=%d compacted_n=%d tiers=%s",
            session_id, new_gen, len(removable), sorted(new_sections),
        )
        return CascadeResult(True, len(removable), new_gen, reason="ok", tiers=records)
    finally:
        release_compaction_lock(persona_dir, session_id)


def emergency_fold_24h(
    persona_dir,
    session_id: str,
    *,
    provider: LLMProvider,
    min_keep_tail: int = 40,
    now: datetime | None = None,
    lock_stale_s: float = 600.0,
) -> CascadeResult:
    """The apply_budget backstop: a 24h-ONLY emergency fold on the sectioned row.

    Folds any extracted raw beyond the protected tail (``older_than=0``) into the
    24h tier (0.60, capped), leaving tiers 2/3 untouched — it bounds the live head
    in-turn without running the full age-gated re-bucket (that stays on the daily
    tick). Writes the sectioned row, holds the lock, archives-before-rewrite (C22).
    """
    from pathlib import Path

    now = now or datetime.now(UTC)
    if not acquire_compaction_lock(persona_dir, session_id, stale_s=lock_stale_s):
        return CascadeResult(False, 0, 0, reason="locked")
    try:
        turns = read_session(persona_dir, session_id)
        existing_summary, raw_turns = _split_buffer(turns)

        cursor = read_cursor(persona_dir, session_id)
        if cursor is None:
            return CascadeResult(False, 0, _summary_gen(existing_summary), reason="cursor_none")
        cursor_dt = _parse_ts(cursor)

        protected = set(range(max(0, len(raw_turns) - min_keep_tail), len(raw_turns)))
        removable: list[dict] = []
        for i, t in enumerate(raw_turns):
            ts = _parse_ts(t.get("ts"))
            if i in protected or ts is None:
                continue
            if cursor_dt is not None and ts > cursor_dt:
                continue
            removable.append(t)  # older_than=0: any extracted turn beyond the tail
        if not removable:
            return CascadeResult(
                False, 0, _summary_gen(existing_summary), reason="nothing_aged"
            )

        existing_sections = _read_sections(existing_summary, now)
        persona_name = Path(persona_dir).name
        sec24_inputs = [existing_sections["24h"]] if "24h" in existing_sections else []
        new24, _fell, _val = _fold_into_section(
            persona_name, sec24_inputs, removable, _FRACTION_24H, _SECTION_24H_CHAR_CAP,
            provider, now=now,
        )
        new_sections: dict[str, dict] = dict(existing_sections)
        if new24 is not None:
            new_sections["24h"] = new24

        new_gen = _install_cascade_row(
            persona_dir, session_id,
            new_sections=new_sections, existing_summary=existing_summary,
            removable=removable, now=now,
        )
        if new_gen is None:
            return CascadeResult(
                False, 0, _summary_gen(existing_summary), reason="archive_failed"
            )
        return CascadeResult(True, len(removable), new_gen, reason="ok")
    finally:
        release_compaction_lock(persona_dir, session_id)

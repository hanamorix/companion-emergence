"""The emotion engine's pure functions (Engine 1, #576).

Everything here works on in-memory state and takes the time as an argument: no files, no lock, no clock, no memory
store. The engine module (Engine 0 and later) orchestrates; this file is what its numbers come from. The JSON form
of `EngineState` and the queue are in `engine_files.py`; the drain and the direct apply are in `engine.py`, the replay in
`engine_rebuild.py`. Callers pass names already resolved (`resolve_emotions`): the fold functions do not canonicalise.

What lives here:

* The state a persona's emotions are kept in: two moving averages per emotion (short and long), each with a weight,
  the day the emotion last had a real entry, and a 30-day snapshot that the shift rule reads.
* `fold_entry` and `fold_event`: Roy's running-average formula, per average, and the four kinds of event that feed it
  (a memory committed, a memory given its feelings later, a memory shown, a signed change), plus the start-only event
  that only keeps the persona's start date.
* `close_days`: at the end of an active day, one entry of 0 (weight 1) for every present emotion that had no entry that
  day, the day's snapshot, and the skip over whole days the app was off.
* `apply_climax_reset`, `fade`, `counted_kind`, `source_modifier`, `resolve_emotions`.
* `run_tracker` and `shift_for`: which feelings have shifted, and which way, worked out from the snapshot.
* `feels_high_from`: the one rule for "this feeling is high" (never below an absolute floor; above it, the shift reads
  stronger, or the short average is in the upper half of its own 30-day range) that `engine.feels_high` answers by.

All constants are read as `dev_constants.NAME` at call time. Times are aware datetimes (UTC); a "day" is the machine's
local calendar day (`day_of`), written `YYYY-MM-DD`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING, NamedTuple, get_args

from brain import dev_constants
from brain.emotion.vocabulary import canonical_name
from brain.initiate.schemas import StateName
from brain.memory import memory_utils
from brain.utils.log_throttle import warn_throttled
from brain.utils.memory import is_conversation_derived
from brain.utils.time import to_local

if TYPE_CHECKING:
    from brain.memory.store import Memory

logger = logging.getLogger(__name__)

# The quiet-day entry: an emotion that had no entry on a day the app ran gets one entry of 0 at weight 1, with no source
# modifier. Written once, here (A3 E8.5): the owner's own numbers, not tunables.
_QUIET_DAY_VALUE = 0.0
_QUIET_DAY_WEIGHT = 1.0

# The source modifier of a peer or research memory runs from this floor up by this span (Step 0 "The source modifier").
_MODIFIER_FLOOR = 0.1
_MODIFIER_SPAN = 0.9
_RATING_SCALE = 10.0  # a research memory's engagement rating is 0 to this

# `initiate_outbound` rows count once they are past "pending": a note, or any state from "delivered" onward. The states
# come from the Literal itself so the set cannot drift from it.
_COUNTED_INITIATE_STATES = (frozenset(get_args(StateName)) - {"pending"}) | {"note"}

_EVENT_KINDS = frozenset({"commit", "assigned", "showing", "change", "start"})
# The kinds that carry a source modifier. A change has none (its weight is 1) and a start folds nothing.
_MODIFIER_KINDS = frozenset({"commit", "assigned", "showing"})
# The kinds dated at the row's creation time, whose repeat record starts there and is not kept once its window has ended.
_ANCHORED_KINDS = frozenset({"commit", "assigned"})


# --- state ---------------------------------------------------------------------------------------------------------


@dataclass
class Average:
    """One moving average and its weight."""

    avg: float = 0.0
    weight: float = 0.0


class RunRecord(NamedTuple):
    """The long average's run as the snapshot keeps it: where it started, which way, the last day it moved that way."""

    start_day: str
    direction: int  # +1 rising, -1 falling
    last_with_day: str


class SnapshotDay(NamedTuple):
    """One closed day of one emotion: the two averages as they stood, whether she felt it that day, the long run record."""

    day: str
    short: float
    long: float
    felt: bool
    run: RunRecord | None


@dataclass
class RepeatEntry:
    """A memory's first full showing and how many showings have followed it (0 right after the first)."""

    first_at: datetime
    count: int = 0


@dataclass
class EmotionState:
    short: Average = field(default_factory=Average)
    long: Average = field(default_factory=Average)
    last_entry_day: str | None = None
    snapshot: list[SnapshotDay] = field(default_factory=list)


@dataclass
class EngineState:
    """Everything the engine keeps about one persona's emotions, in memory.

    `last_update` is the time up to which the weights have been aged; `active_day` is the open day; `persona_start_at` is the
    creation time of the persona's oldest memory (a future time is held to now when it is folded), which the shift's start gate
    counts from (null: no memory yet, the gate is closed). `queue_line` is the
    number of lines of the queue file the engine has already folded: the next drain reads the lines after it. It is saved
    with the state so that the fold and the position reach the disk in one write. `switch_over_at` is the persona's one
    switch-over time and the only record that it was switched over: the first rebuild at start-up sets it, in its one verified
    state write; nothing else sets it, and every later save of a loaded state writes it back unchanged (None: not switched).
    """

    saved_at: datetime | None = None
    reset_at: datetime | None = None
    last_update: datetime | None = None
    active_day: str | None = None
    persona_start_at: datetime | None = None
    repeats: dict[str, RepeatEntry] = field(default_factory=dict)
    emotions: dict[str, EmotionState] = field(default_factory=dict)
    queue_line: int = 0
    switch_over_at: datetime | None = None


@dataclass(frozen=True)
class Event:
    """One thing that happened to the emotions.

    kind "commit": a counted memory was committed (`at` is its creation time, `emotions` its faded values, `climax` its
    stored climax). "assigned": an existing counted memory got its feelings later (`at` is the row's creation time, as for a
    commit, not the time of the assignment). "showing": a counted memory was shown in full. "change": signed amounts to add to
    the stored averages (`source` names the caller). "start": the start-only event for a committed row of a type that
    does not count (a memory's own date event, applied directly like a commit, never queued); it carries only `at`, the row's
    creation time, and keeps the persona's start date.

    `modifier` is the source modifier of a commit, an assigned or a showing event. A change has no modifier (its weight is
    always 1) and a start event folds nothing, so neither reads the field.
    """

    kind: str
    at: datetime
    memory_id: str | None = None
    emotions: Mapping[str, float] = field(default_factory=dict)
    modifier: float = 1.0
    climax: float = 0.0
    source: str | None = None


def copy_state(state: EngineState) -> EngineState:
    """A copy of `state` that shares nothing mutable with it: the checkpoint an apply step restores after a fold that raised.

    The scalar fields are copied by `dataclasses.replace` (so a field added later is carried without a change here); the
    averages, the repeat entries and each emotion's snapshot list are copied one level down. A `SnapshotDay` is an immutable
    tuple, so the list is a shallow copy. Cheaper than `copy.deepcopy`, which an import of thousands of events would pay for
    on every event.
    """
    return replace(
        state,
        repeats={memory_id: RepeatEntry(entry.first_at, entry.count) for memory_id, entry in state.repeats.items()},
        emotions={
            name: EmotionState(
                Average(emotion.short.avg, emotion.short.weight),
                Average(emotion.long.avg, emotion.long.weight),
                emotion.last_entry_day,
                list(emotion.snapshot),
            )
            for name, emotion in state.emotions.items()
        },
    )


def restore_state(state: EngineState, checkpoint: EngineState) -> None:
    """Put the fields of `checkpoint` (a `copy_state` result) back into `state`, in place, so every holder of `state` sees it."""
    for item in fields(EngineState):
        setattr(state, item.name, getattr(checkpoint, item.name))


# What a fold step may raise for an event's own content: an ordinary skip, logged as a line. Anything else is a bug and is
# logged with its traceback (throttled), but the event is still skipped and the others still fold.
_SKIPPABLE = (ValueError, OverflowError, OSError, TypeError)


def fold_guarded(state: EngineState, step: Callable[[], None], what: str, *, key: str) -> bool:
    """Run one fold `step` on `state`; if it raises, put the state back exactly as it was, log, and return False.

    `_fold` changes the averages before its last `day_of`, so a step that raises can leave a half-folded event; the apply step
    would then save it. The state is therefore checkpointed (`copy_state`) before the step and restored on ANY exception. An
    error of the event's own content (`_SKIPPABLE`) is an ordinary warning line; any other is logged with its traceback, once
    per throttle interval for the same `key` (the caller puts the persona and the step in it), so a persistent bug does not
    flood the log. `BaseException`s that are not `Exception` (an interrupt) are not caught.
    """
    checkpoint = copy_state(state)
    try:
        step()
    except _SKIPPABLE as exc:
        restore_state(state, checkpoint)
        logger.warning("emotion engine: %s skipped (%s: %s)", what, type(exc).__name__, exc)
        return False
    except Exception as exc:
        restore_state(state, checkpoint)
        warn_throttled(logger, f"{key}:{type(exc).__name__}", "emotion engine: %s skipped, an unexpected %s", what, type(exc).__name__)
        return False
    return True


# --- days ----------------------------------------------------------------------------------------------------------


def day_of(t: datetime) -> str:
    """The local calendar day of a time, `YYYY-MM-DD`. The one conversion all day arithmetic goes through."""
    return to_local(t).date().isoformat()


def _day_start(day: str) -> datetime:
    """The first instant of a local calendar day, as an aware UTC time."""
    return datetime.combine(date.fromisoformat(day), time.min).astimezone(UTC)


def _add_days(day: str, count: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=count)).isoformat()


def _days_between(earlier: str, later: str) -> int:
    return (date.fromisoformat(later) - date.fromisoformat(earlier)).days


# --- the running average -------------------------------------------------------------------------------------------


def _hold(value: float) -> float:
    """Hold a value to the emotion scale, 0 to the cap."""
    return min(max(value, 0.0), dev_constants.EMOTION_VALUE_CAP)


def _decay(hours: float, half_life_hours: float) -> float:
    return 0.5 ** (hours / half_life_hours)


def _fold_average(average: Average, value: float, w: float) -> None:
    """`weight = aged weight + w`; `average = average + (value - average) x w / weight`.

    The delta form of the weighted mean: an average fed the value it already holds does not move at all, so a steady feeling
    makes an exactly zero daily move and the shift rule needs no tolerance. A weight of 0 (an entry too old to count) folds
    nothing. The result is held to 0-10: the arithmetic can land one rounding step past the cap.
    """
    if w <= 0.0:
        return
    weight = average.weight + w
    average.avg = _hold(average.avg + (value - average.avg) * w / weight)
    average.weight = weight


def _fold(
    state: EngineState,
    emotion: str,
    short_value: float,
    long_value: float,
    w: float,
    at: datetime,
    *,
    zero: bool,
) -> None:
    """Fold one entry into one emotion's two averages. `short_value` and `long_value` differ only for a change."""
    if not (math.isfinite(short_value) and math.isfinite(long_value) and math.isfinite(w)):
        raise ValueError(f"cannot fold a non-finite entry for {emotion!r}: {short_value!r}, {long_value!r}, weight {w!r}")
    existing = state.emotions.get(emotion)
    if zero and existing is None:
        return  # a quiet-day zero goes only to an emotion that already has an average
    short_hours = dev_constants.EMOTION_SHORT_HALF_LIFE_H
    long_hours = dev_constants.EMOTION_LONG_HALF_LIFE_D * 24.0
    if state.last_update is None or at >= state.last_update:
        # In order: the entry ages every weight up to its own time and becomes the reference time.
        if state.last_update is not None:
            elapsed = (at - state.last_update).total_seconds() / 3600.0
            for other in state.emotions.values():
                other.short.weight *= _decay(elapsed, short_hours)
                other.long.weight *= _decay(elapsed, long_hours)
        state.last_update = at
        w_short = w_long = w
    else:
        # Before the reference time: the entry is pre-aged by its own age and ages nothing else. Both orders store the
        # same sum of w x 0.5^(age / H), so the averages do not depend on the order entries arrive in.
        elapsed = (state.last_update - at).total_seconds() / 3600.0
        w_short = w * _decay(elapsed, short_hours)
        w_long = w * _decay(elapsed, long_hours)
    emotion_state = existing if existing is not None else EmotionState()
    _fold_average(emotion_state.short, short_value, w_short)
    _fold_average(emotion_state.long, long_value, w_long)
    if existing is None:
        if emotion_state.long.weight <= 0.0:
            return  # so old that nothing of it survives: no ghost emotion
        state.emotions[emotion] = emotion_state
    if zero:
        return  # a quiet-day zero does not set the last-entry day
    day = day_of(at)
    if emotion_state.last_entry_day is None or day > emotion_state.last_entry_day:
        emotion_state.last_entry_day = day


def fold_entry(
    state: EngineState,
    emotion: str,
    value: float,
    w: float,
    at: datetime,
    *,
    zero: bool = False,
) -> None:
    """Fold one entry (`value` held to 0-10, weight `w`, dated `at`) into both averages of `emotion`.

    An entry at or after `last_update` ages all weights to its time and becomes the new `last_update`; an earlier one is
    pre-aged and ages nothing else. An entry that is not a quiet-day zero (even one whose value is 0) sets the emotion's last-entry
    day (the later of the old and the entry's day). `zero=True` is a quiet-day zero: it goes only to an emotion that
    already has an average and does not set the last-entry day. The caller holds a future `at` to now (`fold_event` does).
    """
    if not math.isfinite(value):
        raise ValueError(f"cannot fold a non-finite value for {emotion!r}: {value!r}")
    held = _hold(value)
    _fold(state, emotion, held, held, w, at, zero=zero)


def _fold_persona_start(state: EngineState, created_at: datetime) -> None:
    if state.persona_start_at is None or created_at < state.persona_start_at:
        state.persona_start_at = created_at


def _repeat_factor(state: EngineState, event: Event, at: datetime, now: datetime) -> float:
    """1 for a memory's first showing, then the repeat factor, its square and so on within the repeat window of that first
    showing (`dev_constants.EMOTION_REPEAT_FACTOR` ** (showings since the first + 1)).

    A commit of a memory less than a window old counts as the first showing and resets any entry. A showing or assignment outside the window
    (an entry not yet pruned, or one dated before the recorded first showing) is a new first showing; whether an entry is live is
    judged at the event's own `at`. A commit and an assignment are dated at the row's creation time, so their record starts
    there, not when they are folded: when that window has already ended by `now` the record is not kept (and a stale one from
    before the event is removed; a record dated after it is left alone), so the memory's next full showing counts at full
    weight. A showing always writes its record.
    """
    if event.memory_id is None:
        return 1.0
    entry = state.repeats.get(event.memory_id)
    window = timedelta(hours=dev_constants.EMOTION_REPEAT_WINDOW_H)
    if event.kind != "commit" and entry is not None and entry.first_at <= at < entry.first_at + window:
        factor = dev_constants.EMOTION_REPEAT_FACTOR ** (entry.count + 1)
        entry.count += 1
        return factor
    if event.kind in _ANCHORED_KINDS and at + window <= now:
        if entry is not None and entry.first_at <= at:
            del state.repeats[event.memory_id]  # a stale record from before this event; a newer one is left alone
        return 1.0
    state.repeats[event.memory_id] = RepeatEntry(first_at=at, count=0)
    return 1.0


def fold_event(state: EngineState, event: Event, now: datetime) -> None:
    """Apply one event to the state.

    A future event time and a future `last_update` are held to `now` first (the persona's start date is dated at the held time too). A commit, an assignment and a showing fold each
    of the event's emotions at weight `modifier x repeat factor`; a commit then keeps the persona's start date and, when its
    stored climax is 7 or more, applies the climax reset after its own emotions are in. An event with no emotions folds
    nothing and writes or advances no repeat record; a commit or an assignment of a memory that is already a window old keeps
    none either (`_repeat_factor`). A change folds, for each average, the stored average plus the signed amount (no history:
    base 0), held to 0-10, at weight 1 (it has no modifier and no repeat factor). A start event keeps the persona's start date
    and folds nothing else.

    An entry belongs to the day of its own time. One dated in a day that has already closed cannot undo the zero folded for that
    day, so the day then holds both the zero and the entry (accepted edge, A3 E5.13: a late-gated memory dated yesterday, or an
    event whose hook was still waiting for the lock when the drain closed its day).

    The skip of a commit or assignment whose memory was already folded in the same drain is the drain's, not this function's.

    The emotions of an event are expected to be resolved already (`resolve_emotions`). A non-finite value, or a non-finite
    modifier on a kind that carries one, raises before anything is changed, so a bad event leaves the state as it was.
    """
    if event.kind not in _EVENT_KINDS:
        raise ValueError(f"unknown event kind {event.kind!r}")
    bad_modifier = event.kind in _MODIFIER_KINDS and not math.isfinite(event.modifier)
    if bad_modifier or (event.kind != "start" and not all(math.isfinite(v) for v in event.emotions.values())):
        raise ValueError(f"cannot fold an event with a non-finite value or modifier: {event.kind} {event.memory_id}")
    if state.last_update is not None and state.last_update > now:
        state.last_update = now
    at = min(event.at, now)
    if event.kind == "start":
        _fold_persona_start(state, at)
        return
    if event.kind == "commit":
        _fold_persona_start(state, at)
    if not event.emotions:
        return
    if event.kind == "change":
        for name, amount in event.emotions.items():
            existing = state.emotions.get(name)
            short_base = existing.short.avg if existing is not None else 0.0
            long_base = existing.long.avg if existing is not None else 0.0
            _fold(state, name, _hold(short_base + amount), _hold(long_base + amount), 1.0, at, zero=False)
        return
    w = event.modifier * _repeat_factor(state, event, at, now)
    for name, value in event.emotions.items():
        fold_entry(state, name, value, w, at)
    if event.kind == "commit" and event.climax >= dev_constants.EMOTION_CLIMAX_THRESHOLD:
        apply_climax_reset(state)


def apply_climax_reset(state: EngineState) -> None:
    """The post-climax reset, on the stored short averages only.

    Not an entry: it adds no weight and leaves the long averages and every last-entry day alone. Arousal x 0.2 (never below
    the floor), desire x 0.6, comfort_seeking and rest_need + 2 (held to 10), climax set to 0; each only when present.
    """
    emotions = state.emotions
    arousal = emotions.get("arousal")
    if arousal is not None:
        arousal.short.avg = max(
            dev_constants.EMOTION_CLIMAX_AROUSAL_FLOOR, arousal.short.avg * dev_constants.EMOTION_CLIMAX_AROUSAL_FACTOR
        )
    desire = emotions.get("desire")
    if desire is not None:
        desire.short.avg *= dev_constants.EMOTION_CLIMAX_DESIRE_FACTOR
    # Accepted edge (A3 E4.6): an absent comfort_seeking or rest_need stays absent, so the +2 is skipped and the bump is
    # lost for a feeling with no history. Absent is not 0 (null is not 0).
    for name in ("comfort_seeking", "rest_need"):
        present = emotions.get(name)
        if present is not None:
            present.short.avg = min(dev_constants.EMOTION_VALUE_CAP, present.short.avg + dev_constants.EMOTION_CLIMAX_RELIEF_BUMP)
    climax = emotions.get("climax")
    if climax is not None:
        climax.short.avg = 0.0


# --- the fade of a stored memory value ------------------------------------------------------------------------------


def fade(
    value: float,
    age_days: float,
    half_life_days: float,
    importance: float,
    floor_fraction: float | None = None,
) -> float:
    """`value x 0.5^(age / (half_life x importance_stretch(importance)))`, never below `floor_fraction x value` when one is given.

    The stretch is the forgetting pass's own, from `brain.memory.memory_utils.importance_stretch` (the one function that holds
    the formula and the gain); an importance outside 0 to 10 is held to it there. The caller measures the age from the memory's
    creation time, or from the switch-over time when that is later, and passes the floor of the emotion
    (`EMOTION_FADE_FLOOR_FRACTION`, love and belonging only). A negative age (a row dated in the future) is held at 0 so a
    stored value never grows.
    """
    stretched = half_life_days * memory_utils.importance_stretch(importance)
    faded = value * 0.5 ** (max(age_days, 0.0) / stretched)
    if floor_fraction is not None:
        faded = max(faded, floor_fraction * value)
    return faded


# --- which memories count, and what they weigh -----------------------------------------------------------------------


def counted_kind(memory: Memory) -> str | None:
    """Which kind of memory this is for the engine: "user", "own", "peer", "research", or None when it does not count."""
    if is_conversation_derived(memory):
        return "user"
    if set(memory.metadata) == {"tags", "importance"}:
        return "user"  # what the add_memory tool writes
    if memory.memory_type == "making":
        return "own"
    if memory.memory_type == "initiate_outbound":
        state_name = memory.metadata.get("initiate_state")
        if isinstance(state_name, str) and state_name in _COUNTED_INITIATE_STATES:
            return "own"
        return None
    if memory.memory_type == "kindled_peer":
        return "peer"
    if memory.memory_type == "research":
        return "research"
    return None


def source_modifier(memory: Memory, trust: float) -> float:
    """How much a counted memory weighs: 1.0 for the user and her own; peer 0.1 + 0.9 x trust; research 0.1 + 0.9 x rating / 10.

    `trust` is the peer's trust score, passed in (0 to 1; anything else is held to that, a NaN reads as 0). A research memory's
    rating is `metadata["engagement_rating"]`; a missing or unparseable rating reads as 0, and the rating is held to 0-10.
    """
    kind = counted_kind(memory)
    if kind is None:
        raise ValueError(f"memory {memory.id} is of a type that does not count, so it has no source modifier")
    if kind in ("user", "own"):
        return 1.0
    if kind == "peer":
        held = min(max(trust, 0.0), 1.0) if math.isfinite(trust) else 0.0
        return _MODIFIER_FLOOR + _MODIFIER_SPAN * held
    try:
        rating = float(memory.metadata.get("engagement_rating"))
    except (TypeError, ValueError, OverflowError):
        rating = 0.0
    if not math.isfinite(rating):
        rating = 0.0
    rating = min(max(rating, 0.0), _RATING_SCALE)
    return _MODIFIER_FLOOR + _MODIFIER_SPAN * rating / _RATING_SCALE


def resolve_emotions(raw: Mapping[str, object], *, signed: bool = False) -> dict[str, float]:
    """The one step that turns an emotion map into the names and values the engine folds.

    Names become canonical (`vocabulary.canonical_name`); two spellings of one name become one entry whose value is the
    average of the two (this is not the reader's "keep the larger"); then each value is held to 0-10. A value that is not a
    finite number is dropped with a log line. `signed=True` is for a change's amounts: the sign is kept and nothing is held.
    """
    groups: dict[str, list[float]] = {}
    for name, value in raw.items():
        if not isinstance(name, str):
            logger.warning("dropping an emotion whose name %r is not text", name)
            continue
        try:
            number = None if isinstance(value, bool) or not isinstance(value, (int, float)) else float(value)
        except OverflowError:  # an int too large for a float
            number = None
        if number is None or not math.isfinite(number):
            logger.warning("dropping emotion %r: %r is not a finite number", name, value)
            continue
        groups.setdefault(canonical_name(name), []).append(number)
    averaged = {name: sum(values) / len(values) for name, values in groups.items()}
    if signed:
        return averaged
    return {name: _hold(value) for name, value in averaged.items()}


# --- runs and the shift rule ----------------------------------------------------------------------------------------


class Move(NamedTuple):
    """One calendar day's move of an average: the day, and the day's value minus the previous snapshot day's."""

    day: str
    delta: float


class Pending(NamedTuple):
    """A first day against a run, not yet followed by a second: the day, its kind ("zero" or "other"), whether it was felt."""

    day: str
    kind: str
    felt: bool


class FinishedRun(NamedTuple):
    start_day: str
    direction: int
    length: int  # calendar days from the start to the last with-direction day, both counted
    last_with_day: str


class CurrentRun(NamedTuple):
    direction: int
    start_day: str
    length: int  # blip days, off days and a pending first reverse day included
    last_with_day: str
    felt_count: int
    pending: Pending | None


class TrackerResult(NamedTuple):
    finished: tuple[FinishedRun, ...]
    current: CurrentRun | None


# A tracker seed: the stored record's start day, direction and last with-direction day, and the pending day's kind, which is
# never stored: None when the record's day is the last with-direction day itself, "zero" or "other" when it is the day after.
Seed = tuple[str, int, str, str | None]


def run_tracker(
    moves: Sequence[Move],
    felt: Sequence[bool] | None = None,
    seed: Seed | None = None,
) -> TrackerResult:
    """One pure pass over daily moves, oldest first, one per calendar day. Rules: A3 E5.21 steps 1 to 5.

    1. No run and a non-zero move: a run starts today (length 1). A zero move with no run starts nothing.
    2. A move with the run: it carries on, length + 1, and any pending first reverse day is cleared.
    3. A move against the run (the other way, or zero) with no pending day: today is the first reverse day, a blip. It adds 1 to
       the length and is marked pending, with its kind and felt flag.
    4. A move against the run with a pending day: the run ends, finishing before the first reverse day (the held length less
       1). The two reverse days are the first two days of the new run: both the same new way gives length 2 starting on the
       pending day; a zero first day and a non-zero today gives length 1 starting today; a zero today leaves no run.
    5. A calendar day between two snapshot days (the app off, or a day dropped as bad) is a zero move with no felt flag; the
       caller puts it in `moves`.

    `felt` (aligned with `moves`) is only passed for the short average; it counts the days she felt the emotion. `seed` makes
    the run already in progress when the first move arrives (the long average's stored record).
    """
    finished: list[FinishedRun] = []
    direction = 0
    start_day = last_with_day = ""
    length = felt_count = 0
    pending: Pending | None = None
    if seed is not None:
        start_day, direction, last_with_day, pending_kind = seed
        record_day = last_with_day if pending_kind is None else _add_days(last_with_day, 1)
        length = _days_between(start_day, record_day) + 1
        if pending_kind is not None:
            pending = Pending(record_day, pending_kind, False)
    for index, move in enumerate(moves):
        felt_today = bool(felt[index]) if felt is not None else False
        sign = (move.delta > 0) - (move.delta < 0)
        if direction == 0:
            if sign != 0:
                direction, start_day, last_with_day, length = sign, move.day, move.day, 1
                felt_count = int(felt_today)
                pending = None
            continue
        if sign == direction:
            length += 1
            felt_count += int(felt_today)
            last_with_day = move.day
            pending = None
        elif pending is None:
            length += 1
            felt_count += int(felt_today)
            pending = Pending(move.day, "zero" if sign == 0 else "other", felt_today)
        else:
            finished.append(FinishedRun(start_day, direction, length - 1, last_with_day))
            if sign == 0:
                direction = 0
                pending = None
            elif pending.kind == "other":
                # Both reverse days moved the same new way: the new run starts on the pending day with length 2.
                # Seeded pending day, oldest in the snapshot: no recorded kind and no previous day to read one from, so it is
                # read as a move the other way (accepted edge, owner 2026-10-28). A zero move would have started the new run a
                # day later with length 1.
                direction, start_day, last_with_day, length = sign, pending.day, move.day, 2
                felt_count = int(pending.felt) + int(felt_today)
                pending = None
            else:
                direction, start_day, last_with_day, length = sign, move.day, move.day, 1
                felt_count = int(felt_today)
                pending = None
    current = (
        CurrentRun(direction, start_day, length, last_with_day, felt_count, pending) if direction != 0 else None
    )
    return TrackerResult(tuple(finished), current)


def _seed_for(days: Sequence[SnapshotDay], index: int) -> Seed | None:
    """The tracker seed for the run record kept on `days[index]`, or None when that day has no run.

    The pending day's kind is worked out here from the snapshot's own move for that day (its long average minus the previous
    snapshot day's, compared exactly), and is "other" when the day is the oldest in the snapshot and so has no move to read.
    """
    record = days[index].run
    if record is None:
        return None
    if days[index].day == record.last_with_day:
        return (record.start_day, record.direction, record.last_with_day, None)
    if index == 0:
        return (record.start_day, record.direction, record.last_with_day, "other")
    kind = "zero" if days[index].long - days[index - 1].long == 0 else "other"
    return (record.start_day, record.direction, record.last_with_day, kind)


def _moves_between(
    days: Sequence[SnapshotDay], value_of: Callable[[SnapshotDay], float]
) -> tuple[list[Move], list[bool]]:
    """The daily moves of one average over consecutive snapshot days, with a zero move (not felt) for each calendar day between."""
    moves: list[Move] = []
    felts: list[bool] = []
    for previous, current in zip(days, days[1:], strict=False):
        for offset in range(1, _days_between(previous.day, current.day)):
            moves.append(Move(_add_days(previous.day, offset), 0.0))
            felts.append(False)
        moves.append(Move(current.day, value_of(current) - value_of(previous)))
        felts.append(current.felt)
    return moves, felts


def _shift_of(days: Sequence[SnapshotDay], persona_start_at: datetime | None) -> int:
    if not days or persona_start_at is None:
        return 0
    # The gate: the end of the last snapshot day must be at least 30 days after the persona's start.
    if _day_start(_add_days(days[-1].day, 1)) - persona_start_at < timedelta(days=dev_constants.EMOTION_LONG_HALF_LIFE_D):
        return 0
    long_moves, _ = _moves_between(days, lambda d: d.long)
    long_runs = run_tracker(long_moves, seed=_seed_for(days, 0))
    if not long_runs.finished:
        return 0  # quiet: the long average has no finished run in the window
    short_moves, short_felt = _moves_between(days, lambda d: d.short)
    run = run_tracker(short_moves, felt=short_felt).current
    if run is None:
        return 0
    if run.length <= max(r.length for r in long_runs.finished):
        return 0  # equal does not count
    if run.direction < 0 and not 2 * run.felt_count > run.length:
        return 0  # a fall counts only when felt on more of its days than not; a rise is never filtered
    return run.direction


def shift_for(snapshot_days: Mapping[str, Sequence[SnapshotDay]], persona_start_at: datetime | None) -> dict[str, int]:
    """For each emotion: +1 (a rising run), -1 (a falling run) or 0, from its 30-day snapshot. Rules: A3 E5.22 to E5.26.

    Shifted only when the start gate is open (the last snapshot day ends 30 or more days after `persona_start_at`; null is
    closed) and the long average has a finished run, and the short average's run in progress is longer than the longest
    finished long run (equal does not count; the long run is measured from its recorded start), and, for a fall, she felt the
    emotion on more than half of the run's days. Takes no `now`: the gate compares the last snapshot day.
    """
    return {name: _shift_of(days, persona_start_at) for name, days in snapshot_days.items()}


def feels_high_from(short: float, shift: int, snapshot: Sequence[SnapshotDay]) -> bool:
    """Whether a feeling is high: the one rule `engine.feels_high` (S52, S53) answers by. Change the rule here.

    Never high when `short`, its short average today, is below `FEELS_HIGH_FLOOR`: the floor comes first and holds for both
    parts below. At or above it, high when its shift reads stronger (`shift > 0`, a rising run; a falling run is not
    "stronger"), or when `short` is at or above `lowest + FEELS_HIGH_RANGE_FRACTION * (highest - lowest)`, where `lowest` and
    `highest` are the lowest and highest short average among the feeling's days in `snapshot` (the 30-day snapshot). With
    fewer than two snapshot days, or a flat range (`highest == lowest`), only the shift part applies. Pure: nothing is stored.
    """
    if short < dev_constants.FEELS_HIGH_FLOOR:
        return False
    if shift > 0:
        return True
    if len(snapshot) < 2:
        return False
    lowest = min(day.short for day in snapshot)
    highest = max(day.short for day in snapshot)
    if highest == lowest:
        return False
    return short >= lowest + dev_constants.FEELS_HIGH_RANGE_FRACTION * (highest - lowest)


# --- the day close ---------------------------------------------------------------------------------------------------


def close_days(state: EngineState, t: datetime) -> None:
    """Close the open active day when `t` falls on a later day. Pure: mutates `state`, writes nothing, takes no flag.

    With no `active_day` it only opens `day_of(t)`. On the same day or an earlier one (the clock was set back) it does
    nothing. Otherwise, for the open day D: every present emotion (long weight above 0) whose last-entry day is not D gets one
    entry of 0, weight 1, dated the first instant after D, folded into both averages (it does not set the last-entry day);
    then D is appended to every present emotion's snapshot, with the long run record carried forward from the previous
    snapshot day, and snapshot days 30 or more calendar days before D are dropped; then `last_update` moves past the whole
    calendar days the app was off (weights do not age across them) and `active_day` becomes `day_of(t)`.

    Accepted edges (`docs/source-spec/2026-10-28-emotion-rework-ThinkerOfThoughts-half-design.md` section 6):

    * A system clock set forward and then corrected: this function closes the open day and sets `active_day` to the far-future
      day, and then day closes stop until real time reaches that day, so the shift stays frozen at its last value and, with
      no day closing, the change-log files the drain would delete pile up. The deletion half belongs to the drain (it
      anchors on the active day as it stood before its closes). The open design call, telling a corrected clock from a long
      absence, needs a chosen number or a new rule; it is tracked in #660.
    * An entry dated in a day that has already closed cannot undo the zero already folded for it: that day holds both the zero
      and the entry.
    """
    today = day_of(t)
    if state.active_day is None:
        state.active_day = today
        return
    if today <= state.active_day:
        return
    closing = state.active_day
    after = _day_start(_add_days(closing, 1))
    needs_zero = [name for name, e in state.emotions.items() if e.long.weight > 0.0 and e.last_entry_day != closing]
    for name in needs_zero:
        _fold(state, name, _QUIET_DAY_VALUE, _QUIET_DAY_VALUE, _QUIET_DAY_WEIGHT, after, zero=True)
    window = dev_constants.EMOTION_LONG_HALF_LIFE_D
    for e in state.emotions.values():
        if e.long.weight <= 0.0:
            continue
        previous = e.snapshot[-1] if e.snapshot else None
        run: RunRecord | None = None
        if previous is not None:
            off_days = [Move(_add_days(previous.day, n), 0.0) for n in range(1, _days_between(previous.day, closing))]
            moves = [*off_days, Move(closing, e.long.avg - previous.long)]
            current = run_tracker(moves, seed=_seed_for(e.snapshot, len(e.snapshot) - 1)).current
            if current is not None:
                run = RunRecord(current.start_day, current.direction, current.last_with_day)
        e.snapshot.append(SnapshotDay(closing, e.short.avg, e.long.avg, e.last_entry_day == closing, run))
        e.snapshot = [d for d in e.snapshot if _days_between(d.day, closing) < window]
    if state.last_update is not None:
        state.last_update += _day_start(today) - after
    state.active_day = today

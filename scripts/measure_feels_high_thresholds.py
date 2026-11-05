"""# Safety tier: 1

Count how often each reader condition fires, old fixed level against the engine's `feels_high`, at several floors.

Usage:
    bash scripts/measure_feels_high_thresholds.sh <path to a memories.db> [--floors 0,0.1,0.25,0.5,1.0,2.0]
    .venv/bin/python scripts/measure_feels_high_thresholds.py <path to a memories.db> [--floors ...]

The path is required: the database is not in the repository. The file is copied into a fresh temporary directory (the main
database file only) and the copy is removed when the script ends. It refuses a path under the brain's home directory and
aborts on a non-empty -wal file (the same checks as scripts/time_old_emotion_paths.py, whose helpers it calls). It starts one
process (`git rev-parse HEAD`), calls no model, uses no network and reads no persona data file other than the path given.

What it does. The history goes through the emotion engine's own pure functions (`fold_event`, `close_days`, the climax reset
inside `fold_event`) from an empty state, with the live fold rules (quiet-day zeros in both averages). Ticks are on the
4-a-day UTC grid the shift sizing used (03, 09, 15 and 21 h). "Old" is `aggregate_state` over the 200 most recent
emotion-bearing rows at the tick, compared to each reader's old fixed level. "New" is the engine's short averages for present,
registered names, with every row asking `engine_utils.feels_high_from`, the one function `engine.feels_high` calls, with the
shift from `engine_utils.shift_for` (start gate included; the persona start is the oldest row's creation time, as the engine
sets it). The new arm is computed at every floor of `--floors` in one engine pass (the floor is `dev_constants.FEELS_HIGH_FLOOR`,
set for the duration of each floor's computation and restored). Floor 0 is the rule without a floor. Because phase B has no
rebuild the engine starts empty, so the comparison is biased for roughly the first 30 days; the "after day 30" counts remove it.

Two runs: every day active, and a fixed set of days off (rows created on an off day are dropped from both arms and no tick
runs that day). The days off are drawn with a fixed seed.

It prints, per run: a host line; tables of firing counts per reader row (old, then each floor); the shift part alone and the
range part alone per floor; the near-zero firings (rows that fire at floor 0 and stop firing once feelings with a short
average under 0.1 are not high); for each floor-0 firing its survival floor (the highest floor at which it would still
fire) in buckets; how many present feelings ever reach each floor; and the floor sweep summary R(F) with the plateau rule
(see "R and the proposal" below). It checks its own invariants and exits 1 when one fails: counts never rise as the floor
rises, a feeling called high at floor F is exactly a feeling called high at floor 0 whose short average is at least F,
near-zero firings are zero at any floor of 0.1 or more, and the dev constant is restored.

R and the proposal. R(F) is the number of firings removed by floor F, summed over both runs after day 30 and over the
reflex and temperature rows only (temperature: arousal, desire, belonging, love, climax, body_grief, touch_hunger; reflex arcs
loneliness_journal, defiance_burst, creative_pitch, self_check): the rows' floor-0 firings minus their firings at F. The
proposal is the smallest candidate floor (floor 0 is the baseline, not a candidate) whose next candidate removes nothing
more (R(next) = R(F)); with no such plateau it is the largest candidate. The proposal is a proposal: the owner decides.

On a failure it prints one line, `measure_feels_high_thresholds: FAILED at <step>: <detail>`, to stderr and exits 1.
"""

from __future__ import annotations

import argparse
import bisect
import importlib.util
import json
import random
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

GRID_HOURS = (3, 9, 15, 21)
SEED = 20261104
MIN_START_DAY = 36
MIN_APART = 5
DEFAULT_FLOORS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
NEAR_ZERO = 0.1
BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0)  # survival-floor bucket edges: under 0.1, 0.1 to 0.25, ... 2.0 and over

# The rows R(F) sums over (the reflex and temperature rows Hana named); every other row is reported, not summed.
R_ROWS = (
    "temperature: arousal",
    "temperature: desire",
    "temperature: belonging",
    "temperature: love",
    "temperature: climax",
    "temperature: body_grief",
    "temperature: touch_hunger",
    "reflex arc loneliness_journal",
    "reflex arc defiance_burst",
    "reflex arc creative_pitch",
    "reflex arc self_check",
)

HighFn = Callable[[str], bool]


def _load_timing_helpers():
    """The timing script's source vetting, copy, store open and vocabulary load, and its host line (one copy of each)."""
    path = REPO_ROOT / "scripts" / "time_old_emotion_paths.py"
    spec = importlib.util.spec_from_file_location("time_old_emotion_paths_helpers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def place_off_days(n_days: int) -> list[tuple[int, int]]:
    """Day-index spans (start, length): three spans of 3-10 days, then a one-day and a two-day gap.

    Fixed seed; none in the first 36 days; every span at least 5 days from every other.
    """
    rng = random.Random(SEED)
    placed: list[tuple[int, int]] = []
    for length in (rng.randint(3, 10), rng.randint(3, 10), rng.randint(3, 10), 1, 2):
        for _ in range(10_000):
            start = rng.randint(MIN_START_DAY, n_days - length - MIN_APART - 1)
            if all(start + length + MIN_APART <= s or start >= s + ln + MIN_APART for s, ln in placed):
                placed.append((start, length))
                break
        else:
            raise SystemExit("could not place the off-day spans")
    return placed


def reflex_arcs() -> list[tuple[str, dict[str, float]]]:
    data = json.loads((REPO_ROOT / "brain" / "engines" / "default_reflex_arcs.json").read_text(encoding="utf-8"))
    return [(a["name"], {k: float(v) for k, v in a["trigger"].items()}) for a in data["arcs"]]


# Old arm: the fixed levels each reader held before `feels_high` (kept only in the old code path).
def old_conditions(arcs: list[tuple[str, dict[str, float]]], many: int) -> dict[str, Callable[[dict[str, float]], bool]]:
    rows: dict[str, Callable[[dict[str, float]], bool]] = {
        f"body: more than {many} feelings high": lambda e: sum(1 for v in e.values() if v >= 7.0) > many,
        "body: rest_need": lambda e: e.get("rest_need", 0.0) >= 7.0,
        "body: peace": lambda e: e.get("peace", 0.0) >= 7.0,
        "temperature: arousal": lambda e: e.get("arousal", 0.0) >= 7.0,
        "temperature: desire": lambda e: e.get("desire", 0.0) >= 7.0,
        "temperature: belonging": lambda e: e.get("belonging", 0.0) >= 8.0,
        "temperature: love": lambda e: e.get("love", 0.0) >= 8.0,
        "temperature: climax": lambda e: e.get("climax", 0.0) >= 5.0,
        "temperature: body_grief": lambda e: e.get("body_grief", 0.0) >= 7.0,
        "temperature: touch_hunger": lambda e: e.get("touch_hunger", 0.0) >= 7.0,
        "research gate: any feeling high": lambda e: max(e.values(), default=0.0) >= 7.0,
    }
    for name, trig in arcs:
        rows[f"reflex arc {name}"] = lambda e, trig=trig: all(e.get(n, 0.0) >= t for n, t in trig.items())
    rows["reflex: some arc eligible"] = lambda e: any(
        all(e.get(n, 0.0) >= t for n, t in trig.items()) for _, trig in arcs
    )
    rows["expression arousal override"] = lambda e: e.get("arousal", 0.0) + e.get("desire", 0.0) >= 12.0
    rows["expression intent"] = lambda e: _top(e) == "creative_hunger" and e.get("creative_hunger", 0.0) >= 7.0
    return rows


def _top(e: dict[str, float]) -> str | None:
    best, value = None, 0.0
    for name, v in e.items():
        if v > value:
            best, value = name, v
    return best


# New arm: the same rows, every fixed level replaced by the predicate. `e` holds the short averages of present names.
def new_conditions(arcs: list[tuple[str, dict[str, float]]], many: int) -> dict[str, Callable[[dict[str, float], HighFn], bool]]:
    def names(*wanted: str) -> Callable[[dict[str, float], HighFn], bool]:
        return lambda e, high: all(n in e and high(n) for n in wanted)

    rows: dict[str, Callable[[dict[str, float], HighFn], bool]] = {
        f"body: more than {many} feelings high": lambda e, high: sum(1 for n in e if high(n)) > many,
        "body: rest_need": names("rest_need"),
        "body: peace": names("peace"),
        "temperature: arousal": names("arousal"),
        "temperature: desire": names("desire"),
        "temperature: belonging": names("belonging"),
        "temperature: love": names("love"),
        "temperature: climax": names("climax"),
        "temperature: body_grief": names("body_grief"),
        "temperature: touch_hunger": names("touch_hunger"),
        "research gate: any feeling high": lambda e, high: any(high(n) for n in e),
    }
    for name, trig in arcs:
        rows[f"reflex arc {name}"] = names(*trig)
    rows["reflex: some arc eligible"] = lambda e, high: any(names(*trig)(e, high) for _, trig in arcs)
    rows["expression arousal override"] = names("arousal", "desire")
    rows["expression intent"] = lambda e, high: (lambda t: t == "creative_hunger" and high(t))(_top(e))
    return rows


def strain_of(body_state) -> float:
    raw_exhaustion = float(body_state.exhaustion) / 9.0
    raw_energy_lack = 1.0 - float(body_state.energy) / 10.0
    return max(0.0, min(1.0, max(raw_exhaustion, raw_energy_lack)))


@dataclass
class RunResult:
    label: str
    n_ticks: int
    n_days: int
    n_after30: int
    n_post_gap: int
    floors: tuple[float, ...]
    # (row, key) -> per-tick 0/1; key is "old" or ("new", floor), ("shift", floor), ("range", floor)
    fire: dict = field(default_factory=dict)
    after30: list[int] = field(default_factory=list)
    post_gap: list[int] = field(default_factory=list)
    day_of_tick: list = field(default_factory=list)
    survival: dict = field(default_factory=lambda: defaultdict(list))  # row -> survival floors of floor-0 firings after day 30
    peak_short: dict = field(default_factory=dict)  # name -> highest short over the run
    high_means: dict = field(default_factory=dict)
    strain_means: dict = field(default_factory=dict)
    maker: dict = field(default_factory=dict)
    avg_present: float = 0.0
    quiet_days: int = 0
    invariant_failures: list = field(default_factory=list)


def run_one(label, rows, eb_rows, off_dates, dates, arcs, floors, rest_threshold) -> RunResult:
    from brain import dev_constants
    from brain.body.state import body_state_from_short, compute_body_state
    from brain.emotion import engine_utils as eu
    from brain.emotion import vocabulary
    from brain.emotion.aggregate import aggregate_state
    from brain.maker import charge as maker_charge
    from brain.maker import config as maker_cfg

    many = dev_constants.BODY_ENERGY_HIGH_COUNT
    old_rows = old_conditions(arcs, many)
    new_rows = new_conditions(arcs, many)
    row_names = list(old_rows)

    rows = [m for m in rows if m.created_at.date() not in off_dates]
    eb = [m for m in eb_rows if m.created_at.date() not in off_dates]
    eb_times = [m.created_at for m in eb]

    events: list = []
    for m in rows:
        kind = eu.counted_kind(m)
        if kind is None or not m.emotions:
            continue
        emotions = eu.resolve_emotions(m.emotions)
        if not emotions:
            continue
        events.append(
            eu.Event(
                kind="commit",
                at=m.created_at,
                memory_id=m.id,
                emotions=emotions,
                modifier=eu.source_modifier(m, 0.0),
                climax=float(m.emotions.get("climax", 0.0) or 0.0),
            )
        )
    event_times = [e.at for e in events]
    row_days = {e.at.date() for e in events}

    state = eu.EngineState()
    # The engine sets this from MIN(created_at) over the memory rows (the shift's start gate counts from it).
    state.persona_start_at = min(m.created_at for m in rows)
    last_tick = datetime.min.replace(tzinfo=UTC)

    result = RunResult(label, 0, 0, 0, 0, tuple(floors))
    fire: dict = defaultdict(list)
    after30: list[int] = []
    post_gap: list[int] = []
    day_of_tick: list = []
    first_date = dates[0]
    strain_base = strain_of(
        compute_body_state(emotions={}, session_hours=0.0, words_written=0, days_since_contact=0.0, now=datetime.now(UTC))
    )
    strain_vals: dict = defaultdict(list)
    arms = ("old",) + tuple(("new", f) for f in floors)
    charge_dir = {a: Path(tempfile.mkdtemp(prefix="measure-maker-")) for a in ("old", "new")}
    maker_state = {a: {"fires": 0, "last_fire": None, "max_charge": 0.0, "intensity_sum": 0.0} for a in ("old", "new")}
    registered_sizes: list[int] = []
    high_counts: dict = defaultdict(list)
    peak_short: dict[str, float] = {}
    prev_active_day = None
    since_gap = None
    quiet_days: set = set()
    failures: list[str] = []
    default_floor = dev_constants.FEELS_HIGH_FLOOR

    try:
        for date in dates:
            if date in off_dates:
                continue
            if date not in row_days:
                quiet_days.add(date)
            if prev_active_day is not None and (date - prev_active_day).days > 1:
                since_gap = 0
            elif since_gap is not None:
                since_gap += 1
            for hour in GRID_HOURS:
                t = datetime(date.year, date.month, date.day, hour, tzinfo=UTC)
                lo = bisect.bisect_right(event_times, last_tick)
                hi = bisect.bisect_right(event_times, t)
                for ev in events[lo:hi]:
                    eu.close_days(state, ev.at)
                    eu.fold_event(state, ev, ev.at)
                eu.close_days(state, t)
                last_tick = t

                idx = bisect.bisect_right(eb_times, t)
                recent = eb[max(0, idx - 200) : idx]
                old = dict(aggregate_state(recent).emotions)
                new = {
                    n: es.short.avg
                    for n, es in state.emotions.items()
                    if es.long.weight > 0.0 and vocabulary.get(n) is not None
                }
                registered_sizes.append(len(new))
                for n, v in new.items():
                    peak_short[n] = max(peak_short.get(n, 0.0), v)
                snaps = {n: state.emotions[n].snapshot for n in new}
                shifts = eu.shift_for(snaps, state.persona_start_at)

                highs: dict = {}
                try:
                    for f in floors:
                        dev_constants.FEELS_HIGH_FLOOR = f
                        highs[("new", f)] = {n for n in new if eu.feels_high_from(new[n], shifts[n], snaps[n])}
                        # the shift part alone (an empty snapshot closes the range part) and the range part alone (shift 0)
                        highs[("shift", f)] = {n for n in new if eu.feels_high_from(new[n], shifts[n], [])}
                        highs[("range", f)] = {n for n in new if eu.feels_high_from(new[n], 0, snaps[n])}
                finally:
                    dev_constants.FEELS_HIGH_FLOOR = default_floor
                base = highs[("new", floors[0])]
                for f in floors:
                    expected = {n for n in base if new[n] >= f}
                    if highs[("new", f)] != expected:
                        failures.append(f"{date} {hour}h floor {f}: the predicate disagrees with floor 0 filtered by the floor")
                for key, hs in highs.items():
                    high_counts[key].append(len(hs))

                day_of_tick.append(date)
                post_gap.append(int(since_gap is not None and since_gap <= 1))
                is_after30 = int((date - first_date).days >= 30)
                after30.append(is_after30)

                for name in row_names:
                    fire[(name, "old")].append(int(old_rows[name](old)))
                    for f in floors:
                        for part in ("new", "shift", "range"):
                            hs = highs[(part, f)]
                            fire[(name, (part, f))].append(int(new_rows[name](new, hs.__contains__)))
                    if is_after30 and new_rows[name](new, base.__contains__):
                        result.survival[name].append(_survival(new_rows[name], new, base))

                for arm in arms:
                    if arm == "old":
                        body = compute_body_state(
                            emotions=old, session_hours=0.0, words_written=0, days_since_contact=0.0, now=t
                        )
                    else:
                        body = body_state_from_short(
                            new,
                            feels_high=highs[arm].__contains__,
                            session_hours=0.0,
                            words_written=0,
                            days_since_contact=0.0,
                            now=t,
                        )
                    strain = strain_of(body)
                    strain_vals[arm].append(strain)
                    fire[("body_strain above its empty-state value", arm)].append(int(strain > strain_base + 1e-9))
                    fire[(f"rest gate (energy <= {rest_threshold})", arm)].append(int(body.energy <= rest_threshold))

                # maker: the real accumulate (soul and dream inputs zero), then the tick's own discharge test; the maker
                # feeds the peak short average, not a "high" test, so it does not move with the floor
                for arm, e in (("old", old), ("new", new)):
                    intensity = max(e.values(), default=0.0)
                    st = maker_charge.accumulate(
                        charge_dir[arm],
                        emotional_intensity=intensity,
                        soul_delta=0,
                        dream_count=0,
                        now=t,
                        w_emotion=maker_cfg.W_EMOTION,
                        w_soul=maker_cfg.W_SOUL,
                        w_dream=maker_cfg.W_DREAM,
                        decay_per_hour=maker_cfg.DECAY_PER_HOUR,
                    )
                    maker_state[arm]["max_charge"] = max(maker_state[arm]["max_charge"], st.charge)
                    maker_state[arm]["intensity_sum"] += intensity
                    if st.charge >= maker_cfg.DISCHARGE_THRESHOLD:
                        last = maker_state[arm]["last_fire"]
                        if last is None or (t - last).total_seconds() / 3600.0 >= maker_cfg.COOLDOWN_HOURS:
                            maker_state[arm]["fires"] += 1
                            maker_state[arm]["last_fire"] = t
                            st.charge = 0.0
                            st.last_fire_ts = t.isoformat()
                            maker_charge.save_charge(charge_dir[arm], st)
            prev_active_day = date
    finally:
        dev_constants.FEELS_HIGH_FLOOR = default_floor
        for d in charge_dir.values():
            shutil.rmtree(d, ignore_errors=True)

    if dev_constants.FEELS_HIGH_FLOOR != default_floor:
        failures.append("the dev constant was not restored")
    n_ticks = len(day_of_tick)
    result.n_ticks = n_ticks
    result.n_days = len(set(day_of_tick))
    result.n_after30 = sum(after30)
    result.n_post_gap = sum(post_gap)
    result.fire = fire
    result.after30 = after30
    result.post_gap = post_gap
    result.day_of_tick = day_of_tick
    result.peak_short = peak_short
    result.high_means = {k: sum(v) / max(1, n_ticks) for k, v in high_counts.items()}
    result.strain_means = {k: sum(v) / max(1, n_ticks) for k, v in strain_vals.items()}
    result.maker = {
        a: {**maker_state[a], "mean_intensity": maker_state[a]["intensity_sum"] / max(1, n_ticks)} for a in maker_state
    }
    result.avg_present = sum(registered_sizes) / max(1, n_ticks)
    result.quiet_days = len(quiet_days)
    result.invariant_failures = failures
    result.invariant_failures += _check_invariants(result, row_names, floors)
    return result


def _survival(cond: Callable[[dict[str, float], HighFn], bool], new: dict[str, float], highs: set[str]) -> float:
    """The highest floor at which this firing still fires (every row is monotone in the set of high feelings)."""
    for level in sorted({new[n] for n in highs}, reverse=True):
        kept = {n for n in highs if new[n] >= level}
        if cond(new, kept.__contains__):
            return level
    return 0.0


def _after(vals: list[int], flags: list[int]) -> int:
    return sum(v for v, f in zip(vals, flags, strict=True) if f)


def _check_invariants(r: RunResult, row_names: list[str], floors: tuple[float, ...]) -> list[str]:
    bad: list[str] = []
    for name in row_names:
        for part in ("new", "shift", "range"):
            for flags in (None, r.after30):
                counts = [
                    sum(r.fire[(name, (part, f))]) if flags is None else _after(r.fire[(name, (part, f))], flags)
                    for f in floors
                ]
                if any(b > a for a, b in zip(counts, counts[1:], strict=False)):
                    bad.append(f"{r.label}: {name} ({part}) fires more at a higher floor: {counts}")
        c0 = _after(r.fire[(name, ("new", floors[0]))], r.after30)
        for f in floors:
            if f > floors[0]:
                # a floor-0 firing survives floor F exactly when its survival floor is at least F (so no firing rests only
                # on a feeling under F, and near-zero firings are gone at any floor of 0.1 or more)
                survivors = c0 - sum(1 for s in r.survival[name] if s < f)
                if _after(r.fire[(name, ("new", f))], r.after30) != survivors:
                    bad.append(f"{r.label}: {name} at floor {f:g}: count differs from the floor-0 firings that survive it")
    return bad


def _count(r: RunResult, name: str, key, flags: list[int] | None) -> int:
    """Firings of one row in one arm: over the ticks `flags` selects, or over all ticks when `flags` is None."""
    values = r.fire[(name, key)]
    return _after(values, flags) if flags is not None else sum(values)


def _fmt(n: int, d: int) -> str:
    return f"{n}/{d}"


def report(results: list[RunResult], floors: tuple[float, ...], row_names: list[str]) -> tuple[list[str], dict]:
    out: list[str] = []
    for r in results:
        out.append(f"### Run: {r.label}")
        out.append(
            f"active days {r.n_days}, ticks {r.n_ticks} (4 a day: 03, 09, 15, 21 h UTC), quiet active days {r.quiet_days}; "
            f"ticks in the first two active days back from an off span {r.n_post_gap}; ticks after the engine's first 30 days {r.n_after30}"
        )
        out.append(f"new-arm average count of present registered feelings per tick: {r.avg_present:.1f}")
        for f in floors:
            out.append(
                f"floor {f:g}: feelings high per tick {r.high_means[('new', f)]:.1f} "
                f"(shift part alone {r.high_means[('shift', f)]:.2f}, range part alone {r.high_means[('range', f)]:.1f})"
            )
        out.append("")
        head = ["reader row", "old"] + [f"floor {f:g}" for f in floors]
        for title, flags in (("after day 30", r.after30), ("all ticks", None)):
            denom = r.n_after30 if flags is not None else r.n_ticks
            out.append(f"Firing counts, {title} (ticks out of {denom}):")
            out.append("")
            out.append("| " + " | ".join(head) + " |")
            out.append("|" + "---|" * len(head))
            for name in _row_order(r, row_names):
                cells = [_fmt(_count(r, name, "old", flags), denom)] + [
                    _fmt(_count(r, name, ("new", f), flags), denom) for f in floors
                ]
                out.append(f"| {name} | " + " | ".join(cells) + " |")
            out.append("")
        out.append("New arm after day 30 by part, per floor (shift part alone / range part alone), ticks out of " f"{r.n_after30}:")
        out.append("")
        out.append("| reader row | " + " | ".join(f"floor {f:g}" for f in floors) + " |")
        out.append("|" + "---|" * (len(floors) + 1))
        for name in _row_order(r, row_names):
            cells = [
                f"{_after(r.fire[(name, ('shift', f))], r.after30)} / {_after(r.fire[(name, ('range', f))], r.after30)}"
                for f in floors
            ]
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out.append("")
        out.append("Near-zero firings after day 30 (fires at floor 0, stops firing once feelings with a short average under "
                   f"{NEAR_ZERO:g} are not high), and the survival floor of every floor-0 firing in buckets "
                   "(the highest floor at which that firing would still fire):")
        out.append("")
        labels = [f"under {BUCKETS[0]:g}"] + [f"{a:g} to {b:g}" for a, b in zip(BUCKETS, BUCKETS[1:], strict=False)] + [f"{BUCKETS[-1]:g} and over"]
        out.append("| reader row | floor-0 firings | near-zero | " + " | ".join(labels) + " |")
        out.append("|" + "---|" * (len(labels) + 3))
        for name in _row_order(r, row_names):
            vals = r.survival[name]
            if not vals:
                continue
            counts = [0] * len(labels)
            for v in vals:
                counts[bisect.bisect_right(BUCKETS, v)] += 1
            near = sum(1 for v in vals if v < NEAR_ZERO)
            out.append(f"| {name} | {len(vals)} | {near} | " + " | ".join(str(c) for c in counts) + " |")
        out.append("")
        out.append("Present feelings whose highest short average over the run reaches each floor "
                   f"(of {len(r.peak_short)} that were present at some tick): "
                   + "; ".join(f"floor {f:g}: {sum(1 for v in r.peak_short.values() if v >= f)}" for f in floors))
        out.append("")
        out.append("body_strain mean: " + ", ".join(
            [f"old {r.strain_means['old']:.3f}"] + [f"floor {f:g} {r.strain_means[('new', f)]:.3f}" for f in floors]))
        for arm in ("old", "new"):
            m = r.maker[arm]
            out.append(
                f"maker discharges ({arm}): {m['fires']} over {r.n_days} simulated active days; mean peak intensity fed in "
                f"{m['mean_intensity']:.2f}; highest charge reached {m['max_charge']:.2f} (the maker does not use the predicate)"
            )
        out.append("")

    # R(F) over both runs, after day 30, over R_ROWS
    candidates = [f for f in floors if f > 0]
    baseline = floors[0]
    base_total = sum(_after(r.fire[(n, ("new", baseline))], r.after30) for r in results for n in R_ROWS)
    removed = {
        f: base_total - sum(_after(r.fire[(n, ("new", f))], r.after30) for r in results for n in R_ROWS)
        for f in floors
    }
    proposal = None
    for f, nxt in zip(candidates, candidates[1:], strict=False):
        if removed[nxt] == removed[f]:
            proposal = f
            break
    plateau = proposal is not None
    if proposal is None and candidates:
        proposal = candidates[-1]
    out.append("### R and the proposal")
    out.append("")
    out.append(f"Firings of the reflex and temperature rows at floor {baseline:g} (both runs, after day 30): {base_total}.")
    out.append("")
    out.append("| floor | firings of those rows | removed, R(F) |")
    out.append("|---|---|---|")
    for f in floors:
        total = base_total - removed[f]
        out.append(f"| {f:g} | {total} | {removed[f]} |")
    out.append("")
    if plateau:
        out.append(f"Proposal by the plateau rule: {proposal:g} (the next candidate removes nothing more).")
    else:
        out.append(f"No plateau through {proposal:g}: firings keep thinning as the floor rises; the plateau rule's fallback is {proposal:g}.")
    out.append("")
    return out, {"proposal": proposal, "plateau": plateau, "removed": removed, "base_total": base_total}


def _row_order(r: RunResult, row_names: list[str]) -> list[str]:
    rest = sorted({k[0] for k in r.fire if k[0] not in row_names})
    return row_names + rest


def parse_floors(text: str) -> tuple[float, ...]:
    try:
        floors = tuple(float(x) for x in text.split(",") if x.strip())
    except ValueError as exc:
        raise SystemExit(f"measure_feels_high_thresholds: --floors must be a comma list of numbers: {exc}") from exc
    if not floors or floors[0] != 0.0 or list(floors) != sorted(set(floors)):
        raise SystemExit("measure_feels_high_thresholds: --floors must start at 0 and ascend without repeats")
    return floors


def run(db: str, floors: tuple[float, ...]) -> tuple[list[str], int]:
    helpers = _load_timing_helpers()
    started = time.perf_counter()
    source, commit, code = helpers._step_source(db)
    from brain import dev_constants
    from brain.emotion import vocabulary
    from brain.memory.store import _row_to_memory
    from brain.paths import get_home

    scratch = helpers._guarded("copy", lambda: Path(tempfile.mkdtemp(prefix="measure-feels-high-")).resolve())
    store = None
    try:
        if helpers._under_home(scratch, get_home()):
            raise helpers.StepError("copy", f"the temporary directory is under the brain's home directory: {scratch}")
        digest = helpers._step_copy(source, scratch)
        store = helpers._step_open(scratch)
        registered, registry_size = helpers._step_vocabulary(scratch, store)
        rows = [
            _row_to_memory(r)
            for r in store._conn.execute("SELECT * FROM memories WHERE active = 1 ORDER BY created_at").fetchall()  # noqa: SLF001
        ]
        rows = [m for m in rows if m.created_at is not None]
        rows.sort(key=lambda m: m.created_at)
        eb_rows = [m for m in rows if m.emotions]
        first, last = rows[0].created_at.date(), rows[-1].created_at.date()
        dates = [first + timedelta(days=i) for i in range((last - first).days + 1)]
        arcs = reflex_arcs()
        from brain.initiate import review

        rest_threshold = review._REST_ENERGY_THRESHOLD  # noqa: SLF001

        spans = place_off_days(len(dates))
        off_dates = {dates[s + k] for s, ln in spans for k in range(ln)}
        row_names = list(old_conditions(arcs, dev_constants.BODY_ENERGY_HIGH_COUNT))
        lines = ["# Feels-high threshold measurement (old fixed levels against the predicate, at several floors)", ""]
        lines.append(helpers._host_line())
        lines.append(f"commit: {commit}")
        lines.append(f"database: sha256 {digest}; {len(rows)} active rows, {len(eb_rows)} with emotions; {first} to {last} ({len(dates)} days)")
        lines.append(f"registered emotion names from the first vocabulary load: {registered}; vocabulary size after load {len(vocabulary.list_all())}")
        lines.append(f"floors: {', '.join(f'{f:g}' for f in floors)} (floor 0 is the rule without a floor)")
        lines.append("Days off in run 2 (UTC dates; index counts from the first row's day = 0; seed "
                     f"{SEED}): " + "; ".join(
                         f"{dates[s]} to {dates[s + ln - 1]} ({ln} day{'s' if ln > 1 else ''}, index {s})" for s, ln in spans))
        lines.append("")
        results = [
            run_one("every day active", rows, eb_rows, set(), dates, arcs, floors, rest_threshold),
            run_one("days off (the set above)", rows, eb_rows, off_dates, dates, arcs, floors, rest_threshold),
        ]
        body, summary = report(results, floors, row_names)
        lines += body
        failures = [f for r in results for f in r.invariant_failures]
        if failures:
            lines.append("INVARIANT FAILED:")
            lines += [f"- {f}" for f in failures]
        lines.append(f"elapsed: {time.perf_counter() - started:.0f} seconds")
        return lines, (1 if failures else 0)
    finally:
        try:
            if store is not None:
                store.close()
        except Exception as exc:  # noqa: BLE001 - a failed close must not stop the directory being removed
            print(f"measure_feels_high_thresholds: could not close the copy: {exc}", file=sys.stderr)
        try:
            shutil.rmtree(scratch)
        except OSError as exc:
            print(f"measure_feels_high_thresholds: could not remove {scratch}: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("db", help="path to a memories.db (not in the repository)")
    parser.add_argument(
        "--floors",
        default=",".join(f"{f:g}" for f in DEFAULT_FLOORS),
        help="comma list of floors, ascending, starting at 0 (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    floors = parse_floors(args.floors)
    helpers = _load_timing_helpers()
    try:
        lines, code = run(args.db, floors)
    except helpers.StepError as exc:
        print(f"measure_feels_high_thresholds: FAILED at {exc.step}: {exc.detail}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main())

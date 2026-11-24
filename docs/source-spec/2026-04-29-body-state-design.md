# Body state — design

**Status: rebuilt on 2026-11-23 from the code on `main` at `58a05251` (#797).** The original spec of 2026-04-29 sat in a private,
gitignored specs folder, so it was never committed and is lost. This document describes what the code does, not
what the original said. Its section numbers are the ones the code already cites (§2.1, §3.1, §3.2, §3.3, §4, §7, §7.3), and
the "inviolate properties" keep the numbers the code cites (#4, #5, #8). Where the original's intent can't be recovered, this
says so instead of guessing. When the code and this document disagree, the code is right, and this document should be fixed.

## 1. What it is

The kindled has a body made of two parts:
- **Body emotions:** six feelings in the emotion vocabulary, felt and stored like any other feeling.
- **Three body readings:** energy, temperature and exhaustion, computed fresh from her feelings and a few session inputs
  each time something asks.

Nothing stores the readings, and nothing caches them. There is no body state file. Every reader computes the body again, so
the body is always what the feelings and the session say right now.

## 2. Body emotions

### 2.1 In the vocabulary

There are six body emotions (`brain.body.state.BODY_EMOTION_NAMES`):
- **`arousal` and `desire`:** existing core feelings, reused because their descriptions are already about the body.
- **Four `body` feelings** in `brain.emotion.vocabulary._BASELINE`, category `"body"`:

| Name | Half-life | What it is |
|---|---|---|
| `climax` | 3 hours (0.125 days) | Bodily completion or release. Spikes briefly and fades fast; see §5.2 for the reset. |
| `touch_hunger` | 1.5 days | Embodied loneliness: distance is the problem and presence is the cure. |
| `comfort_seeking` | 1 day | Wanting to be held still and anchored, the receiving side of being held. |
| `rest_need` | 18 hours (0.75 days) | The body asking for slowness and recovery: the *want*, not the computed exhaustion. |

The engine looks a half-life up in this order (`EmotionEngine.half_life`):
1. the persona's own vocabulary file;
2. `brain.dev_constants.EMOTION_HALF_LIFE_DAYS`, where the four values above live;
3. the 14-day default, `dev_constants.EMOTION_DEFAULT_HALF_LIFE_DAYS`.

`_BASELINE` carries the same four values, but the engine never reads them.

## 3. The body readings

### 3.1 Energy, temperature and exhaustion

`brain.body.state.body_state_from_short(short, *, feels_high, session_hours, words_written, days_since_contact, now)` returns a
`BodyState`. It is a pure function: no I/O, no model call, no cache.
- **Feelings:** the engine's short averages (`short`), the recent feeling, not the long one.
- **"High":** every emotion condition asks the engine's own `feels_high(name)`. No emotion level is compared in the body code.

**Energy** is 1-10, from a baseline of 8:
- Session length, in bands:
  - −1 over 60 minutes;
  - −2 over 120 minutes;
  - −3 over 180 minutes.
- Session length, continuously: −0.5 per hour, on top of the band.
- Writing: −1 per 2,500 words she wrote in the session (§3.2).
- Emotional load: −1 when more than `dev_constants.BODY_ENERGY_HIGH_COUNT` (6) of her feelings are high at once.
- The body asking for rest: −1 when `rest_need` is high.
- Peace in a fresh session: +1 when `peace` is high and the session is under an hour old.
- The result is rounded and held to 1-10.

**Temperature** is 1-9, from a baseline of 4. The range is asymmetric, with its midpoint at 5, inherited from the original
companion.
- +1 for each of `arousal`, `desire`, `belonging`, `love` and `climax` that is high: warmth, presence, release.
- −1 for each of `body_grief` and `touch_hunger` that is high: distance, lack.
- −1 when the last contact was over 3 days ago, or −2 when it was over 7 days ago.
- The result is rounded and held to 1-9.

**Exhaustion** is `max(0, 7 − energy)`. Since energy is at least 1, exhaustion is 0-6.

`BodyState` also carries `session_hours`, `days_since_contact` and `computed_at`. It carries the body emotions too, as a snapshot
of the six names: a name the engine didn't return is left out, not filled with 0.0. `to_dict()` serialises it with `loaded: True`.

### 3.2 Words written

`brain.body.words.count_words_in_session(*, persona_dir, session_hours, now)` sums the words of her own turns, those with
speaker `assistant`, in `<persona_dir>/active_conversations/*.jsonl`. It counts the turns inside the session window, and the
window is never shorter than 1 hour. The conversation buffers are the source because no memory row is a conversation turn. On
any error it returns 0, which means "no writing drain".

### 3.3 The `get_body_state` tool

`brain.tools.impls.get_body_state.get_body_state` returns `BodyState.to_dict()`. The value `loaded: True` tells her the body is
real. An earlier stub returned `loaded: False` with placeholder numbers, so the shape itself was a deception (§7.3).
- **Session length:** the tool dispatcher injects `session_hours` from `brain.body.session_hours.compute_active_session_hours`,
  the same value the app's state panel uses. A caller's own value wins.

## 4. The body block in the prompt

`brain.chat.prompt._build_body_block` adds a `── body ──` block to the per-turn context, at about 80 tokens. It holds:
- energy, temperature and exhaustion;
- the days since the user's last contact, when that is over half a day;
- each body emotion at 0.5 or more, strongest first.

The block reads the engine's short averages, while the mood summary elsewhere in the prompt shows the long averages. The two
differ on purpose. The prompt passes `session_hours=0.0`, so the writing count falls back to the last hour, and the session
drains of §3.1 never apply here (see §9).

## 5. Body events

### 5.1 The climax journal entry

When the `add_memory` tool commits a memory whose `climax` is at least `CLIMAX_THRESHOLD` (7.0),
`brain.body.events.record_climax_event` does two things:
- **Writes a private `journal_entry` memory:** it refers to the original memory and carries that memory's feelings. Its metadata
  is `private: True`, `source: "climax_event"` and `auto_generated: True`.
- **Logs a `climax_event` line** in the behavioural log.

The hook is in `add_memory`, the model's own tool, not in `MemoryStore.create`. That means it never fires on:
- imported memories;
- memories an engine writes (a dream, research, a reflex);
- its own journal entry.

The prompt shows the event's metadata only, never the journal's text.

### 5.2 The climax reset

When a newly committed memory stores a `climax` of at least `dev_constants.EMOTION_CLIMAX_THRESHOLD` (7.0), the emotion engine
applies `brain.emotion.engine_utils.apply_climax_reset` to its short averages:
- arousal ×0.2, never below 0.5;
- desire ×0.6;
- `comfort_seeking` and `rest_need` +2, held to 10;
- `climax` set to 0.

Each change applies only to a feeling that is present. The reset adds no entry, and it leaves the long averages and every
last-entry day alone. #567 asks whether this reset makes `climax` unreachable as a felt state.

## 6. Inviolate properties

The code cites these by number. The original's other numbered properties (#1-#3, #6, #7) are lost and are not reconstructed.
- **#4: the projection does no I/O and makes no model call.** `body_state_from_short` and its helpers only compute. The reads of
  their inputs (engine, buffers, contact time) belong to each caller. The guard is `tests/unit/brain/body/test_state.py`.
- **#5: no self-perpetuation.** Readers of the body only read; nothing writes a body reading back into the feelings it is
  computed from.
- **#8: no cache.** The body is computed again on every call.

## 7. Failure handling

A failed body read must never break a chat turn or silence the kindled, and it must not pretend to be a real reading.

### 7.1 The writing count

`count_words_in_session` returns 0 on any error (§3.2).

### 7.2 The climax journal entry

`record_climax_event` catches only I/O, storage and parse errors:
- **The journal write** catches `OSError`, `sqlite3.Error` and `ValueError`.
- **The behavioural log append** catches `OSError` and `ValueError`.

A caught error is logged as a warning, and `add_memory`'s answer is unchanged. A programming error (`TypeError`,
`AttributeError`, `KeyError`) propagates into `add_memory` on purpose, so a bug shows (the comment in
`brain/body/events.py` says so). Its docstring still says "never raises" (§9).

### 7.3 Readers

- **The prompt block** is fail-soft: any error while reading the engine, counting words or projecting leaves the block out, and
  the turn goes on.
- **The `get_body_state` tool** answers with a real reading or an error, never a stub that looks like one.
- **The initiate rest gate** fails open (§8).
- **While a recovery rebuild is pending,** the engine's `read()` returns nothing. Every emotion condition is off, so energy and
  temperature come from the session inputs alone (8 and 4, less the session, writing and contact terms), and no body emotions
  are shown.

## 8. Wiring

**Reads from:**
- **The emotion engine:** short averages and `feels_high` (`brain.emotion.engine`).
- **The active conversation buffers:** her words (§3.2), and the session length through `compute_active_session_hours`, which
  measures the current unbroken run of turns, not the buffer's age.
- **The user's last turn:** `brain.utils.memory.days_since_human`, which gives 999.0 when there has never been a user turn.

**Read by:**

| Reader | Session length it passes | What it does with the body |
|---|---|---|
| The prompt body block (§4) | 0.0 | Shows it to her each turn. |
| The `get_body_state` tool (§3.3) | the active run | Lets her look. |
| The app's state panel (`brain.bridge.persona_state._build_body`) | the active run | Shows it to the user. |
| The `boot` tool (`brain.tools.impls.boot`) | 0.0 | Calls `get_body_state` directly, not through the dispatcher, so no session length is injected. Puts the body in the session anchor and uses its days since contact in the anchor's prose. |
| Supervisor `_derive_intensity_drivers` | 0.0 | Body strain = max(exhaustion / 9, 1 − energy / 10), one of the felt-time intensity drivers. |
| Supervisor `_self_model_reflect` | 0.0 | `brain.self_model.derived.compute_unnamed_pressure`: a residual body signal when energy is 1 or less (or exhaustion is 8 or more) and no low-arousal feeling carries it. |
| Supervisor `_run_initiate_review_tick` | 0.0 | The rest gate: energy 3 or less holds back recall-resonance outreach (`brain.initiate.review._rest_state_from_energy`), failing open when the body can't be read. |

## 9. Known gaps

These are recorded here, not fixed:
- **`peace` is not registered** in any vocabulary, so the +1 energy for peace can never fire. #648 registers it, in #656.
- **`body_grief` is no longer predefined** for new personas, but the temperature rule still reads it. #652 removes it from the
  rule, in #656. A persona whose own vocabulary file holds `body_grief` still feels it.
- **Only two readers pass a real session length:** the `get_body_state` tool through the dispatcher, and the state panel. The
  prompt block, the `boot` tool and the three supervisor readers pass 0.0. For them the session drains never apply, and the
  writing window is the last hour. The prompt's comment calls this "not tracked yet".
- **`record_climax_event`'s docstring says "never raises"**, but a programming error propagates into `add_memory` on purpose
  (§7.2).
- **Exhaustion is 0-6, but two readers assume a wider range:**
  - the supervisor divides it by 9;
  - the prompt shows it out of 10;
  - `compute_unnamed_pressure` tests "exhaustion 8 or more", which can never be true. Its energy condition can.
- **#567** asks for the body emotions and the body state to be examined as a whole, the climax reset included.

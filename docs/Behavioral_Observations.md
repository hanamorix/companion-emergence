# Behavioral Observations

A running, informal log of small behavioral quirks observed in the companion's substrate model, kept so we do not rediscover them the hard way. Append new entries at the top with a date. Each entry should state the general behavior first, then any concrete case that surfaced it, and why it may matter later.

---

## 2026-08-25 — Small/fast substrate models take prompt phrasing literally

**Behavior:** a smaller/faster substrate model (Haiku) tends to answer a figure of speech phrased as a bare question literally, rather than as the intended metaphor. The larger the model, the more it tolerates figurative phrasing; the smaller, the more literal.

**Case:** the self_model note prompt framed the persona's emotional state as "weather" and closed with `What's the weather?`. On Haiku, a faithful context-free call answered it as a real weather request 10/10 (offering a forecast, asking for a location) instead of the intended one-sentence note. Rephrasing to `What's the (metaphorical) weather?` dropped it to 0/10 (a metaphor-consistent embellishment remained ~3/10).

**Why it matters:** when prompting the small/housekeeping model, do not rely on it to infer that a question is figurative. Mark the metaphor explicitly, or state the task plainly rather than as a question. A prompt that reads fine on the chat model can misfire on the smaller one.

---

### Disproven candidates (kept so they are not re-investigated)

- "The substrate under-uses context already in front of it, preferring an active tool call" — **disproven** on 2026-08-19. It was a live-test harness artifact, not a behavior: the harness ran a split-brain where the tool-serving process loaded stale code, so `read_full_memory` was uncallable. On the fixed harness a valid 15-turn discrimination test showed the model does open relevant surfaced snippets and skip irrelevant ones.

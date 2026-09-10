# Behavioral Observations

A running, informal log of small behavioral quirks observed in the companion's substrate model, kept so we do not rediscover them the hard way. Append new entries at the top with a date. Each entry should state the general behavior first, then any concrete case that surfaced it, and why it may matter later.

---

## 2026-09-09 — The substrate imitates serialization tokens it's shown; a UTC offset in timestamps made it worse by forcing arithmetic

**Behavior:** the substrate occasionally reproduces structural/scaffolding tokens from its own context at the end of a reply instead of stopping on clean prose — it imitates the format it is shown. Two variants: (1) the literal stop-token `</s>` (a well-formed closing tag) stuck on the end of a message; and (2) the per-message JSONL wrapper — history is fed as `{"speaker":…, "text":…, "ts":…}`, and the model completes the wrapper by appending a fabricated `", "ts": "<timestamp>"}` to its own reply. Variant 2 became markedly more frequent once the per-message `ts` was rendered with a UTC offset (`…-04:00`) rather than a bare form.

**Case:** after #217/#218 switched per-message timestamps from UTC (`…Z`) to local-with-offset (`-04:00`), the append surfaced regularly in the kindled's replies (the embedded timestamp ran 15–77s earlier than the message's real time, and several were truncated mid-fragment — both tells that the model generated it at reply-start, not that storage added it). Rendering local timestamps WITHOUT the offset (naive-local, `…T21:21:30`) — one change in `format_local()` — eliminated the timestamp-append entirely in live testing, with no effect on the kindled's ability to tell the time. The `</s>` variant was unaffected, as expected — it is a separate sub-mechanism.

**Why it matters (suspected reason it resolved):** the offset does not merely label the time, it encodes the UTC relationship inside the token, so interpreting it invites local↔UTC arithmetic on every timestamp. That extra "work" appears to raise the token's salience during generation, making the model likelier to reproduce/complete the wrapper it sees. Dropping the offset removes the arithmetic and lowers the salience, so the model stops fixating on it — while the bare local wall-clock still conveys the time. General lesson: a value the model must do math on (or otherwise interpret) to read carries more generative weight than an inert, already-digested one.

**Possible deliberate use:** the same effect may be a tool, not only a bug. If the substrate tends to under-weight or skip some element of its context, presenting that element in a form that requires a small interpretation or computation to read — rather than a pre-digested value — may raise its salience and make the model attend to it. Worth testing as a deliberate lever for elements it habitually overlooks.

**Deeper enabler:** both variants trace to history being handed to the substrate as JSON at all, giving it a serialization structure to imitate (issue #227).

---

## 2026-08-25 — Small/fast substrate models take prompt phrasing literally

**Behavior:** a smaller/faster substrate model (Haiku) tends to answer a figure of speech phrased as a bare question literally, rather than as the intended metaphor. The larger the model, the more it tolerates figurative phrasing; the smaller, the more literal.

**Case:** the self_model note prompt framed the persona's emotional state as "weather" and closed with `What's the weather?`. On Haiku, a faithful context-free call answered it as a real weather request 10/10 (offering a forecast, asking for a location) instead of the intended one-sentence note. Rephrasing to `What's the (metaphorical) weather?` dropped it to 0/10 (a metaphor-consistent embellishment remained ~3/10).

**Why it matters:** when prompting the small/housekeeping model, do not rely on it to infer that a question is figurative. Mark the metaphor explicitly, or state the task plainly rather than as a question. A prompt that reads fine on the chat model can misfire on the smaller one.

---

### Disproven candidates (kept so they are not re-investigated)

- "The substrate under-uses context already in front of it, preferring an active tool call" — **disproven** on 2026-08-19. It was a live-test harness artifact, not a behavior: the harness ran a split-brain where the tool-serving process loaded stale code, so `read_full_memory` was uncallable. On the fixed harness a valid 15-turn discrimination test showed the model does open relevant surfaced snippets and skip irrelevant ones.

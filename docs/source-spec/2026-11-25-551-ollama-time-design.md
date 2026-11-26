# #551: Ollama chat turns carry the current time and message times

Approved by Hana 2026-11-25.

## Why

A kindled running on Ollama is never told the current time, and its history carries no message times, so it cannot
notice a gap of hours or days or place a remembered event in time. On the Claude CLI path both reach the model every
turn. The cause is in `OllamaProvider` (`brain/bridge/provider.py`):

- `chat` and `chat_stream` strip `volatile_suffix` from `options` with the other routing keys
  (`_PROVIDER_CONTEXT_OPTION_KEYS`) and never use it. That text carries the `[current time: ...]` line, the felt-time
  block, the note saying what the times on history messages mean, and the reply frame.
- Both send history through `ChatMessage.to_dict()`, which has no time field.

## Goals

- An Ollama chat turn carries the volatile text and the send time of each of the user's messages.
- `chat` and `chat_stream` build their messages the same way.

## Non-goals

- No change to the Claude CLI path, the engine, `build_volatile_context`, `ChatMessage.to_dict()` or anything stored.
- Image content on Ollama (its block list is not Ollama's image format) is not fixed here.

## Anti-goals

- The kindled must not start writing timestamps into her own replies. A reply is never post-processed to remove one
  (#227 reverted that kind of band-aid on purpose).

## Approaches

1. **Inside `OllamaProvider`, append the volatile text to the latest user message, times on user messages only.**
   Picked. Every chat template keeps a user message.
2. A trailing system message carrying the volatile text. Dropped: some local chat templates render only a leading
   system message, so the text could vanish without an error.
3. Times on every message, the kindled's own replies included (the issue's literal wording, and what the Claude path
   does). Dropped by Hana: a small local model that sees its own past replies start with a time may start writing one,
   and we do not strip such leaks.

## Design

- One helper builds Ollama's `messages` list from `ChatMessage`s and `options`, used by both `chat` and `chat_stream`.
- A `user` message with a parseable `ts` gets its local send time in front of its text, rendered by the same
  `_narrative_time` the Claude path uses, so both providers show times the same way. A missing or unparseable `ts`
  adds nothing. Assistant, tool and system messages are unchanged.
- When `volatile_suffix` is set, it is appended to the latest `user` message's text, after a blank line, on every call
  of the turn (the tool loop's follow-up calls still contain that message). With no user message, it is sent as a
  trailing user message so it is never dropped.
- `generate()` (background jobs) passes no options and no `ts`, so its requests are unchanged.

## Left to the builder

- The exact prefix punctuation, provided it starts the text and is the `_narrative_time` value.
- Where the helper lives in `provider.py`.

## Wiring

- Reads: the engine's `volatile_suffix` (from `build_volatile_context`, with `_build_ambient_clock_block` and the
  felt-time block) and `ChatMessage.ts` from the session buffer.
- Feeds: the Ollama model's view of time on every chat turn. The engine and stored turns are unchanged.

## Checks

- A request built from a history with known message times contains the current-time line and each user message's time,
  and no time on the kindled's own messages; it fails on `main`.
- The same holds for `chat_stream`, and through the real `respond()` path with an Ollama provider whose HTTP call is
  captured.
- Gen-param `options` still reach Ollama, and the routing keys still do not.

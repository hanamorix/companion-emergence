# #551 Ollama time Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An Ollama chat turn carries the per-turn volatile text (current time, felt time, reply frame) and each user
message's send time.

**Architecture:** One module-level helper in `brain/bridge/provider.py` turns `ChatMessage`s plus `options` into
Ollama's `messages` list; `OllamaProvider.chat` and `chat_stream` both call it instead of `[m.to_dict() for m in
messages]`.

**Tech Stack:** Python 3.13, pytest, httpx (patched in tests).

**Spec:** `docs/source-spec/2026-11-25-551-ollama-time-design.md`

## Global Constraints

- Times go on `user` messages only; assistant, tool and system messages are sent exactly as `to_dict()` gives them.
- The time is `_narrative_time(msg.ts)`; an empty result adds nothing.
- No change to the Claude CLI path, the engine, `ChatMessage`, `build_volatile_context` or anything stored.
- Replies are never post-processed.
- Tests: `scripts/clean_shell.sh uv run pytest`; every new test must fail on `origin/main`
  (`scripts/new_tests_fail_on_base.py`). tdd-guard allows one new test per edit.

## Review Focus

1. Tool-loop follow-up calls: the latest user message is no longer last (assistant tool call + tool result follow);
   the suffix still lands on it, once, and calling `chat` again with the same list gives the same payload (no
   accumulation).
2. A request with no user message but a suffix: the suffix is sent as a trailing `user` message, not dropped.
3. A user message whose `ts` is missing or unparseable gets no prefix and no crash.
4. A user message with image blocks (`to_dict()` gives a list): the time and suffix go in as `{"type": "text",
   "text": ...}` entries (first and last), and nothing raises.
5. Gen-param options still reach `payload["options"]`; routing keys, `volatile_suffix` included, still do not.

---

### Task 1: Ollama messages carry the volatile text and user send times

**Files:**
- Modify: `brain/bridge/provider.py` (`OllamaProvider.chat`, `OllamaProvider.chat_stream`, new helper next to
  `_claude_context_lines`)
- Test: `tests/unit/brain/bridge/test_ollama_time_551.py` (new)

**Interfaces:**
- Produces: `_ollama_messages(messages: list[ChatMessage], options: dict[str, Any] | None) -> list[dict[str, Any]]`.
  String content of a timed user message becomes `f"{when} {text}"`; the latest user message's string content becomes
  `f"{content}\n\n{suffix}"`; with no user message, `{"role": "user", "content": suffix}` is appended.

- [ ] **Step 1: Write the failing `chat` test** (`httpx.post` patched with the `_make_ollama_response` shape from
  `test_provider_chat.py`)

```python
def test_551_chat_sends_the_volatile_text_and_each_user_messages_time() -> None:
    # history: system "SYS"; user "hey nell" ts T1; assistant "i am here" ts T2; user "no time" ts None;
    # user "bad time" ts "not-a-time"; user "and now?" ts T3
    # options: {"volatile_suffix": "[current time: NOW]\nNOTE", "include_block_clock": False,
    #           "session_id": "s", "temperature": 0.4}
    sent = payload["messages"]
    assert sent[0] == {"role": "system", "content": "SYS"}
    assert sent[1]["content"] == f"{_narrative_time(T1)} hey nell"
    assert sent[2] == {"role": "assistant", "content": "i am here"}
    assert sent[3]["content"] == "no time"
    assert sent[4]["content"] == "bad time"
    assert sent[5]["content"] == f"{_narrative_time(T3)} and now?\n\n[current time: NOW]\nNOTE"
    assert payload["options"] == {"temperature": 0.4}
```

- [ ] **Step 2: Run it, expect FAIL** (`sent[1]["content"] == "hey nell"` on main)

Run: `scripts/clean_shell.sh uv run pytest tests/unit/brain/bridge/test_ollama_time_551.py -q -o "tdd_guard_project_root=<main checkout>"`

- [ ] **Step 3: Implement `_ollama_messages` and call it from `chat`.**

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Add, one per edit, each red on main then green:**
  - `test_551_chat_stream_sends_the_same_messages` — `httpx.stream` patched; the streamed payload's `messages` equal
    what `chat` sends for the same input (call `chat_stream` and assert the same three facts as Step 1's last line).
  - `test_551_the_suffix_lands_on_the_latest_user_message_in_a_tool_loop_follow_up` — messages: system, user T1,
    assistant with a tool call, tool result. The user entry ends with `"\n\nSUFFIX"`, the tool entry is unchanged, and
    two `chat` calls with the same list send identical `messages`.
  - `test_551_with_no_user_message_the_suffix_is_sent_as_a_trailing_user_message` — `[system]` + suffix → last entry
    `{"role": "user", "content": "SUFFIX"}`.
  - `test_551_an_image_user_message_gets_the_time_and_suffix_as_text_blocks` — user content
    `(TextBlock("look"), ImageBlock(...))` with ts T1 + suffix: list starts with `{"type": "text", "text": when}` and
    ends with `{"type": "text", "text": "SUFFIX"}`.
  - `test_551_respond_through_ollama_carries_the_clock_and_user_times` — the `test_context_narrative_path.py` harness
    (`_persona`, `_seed_history`, `respond`) with `OllamaProvider()` and `httpx.post` patched: the last user entry
    contains `"[current time: "` and `"The time at the start of each conversation line above"`, the seeded user
    entries start with their `_narrative_time`, and the seeded assistant entries equal their raw text.

- [ ] **Step 6: Wire `chat_stream` to the helper; full file green.**

- [ ] **Step 7: Full gate, then commit** (`pytest`, `ruff`, `lint_windows_pitfalls.py`, `pnpm test`, `pnpm build`,
  then `new_tests_fail_on_base.py --base origin/main` and the leak scan).

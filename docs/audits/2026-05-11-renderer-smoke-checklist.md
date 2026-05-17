# v0.0.7-alpha Renderer Smoke Checklist

**For:** Hana, against a live macOS v0.0.6-alpha (or v0.0.7-alpha if it's already built) bundle.
**Output:** Tick boxes below. Anything broken or off becomes F-2NN findings appended to `2026-05-11-v0.0.7-comprehensive-audit.md` (the master audit doc).

## Setup

- [ ] Boot Companion Emergence app from `/Applications`.
- [ ] No console errors in the Tauri devtools window on launch.
- [ ] Connection panel shows expected provider (claude-cli or whatever you've configured) + bridge state (auto-starting / running).

## Conversation flow

- [ ] Send a chat turn. Reply lands. No console errors.
- [ ] Send 5+ turns back-to-back; verify autoscroll-on-new-message works.
- [ ] Stream a long reply (ask Nell something open-ended). Mid-stream, hit Cmd-Q. App quits cleanly — no hang, no traceback in the saved logs.
- [ ] Reopen. Open same persona. Session resumes mid-thread (Phase B sticky-session behaviour).
- [ ] Walk away 6+ minutes (you can fake this by manually editing the buffer's most-recent turn timestamp to be 6+ min in the past, then triggering a supervisor tick). Return. Session still alive, conversation continues seamlessly.

## Persona management

- [ ] Create a new persona via the wizard. Try a bad name (e.g. `nell smith` with a space) — validation rejects it with a clear error.
- [ ] Try a very long name (>40 chars) — validation rejects it.
- [ ] Try a name with a slash (e.g. `nell/two`) — validation rejects it.
- [ ] Create a valid persona. Switch back to your existing one. Chat history of each is isolated.

## Image flow

- [ ] Drag-drop an image into the chat. Preview renders.
- [ ] Send. Reply lands (FakeProvider won't reference the image but a real provider should). Open DevTools → Network → confirm the `/chat` request body carried the image sha.
- [ ] Stage an image, then remove it BEFORE sending. The preview disappears.
- [ ] Stage an image, then switch persona BEFORE sending. *(This is the F-007 scenario — the URL leak. From the user's POV the staged image just disappears on switch, which is the expected behaviour. Note any visible artifacts.)*

## Always-on-top + window state

- [ ] Toggle always-on-top from the Connection panel. Window stays on top of other apps.
- [ ] Toggle off. Window behaves normally.
- [ ] Quit + reopen with always-on-top previously enabled — setting persists AND the window comes back on-top (not just persisted in config).

## Operator surfaces (CLI smoke from Terminal)

- [ ] `nell status --persona <yours>` reports persona path, provider, memory count, bridge state.
- [ ] `nell soul review --persona <yours>` runs without traceback. If there are pending soul candidates, walks through them.
- [ ] `nell memory list --persona <yours>` returns rows after some chat history exists.

## Recovery surfaces

- [ ] In Activity Monitor: find the `nell` bridge Python process. Force-quit it (kill -9).
- [ ] Renderer detects loss + shows a recovery banner within ~10s.
- [ ] Click "reconnect" (or whatever the banner offers). Bridge restarts. Chat works again.

## Visual + interaction polish

- [ ] Hover every clickable element — has a hover state.
- [ ] Tab-navigate through the connection panel + chat input — focus rings visible at every stop.
- [ ] System Preferences → Accessibility → Display → enable "reduce motion". Open the app. Major animations (entrance, scroll-driven, pulse-loop) are subdued or skipped. Static layout still works.
- [ ] Verify Cmd-, opens preferences (or whatever the app's expected keyboard shortcut is — leave finding if a documented shortcut doesn't work).

## Anything else surprising?

Use this section for ad-hoc findings. Each becomes an F-2NN entry in the master audit doc:

```
F-2NN — <title>
Severity: <P0/P1/P2/P3>
Axis: <code|feature|hygiene|polish|quality>
Location: <where in the UI>
Observation: <what happened>
Why it matters: <user impact>
Suggested fix: <your guess; ok if vague — the audit doc author refines>
```

## When done

Either:
- (a) Paste your findings into this file under "Anything else surprising?" and tell me — I'll integrate them into the master audit doc with proper F-2NN IDs, OR
- (b) Just tell me what broke and I'll write them up.

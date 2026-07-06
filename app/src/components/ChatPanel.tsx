import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { acceptVoiceEdit, fetchActiveSession, fetchChatHistory, getBridgeCredentials, newSession, rejectVoiceEdit, snapshotSession, uploadImage } from "../bridge";
import { subscribeToBridgeEvents, type EventStream } from "../bridgeEvents";
import { InitiateBanner, type InitiateMessage } from "./InitiateBanner";
import { VoiceEditPanel, type VoiceEditProposal } from "./VoiceEditPanel";
import { streamChat } from "../streamChat";
import { errString } from "../lib/errString";
import { friendlyChatError } from "../lib/friendlyChatError";
import { resolveFrameUrl } from "../expressions";

// Tauri invocation is best-effort here: browser dev or older Tauri builds
// may not have the notification command registered yet.
async function tryInvoke(cmd: string, args: Record<string, unknown>): Promise<void> {
  try {
    await invoke(cmd, args);
  } catch {
    // Tauri not present (browser dev) or command unregistered (Task 27
    // hasn't shipped yet) — silent no-op is the right behaviour.
  }
}

// v0.0.15-alpha.2 Phase 4 — defensive empty-error fallback.
//
// A Linux user reported "(Nell couldn't answer — see the error below.)"
// rendering with no error text underneath. streamChat's onError can
// fire with "" or a stringified Error with no message, and the raw
// setError("") leaves the error banner blank. setErrorSafe (below)
// substitutes a fixed copy pointing the user at the bridge restart
// button in Connection.
const EMPTY_ERROR_FALLBACK =
  "The bridge couldn't respond. The supervisor may have stalled — try the bridge restart button in Connection, or close and reopen the app.";

interface Message {
  id: number;
  from: "hana" | "nell";
  text: string;
  time: string;
  /** Words still arriving — the bubble is rendering live during stream. */
  streaming?: boolean;
  /** Thumbnail data URL for any image attached to this message. */
  imageThumb?: string;
  /** Initiate reach-out bubble — renders a ✶ marker. */
  reachedOut?: boolean;
}

interface StagedImage {
  /** Object URL for the local preview thumbnail. */
  previewUrl: string;
  /** Human-readable file name. */
  fileName: string;
  /** Bridge-returned sha after upload. Empty until upload completes. */
  sha: string;
  /** Upload state — driven by the async POST /upload call. */
  status: "uploading" | "ready" | "error";
  /** Error text on failure. */
  error?: string;
}

const ACCEPTED_IMAGE_TYPES = "image/png,image/jpeg,image/webp,image/gif";
const MAX_BYTES = 20 * 1024 * 1024;

// Curated emoji palette — small enough to inline, broad enough for chat.
const EMOJI_GROUPS: { label: string; chars: string[] }[] = [
  {
    label: "feelings",
    chars: [
      "🙂", "😊", "😌", "😍", "🥰", "😘", "😉", "😏",
      "😢", "🥺", "😭", "😞", "😔", "😩", "😤", "😡",
      "😴", "🤔", "🙃", "😶", "😐", "😬", "🥲", "😅",
    ],
  },
  {
    label: "love",
    chars: [
      "❤️", "🤍", "🖤", "💔", "💞", "💗", "💕", "💋",
    ],
  },
  {
    label: "hands",
    chars: [
      "👋", "🤝", "🤗", "🙏", "👍", "✌️", "🤞", "👀",
    ],
  },
  {
    label: "small things",
    chars: [
      "✨", "🌙", "☕", "🌧️", "🔥", "💭", "📖", "🕯️",
    ],
  },
];

const formatTime = () =>
  new Date().toLocaleTimeString("en", { hour: "numeric", minute: "2-digit" });

// Persona names match [A-Za-z0-9_-]{1,40}; render them human:
// underscores → spaces, then capitalise. ``my_companion`` -> "My companion".
const capitalize = (s: string) => {
  if (!s) return s;
  const cleaned = s.replace(/_/g, " ");
  return cleaned[0].toUpperCase() + cleaned.slice(1);
};

interface Props {
  /** Persona name — every bridge call is scoped to this so the UI
   *  cannot accidentally talk to a different persona's daemon. */
  persona: string;
  /** Notifier called when chat is in-flight; drives NellAvatar's
   * speaking animation. true = streaming reply, false = idle. */
  onSpeakingChange?: (speaking: boolean) => void;
  /** True when orphan session buffers from a previous shutdown are
   *  still being re-ingested. Drives the brief "reconnecting your
   *  previous chat" banner so a user opening the app after a hard
   *  quit sees that their last conversation is being saved, not
   *  forgotten. Source: PersonaState.recovering from /persona/state. */
  recovering?: boolean;
  /** True iff felt-time state was rebuilt from JSONL logs after a
   *  stale or missing felt_time_state.json. Cleared on the next
   *  supervisor tick — banner is naturally short-lived.
   *  Source: PersonaState.felt_time_recovered from /persona/state. */
  feltTimeRecovered?: boolean;
  /** Optional event-stream override — used in tests to inject a mock
   *  bridge /events subscription without opening a real WebSocket. When
   *  omitted, ChatPanel opens its own /events socket scoped to the
   *  persona. The stream is responsible for ``initiate_delivered``
   *  events that drive InitiateBanner rendering. */
  eventStream?: EventStream;
  /** Bridge health/liveness signal — same source `GlobalStatusDot` reads
   *  (`PersonaState.mode`). Drives the header "● live" pill's color and
   *  text so the chat card doubles as a health readout. Defaults to
   *  "live" so existing callers/tests that don't pass it see the
   *  original green-pill behavior. */
  mode?: "live" | "bridge_down" | "provider_down" | "offline";
}

/**
 * ChatPanel — WebSocket streaming via /stream/{sid}.
 *
 * Each user turn opens a WS to the bridge, receives tool_call events
 * + reply_chunk frames (word-by-word), and commits the assembled
 * message on the `done` frame. Avatar speaking animation runs from
 * stream-open until done, so the mouth animates while text appears.
 */
export function ChatPanel({ persona, onSpeakingChange, recovering = false, feltTimeRecovered = false, eventStream, mode = "live" }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Phase 4 (v0.0.15-alpha.2): map empty/whitespace-only strings to a
  // fixed user-facing copy so the error banner never renders blank
  // beneath the "see the error below" failure marker.
  const setErrorSafe = (msg: string | null) => {
    if (msg === null) {
      setError(null);
      return;
    }
    const trimmed = msg.trim();
    setError(trimmed.length > 0 ? trimmed : EMPTY_ERROR_FALLBACK);
  };
  const [memorySaveWarning, setMemorySaveWarning] = useState<string | null>(null);
  const [stagedImage, setStagedImage] = useState<StagedImage | null>(null);
  const [emojiOpen, setEmojiOpen] = useState(false);
  const [activeBanners, setActiveBanners] = useState<InitiateMessage[]>([]);
  const [activeVoiceEdits, setActiveVoiceEdits] = useState<VoiceEditProposal[]>([]);
  const sessionRef = useRef<string | null>(null);
  const cancelRef = useRef<(() => void) | null>(null);
  // Monotonic bubble-id counter. Seeded from Date.now() so it starts well
  // above the small turn-number ids used for history bubbles (id: h.turn).
  const bubbleIdRef = useRef<number>(Date.now());
  const nextBubbleId = () => (bubbleIdRef.current += 1);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // All bubble-resident object URLs — revoked on unmount so long
  // image-heavy sessions don't leak browser memory for preview blobs.
  const trackedUrlsRef = useRef<Set<string>>(new Set());
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Pipe streaming state up so the avatar speaks during chunks
  useEffect(() => {
    onSpeakingChange?.(streaming);
  }, [streaming, onSpeakingChange]);

  // Snapshot the active session on unmount/app unload — non-destructive:
  // the replay buffer is preserved on disc so recovery can re-extract if
  // the process is interrupted. Sessions are created lazily on first send —
  // opening the app should not create empty bridge sessions that then linger
  // after quit/reload.
  useEffect(() => {
    const snapshotCurrentSession = (keepalive = false) => {
      const sid = sessionRef.current;
      sessionRef.current = null;
      if (sid) {
        void snapshotSession(persona, sid, { keepalive })
          .then(() => {
            if (!keepalive && mountedRef.current) setMemorySaveWarning(null);
          })
          .catch((e) => {
            // Page unload remains best-effort. In-app persona changes keep the
            // component mounted, so surface the retryable failure instead of
            // silently discarding the user's expectation that memory was saved.
            if (!keepalive && mountedRef.current) {
              setMemorySaveWarning(
                `Memory snapshot pending for the previous chat: ${errString(e)}`,
              );
            }
          });
      }
    };
    const onBeforeUnload = () => snapshotCurrentSession(true);
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", onBeforeUnload);
      cancelRef.current?.();
      snapshotCurrentSession();
      // Free any object URLs we still hold for bubble thumbnails so
      // the renderer doesn't keep blob bytes pinned across unmount.
      for (const url of trackedUrlsRef.current) URL.revokeObjectURL(url);
      trackedUrlsRef.current.clear();
    };
  }, [persona]);

  // ── Mount-time history hydration (Phase 3B, v0.0.15-alpha.2) ──────────
  // When the user reopens the app, the bridge's session JSONL is still on
  // disc and /sessions/active will hand back the previous sid. Replay the
  // turn log so the conversation doesn't appear to have evaporated. Lazy
  // session creation (deferred until first send) stays the default when
  // no active session exists — we don't eagerly mint an empty one.
  //
  // Race note: hydration only writes to setMessages BEFORE any send() can
  // possibly have appended a new turn — sessionRef.current is null until
  // the user types and clicks send, and we set it here before any visible
  // affordance reveals the hydrated state. The cancelled guard ensures a
  // late resolution after persona switch / unmount doesn't clobber the
  // new panel's empty state.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      let sid: string | null;
      try {
        sid = await fetchActiveSession(persona);
      } catch {
        // Bridge can't tell us — treat as fresh session, no hydration.
        return;
      }
      if (cancelled || !sid) return;
      sessionRef.current = sid;
      try {
        const { messages: history } = await fetchChatHistory(persona, sid, 200);
        if (cancelled) return;
        setMessages(
          history.map((h) => ({
            id: h.turn,
            from: h.role === "assistant" ? "nell" : "hana",
            text: h.content,
            time: h.ts
              ? new Date(h.ts).toLocaleTimeString("en", {
                  hour: "numeric",
                  minute: "2-digit",
                })
              : "",
          })),
        );
      } catch {
        // Non-fatal: empty messages array is the legitimate fresh-session
        // UX. A bridge that briefly 5xxs on /chat/history just renders an
        // empty panel; the next send still works.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [persona]);

  // Auto-scroll on new message or chunk
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, streaming]);

  // ── Bridge /events subscription — initiate_delivered banners ───────────
  // When an injected stream is provided (tests), use it as-is. Otherwise
  // open a real /events WebSocket scoped to this persona. The subscription
  // adds an active banner per initiate_delivered event; the banner itself
  // calls onMounted after 2s on-screen (read receipt), and onDismiss /
  // onReply on user action.
  useEffect(() => {
    let owned: { close: () => void } | null = null;
    let stream: EventStream;
    if (eventStream) {
      stream = eventStream;
    } else {
      const opened = subscribeToBridgeEvents(persona);
      owned = opened;
      stream = opened;
    }
    const unsubscribe = stream.subscribe((event) => {
      if (event.type !== "initiate_delivered") return;
      const auditId = event.audit_id;
      const body = event.body;
      if (typeof auditId !== "string" || typeof body !== "string") return;

      // Branch: voice_edit_proposal → inline VoiceEditPanel; all others → InitiateBanner.
      if (event.kind === "voice_edit_proposal") {
        const diff = typeof event.diff === "string" ? event.diff : "";
        const oldText = (diff.match(/^- (.*)$/m)?.[1] ?? "").trim();
        const newText = (diff.match(/^\+ (.*)$/m)?.[1] ?? "").trim();
        setActiveVoiceEdits((prev) =>
          prev.some((v) => v.auditId === auditId)
            ? prev
            : [...prev, { auditId, oldText, newText, rationale: body, evidence: [], voiceTemplate: "" }],
        );
        return;
      }

      const urgency = event.urgency === "notify" ? "notify" : "quiet";
      const state =
        typeof event.state === "string"
          ? (event.state as InitiateMessage["state"])
          : "delivered";
      const timestamp =
        typeof event.timestamp === "string" ? event.timestamp : new Date().toISOString();
      setActiveBanners((prev) =>
        prev.some((b) => b.auditId === auditId)
          ? prev
          : [...prev, { auditId, body, urgency, state, timestamp }],
      );
      // show_initiate_notification is registered (Task 27 done). tryInvoke
      // still provides a silent catch for forward-compat safety.
      if (urgency === "notify") {
        void tryInvoke("show_initiate_notification", { title: capitalize(persona), body });
      }
    });
    return () => {
      unsubscribe();
      owned?.close();
    };
  }, [persona, eventStream]);

  async function postInitiateState(auditId: string, newState: string): Promise<void> {
    // Best-effort: a network failure here just means the audit row stays
    // in its previous state until the next user action. We don't want to
    // surface an error to the user for an ambient receipt.
    try {
      const creds = await getBridgeCredentials(persona);
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (creds.authToken) headers.Authorization = `Bearer ${creds.authToken}`;
      await fetch(`${creds.url}/initiate/state`, {
        method: "POST",
        headers,
        body: JSON.stringify({ audit_id: auditId, new_state: newState }),
      });
    } catch {
      // Silent — see comment above.
    }
  }

  const onBannerMounted = (auditId: string) => {
    void postInitiateState(auditId, "read");
  };
  const onBannerDismiss = (auditId: string) => {
    void postInitiateState(auditId, "dismissed");
    setActiveBanners((prev) => prev.filter((b) => b.auditId !== auditId));
  };

  // Paste-from-clipboard: when an image is on the clipboard and the
  // user pastes inside the textarea, stage it instead of dumping the
  // base64 string into the input. Mirrors how Slack / Discord handle
  // Cmd-V'd screenshots.
  useEffect(() => {
    function onPaste(e: ClipboardEvent) {
      if (!e.clipboardData || stagedImage) return;
      for (const item of e.clipboardData.items) {
        if (item.kind === "file" && item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) {
            e.preventDefault();
            void handleFile(file);
            return;
          }
        }
      }
    }
    const ta = textareaRef.current;
    ta?.addEventListener("paste", onPaste);
    return () => ta?.removeEventListener("paste", onPaste);
    // handleFile is closed-over and stable enough; stagedImage gates
    // the re-stage so we re-bind when it clears.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stagedImage]);

  async function handleFile(file: File) {
    if (!file.type.startsWith("image/")) {
      setErrorSafe(`Unsupported file type: ${file.type}.`);
      return;
    }
    if (file.size > MAX_BYTES) {
      setErrorSafe(`Image too large (${(file.size / 1024 / 1024).toFixed(1)} MB; max 20 MB).`);
      return;
    }
    setErrorSafe(null);
    const previewUrl = URL.createObjectURL(file);
    // F-007 (v0.0.7 audit, polish): track the preview URL the moment it's
    // created — not when the message is sent. If the user stages an image
    // and then unmounts (persona switch, app quit, route change) without
    // sending, the URL is held only in React state. The unmount sweep at
    // lines 159-160 iterates trackedUrlsRef, so adding here ensures it's
    // freed. Set semantics make subsequent track-on-send (line ~279)
    // idempotent, and clearStagedImage's explicit revoke is unaffected
    // since the URL stays in the set until unmount.
    trackedUrlsRef.current.add(previewUrl);
    setStagedImage({
      previewUrl,
      fileName: file.name,
      sha: "",
      status: "uploading",
    });
    try {
      const result = await uploadImage(persona, file);
      setStagedImage((prev) =>
        prev && prev.previewUrl === previewUrl
          ? { ...prev, sha: result.sha, status: "ready" }
          : prev,
      );
    } catch (e) {
      setStagedImage((prev) =>
        prev && prev.previewUrl === previewUrl
          ? { ...prev, status: "error", error: errString(e) }
          : prev,
      );
    }
  }

  function clearStagedImage() {
    if (stagedImage?.previewUrl) URL.revokeObjectURL(stagedImage.previewUrl);
    setStagedImage(null);
  }

  function insertEmoji(emoji: string) {
    const ta = textareaRef.current;
    if (!ta) {
      setInput((v) => v + emoji);
      return;
    }
    const start = ta.selectionStart ?? input.length;
    const end = ta.selectionEnd ?? input.length;
    const next = input.slice(0, start) + emoji + input.slice(end);
    setInput(next);
    // Restore caret after the inserted emoji on next paint.
    requestAnimationFrame(() => {
      ta.focus();
      const pos = start + emoji.length;
      ta.setSelectionRange(pos, pos);
    });
  }

  // ── Shared stream core ────────────────────────────────────────────────────
  // Both the main composer and the card-reply path funnel through here.
  // opts.prepend: extra Message[] to insert before the hana user bubble
  //   (used by onCardSendReply to prepend the reach-out body bubble).
  // opts.replyToAuditId: threaded to streamChat so the server records
  //   replied_explicit atomically with the chat turn (Bundle A #4).
  // opts.imageShas: forwarded to streamChat for image turns.
  // opts.text: the outbound message text.
  // opts.imageThumb: thumbnail URL for the user bubble (image sends).
  async function streamTurn(opts: {
    text: string;
    replyToAuditId?: string;
    imageShas?: string[];
    imageThumb?: string;
    prepend?: Message[];
  }): Promise<void> {
    const { text: outboundText, replyToAuditId, imageShas, imageThumb, prepend = [] } = opts;

    // Session resolve: reattach or create.
    let sessionId = sessionRef.current;
    if (!sessionId) {
      // Phase B sticky-session reattach (F-201): if the bridge has a
      // recent session for this persona (younger than the finalise
      // threshold), pick up where we left off instead of creating a
      // fresh one. A transient /sessions/active failure (network flake,
      // older bridge build without the endpoint) shouldn't block the
      // send — we just fall through to newSession.
      try {
        sessionId = await fetchActiveSession(persona);
      } catch {
        sessionId = null;
      }
      if (!sessionId) {
        try {
          sessionId = await newSession(persona);
        } catch (e) {
          setErrorSafe(`Bridge unreachable: ${errString(e)}`);
          return;
        }
      }
      sessionRef.current = sessionId;
    }

    const userMsg: Message = {
      id: nextBubbleId(),
      from: "hana",
      text: outboundText,
      time: formatTime(),
      imageThumb,
    };
    // The bubble holds onto the previewUrl until unmount — track it so
    // the unmount cleanup revokes it. Don't revoke here.
    if (userMsg.imageThumb) trackedUrlsRef.current.add(userMsg.imageThumb);

    const replyId = nextBubbleId();
    const replyStub: Message = {
      id: replyId,
      from: "nell",
      text: "",
      time: formatTime(),
      streaming: true,
    };

    setMessages((m) => [...m, ...prepend, userMsg, replyStub]);
    setStreaming(true);
    setErrorSafe(null);
    setMemorySaveWarning(null);

    // Wraps the streamChat call so we can rerun it with a fresh
    // session on ``session_not_found`` (the bridge dropped the
    // session — usually after KeepAlive restart, idle shutdown, or
    // a launchctl kickstart). One retry is enough; a second
    // failure is real.
    const runStream = async (sid: string, isRetry: boolean): Promise<void> => {
      cancelRef.current = await streamChat(
        persona,
        sid,
        outboundText,
        {
          onChunk: (chunkText) => {
            setMessages((m) =>
              m.map((msg) =>
                msg.id === replyId ? { ...msg, text: msg.text + chunkText } : msg,
              ),
            );
          },
          onDone: () => {
            setMessages((m) =>
              m.map((msg) =>
                msg.id === replyId
                  ? { ...msg, streaming: false, time: formatTime() }
                  : msg,
              ),
            );
            setStreaming(false);
            cancelRef.current = null;
          },
          onError: (msg) => {
            // Self-healing path: the bridge forgot our session
            // (KeepAlive restart, idle shutdown, kickstart), so the
            // sessionRef cached here is stale. Clear it, ask the
            // bridge for a fresh session, and replay this turn.
            // Cap at one retry — a second session_not_found is real.
            const isSessionGone =
              !isRetry && /session_not_found/i.test(msg);
            if (isSessionGone) {
              cancelRef.current = null;
              sessionRef.current = null;
              void (async () => {
                try {
                  const fresh = await newSession(persona);
                  sessionRef.current = fresh;
                  await runStream(fresh, /* isRetry */ true);
                } catch (e) {
                  setErrorSafe(`Bridge unreachable: ${errString(e)}`);
                  setStreaming(false);
                  setMessages((m) =>
                    m.map((b) =>
                      b.id === replyId
                        ? {
                            ...b,
                            text: `(${capitalize(persona)} couldn't answer — see the error below.)`,
                            streaming: false,
                            time: formatTime(),
                          }
                        : b,
                    ),
                  );
                }
              })();
              return;
            }
            setErrorSafe(msg);
            setStreaming(false);
            cancelRef.current = null;
            // Audit 2026-05-07 P2-10: replace the empty streaming
            // stub with a visible failure marker. Previously onError
            // only set the error string, leaving an empty persona
            // bubble in the transcript that didn't match what was
            // actually said. The bubble now shows a clear failure
            // note so the transcript matches reality.
            setMessages((m) =>
              m.map((b) =>
                b.id === replyId
                  ? {
                      ...b,
                      text: `(${capitalize(persona)} couldn't answer — see the error below.)`,
                      streaming: false,
                      time: formatTime(),
                    }
                  : b,
              ),
            );
          },
        },
        (() => {
          const streamOpts: { imageShas?: string[]; replyToAuditId?: string } = {};
          if (imageShas?.length) streamOpts.imageShas = imageShas;
          if (replyToAuditId) streamOpts.replyToAuditId = replyToAuditId;
          return Object.keys(streamOpts).length > 0 ? streamOpts : undefined;
        })(),
      );
    };

    try {
      await runStream(sessionId, /* isRetry */ false);
    } catch (e) {
      setErrorSafe(errString(e));
      setStreaming(false);
      // Same defense for synchronous failures before streamChat returns.
      setMessages((m) =>
        m.map((b) =>
          b.id === replyId
            ? {
                ...b,
                text: `(${capitalize(persona)} couldn't answer — see the error below.)`,
                streaming: false,
                time: formatTime(),
              }
            : b,
        ),
      );
    }
  }

  // Main composer submit — validates input, clears UI state, calls streamTurn.
  async function send() {
    const text = input.trim();
    const readySha =
      stagedImage?.status === "ready" && stagedImage.sha ? stagedImage.sha : null;
    if ((!text && !readySha) || streaming) return;
    if (stagedImage && stagedImage.status === "uploading") return;
    const outboundText = text || "Please look at this image.";
    const imageThumb = readySha ? stagedImage?.previewUrl : undefined;

    setStagedImage(null);
    setEmojiOpen(false);
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    await streamTurn({
      text: outboundText,
      imageShas: readySha ? [readySha] : undefined,
      imageThumb,
    });
  }

  // Card-reply handler — called by InitiateBanner's onSendReply prop.
  // Prepends the reach-out body as a ✶-marked nell bubble, then sends
  // the reply through streamTurn with replyToAuditId threaded.
  const onCardSendReply = (auditId: string, text: string) => {
    if (streaming || !text.trim()) return;   // don't double-stream; don't send empty
    const banner = activeBanners.find((b) => b.auditId === auditId);
    setActiveBanners((prev) => prev.filter((b) => b.auditId !== auditId));
    const prepend: Message[] = banner
      ? [{ id: nextBubbleId(), from: "nell", text: banner.body, time: formatTime(), reachedOut: true }]
      : [];
    // fire-and-forget: streamTurn owns its own error surfacing (failure bubble + setErrorSafe)
    void streamTurn({ text, replyToAuditId: auditId, prepend });
  };

  function stopStreaming() {
    cancelRef.current?.();
    cancelRef.current = null;
    setStreaming(false);
    setMessages((m) =>
      m.map((msg) =>
        msg.streaming
          ? { ...msg, text: msg.text || "(stopped)", streaming: false, time: formatTime() }
          : msg,
      ),
    );
  }

  function onKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  const [dragOver, setDragOver] = useState(false);

  function onDragOver(e: React.DragEvent<HTMLDivElement>) {
    if (stagedImage || streaming) return;
    if (Array.from(e.dataTransfer.items).some(
      (i) => i.kind === "file" && i.type.startsWith("image/"),
    )) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
      setDragOver(true);
    }
  }

  function onDragLeave(_e: React.DragEvent<HTMLDivElement>) {
    setDragOver(false);
  }

  function onDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    if (stagedImage || streaming) return;
    const file = Array.from(e.dataTransfer.files).find((f) =>
      f.type.startsWith("image/"),
    );
    if (file) void handleFile(file);
  }

  const hasReadyImage = stagedImage?.status === "ready" && !!stagedImage.sha;
  const sendDisabled =
    !streaming && ((!input.trim() && !hasReadyImage) || stagedImage?.status === "uploading");

  return (
    <div
      className="chat-panel glass"
      style={{
        position: "relative",
        width: "398px",
        height: "566px",
        borderRadius: 26,
        boxShadow: "var(--shadow)",
        background: "rgba(36,26,29,0.50)",
        backdropFilter: "blur(36px) saturate(1.5)",
        WebkitBackdropFilter: "blur(36px) saturate(1.5)",
        outline: dragOver ? "2px dashed var(--accent)" : "none",
        outlineOffset: 4,
        transition: "outline 0.15s ease",
        overflow: "hidden",
      }}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <ChatHeader persona={persona} mode={mode} />
      <div
        ref={scrollRef}
        data-testid="chat-messages"
        style={{
          flex: "1 1 auto",
          minHeight: "0",
          overflowY: "auto",
          padding: "16px 16px 10px",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {messages.map((m, i) => (
          <Bubble
            key={m.id}
            msg={m}
            prevFrom={i > 0 ? messages[i - 1].from : null}
            nextFrom={i < messages.length - 1 ? messages[i + 1].from : null}
          />
        ))}
        {error && (
          <div style={{ fontSize: 11, color: "var(--crimson)", padding: "6px 4px" }}>
            {friendlyChatError(error)}
          </div>
        )}
        {memorySaveWarning && (
          <div style={{ fontSize: 11, color: "var(--lacquer)", padding: "6px 4px" }}>
            {memorySaveWarning}
          </div>
        )}
        {recovering && (
          <div
            data-testid="recovery-banner"
            role="status"
            style={{
              fontSize: 11,
              color: "var(--text-mute)",
              padding: "6px 4px",
              fontStyle: "italic",
            }}
          >
            Reconnecting your previous chat — give it a moment.
          </div>
        )}
        {feltTimeRecovered && (
          <div
            data-testid="felt-time-recovery-banner"
            role="status"
            style={{
              fontSize: 11,
              color: "var(--text-mute)",
              padding: "6px 4px",
              fontStyle: "italic",
            }}
          >
            Felt time recovered from logs.
          </div>
        )}
      </div>
      {activeBanners.length > 0 && (
        <div
          data-testid="initiate-banner-list"
          style={{ display: "flex", flexDirection: "column", gap: 6, padding: "0 16px 8px" }}
        >
          {activeBanners.map((b) => (
            <InitiateBanner
              key={b.auditId}
              message={b}
              companionName={capitalize(persona)}
              onSendReply={onCardSendReply}
              onDismiss={onBannerDismiss}
              onMounted={onBannerMounted}
              isStreaming={streaming}
            />
          ))}
        </div>
      )}
      {activeVoiceEdits.map((p) => (
        <VoiceEditPanel
          key={p.auditId}
          proposal={p}
          persona={persona}
          onAccept={(id, withEdits) => {
            void acceptVoiceEdit(persona, id, withEdits)
              .then(() => setActiveVoiceEdits((prev) => prev.filter((v) => v.auditId !== id)))
              .catch((e) => console.error("voice-edit accept failed", e));
          }}
          onReject={(id) => {
            void rejectVoiceEdit(persona, id)
              .then(() => setActiveVoiceEdits((prev) => prev.filter((v) => v.auditId !== id)))
              .catch((e) => console.error("voice-edit reject failed", e));
          }}
        />
      ))}
      {stagedImage && (
        <div style={{ padding: "0 16px" }}>
          <StagedImageRow staged={stagedImage} onRemove={clearStagedImage} />
        </div>
      )}
      <div
        style={{
          flexShrink: 0,
          display: "flex",
          gap: 9,
          alignItems: "flex-end",
          position: "relative",
          padding: "12px 14px 14px",
          borderTop: "1px solid var(--hairline)",
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_IMAGE_TYPES}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
            e.target.value = ""; // allow re-selecting same file later
          }}
          style={{ display: "none" }}
        />
        <IconButton
          aria-label="attach image"
          title="attach image"
          onClick={() => fileInputRef.current?.click()}
          disabled={streaming || !!stagedImage}
        >
          <PaperclipIcon />
        </IconButton>
        <IconButton
          aria-label="insert emoji"
          title="insert emoji"
          onClick={() => setEmojiOpen((v) => !v)}
          active={emojiOpen}
          disabled={streaming}
        >
          <EmojiIcon />
        </IconButton>
        {emojiOpen && (
          <EmojiPicker
            onPick={(c) => {
              insertEmoji(c);
              // Keep the picker open so quick multi-emoji bursts don't
              // require re-clicking. ESC / outside-click closes it.
            }}
            onClose={() => setEmojiOpen(false)}
          />
        )}
        <div
          style={{
            flex: 1,
            display: "flex",
            alignItems: "center",
            background: "var(--field)",
            border: "1px solid var(--hairline)",
            borderRadius: 999,
            padding: "10px 17px",
          }}
        >
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
            onInput={(e) => {
              const ta = e.currentTarget;
              ta.style.height = "auto";
              ta.style.height = `${Math.min(ta.scrollHeight, 192)}px`;
            }}
            placeholder={`Write to ${capitalize(persona)}…`}
            className="chat-input"
            style={{
              flex: 1,
              background: "transparent",
              border: "none",
              padding: 0,
              color: "var(--text)",
              fontFamily: "var(--font-ui)",
              fontSize: 13.5,
              resize: "none",
              minHeight: 18,
              maxHeight: 192,
              overflow: "hidden",
              outline: "none",
            }}
          />
        </div>
        <button
          onClick={streaming ? stopStreaming : send}
          disabled={sendDisabled}
          style={{
            flexShrink: 0,
            width: 36,
            height: 36,
            borderRadius: "50%",
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            background: "var(--accent)",
            color: "#ffffff",
            fontSize: 16,
            boxShadow: sendDisabled
              ? "none"
              : "0 6px 16px color-mix(in srgb, var(--accent) 45%, transparent)",
            opacity: sendDisabled ? 0.45 : 1,
            transition: "opacity 0.2s, box-shadow 0.2s",
          }}
          aria-label={streaming ? "stop response" : "send"}
        >
          {streaming ? "×" : "↑"}
        </button>
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────────
// ChatHeader — purely presentational card-top block (DESIGN-SPEC §6).
// The avatar thumb is a static small render, not state-driven — the real
// expressive avatar lives in NellAvatar next to the presence column.
// ──────────────────────────────────────────────────────────────────────────

// Colors: live = DESIGN-SPEC §1 "Success text" (#7fc9a0) / live-dot bg
// formula. bridge_down/offline = DESIGN-SPEC §1 "Error / invalid"
// (--crimson, #e07a6a) — the same token InitiateBanner-adjacent error
// text uses elsewhere in ChatPanel. provider_down = the existing app
// amber (App.tsx GlobalStatusDot "warn" palette, rgb(216,154,88)) — the
// redesign's token table doesn't define a distinct amber, so this reuses
// the app's one existing amber rather than inventing a new hex.
const MODE_PILL: Record<
  NonNullable<Props["mode"]>,
  { label: string; color: string; bg: string }
> = {
  live: { label: "● live", color: "#7fc9a0", bg: "rgba(79,168,118,0.14)" },
  provider_down: { label: "● provider down", color: "rgb(216,154,88)", bg: "rgba(216,154,88,0.14)" },
  bridge_down: { label: "● bridge down", color: "var(--crimson)", bg: "rgba(224,122,106,0.14)" },
  offline: { label: "● offline", color: "var(--crimson)", bg: "rgba(224,122,106,0.14)" },
};

function ChatHeader({ persona, mode }: { persona: string; mode: NonNullable<Props["mode"]> }) {
  const pill = MODE_PILL[mode] ?? MODE_PILL.live;
  return (
    <div
      style={{
        flexShrink: 0,
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "12px 18px",
        borderBottom: "1px solid var(--hairline)",
        background: "rgba(255,255,255,0.03)",
      }}
    >
      <div
        style={{
          width: 34,
          height: 34,
          borderRadius: "50%",
          background: "#241a1c",
          flexShrink: 0,
          overflow: "hidden",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <img
          src={resolveFrameUrl("smile", "base")}
          alt=""
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
        />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text)" }}>
          {capitalize(persona)}
        </div>
        <div style={{ fontSize: 10.5, color: "var(--text-mute)" }}>Kindled · local-only</div>
      </div>
      <div
        style={{
          fontSize: 11,
          fontWeight: 600,
          color: pill.color,
          background: pill.bg,
          padding: "4px 10px",
          borderRadius: 999,
          flexShrink: 0,
          whiteSpace: "nowrap",
        }}
      >
        {pill.label}
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────────
// Sub-components — paperclip, emoji picker, staged-image row
// ──────────────────────────────────────────────────────────────────────────

function IconButton({
  children,
  active,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { active?: boolean }) {
  return (
    <button
      {...rest}
      style={{
        background: active ? "var(--accent-dim)" : "transparent",
        border: "1px solid var(--border-dk)",
        borderRadius: 6,
        width: 32,
        height: 32,
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        color: "var(--text-mid)",
        cursor: rest.disabled ? "default" : "pointer",
        opacity: rest.disabled ? 0.35 : 1,
        transition: "background 0.15s, opacity 0.15s",
        flexShrink: 0,
      }}
    >
      {children}
    </button>
  );
}

function PaperclipIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
    </svg>
  );
}

function EmojiIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <path d="M8 14s1.5 2 4 2 4-2 4-2" />
      <line x1="9" y1="9" x2="9.01" y2="9" />
      <line x1="15" y1="9" x2="15.01" y2="9" />
    </svg>
  );
}

function EmojiPicker({
  onPick,
  onClose,
}: {
  onPick: (c: string) => void;
  onClose: () => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    function onDoc(e: MouseEvent) {
      if (!containerRef.current?.contains(e.target as Node)) onClose();
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    // Defer attach by one tick so the click that opened us doesn't
    // immediately match outside-click and close again.
    const t = setTimeout(() => {
      document.addEventListener("mousedown", onDoc);
      document.addEventListener("keydown", onKey);
    }, 0);
    return () => {
      clearTimeout(t);
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  return (
    <div
      ref={containerRef}
      style={{
        position: "absolute",
        bottom: 40,
        left: 0,
        background: "rgba(26, 18, 18, 0.97)",
        border: "1px solid var(--border-dk)",
        borderRadius: 8,
        padding: 8,
        width: 232,
        maxHeight: 220,
        overflowY: "auto",
        boxShadow: "0 4px 14px rgba(0,0,0,0.3)",
        zIndex: 10,
        animation: "msg-in 0.18s ease",
      }}
    >
      {EMOJI_GROUPS.map((group) => (
        <div key={group.label} style={{ marginBottom: 8 }}>
          <div
            style={{
              fontSize: 9,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "var(--mauve)",
              fontFamily: "var(--font-disp)",
              marginBottom: 4,
            }}
          >
            {group.label}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(8, 1fr)", gap: 2 }}>
            {group.chars.map((c, i) => (
              <button
                key={`${group.label}-${i}`}
                onClick={() => onPick(c)}
                title={c}
                style={{
                  background: "transparent",
                  border: "none",
                  fontSize: 16,
                  padding: "3px 0",
                  cursor: "pointer",
                  borderRadius: 4,
                  transition: "background 0.1s",
                }}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.background = "var(--accent-dim)")
                }
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                {c}
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function StagedImageRow({
  staged,
  onRemove,
}: {
  staged: StagedImage;
  onRemove: () => void;
}) {
  return (
    <div
      style={{
        marginTop: 8,
        padding: "6px 8px",
        background: "rgba(26, 18, 18, 0.45)",
        border: "1px solid var(--border-dk)",
        borderRadius: 6,
        display: "flex",
        gap: 8,
        alignItems: "center",
        animation: "msg-in 0.22s ease",
      }}
    >
      <img
        src={staged.previewUrl}
        alt={staged.fileName}
        style={{ width: 36, height: 36, objectFit: "cover", borderRadius: 4 }}
      />
      <div style={{ flex: 1, fontSize: 11, color: "var(--linen)", overflow: "hidden" }}>
        <div
          style={{
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
            opacity: 0.85,
          }}
        >
          {staged.fileName}
        </div>
        <div style={{ fontSize: 9, color: "var(--mauve)", marginTop: 2, letterSpacing: "0.04em" }}>
          {staged.status === "uploading" && "uploading…"}
          {staged.status === "ready" && `staged · ${staged.sha.slice(0, 8)}`}
          {staged.status === "error" && `failed: ${staged.error ?? ""}`}
        </div>
      </div>
      <button
        onClick={onRemove}
        aria-label="remove image"
        title="remove"
        style={{
          background: "transparent",
          border: "none",
          color: "var(--mauve)",
          cursor: "pointer",
          fontSize: 14,
          padding: "0 4px",
        }}
      >
        ×
      </button>
    </div>
  );
}

/**
 * Bubble — DESIGN-SPEC §7 iMessage grammar.
 *
 * prevFrom/nextFrom are the neighboring messages' `from` (or null at the
 * ends of the list) — used purely to compute grouping margins and to
 * decide whether this bubble is the LAST of a consecutive same-role run
 * (only the last bubble of a group shows its timestamp).
 */
function Bubble({
  msg,
  prevFrom,
  nextFrom,
}: {
  msg: Message;
  prevFrom: Message["from"] | null;
  nextFrom: Message["from"] | null;
}) {
  const isHana = msg.from === "hana";
  const sameAsPrev = prevFrom === msg.from;
  const isLastOfGroup = nextFrom !== msg.from;
  const marginTop = prevFrom === null ? 0 : sameAsPrev ? 3 : 14;
  const isTyping = !!msg.streaming && !msg.text;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: isHana ? "flex-end" : "flex-start",
        marginTop,
        animation: "msg-in 0.22s ease",
      }}
    >
      <div
        style={{
          maxWidth: "76%",
          padding: isTyping ? "12px 15px" : "9px 14px",
          background: isHana
            ? "linear-gradient(180deg, color-mix(in srgb, var(--accent) 88%, #ffffff), var(--accent))"
            : "var(--bubble-in)",
          color: isHana ? "#ffffff" : "var(--text)",
          border: isHana ? "none" : "1px solid rgba(255,255,255,0.08)",
          borderRadius: isHana ? "20px 20px 6px 20px" : "20px 20px 20px 6px",
          fontSize: 13.5,
          lineHeight: 1.4,
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}
      >
        {msg.imageThumb && (
          <img
            src={msg.imageThumb}
            alt=""
            style={{
              display: "block",
              maxWidth: "100%",
              maxHeight: 160,
              borderRadius: 5,
              marginBottom: msg.text ? 6 : 0,
            }}
          />
        )}
        {msg.reachedOut && (
          <span className="msg-reachedout" title="Nell reached out">✶ </span>
        )}
        {msg.streaming && !msg.text ? <TypingDots /> : renderBubbleText(msg.text)}
      </div>
      {isLastOfGroup && (
        <div
          style={{
            fontSize: 10,
            color: "var(--text-mute)",
            margin: "4px 6px 0",
          }}
        >
          {msg.time}
        </div>
      )}
    </div>
  );
}

/**
 * Inline rendering for chat bubble text:
 *   - Normalises literal ``\n`` / ``\r\n`` escape sequences (some
 *     models emit those as text rather than real newlines) into
 *     actual newlines so ``white-space: pre-wrap`` can show them
 *     as paragraph breaks.
 *   - Renders ``*italic*`` as ``<em>`` (single-asterisk-pair, no
 *     nesting). Pairs spread across line breaks aren't matched —
 *     the regex stops at newlines so an unclosed ``*`` doesn't eat
 *     the rest of the bubble.
 *
 * Deliberately small: full markdown (bold, links, code, lists) is
 * out of scope for chat — this just covers the two formatting
 * shapes the model actually emits in voice prose.
 */
function renderBubbleText(text: string): React.ReactNode {
  if (!text) return text;
  // Normalise escaped newlines (``\\n``, ``\\r\\n``) to real ones.
  // Some Claude outputs interleave escape sequences with prose when
  // the model is reasoning about its own format — normalise before
  // splitting so the italic pass sees clean line content.
  const normalized = text.replace(/\\r\\n/g, "\n").replace(/\\n/g, "\n");

  const parts: React.ReactNode[] = [];
  const re = /\*([^*\n]+)\*/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let key = 0;
  while ((match = re.exec(normalized)) !== null) {
    if (match.index > last) parts.push(normalized.slice(last, match.index));
    parts.push(<em key={`em-${key++}`}>{match[1]}</em>);
    last = match.index + match[0].length;
  }
  if (last < normalized.length) parts.push(normalized.slice(last));
  return parts.length ? parts : normalized;
}

function TypingDots() {
  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 5,
        height: 14,
      }}
      aria-label="thinking"
    >
      {[0, 0.15, 0.3].map((delay, i) => (
        <div
          key={i}
          style={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            background: "var(--text-mute)",
            animation: `typing-bounce 1.2s ease-in-out infinite`,
            animationDelay: `${delay}s`,
          }}
        />
      ))}
    </div>
  );
}

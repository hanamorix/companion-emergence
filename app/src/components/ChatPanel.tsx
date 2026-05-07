import { useEffect, useRef, useState } from "react";
import { closeSession, newSession, uploadImage } from "../bridge";
import { streamChat } from "../streamChat";

interface Message {
  id: number;
  from: "hana" | "nell";
  text: string;
  time: string;
  /** Words still arriving — the bubble is rendering live during stream. */
  streaming?: boolean;
  /** Thumbnail data URL for any image attached to this message. */
  imageThumb?: string;
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

interface Props {
  /** Persona name — every bridge call is scoped to this so the UI
   *  cannot accidentally talk to a different persona's daemon. */
  persona: string;
  /** Notifier called when chat is in-flight; drives NellAvatar's
   * speaking animation. true = streaming reply, false = idle. */
  onSpeakingChange?: (speaking: boolean) => void;
}

/**
 * ChatPanel — WebSocket streaming via /stream/{sid}.
 *
 * Each user turn opens a WS to the bridge, receives tool_call events
 * + reply_chunk frames (word-by-word), and commits the assembled
 * message on the `done` frame. Avatar speaking animation runs from
 * stream-open until done, so the mouth animates while text appears.
 */
export function ChatPanel({ persona, onSpeakingChange }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stagedImage, setStagedImage] = useState<StagedImage | null>(null);
  const [emojiOpen, setEmojiOpen] = useState(false);
  const sessionRef = useRef<string | null>(null);
  const cancelRef = useRef<(() => void) | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // All bubble-resident object URLs — revoked on unmount so long
  // image-heavy sessions don't leak browser memory for preview blobs.
  const trackedUrlsRef = useRef<Set<string>>(new Set());

  // Pipe streaming state up so the avatar speaks during chunks
  useEffect(() => {
    onSpeakingChange?.(streaming);
  }, [streaming, onSpeakingChange]);

  // Open session on mount, close it on unmount so the buffer flushes
  // through ingest immediately instead of waiting on the supervisor's
  // stale-close loop. Errors during close surface as a visible status.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const sid = await newSession(persona);
        if (!cancelled) sessionRef.current = sid;
      } catch (e) {
        if (!cancelled) setError(`bridge unreachable: ${(e as Error).message}`);
      }
    })();
    return () => {
      cancelled = true;
      cancelRef.current?.();
      const sid = sessionRef.current;
      if (sid) {
        // Best-effort — fire and forget. The component is unmounting,
        // there's nowhere meaningful to surface a close failure; the
        // supervisor's stale-close still catches it as a fallback.
        void closeSession(persona, sid).catch(() => undefined);
      }
      // Free any object URLs we still hold for bubble thumbnails so
      // the renderer doesn't keep blob bytes pinned across unmount.
      for (const url of trackedUrlsRef.current) URL.revokeObjectURL(url);
      trackedUrlsRef.current.clear();
    };
  }, [persona]);

  // Auto-scroll on new message or chunk
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, streaming]);

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
      setError(`unsupported file type: ${file.type}`);
      return;
    }
    if (file.size > MAX_BYTES) {
      setError(`image too large (${(file.size / 1024 / 1024).toFixed(1)} MB; max 20 MB)`);
      return;
    }
    setError(null);
    const previewUrl = URL.createObjectURL(file);
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
          ? { ...prev, status: "error", error: (e as Error).message }
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

  async function send() {
    const text = input.trim();
    if (!text || streaming || !sessionRef.current) return;
    if (stagedImage && stagedImage.status === "uploading") return;

    const readySha =
      stagedImage?.status === "ready" && stagedImage.sha ? stagedImage.sha : null;
    const userMsg: Message = {
      id: Date.now(),
      from: "hana",
      text,
      time: formatTime(),
      imageThumb: readySha ? stagedImage?.previewUrl : undefined,
    };
    // The bubble holds onto the previewUrl until unmount — track it so
    // the unmount cleanup revokes it. Don't revoke here.
    if (userMsg.imageThumb) trackedUrlsRef.current.add(userMsg.imageThumb);
    setStagedImage(null);
    setEmojiOpen(false);
    const replyId = Date.now() + 1;
    const replyStub: Message = {
      id: replyId,
      from: "nell",
      text: "",
      time: formatTime(),
      streaming: true,
    };

    setMessages((m) => [...m, userMsg, replyStub]);
    setInput("");
    setStreaming(true);
    setError(null);

    try {
      cancelRef.current = await streamChat(
        persona,
        sessionRef.current,
        text,
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
            setError(msg);
            setStreaming(false);
            cancelRef.current = null;
          },
        },
        readySha ? { imageShas: [readySha] } : undefined,
      );
    } catch (e) {
      setError((e as Error).message);
      setStreaming(false);
    }
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

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        width: 290,
        height: 380,
        position: "relative",
        outline: dragOver ? "2px dashed var(--accent)" : "none",
        outlineOffset: 4,
        borderRadius: 8,
        transition: "outline 0.15s ease",
      }}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <div
        ref={scrollRef}
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "0 2px",
          display: "flex",
          flexDirection: "column",
          gap: 8,
        }}
      >
        {messages.map((m) => (
          <Bubble key={m.id} msg={m} />
        ))}
        {streaming && messages.length > 0 && messages[messages.length - 1]?.text === "" && (
          <TypingDots />
        )}
        {error && (
          <div style={{ fontSize: 11, color: "var(--crimson)", padding: "6px 4px" }}>
            {error}
          </div>
        )}
      </div>
      {stagedImage && (
        <StagedImageRow staged={stagedImage} onRemove={clearStagedImage} />
      )}
      <div
        style={{
          paddingTop: 10,
          display: "flex",
          gap: 6,
          alignItems: "flex-end",
          position: "relative",
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
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKey}
          placeholder="write to nell..."
          rows={1}
          style={{
            flex: 1,
            background: "rgba(26, 18, 18, 0.6)",
            border: "1px solid var(--border-dk)",
            borderRadius: 6,
            padding: "8px 10px",
            color: "var(--linen)",
            fontFamily: "var(--font-ui)",
            fontSize: 12,
            resize: "none",
            minHeight: 32,
            maxHeight: 120,
            outline: "none",
          }}
        />
        <button
          onClick={send}
          disabled={!input.trim() || streaming || stagedImage?.status === "uploading"}
          style={{
            background: "var(--accent)",
            color: "var(--linen)",
            padding: "7px 11px",
            borderRadius: 6,
            fontSize: 14,
            opacity:
              !input.trim() || streaming || stagedImage?.status === "uploading" ? 0.4 : 1,
            transition: "opacity 0.2s",
          }}
          aria-label="send"
        >
          ↑
        </button>
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
        color: "var(--linen)",
        cursor: rest.disabled ? "default" : "pointer",
        opacity: rest.disabled ? 0.35 : 0.85,
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

function Bubble({ msg }: { msg: Message }) {
  const isHana = msg.from === "hana";
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: isHana ? "flex-end" : "flex-start",
        animation: "msg-in 0.28s ease",
      }}
    >
      <div
        style={{
          maxWidth: "84%",
          padding: "8px 12px",
          background: isHana ? "var(--bubble-user)" : "var(--bubble-nell)",
          color: isHana ? "var(--linen)" : "var(--text)",
          border: `1px solid ${isHana ? "rgba(130,51,41,0.5)" : "var(--border)"}`,
          borderRadius: 9,
          fontSize: 12,
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
        {msg.text}
      </div>
      <div
        style={{
          fontSize: 10,
          color: "var(--mauve)",
          margin: "2px 6px 0",
          fontFamily: "var(--font-disp)",
          letterSpacing: "0.04em",
        }}
      >
        {msg.time}
      </div>
    </div>
  );
}

function TypingDots() {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 5,
        padding: "8px 0",
        animation: "msg-in 0.28s ease",
      }}
    >
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: "var(--mauve)",
            animation: `typing-bounce 1.2s ease-in-out infinite`,
            animationDelay: `${i * 0.18}s`,
          }}
        />
      ))}
    </div>
  );
}

import { useEffect, useRef, useState } from "react";
import { newSession } from "../bridge";
import { streamChat } from "../streamChat";

interface Message {
  id: number;
  from: "hana" | "nell";
  text: string;
  time: string;
  /** Words still arriving — the bubble is rendering live during stream. */
  streaming?: boolean;
}

const formatTime = () =>
  new Date().toLocaleTimeString("en", { hour: "numeric", minute: "2-digit" });

interface Props {
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
export function ChatPanel({ onSpeakingChange }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionRef = useRef<string | null>(null);
  const cancelRef = useRef<(() => void) | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Pipe streaming state up so the avatar speaks during chunks
  useEffect(() => {
    onSpeakingChange?.(streaming);
  }, [streaming, onSpeakingChange]);

  // Open session on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const sid = await newSession();
        if (!cancelled) sessionRef.current = sid;
      } catch (e) {
        if (!cancelled) setError(`bridge unreachable: ${(e as Error).message}`);
      }
    })();
    return () => {
      cancelled = true;
      cancelRef.current?.();
    };
  }, []);

  // Auto-scroll on new message or chunk
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, streaming]);

  async function send() {
    const text = input.trim();
    if (!text || streaming || !sessionRef.current) return;

    const userMsg: Message = { id: Date.now(), from: "hana", text, time: formatTime() };
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
      cancelRef.current = await streamChat(sessionRef.current, text, {
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
              msg.id === replyId ? { ...msg, streaming: false, time: formatTime() } : msg,
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
      });
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

  return (
    <div style={{ display: "flex", flexDirection: "column", width: 290, height: 380 }}>
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
      <div style={{ paddingTop: 10, display: "flex", gap: 7, alignItems: "flex-end" }}>
        <textarea
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
          disabled={!input.trim() || streaming}
          style={{
            background: "var(--accent)",
            color: "var(--linen)",
            padding: "7px 11px",
            borderRadius: 6,
            fontSize: 14,
            opacity: !input.trim() || streaming ? 0.4 : 1,
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

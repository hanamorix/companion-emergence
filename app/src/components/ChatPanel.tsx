import { useEffect, useRef, useState } from "react";
import { newSession, sendChat } from "../bridge";

interface Message {
  id: number;
  from: "hana" | "nell";
  text: string;
  time: string;
}

const formatTime = () =>
  new Date().toLocaleTimeString("en", { hour: "numeric", minute: "2-digit" });

interface Props {
  /** Notifier called when chat is in-flight; drives NellAvatar's
   * speaking animation. true = waiting for reply, false = idle. */
  onSpeakingChange?: (speaking: boolean) => void;
}

/**
 * ChatPanel — text-only HTTP /chat for Phase 2. Phase 6 will switch
 * to WS /stream for token-by-token streaming.
 *
 * Reports its in-flight state via onSpeakingChange so the avatar can
 * cycle to the speaking frames while a reply is being composed.
 */
export function ChatPanel({ onSpeakingChange }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionRef = useRef<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Pipe typing state up so the avatar can do the talking animation
  useEffect(() => {
    onSpeakingChange?.(typing);
  }, [typing, onSpeakingChange]);

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
    };
  }, []);

  // Auto-scroll on new message
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, typing]);

  async function send() {
    const text = input.trim();
    if (!text || typing || !sessionRef.current) return;
    const userMsg: Message = { id: Date.now(), from: "hana", text, time: formatTime() };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setTyping(true);
    setError(null);
    try {
      const resp = await sendChat(sessionRef.current, text);
      const nellMsg: Message = {
        id: Date.now() + 1,
        from: "nell",
        text: resp.reply,
        time: formatTime(),
      };
      setMessages((m) => [...m, nellMsg]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setTyping(false);
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
        {typing && <TypingDots />}
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
          disabled={!input.trim() || typing}
          style={{
            background: "var(--accent)",
            color: "var(--linen)",
            padding: "7px 11px",
            borderRadius: 6,
            fontSize: 14,
            opacity: !input.trim() || typing ? 0.4 : 1,
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

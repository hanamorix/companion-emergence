import { useEffect, useRef, useState } from "react";

export type InitiateMessage = {
  auditId: string;
  body: string;
  urgency: "notify" | "quiet";
  state:
    | "pending"
    | "delivered"
    | "read"
    | "replied_explicit"
    | "acknowledged_unclear"
    | "unanswered"
    | "dismissed";
  timestamp: string;
};

type Props = {
  message: InitiateMessage;
  companionName: string;
  onSendReply: (auditId: string, text: string) => void;
  onDismiss: (auditId: string) => void;
  onMounted: (auditId: string) => void;
  isStreaming?: boolean;
};

export function InitiateBanner({ message, companionName, onSendReply, onDismiss, onMounted, isStreaming = false }: Props) {
  const firedRef = useRef(false);
  const [draft, setDraft] = useState("");

  useEffect(() => {
    // 2-second on-screen timer that respects document visibility:
    // if the user minimizes / switches tabs, pause; resume when visible.
    // Otherwise a backgrounded app would silently mark messages "read"
    // without Hana ever having looked at them.
    let timer: ReturnType<typeof setTimeout> | null = null;

    function fire() {
      if (!firedRef.current && !document.hidden) {
        firedRef.current = true;
        onMounted(message.auditId);
      }
    }

    function scheduleTimer() {
      if (firedRef.current) return;
      if (typeof document !== "undefined" && document.hidden) return;
      timer = setTimeout(fire, 2000);
    }

    function handleVisibilityChange() {
      if (document.hidden) {
        if (timer !== null) {
          clearTimeout(timer);
          timer = null;
        }
      } else {
        scheduleTimer();
      }
    }

    scheduleTimer();
    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", handleVisibilityChange);
    }
    return () => {
      if (timer !== null) clearTimeout(timer);
      if (typeof document !== "undefined") {
        document.removeEventListener("visibilitychange", handleVisibilityChange);
      }
    };
  }, [message.auditId, onMounted]);

  function send() {
    if (isStreaming) return;
    const t = draft.trim();
    if (!t) return;
    onSendReply(message.auditId, t);
    setDraft("");
  }

  return (
    <div className="initiate-banner" role="region" aria-label={`${companionName} reached out`}>
      <div className="initiate-banner__header">✶ {companionName} reached out</div>
      <div className="initiate-banner__body">{message.body}</div>
      <div className="initiate-banner__reply">
        <textarea
          aria-label={`Reply to ${companionName}`}
          placeholder="reply…"
          rows={1}
          value={draft}
          disabled={isStreaming}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        <button type="button" onClick={send} disabled={isStreaming} aria-label="Send reply">
          ↵
        </button>
      </div>
      <div className="initiate-banner__actions">
        <button
          type="button"
          onClick={() => onDismiss(message.auditId)}
          aria-label="Dismiss"
        >
          ×
        </button>
      </div>
    </div>
  );
}

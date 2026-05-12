import { useEffect, useRef } from "react";

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
  onReply: (auditId: string) => void;
  onDismiss: (auditId: string) => void;
  onMounted: (auditId: string) => void;
};

const STATE_LABEL: Record<InitiateMessage["state"], string> = {
  pending: "pending",
  delivered: "delivered",
  read: "read",
  replied_explicit: "replied",
  acknowledged_unclear: "acknowledged unclear",
  unanswered: "unanswered",
  dismissed: "dismissed",
};

export function InitiateBanner({ message, onReply, onDismiss, onMounted }: Props) {
  const firedRef = useRef(false);
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

  return (
    <div className="initiate-banner" role="region" aria-label="Nell reached out">
      <div className="initiate-banner__body">{message.body}</div>
      <div className="initiate-banner__meta">
        <span className="initiate-banner__urgency">{message.urgency}</span>
        <span className="initiate-banner__state">{STATE_LABEL[message.state]}</span>
      </div>
      <div className="initiate-banner__actions">
        <button
          type="button"
          onClick={() => onReply(message.auditId)}
          aria-label="Reply (↩)"
        >
          ↩ reply
        </button>
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

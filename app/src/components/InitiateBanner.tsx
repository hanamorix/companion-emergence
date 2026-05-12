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
    const timer = setTimeout(() => {
      if (!firedRef.current) {
        firedRef.current = true;
        onMounted(message.auditId);
      }
    }, 2000);
    return () => clearTimeout(timer);
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

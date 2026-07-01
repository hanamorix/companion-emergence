import { useEffect, useState } from "react";
import { brainLoginStatus } from "../../appConfig";
import { BrainLoginPrompt } from "./BrainLoginPrompt";

/**
 * "Clean connection" — Connection-panel entry point for the brain's own
 * clean `claude` login (mirrors NotesToggle / KindledLinkToggle placement).
 *
 * Purely an offer: shows the current authorized state read-only, plus an
 * Authorize / Re-authorize control that reveals the same BrainLoginPrompt
 * used by the on-open banner. Never blocks anything else in this panel.
 */
export function CleanConnectionSection() {
  const [authorized, setAuthorized] = useState<boolean | null>(null);
  const [showPrompt, setShowPrompt] = useState(false);

  useEffect(() => {
    let cancelled = false;
    brainLoginStatus()
      .then((res) => {
        if (!cancelled) setAuthorized(res.authorized);
      })
      .catch(() => {
        if (!cancelled) setAuthorized(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 6,
        padding: "4px 0",
      }}
    >
      <div
        style={{
          fontSize: 10.5,
          color: "var(--text-mute)",
          lineHeight: 1.55,
          letterSpacing: "0.01em",
        }}
      >
        {authorized === true
          ? "She's signed in with her own clean Claude login."
          : "Optional — give her a Claude login separate from yours, so plugin/skill noise from your setup stays out of her replies."}
      </div>

      {showPrompt ? (
        <BrainLoginPrompt
          onAuthorized={() => {
            setAuthorized(true);
            setShowPrompt(false);
          }}
          onDismiss={() => setShowPrompt(false)}
        />
      ) : (
        <button
          type="button"
          onClick={() => setShowPrompt(true)}
          style={{
            width: "100%",
            padding: "7px 10px",
            fontSize: 11,
            fontFamily: "var(--font-ui)",
            background: "var(--accent-dim)",
            color: "var(--text)",
            border: "1px solid rgba(130, 51, 41, 0.3)",
            borderRadius: 6,
            cursor: "pointer",
          }}
        >
          {authorized === true ? "re-authorize" : "authorize"}
        </button>
      )}
    </div>
  );
}

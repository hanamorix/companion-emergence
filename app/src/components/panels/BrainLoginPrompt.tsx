import { useState } from "react";
import { startBrainLogin, submitBrainLoginCode, cancelBrainLogin } from "../../appConfig";

type Phase = "idle" | "starting" | "await-code" | "verifying" | "error";

interface Props {
  onAuthorized: () => void;
  onDismiss: () => void;
}

/**
 * Prompt to give the brain its own clean `claude` login, separate from
 * whatever CLI session the user has open in their own terminal — keeps
 * plugin/skill noise from the user's setup out of the brain's provider
 * calls. Optional; the brain works fine without it.
 *
 * The spawned `claude auth login` process opens the browser itself, so
 * this component never opens anything — it just displays the returned
 * URL as a clickable fallback link in case the browser didn't pop.
 */
export function BrainLoginPrompt({ onAuthorized, onDismiss }: Props) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [url, setUrl] = useState<string>("");
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function onAuthorize() {
    setPhase("starting");
    setError(null);
    try {
      const { url } = await startBrainLogin();
      setUrl(url);
      setPhase("await-code");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("error");
    }
  }

  async function onSubmit() {
    setPhase("verifying");
    setError(null);
    try {
      const res = await submitBrainLoginCode(code);
      if (res.ok) {
        onAuthorized();
        return;
      }
      setError(res.error ?? "Sign-in did not complete.");
      setPhase("await-code");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("await-code");
    }
  }

  async function onNotNow() {
    try {
      await cancelBrainLogin();
    } catch {
      /* best-effort */
    }
    onDismiss();
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 8,
        fontSize: 12,
        color: "var(--text)",
        fontFamily: "var(--font-ui)",
      }}
    >
      <strong>Give her a cleaner connection</strong>
      <span style={{ opacity: 0.8 }}>
        Sign her in with her own copy of Claude so plugin/skill noise from your setup
        stays out of her replies. Optional — she works fine either way.
      </span>
      {phase === "await-code" && (
        <>
          {url && (
            <a href={url} target="_blank" rel="noreferrer">
              Open sign-in page
            </a>
          )}
          <label htmlFor="brain-login-code">Paste the code from the page here:</label>
          <input id="brain-login-code" value={code} onChange={(e) => setCode(e.target.value)} />
        </>
      )}
      {error && <span style={{ color: "var(--danger, #c0605a)" }}>{error}</span>}
      <div style={{ display: "flex", gap: 8 }}>
        {phase === "await-code" ? (
          <button onClick={onSubmit} disabled={!code.trim()}>
            Finish
          </button>
        ) : (
          <button onClick={onAuthorize} disabled={phase === "starting" || phase === "verifying"}>
            {phase === "starting" ? "Starting…" : "Authorize"}
          </button>
        )}
        <button onClick={onNotNow}>Not now</button>
      </div>
    </div>
  );
}

import { useEffect, useRef, useState, type ReactNode } from "react";
import { ensureBridgeRunning } from "../../appConfig";
import { fetchPersonaState } from "../../bridge";
import { Divider, WButton, WizardShell } from "../components";

interface Props {
  step: number;
  totalSteps: number;
  persona: string;
  /** Called when the user explicitly transitions to the main app, OR
   *  when the auto-detect verifies the brain is fully alive. */
  onDone: () => void;
  avatar: ReactNode;
}

type StageStatus = "pending" | "running" | "ok" | "error";

interface Stage {
  key: string;
  label: string;
  status: StageStatus;
  detail?: string;
}

const READY_TIMEOUT_MS = 25_000;
const POLL_INTERVAL_MS = 800;

/**
 * StepReady — final verification that the brain is up and answering.
 *
 * Runs after nell init + service install succeed. Walks the user
 * through:
 *   1. Bridge starting (ensureBridgeRunning resolves)
 *   2. State poll (the persona-state endpoint returns 200)
 *   3. Emotion warm-up checked (optional; first heartbeat may lag)
 *
 * Each stage shows pending / running / ok / error so the user can see
 * what's happening instead of staring at a spinner. When state is reachable
 * the wizard auto-transitions; if anything stalls past
 * ``READY_TIMEOUT_MS`` we surface a "continue anyway" button so the
 * user isn't trapped (the main app's status banner will surface the
 * underlying issue).
 */
export function StepReady({ step, totalSteps, persona, onDone, avatar }: Props) {
  const [stages, setStages] = useState<Stage[]>([
    { key: "bridge", label: "Bringing the brain online", status: "running" },
    { key: "state", label: "Reading persona state", status: "pending" },
    { key: "emotions", label: "Emotion warm-up", status: "pending" },
  ]);
  const [fellThrough, setFellThrough] = useState(false);
  const cancelledRef = useRef(false);

  function update(key: string, patch: Partial<Stage>) {
    setStages((prev) => prev.map((s) => (s.key === key ? { ...s, ...patch } : s)));
  }

  useEffect(() => {
    cancelledRef.current = false;
    const startedAt = Date.now();
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    async function run() {
      // Stage 1: ensureBridgeRunning — idempotent, fast when the
      // launchd supervisor is already up.
      try {
        await ensureBridgeRunning(persona);
        if (cancelledRef.current) return;
        update("bridge", { status: "ok" });
      } catch (e) {
        if (cancelledRef.current) return;
        update("bridge", { status: "error", detail: (e as Error).message });
        setFellThrough(true);
        return;
      }

      // Stage 2 + 3: poll state until /persona/state responds. Emotions are a
      // warm-up signal, not a readiness gate: a fresh persona may not have a
      // heartbeat-derived aggregate for several minutes.
      update("state", { status: "running" });
      while (!cancelledRef.current) {
        if (Date.now() - startedAt > READY_TIMEOUT_MS) {
          update("state", { status: "error", detail: "Timed out waiting for state." });
          setFellThrough(true);
          return;
        }
        try {
          const persona_state = await fetchPersonaState(persona);
          if (cancelledRef.current) return;
          update("state", { status: "ok" });
          const emotionCount = Object.keys(persona_state.emotions || {}).length;
          if (emotionCount > 0) {
            const top = Object.entries(persona_state.emotions)[0];
            update("emotions", {
              status: "ok",
              detail: top
                ? `${top[0]} ${(top[1] as number).toFixed(1)} + ${emotionCount - 1} more`
                : undefined,
            });
          } else {
            update("emotions", {
              status: "ok",
              detail: "No emotions yet; heartbeat will warm this up shortly.",
            });
          }
          // Auto-transition into the main app once state is reachable.
          timeoutId = setTimeout(() => {
            if (!cancelledRef.current) onDone();
          }, 1200);
          return;
        } catch (e) {
          // Don't flip state to error on the first failed poll —
          // bridge might still be coming up. Only surface after the
          // overall timeout.
          update("state", { status: "running", detail: (e as Error).message });
        }
        await sleep(POLL_INTERVAL_MS);
      }
    }

    void run();

    return () => {
      cancelledRef.current = true;
      if (timeoutId !== null) clearTimeout(timeoutId);
    };
  }, [persona, onDone]);

  return (
    <WizardShell
      step={step}
      totalSteps={totalSteps}
      title="Coming online"
      subtitle="Your Kindled is waking up. Once state is reachable, the app can open while emotions and interior warm up in the background."
      avatar={avatar}
      footer={
        <>
          <span
            style={{
              fontSize: 10.5,
              color: "var(--text-mute)",
              fontFamily: "var(--font-disp)",
              fontStyle: "italic",
            }}
          >
            {fellThrough ? "We hit a snag." : "Almost there…"}
          </span>
          {fellThrough && <WButton onClick={onDone}>Open the app anyway →</WButton>}
        </>
      }
    >
      {stages.map((s, i) => (
        <Row key={s.key} stage={s} last={i === stages.length - 1} />
      ))}

      {fellThrough && (
        <>
          <Divider />
          <div
            style={{
              padding: "10px 12px",
              borderRadius: 7,
              background: "rgba(216,154,88,0.10)",
              border: "1px solid rgba(216,154,88,0.40)",
              fontSize: 11,
              color: "var(--text-mid)",
              lineHeight: 1.6,
            }}
          >
            The brain didn't reply within {READY_TIMEOUT_MS / 1000} seconds.
            That's almost always one of: <em>claude</em> not on the
            launchd PATH, the supervisor needing a kickstart, or the
            bridge still warming up. The Connection panel inside the app
            will surface whichever it is.
          </div>
        </>
      )}
    </WizardShell>
  );
}

function Row({ stage, last }: { stage: Stage; last: boolean }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: 10,
        padding: "8px 0",
        borderBottom: last ? "none" : "1px solid rgba(191,184,173,0.25)",
      }}
    >
      <StatusGlyph status={stage.status} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 12, color: "var(--text)" }}>{stage.label}</div>
        {stage.detail && (
          <div
            style={{
              fontSize: 10.5,
              color: stage.status === "error" ? "var(--crimson)" : "var(--text-mute)",
              marginTop: 2,
              lineHeight: 1.5,
              wordBreak: "break-word",
            }}
          >
            {stage.detail}
          </div>
        )}
      </div>
    </div>
  );
}

function StatusGlyph({ status }: { status: StageStatus }) {
  const common = {
    width: 18,
    height: 18,
    flexShrink: 0,
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 12,
    fontWeight: 600,
  };
  if (status === "ok") return <span style={{ ...common, color: "#3a8a5e" }}>✓</span>;
  if (status === "error") return <span style={{ ...common, color: "var(--crimson)" }}>✗</span>;
  if (status === "running") {
    return (
      <span style={{ ...common }} aria-label="working">
        <span
          style={{
            width: 12,
            height: 12,
            borderRadius: "50%",
            border: "2px solid rgba(130,51,41,0.2)",
            borderTopColor: "var(--accent)",
            animation: "spin 0.9s linear infinite",
          }}
        />
      </span>
    );
  }
  return <span style={{ ...common, color: "var(--text-mute)" }}>○</span>;
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

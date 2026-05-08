import { useEffect, useState, type ReactNode } from "react";
import { checkClaudeCli, type ClaudeCliCheck } from "../../appConfig";
import { Divider, SectionLabel, WButton, WizardShell } from "../components";

interface Props {
  step: number;
  totalSteps: number;
  onNext: () => void;
  onBack: () => void;
  avatar: ReactNode;
}

/**
 * StepPrerequisites — verifies the one external dependency the
 * framework needs: Anthropic's ``claude`` CLI on PATH.
 *
 * The wizard probes via the Tauri ``check_claude_cli`` command on
 * mount and on every "Re-check" click. Forward navigation is gated
 * until claude is found, so users can't get to the install step
 * with a misconfigured environment that would only fail at first
 * chat. The OS-specific install commands are surfaced inline.
 */
export function StepPrerequisites({ step, totalSteps, onNext, onBack, avatar }: Props) {
  const [check, setCheck] = useState<ClaudeCliCheck | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runCheck() {
    setChecking(true);
    setError(null);
    try {
      const result = await checkClaudeCli();
      setCheck(result);
    } catch (e) {
      setError((e as Error).message);
      setCheck(null);
    } finally {
      setChecking(false);
    }
  }

  useEffect(() => {
    void runCheck();
  }, []);

  const ready = check?.found === true;

  return (
    <WizardShell
      step={step}
      totalSteps={totalSteps}
      title="Prerequisites"
      subtitle="Companion Emergence shells out to Anthropic's claude CLI for every reply, every dream, every reflex. Let's make sure it's installed before we set up your persona."
      avatar={avatar}
      footer={
        <>
          <WButton variant="ghost" onClick={onBack} small>
            ← Back
          </WButton>
          <WButton onClick={onNext} disabled={!ready}>
            Continue →
          </WButton>
        </>
      }
    >
      <SectionLabel>Claude CLI</SectionLabel>
      <CheckCard check={check} checking={checking} error={error} onRetry={runCheck} />

      {!ready && (
        <>
          <Divider />
          <SectionLabel>How to install</SectionLabel>
          <InstallInstructions />
          <Divider />
          <Note>
            <strong>Already paid for Claude?</strong> Your Claude Code
            subscription powers the CLI; no separate API key is
            needed. Run <Code>claude</Code> once after install to
            sign in, then come back and click <em>Re-check</em>.
          </Note>
        </>
      )}

      {ready && (
        <>
          <Divider />
          <Note>
            That's the only dependency. The next steps live entirely
            on this machine — your brain, memories, and conversations
            never leave it.
          </Note>
        </>
      )}
    </WizardShell>
  );
}

function CheckCard({
  check,
  checking,
  error,
  onRetry,
}: {
  check: ClaudeCliCheck | null;
  checking: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  const found = check?.found === true;
  const palette = found
    ? {
        bg: "rgba(60, 130, 90, 0.10)",
        border: "rgba(60, 130, 90, 0.45)",
        accent: "#3a8a5e",
      }
    : {
        bg: "rgba(178, 42, 42, 0.08)",
        border: "rgba(178, 42, 42, 0.40)",
        accent: "var(--crimson)",
      };

  return (
    <div
      style={{
        padding: "12px 14px",
        borderRadius: 8,
        background: palette.bg,
        border: `1px solid ${palette.border}`,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          fontSize: 12.5,
          fontWeight: 500,
          color: palette.accent,
          marginBottom: 4,
        }}
      >
        <span aria-hidden="true">{checking ? "…" : found ? "✓" : "✗"}</span>
        <span>
          {checking
            ? "Checking…"
            : found
              ? "Claude CLI found."
              : "Claude CLI not found."}
        </span>
      </div>
      {check?.path && (
        <div
          style={{
            fontSize: 10.5,
            color: "var(--text-mid)",
            fontFamily: "DM Mono, Courier New, monospace",
            marginTop: 4,
            wordBreak: "break-all",
          }}
        >
          {check.path}
        </div>
      )}
      {check?.version && (
        <div
          style={{
            fontSize: 10.5,
            color: "var(--text-mute)",
            fontFamily: "var(--font-disp)",
            marginTop: 2,
          }}
        >
          {check.version}
        </div>
      )}
      {error && (
        <div
          style={{
            fontSize: 11,
            color: "var(--crimson)",
            marginTop: 6,
            lineHeight: 1.5,
          }}
        >
          {error}
        </div>
      )}
      {!found && !checking && (
        <div style={{ marginTop: 10 }}>
          <WButton small onClick={onRetry}>
            Re-check
          </WButton>
        </div>
      )}
    </div>
  );
}

function InstallInstructions() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <Platform name="macOS / Linux">
        <div>
          The official one-liner installs <Code>claude</Code> to{" "}
          <Code>~/.local/bin/claude</Code>:
        </div>
        <CodeBlock>{`curl -fsSL https://claude.ai/install.sh | bash`}</CodeBlock>
        <div style={{ fontSize: 10.5, color: "var(--text-mute)", marginTop: 4 }}>
          Re-open your terminal so PATH picks up <Code>~/.local/bin</Code>,
          then run <Code>claude</Code> once and sign in.
        </div>
      </Platform>
      <Platform name="Windows">
        <div>
          Download the installer from{" "}
          <Code>docs.claude.com/en/docs/claude-code/setup</Code> and run
          it. After install, open a new PowerShell and run{" "}
          <Code>claude</Code> to sign in.
        </div>
      </Platform>
    </div>
  );
}

function Platform({ name, children }: { name: string; children: ReactNode }) {
  return (
    <div>
      <div
        style={{
          fontSize: 10,
          color: "var(--text-mute)",
          fontFamily: "var(--font-disp)",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          marginBottom: 4,
        }}
      >
        {name}
      </div>
      <div style={{ fontSize: 11, color: "var(--text-mid)", lineHeight: 1.6 }}>
        {children}
      </div>
    </div>
  );
}

function CodeBlock({ children }: { children: string }) {
  return (
    <pre
      style={{
        fontSize: 10.5,
        fontFamily: "DM Mono, Courier New, monospace",
        background: "rgba(0,0,0,0.06)",
        color: "var(--text)",
        padding: "8px 10px",
        borderRadius: 5,
        margin: "6px 0 0",
        whiteSpace: "pre-wrap",
        wordBreak: "break-all",
      }}
    >
      {children}
    </pre>
  );
}

function Code({ children }: { children: ReactNode }) {
  return (
    <code
      style={{
        fontFamily: "DM Mono, Courier New, monospace",
        fontSize: "0.92em",
        background: "rgba(0,0,0,0.05)",
        padding: "1px 4px",
        borderRadius: 3,
      }}
    >
      {children}
    </code>
  );
}

function Note({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        fontSize: 11,
        color: "var(--text-mid)",
        lineHeight: 1.65,
      }}
    >
      {children}
    </div>
  );
}

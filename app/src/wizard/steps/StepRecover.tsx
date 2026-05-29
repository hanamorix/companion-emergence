import { useEffect, useState, type ReactNode } from "react";
import {
  invokeRecoverPreflight, invokeRunRecover, type RecoverPreflight,
} from "../../bridge";
import { errString } from "../../lib/errString";
import { WizardShell } from "../components";

export function StepRecover({
  persona, sourceDir, onDone, avatar,
}: {
  persona: string;
  sourceDir: string | null;
  onDone: () => void;
  avatar?: ReactNode;
}) {
  const [preflight, setPreflight] = useState<RecoverPreflight | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    let alive = true;
    invokeRecoverPreflight(persona, sourceDir)
      .then((p) => { if (alive) setPreflight(p); })
      .catch((e) => { if (alive) setError(errString(e)); });
    return () => { alive = false; };
  }, [persona, sourceDir]);

  async function run() {
    setRunning(true);
    setError(null);
    try {
      const r = await invokeRunRecover(persona, sourceDir, false, false);
      if (!r.success) setError(r.stderr || "Recovery failed.");
      else onDone();
    } catch (e) {
      setError(errString(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <WizardShell title="Recover memories" avatar={avatar}>
      {error ? (
        <div role="alert">Recovery problem: {error}</div>
      ) : !preflight ? (
        <div>Checking what can be restored…</div>
      ) : (
        <div>
          <p>{preflight.missing} memories to restore, {preflight.unfade} to un-fade.</p>
          <button disabled={running} onClick={run}>
            {running ? "Restoring…" : "Restore now"}
          </button>
        </div>
      )}
    </WizardShell>
  );
}

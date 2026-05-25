import { useEffect, useState } from "react";
import {
  invokeRecoverPreflight, invokeRunRecover, type RecoverPreflight,
} from "../../bridge";
import { errString } from "../../lib/errString";

export function StepRecover({
  persona, sourceDir, onDone,
}: {
  persona: string;
  sourceDir: string | null;
  onDone: () => void;
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

  if (error) return <div role="alert">Recovery problem: {error}</div>;
  if (!preflight) return <div>Checking what can be restored…</div>;

  return (
    <div>
      <h2>Recover {persona}</h2>
      <p>{preflight.missing} memories to restore, {preflight.unfade} to un-fade.</p>
      <button disabled={running} onClick={run}>
        {running ? "Restoring…" : "Restore now"}
      </button>
    </div>
  );
}

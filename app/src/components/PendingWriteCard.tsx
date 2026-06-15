/** PendingWriteCard — the consent gate for a proposed file write.
 *
 * `propose_write` queues a guarded pending write (writing nothing); the
 * frontend polls `PersonaState.pending_writes` and renders one of these per
 * row. The user approves (commit to disc, guard re-run) or declines (discard).
 * Mirrors the InitiateBanner reach-out card: a soft card with the gist + two
 * actions, no settings, no knobs — install + name + talk, plus "can I write
 * this?".
 */

export type PendingWrite = {
  id: string;
  op: "create" | "append";
  path: string;
  /** Content preview, capped server-side at 2000 chars. */
  preview: string;
  /** True iff the real content is longer than the preview. */
  truncated: boolean;
  proposed_at: string;
};

type Props = {
  write: PendingWrite;
  onApprove: (id: string) => void;
  onDecline: (id: string) => void;
  /** True while an approve/decline call is in flight — disables both buttons
   *  so the user can't double-submit. */
  busy?: boolean;
};

export function PendingWriteCard({ write, onApprove, onDecline, busy = false }: Props) {
  return (
    <div
      className="pending-write-card"
      role="region"
      aria-label={`Proposed file ${write.op}`}
    >
      <div className="pending-write-card__header">
        <span className="pending-write-card__badge">{write.op}</span>
        <span className="pending-write-card__path">{write.path}</span>
      </div>
      <pre className="pending-write-card__preview">{write.preview}</pre>
      {write.truncated && (
        <div className="pending-write-card__truncated">(truncated)</div>
      )}
      <div className="pending-write-card__actions">
        <button
          type="button"
          className="pending-write-card__approve"
          onClick={() => onApprove(write.id)}
          disabled={busy}
        >
          Approve
        </button>
        <button
          type="button"
          className="pending-write-card__decline"
          onClick={() => onDecline(write.id)}
          disabled={busy}
        >
          Decline
        </button>
      </div>
    </div>
  );
}

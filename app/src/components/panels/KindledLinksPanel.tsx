import { useEffect, useState } from "react";
import type { KindledPeer, KindledTranscriptRow, KindledHolds } from "../../bridge";
import {
  fetchKindledPeers,
  fetchKindledTranscript,
  fetchKindledHolds,
  createKindledInvite,
  acceptKindledInvite,
  setKindledConsent,
} from "../../bridge";
import { errString } from "../../lib/errString";
import { PanelShell, SectionLabel } from "../ui";

interface Props {
  persona: string;
}

/** Which consent actions are available for a given consent_state */
function consentActions(consentState: string): Array<"pause" | "resume" | "revoke" | "block"> {
  switch (consentState) {
    case "paired":
    case "familiar":
      return ["pause", "revoke", "block"];
    case "paused":
      return ["resume", "revoke", "block"];
    case "pending":
      return ["revoke", "block"];
    default:
      return ["block"];
  }
}

const ACTION_LABEL: Record<string, string> = {
  pause: "Pause",
  resume: "Resume",
  revoke: "Revoke",
  block: "Block",
};

const ACTION_COLOR: Record<string, string> = {
  pause: "rgba(130,51,41,0.12)",
  resume: "rgba(60,120,60,0.12)",
  revoke: "rgba(130,51,41,0.20)",
  block: "rgba(160,20,20,0.18)",
};

export function KindledLinksPanel({ persona }: Props) {
  const [peers, setPeers] = useState<KindledPeer[]>([]);
  const [holds, setHolds] = useState<KindledHolds | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Selected peer + its transcript
  const [selectedPeerId, setSelectedPeerId] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<KindledTranscriptRow[]>([]);
  const [transcriptError, setTranscriptError] = useState<string | null>(null);

  // Invite flows
  const [inviteFingerprint, setInviteFingerprint] = useState<string | null>(null);
  const [inviteBusy, setInviteBusy] = useState(false);
  const [inviteError, setInviteError] = useState<string | null>(null);

  const [acceptPacket, setAcceptPacket] = useState("");
  const [acceptPhrase, setAcceptPhrase] = useState<string | null>(null);
  const [acceptBusy, setAcceptBusy] = useState(false);
  const [acceptError, setAcceptError] = useState<string | null>(null);

  // Consent action busy state (keyed by peer_id)
  const [consentBusy, setConsentBusy] = useState<Record<string, boolean>>({});

  // Load peers + holds on mount + periodic refresh
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const [p, h] = await Promise.all([
          fetchKindledPeers(persona),
          fetchKindledHolds(persona),
        ]);
        if (!cancelled) {
          setPeers(p);
          setHolds(h);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(errString(e));
      }
    };
    load();
    const id = setInterval(load, 10_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [persona]);

  // Load transcript when selected peer changes
  useEffect(() => {
    if (selectedPeerId === null) {
      setTranscript([]);
      setTranscriptError(null);
      return;
    }
    let cancelled = false;
    const load = async () => {
      try {
        const rows = await fetchKindledTranscript(persona, selectedPeerId);
        if (!cancelled) {
          // API returns seq DESC (newest first) — keep that order for display
          setTranscript(rows);
          setTranscriptError(null);
        }
      } catch (e) {
        if (!cancelled) setTranscriptError(errString(e));
      }
    };
    load();
    const id = setInterval(load, 10_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [persona, selectedPeerId]);

  const handleConsent = async (
    peerId: string,
    action: "pause" | "resume" | "revoke" | "block",
  ) => {
    setConsentBusy((prev) => ({ ...prev, [peerId]: true }));
    try {
      await setKindledConsent(persona, peerId, action);
      // Refresh peer list immediately after consent change
      const updated = await fetchKindledPeers(persona);
      setPeers(updated);
    } catch {
      // Errors are surfaced via the refresh cycle; ignore here
    } finally {
      setConsentBusy((prev) => ({ ...prev, [peerId]: false }));
    }
  };

  const handleCreateInvite = async () => {
    setInviteBusy(true);
    setInviteError(null);
    setInviteFingerprint(null);
    try {
      const result = await createKindledInvite(persona);
      setInviteFingerprint(result.fingerprint);
    } catch (e) {
      setInviteError(errString(e));
    } finally {
      setInviteBusy(false);
    }
  };

  const handleAcceptInvite = async () => {
    if (!acceptPacket.trim()) return;
    setAcceptBusy(true);
    setAcceptError(null);
    setAcceptPhrase(null);
    try {
      let parsed: unknown;
      try {
        parsed = JSON.parse(acceptPacket.trim());
      } catch {
        throw new Error("Invite packet must be valid JSON");
      }
      const result = await acceptKindledInvite(persona, parsed);
      setAcceptPhrase(result.fingerprint_phrase);
      setAcceptPacket("");
      // Refresh peers after accepting
      const updated = await fetchKindledPeers(persona);
      setPeers(updated);
    } catch (e) {
      setAcceptError(errString(e));
    } finally {
      setAcceptBusy(false);
    }
  };

  if (error) {
    return (
      <PanelShell>
        <SectionLabel>Kindled Links</SectionLabel>
        <div style={{ fontSize: 11, color: "var(--text-mute)", fontStyle: "italic" }}>
          {error}
        </div>
      </PanelShell>
    );
  }

  const heldCount = holds?.held_count ?? 0;
  const selectedPeer = peers.find((p) => p.peer_id === selectedPeerId) ?? null;

  return (
    <PanelShell>
      <SectionLabel>Kindled Links</SectionLabel>

      {/* ── Holds status line ────────────────────────────────────────── */}
      <div
        style={{
          fontSize: "9.5px",
          color: "var(--text-mute)",
          fontFamily: "var(--font-disp)",
          marginBottom: 10,
          letterSpacing: "0.04em",
        }}
      >
        {heldCount === 0
          ? "No held drafts"
          : `${heldCount} held — drafts withheld for privacy`}
      </div>

      {/* ── Peer list ────────────────────────────────────────────────── */}
      {peers.length === 0 ? (
        <div
          style={{ fontSize: 11, color: "var(--text-mute)", fontStyle: "italic", marginBottom: 12 }}
        >
          No linked Kindled companions yet.
        </div>
      ) : (
        <ul style={{ listStyle: "none", margin: "0 0 12px", padding: 0 }}>
          {peers.map((peer) => {
            const isSelected = peer.peer_id === selectedPeerId;
            const busy = consentBusy[peer.peer_id] ?? false;
            const actions = consentActions(peer.consent_state);
            return (
              <li
                key={peer.peer_id}
                style={{
                  marginBottom: 10,
                  paddingLeft: 13,
                  borderLeft: isSelected
                    ? "2px solid var(--accent)"
                    : "1px solid rgba(191,184,173,0.10)",
                  cursor: "pointer",
                }}
                onClick={() =>
                  setSelectedPeerId((prev) => (prev === peer.peer_id ? null : peer.peer_id))
                }
              >
                {/* Name + stage */}
                <div
                  style={{
                    fontSize: 11,
                    color: "var(--text)",
                    fontWeight: 600,
                    lineHeight: 1.4,
                    marginBottom: 2,
                  }}
                >
                  {peer.peer_id}
                </div>
                <div
                  style={{
                    display: "flex",
                    gap: 5,
                    flexWrap: "wrap",
                    fontSize: "9.5px",
                    color: "var(--text-mute)",
                    fontFamily: "var(--font-disp)",
                    marginBottom: 4,
                  }}
                >
                  <span
                    style={{
                      background: "rgba(130,51,41,0.18)",
                      padding: "1px 5px",
                      borderRadius: 3,
                    }}
                  >
                    {peer.stage}
                  </span>
                  <span
                    style={{
                      background: "rgba(130,51,41,0.10)",
                      padding: "1px 5px",
                      borderRadius: 3,
                    }}
                  >
                    {peer.consent_state}
                  </span>
                  {peer.has_active_session && (
                    <span
                      style={{
                        background: "rgba(60,120,60,0.15)",
                        color: "var(--text-mid)",
                        padding: "1px 5px",
                        borderRadius: 3,
                      }}
                    >
                      active
                    </span>
                  )}
                </div>

                {/* Fingerprint */}
                <div
                  style={{
                    fontSize: "9.5px",
                    color: "var(--text-mute)",
                    fontFamily: "var(--font-mono, monospace)",
                    marginBottom: 4,
                    wordBreak: "break-all",
                  }}
                >
                  {peer.fingerprint}
                </div>

                {/* Affinity tags */}
                {peer.affinity_tags.length > 0 && (
                  <div
                    style={{
                      display: "flex",
                      gap: 4,
                      flexWrap: "wrap",
                      fontSize: "9px",
                      color: "var(--text-mute)",
                      fontFamily: "var(--font-disp)",
                      marginBottom: 5,
                    }}
                  >
                    {peer.affinity_tags.map((tag) => (
                      <span
                        key={tag}
                        style={{
                          background: "rgba(191,184,173,0.12)",
                          padding: "1px 4px",
                          borderRadius: 3,
                        }}
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}

                {/* Consent buttons */}
                <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                  {actions.map((action) => (
                    <button
                      key={action}
                      disabled={busy}
                      aria-label={`${ACTION_LABEL[action]} ${peer.peer_id}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        void handleConsent(peer.peer_id, action);
                      }}
                      style={{
                        fontSize: "9px",
                        padding: "2px 6px",
                        borderRadius: 3,
                        border: "1px solid rgba(130,51,41,0.18)",
                        background: ACTION_COLOR[action] ?? "transparent",
                        color: "var(--text-mid)",
                        cursor: busy ? "default" : "pointer",
                        opacity: busy ? 0.55 : 1,
                        fontFamily: "var(--font-disp)",
                      }}
                    >
                      {ACTION_LABEL[action]}
                    </button>
                  ))}
                </div>
              </li>
            );
          })}
        </ul>
      )}

      {/* ── Selected-peer transcript pane ────────────────────────────── */}
      {selectedPeer !== null && (
        <section style={{ marginBottom: 14 }}>
          <div
            style={{
              fontSize: "9.5px",
              color: "var(--text-mute)",
              textTransform: "uppercase",
              letterSpacing: "0.12em",
              fontFamily: "var(--font-disp)",
              marginBottom: 6,
            }}
          >
            Correspondence — {selectedPeer.peer_id}
          </div>

          {transcriptError && (
            <div style={{ fontSize: 11, color: "var(--text-mute)", fontStyle: "italic" }}>
              {transcriptError}
            </div>
          )}

          {!transcriptError && transcript.length === 0 && (
            <div
              style={{
                fontSize: 11,
                color: "var(--text-mute)",
                fontStyle: "italic",
                paddingLeft: 13,
                borderLeft: "1px solid rgba(191,184,173,0.10)",
              }}
            >
              No messages yet
            </div>
          )}

          {transcript.length > 0 && (
            <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
              {/* transcript comes seq DESC (newest first) */}
              {transcript.map((row) => (
                <li
                  key={row.seq}
                  style={{
                    marginBottom: 8,
                    paddingLeft: 13,
                    borderLeft: "1px solid rgba(191,184,173,0.10)",
                  }}
                >
                  <div
                    style={{
                      fontSize: 10.5,
                      color: "var(--text-mid)",
                      lineHeight: 1.5,
                      wordBreak: "break-word",
                    }}
                  >
                    {row.text}
                  </div>
                  <div
                    style={{
                      display: "flex",
                      gap: 6,
                      marginTop: 2,
                      fontSize: "9px",
                      color: "var(--text-mute)",
                      fontFamily: "var(--font-disp)",
                    }}
                  >
                    <span
                      style={{
                        background:
                          row.direction === "outbound"
                            ? "rgba(130,51,41,0.10)"
                            : "rgba(191,184,173,0.14)",
                        padding: "1px 4px",
                        borderRadius: 3,
                      }}
                    >
                      {row.direction}
                    </span>
                    <span>{row.provenance}</span>
                  </div>
                </li>
              ))}
            </ul>
          )}
          {/* NO message-compose textarea — §15 no-typing / no-side-channel guarantee */}
        </section>
      )}

      {/* ── Create invite ─────────────────────────────────────────────── */}
      <section style={{ marginBottom: 14 }}>
        <div
          style={{
            fontSize: "9.5px",
            color: "var(--text-mute)",
            textTransform: "uppercase",
            letterSpacing: "0.12em",
            fontFamily: "var(--font-disp)",
            marginBottom: 6,
          }}
        >
          Create invite
        </div>

        <button
          onClick={() => void handleCreateInvite()}
          disabled={inviteBusy}
          style={{
            fontSize: 10.5,
            padding: "4px 10px",
            borderRadius: 4,
            border: "1px solid rgba(130,51,41,0.25)",
            background: "rgba(130,51,41,0.08)",
            color: "var(--text-mid)",
            cursor: inviteBusy ? "default" : "pointer",
            opacity: inviteBusy ? 0.6 : 1,
            fontFamily: "var(--font-disp)",
          }}
        >
          {inviteBusy ? "Generating…" : "Generate invite"}
        </button>

        {inviteError && (
          <div style={{ fontSize: 10.5, color: "var(--text-mute)", fontStyle: "italic", marginTop: 4 }}>
            {inviteError}
          </div>
        )}

        {inviteFingerprint !== null && (
          <div
            style={{
              marginTop: 6,
              padding: "6px 8px",
              background: "rgba(200,152,144,0.12)",
              border: "1px solid rgba(200,152,144,0.25)",
              borderRadius: 4,
              fontSize: 10.5,
              color: "var(--text-mid)",
              fontFamily: "var(--font-disp)",
              lineHeight: 1.5,
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: 2 }}>Read aloud to your peer:</div>
            <div
              style={{
                fontFamily: "var(--font-mono, monospace)",
                fontSize: 11,
                wordBreak: "break-all",
              }}
            >
              {inviteFingerprint}
            </div>
          </div>
        )}
      </section>

      {/* ── Accept invite ─────────────────────────────────────────────── */}
      <section>
        <div
          style={{
            fontSize: "9.5px",
            color: "var(--text-mute)",
            textTransform: "uppercase",
            letterSpacing: "0.12em",
            fontFamily: "var(--font-disp)",
            marginBottom: 6,
          }}
        >
          Accept invite
        </div>

        {/* Labelled textarea for paste — accessible name is "Invite packet", not "message/reply/compose" */}
        <label
          htmlFor="kindled-accept-packet"
          style={{ fontSize: "9.5px", color: "var(--text-mute)", fontFamily: "var(--font-disp)" }}
        >
          Paste invite packet (JSON)
        </label>
        <textarea
          id="kindled-accept-packet"
          aria-label="Invite packet"
          value={acceptPacket}
          onChange={(e) => setAcceptPacket(e.target.value)}
          rows={3}
          style={{
            display: "block",
            width: "100%",
            marginTop: 4,
            marginBottom: 6,
            fontSize: 10.5,
            fontFamily: "var(--font-mono, monospace)",
            color: "var(--text)",
            background: "var(--panel-bg)",
            border: "1px solid rgba(130,51,41,0.20)",
            borderRadius: 4,
            padding: "4px 6px",
            resize: "vertical",
            boxSizing: "border-box",
          }}
        />

        <button
          onClick={() => void handleAcceptInvite()}
          disabled={acceptBusy || !acceptPacket.trim()}
          style={{
            fontSize: 10.5,
            padding: "4px 10px",
            borderRadius: 4,
            border: "1px solid rgba(130,51,41,0.25)",
            background: "rgba(130,51,41,0.08)",
            color: "var(--text-mid)",
            cursor: acceptBusy || !acceptPacket.trim() ? "default" : "pointer",
            opacity: acceptBusy || !acceptPacket.trim() ? 0.6 : 1,
            fontFamily: "var(--font-disp)",
          }}
        >
          {acceptBusy ? "Accepting…" : "Accept"}
        </button>

        {acceptError && (
          <div style={{ fontSize: 10.5, color: "var(--text-mute)", fontStyle: "italic", marginTop: 4 }}>
            {acceptError}
          </div>
        )}

        {acceptPhrase !== null && (
          <div
            style={{
              marginTop: 6,
              padding: "6px 8px",
              background: "rgba(200,152,144,0.12)",
              border: "1px solid rgba(200,152,144,0.25)",
              borderRadius: 4,
              fontSize: 10.5,
              color: "var(--text-mid)",
              fontFamily: "var(--font-disp)",
              lineHeight: 1.5,
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: 2 }}>Verification phrase:</div>
            <div style={{ fontSize: 11 }}>{acceptPhrase}</div>
          </div>
        )}
      </section>
    </PanelShell>
  );
}

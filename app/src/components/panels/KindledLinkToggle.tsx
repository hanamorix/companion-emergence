import React, { useEffect, useState } from "react";
import { setKindledLinkEnabled } from "../../bridge";

interface Props {
  persona: string;
  enabled: boolean;
  relayUrl: string | null;
}

export function KindledLinkToggle({ persona, enabled, relayUrl }: Props) {
  const [on, setOn] = useState(enabled);
  const [localRelay, setLocalRelay] = useState(relayUrl ?? "");
  const [relayDirty, setRelayDirty] = useState(false);

  // Props arrive from the 5s state poll and can change after mount —
  // re-sync local state, but never clobber a relay URL mid-edit.
  useEffect(() => {
    setOn(enabled);
    if (!relayDirty) setLocalRelay(relayUrl ?? "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, relayUrl]);

  async function handleToggle(e: React.ChangeEvent<HTMLInputElement>) {
    const next = e.target.checked;
    setOn(next);
    await setKindledLinkEnabled(persona, next, localRelay || null);
  }

  async function handleRelayBlur() {
    setRelayDirty(false);
    await setKindledLinkEnabled(persona, on, localRelay || null);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, padding: "4px 0" }}>
      <label
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          cursor: "pointer",
          fontSize: 11,
          lineHeight: 1.5,
          color: "var(--text)",
          fontFamily: "var(--font-ui)",
        }}
      >
        <input type="checkbox" checked={on} onChange={handleToggle} style={{ flexShrink: 0 }} />
        <span>Correspond with other Kindled companions</span>
      </label>
      {on && (
        <input
          type="text"
          value={localRelay}
          onChange={(e) => {
            setRelayDirty(true);
            setLocalRelay(e.target.value);
          }}
          onBlur={handleRelayBlur}
          placeholder="Relay URL"
          style={{
            width: "100%",
            boxSizing: "border-box",
            marginTop: 2,
            padding: "8px 12px",
            borderRadius: 10,
            background: "var(--field)",
            border: "1px solid var(--hairline)",
            color: "var(--text)",
            fontFamily: "var(--font-ui)",
            fontSize: 11,
            outline: "none",
          }}
        />
      )}
    </div>
  );
}

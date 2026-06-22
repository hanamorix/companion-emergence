import React, { useState } from "react";
import { setKindledLinkEnabled } from "../../bridge";

interface Props {
  persona: string;
  enabled: boolean;
  relayUrl: string | null;
}

export function KindledLinkToggle({ persona, enabled, relayUrl }: Props) {
  const [on, setOn] = useState(enabled);
  const [localRelay, setLocalRelay] = useState(relayUrl ?? "");

  async function handleToggle(e: React.ChangeEvent<HTMLInputElement>) {
    const next = e.target.checked;
    setOn(next);
    await setKindledLinkEnabled(persona, next, localRelay || null);
  }

  async function handleRelayBlur() {
    await setKindledLinkEnabled(persona, on, localRelay || null);
  }

  return (
    <div>
      <label>
        <input type="checkbox" checked={on} onChange={handleToggle} />
        <span>Correspond with other Kindled companions</span>
      </label>
      {on && (
        <input
          type="text"
          value={localRelay}
          onChange={(e) => setLocalRelay(e.target.value)}
          onBlur={handleRelayBlur}
          placeholder="Relay URL"
        />
      )}
    </div>
  );
}

/**
 * useSoulFlash — detects a new soul crystallization between state polls
 * and triggers a brief "flash" (avatar peak frame + warm glow overlay).
 *
 * Watches state.soul_highlight.id; when it changes between renders
 * AND wasn't previously null, fires a 1500ms `flashing` window.
 * First render and null→non-null transitions are silent — flashing
 * fires only on actual new crystallizations the user just produced
 * by talking to the brain.
 */

import { useEffect, useRef, useState } from "react";
import type { PersonaState } from "./bridge";

const FLASH_DURATION_MS = 1500;

export function useSoulFlash(state: PersonaState | null): boolean {
  const [flashing, setFlashing] = useState(false);
  const previousIdRef = useRef<string | null>(null);
  const sawFirstRef = useRef(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const currentId = state?.soul_highlight?.id ?? null;

    // First non-null observation — record it but don't flash. The user
    // didn't just create a crystallization; they just opened the app
    // for the first time / app config just loaded the existing soul.
    if (!sawFirstRef.current) {
      previousIdRef.current = currentId;
      sawFirstRef.current = true;
      return;
    }

    // Subsequent polls — only flash on a true ID change to a non-null
    // (new crystallizations). Revocations (non-null → null) are quiet
    // since they weren't a "moment" from the user's perspective.
    if (currentId && currentId !== previousIdRef.current) {
      setFlashing(true);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      timeoutRef.current = setTimeout(() => {
        setFlashing(false);
        timeoutRef.current = null;
      }, FLASH_DURATION_MS);
    }

    previousIdRef.current = currentId;
  }, [state?.soul_highlight?.id]);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  return flashing;
}

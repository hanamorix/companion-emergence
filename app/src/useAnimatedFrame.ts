/**
 * useAnimatedFrame — schedules which frame the avatar should display.
 *
 * The runtime drives the 4-frame matrix:
 *
 *   IDLE  : base, with random blinks (3-6s interval, 150ms duration).
 *   SPEAKING: alternates speaking/base at ~6Hz to simulate mouth motion.
 *           Blinks land on whichever frame the timer hits, producing
 *           speaking-blink (peak intensity) when both fire together.
 *
 * Pure hook — owns its own setInterval/setTimeout cleanup. No props
 * other than current category + isSpeaking flag, so it can drive
 * NellAvatar without lifting more state up.
 */

import { useEffect, useRef, useState } from "react";
import type { Frame } from "./expressions";

const BLINK_DURATION_MS = 150;
const BLINK_INTERVAL_MIN_MS = 3000;
const BLINK_INTERVAL_RANGE_MS = 3000; // adds 0-3000 jitter
const SPEAKING_PHASE_MS = 167; // ~6Hz mouth alternation

interface Options {
  isSpeaking: boolean;
  /** Disable all animation (e.g. user toggled reduced-motion). */
  reducedMotion?: boolean;
}

export function useAnimatedFrame({ isSpeaking, reducedMotion = false }: Options): Frame {
  const [blinking, setBlinking] = useState(false);
  const [speakingPhase, setSpeakingPhase] = useState(false);

  // Blink scheduler — random interval, brief flash
  useEffect(() => {
    if (reducedMotion) return;
    let cancelled = false;
    let nestedTimeout: ReturnType<typeof setTimeout> | null = null;

    function scheduleNext() {
      if (cancelled) return;
      const delay = BLINK_INTERVAL_MIN_MS + Math.random() * BLINK_INTERVAL_RANGE_MS;
      nestedTimeout = setTimeout(() => {
        if (cancelled) return;
        setBlinking(true);
        nestedTimeout = setTimeout(() => {
          if (cancelled) return;
          setBlinking(false);
          scheduleNext();
        }, BLINK_DURATION_MS);
      }, delay);
    }

    scheduleNext();
    return () => {
      cancelled = true;
      if (nestedTimeout) clearTimeout(nestedTimeout);
    };
  }, [reducedMotion]);

  // Speaking-phase scheduler — alternates while isSpeaking is true
  useEffect(() => {
    if (!isSpeaking || reducedMotion) {
      setSpeakingPhase(false);
      return;
    }
    const id = setInterval(() => setSpeakingPhase((p) => !p), SPEAKING_PHASE_MS);
    return () => clearInterval(id);
  }, [isSpeaking, reducedMotion]);

  // Pick the frame from the two phases.
  if (isSpeaking && speakingPhase) {
    return blinking ? "speaking-blink" : "speaking";
  }
  return blinking ? "blink" : "base";
}

/** Imperative helper: trigger a one-shot peak frame for N ms (e.g.
 * soul-crystallization flash, climax event). Returns a setter that
 * caller wires into a useState. */
export function usePeakFlash(
  durationMs = 800,
): { flashing: boolean; flash: () => void } {
  const [flashing, setFlashing] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  function flash() {
    setFlashing(true);
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => {
      setFlashing(false);
      timeoutRef.current = null;
    }, durationMs);
  }

  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  return { flashing, flash };
}

import { useEffect, useMemo, useState } from "react";
import {
  pickExpressionFromState,
  resolveFrameUrl,
  type ExpressionCategory,
} from "../expressions";
import { useAnimatedFrame } from "../useAnimatedFrame";
import type { PersonaState } from "../bridge";

interface Props {
  state: PersonaState | null;
  /** True while the chat is awaiting a reply — drives the speaking animation. */
  isSpeaking?: boolean;
  /** Honor the user's reduced-motion toggle / OS pref. */
  reducedMotion?: boolean;
  size?: number;
}

/**
 * NellAvatar — breathing face with the 4-frame animation engine.
 *
 * Pipeline:
 *   1. pickExpressionFromState(state) → category (heuristic stack:
 *      body emotions override social emotions, etc.)
 *   2. useAnimatedFrame({isSpeaking}) → frame (idle blinks, speaking
 *      mouth-cycling, speaking-blink at intensity peaks)
 *   3. resolveFrameUrl(category, frame) → asset URL (prefers new
 *      4-frame directory format; falls back to legacy single-file
 *      variants when art for that category isn't ready yet)
 *
 * Mode-aware: glow color (crimson when provider_down) + grayscale
 * filter (when offline). Breathing animation continues.
 */
export function NellAvatar({
  state,
  isSpeaking = false,
  reducedMotion = false,
  size = 280,
}: Props) {
  const mode = state?.mode ?? "live";
  const dimmed = mode === "offline";
  const glowColor = mode === "provider_down" ? "#8A3033" : "#823329";

  const category: ExpressionCategory = useMemo(
    () => pickExpressionFromState(state),
    [state],
  );

  const frame = useAnimatedFrame({ isSpeaking, reducedMotion });

  // Resolve asset URL whenever category or frame changes
  const [src, setSrc] = useState(() => resolveFrameUrl(category, frame));
  useEffect(() => {
    setSrc(resolveFrameUrl(category, frame));
  }, [category, frame]);

  // Energy-driven breath cadence: faster when fresh, slower when tired
  const energy = state?.body?.energy ?? 6;
  const breathSeconds = clamp(5 + (10 - energy) * 0.35, 4.5, 8);

  return (
    <div
      style={{ position: "relative", width: size, height: size, flexShrink: 0 }}
      data-category={category}
    >
      {/* warm lacquer glow */}
      <div
        style={{
          position: "absolute",
          inset: -28,
          borderRadius: "50%",
          background: `radial-gradient(ellipse at 50% 58%, ${glowColor}44 0%, transparent 68%)`,
          filter: "blur(20px)",
          animation: reducedMotion ? "none" : `breathe ${breathSeconds}s ease-in-out infinite`,
          opacity: dimmed ? 0.15 : 0.9,
          transition: "opacity 1.2s ease",
          pointerEvents: "none",
          zIndex: 0,
        }}
      />
      <img
        src={src}
        alt="Nell"
        draggable={false}
        style={{
          position: "relative",
          zIndex: 1,
          width: size,
          height: size,
          objectFit: "contain",
          objectPosition: "center top",
          animation: reducedMotion ? "none" : `breathe ${breathSeconds}s ease-in-out infinite`,
          filter: dimmed ? "grayscale(0.6) brightness(0.65) sepia(0.2)" : "none",
          transition: "filter 1.2s ease, opacity 0.18s ease",
          userSelect: "none",
        }}
      />
    </div>
  );
}

function clamp(v: number, min: number, max: number): number {
  return Math.min(Math.max(v, min), max);
}

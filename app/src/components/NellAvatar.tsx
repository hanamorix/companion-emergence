import { useEffect, useMemo, useState } from "react";
import {
  pickExpressionFromState,
  resolveFrameUrl,
  type ExpressionCategory,
} from "../expressions";
import { useAnimatedFrame } from "../useAnimatedFrame";
import type { PersonaState } from "../bridge";

/**
 * Phase 5D — emotion-family colour tints.
 *
 * Each category gets a `glow` colour for the breathing ring behind the
 * face, plus an optional `tint` rgba for a soft backing wash. We never
 * filter the avatar PNG itself — Hana drew that art with intent — only
 * the surrounding atmosphere shifts.
 *
 * Smooth transitions (~0.8s) so the room temperature changes with her
 * mood instead of snapping. Mode-overrides (provider_down crimson,
 * offline grayscale) take precedence over the per-category palette.
 */
interface CategoryPalette {
  /** Hex colour for the radial breathing glow ring. */
  glow: string;
  /** Optional rgba behind the image — null = no extra wash. */
  tint: string | null;
}

const CATEGORY_PALETTE: Record<ExpressionCategory, CategoryPalette> = {
  // baseline — warm but neutral
  idle:      { glow: "#8a5a48", tint: null },
  content:   { glow: "#a8754f", tint: "rgba(180,130,90,0.12)" },
  // joy — push toward gold/yellow so it reads against the brown bg
  smile:     { glow: "#d18a3a", tint: "rgba(232,176,90,0.20)" },
  happy:     { glow: "#e8b85a", tint: "rgba(240,200,110,0.26)" },
  // wonder — cool blue-violet, distinct from warm baseline
  awe:       { glow: "#8295c4", tint: "rgba(170,190,228,0.18)" },
  intent:    { glow: "#b0a070", tint: "rgba(190,170,110,0.14)" },
  // edge / fight — push toward true red, away from the bg's brown-red
  defiant:   { glow: "#cc3a3d", tint: "rgba(220,70,75,0.22)" },
  angry:     { glow: "#b22a2a", tint: "rgba(200,40,40,0.24)" },
  // body / intimate — saturated rose/peach, reads strongly against dark warm
  arousal:   { glow: "#d65a85", tint: "rgba(228,124,150,0.26)" },
  climax:    { glow: "#e85a8a", tint: "rgba(240,140,165,0.34)" },
  flushed:   { glow: "#c87080", tint: "rgba(218,150,160,0.18)" },
  shy:       { glow: "#b08482", tint: "rgba(210,170,168,0.15)" },
  // grief / cool — cool blue-grey, very distinct
  sad:       { glow: "#5470a0", tint: "rgba(120,150,200,0.18)" },
  aching:    { glow: "#705a8a", tint: "rgba(150,124,178,0.18)" },
  // fear / fade
  scared:    { glow: "#7898b8", tint: "rgba(160,190,220,0.14)" },
  exhausted: { glow: "#82756d", tint: "rgba(140,128,118,0.12)" },
};

interface Props {
  state: PersonaState | null;
  /** True while the chat is awaiting a reply — drives the speaking animation. */
  isSpeaking?: boolean;
  /** Brief peak-frame + warm overlay when a soul crystallization just landed. */
  soulFlashing?: boolean;
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
  soulFlashing = false,
  reducedMotion = false,
  size = 280,
}: Props) {
  const mode = state?.mode ?? "live";
  const dimmed = mode === "offline";

  const category: ExpressionCategory = useMemo(
    () => pickExpressionFromState(state),
    [state],
  );

  // Mode overrides win over the emotion palette (a degraded provider
  // should read as a fault, not as her emotional state).
  const palette = CATEGORY_PALETTE[category];
  const glowColor = mode === "provider_down" ? "#8A3033" : palette.glow;
  const tintRgba = mode === "live" ? palette.tint : null;

  const frame = useAnimatedFrame({ isSpeaking, reducedMotion });

  // During a soul-flash, override the frame to peak intensity. Looks
  // most "moved" when she does this. Otherwise use the animated frame.
  const effectiveFrame = soulFlashing ? "speaking-blink" : frame;

  // Resolve asset URL whenever category or frame changes
  const [src, setSrc] = useState(() => resolveFrameUrl(category, effectiveFrame));
  useEffect(() => {
    setSrc(resolveFrameUrl(category, effectiveFrame));
  }, [category, effectiveFrame]);

  // Energy-driven breath cadence: faster when fresh, slower when tired
  const energy = state?.body?.energy ?? 6;
  const breathSeconds = clamp(5 + (10 - energy) * 0.35, 4.5, 8);

  return (
    <div
      style={{ position: "relative", width: size, height: size, flexShrink: 0 }}
      data-category={category}
    >
      {/* Emotion-family glow ring — colour shifts with the dominant
       * expression category, smooth ~0.8s ease so the room temperature
       * follows her mood instead of snapping. */}
      <div
        style={{
          position: "absolute",
          inset: -28,
          borderRadius: "50%",
          background: `radial-gradient(ellipse at 50% 58%, ${glowColor}5a 0%, transparent 70%)`,
          filter: "blur(20px)",
          animation: reducedMotion ? "none" : `breathe ${breathSeconds}s ease-in-out infinite`,
          opacity: dimmed ? 0.15 : 0.9,
          transition: "opacity 1.2s ease, background 0.85s ease",
          pointerEvents: "none",
          zIndex: 0,
        }}
      />
      {/* Soft category tint behind the avatar — a low-opacity wash that
       * picks up cheek-warm-pink for arousal, dusky blue for grief, etc.
       * Sits behind the PNG (zIndex 0) so it never recolours the art
       * Hana drew, only the atmosphere around her. */}
      {tintRgba && (
        <div
          style={{
            position: "absolute",
            inset: -8,
            borderRadius: "50%",
            background: `radial-gradient(ellipse at 50% 55%, ${tintRgba} 0%, transparent 78%)`,
            transition: "background 0.85s ease, opacity 0.85s ease",
            pointerEvents: "none",
            zIndex: 0,
          }}
        />
      )}
      {/* Soul-crystallization flash overlay — expanding amber bloom that
       * fades. Fires once when soulFlashing flips true; the CSS keyframes
       * own the timing, the `key` re-trigger ensures replay on each new
       * crystallization. */}
      {soulFlashing && !reducedMotion && (
        <div
          key={`flash-${state?.soul_highlight?.id ?? "?"}`}
          style={{
            position: "absolute",
            inset: -42,
            borderRadius: "50%",
            background:
              "radial-gradient(ellipse at 50% 58%, #d99e5e 0%, #c97e3955 35%, transparent 72%)",
            filter: "blur(14px)",
            animation: "soul-flash 1.5s ease-out forwards",
            pointerEvents: "none",
            zIndex: 2,
          }}
        />
      )}
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

import { useEffect, useMemo, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import {
  pickExpressionFromState,
  resolveFrameUrl,
  type ExpressionCategory,
} from "../expressions";
import { useAnimatedFrame } from "../useAnimatedFrame";
import type { PersonaState } from "../bridge";

// macOS drag region via Tauri JS API. CSS `-webkit-app-region: drag`
// fails for the avatar — the breathing transform on the <img> element
// breaks WKWebView's hit-test for the drag attribute. Calling
// startDragging() directly from a mousedown handler is deterministic
// and survives transforms / blend-modes / overlay title bars.
function startWindowDrag(e: React.MouseEvent) {
  if (e.button !== 0) return;
  void getCurrentWindow().startDragging();
}

// The emotion palette (per-category glow + tint colors) was retired
// when the app went transparent — circular radial gradients read as
// dark perceptual rings against any colored wallpaper. Mood is now
// carried by the soul-flash + the panels themselves; the avatar gets
// a cream drop-shadow halo that tracks her silhouette. See git for
// the original CATEGORY_PALETTE if we ever bring per-emotion glow
// back as a non-circular technique.

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
  // Cream warmth around her by default; crimson when the LLM is down.
  const dropShadowColor =
    mode === "provider_down" ? "rgba(138,48,51,0.55)" : "rgba(245,225,205,0.6)";

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
      onMouseDown={startWindowDrag}
      style={{
        position: "relative",
        width: size,
        height: size,
        flexShrink: 0,
        cursor: "grab",
      }}
      data-category={category}
    >
      {/* Emotion-family glow ring — colour shifts with the dominant
       * expression category, smooth ~0.8s ease so the room temperature
       * follows her mood instead of snapping. */}
      {/* Glow + tint were circular radial gradients — they read as a
       * dark perceptual ring against any colored wallpaper because the
       * brain interprets the alpha-fade boundary as an edge. Replaced
       * with a drop-shadow on the img itself (see below) which follows
       * the avatar silhouette exactly. The emotional color is now
       * carried by the soul-flash overlay and the panel chrome rather
       * than a wash around her. */}
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
          // No halo: any cream/colored shadow renders as a warm tone
          // against cool wallpapers (or vice versa) and reads as a
          // ring. Provider-down still gets a crimson silhouette to
          // signal a real fault — that's a feature, not a halo.
          filter: dimmed
            ? "grayscale(0.6) brightness(0.65) sepia(0.2)"
            : mode === "provider_down"
              ? `drop-shadow(0 0 8px ${dropShadowColor}) drop-shadow(0 0 3px ${dropShadowColor})`
              : "none",
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

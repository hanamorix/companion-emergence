import { useEffect, useState } from "react";
import { defaultExpression, expressionForEmotions } from "../expressions";
import type { PersonaState } from "../bridge";

interface Props {
  state: PersonaState | null;
  /** "live" | "bridge_down" | "provider_down" | "offline" — drives glow color + filter */
  size?: number;
}

/**
 * NellAvatar — breathing face with mode-aware glow.
 *
 * The image src is driven by persona state's top emotion. While
 * loading or when state is null, falls back to the default smile
 * expression so the screen never goes blank.
 */
export function NellAvatar({ state, size = 280 }: Props) {
  const mode = state?.mode ?? "live";
  const dimmed = mode === "offline";
  const glowColor = mode === "provider_down" ? "#8A3033" : "#823329";

  const [src, setSrc] = useState(defaultExpression());
  useEffect(() => {
    if (!state || !state.emotions) {
      setSrc(defaultExpression());
      return;
    }
    setSrc(expressionForEmotions(state.emotions));
  }, [state]);

  return (
    <div style={{ position: "relative", width: size, height: size, flexShrink: 0 }}>
      {/* warm lacquer glow */}
      <div
        style={{
          position: "absolute",
          inset: -28,
          borderRadius: "50%",
          background: `radial-gradient(ellipse at 50% 58%, ${glowColor}44 0%, transparent 68%)`,
          filter: "blur(20px)",
          animation: "breathe 5s ease-in-out infinite",
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
          animation: "breathe 5s ease-in-out infinite",
          filter: dimmed ? "grayscale(0.6) brightness(0.65) sepia(0.2)" : "none",
          transition: "filter 1.2s ease, opacity 0.3s ease",
          userSelect: "none",
        }}
      />
    </div>
  );
}

import { useEffect, useState } from "react";
import { expressionPath } from "../expressions";

const STEP_TO_EXPRESSION: Record<string, string> = {
  welcome: expressionPath("smile", 4),
  name: expressionPath("shy", 1),
  user: expressionPath("happy", 1),
  voice: expressionPath("smile", 1),
  migrate: expressionPath("exhausted", 2),
  review: expressionPath("happy", 3),
  installing: expressionPath("happy", 2),
  done: expressionPath("happy", 1),
  error: expressionPath("scared", 1),
};

interface Props {
  step: keyof typeof STEP_TO_EXPRESSION;
  size?: number;
}

/**
 * Wizard avatar — fades between expressions as the user moves through
 * the steps. Same breathing animation as the main app.
 */
export function WizardAvatar({ step, size = 200 }: Props) {
  const [current, setCurrent] = useState(step);
  const [fading, setFading] = useState(false);

  useEffect(() => {
    if (step === current) return;
    setFading(true);
    const t = setTimeout(() => {
      setCurrent(step);
      setFading(false);
    }, 200);
    return () => clearTimeout(t);
  }, [step, current]);

  const src = STEP_TO_EXPRESSION[current] ?? STEP_TO_EXPRESSION.welcome;

  return (
    <div style={{ position: "relative", width: size, height: size, flexShrink: 0 }}>
      <div
        style={{
          position: "absolute",
          inset: -24,
          borderRadius: "50%",
          background:
            "radial-gradient(ellipse at 50% 58%, rgba(130,51,41,0.35) 0%, transparent 68%)",
          filter: "blur(18px)",
          animation: "breathe 5s ease-in-out infinite",
          pointerEvents: "none",
        }}
      />
      <img
        src={src}
        alt="Persona avatar"
        style={{
          position: "relative",
          zIndex: 1,
          width: size,
          height: size,
          objectFit: "contain",
          objectPosition: "center top",
          animation: "breathe 5s ease-in-out infinite",
          opacity: fading ? 0 : 1,
          transition: "opacity 0.2s ease",
          userSelect: "none",
        }}
        draggable={false}
      />
    </div>
  );
}

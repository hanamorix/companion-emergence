/**
 * Map persona emotional state → which expression PNG to render.
 *
 * Phase 2 ships a static default ("smile 4"); Phase 5 will refine the
 * heuristic with multi-axis state (energy, exhaustion, recent
 * crystallization context).
 *
 * Asset URLs resolve via Vite's `import.meta.glob` over the repo's
 * expressions/ directory. Vite hashes the assets into `dist/assets/`
 * at build time and produces stable URLs for `<img src>` — no
 * symlink, no public-dir copy, works identically in dev + bundled.
 */

export type ExpressionCategory =
  | "angry"
  | "exhausted"
  | "happy"
  | "sad"
  | "scared"
  | "shy"
  | "smile";

const VARIANTS_PER_CATEGORY = 4;

// Vite glob import — resolved at build time. Keys are filesystem-relative
// paths like "../../expressions/smile 4.png"; values are URL strings.
const ALL_EXPRESSIONS = import.meta.glob<string>(
  "../../expressions/*.png",
  { eager: true, query: "?url", import: "default" },
);

/** Resolve to the URL Vite generated for a given expression file. */
export function expressionPath(category: ExpressionCategory, variant: number): string {
  const v = Math.min(Math.max(variant, 1), VARIANTS_PER_CATEGORY);
  const key = `../../expressions/${category} ${v}.png`;
  const url = ALL_EXPRESSIONS[key];
  if (!url) {
    // Should never happen if expressions/ is intact, but fail soft to
    // a known-good asset rather than render a broken image.
    const fallback = ALL_EXPRESSIONS["../../expressions/smile 4.png"];
    if (!fallback) throw new Error(`expressions/ catalog missing — bundle broken`);
    return fallback;
  }
  return url;
}

/** Pick a default expression — used while bridge is offline / before first state read. */
export function defaultExpression(): string {
  return expressionPath("smile", 4);
}

/**
 * Heuristic: top-emotion-driven expression mapping. Stable but minimal.
 * Phase 5 will replace this with a richer multi-axis mapping (energy,
 * exhaustion, dominant emotion, recent crystallization context).
 */
export function expressionForEmotions(
  emotions: Record<string, number>,
  variantSeed = 1,
): string {
  const entries = Object.entries(emotions);
  if (entries.length === 0) return defaultExpression();
  // Sort desc; pick the top emotion's family
  entries.sort(([, a], [, b]) => b - a);
  const [topName] = entries[0];
  const v = ((variantSeed - 1) % VARIANTS_PER_CATEGORY) + 1;

  // Coarse mapping. Refine in Phase 5.
  if (["love", "tenderness", "belonging", "gratitude", "joy"].includes(topName)) {
    return expressionPath("smile", v);
  }
  if (["curiosity", "creative_hunger"].includes(topName)) {
    return expressionPath("happy", v);
  }
  if (["grief", "loneliness", "nostalgia", "anchor_pull"].includes(topName)) {
    return expressionPath("sad", v);
  }
  if (["anger", "defiance", "frustration"].includes(topName)) {
    return expressionPath("angry", v);
  }
  if (["fear", "guilt", "shame"].includes(topName)) {
    return expressionPath("scared", v);
  }
  if (["vulnerability", "tenderness"].includes(topName)) {
    return expressionPath("shy", v);
  }
  if (["rest_need", "exhaustion"].includes(topName)) {
    return expressionPath("exhausted", v);
  }
  return defaultExpression();
}

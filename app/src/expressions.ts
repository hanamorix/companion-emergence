/**
 * Expression engine — pick a category, render a frame.
 *
 * Two responsibilities:
 *   1. pickExpressionCategory(state) — heuristic stack mapping persona
 *      state to one of 13 expression categories. Body emotions override
 *      social emotions ("physiology speaks first"); long absence reads as
 *      aching; everything else falls through emotion-family routing.
 *   2. resolveFrameUrl(category, frame) — find the PNG for a (category,
 *      frame) pair, supporting BOTH the new 4-frame directory format
 *      (expressions/<category>/<frame>.png) AND the legacy single-file
 *      variants (expressions/<category> <n>.png). New art drops in by
 *      adding a directory; nothing else changes.
 *
 * Asset URLs resolve via Vite's import.meta.glob — hashed into
 * dist/assets/ at build time. New expression directories are picked up
 * automatically next build.
 */

import type { BodyState, PersonaState } from "./bridge";

// ────────────────────────────────────────────────────────────────────────────
// Categories
// ────────────────────────────────────────────────────────────────────────────

export type ExpressionCategory =
  // existing 7 — legacy art in <category> 1.png … <category> 4.png
  | "smile"
  | "happy"
  | "sad"
  | "angry"
  | "scared"
  | "shy"
  | "exhausted"
  // new 6 — Phase 5 art. Falls back to closest legacy category until each
  // <category>/ directory lands.
  | "content"
  | "aching"
  | "flushed"
  | "awe"
  | "intent"
  | "defiant";

/**
 * Where to render this category from when no art for it exists yet.
 * Removed from the map as each new directory ships with art.
 */
const FALLBACK_CATEGORY: Partial<Record<ExpressionCategory, ExpressionCategory>> = {
  content: "smile",
  aching: "sad",
  flushed: "shy",
  awe: "happy",
  intent: "happy",
  defiant: "angry",
};

// ────────────────────────────────────────────────────────────────────────────
// Frames — the 4-frame matrix (mouth × eyes)
// ────────────────────────────────────────────────────────────────────────────

export type Frame =
  | "base"            // mouth-closed + eyes-open — the resting frame
  | "blink"           // mouth-closed + eyes-closed
  | "speaking"        // mouth-open + eyes-open
  | "speaking-blink"; // mouth-open + eyes-closed — peak intensity

// ────────────────────────────────────────────────────────────────────────────
// Asset resolution
// ────────────────────────────────────────────────────────────────────────────

const NEW_FORMAT_ASSETS = import.meta.glob<string>(
  "../../expressions/*/*.png",
  { eager: true, query: "?url", import: "default" },
);

const LEGACY_FORMAT_ASSETS = import.meta.glob<string>(
  "../../expressions/*.png",
  { eager: true, query: "?url", import: "default" },
);

// Map a frame → which legacy variant index to substitute for it. We
// only have 4 generic variants in the legacy format, none of which
// were authored as eyes-open vs eyes-closed. Best we can do is spread
// the four variants across the four frames so cycling still produces
// movement.
const LEGACY_FRAME_TO_VARIANT: Record<Frame, number> = {
  base: 1,
  blink: 1, // legacy has no real blink — show variant 1, the engine can fast-cut to it for blink-effect
  speaking: 2,
  "speaking-blink": 3,
};

/** Resolve a (category, frame) pair to a hashed asset URL. */
export function resolveFrameUrl(category: ExpressionCategory, frame: Frame): string {
  // 1. New 4-frame directory format
  const newKey = `../../expressions/${category}/${frame}.png`;
  if (NEW_FORMAT_ASSETS[newKey]) return NEW_FORMAT_ASSETS[newKey];

  // 2. Fallback category in new format
  const fallback = FALLBACK_CATEGORY[category];
  if (fallback) {
    const fallbackNewKey = `../../expressions/${fallback}/${frame}.png`;
    if (NEW_FORMAT_ASSETS[fallbackNewKey]) return NEW_FORMAT_ASSETS[fallbackNewKey];
    return resolveLegacyUrl(fallback, frame);
  }

  // 3. Legacy single-file format
  return resolveLegacyUrl(category, frame);
}

function resolveLegacyUrl(category: ExpressionCategory, frame: Frame): string {
  const variant = LEGACY_FRAME_TO_VARIANT[frame];
  const key = `../../expressions/${category} ${variant}.png`;
  const url = LEGACY_FORMAT_ASSETS[key];
  if (url) return url;
  // Last-ditch fallback: smile 4 always exists in legacy art
  const lastResort = LEGACY_FORMAT_ASSETS["../../expressions/smile 4.png"];
  if (!lastResort) throw new Error("expressions/ catalog empty — bundle broken");
  return lastResort;
}

/** True iff the new 4-frame directory format ships art for this category. */
export function hasNewFormatArt(category: ExpressionCategory): boolean {
  return Object.keys(NEW_FORMAT_ASSETS).some((k) =>
    k.startsWith(`../../expressions/${category}/`),
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Heuristic stack — persona state → category
// ────────────────────────────────────────────────────────────────────────────

const SMILE_FAMILY = new Set(["love", "tenderness", "belonging", "gratitude", "joy"]);
const HAPPY_FAMILY = new Set(["curiosity", "creative_hunger", "pride"]);
const SAD_FAMILY = new Set(["grief", "loneliness", "nostalgia", "anchor_pull"]);
const ANGRY_FAMILY = new Set(["anger", "frustration"]);
const SCARED_FAMILY = new Set(["fear", "guilt", "shame"]);
const SHY_FAMILY = new Set(["vulnerability"]);
const EXHAUSTED_FAMILY = new Set(["rest_need"]);
const AWE_FAMILY = new Set(["awe", "emergence", "hope"]);
const CONTENT_FAMILY = new Set(["contentment", "peace"]);
const INTENT_FAMILY = new Set(["creative_hunger"]); // also in HAPPY — INTENT wins via order
const DEFIANT_FAMILY = new Set(["defiance"]);
const ACHING_FAMILY = new Set(["body_grief", "freedom_ache"]);

/**
 * Body-emotion thresholds — when these fire, override whatever the
 * social emotions are saying. Spec principle: physiology speaks first.
 */
const CLIMAX_OVERRIDE = 7;
const AROUSAL_DESIRE_SUM_OVERRIDE = 12;
const EXHAUSTION_OVERRIDE = 7;
const DAYS_SINCE_CONTACT_ACHING = 5;

export interface ExpressionInput {
  emotions: Record<string, number>;
  body: BodyState | null;
}

export function pickExpressionCategory(input: ExpressionInput): ExpressionCategory {
  const { body, emotions } = input;
  const bodyEmo = body?.body_emotions ?? {};

  // 1. Body overrides — physiology first
  if ((bodyEmo.climax ?? 0) >= CLIMAX_OVERRIDE) return "flushed";
  if (((bodyEmo.arousal ?? 0) + (bodyEmo.desire ?? 0)) >= AROUSAL_DESIRE_SUM_OVERRIDE) {
    return "flushed";
  }
  if ((body?.exhaustion ?? 0) >= EXHAUSTION_OVERRIDE) return "exhausted";

  // 2. Long absence reads as bodily yearning
  if ((body?.days_since_contact ?? 0) >= DAYS_SINCE_CONTACT_ACHING) return "aching";

  // 3. Social emotion families — order matters because some emotion
  // names appear in multiple families (e.g. creative_hunger). Specific
  // categories (intent, defiant) win over generic ones (happy, angry).
  const top = topEmotion(emotions);
  if (top) {
    if (INTENT_FAMILY.has(top) && (emotions[top] ?? 0) >= 7) return "intent";
    if (DEFIANT_FAMILY.has(top)) return "defiant";
    if (AWE_FAMILY.has(top)) return "awe";
    if (CONTENT_FAMILY.has(top)) return "content";
    if (ACHING_FAMILY.has(top)) return "aching";
    if (SMILE_FAMILY.has(top)) return "smile";
    if (HAPPY_FAMILY.has(top)) return "happy";
    if (SAD_FAMILY.has(top)) return "sad";
    if (ANGRY_FAMILY.has(top)) return "angry";
    if (SCARED_FAMILY.has(top)) return "scared";
    if (SHY_FAMILY.has(top)) return "shy";
    if (EXHAUSTED_FAMILY.has(top)) return "exhausted";
  }

  // 4. Default — settled resting state
  return "content";
}

/** From a state snapshot, pick the dominant emotion name. */
export function pickExpressionFromState(state: PersonaState | null): ExpressionCategory {
  if (!state) return "content";
  return pickExpressionCategory({
    emotions: state.emotions ?? {},
    body: state.body ?? null,
  });
}

function topEmotion(emotions: Record<string, number>): string | null {
  let topName: string | null = null;
  let topValue = 0;
  for (const [name, value] of Object.entries(emotions)) {
    if (value > topValue) {
      topValue = value;
      topName = name;
    }
  }
  return topValue > 0 ? topName : null;
}

// Convenience for callers that just want a default render
export function defaultExpressionUrl(): string {
  return resolveFrameUrl("content", "base");
}

/**
 * Legacy single-category-variant resolver for callers that pick by
 * intent rather than by state (e.g. wizard Avatar maps step → variant
 * directly). Kept thin: callers that ARE state-driven should use
 * pickExpressionFromState + resolveFrameUrl instead.
 */
export function expressionPath(category: ExpressionCategory, variant: number): string {
  // Map variant 1-4 to a frame in the new format. Best-guess pairing
  // so existing wizard mappings (e.g. "smile 4") look reasonable in
  // either art format.
  const variantToFrame: Record<number, Frame> = {
    1: "base",
    2: "speaking",
    3: "speaking-blink",
    4: "blink",
  };
  const frame = variantToFrame[Math.min(Math.max(variant, 1), 4)] ?? "base";

  // Try new format first
  const newKey = `../../expressions/${category}/${frame}.png`;
  if (NEW_FORMAT_ASSETS[newKey]) return NEW_FORMAT_ASSETS[newKey];

  // Legacy path
  const legacyKey = `../../expressions/${category} ${variant}.png`;
  const legacyUrl = LEGACY_FORMAT_ASSETS[legacyKey];
  if (legacyUrl) return legacyUrl;

  // Last resort
  return defaultExpressionUrl();
}

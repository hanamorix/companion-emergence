// The avatar's expression picker. The engine decides what is "high": the server sends the list of names it calls
// high (`feels_high`), and the picker follows that list and nothing else. A missing list (null, or absent from an
// older bridge) means nothing counts as high; no state at all means the neutral expression.

import { describe, it, expect } from "vitest";
import type { BodyState, PersonaState } from "./bridge";
import { pickExpressionCategory, pickExpressionFromState } from "./expressions";

function body(overrides: Partial<BodyState> = {}): BodyState {
  return {
    energy: 6,
    temperature: 5,
    exhaustion: 0,
    session_hours: 0,
    days_since_contact: 0,
    body_emotions: {},
    computed_at: "2026-11-10T09:00:00+00:00",
    ...overrides,
  };
}

function state(overrides: Partial<PersonaState> = {}): PersonaState {
  return {
    persona: "test",
    emotions: {},
    body: null,
    interior: { dream: null, research: null, heartbeat: null, reflex: null },
    soul_highlight: null,
    connection: { provider: "claude-cli", model: null, last_heartbeat_at: null },
    mode: "live",
    recovering: false,
    felt_time_recovered: false,
    ...overrides,
  };
}

describe("expression picker, feels_high present", () => {
  it("shows arousal when the list holds arousal and desire, whatever the body numbers are", () => {
    const input = {
      emotions: {},
      body: body({ body_emotions: { arousal: 1, desire: 1 } }),
      feelsHigh: ["arousal", "desire"],
    };
    expect(pickExpressionCategory(input)).toBe("arousal");
  });

  it("does not show arousal when the list holds only one of the two", () => {
    const only = (name: string) =>
      pickExpressionCategory({ emotions: {}, body: body(), feelsHigh: [name] });
    expect(only("arousal")).toBe("idle");
    expect(only("desire")).toBe("idle");
  });

  it("does not show arousal without a body, even when the list holds both", () => {
    expect(
      pickExpressionCategory({ emotions: {}, body: null, feelsHigh: ["arousal", "desire"] }),
    ).toBe("idle");
  });

  it("does not show arousal from big body numbers when the list is empty", () => {
    const input = {
      emotions: {},
      body: body({ body_emotions: { arousal: 10, desire: 10 } }),
      feelsHigh: [],
    };
    expect(pickExpressionCategory(input)).toBe("idle");
  });

  it("shows intent when creative hunger is the top feeling and the list holds it", () => {
    expect(
      pickExpressionCategory({ emotions: { creative_hunger: 3 }, body: null, feelsHigh: ["creative_hunger"] }),
    ).toBe("intent");
  });

  it("shows the ordinary expression for creative hunger when the list does not hold it", () => {
    expect(
      pickExpressionCategory({ emotions: { creative_hunger: 9 }, body: null, feelsHigh: [] }),
    ).toBe("happy");
  });
});

describe("expression picker, feels_high null or absent: nothing counts as high", () => {
  const missing: Array<[string, string[] | null | undefined]> = [
    ["null", null],
    ["undefined", undefined],
  ];

  it.each(missing)("does not show arousal from body arousal plus desire at 12 or more (list %s)", (_label, feelsHigh) => {
    const input = {
      emotions: {},
      body: body({ body_emotions: { arousal: 8, desire: 8 } }),
      feelsHigh,
    };
    expect(pickExpressionCategory(input)).toBe("idle");
  });

  it.each(missing)("does not show arousal at the old boundary of exactly 12 (list %s)", (_label, feelsHigh) => {
    const input = {
      emotions: {},
      body: body({ body_emotions: { arousal: 6, desire: 6 } }),
      feelsHigh,
    };
    expect(pickExpressionCategory(input)).toBe("idle");
  });

  it.each(missing)("does not show intent for creative hunger at 7 or more (list %s)", (_label, feelsHigh) => {
    expect(
      pickExpressionCategory({ emotions: { creative_hunger: 9 }, body: null, feelsHigh }),
    ).toBe("happy");
  });

  it("a state without the field and a state with feels_high null both count nothing as high", () => {
    const loud = {
      emotions: { creative_hunger: 9 },
      body: body({ body_emotions: { arousal: 8, desire: 8 } }),
    };
    expect(pickExpressionFromState(state(loud))).toBe("happy");
    expect(pickExpressionFromState(state({ ...loud, feels_high: null }))).toBe("happy");
  });

  it("keeps the climax and exhaustion expressions, which are not governed by the list", () => {
    expect(
      pickExpressionCategory({ emotions: {}, body: body({ body_emotions: { climax: 7 } }), feelsHigh: null }),
    ).toBe("climax");
    expect(
      pickExpressionCategory({ emotions: {}, body: body({ exhaustion: 7 }), feelsHigh: null }),
    ).toBe("exhausted");
  });
});

describe("expression picker, the content family", () => {
  it.each(["contentment", "peace", "calm", "groundedness"])(
    "shows the content expression when %s is the top feeling",
    (name) => {
      expect(pickExpressionCategory({ emotions: { [name]: 6 }, body: null })).toBe("content");
    },
  );

  it("does not show it for a feeling outside the family", () => {
    expect(pickExpressionCategory({ emotions: { relief: 6 }, body: null })).not.toBe("content");
  });
});

describe("expression picker, no state at all", () => {
  it("shows the neutral expression", () => {
    expect(pickExpressionFromState(null)).toBe("idle");
  });
});

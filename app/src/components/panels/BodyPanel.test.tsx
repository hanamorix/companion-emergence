import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

import { BodyPanel } from "./BodyPanel";

// Minimal PersonaState shape — only the body fields BodyPanel reads.
function fakeState(body: Record<string, unknown> | null) {
  return { body } as any;
}

const BODY_EMOTIONS_QUIET = {
  arousal: 0,
  desire: 0,
  climax: 0,
  touch_hunger: 0.2,
  comfort_seeking: 0.4,
  rest_need: 0.3,
};

const BODY_EMOTIONS_STIRRING = {
  arousal: 0,
  desire: 6.2,
  climax: 0,
  touch_hunger: 4.1,
  comfort_seeking: 0,
  rest_need: 2.0,
};

function makeBody(overrides: Partial<{ body_emotions: Record<string, number> }>) {
  return {
    energy: 7,
    temperature: 5,
    exhaustion: 0,
    session_hours: 0.5,
    days_since_contact: 0,
    body_emotions: overrides.body_emotions ?? BODY_EMOTIONS_QUIET,
  };
}

describe("BodyPanel", () => {
  afterEach(() => cleanup());

  it("shows 'Body quiet.' (body-scoped) when all six body emotions are ≤ 0.4", () => {
    render(<BodyPanel state={fakeState(makeBody({}))} />);
    // The label must be obviously body-scoped — bare "Quiet." reads as a
    // global state ("Nell feels nothing") even though it sits under the
    // "Body Emotions" header. "Body quiet." anchors the empty-state to
    // its section.
    expect(screen.getByText(/Body quiet\./)).toBeInTheDocument();
    // Must NOT render the ambiguous bare "Quiet." regression.
    expect(screen.queryByText(/^Quiet\.$/)).not.toBeInTheDocument();
  });

  it("renders body-emotion bars when any value is > 0.4 and skips the empty-state label", () => {
    render(<BodyPanel state={fakeState(makeBody({ body_emotions: BODY_EMOTIONS_STIRRING }))} />);
    // Bar component renders the label with `_` → ` ` substitution.
    expect(screen.getByText("desire")).toBeInTheDocument();
    expect(screen.getByText("touch hunger")).toBeInTheDocument();
    expect(screen.getByText("rest need")).toBeInTheDocument();
    expect(screen.queryByText(/Body quiet\./)).not.toBeInTheDocument();
  });

  it("shows 'Body offline.' when state.body is null (no signal yet)", () => {
    render(<BodyPanel state={fakeState(null)} />);
    expect(screen.getByText(/Body offline\./)).toBeInTheDocument();
  });
});

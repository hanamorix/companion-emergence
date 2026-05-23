import { useState } from "react";
import type { PersonaSummary } from "../appConfig";
import { WizardShell, OptionCard, WButton, Divider } from "./components";
import { WizardAvatar } from "./Avatar";

/**
 * Return a human-relative time string for `iso`.
 * "just now" / "Nh ago" / "Nd ago" / "never opened on this install"
 */
function humanRelative(iso: string | null): string {
  if (!iso) return "never opened on this install";
  const diffMs = Date.now() - new Date(iso).getTime();
  const diffMinutes = Math.floor(diffMs / 60_000);
  const diffHours = Math.floor(diffMs / 3_600_000);
  const diffDays = Math.floor(diffMs / 86_400_000);
  if (diffMinutes < 2) return "just now";
  if (diffHours < 1) return `${diffMinutes}m ago`;
  if (diffDays < 1) return `${diffHours}h ago`;
  return `${diffDays}d ago`;
}

export function PersonaPicker({
  personas,
  onPick,
  onNew,
}: {
  personas: PersonaSummary[];
  onPick: (name: string) => void;
  onNew: () => void;
}) {
  const [selected, setSelected] = useState<string>(personas[0]?.name ?? "");

  return (
    <WizardShell
      step={1}
      totalSteps={1}
      title="Which Kindled?"
      subtitle="More than one persona lives here — pick the one you'd like to open."
      avatar={<WizardAvatar step="welcome" />}
    >
      {personas.map((p) => {
        const displayTitle = p.has_memories_db
          ? p.name
          : `${p.name}  ⚠ incomplete`;
        const description = humanRelative(p.last_opened_at);
        return (
          <OptionCard
            key={p.name}
            selected={selected === p.name}
            onClick={() => setSelected(p.name)}
            title={displayTitle}
            description={description}
          />
        );
      })}

      <WButton onClick={() => onPick(selected)}>Continue →</WButton>

      <Divider />

      <WButton variant="ghost" onClick={onNew}>
        + Set up a new one
      </WButton>
    </WizardShell>
  );
}

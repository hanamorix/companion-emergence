import type { TunablesNotice } from "../bridge";

/** The kept-aside file name is produced by the brain (`tunables.json.corrupt-<time>`), so nothing unusual is expected in it.
 *  It is still shown defensively: control characters are removed and it is cut at this many characters. */
export const TUNABLES_NOTICE_NAME_MAX = 80;

function cleanName(name: string | null): string {
  if (!name) return "(unknown)";
  const kept = Array.from(name).filter((ch) => {
    const code = ch.codePointAt(0) ?? 0;
    return code >= 32 && !(code >= 127 && code <= 159);
  });
  return kept.join("").slice(0, TUNABLES_NOTICE_NAME_MAX);
}

/** Plain-language text for a damaged `tunables.json` (#462). Each text says what happened and how to resolve or dismiss it. */
export function tunablesNoticeMessage(notice: TunablesNotice): string {
  const name = cleanName(notice.kept_as);
  if (notice.kind === "reset") {
    return (
      `Your tunables.json could not be read, so your overrides are not in effect. The damaged file was kept as ${name}. ` +
      "Copy your overrides from it back into tunables.json to use them again, or delete that file to dismiss this message."
    );
  }
  if (notice.kind === "restored") {
    return (
      `Your tunables.json could not be read, so an older saved copy was put in its place. The damaged file was kept as ${name}. ` +
      "Check that your overrides are what you expect, then delete that file to dismiss this message."
    );
  }
  return (
    "Your tunables.json cannot be read right now, so your overrides are not in effect. The file was left as it is. " +
    "Fix its permissions or content, then restart; this message goes away by itself after the restart."
  );
}

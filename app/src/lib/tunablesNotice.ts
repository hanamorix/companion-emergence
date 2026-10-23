import type { TunablesNotice } from "../bridge";

/** The reason is the TOML parser's short text, cleaned by the brain already. It is still shown defensively:
 *  control characters are removed and it is cut at this many characters. */
export const TUNABLES_NOTICE_REASON_MAX = 120;

function cleanReason(reason: string | null): string {
  if (!reason) return "";
  const kept = Array.from(reason).filter((ch) => {
    const code = ch.codePointAt(0) ?? 0;
    return code >= 32 && !(code >= 127 && code <= 159);
  });
  return kept.join("").slice(0, TUNABLES_NOTICE_REASON_MAX);
}

/** Plain-language text for a `tunables.toml` the brain could not read (#462, #529). The file is never repaired or reset:
 *  the text says where the problem is, that the defaults are in use meanwhile, and that the file was left as it is. */
export function tunablesNoticeMessage(notice: TunablesNotice): string {
  const hasLine = typeof notice.line === "number";
  const where = hasLine
    ? ` at line ${notice.line}${typeof notice.column === "number" ? `, column ${notice.column}` : ""}`
    : "";
  const reason = cleanReason(notice.reason);
  const why = reason ? ` (${reason})` : "";
  const fix = hasLine ? "Fix that line to use your settings again." : "Fix the file to use your settings again.";
  return (
    `Your tunables.toml could not be read${where}${why}, so your settings are not in effect and the defaults are being used. ` +
    `The file was left as it is. ${fix}`
  );
}

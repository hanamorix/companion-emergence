import type { RebuildNotice } from "../bridge";
import { cleanReason } from "./tunablesNotice";

/** Plain-language text for a switch-over that did not complete while the bridge started anyway on the feelings saved before
 *  (#633 review, 11-18, point 5; the persona is named from 11-18b item 1). `name` is the persona's display name as the window
 *  shows it (the caller capitalizes the folder name), put in the possessive; a blank name gives the neutral "The feelings".
 *  The reason is the brain's one-line text, cleaned and cut by `cleanReason`; the log path is shown whole, and the sentence
 *  naming it is left out when the brain sent none. The full reason stays in the log. */
export function rebuildNoticeMessage(notice: RebuildNotice, name: string): string {
  const reason = cleanReason(notice.reason);
  const why = reason ? `: ${reason}` : "";
  const where = notice.log_path ? ` Details are in the log: ${notice.log_path}.` : "";
  const who = name.trim();
  const subject = who ? `${who}'s feelings` : "The feelings";
  return `${subject} could not be brought up to date${why}. The app is using the ones saved before.${where}`;
}

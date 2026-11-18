import type { RebuildNotice } from "../bridge";
import { cleanReason } from "./tunablesNotice";

/** Plain-language text for a switch-over that did not complete while the bridge started anyway on the feelings saved before
 *  (#633 review, 11-18, point 5). The reason is the brain's one-line text, cleaned and cut by `cleanReason`; the log
 *  path is shown whole, and the sentence naming it is left out when the brain sent none. The full reason stays in the log. */
export function rebuildNoticeMessage(notice: RebuildNotice): string {
  const reason = cleanReason(notice.reason);
  const why = reason ? `: ${reason}` : "";
  const where = notice.log_path ? ` Details are in the log: ${notice.log_path}.` : "";
  return `Your feelings could not be brought up to date${why}. The app is using the ones saved before.${where}`;
}

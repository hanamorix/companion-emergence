import type { EmbeddingDownloadNotice } from "../bridge";

/** Plain-language text for the memory-search model the brain downloads in the background (#833). While it downloads, memory
 *  search uses keywords only; after a failed download the brain tries again on its own. The size is the library's own figure,
 *  left out when the brain sent none or a number that is not a size. */
export function embeddingDownloadNoticeMessage(notice: EmbeddingDownloadNotice): string {
  if (notice.state === "failed") {
    return "The memory search model could not be downloaded. Memory search uses keywords only, and the download will be tried again.";
  }
  const size =
    typeof notice.size_gb === "number" && Number.isFinite(notice.size_gb) && notice.size_gb > 0
      ? ` (${notice.size_gb.toFixed(1)} GB)`
      : "";
  return `Downloading the memory search model${size}. Memory search uses keywords until it finishes.`;
}

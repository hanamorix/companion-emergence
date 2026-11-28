import { describe, expect, it } from "vitest";

import { embeddingDownloadNoticeMessage } from "./embeddingDownloadNotice";

describe("embeddingDownloadNoticeMessage", () => {
  it("says the model is downloading, its size, and that keywords are used meanwhile", () => {
    expect(embeddingDownloadNoticeMessage({ state: "downloading", size_gb: 2.24 })).toBe(
      "Downloading the memory search model (2.2 GB). Memory search uses keywords until it finishes.",
    );
  });

  it.each([[null], [0], [-1], [Number.NaN]])("leaves the size out when it is %j", (size_gb) => {
    expect(embeddingDownloadNoticeMessage({ state: "downloading", size_gb })).toBe(
      "Downloading the memory search model. Memory search uses keywords until it finishes.",
    );
  });

  it("says the download failed and will be tried again", () => {
    expect(embeddingDownloadNoticeMessage({ state: "failed", size_gb: 2.24 })).toBe(
      "The memory search model could not be downloaded. Memory search uses keywords only, and the download will be tried again.",
    );
  });
});

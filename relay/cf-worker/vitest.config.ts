import { defineWorkersConfig } from "@cloudflare/vitest-pool-workers/config";

export default defineWorkersConfig({
  test: {
    reporters: ["default", ["tdd-guard-vitest"]],
    poolOptions: {
      workers: {
        wrangler: { configPath: "./wrangler.toml" },
        miniflare: { d1Databases: ["DB"] },
      },
    },
  },
});

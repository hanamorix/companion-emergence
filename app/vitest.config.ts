import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

// The expression art lives in <repo>/expressions, one level above this app folder; src/expressions.ts loads it
// with import.meta.glob. Vite refuses to serve a file outside its allowed roots ("Denied ID"), so a test that
// imports the real expressions module needs the repository root allowed.
const repoRoot = fileURLToPath(new URL("..", import.meta.url));

export default defineConfig({
  plugins: [react()],
  server: { fs: { allow: [repoRoot] } },
  test: {
    environment: "jsdom",
    globals: false,
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
    reporters: ["default", ["tdd-guard-vitest"]],
  },
});

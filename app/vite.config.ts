import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Tauri exposes the dev server's host through the TAURI_DEV_HOST env var
const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  plugins: [react()],
  // Vite options tailored for Tauri development
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? { protocol: "ws", host, port: 1421 }
      : undefined,
    watch: {
      // Don't watch the Rust crate
      ignored: ["**/src-tauri/**"],
    },
  },
});

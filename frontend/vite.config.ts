import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig(({ mode }) => {
  // Read backend/.env so PORT is configured in one place.
  // Empty prefix → load all keys (not just VITE_*).
  const backendEnv = loadEnv(mode, path.resolve(__dirname, "../backend"), "");
  const backendPort = backendEnv.BACKEND_PORT || "8000";
  const backendHost = backendEnv.BACKEND_HOST || "127.0.0.1";
  const target = `http://${backendHost}:${backendPort}`;

  return {
    plugins: [react()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    server: {
      proxy: {
        "/api": {
          target,
          changeOrigin: true,
          ws: true,
          configure: (proxy) => {
            proxy.on("error", (err, req, res) => {
              const code = (err as NodeJS.ErrnoException).code ?? err.message;
              console.warn(`[vite proxy] ${req.method} ${req.url} → ${code}`);
              if (res && "writeHead" in res && !res.headersSent) {
                try {
                  res.writeHead(502, { "Content-Type": "text/plain" });
                  res.end(`backend unreachable (${code})`);
                } catch {}
              }
            });
          },
        },
      },
    },
  };
});

import { useState, useEffect } from "react";
import { api, type ModelStatus } from "../lib/api";
import { connectWs } from "../lib/ws";

export function useModelStatus() {
  const [models, setModels] = useState<ModelStatus[]>([]);

  useEffect(() => {
    api.listModels().then(setModels).catch(() => {});

    const disconnect = connectWs("/ws/models", (data) => {
      const ev = data as {
        name: string;
        status: string;
        bytes_downloaded: number;
        bytes_total: number;
        attempt?: number;
        note?: string;
      };
      setModels((prev) =>
        prev.map((m) =>
          m.name === ev.name
            ? {
                ...m,
                status: ev.status,
                bytes_downloaded: ev.bytes_downloaded,
                bytes_total: ev.bytes_total,
                attempt: ev.attempt,
                note: ev.note,
              }
            : m
        )
      );
    });

    // Also poll to pick up models that were already downloading
    const interval = setInterval(() => {
      api.listModels().then(setModels).catch(() => {});
    }, 10000);

    return () => {
      disconnect();
      clearInterval(interval);
    };
  }, []);

  return { models };
}

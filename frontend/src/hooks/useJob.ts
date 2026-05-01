import { useState, useEffect, useRef } from "react";
import { api, type JobDetail } from "../lib/api";
import { connectWs } from "../lib/ws";

export function useJob(jobId: string | undefined) {
  const [job, setJob] = useState<JobDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!jobId) return;

    let wsActive = false;

    // Fetch initial state
    api.getJob(jobId).then(setJob).catch((e) => setError(String(e)));

    // Connect WebSocket
    const disconnect = connectWs(
      `/ws/jobs/${jobId}`,
      (data) => {
        wsActive = true;
        setJob((prev) => {
          if (!prev) return prev;
          const ev = data as Partial<JobDetail>;
          return { ...prev, ...ev };
        });
      },
      () => { wsActive = false; }
    );

    // Fallback polling when WS is down
    pollRef.current = setInterval(async () => {
      if (wsActive) return;
      try {
        const j = await api.getJob(jobId);
        setJob(j);
      } catch {}
    }, 3000);

    return () => {
      disconnect();
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [jobId]);

  return { job, error };
}

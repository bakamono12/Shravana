export type WsHandler = (data: unknown) => void;

export function connectWs(path: string, onMessage: WsHandler, onClose?: () => void): () => void {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.host;
  let ws: WebSocket;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let closed = false;
  let attempt = 0;

  function connect() {
    ws = new WebSocket(`${protocol}://${host}/api${path}`);
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data?.type !== "ping") onMessage(data);
      } catch {}
    };
    ws.onclose = () => {
      if (closed) return;
      attempt++;
      const delay = Math.min(1000 * 2 ** attempt, 30000);
      reconnectTimer = setTimeout(connect, delay);
      onClose?.();
    };
    ws.onerror = () => ws.close();
  }

  connect();

  return () => {
    closed = true;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    ws?.close();
  };
}

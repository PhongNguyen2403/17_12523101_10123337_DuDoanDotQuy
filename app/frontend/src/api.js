const API_URL = import.meta.env.VITE_API_URL || "";

function requestId() {
  return globalThis.crypto?.randomUUID?.() || `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function handle(resp) {
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    const err = new Error(body.message || `Lỗi ${resp.status}`);
    err.details = body;
    throw err;
  }
  return resp.json();
}

export async function fetchSchema() {
  return handle(await fetch(`${API_URL}/api/schema`));
}

export async function fetchModelInfo() {
  return handle(await fetch(`${API_URL}/api/model-info`));
}

export async function submitPrediction(payload) {
  return handle(
    await fetch(`${API_URL}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Request-ID": requestId() },
      body: JSON.stringify(payload),
    })
  );
}

export async function fetchHistory(limit = 10) {
  return handle(await fetch(`${API_URL}/api/history?limit=${limit}`));
}

export async function fetchHealth() {
  return handle(await fetch(`${API_URL}/health`));
}

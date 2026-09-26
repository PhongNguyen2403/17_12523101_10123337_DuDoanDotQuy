const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

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

export async function submitPrediction(payload) {
  return handle(
    await fetch(`${API_URL}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  );
}

export async function fetchHistory(limit = 10) {
  return handle(await fetch(`${API_URL}/api/predictions?limit=${limit}`));
}

export async function fetchHealth() {
  return handle(await fetch(`${API_URL}/health`));
}

export function ragBackendUrl() {
  return (process.env.RAG_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

export function ragApiKey() {
  const key = process.env.RAG_API_KEY?.trim() || process.env.RAG_API_KEYS?.split(",")[0]?.trim();
  if (!key) {
    throw new Error("RAG_API_KEY is not configured on the server.");
  }
  return key;
}

export async function proxyToRag(path: string, init: RequestInit = {}) {
  const headers = new Headers(init.headers);
  headers.set("X-API-Key", ragApiKey());
  if (!(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${ragBackendUrl()}${path}`, {
    ...init,
    headers,
    cache: "no-store"
  });

  const text = await response.text();
  let payload: unknown = text;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = text;
  }

  return { response, payload };
}

export function errorMessage(payload: unknown, status: number) {
  if (typeof payload === "string" && payload.trim()) return payload;
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    return JSON.stringify(detail);
  }
  return `Request failed with ${status}`;
}

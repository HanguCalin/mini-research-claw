const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed with ${response.status}`);
  }

  return response.json();
}

export function createPipelineRun(topic) {
  return request("/runs", {
    method: "POST",
    body: JSON.stringify({ topic }),
  });
}

export function getPipelineRun(runId) {
  return request(`/runs/${runId}`);
}

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

export function createPipelineRun(topic, overrides = {}) {
  // Strip null/undefined/empty so the server only sees explicit overrides.
  const body = { topic };
  if (overrides.maxCodeRetries != null && overrides.maxCodeRetries !== "") {
    body.max_code_retries = Number(overrides.maxCodeRetries);
  }
  if (overrides.arxivResultsPerRound != null && overrides.arxivResultsPerRound !== "") {
    body.arxiv_results_per_round = Number(overrides.arxivResultsPerRound);
  }
  if (overrides.modelOverride) {
    body.model_override = overrides.modelOverride;
  }
  return request("/runs", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function getPipelineRun(runId) {
  return request(`/runs/${runId}`);
}

export function getPendingGate(runId) {
  return request(`/runs/${runId}/pending-gate`);
}

export function submitGateDecision(runId, { gateId, action, reason = "" }) {
  return request(`/runs/${runId}/gate-decision`, {
    method: "POST",
    body: JSON.stringify({ gate_id: gateId, action, reason }),
  });
}

export function listArtifacts(runId) {
  return request(`/runs/${runId}/artifacts`);
}

// URL the browser can hit directly via <a download> to trigger a file save.
// Goes through the same Vite/nginx /api proxy as the rest of the calls.
export function artifactDownloadUrl(runId, filename) {
  return `${API_BASE_URL}/runs/${runId}/artifacts/${encodeURIComponent(filename)}`;
}

import { useState } from "react";
import { CheckCircle2, XCircle, AlertCircle, FileText, Network } from "lucide-react";

/* ─── Helpers ─────────────────────────────────────────────────────────────── */

function PolarityBadge({ polarity }) {
  const styles = {
    supports: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30",
    contradicts: "bg-rose-500/15 text-rose-300 ring-rose-500/30",
    neutral: "bg-white/5 text-text-muted ring-white/10",
  };
  return (
    <span
      className={`rounded-md px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide ring-1 ${
        styles[polarity] || styles.neutral
      }`}
    >
      {polarity}
    </span>
  );
}

function StatusPill({ ok, label }) {
  return (
    <span
      className={`rounded-md px-2 py-0.5 text-[11px] font-semibold uppercase ring-1 ${
        ok
          ? "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30"
          : "bg-rose-500/15 text-rose-300 ring-rose-500/30"
      }`}
    >
      {label}
    </span>
  );
}

function SectionTitle({ icon: Icon, children }) {
  return (
    <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-text-secondary">
      {Icon && <Icon size={14} strokeWidth={1.8} className="text-accent" />}
      {children}
    </div>
  );
}

/* ─── Gate 1 — Hypothesis Approval ────────────────────────────────────────── */

function HypothesisGateBody({ payload }) {
  const { hypothesis, incremental_delta, novelty, prior_art, kg_triples, papers, paper_total } =
    payload;
  return (
    <div className="space-y-5">
      <section className="space-y-2">
        <SectionTitle icon={FileText}>Generated Hypothesis</SectionTitle>
        <p className="rounded-lg border border-white/8 bg-black/30 p-4 text-sm leading-relaxed text-text-primary">
          {hypothesis}
        </p>
      </section>

      <section className="space-y-2">
        <SectionTitle>Incremental Delta (what&rsquo;s new)</SectionTitle>
        <p className="rounded-lg border border-white/8 bg-black/30 p-4 text-sm leading-relaxed text-text-secondary">
          {incremental_delta}
        </p>
      </section>

      <section className="space-y-2">
        <SectionTitle>Novelty Scores</SectionTitle>
        <div className="overflow-hidden rounded-lg border border-white/8">
          <table className="w-full text-sm">
            <thead className="bg-white/5 text-left text-[11px] uppercase tracking-wide text-text-muted">
              <tr>
                <th className="px-3 py-2">Metric</th>
                <th className="px-3 py-2">Value</th>
                <th className="px-3 py-2">Threshold</th>
                <th className="px-3 py-2">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              <tr>
                <td className="px-3 py-2 font-medium">Novelty (RND)</td>
                <td className="px-3 py-2 font-mono">{novelty.score.toFixed(3)}</td>
                <td className="px-3 py-2 font-mono text-text-muted">
                  &ge; {novelty.threshold}
                </td>
                <td className="px-3 py-2">
                  <StatusPill ok={novelty.pass} label={novelty.pass ? "Pass" : "Fail"} />
                </td>
              </tr>
              <tr>
                <td className="px-3 py-2 font-medium">Prior-Art Similarity</td>
                <td className="px-3 py-2 font-mono">{prior_art.similarity.toFixed(3)}</td>
                <td className="px-3 py-2 font-mono text-text-muted">
                  &lt; {prior_art.ceiling}
                </td>
                <td className="px-3 py-2">
                  <StatusPill ok={prior_art.pass} label={prior_art.pass ? "Pass" : "Fail"} />
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {kg_triples.length > 0 && (
        <section className="space-y-2">
          <SectionTitle icon={Network}>Key KG Triples</SectionTitle>
          <div className="overflow-hidden rounded-lg border border-white/8">
            <table className="w-full text-sm">
              <thead className="bg-white/5 text-left text-[11px] uppercase tracking-wide text-text-muted">
                <tr>
                  <th className="px-3 py-2">Source</th>
                  <th className="px-3 py-2">Relation</th>
                  <th className="px-3 py-2">Target</th>
                  <th className="px-3 py-2">Polarity</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {kg_triples.map((edge, i) => (
                  <tr key={i}>
                    <td className="px-3 py-2">{edge.source}</td>
                    <td className="px-3 py-2 font-mono text-text-secondary">{edge.relation}</td>
                    <td className="px-3 py-2">{edge.target}</td>
                    <td className="px-3 py-2">
                      <PolarityBadge polarity={edge.polarity} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {papers.length > 0 && (
        <section className="space-y-2">
          <SectionTitle>Retrieved Papers ({paper_total})</SectionTitle>
          <div className="overflow-hidden rounded-lg border border-white/8">
            <table className="w-full text-sm">
              <thead className="bg-white/5 text-left text-[11px] uppercase tracking-wide text-text-muted">
                <tr>
                  <th className="px-3 py-2">ArXiv ID</th>
                  <th className="px-3 py-2">Title</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {papers.map((p, i) => (
                  <tr key={i}>
                    <td className="px-3 py-2 font-mono text-text-secondary">{p.arxiv_id}</td>
                    <td className="px-3 py-2">{p.title}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}

/* ─── Gate 2 — ExperimentSpec Approval ────────────────────────────────────── */

function ExperimentGateBody({ payload }) {
  const { hypothesis, incremental_delta, spec, kg_edges } = payload;
  const specRows = [
    ["Independent Variable", spec.independent_var],
    ["Dependent Variable", spec.dependent_var],
    ["Control Description", spec.control_description],
    ["Dataset ID", spec.dataset_id],
    ["Evaluation Metrics", spec.evaluation_metrics.join(", ")],
    ["Expected Outcome", spec.expected_outcome],
  ];
  return (
    <div className="space-y-5">
      <section className="space-y-2">
        <SectionTitle icon={FileText}>Hypothesis</SectionTitle>
        <p className="rounded-lg border border-white/8 bg-black/30 p-4 text-sm leading-relaxed text-text-primary">
          {hypothesis}
        </p>
      </section>

      <section className="space-y-2">
        <SectionTitle>Incremental Delta</SectionTitle>
        <p className="rounded-lg border border-white/8 bg-black/30 p-4 text-sm leading-relaxed text-text-secondary">
          {incremental_delta}
        </p>
      </section>

      <section className="space-y-2">
        <SectionTitle>Experiment Specification</SectionTitle>
        <div className="overflow-hidden rounded-lg border border-white/8">
          <table className="w-full text-sm">
            <tbody className="divide-y divide-white/5">
              {specRows.map(([field, value]) => (
                <tr key={field}>
                  <td className="w-56 bg-white/5 px-3 py-2 text-[12px] font-semibold uppercase tracking-wide text-text-secondary">
                    {field}
                  </td>
                  <td className="px-3 py-2 text-text-primary">{value || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {kg_edges.length > 0 && (
        <section className="space-y-2">
          <SectionTitle icon={Network}>Relevant KG Edges</SectionTitle>
          <div className="overflow-hidden rounded-lg border border-white/8">
            <table className="w-full text-sm">
              <thead className="bg-white/5 text-left text-[11px] uppercase tracking-wide text-text-muted">
                <tr>
                  <th className="px-3 py-2">Source</th>
                  <th className="px-3 py-2">Relation</th>
                  <th className="px-3 py-2">Target</th>
                  <th className="px-3 py-2">Polarity</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {kg_edges.map((edge, i) => (
                  <tr key={i}>
                    <td className="px-3 py-2">{edge.source}</td>
                    <td className="px-3 py-2 font-mono text-text-secondary">{edge.relation}</td>
                    <td className="px-3 py-2">{edge.target}</td>
                    <td className="px-3 py-2">
                      <PolarityBadge polarity={edge.polarity} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}

/* ─── Outer panel ─────────────────────────────────────────────────────────── */

const GATE_META = {
  hypothesis: {
    title: "HITL Gate 1 — Hypothesis Approval",
    blurb:
      "The pipeline has paused for your review. Approve to proceed to experiment design, or reject with a reason to regenerate.",
    rejectPlaceholder:
      "Why are you rejecting? Be specific — the feedback steers the regenerator (e.g. 'wrong model class', 'dataset not available offline').",
  },
  experiment: {
    title: "HITL Gate 2 — Experiment Approval",
    blurb:
      "Review the ExperimentSpec before any code is written. Approve to run, reject for redesign, or type 'abort' as the reason to terminate.",
    rejectPlaceholder:
      "Reason (or type 'abort' to terminate the run instead of redesigning the experiment)",
  },
};

export default function GatePanel({ gate, onSubmit, busy }) {
  const [showReject, setShowReject] = useState(false);
  const [reason, setReason] = useState("");
  const meta = GATE_META[gate.gate_id] || {
    title: `HITL Gate (${gate.gate_id})`,
    blurb: "Awaiting decision.",
    rejectPlaceholder: "Reason",
  };

  function approve() {
    onSubmit({ action: "approve", reason: "" });
  }
  function reject() {
    onSubmit({ action: "reject", reason: reason.trim() || "No reason provided" });
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto bg-black/70 backdrop-blur-sm">
      <div className="my-8 w-full max-w-4xl rounded-xl border border-accent/30 bg-bg-deep shadow-2xl">
        <header className="flex items-start gap-3 border-b border-white/8 p-5">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-accent-dim ring-1 ring-accent/30">
            <AlertCircle size={20} strokeWidth={1.8} className="text-accent" />
          </div>
          <div className="flex-1">
            <h2 className="text-lg font-semibold text-text-primary">{meta.title}</h2>
            <p className="mt-1 text-sm text-text-secondary">{meta.blurb}</p>
          </div>
        </header>

        <div className="max-h-[60vh] overflow-y-auto p-5">
          {gate.gate_id === "hypothesis" ? (
            <HypothesisGateBody payload={gate.payload} />
          ) : (
            <ExperimentGateBody payload={gate.payload} />
          )}
        </div>

        <footer className="space-y-3 border-t border-white/8 p-5">
          {showReject && (
            <textarea
              autoFocus
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              rows={3}
              placeholder={meta.rejectPlaceholder}
              className="w-full resize-none rounded-lg border border-white/10 bg-black/40 p-3 text-sm text-text-primary placeholder:text-text-muted focus:border-accent/60 focus:outline-none"
            />
          )}
          <div className="flex flex-wrap justify-end gap-2">
            {!showReject ? (
              <>
                <button
                  type="button"
                  onClick={() => setShowReject(true)}
                  disabled={busy}
                  className="inline-flex items-center gap-2 rounded-lg border border-rose-500/40 bg-rose-500/10 px-4 py-2 text-sm font-medium text-rose-200 transition hover:bg-rose-500/20 disabled:opacity-50"
                >
                  <XCircle size={16} strokeWidth={1.8} /> Reject
                </button>
                <button
                  type="button"
                  onClick={approve}
                  disabled={busy}
                  className="inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-black transition hover:bg-accent/90 disabled:opacity-50"
                >
                  <CheckCircle2 size={16} strokeWidth={1.8} /> Approve
                </button>
              </>
            ) : (
              <>
                <button
                  type="button"
                  onClick={() => {
                    setShowReject(false);
                    setReason("");
                  }}
                  disabled={busy}
                  className="rounded-lg border border-white/10 bg-black/30 px-4 py-2 text-sm text-text-secondary transition hover:bg-black/50 disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={reject}
                  disabled={busy}
                  className="inline-flex items-center gap-2 rounded-lg bg-rose-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-rose-500/90 disabled:opacity-50"
                >
                  <XCircle size={16} strokeWidth={1.8} /> Confirm Reject
                </button>
              </>
            )}
          </div>
        </footer>
      </div>
    </div>
  );
}

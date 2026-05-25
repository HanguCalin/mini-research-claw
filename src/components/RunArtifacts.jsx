import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FileText,
  RefreshCw,
} from "lucide-react";
import { artifactDownloadUrl, listArtifacts } from "../services/api";

/* Human-readable labels for each artifact the uploader produces. */
const FILE_LABELS = {
  "final_paper.pdf":      "Compiled paper (NeurIPS-formatted PDF)",
  "draft.tex":            "LaTeX source the compiler ran",
  "references.bib":       "BibTeX references cited in the paper",
  "metrics.json":         "Experiment metrics and hyperparameters",
  "claim_ledger.json":    "Every paper claim mapped to KG evidence",
  "debate_log.json":      "Critique panel debate log (challenges + retractions)",
  "python_code.py":       "The script the ML Coder generated and the sandbox ran",
  "execution_logs.txt":   "Full sandbox stdout/stderr",
  "hypothesis.txt":       "The hypothesis the run was testing",
  "experiment_spec.json": "ExperimentSpec — IV, DV, dataset, metrics",
  "failure_report.json":  "Terminal status, retry counts, last 4KB of logs",
};

/* Recommended display order, per variant. */
const PRIORITY = {
  success: [
    "final_paper.pdf",
    "draft.tex",
    "references.bib",
    "metrics.json",
    "claim_ledger.json",
    "debate_log.json",
    "python_code.py",
    "execution_logs.txt",
    "hypothesis.txt",
    "experiment_spec.json",
    "failure_report.json",
  ],
  failure: [
    "failure_report.json",
    "execution_logs.txt",
    "python_code.py",
    "hypothesis.txt",
    "experiment_spec.json",
    "metrics.json",
    "claim_ledger.json",
    "debate_log.json",
    "draft.tex",
    "references.bib",
    "final_paper.pdf",
  ],
};

/* Per-variant chrome: colours, icon, default title, default intro. */
const VARIANT_STYLES = {
  success: {
    Icon: CheckCircle2,
    panel: "border-emerald-500/30 bg-emerald-500/5",
    headerBorder: "border-emerald-500/20",
    iconWrap: "bg-emerald-500/15 ring-emerald-500/30",
    iconText: "text-emerald-300",
    title: "Run complete — download paper & artifacts",
    intro:
      "The pipeline finished. The compiled PDF, LaTeX source, metrics, and supporting evidence are below.",
  },
  failure: {
    Icon: AlertTriangle,
    panel: "border-rose-500/30 bg-rose-500/5",
    headerBorder: "border-rose-500/20",
    iconWrap: "bg-rose-500/15 ring-rose-500/30",
    iconText: "text-rose-300",
    title: "Run failed — download diagnostics",
    intro:
      "The pipeline terminated before producing a paper. The files below are everything the uploader managed to persist for this run.",
  },
};

function sortByPriority(names, variant) {
  const order = new Map(PRIORITY[variant].map((n, i) => [n, i]));
  return [...names].sort(
    (a, b) => (order.get(a) ?? 999) - (order.get(b) ?? 999),
  );
}

export default function RunArtifacts({ runId, variant = "success", error }) {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(null);

  const styles = VARIANT_STYLES[variant] || VARIANT_STYLES.success;
  const { Icon } = styles;

  const refresh = () => {
    if (!runId) return;
    setLoading(true);
    setLoadError(null);
    listArtifacts(runId)
      .then((r) => setFiles(sortByPriority(r.files || [], variant)))
      .catch((e) => setLoadError(e.message || "Failed to list artifacts"))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, variant]);

  if (!runId) return null;

  return (
    <div className={`panel-raised mt-4 rounded-lg border ${styles.panel}`}>
      <header className={`flex items-start gap-3 border-b ${styles.headerBorder} p-5`}>
        <div
          className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ring-1 ${styles.iconWrap}`}
        >
          <Icon size={20} strokeWidth={1.8} className={styles.iconText} />
        </div>
        <div className="flex-1">
          <h3 className="text-base font-semibold text-text-primary">
            {styles.title}
          </h3>
          <p className="mt-1 text-sm text-text-secondary">
            {error || styles.intro}
          </p>
          <p className="mt-1 text-[11px] font-mono text-text-muted">
            run_id: {runId}
          </p>
        </div>
        <button
          type="button"
          onClick={refresh}
          className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 bg-black/30 text-text-secondary transition hover:bg-black/50"
          title="Refresh artifact list"
        >
          <RefreshCw
            size={14}
            strokeWidth={1.8}
            className={loading ? "animate-spin" : ""}
          />
        </button>
      </header>

      <div className="p-5">
        {loadError && (
          <p className="rounded-md border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-200">
            {loadError}
          </p>
        )}

        {!loadError && !loading && files.length === 0 && (
          <p className="text-sm text-text-muted">
            No artifacts were uploaded for this run. The pipeline likely
            crashed before reaching the upload step.
          </p>
        )}

        {files.length > 0 && (
          <ul className="space-y-2">
            {files.map((name, idx) => {
              // Highlight the most useful download for each variant: the PDF
              // on success, the failure report on failure.
              const isPrimary = idx === 0;
              return (
                <li
                  key={name}
                  className={`flex items-center gap-3 rounded-lg border p-3 ${
                    isPrimary
                      ? "border-accent/30 bg-accent/5"
                      : "border-white/8 bg-black/20"
                  }`}
                >
                  <FileText
                    size={16}
                    strokeWidth={1.8}
                    className={`shrink-0 ${
                      isPrimary ? "text-accent" : "text-text-secondary"
                    }`}
                  />
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-mono text-sm text-text-primary">
                      {name}
                    </p>
                    {FILE_LABELS[name] && (
                      <p className="mt-0.5 text-[12px] text-text-muted">
                        {FILE_LABELS[name]}
                      </p>
                    )}
                  </div>
                  <a
                    href={artifactDownloadUrl(runId, name)}
                    download={name}
                    className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold ring-1 transition ${
                      isPrimary
                        ? "bg-accent text-black ring-accent hover:brightness-110"
                        : "bg-accent/20 text-accent ring-accent/30 hover:bg-accent/30"
                    }`}
                  >
                    <Download size={13} strokeWidth={2} />
                    Download
                  </a>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}

import { CheckCircle2, XCircle, Clock, ExternalLink } from "lucide-react";

const mockRuns = [
  {
    id: "run-007",
    topic: "Compare Random Forest vs Logistic Regression on Iris dataset",
    status: "success",
    duration: "29s",
    date: "2026-03-23 14:32",
    papers: 5,
    retries: 0,
  },
  {
    id: "run-006",
    topic: "Benchmark neural network pruning techniques on CIFAR-10",
    status: "success",
    duration: "42s",
    date: "2026-03-22 10:15",
    papers: 4,
    retries: 1,
  },
  {
    id: "run-005",
    topic: "Evaluate sentiment analysis: transformer vs RNN",
    status: "failed",
    duration: "58s",
    date: "2026-03-21 16:44",
    papers: 3,
    retries: 3,
  },
  {
    id: "run-004",
    topic: "K-means vs DBSCAN clustering on synthetic data",
    status: "success",
    duration: "31s",
    date: "2026-03-20 09:22",
    papers: 6,
    retries: 0,
  },
  {
    id: "run-003",
    topic: "Transfer learning efficiency on small medical imaging datasets",
    status: "success",
    duration: "37s",
    date: "2026-03-19 11:05",
    papers: 5,
    retries: 1,
  },
];

const statusConfig = {
  success: {
    icon: CheckCircle2,
    class: "text-success",
    bg: "bg-success-dim",
    label: "Success",
  },
  failed: {
    icon: XCircle,
    class: "text-error",
    bg: "bg-error-dim",
    label: "Failed",
  },
};

export default function History() {
  return (
    <div className="space-y-8 animate-slide-in">
      <div>
        <p className="section-kicker">Archive</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">
          History
        </h1>
        <p className="mt-2 text-sm leading-7 text-text-secondary">
          Browse past research pipeline runs.
        </p>
      </div>

      {/* Runs table */}
      <div className="panel-soft panel-cyber overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-black/10 text-xs uppercase tracking-wider text-text-muted">
              <th className="text-left px-5 py-3 font-medium">Status</th>
              <th className="text-left px-5 py-3 font-medium">Topic</th>
              <th className="text-left px-5 py-3 font-medium">Papers</th>
              <th className="text-left px-5 py-3 font-medium">Retries</th>
              <th className="text-left px-5 py-3 font-medium">Duration</th>
              <th className="text-left px-5 py-3 font-medium">Date</th>
              <th className="px-5 py-3" />
            </tr>
          </thead>
          <tbody>
            {mockRuns.map((run) => {
              const s = statusConfig[run.status];
              const Icon = s.icon;
              return (
                <tr
                  key={run.id}
                  className="border-b border-border last:border-0 transition-colors hover:bg-white/3"
                >
                  <td className="px-5 py-4">
                    <span
                      className={`inline-flex items-center gap-1.5 text-xs font-mono font-medium px-2.5 py-1 rounded-full ${s.bg} ${s.class}`}
                    >
                      <Icon size={12} />
                      {s.label}
                    </span>
                  </td>
                  <td className="px-5 py-4 text-text-primary max-w-xs truncate">
                    {run.topic}
                  </td>
                  <td className="px-5 py-4 text-text-secondary font-mono">
                    {run.papers}
                  </td>
                  <td className="px-5 py-4 font-mono">
                    <span
                      className={
                        run.retries > 0 ? "text-warning" : "text-text-muted"
                      }
                    >
                      {run.retries}
                    </span>
                  </td>
                  <td className="px-5 py-4 text-text-secondary font-mono flex items-center gap-1.5">
                    <Clock size={12} className="text-text-muted" />
                    {run.duration}
                  </td>
                  <td className="px-5 py-4 text-text-muted font-mono text-xs">
                    {run.date}
                  </td>
                  <td className="px-5 py-4">
                    <button className="text-text-muted hover:text-accent transition-colors cursor-pointer">
                      <ExternalLink size={14} />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

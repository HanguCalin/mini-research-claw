import { Terminal } from "lucide-react";

const sampleLogs = [
  { time: "14:32:01", level: "info", agent: "system", msg: "Pipeline initialized for topic: 'Random Forest vs Logistic Regression on Iris'" },
  { time: "14:32:02", level: "info", agent: "researcher", msg: "Searching arXiv for relevant papers..." },
  { time: "14:32:05", level: "info", agent: "researcher", msg: "Found 5 papers. Synthesizing hypothesis..." },
  { time: "14:32:08", level: "success", agent: "researcher", msg: "Hypothesis generated: 'Random Forest outperforms LR on Iris by ≥5% accuracy'" },
  { time: "14:32:09", level: "info", agent: "coder", msg: "Generating experiment script..." },
  { time: "14:32:14", level: "success", agent: "coder", msg: "Script generated (47 lines). Passing to executor." },
  { time: "14:32:15", level: "info", agent: "executor", msg: "Spinning up Docker sandbox..." },
  { time: "14:32:18", level: "info", agent: "executor", msg: "Running experiment... (timeout: 60s)" },
  { time: "14:32:23", level: "success", agent: "executor", msg: "Execution complete. Exit code: 0" },
  { time: "14:32:24", level: "info", agent: "writer", msg: "Composing research paper..." },
  { time: "14:32:30", level: "success", agent: "writer", msg: "Paper saved → output/research_20260323_143230.md" },
  { time: "14:32:30", level: "info", agent: "system", msg: "Pipeline complete. Duration: 29s" },
];

const levelColors = {
  info: "text-text-secondary",
  success: "text-success",
  warn: "text-warning",
  error: "text-error",
};

export default function LogPanel({ logs = sampleLogs, maxHeight = "400px" }) {
  return (
    <div className="rounded-xl border border-border bg-surface-1 overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border bg-surface-2">
        <Terminal size={14} className="text-text-muted" />
        <span className="text-xs font-mono text-text-muted">
          pipeline output
        </span>
        <div className="ml-auto flex gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-error/60" />
          <div className="w-2.5 h-2.5 rounded-full bg-warning/60" />
          <div className="w-2.5 h-2.5 rounded-full bg-success/60" />
        </div>
      </div>

      {/* Log content */}
      <div
        className="p-4 overflow-y-auto font-mono text-xs leading-relaxed"
        style={{ maxHeight }}
      >
        {logs.map((log, i) => (
          <div
            key={i}
            className="flex gap-3 py-0.5 animate-slide-in"
            style={{ animationDelay: `${i * 40}ms` }}
          >
            <span className="text-text-muted shrink-0">{log.time}</span>
            <span className="text-accent shrink-0 w-24 text-right">
              [{log.agent}]
            </span>
            <span className={levelColors[log.level] || "text-text-secondary"}>
              {log.msg}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

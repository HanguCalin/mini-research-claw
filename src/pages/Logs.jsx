import { useState } from "react";
import LogPanel from "../components/LogPanel";
import { Filter } from "lucide-react";

const allLogs = [
  { time: "14:32:01", level: "info", agent: "system", msg: "Pipeline initialized for topic: 'Random Forest vs Logistic Regression on Iris'" },
  { time: "14:32:02", level: "info", agent: "researcher", msg: "Searching arXiv for relevant papers..." },
  { time: "14:32:03", level: "info", agent: "researcher", msg: "Query: 'random forest logistic regression iris classification'" },
  { time: "14:32:05", level: "info", agent: "researcher", msg: "Found 5 papers. Synthesizing hypothesis..." },
  { time: "14:32:06", level: "info", agent: "researcher", msg: "API call: claude-3-haiku — 1,240 input tokens, 380 output tokens" },
  { time: "14:32:08", level: "success", agent: "researcher", msg: "Hypothesis generated: 'Random Forest outperforms LR on Iris by ≥5% accuracy'" },
  { time: "14:32:09", level: "info", agent: "coder", msg: "Generating experiment script..." },
  { time: "14:32:10", level: "info", agent: "coder", msg: "API call: claude-3.7-sonnet — 2,100 input tokens, 890 output tokens" },
  { time: "14:32:14", level: "success", agent: "coder", msg: "Script generated (47 lines). Passing to executor." },
  { time: "14:32:15", level: "info", agent: "executor", msg: "Spinning up Docker sandbox..." },
  { time: "14:32:16", level: "info", agent: "executor", msg: "Container ID: a3f8c2d1 — Image: research-sandbox:latest" },
  { time: "14:32:18", level: "info", agent: "executor", msg: "Running experiment... (timeout: 60s)" },
  { time: "14:32:23", level: "success", agent: "executor", msg: "Execution complete. Exit code: 0" },
  { time: "14:32:23", level: "info", agent: "executor", msg: "Results: RF accuracy=0.967, LR accuracy=0.933 — hypothesis supported" },
  { time: "14:32:24", level: "info", agent: "writer", msg: "Composing research paper..." },
  { time: "14:32:25", level: "info", agent: "writer", msg: "API call: claude-3.7-sonnet — 3,400 input tokens, 1,600 output tokens" },
  { time: "14:32:30", level: "success", agent: "writer", msg: "Paper saved → output/research_20260323_143230.md" },
  { time: "14:32:30", level: "info", agent: "system", msg: "Pipeline complete. Duration: 29s. Total API cost: $0.018" },
];

const agents = ["all", "system", "researcher", "coder", "executor", "writer"];

export default function Logs() {
  const [filter, setFilter] = useState("all");

  const filtered =
    filter === "all" ? allLogs : allLogs.filter((l) => l.agent === filter);

  return (
    <div className="space-y-8 animate-slide-in">
      <div>
        <p className="section-kicker">Trace View</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">
          Logs
        </h1>
        <p className="mt-2 text-sm leading-7 text-text-secondary">
          Full pipeline output with API call details.
        </p>
      </div>

      {/* Filters */}
      <div className="panel-soft panel-cyber flex flex-wrap items-center gap-3 px-4 py-3">
        <Filter size={14} className="text-text-muted" />
        {agents.map((a) => (
          <button
            key={a}
            onClick={() => setFilter(a)}
            className={`cursor-pointer rounded-full border px-3 py-1.5 text-xs font-mono transition-all ${
              filter === a
                ? "border-accent bg-accent-dim text-accent shadow-[0_0_24px_rgba(103,232,249,0.14)]"
                : "border-border bg-black/10 text-text-secondary hover:text-text-primary"
            }`}
          >
            {a}
          </button>
        ))}
      </div>

      <LogPanel logs={filtered} maxHeight="600px" />
    </div>
  );
}

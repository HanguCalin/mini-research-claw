import {
  Search,
  Code,
  Play,
  FileText,
  ArrowRight,
  RotateCcw,
  BrainCircuit,
  ClipboardCheck,
  DatabaseZap,
  Network,
} from "lucide-react";
import { pipelineStages } from "../data/pipelineStages";

const statusStyles = {
  idle: {
    ring: "border-border",
    bg: "bg-surface-2",
    text: "text-text-muted",
    badge: "bg-surface-3 text-text-muted",
    label: "Idle",
  },
  running: {
    ring: "border-accent glow-active",
    bg: "bg-accent-dim",
    text: "text-accent",
    badge: "bg-accent-dim text-accent",
    label: "Running",
  },
  success: {
    ring: "border-success",
    bg: "bg-success-dim",
    text: "text-success",
    badge: "bg-success-dim text-success",
    label: "Done",
  },
  failed: {
    ring: "border-error",
    bg: "bg-error-dim",
    text: "text-error",
    badge: "bg-error-dim text-error",
    label: "Failed",
  },
  retrying: {
    ring: "border-warning glow-active",
    bg: "bg-warning-dim",
    text: "text-warning",
    badge: "bg-warning-dim text-warning",
    label: "Retrying",
  },
};

const icons = {
  retriever: Search,
  kg: Network,
  hypothesis: BrainCircuit,
  design: ClipboardCheck,
  coder: Code,
  executor: Play,
  writer: FileText,
  compiler: DatabaseZap,
};

export default function PipelineStepper({ agentStates = {} }) {
  return (
    <div className="flex min-w-max items-stretch gap-3">
      {pipelineStages.map((agent, i) => {
        const status = agentStates[agent.id] || "idle";
        const s = statusStyles[status];
        const Icon = icons[agent.id] || Search;
        const isRetry = agent.id === "executor" && status === "retrying";

        return (
          <div key={agent.id} className="flex items-stretch gap-3">
            <div
              className={`relative flex w-44 flex-col gap-3 rounded-lg border p-4 transition-all duration-300 ${s.ring} ${s.bg}`}
              style={
                status === "running" || status === "retrying"
                  ? {
                      "--glow-color":
                        status === "retrying"
                          ? "var(--color-warning)"
                          : "var(--color-accent)",
                    }
                  : undefined
              }
            >
              <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/30 to-transparent" />
              <div className="flex items-center justify-between gap-3">
                <div
                  className={`flex h-10 w-10 items-center justify-center rounded-lg border ${s.ring} ${s.bg}`}
                >
                  {isRetry ? (
                    <RotateCcw size={18} className={s.text} />
                  ) : (
                    <Icon size={18} className={s.text} />
                  )}
                </div>
                <span
                  className={`rounded-full px-2.5 py-1 text-[10px] font-medium ${s.badge}`}
                >
                  {s.label}
                </span>
              </div>
              <div>
                <p className="text-sm font-semibold text-text-primary">
                  {agent.name}
                </p>
                <p className="mt-0.5 text-[11px] font-mono text-text-muted">
                  {agent.model}
                </p>
              </div>
              <p className="text-xs leading-relaxed text-text-muted">
                {agent.description}
              </p>
            </div>

            {i < pipelineStages.length - 1 && (
              <div className="flex items-center text-text-muted">
                <ArrowRight size={18} />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

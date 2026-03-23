import {
  Search,
  Code,
  Play,
  FileText,
  ArrowRight,
  RotateCcw,
} from "lucide-react";

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

const agents = [
  {
    id: "researcher",
    name: "Researcher",
    model: "Haiku",
    icon: Search,
    description: "Searching arXiv & forming hypothesis",
  },
  {
    id: "coder",
    name: "Coder",
    model: "Sonnet",
    icon: Code,
    description: "Generating experiment code",
  },
  {
    id: "executor",
    name: "Executor",
    model: "Sandbox",
    icon: Play,
    description: "Running code in Docker",
  },
  {
    id: "writer",
    name: "Writer",
    model: "Sonnet",
    icon: FileText,
    description: "Composing research paper",
  },
];

export default function PipelineStepper({ agentStates = {} }) {
  return (
    <div className="flex items-start gap-3">
      {agents.map((agent, i) => {
        const status = agentStates[agent.id] || "idle";
        const s = statusStyles[status];
        const Icon = agent.icon;
        const isRetry =
          agent.id === "executor" && status === "retrying";

        return (
          <div key={agent.id} className="flex items-start gap-3">
            {/* Agent Card */}
            <div
              className={`relative flex flex-col items-center gap-3 p-5 rounded-xl border-2 ${s.ring} ${s.bg} w-44 transition-all duration-300`}
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
              <div
                className={`w-11 h-11 rounded-full border ${s.ring} flex items-center justify-center ${s.bg}`}
              >
                {isRetry ? (
                  <RotateCcw size={20} className={s.text} />
                ) : (
                  <Icon size={20} className={s.text} />
                )}
              </div>
              <div className="text-center">
                <p className="text-sm font-semibold text-text-primary">
                  {agent.name}
                </p>
                <p className="text-[11px] font-mono text-text-muted mt-0.5">
                  {agent.model}
                </p>
              </div>
              <span
                className={`text-[10px] font-mono font-medium px-2.5 py-1 rounded-full ${s.badge}`}
              >
                {s.label}
              </span>
              <p className="text-xs text-text-muted text-center leading-relaxed">
                {agent.description}
              </p>
            </div>

            {/* Connector arrow */}
            {i < agents.length - 1 && (
              <div className="flex items-center pt-10 text-text-muted">
                <ArrowRight size={18} />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

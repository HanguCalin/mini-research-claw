import { useMemo, useState } from "react";
import {
  ArrowRight,
  CheckCircle2,
  CircleDot,
  Clock3,
  Code,
  DatabaseZap,
  FileText,
  Microscope,
  Network,
  Play,
  Search,
  Sparkles,
  X,
} from "lucide-react";
import { pipelineStages } from "../data/pipelineStages";

const icons = {
  retriever: Search,
  kg: Network,
  hypothesis: Sparkles,
  design: Microscope,
  coder: Code,
  executor: Play,
  writer: FileText,
  compiler: DatabaseZap,
};

const statusLabel = {
  idle: "Waiting",
  running: "Running",
  success: "Complete",
};

function statusFor(index, activeStageIndex, isRunning) {
  if (index < activeStageIndex) return "success";
  if (index === activeStageIndex && isRunning) return "running";
  if (!isRunning && activeStageIndex >= pipelineStages.length) return "success";
  return "idle";
}

export default function AgentTimeline({ activeStageIndex, isRunning }) {
  const [selectedStage, setSelectedStage] = useState(null);

  const selectedStatus = useMemo(() => {
    if (!selectedStage) return "idle";
    const index = pipelineStages.findIndex((stage) => stage.id === selectedStage.id);
    return statusFor(index, activeStageIndex, isRunning);
  }, [activeStageIndex, isRunning, selectedStage]);

  return (
    <>
      <div className="panel-raised overflow-hidden rounded-lg">
        <div className="flex flex-col gap-2 border-b border-border bg-black/10 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="section-kicker">Animated Agent Timeline</p>
            <h2 className="mt-1 text-lg font-semibold text-text-primary">
              Paper assembly line
            </h2>
          </div>
          <div className="text-xs font-mono text-text-muted">
            click any stage for details
          </div>
        </div>

        <div className="overflow-x-auto p-5">
          <div className="flex min-w-max items-center gap-3">
            {pipelineStages.map((stage, index) => {
              const status = statusFor(index, activeStageIndex, isRunning);
              const Icon = icons[stage.id] || CircleDot;
              const isActive = status === "running";
              const isComplete = status === "success";

              return (
                <div key={stage.id} className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={() => setSelectedStage(stage)}
                    className={`group relative flex w-52 cursor-pointer flex-col gap-4 overflow-hidden rounded-lg border p-4 text-left transition-all duration-300 hover:-translate-y-1 hover:border-border-bright ${
                      isActive
                        ? "border-accent bg-accent-dim shadow-[0_0_34px_rgba(97,216,255,0.16)]"
                        : isComplete
                          ? "border-success/45 bg-success-dim"
                          : "border-border bg-black/20"
                    }`}
                  >
                    <div className="absolute -right-10 -top-10 h-24 w-24 rounded-full bg-accent-dim opacity-0 blur-2xl transition-opacity group-hover:opacity-70" />
                    <div
                      className={`absolute inset-x-4 top-0 h-px bg-gradient-to-r from-transparent via-white/40 to-transparent transition-opacity ${
                        isActive ? "opacity-100" : "opacity-30"
                      }`}
                    />
                    {isActive && (
                      <span className="absolute right-3 top-3 h-2.5 w-2.5 rounded-full bg-accent shadow-[0_0_18px_rgba(97,216,255,0.9)]" />
                    )}
                    <div className="flex items-start justify-between gap-3">
                      <div
                        className={`flex h-11 w-11 items-center justify-center rounded-lg border ${
                          isActive
                            ? "border-accent text-accent"
                            : isComplete
                              ? "border-success text-success"
                              : "border-border text-text-muted"
                        }`}
                      >
                        <Icon size={19} />
                      </div>
                      <span
                        className={`rounded-full px-2.5 py-1 text-[10px] font-medium ${
                          isActive
                            ? "bg-accent-dim text-accent"
                            : isComplete
                              ? "bg-success-dim text-success"
                              : "bg-surface-3 text-text-muted"
                        }`}
                      >
                        {statusLabel[status]}
                      </span>
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-text-primary">
                        {stage.name}
                      </p>
                      <p className="mt-1 text-[11px] font-mono text-text-muted">
                        {stage.model} / {(stage.duration / 1000).toFixed(1)}s
                      </p>
                    </div>
                    <p className="text-xs leading-relaxed text-text-secondary">
                      {stage.insight}
                    </p>
                    {isActive && (
                      <div className="h-1 overflow-hidden rounded-full bg-black/20">
                        <div className="h-full w-2/3 animate-timeline-scan rounded-full bg-accent" />
                      </div>
                    )}
                  </button>
                  {index < pipelineStages.length - 1 && (
                    <ArrowRight size={18} className="text-text-muted" />
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {selectedStage && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4 py-6 backdrop-blur-sm">
          <div className="panel-raised max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-lg">
            <div className="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
              <div>
                <p className="section-kicker">Stage Detail</p>
                <h3 className="mt-2 text-2xl font-semibold text-text-primary">
                  {selectedStage.name}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setSelectedStage(null)}
                className="flex h-9 w-9 cursor-pointer items-center justify-center rounded-lg border border-border bg-black/10 text-text-muted transition-colors hover:text-text-primary"
                aria-label="Close stage details"
              >
                <X size={17} />
              </button>
            </div>
            <div className="space-y-5 p-5">
              <div className="grid gap-3 sm:grid-cols-3">
                <div className="rounded-lg border border-border bg-black/10 p-4">
                  <Clock3 size={16} className="text-accent" />
                  <p className="mt-3 text-xs text-text-muted">Duration</p>
                  <p className="mt-1 font-mono text-sm text-text-primary">
                    {(selectedStage.duration / 1000).toFixed(1)}s
                  </p>
                </div>
                <div className="rounded-lg border border-border bg-black/10 p-4">
                  <CircleDot size={16} className="text-warning" />
                  <p className="mt-3 text-xs text-text-muted">Status</p>
                  <p className="mt-1 text-sm font-semibold text-text-primary">
                    {statusLabel[selectedStatus]}
                  </p>
                </div>
                <div className="rounded-lg border border-border bg-black/10 p-4">
                  <Sparkles size={16} className="text-success" />
                  <p className="mt-3 text-xs text-text-muted">Agent</p>
                  <p className="mt-1 text-sm font-semibold text-text-primary">
                    {selectedStage.model}
                  </p>
                </div>
              </div>

              <div>
                <p className="text-sm leading-7 text-text-secondary">
                  {selectedStage.detail}
                </p>
              </div>

              <div>
                <p className="section-kicker mb-3">Expected Outputs</p>
                <div className="grid gap-2 sm:grid-cols-3">
                  {selectedStage.outputs.map((output) => (
                    <div
                      key={output}
                      className="flex items-center gap-2 rounded-lg border border-border bg-black/10 px-3 py-2 text-sm text-text-secondary"
                    >
                      <CheckCircle2 size={15} className="text-success" />
                      {output}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

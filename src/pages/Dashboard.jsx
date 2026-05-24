import {
  Activity,
  BookOpenCheck,
  BrainCircuit,
  Clock,
  Cpu,
  FlaskConical,
  Gauge,
  Lightbulb,
  PlayCircle,
  Radio,
} from "lucide-react";
import AgentTimeline from "../components/AgentTimeline";
import ArtifactDock from "../components/ArtifactDock";
import LogPanel from "../components/LogPanel";
import PaperPreview from "../components/PaperPreview";
import StatCard from "../components/StatCard";
import { usePipeline } from "../context/usePipeline";
import { pipelineStages } from "../data/pipelineStages";

export default function Dashboard() {
  const {
    activeStage,
    activeStageIndex,
    backendError,
    backendRunId,
    backendStatus,
    elapsedSeconds,
    isRunning,
    logs,
    progress,
    topic,
  } = usePipeline();

  const completedStages = pipelineStages.filter(
    (_, index) => index < activeStageIndex,
  ).length;

  return (
    <div className="space-y-8 animate-slide-in">
      <div className="control-hero px-5 py-6 sm:px-7 lg:px-8">
        <div className="relative z-10 grid gap-7 xl:grid-cols-[1.15fr_0.85fr] xl:items-center">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="data-chip rounded-full px-3 py-1 text-[11px] font-mono uppercase tracking-[0.18em] text-accent">
                Research Control Room
              </span>
              <span className="data-chip rounded-full px-3 py-1 text-[11px] font-mono uppercase tracking-[0.18em] text-success">
                API {backendStatus}
              </span>
            </div>
            <h1 className="mt-5 max-w-3xl text-4xl font-semibold leading-tight tracking-tight text-text-primary sm:text-5xl">
              Autonomous paper generation, visible at every handoff.
            </h1>
            <p className="mt-4 max-w-2xl text-sm leading-7 text-text-secondary">
              Monitor literature search, graph extraction, experiment design,
              sandbox execution, paper drafting, and compiled artifacts from a
              single control deck.
            </p>

            <div className="mt-6 grid gap-3 sm:grid-cols-3">
              {[
                { icon: Gauge, label: "Progress", value: `${progress}%` },
                { icon: Cpu, label: "Active agent", value: activeStage.shortName },
                { icon: Radio, label: "Backend run", value: backendRunId ? backendRunId.slice(0, 8) : "none" },
              ].map(({ icon: Icon, label, value }) => (
                <div key={label} className="data-chip rounded-lg p-3">
                  <Icon size={16} className="text-accent" />
                  <p className="mt-2 text-[11px] text-text-muted">{label}</p>
                  <p className="mt-1 text-sm font-semibold text-text-primary">
                    {value}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <div className="hero-orbit">
            <div className="hero-core">
              <BrainCircuit size={34} className="text-accent" />
            </div>
            <div className="absolute bottom-2 left-0 right-0 mx-auto w-full max-w-sm rounded-lg border border-border bg-black/25 p-4 backdrop-blur">
              <p className="text-[11px] font-mono uppercase tracking-[0.18em] text-text-muted">
                Active Topic
              </p>
              <p className="mt-2 line-clamp-3 text-sm leading-6 text-text-primary">
                {topic}
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.25fr_0.75fr]">
        <div className="panel-raised rounded-lg p-6 sm:p-7">
          <div className="max-w-2xl">
            <p className="section-kicker">Live Run Snapshot</p>
            <h2 className="mt-2 text-2xl font-semibold tracking-tight">
              {isRunning
                ? `${activeStage.name} is working.`
                : "The last generated paper package is ready."}
            </h2>
            <p className="mt-2 text-sm leading-7 text-text-secondary">
              {backendError
                ? backendError
                : isRunning
                ? activeStage.description
                : "Start a new run to watch the agents move through literature search, experiment design, code execution, writing, and compilation."}
            </p>
          </div>
          <div className="mt-6">
            <div className="flex items-center justify-between text-xs font-mono text-text-muted">
              <span>
                {completedStages} of {pipelineStages.length} stages complete
              </span>
              <span>{progress}%</span>
            </div>
            <div className="mt-2 h-2 overflow-hidden rounded-full bg-black/20 ring-1 ring-white/5">
              <div
                className="h-full rounded-full bg-[linear-gradient(90deg,#61d8ff,#49e6a7,#f6c85f)] transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
          <div className="panel-raised rounded-lg px-4 py-4">
            <div className="text-[11px] font-mono uppercase tracking-[0.18em] text-text-muted">
              Stage
            </div>
            <div className="mt-3 flex items-center gap-2 text-lg text-text-primary">
              <PlayCircle size={18} className="text-accent" />
              {activeStage.name}
            </div>
          </div>
          <div className="panel-raised rounded-lg px-4 py-4">
            <div className="text-[11px] font-mono uppercase tracking-[0.18em] text-text-muted">
              Status
            </div>
            <div
              className={
                isRunning ? "mt-3 text-lg text-accent" : "mt-3 text-lg text-success"
              }
            >
              {isRunning ? "Generating" : "Ready"}
            </div>
            <p className="mt-2 font-mono text-[11px] text-text-muted">
              backend: {backendStatus}
            </p>
          </div>
          <div className="panel-raised rounded-lg px-4 py-4 sm:col-span-2 xl:col-span-1">
            <div className="text-[11px] font-mono uppercase tracking-[0.18em] text-text-muted">
              Current Insight
            </div>
            <div className="mt-3 flex gap-2 text-sm leading-6 text-text-secondary">
              <Lightbulb size={17} className="mt-1 shrink-0 text-warning" />
              {activeStage.insight}
            </div>
          </div>
        </div>
      </div>

      <div>
        <p className="section-kicker mb-4">Pipeline Metrics</p>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <StatCard
            icon={FlaskConical}
            label="Current Stage"
            value={`${Math.min(activeStageIndex + 1, pipelineStages.length)}/${pipelineStages.length}`}
            sub={activeStage.shortName}
            accent
          />
          <StatCard
            icon={BookOpenCheck}
            label="Evidence Loop"
            value="KG"
            sub="claims stay traceable"
          />
          <StatCard
            icon={Clock}
            label="Elapsed"
            value={`${elapsedSeconds}s`}
            sub={isRunning ? "active run" : "latest run"}
          />
          <StatCard
            icon={Activity}
            label="Events"
            value={logs.length}
            sub="visible trace entries"
          />
        </div>
      </div>

      <div>
        <AgentTimeline
          activeStageIndex={activeStageIndex}
          isRunning={isRunning}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-[0.95fr_1.05fr]">
        <PaperPreview
          activeStageIndex={activeStageIndex}
          isRunning={isRunning}
          topic={topic}
        />
        <div>
          <h2 className="section-kicker mb-4">Recent Output</h2>
          <LogPanel logs={logs} maxHeight="320px" />
        </div>
      </div>

      <ArtifactDock
        activeStageIndex={activeStageIndex}
        isRunning={isRunning}
      />
    </div>
  );
}

import {
  Activity,
  BookOpenCheck,
  CheckCircle2,
  Clock,
  FileText,
  FlaskConical,
  Lightbulb,
  PlayCircle,
} from "lucide-react";
import PipelineStepper from "../components/PipelineStepper";
import LogPanel from "../components/LogPanel";
import StatCard from "../components/StatCard";
import { usePipeline } from "../context/usePipeline";
import { pipelineStages } from "../data/pipelineStages";

export default function Dashboard() {
  const {
    activeStage,
    activeStageIndex,
    elapsedSeconds,
    isRunning,
    logs,
    progress,
    stageStates,
    topic,
  } = usePipeline();

  const completedStages = pipelineStages.filter(
    (_, index) => index < activeStageIndex,
  ).length;

  return (
    <div className="space-y-8 animate-slide-in">
      <div className="grid gap-4 xl:grid-cols-[1.45fr_0.85fr]">
        <div>
          <p className="section-kicker">Research Control Room</p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">
            Dashboard
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-7 text-text-secondary">
            Monitor the pipeline, inspect stage transitions, and track system
            output from a single command-center layout.
          </p>
        </div>
        <div className="panel-soft rounded-lg px-5 py-5">
          <p className="section-kicker">Active Topic</p>
          <p className="mt-3 max-w-md text-sm leading-7 text-text-primary">
            {topic}
          </p>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.25fr_0.75fr]">
        <div className="panel-soft rounded-lg p-6 sm:p-7">
          <div className="max-w-2xl">
            <p className="section-kicker">Live Run Snapshot</p>
            <h2 className="mt-2 text-2xl font-semibold tracking-tight">
              {isRunning
                ? `${activeStage.name} is working.`
                : "The last generated paper package is ready."}
            </h2>
            <p className="mt-2 text-sm leading-7 text-text-secondary">
              {isRunning
                ? activeStage.description
                : "Start a new run to watch the agents move through literature search, experiment design, code execution, writing, and compilation."}
            </p>
          </div>
          <div className="mt-6">
            <div className="flex items-center justify-between text-xs font-mono text-text-muted">
              <span>{completedStages} of {pipelineStages.length} stages complete</span>
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
          <div className="panel-soft rounded-lg px-4 py-4">
            <div className="text-[11px] font-mono uppercase tracking-[0.18em] text-text-muted">
              Stage
            </div>
            <div className="mt-3 flex items-center gap-2 text-lg text-text-primary">
              <PlayCircle size={18} className="text-accent" />
              {activeStage.name}
            </div>
          </div>
          <div className="panel-soft rounded-lg px-4 py-4">
            <div className="text-[11px] font-mono uppercase tracking-[0.18em] text-text-muted">
              Status
            </div>
            <div className={isRunning ? "mt-3 text-lg text-accent" : "mt-3 text-lg text-success"}>
              {isRunning ? "Generating" : "Ready"}
            </div>
          </div>
          <div className="panel-soft rounded-lg px-4 py-4 sm:col-span-2 xl:col-span-1">
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
        <h2 className="section-kicker mb-4">Active Pipeline</h2>
        <div className="panel-soft overflow-x-auto rounded-lg p-5">
          <PipelineStepper agentStates={stageStates} />
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[0.75fr_1.25fr]">
        <div className="panel-soft rounded-lg p-5">
          <div className="flex items-center gap-3">
            <FileText size={20} className="text-accent" />
            <div>
              <p className="section-kicker">Paper Package</p>
              <h2 className="mt-1 text-lg font-semibold">Artifacts in motion</h2>
            </div>
          </div>
          <div className="mt-5 space-y-3 text-sm text-text-secondary">
            {["LaTeX source", "metrics JSON", "claim ledger", "compiled PDF"].map((item, index) => (
              <div key={item} className="flex items-center gap-3 rounded-lg border border-border bg-black/10 px-3 py-2.5">
                <CheckCircle2
                  size={16}
                  className={index < completedStages / 2 ? "text-success" : "text-text-muted"}
                />
                {item}
              </div>
            ))}
          </div>
        </div>
        <div>
          <h2 className="section-kicker mb-4">Recent Output</h2>
          <LogPanel logs={logs} maxHeight="320px" />
        </div>
      </div>
    </div>
  );
}

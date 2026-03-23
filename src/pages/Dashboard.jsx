import { Activity, Clock, FlaskConical, CheckCircle2 } from "lucide-react";
import PipelineStepper from "../components/PipelineStepper";
import LogPanel from "../components/LogPanel";
import StatCard from "../components/StatCard";

export default function Dashboard() {
  // Mock state — will be replaced by real pipeline state
  const agentStates = {
    researcher: "success",
    coder: "success",
    executor: "running",
    writer: "idle",
  };

  return (
    <div className="space-y-8 animate-slide-in">
      {/* Header */}
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
        <div className="panel-soft panel-cyber px-5 py-5">
          <p className="section-kicker">Active Topic</p>
          <p className="mt-3 max-w-md text-sm leading-7 text-text-primary">
            Random Forest vs Logistic Regression on the Iris dataset
          </p>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.35fr_0.65fr]">
        <div className="panel-soft panel-cyber p-6 sm:p-7">
          <div className="max-w-2xl">
            <p className="section-kicker">Live Run Snapshot</p>
            <h2 className="mt-2 text-2xl font-semibold tracking-tight">
              The executor is validating the generated experiment.
            </h2>
            <p className="mt-2 text-sm leading-7 text-text-secondary">
              Research and code generation completed successfully. The next
              handoff will send execution logs to the writer agent for paper
              synthesis.
            </p>
          </div>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
          <div className="panel-soft panel-cyber px-4 py-4">
            <div className="text-[11px] font-mono uppercase tracking-[0.18em] text-text-muted">
              Stage
            </div>
            <div className="mt-3 text-lg text-text-primary">Executor</div>
          </div>
          <div className="panel-soft panel-cyber px-4 py-4">
            <div className="text-[11px] font-mono uppercase tracking-[0.18em] text-text-muted">
              Status
            </div>
            <div className="mt-3 text-lg text-success">Running</div>
          </div>
        </div>
      </div>

      <div>
        <p className="section-kicker mb-4">Pipeline Metrics</p>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <StatCard
            icon={FlaskConical}
            label="Total Runs"
            value="12"
            sub="3 this week"
            accent
          />
          <StatCard
            icon={CheckCircle2}
            label="Successful"
            value="9"
            sub="75% success rate"
          />
          <StatCard
            icon={Clock}
            label="Avg. Duration"
            value="34s"
            sub="per pipeline run"
          />
          <StatCard
            icon={Activity}
            label="API Calls"
            value="147"
            sub="~$2.40 total"
          />
        </div>
      </div>

      {/* Pipeline visualizer */}
      <div>
        <h2 className="section-kicker mb-4">Active Pipeline</h2>
        <div className="panel-soft panel-cyber overflow-x-auto p-6">
          <PipelineStepper agentStates={agentStates} />
        </div>
      </div>

      {/* Live logs */}
      <div>
        <h2 className="section-kicker mb-4">Recent Output</h2>
        <LogPanel maxHeight="280px" />
      </div>
    </div>
  );
}

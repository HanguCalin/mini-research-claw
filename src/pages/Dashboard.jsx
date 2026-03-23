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
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
        <p className="text-sm text-text-secondary mt-1">
          Monitor your research pipeline in real time.
        </p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-4 gap-4">
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

      {/* Pipeline visualizer */}
      <div>
        <h2 className="text-sm font-medium text-text-secondary mb-4 uppercase tracking-wider">
          Active Pipeline
        </h2>
        <div className="p-6 rounded-xl border border-border bg-surface-1 overflow-x-auto">
          <PipelineStepper agentStates={agentStates} />
        </div>
      </div>

      {/* Live logs */}
      <div>
        <h2 className="text-sm font-medium text-text-secondary mb-4 uppercase tracking-wider">
          Recent Output
        </h2>
        <LogPanel maxHeight="280px" />
      </div>
    </div>
  );
}

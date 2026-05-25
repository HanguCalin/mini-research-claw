import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  BookOpen,
  ChevronDown,
  FileText,
  Gauge,
  Network,
  ShieldCheck,
  Sparkles,
  Zap,
} from "lucide-react";
import { usePipeline } from "../context/usePipeline";
import { pipelineStages } from "../data/pipelineStages";

const exampleTopics = [
  "Compare Random Forest vs Logistic Regression on the Iris dataset",
  "Analyze sentiment analysis accuracy of transformer vs RNN models",
  "Evaluate k-means vs DBSCAN clustering on synthetic datasets",
  "Benchmark neural network pruning techniques on CIFAR-10",
];

export default function NewResearch() {
  const [topic, setTopic] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const navigate = useNavigate();
  const { startRun } = usePipeline();
  const [config, setConfig] = useState({
    modelOverride: "",  // empty = use per-node defaults from backend/config.py
    maxRetries: 3,
    maxPapers: 5,
  });

  const launch = () => {
    if (!topic.trim()) return;
    startRun(topic, {
      modelOverride: config.modelOverride || null,
      maxCodeRetries: config.maxRetries,
      arxivResultsPerRound: config.maxPapers,
    });
    navigate("/");
  };

  return (
    <div className="grid gap-6 animate-slide-in xl:grid-cols-[0.95fr_1.05fr]">
      <div className="space-y-6">
        <div>
          <p className="section-kicker">Launch Pad</p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">
            New Research
          </h1>
          <p className="mt-2 max-w-xl text-sm leading-7 text-text-secondary">
            Shape the topic, choose constraints, and send the run into a visible
            generation workflow.
          </p>
        </div>

        <div className="grid gap-3 sm:grid-cols-3">
          {[
            { icon: BookOpen, label: "Paper cache", value: `${config.maxPapers} papers` },
            { icon: Gauge, label: "Retries", value: `${config.maxRetries} max` },
            { icon: ShieldCheck, label: "Sandbox", value: "isolated" },
          ].map(({ icon: Icon, label, value }) => (
            <div key={label} className="panel-soft rounded-lg p-4">
              <Icon size={16} className="text-accent" />
              <p className="mt-3 text-xs text-text-muted">{label}</p>
              <p className="mt-1 text-sm font-semibold text-text-primary">
                {value}
              </p>
            </div>
          ))}
        </div>
      </div>

      <div className="panel-soft space-y-5 rounded-lg p-5 sm:p-6">
        <label className="text-sm font-medium text-text-secondary">
          Research Topic
        </label>
        <div className="relative">
          <textarea
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="Describe what you want to research..."
            rows={3}
            className="w-full resize-none rounded-lg border border-border bg-black/15 px-4 py-4 text-sm text-text-primary shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] transition-all placeholder:text-text-muted focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30"
          />
          <Sparkles
            size={16}
            className="absolute top-3.5 right-3.5 text-text-muted"
          />
        </div>

        {/* Quick topics */}
        <div className="flex flex-wrap gap-2">
          {exampleTopics.map((t, i) => (
            <button
              key={i}
            onClick={() => setTopic(t)}
              className="cursor-pointer rounded-full border border-border bg-black/10 px-3 py-2 text-xs text-text-secondary transition-all hover:border-accent/40 hover:text-accent"
            >
              {t.length > 50 ? t.slice(0, 50) + "…" : t}
            </button>
          ))}
        </div>

        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex cursor-pointer items-center gap-2 text-sm text-text-secondary transition-colors hover:text-text-primary"
        >
          <ChevronDown
            size={16}
            className={`transition-transform ${showAdvanced ? "rotate-180" : ""}`}
          />
          Agent Configuration
        </button>

        {showAdvanced && (
          <div className="animate-slide-in space-y-4">
            <div className="space-y-1.5 sm:col-span-2">
              <label className="text-xs font-medium text-text-muted">
                Model Override (applies to every AI node)
              </label>
              <select
                value={config.modelOverride}
                onChange={(e) =>
                  setConfig((c) => ({ ...c, modelOverride: e.target.value }))
                }
                className="w-full cursor-pointer appearance-none rounded-lg border border-border bg-black/10 px-3 py-2.5 text-sm text-text-primary focus:border-accent focus:outline-none"
              >
                <option value="">Use per-node defaults (recommended)</option>
                <option value="claude-haiku-4-5-20251001">claude-haiku-4-5 &mdash; cheap, fast (every node)</option>
                <option value="claude-sonnet-4-6">claude-sonnet-4-6 &mdash; strongest (every node)</option>
                <option value="claude-opus-4-7">claude-opus-4-7 &mdash; max quality, slow + expensive</option>
              </select>
              <p className="text-[11px] text-text-muted">
                Defaults pick Haiku for cheap structured tasks (KG extraction,
                LaTeX repair) and Sonnet for reasoning (hypothesis, coding,
                writing). Overriding swaps the whole pipeline to one model.
              </p>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-text-muted">
                  Max Code Retries (sandbox self-heal loop)
                </label>
                <input
                  type="number"
                  min={0}
                  max={10}
                  value={config.maxRetries}
                  onChange={(e) =>
                    setConfig((c) => ({
                      ...c,
                      maxRetries: parseInt(e.target.value, 10) || 0,
                    }))
                  }
                  className="w-full rounded-lg border border-border bg-black/10 px-3 py-2.5 text-sm text-text-primary focus:border-accent focus:outline-none"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-text-muted">
                  Max Papers per ArXiv Round
                </label>
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={config.maxPapers}
                  onChange={(e) =>
                    setConfig((c) => ({
                      ...c,
                      maxPapers: parseInt(e.target.value, 10) || 5,
                    }))
                  }
                  className="w-full rounded-lg border border-border bg-black/10 px-3 py-2.5 text-sm text-text-primary focus:border-accent focus:outline-none"
                />
              </div>
            </div>
          </div>
        )}

        <button
          onClick={launch}
          disabled={!topic.trim()}
          className="action-primary flex w-full cursor-pointer items-center justify-center gap-2 rounded-lg px-6 py-3 text-sm font-semibold text-surface-0 transition-all hover:-translate-y-0.5 hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40 sm:w-auto"
        >
          <Zap size={16} />
          Launch Pipeline
        </button>
      </div>

      <div className="xl:col-span-2">
        <div className="panel-soft rounded-lg p-5">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="section-kicker">Generation Preview</p>
              <h2 className="mt-2 text-xl font-semibold tracking-tight">
                What the user will see while the paper is generated
              </h2>
            </div>
            <FileText size={22} className="text-accent" />
          </div>
          <div className="mt-5 grid gap-3 md:grid-cols-4">
            {pipelineStages.slice(0, 8).map((stage) => (
              <div key={stage.id} className="rounded-lg border border-border bg-black/10 p-4">
                <div className="flex items-center gap-2">
                  <Network size={14} className="text-accent" />
                  <p className="text-sm font-semibold text-text-primary">
                    {stage.shortName}
                  </p>
                </div>
                <p className="mt-2 text-xs leading-relaxed text-text-muted">
                  {stage.insight}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

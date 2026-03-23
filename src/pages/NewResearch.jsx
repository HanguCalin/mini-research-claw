import { useState } from "react";
import { Sparkles, ChevronDown, Zap } from "lucide-react";

const exampleTopics = [
  "Compare Random Forest vs Logistic Regression on the Iris dataset",
  "Analyze sentiment analysis accuracy of transformer vs RNN models",
  "Evaluate k-means vs DBSCAN clustering on synthetic datasets",
  "Benchmark neural network pruning techniques on CIFAR-10",
];

export default function NewResearch() {
  const [topic, setTopic] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [config, setConfig] = useState({
    researcherModel: "haiku",
    coderModel: "sonnet",
    writerModel: "sonnet",
    maxRetries: 3,
    maxPapers: 5,
  });

  return (
    <div className="max-w-3xl space-y-8 animate-slide-in">
      {/* Header */}
      <div>
        <p className="section-kicker">Launch Pad</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">
          New Research
        </h1>
        <p className="mt-2 text-sm leading-7 text-text-secondary">
          Enter a research topic and the AI agents will handle the rest.
        </p>
      </div>

      {/* Topic input */}
      <div className="panel-soft panel-cyber space-y-4 p-6">
        <label className="text-sm font-medium text-text-secondary">
          Research Topic
        </label>
        <div className="relative">
          <textarea
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="Describe what you want to research..."
            rows={3}
            className="w-full rounded-2xl border border-border bg-black/10 px-4 py-4 text-sm text-text-primary placeholder-text-muted resize-none shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] transition-all focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30"
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
      </div>

      {/* Advanced config toggle */}
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
        <div className="panel-soft panel-cyber grid gap-4 p-5 animate-slide-in sm:grid-cols-2">
          {/* Model selectors */}
          {[
            { key: "researcherModel", label: "Researcher Model" },
            { key: "coderModel", label: "Coder Model" },
            { key: "writerModel", label: "Writer Model" },
          ].map(({ key, label }) => (
            <div key={key} className="space-y-1.5">
              <label className="text-xs font-medium text-text-muted">
                {label}
              </label>
              <select
                value={config[key]}
                onChange={(e) =>
                  setConfig((c) => ({ ...c, [key]: e.target.value }))
                }
                className="w-full cursor-pointer appearance-none rounded-xl border border-border bg-black/10 px-3 py-2.5 text-sm text-text-primary focus:border-accent focus:outline-none"
              >
                <option value="haiku">Claude 3.5 Haiku</option>
                <option value="sonnet">Claude 3.7 Sonnet</option>
              </select>
            </div>
          ))}

          {/* Max retries */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-text-muted">
              Max Retries
            </label>
            <input
              type="number"
              min={1}
              max={5}
              value={config.maxRetries}
              onChange={(e) =>
                setConfig((c) => ({
                  ...c,
                  maxRetries: parseInt(e.target.value) || 3,
                }))
              }
              className="w-full rounded-xl border border-border bg-black/10 px-3 py-2.5 text-sm text-text-primary focus:border-accent focus:outline-none"
            />
          </div>

          {/* Max papers */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-text-muted">
              Max Papers to Fetch
            </label>
            <input
              type="number"
              min={1}
              max={20}
              value={config.maxPapers}
              onChange={(e) =>
                setConfig((c) => ({
                  ...c,
                  maxPapers: parseInt(e.target.value) || 5,
                }))
              }
              className="w-full rounded-xl border border-border bg-black/10 px-3 py-2.5 text-sm text-text-primary focus:border-accent focus:outline-none"
            />
          </div>
        </div>
      )}

      {/* Submit */}
      <button
        disabled={!topic.trim()}
        className="flex cursor-pointer items-center gap-2 rounded-2xl bg-[linear-gradient(135deg,#67e8f9,#34d399)] px-6 py-3 text-sm font-semibold text-surface-0 shadow-[0_18px_40px_rgba(52,211,153,0.16)] transition-all hover:-translate-y-0.5 hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Zap size={16} />
        Launch Pipeline
      </button>
    </div>
  );
}

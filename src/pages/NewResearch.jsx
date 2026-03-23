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
    <div className="max-w-2xl space-y-8 animate-slide-in">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">
          New Research
        </h1>
        <p className="text-sm text-text-secondary mt-1">
          Enter a research topic and the AI agents will handle the rest.
        </p>
      </div>

      {/* Topic input */}
      <div className="space-y-3">
        <label className="text-sm font-medium text-text-secondary">
          Research Topic
        </label>
        <div className="relative">
          <textarea
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="Describe what you want to research..."
            rows={3}
            className="w-full bg-surface-1 border border-border rounded-xl px-4 py-3 text-sm text-text-primary placeholder-text-muted resize-none focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent/30 transition-all"
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
              className="text-xs px-3 py-1.5 rounded-full border border-border bg-surface-2 text-text-secondary hover:text-accent hover:border-accent/40 transition-all cursor-pointer"
            >
              {t.length > 50 ? t.slice(0, 50) + "…" : t}
            </button>
          ))}
        </div>
      </div>

      {/* Advanced config toggle */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="flex items-center gap-2 text-sm text-text-secondary hover:text-text-primary transition-colors cursor-pointer"
      >
        <ChevronDown
          size={16}
          className={`transition-transform ${showAdvanced ? "rotate-180" : ""}`}
        />
        Agent Configuration
      </button>

      {showAdvanced && (
        <div className="grid grid-cols-2 gap-4 p-5 rounded-xl border border-border bg-surface-1 animate-slide-in">
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
                className="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent appearance-none cursor-pointer"
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
              className="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent"
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
              className="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent"
            />
          </div>
        </div>
      )}

      {/* Submit */}
      <button
        disabled={!topic.trim()}
        className="flex items-center gap-2 px-6 py-3 rounded-xl bg-accent text-surface-0 font-semibold text-sm hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed transition-all cursor-pointer"
      >
        <Zap size={16} />
        Launch Pipeline
      </button>
    </div>
  );
}

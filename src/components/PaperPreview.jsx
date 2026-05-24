import { FileText, Lock, Sparkles } from "lucide-react";
import { pipelineStages } from "../data/pipelineStages";

export default function PaperPreview({ activeStageIndex, isRunning, topic }) {
  const unlockedCount =
    !isRunning && activeStageIndex >= pipelineStages.length
      ? pipelineStages.length
      : Math.max(1, activeStageIndex + 1);

  return (
    <div className="panel-raised overflow-hidden rounded-lg">
      <div className="flex items-start justify-between gap-4 border-b border-border bg-black/10 px-5 py-4">
        <div>
          <p className="section-kicker">Growing Paper Preview</p>
          <h2 className="mt-1 text-lg font-semibold text-text-primary">
            Draft taking shape
          </h2>
        </div>
        <FileText size={21} className="text-accent" />
      </div>

      <div className="p-5">
        <div className="rounded-lg border border-border-bright bg-black/20 p-4">
          <p className="text-xs font-mono uppercase tracking-[0.18em] text-text-muted">
            Working title
          </p>
          <h3 className="mt-2 text-xl font-semibold leading-snug text-text-primary">
            {topic}
          </h3>
        </div>

        <div className="mt-4 space-y-3">
          {pipelineStages.map((stage, index) => {
            const unlocked = index < unlockedCount;
            const active = index === activeStageIndex && isRunning;

            return (
              <section
                key={stage.id}
                className={`rounded-lg border p-4 transition-all duration-300 ${
                  unlocked
                    ? "border-border-bright bg-black/20"
                    : "border-border bg-black/5 opacity-60"
                }`}
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    {unlocked ? (
                      <Sparkles
                        size={15}
                        className={active ? "text-accent" : "text-success"}
                      />
                    ) : (
                      <Lock size={14} className="text-text-muted" />
                    )}
                    <h4 className="text-sm font-semibold text-text-primary">
                      {stage.paperSection.title}
                    </h4>
                  </div>
                  {active && (
                    <span className="rounded-full bg-accent-dim px-2.5 py-1 text-[10px] font-medium text-accent">
                      writing
                    </span>
                  )}
                </div>
                <p className="mt-3 text-sm leading-7 text-text-secondary">
                  {unlocked
                    ? stage.paperSection.body
                    : "This section will unlock after the preceding agents produce enough context."}
                </p>
              </section>
            );
          })}
        </div>
      </div>
    </div>
  );
}

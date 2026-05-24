import {
  CheckCircle2,
  FileArchive,
  FileCode2,
  FileJson,
  FileText,
  PackageCheck,
} from "lucide-react";
import { pipelineStages } from "../data/pipelineStages";

const iconFor = (name) => {
  if (name.endsWith(".json")) return FileJson;
  if (name.endsWith(".py") || name.endsWith(".tex") || name.endsWith(".lock")) return FileCode2;
  if (name.endsWith(".pdf") || name.endsWith(".bib")) return FileText;
  return FileArchive;
};

export default function ArtifactDock({ activeStageIndex, isRunning }) {
  const readyArtifacts = pipelineStages.flatMap((stage, index) => {
    const complete = index < activeStageIndex || (!isRunning && activeStageIndex >= pipelineStages.length);
    return complete
      ? stage.artifacts.map((artifact) => ({
          artifact,
          stage: stage.shortName,
        }))
      : [];
  });

  return (
    <div className="panel-raised overflow-hidden rounded-lg">
      <div className="flex items-start justify-between gap-4 border-b border-border bg-black/10 px-5 py-4">
        <div>
          <p className="section-kicker">Artifact Dock</p>
          <h2 className="mt-1 text-lg font-semibold text-text-primary">
            Files appearing as agents finish
          </h2>
        </div>
        <PackageCheck size={21} className="text-success" />
      </div>

      <div className="p-5">
        {readyArtifacts.length === 0 ? (
          <div className="rounded-lg border border-dashed border-border bg-black/10 px-4 py-8 text-center">
            <FileArchive size={24} className="mx-auto text-text-muted" />
            <p className="mt-3 text-sm text-text-secondary">
              Artifacts will dock here as soon as the first stage completes.
            </p>
          </div>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {readyArtifacts.map(({ artifact, stage }, index) => {
              const Icon = iconFor(artifact);

              return (
                <div
                  key={`${artifact}-${index}`}
                  className="animate-dock-in rounded-lg border border-border bg-black/20 p-4 transition-all hover:-translate-y-0.5 hover:border-border-bright"
                  style={{ animationDelay: `${Math.min(index * 45, 360)}ms` }}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-border-bright bg-success-dim text-success">
                      <Icon size={18} />
                    </div>
                    <CheckCircle2 size={16} className="text-success" />
                  </div>
                  <p className="mt-4 break-all font-mono text-xs text-text-primary">
                    {artifact}
                  </p>
                  <p className="mt-2 text-[11px] text-text-muted">
                    created by {stage}
                  </p>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

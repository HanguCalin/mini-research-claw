import { useContext } from "react";
import { PipelineContext } from "./pipelineContextObject";

export function usePipeline() {
  const context = useContext(PipelineContext);
  if (!context) {
    throw new Error("usePipeline must be used inside PipelineProvider");
  }
  return context;
}

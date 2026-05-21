import { useEffect, useMemo, useState } from "react";
import { pipelineStages } from "../data/pipelineStages";
import { PipelineContext } from "./pipelineContextObject";

const seedLogs = [
  {
    time: "14:32:01",
    level: "info",
    agent: "system",
    msg: "Ready for a new research run.",
  },
];

function stamp() {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function makeStageLog(stage, topic, index) {
  const messages = [
    `Started ${stage.name.toLowerCase()} for: "${topic}"`,
    stage.insight,
    `${stage.name} completed. Passing context to the next agent.`,
  ];

  return messages.map((msg, i) => ({
    time: stamp(),
    level: i === 2 ? "success" : "info",
    agent: stage.id,
    msg: index === 0 && i === 0 ? `Pipeline initialized. ${msg}` : msg,
  }));
}

export function PipelineProvider({ children }) {
  const [topic, setTopic] = useState(
    "Random Forest vs Logistic Regression on the Iris dataset",
  );
  const [activeStageIndex, setActiveStageIndex] = useState(2);
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState(seedLogs);
  const [startedAt, setStartedAt] = useState(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  const startRun = (nextTopic) => {
    const trimmed = nextTopic.trim();
    if (!trimmed) return;

    setTopic(trimmed);
    setActiveStageIndex(0);
    setIsRunning(true);
    const startTime = Date.now();
    setStartedAt(startTime);
    setElapsedSeconds(0);
    const firstStage = pipelineStages[0];
    setLogs([
      {
        time: stamp(),
        level: "info",
        agent: "system",
        msg: `Queued paper generation for: "${trimmed}"`,
      },
      ...makeStageLog(firstStage, trimmed, 0),
    ]);
  };

  useEffect(() => {
    if (!isRunning) return undefined;

    const stage = pipelineStages[activeStageIndex];
    if (!stage) {
      return undefined;
    }

    const timer = window.setTimeout(() => {
      const nextIndex = activeStageIndex + 1;
      setActiveStageIndex(nextIndex);
      const nextStage = pipelineStages[nextIndex];
      if (nextStage) {
        setLogs((current) => [
          ...current,
          ...makeStageLog(nextStage, topic, nextIndex),
        ]);
      } else {
        setIsRunning(false);
        setLogs((current) => [
          ...current,
          {
            time: stamp(),
            level: "success",
            agent: "system",
            msg: "Paper package prepared: LaTeX source, metrics, claim ledger, and PDF artifact.",
          },
        ]);
      }
    }, stage.duration);

    return () => window.clearTimeout(timer);
  }, [activeStageIndex, isRunning, topic]);

  useEffect(() => {
    if (!isRunning || !startedAt) return undefined;

    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.max(1, Math.round((Date.now() - startedAt) / 1000)));
    }, 1000);

    return () => window.clearInterval(timer);
  }, [isRunning, startedAt]);

  const stageStates = useMemo(() => {
    return pipelineStages.reduce((states, stage, index) => {
      let status = "idle";
      if (index < activeStageIndex) status = "success";
      if (index === activeStageIndex && isRunning) status = "running";
      if (!isRunning && activeStageIndex >= pipelineStages.length) status = "success";
      states[stage.id] = status;
      return states;
    }, {});
  }, [activeStageIndex, isRunning]);

  const progress = Math.min(
    100,
    Math.round((activeStageIndex / pipelineStages.length) * 100),
  );
  const activeStage = pipelineStages[Math.min(activeStageIndex, pipelineStages.length - 1)];

  const value = {
    topic,
    activeStage,
    activeStageIndex,
    elapsedSeconds,
    isRunning,
    logs,
    progress,
    stageStates,
    startRun,
  };

  return (
    <PipelineContext.Provider value={value}>
      {children}
    </PipelineContext.Provider>
  );
}

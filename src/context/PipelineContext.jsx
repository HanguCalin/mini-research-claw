import { useEffect, useMemo, useState } from "react";
import { pipelineStages } from "../data/pipelineStages";
import { createPipelineRun, getPipelineRun } from "../services/api";
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
  const [backendRunId, setBackendRunId] = useState(null);
  const [backendStatus, setBackendStatus] = useState("idle");
  const [backendError, setBackendError] = useState(null);
  const [backendResult, setBackendResult] = useState(null);

  const startRun = (nextTopic) => {
    const trimmed = nextTopic.trim();
    if (!trimmed) return;

    setTopic(trimmed);
    setActiveStageIndex(0);
    setIsRunning(true);
    setBackendRunId(null);
    setBackendStatus("queued");
    setBackendError(null);
    setBackendResult(null);
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

    createPipelineRun(trimmed)
      .then((record) => {
        setBackendRunId(record.client_run_id);
        setBackendStatus(record.status);
        setLogs((current) => [
          ...current,
          {
            time: stamp(),
            level: "success",
            agent: "api",
            msg: `Backend run accepted: ${record.client_run_id}`,
          },
        ]);
      })
      .catch((error) => {
        setBackendStatus("failed");
        setBackendError(error.message);
        setIsRunning(false);
        setLogs((current) => [
          ...current,
          {
            time: stamp(),
            level: "error",
            agent: "api",
            msg: `Backend launch failed: ${error.message}`,
          },
        ]);
      });
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
        if (!backendRunId || backendStatus === "success" || backendStatus === "failed") {
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
        } else {
          setLogs((current) => [
            ...current,
            {
              time: stamp(),
              level: "info",
              agent: "api",
              msg: "Visual walkthrough complete. Waiting for backend pipeline result...",
            },
          ]);
        }
      }
    }, stage.duration);

    return () => window.clearTimeout(timer);
  }, [activeStageIndex, backendRunId, backendStatus, isRunning, topic]);

  useEffect(() => {
    if (!isRunning || !startedAt) return undefined;

    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.max(1, Math.round((Date.now() - startedAt) / 1000)));
    }, 1000);

    return () => window.clearInterval(timer);
  }, [isRunning, startedAt]);

  useEffect(() => {
    if (!backendRunId || backendStatus === "success" || backendStatus === "failed") {
      return undefined;
    }

    const poll = window.setInterval(() => {
      getPipelineRun(backendRunId)
        .then((record) => {
          setBackendStatus(record.status);
          if (record.result) setBackendResult(record.result);

          if (record.status === "success") {
            setActiveStageIndex(pipelineStages.length);
            setIsRunning(false);
            setLogs((current) => [
              ...current,
              {
                time: stamp(),
                level: "success",
                agent: "api",
                msg: "Backend pipeline completed successfully.",
              },
            ]);
          }

          if (record.status === "failed") {
            setIsRunning(false);
            setBackendError(record.error || "Backend pipeline failed.");
            setLogs((current) => [
              ...current,
              {
                time: stamp(),
                level: "error",
                agent: "api",
                msg: record.error || "Backend pipeline failed.",
              },
            ]);
          }
        })
        .catch((error) => {
          setBackendError(error.message);
          setLogs((current) => [
            ...current,
            {
              time: stamp(),
              level: "warn",
              agent: "api",
              msg: `Could not refresh backend status: ${error.message}`,
            },
          ]);
        });
    }, 3000);

    return () => window.clearInterval(poll);
  }, [backendRunId, backendStatus]);

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
    backendError,
    backendResult,
    backendRunId,
    backendStatus,
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

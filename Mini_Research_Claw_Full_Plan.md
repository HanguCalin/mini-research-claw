# Mini-Research-Claw (Claude Edition)

## Full Implementation & Development Process Plan

This document outlines the complete plan for building a 4-stage autonomous research pipeline driven by Claude AI agents, along with the full AI-assisted software development process.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [AI Agents Architecture](#2-ai-agents-architecture)
3. [User Stories & Product Backlog](#3-user-stories--product-backlog)
4. [Diagrams](#4-diagrams)
5. [Implementation Phases](#5-implementation-phases)
6. [Source Control Strategy (Git)](#6-source-control-strategy-git)
7. [Automated Testing & Agent Evals](#7-automated-testing--agent-evals)
8. [Bug Reporting & Resolution Workflow](#8-bug-reporting--resolution-workflow)
9. [CI/CD Pipeline](#9-cicd-pipeline)
10. [Demo Plan](#10-demo-plan)
11. [Report on AI Tool Usage During Development](#11-report-on-ai-tool-usage-during-development)

---

## 1. Project Overview

Mini-Research-Claw is an autonomous research assistant that takes a natural-language research topic from a user, searches academic literature, generates a testable hypothesis, writes and executes code to validate it, and produces a complete Markdown research paper — all without human intervention.

The system uses LangGraph for multi-agent orchestration and leverages multiple Claude models optimized per task. A Docker sandbox ensures safe code execution.

**Key Constraint:** This project is a CLI/desktop-based AI agent pipeline.

---

## 2. AI Agents Architecture

The system includes **4 agents, of which 3 are AI-powered**.

### Agent 1: The Researcher Agent (AI)

- **Model:** Claude 3.5 Haiku (fast, cost-effective for reading and summarization).
- **Tools:** A bound Python function using the `arxiv` library to search by topic.
- **Responsibility:** Execute the arXiv search, synthesize retrieved abstracts, and produce a structured hypothesis written into the Global State.

### Agent 2: The Coder Agent (AI)

- **Model:** Claude 3.7 Sonnet (advanced reasoning and software engineering).
- **System Prompt:** "You are an expert data scientist. Read the hypothesis in the state. Write a self-contained Python script to test it using public datasets (e.g., scikit-learn). Save results to `results.txt`. Output ONLY valid Python code."
- **Responsibility:** Read the hypothesis from state, generate a complete Python script, and write it to `code_payload`.

### Agent 3: The Executor Agent (Non-AI — Sandbox Manager)

- **Model:** None — pure Python logic.
- **Responsibility:** Take `code_payload`, run it inside a Docker container, and route based on outcome:
  - `exit code == 0` → update `execution_logs` with results, route to Writer.
  - `exit code != 0` → update `execution_logs` with the stack trace, increment `error_count`, route back to Coder with the error context.

### Agent 4: The Writer Agent (AI)

- **Model:** Claude 3.7 Sonnet (excellent at academic tone and long-context synthesis).
- **System Prompt:** "You are an academic writer. Synthesize the provided literature and execution logs into a 4-section Markdown paper: Introduction, Methodology, Results, and Conclusion."
- **Responsibility:** Compile the final Markdown research paper and save it to disk.

---

## 3. User Stories & Product Backlog

### 3.1 User Stories (12 total)

| ID    | User Story | Priority | Story Points |
| ----- | ---------- | -------- | ------------ |
| US-01 | As a researcher, I want to enter a natural-language research topic so that the system starts an autonomous pipeline. | Must Have | 3 |
| US-02 | As a researcher, I want the system to search arXiv for relevant papers so that I get a literature review without manual searching. | Must Have | 5 |
| US-03 | As a researcher, I want the Researcher Agent to synthesize abstracts into a testable hypothesis so that the pipeline has a clear goal. | Must Have | 5 |
| US-04 | As a researcher, I want the Coder Agent to generate a self-contained Python experiment so that the hypothesis can be validated programmatically. | Must Have | 8 |
| US-05 | As a researcher, I want all generated code to run in a sandboxed Docker container so that my local machine is protected from arbitrary execution. | Must Have | 5 |
| US-06 | As a researcher, I want the system to automatically retry failed code up to 3 times with error feedback so that transient or fixable errors are self-healed. | Must Have | 5 |
| US-07 | As a researcher, I want a Writer Agent to produce a 4-section Markdown paper from the results so that I receive a publication-ready draft. | Must Have | 5 |
| US-08 | As a researcher, I want to see a progress log in the terminal showing which agent is currently active so that I can monitor the pipeline's state. | Should Have | 3 |
| US-09 | As a researcher, I want the system to output a failure report if all retries are exhausted so that I understand what went wrong. | Should Have | 3 |
| US-10 | As a researcher, I want to configure which Claude models are used per agent via a config file so that I can optimize cost vs. quality. | Could Have | 2 |
| US-11 | As a researcher, I want the final paper saved with a timestamped filename so that I can keep multiple runs organized. | Could Have | 1 |
| US-12 | As a developer, I want comprehensive logs of every API call and state transition so that I can debug and evaluate the system. | Should Have | 3 |

### 3.2 Product Backlog

The backlog is organized into 4 sprints:

**Sprint 1 — Foundation (Week 1):**
US-01, US-05, US-12 — CLI entry point, Docker sandbox setup, logging infrastructure.

**Sprint 2 — Core Agents (Week 2):**
US-02, US-03, US-04 — Researcher, Coder agents, arXiv integration.

**Sprint 3 — Orchestration & Resilience (Week 3):**
US-06, US-07, US-08, US-09 — Self-healing loop, Writer agent, progress display, failure reports.

**Sprint 4 — Polish & Config (Week 4):**
US-10, US-11 — Model configuration, timestamped outputs, final integration testing.

---

## 4. Diagrams

All diagrams are stored in the repository under the `docs/diagrams/` directory.

### 4.1 Component Architecture Diagram

High-level system components: CLI Interface, LangGraph Orchestrator, Researcher Agent, Coder Agent, Executor Agent (Docker Sandbox), Writer Agent, arXiv API, File System Output.

### 4.2 LangGraph Workflow Diagram (State Machine)

```
START → [Researcher] → [Coder] → [Executor]
                                      │
                          ┌───────────┼───────────┐
                          │           │           │
                       success    fail (<3)    fail (≥3)
                          │           │           │
                          ▼           ▼           ▼
                       [Writer]    [Coder]      END
                          │                  (failure report)
                          ▼
                        END
                  (final paper)
```

### 4.3 UML Sequence Diagram

Illustrates the message flow between User → CLI → LangGraph → each Agent → external services (arXiv, Docker) → file output.

### 4.4 Global State Data Model (Class Diagram)

```
┌─────────────────────────────────────┐
│           ResearchState             │
├─────────────────────────────────────┤
│ + topic: str                        │
│ + literature: List[Dict]            │
│ + hypothesis: str                   │
│ + code_payload: str                 │
│ + execution_logs: str               │
│ + error_count: int                  │
│ + final_paper: str                  │
├─────────────────────────────────────┤
│  TypedDict used as LangGraph State  │
└─────────────────────────────────────┘
```

### 4.5 Deployment / Infrastructure Diagram

Shows: Host machine, Docker daemon, Python virtual environment, API calls to Anthropic, arXiv REST API, file system I/O.

**Deliverable:** All diagrams rendered as `.png` or `.svg` and stored in `docs/diagrams/`. Mermaid source files kept alongside for version control.

---

## 5. Implementation Phases

### Phase 1: Environment & Infrastructure Setup

1. Initialize a clean Python project with `pyproject.toml` or `requirements.txt`.
2. Install core dependencies: `anthropic`, `langgraph`, `arxiv`, `docker`.
3. Create a `Dockerfile` with a base Python image and data science libraries (pandas, scikit-learn, numpy).
4. Store `ANTHROPIC_API_KEY` securely in a `.env` file (excluded from git via `.gitignore`).

### Phase 2: Define the Global State

| State Variable   | Type       | Description                                                |
| ---------------- | ---------- | ---------------------------------------------------------- |
| `topic`          | str        | The user's initial research prompt.                        |
| `literature`     | List[Dict] | Extracted abstracts and metadata from arXiv.               |
| `hypothesis`     | str        | The testable idea generated by the Researcher.             |
| `code_payload`   | str        | The Python script generated by the Coder.                  |
| `execution_logs` | str        | Stdout/Stderr output from the Docker sandbox.              |
| `error_count`    | int        | Tracks retry attempts to prevent infinite loops (max 3).   |
| `final_paper`    | str        | The complete Markdown output.                              |

### Phase 3: Build the Agents

Implement each of the 4 agents as described in Section 2 above, each in its own Python module under `src/agents/`.

### Phase 4: Orchestration with LangGraph

1. Register the 4 agents as graph nodes.
2. Define linear edges: START → Researcher → Coder → Executor.
3. Define conditional edges at Executor:
   - Success → Writer → END.
   - Failure (error_count < 3) → Coder (retry loop).
   - Failure (error_count ≥ 3) → END (failure report).
4. Compile and expose the graph via a `run_pipeline(topic: str)` function.

### Phase 5: Testing & Iteration

1. **Happy Path:** "Compare the accuracy of a Random Forest vs. Logistic Regression on the Iris dataset."
2. **Forced Failure:** Inject a deliberate error (e.g., reference a non-installed library) to validate the self-healing retry loop.

---

## 6. Source Control Strategy (Git)

### 6.1 Repository Structure

```
mini-research-claw/
├── src/
│   ├── agents/
│   │   ├── researcher.py
│   │   ├── coder.py
│   │   ├── executor.py
│   │   └── writer.py
│   ├── state.py
│   ├── graph.py
│   └── main.py
├── tests/
│   ├── test_researcher.py
│   ├── test_coder.py
│   ├── test_executor.py
│   ├── test_writer.py
│   └── evals/
│       ├── eval_researcher.py
│       └── eval_coder.py
├── docs/
│   ├── diagrams/
│   └── ai-usage-report.md
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

### 6.2 Branching Strategy

- `main` — stable, production-ready code. Protected branch requiring pull request reviews.
- `develop` — integration branch for feature merges.
- `feature/<name>` — one branch per user story or feature (e.g., `feature/researcher-agent`, `feature/docker-sandbox`).
- `bugfix/<id>` — one branch per reported bug (e.g., `bugfix/BUG-003-retry-overflow`).

### 6.3 Commit Requirements

- Conventional commit style with frequent, meaningful commits.
- Conventional commit messages: `feat:`, `fix:`, `test:`, `docs:`, `ci:`.
- Every feature branch merged via **Pull Request** with at least one reviewer.
- Use **rebase** to keep feature branches up to date with `develop`; use **merge commits** when merging PRs into `develop`.

---

## 7. Automated Testing & Agent Evals

### 7.1 Unit Tests

| Test File              | What It Tests                                                        |
| ---------------------- | -------------------------------------------------------------------- |
| `test_researcher.py`   | arXiv search returns valid results; hypothesis is non-empty string.  |
| `test_coder.py`        | Generated code is syntactically valid Python; contains required imports. |
| `test_executor.py`     | Docker container starts/stops; exit codes are captured correctly.     |
| `test_writer.py`       | Output Markdown contains all 4 sections; is non-empty.               |

### 7.2 Integration Tests

- **End-to-end pipeline test:** Run the full graph on a known-good topic and assert that `final_paper` is populated.
- **Retry loop test:** Feed a deliberately broken `code_payload`, assert that `error_count` increments and the Coder is re-invoked.

### 7.3 Agent Evals (LLM-Specific)

These evals measure agent quality beyond pass/fail:

| Eval                          | Method                                                                                                    | Pass Criteria                          |
| ----------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| **Researcher: Relevance**     | Run Researcher on 5 known topics. Use a separate Claude call to score hypothesis relevance (1–5 scale).   | Average relevance ≥ 3.5/5             |
| **Researcher: Completeness**  | Check that `literature` contains ≥ 3 papers with non-empty abstracts.                                     | ≥ 3 papers per query                  |
| **Coder: Syntax Validity**    | Parse `code_payload` with `ast.parse()`.                                                                  | No `SyntaxError` on 10/10 runs        |
| **Coder: Execution Success**  | Run the generated code in sandbox across 5 different hypotheses.                                          | ≥ 4/5 execute without error on first try |
| **Writer: Structure**         | Regex-check that output contains `# Introduction`, `# Methodology`, `# Results`, `# Conclusion`.         | All 4 headers present                 |
| **Writer: Coherence**         | Use a separate Claude call to score coherence and factual consistency of the paper (1–5 scale).            | Average coherence ≥ 3.5/5             |

### 7.4 Test Framework

- **Framework:** `pytest` with `pytest-asyncio` for async agent calls.
- **Mocking:** Use `unittest.mock` to stub Anthropic API calls in unit tests (avoid burning tokens on every CI run).
- **Eval runner:** A standalone `scripts/run_evals.py` that executes all agent evals and outputs a summary table.

---

## 8. Bug Reporting & Resolution Workflow

### 8.1 Bug Report Template (GitHub Issues)

All bugs are filed as GitHub Issues using this template:

```
**Bug ID:** BUG-XXX
**Title:** [Short description]
**Severity:** Critical / Major / Minor
**Reported by:** [Name]
**Date:** [YYYY-MM-DD]

**Steps to Reproduce:**
1. ...
2. ...

**Expected Behavior:** ...
**Actual Behavior:** ...
**Logs / Screenshots:** ...

**Environment:**
- OS: ...
- Python version: ...
- Docker version: ...
```

### 8.2 Resolution Process

1. A bug is filed as a GitHub Issue with the template above.
2. A developer creates a `bugfix/<BUG-ID>` branch from `develop`.
3. The fix is implemented, with a test added that would have caught the bug.
4. A **Pull Request** is opened referencing the Issue (e.g., "Fixes #12").
5. At least one team member reviews the PR.
6. After approval, the PR is merged and the Issue is closed.

### 8.3 Minimum Deliverables

- At least **3 documented bugs** filed as Issues.
- At least **3 bugfix PRs** merged, each linking to its Issue.
- AI tools (e.g., Claude, Copilot) used to help diagnose root causes — documented in the AI usage report.

---

## 9. CI/CD Pipeline

### 9.1 CI Pipeline (GitHub Actions)

File: `.github/workflows/ci.yml`

**Triggers:** On every push to `develop` and on every Pull Request to `develop` or `main`.

**Jobs:**

1. **Lint:** Run `ruff` or `flake8` for code style enforcement.
2. **Type Check:** Run `mypy` for static type analysis.
3. **Unit Tests:** Run `pytest tests/` (excluding evals) with mocked API calls.
4. **Integration Tests:** Run on merge to `develop` only (requires Docker and optionally a real API key stored as a GitHub Secret).
5. **Build Docker Image:** Verify the sandbox Dockerfile builds successfully.

### 9.2 CD Pipeline

**Trigger:** On merge to `main`.

**Steps:**

1. Build and tag the Docker sandbox image.
2. Package the application (e.g., as a pip-installable package or a Docker Compose stack).
3. Publish a GitHub Release with the generated artifacts.

### 9.3 Secrets Management

- `ANTHROPIC_API_KEY` stored as a GitHub Actions secret.
- `.env` file is in `.gitignore` — never committed.

---

## 10. Demo Plan

### 10.1 Live Demo

- Run the full pipeline from the CLI with a live topic.
- Show the terminal output as each agent activates in sequence.
- Open the generated Markdown paper and walk through each section.
- Trigger a forced failure to demonstrate the self-healing retry loop.

### 10.2 Offline Demo (Screencast)

- **Tool:** OBS Studio, Loom, or any screen recorder with audio.
- **Format:** MP4 uploaded to YouTube (unlisted).
- **Content:** Full walkthrough identical to the live demo, with voiceover explaining each stage.
- **Duration:** 5–10 minutes.
- **Deliverable:** YouTube link included in `README.md`.

---

## 11. Report on AI Tool Usage During Development

This section documents how AI tools were used throughout every phase of development. The full report is available at `docs/ai-usage-report.md`.

### 11.1 Areas of AI Usage

| Development Phase            | AI Tool(s) Used                        | How It Was Used                                                                                   |
| ---------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------- |
| User Stories & Backlog       | Claude / ChatGPT                       | Generated initial user stories from the project description; refined acceptance criteria.          |
| Architecture & Diagrams      | Claude (Mermaid), ChatGPT (PlantUML)   | Generated UML sequence diagrams, component diagrams, and state machine diagrams from descriptions. |
| Code Implementation          | GitHub Copilot, Claude                 | Auto-completed boilerplate, generated agent prompt templates, wrote Docker configuration.          |
| Test Writing                 | Claude / Copilot                       | Generated pytest test cases and mock fixtures from function signatures.                           |
| Agent Eval Design            | Claude                                 | Designed evaluation rubrics and LLM-as-judge prompts for coherence and relevance scoring.         |
| Bug Diagnosis                | Claude                                 | Pasted stack traces to Claude for root-cause analysis and suggested fixes.                        |
| CI/CD Setup                  | Claude / Copilot                       | Generated GitHub Actions YAML workflow from project requirements.                                 |
| Documentation & Report       | Claude                                 | Drafted README, this implementation plan, and the AI usage report itself.                         |
| Code Review                  | Claude                                 | Used as a reviewer on PRs — pasted diffs and asked for feedback before human review.              |
| Commit Messages              | Copilot                                | Auto-generated conventional commit messages from staged diffs.                                    |

### 11.2 Report Structure

The final `ai-usage-report.md` will include, for each area:

1. The specific AI tool used and the model version.
2. The prompt or input given.
3. The output received.
4. How the output was validated, edited, or rejected.
5. A reflection on effectiveness: what worked, what didn't, and what was faster or slower with AI.

### 11.3 Metrics to Track

- Total number of AI-assisted interactions (prompts sent).
- Percentage of generated code accepted vs. rejected.
- Time saved estimate per task category.
- Number of hallucinations or incorrect outputs encountered and how they were handled.

---

## Project Status Summary

| Area | Section | Status |
| ---- | ------- | ------ |
| AI Agents (3 AI-powered + 1 non-AI) | §2 | Planned |
| User Stories (12) + Product Backlog (4 sprints) | §3 | Planned |
| Diagrams (5 total) | §4 | Planned |
| Git Strategy (branches, PRs, conventional commits) | §6 | Planned |
| Automated Tests + Agent Evals (6 evals) | §7 | Planned |
| Bug Reporting + Resolution via PR | §8 | Planned |
| CI/CD Pipeline (GitHub Actions) | §9 | Planned |
| AI Usage Report (10 areas + metrics) | §11 | Planned |
| Live Demo + Screencast | §10 | Planned |

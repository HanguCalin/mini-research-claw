# Auto-Mini-Claw (Deep Research & Peer-Review Edition)

## Full Implementation & Development Process Plan

This document outlines the complete plan for building a 100% autonomous research pipeline. The user inputs a research topic, and the system orchestrates multiple AI agents via **LangGraph** to conduct full-text literature retrieval, execute data science experiments, draft an academic paper, and subject it to an automated multi-agent Peer Review before generating the final NeurIPS-formatted PDF.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Multi-Agent Architecture](#2-multi-agent-architecture-dag-execution-flow)
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

Auto-Mini-Claw is a 100% autonomous research assistant that takes a natural-language research topic from a user, retrieves and parses full-text academic literature, builds a deep Knowledge Graph to ground all claims, generates a testable hypothesis, writes and executes experiments, drafts an academic paper in LaTeX, subjects it to an automated multi-agent Peer Review panel, and compiles the final NeurIPS-formatted PDF — all without human intervention.

The system uses LangGraph for multi-agent orchestration with 9 sequential nodes organized in 5 phases, leveraging multiple Claude models optimized per task. A Docker sandbox ensures safe code execution, a deep Knowledge Graph serves as the "single source of truth" for anti-hallucination validation, and a 3-agent peer review panel ensures quality before final compilation.

**Key Constraint:** This project is a CLI/desktop-based AI agent pipeline.

---

## 2. Multi-Agent Architecture (DAG Execution Flow)

The system consists of 9 sequential nodes, with 7 AI-powered agents and 2 non-AI nodes, organized in 5 phases.

### Phase 1: Deep Literature Parsing & Knowledge Graph (Data Grounding)

#### Node 1: Full-Text ArXiv Retriever (Non-AI)

- **Model:** None — pure Python logic.
- **Tools:** A Python script using the `arxiv` library and arXiv's HTML endpoints or LaTeX source parsing.
- **Responsibility:** Query arXiv based on the user's prompt. Instead of stopping at abstracts, it fetches the full text of the top 3–5 papers and extracts the `Methodology`, `Implementation`, and `Results` sections. Writes results to `arxiv_papers_full_text` in the state.
- **Security & Compliance:**
  - Enforces a strict `time.sleep(3)` between consecutive arXiv API/HTTP requests to comply with arXiv's Terms of Service and prevent rate-limiting or IP bans.
  - Parsed full-text content is kept in memory only — no PDFs are saved to disk for redistribution.
  - Paper metadata (authors, year, title, arXiv ID) is preserved for BibTeX generation downstream.

#### Node 2: Deep KG Extractor (AI)

- **Model:** Claude 3.5 Haiku (fast, cost-effective for structured extraction).
- **System Prompt:** "Analyze the following full-text academic papers and extract a granular JSON Knowledge Graph of logical triplets in the format {subject, relation, object}, including specific technical details such as hyperparameters, preprocessing steps, and model configurations."
- **Responsibility:** Analyze the full texts and extract a granular JSON Knowledge Graph. Because it reads the full paper, the triplets include specific technical details (e.g., `Model_X → uses_hyperparameter → Learning_Rate=0.001` or `Dataset_Y → requires_preprocessing → Normalization`). Writes to `knowledge_graph` in the state.

#### Node 3: Hypothesis Generator (AI)

- **Model:** Claude 3.7 Sonnet (advanced reasoning for hypothesis formulation).
- **System Prompt:** "Formulate a highly specific, testable research hypothesis strictly grounded in the technical entities extracted into the Knowledge Graph. When referencing datasets, you MUST use real, verifiable, public dataset IDs from the Hugging Face Hub (e.g., `imdb`, `glue`, `squad`) or scikit-learn. Do NOT hallucinate dataset names or local file paths."
- **Responsibility:** Formulate a highly specific, testable research hypothesis strictly grounded in the technical entities extracted into the deep KG. The hypothesis is validated against the KG entities to prevent hallucination. Dataset references default to the Hugging Face Hub ecosystem.

### Phase 2: Autonomous Experimentation & Self-Healing

#### Node 4: ML Coder (AI)

- **Model:** Claude 3.7 Sonnet (advanced reasoning and software engineering).
- **System Prompt:** "You are an expert data scientist. Read the validated hypothesis and the rich KG with actual implementation details. Write a self-contained, methodologically rigorous Python script. You MUST: (1) explicitly separate train and test data to prevent data leakage; (2) use cross-validation where applicable; (3) set random seeds (e.g., `random_state=42`) for full reproducibility; (4) use real, verifiable, public dataset IDs from Hugging Face Hub (e.g., `load_dataset('imdb')`) or scikit-learn — do NOT hallucinate local file paths or custom dataset names; (5) save a detailed log of all hyperparameters used alongside evaluation metrics into `metrics.json`. Output ONLY valid Python code."
- **Responsibility:** Receive the hypothesis and the rich KG. Armed with actual implementation details (not just abstract concepts), generate a robust Python script (`experiment.py`) to train and evaluate a model with methodologically rigorous ML practices (train/test split, cross-validation, reproducible seeds, hyperparameter logging), saving results into `metrics.json`. Defaults to the Hugging Face `datasets` library for data loading and `transformers`/`huggingface_hub` for pre-trained models where applicable.

#### Node 5: Executor Sandbox (Non-AI)

- **Model:** None — pure Python logic.
- **Responsibility:** Run `experiment.py` in an isolated environment (Docker/Subprocess) and route based on outcome:
  - **Error:** Send the stack trace back to **Node 4**. The Coder performs self-reflection ("Why did the code fail?") and rewrites the code (loop allowed up to 3 times).
  - **Success:** Save the results into `metrics_json` in the state and proceed to the next phase.
- **Docker Hardening:**
  - Container runs in rootless mode with `--security-opt=no-new-privileges` to drop unnecessary privileges.
  - No volume mounts to the host's sensitive directories — only a temporary working directory is mounted.
  - Network access is disabled (`--network=none`) to prevent data exfiltration from LLM-generated code.

### Phase 3: Paper Drafting

#### Node 6: Academic Writer (AI)

- **Model:** Claude 3.7 Sonnet (excellent at academic tone and long-context synthesis).
- **System Prompt:** "You are an academic writer. Synthesize the full-text literature, the hypothesis, and the experiment metrics to write an academic paper directly in LaTeX, following the IMRaD structure (Introduction, Methods, Results, Conclusion). You MUST ground your technical claims in the provided Knowledge Graph, but do NOT use raw IDs or inline provenance tags (like '[Source: arXiv:...]'). Instead, use standard LaTeX citation commands (e.g., `\cite{AuthorYear}`) seamlessly in the text. You MUST also generate a corresponding `references.bib` file containing the BibTeX entries for all papers in the Knowledge Graph. Ensure the bibliography is rendered at the bottom of the final NeurIPS paper via `\bibliography{references}`. Do not state information as absolute truth if it cannot be traced back to the literature context."
- **Responsibility:** Synthesize the full-text literature, the hypothesis, and `metrics_json` results to write the first draft of the paper directly in **LaTeX** (`draft.tex`), following the IMRaD structure. Generate a companion `references.bib` with proper BibTeX entries (using paper metadata: authors, year, title, arXiv ID from the state). All citations use `\cite{AuthorYear}` — no raw arXiv IDs in prose. Writes to `latex_draft` and `bibtex_source` in the state.

### Phase 4: Critique & Linting Engine (Automated Quality Assurance)

The drafted LaTeX is evaluated by an ensemble of agents acting as a critique and linting engine. Rather than binary PASS/FAIL votes, each agent returns a structured list of warnings that the Writer must address in exactly one mandatory revision pass.

#### Node 7: Critique Panel (AI — 3x Claude 3.5 Haiku)

Three independent agents read the `draft.tex` and produce structured warnings:

- **Agent A (The Fact-Checker):** Cross-references the draft with the deep Phase 1 Knowledge Graph. Returns warnings for any algorithms, mathematical formulas, datasets, or claims that cannot be traced back to the KG. Example: `{"warnings": ["Claim X is not supported by metrics", "Dataset Z not found in KG"]}`.
- **Agent B (The Methodologist):** Evaluates if the code results in `metrics.json` logically support the conclusions drawn in the text. Flags unsupported claims, missing error bars, or unjustified generalizations. Example: `{"warnings": ["Accuracy improvement of 2% claimed as 'significant' without statistical test"]}`.
- **Agent C (The Formatter):** Checks for "AI-slop" writing style, excessive verbosity, missing NeurIPS checklist items, and verifies LaTeX structural integrity. Example: `{"warnings": ["Missing NeurIPS checklist", "Section 3 exceeds recommended length"]}`.

#### Node 8: Critique Aggregator & Mandatory Revision (Non-AI)

- **Model:** None — pure Python logic.
- **Responsibility:** Aggregates all warnings from the 3 critique agents into a single structured feedback list. The aggregated warnings are always routed back to **Node 6 (Academic Writer)** for exactly **one mandatory revision pass**. The Writer must address the critique, produce a revised `draft.tex`, and append a "Confidence Score" (self-assessed 1–10) and the NeurIPS reproducibility checklist to the final draft. No further review rounds occur — the revised draft proceeds directly to the LaTeX Compiler.

### Phase 5: Publication

#### Node 9: LaTeX Compiler (Non-AI)

- **Model:** None — pure Python logic.
- **Responsibility:** Execute system commands `pdflatex` and `bibtex` to compile the approved LaTeX source into the final two-column NeurIPS-formatted PDF, ready for download. Writes the output path to `final_pdf_path` in the state.
- **LaTeX Security:** Uses `subprocess.run(['pdflatex', '--no-shell-escape', 'main.tex'], ...)` to explicitly disable shell escapes, preventing malicious code execution from LLM-generated LaTeX content.

---

## 3. User Stories & Product Backlog

### 3.1 User Stories (15 total)

| ID    | User Story | Priority | Story Points |
| ----- | ---------- | -------- | ------------ |
| US-01 | As a researcher, I want to enter a natural-language research topic so that the system starts an autonomous pipeline. | Must Have | 3 |
| US-02 | As a researcher, I want the system to retrieve full-text papers from arXiv (not just abstracts) so that the pipeline has deep technical context. | Must Have | 5 |
| US-03 | As a researcher, I want the Deep KG Extractor to build a granular Knowledge Graph with technical details (hyperparameters, preprocessing) from full texts so that the system has a verified source of truth. | Must Have | 5 |
| US-04 | As a researcher, I want the Hypothesis Generator to synthesize a testable hypothesis strictly grounded in KG entities so that hallucinated claims are prevented. | Must Have | 8 |
| US-05 | As a researcher, I want all generated code to run in a sandboxed Docker container so that my local machine is protected from arbitrary execution. | Must Have | 5 |
| US-06 | As a researcher, I want the system to automatically retry failed code up to 3 times with error feedback so that transient or fixable errors are self-healed. | Must Have | 5 |
| US-07 | As a researcher, I want an Academic Writer Agent to draft the paper in LaTeX (IMRaD format) so that I receive a structured first draft. | Must Have | 5 |
| US-08 | As a researcher, I want a 3-agent Critique & Linting Panel (Fact-Checker, Methodologist, Formatter) to produce structured warnings on the draft so that quality issues are identified before compilation. | Must Have | 8 |
| US-09 | As a researcher, I want the Writer to perform one mandatory revision pass addressing critique warnings, appending a Confidence Score and NeurIPS checklist, so that the final paper meets quality standards. | Must Have | 5 |
| US-10 | As a researcher, I want the approved LaTeX to be compiled into a two-column NeurIPS-formatted PDF so that I receive a publication-ready paper. | Must Have | 3 |
| US-11 | As a researcher, I want to see a progress log in the terminal showing which agent is currently active so that I can monitor the pipeline's state. | Should Have | 3 |
| US-12 | As a researcher, I want the system to output a failure report if all retries are exhausted so that I understand what went wrong. | Should Have | 3 |
| US-13 | As a researcher, I want to configure which Claude models are used per agent via a config file so that I can optimize cost vs. quality. | Could Have | 2 |
| US-14 | As a researcher, I want the final PDF saved with a timestamped filename so that I can keep multiple runs organized. | Could Have | 1 |
| US-15 | As a developer, I want comprehensive logs of every API call and state transition so that I can debug and evaluate the system. | Should Have | 3 |

### 3.2 Product Backlog

The backlog is organized into 4 sprints:

**Sprint 1 — Foundation (Week 1):**
US-01, US-05, US-15 — CLI entry point, Docker sandbox setup, logging infrastructure.

**Sprint 2 — Core Agents (Week 2):**
US-02, US-03, US-04 — Full-text ArXiv retrieval, deep KG extraction, hypothesis generation with anti-hallucination eval.

**Sprint 3 — Orchestration & Resilience (Week 3):**
US-06, US-07, US-08, US-09, US-11, US-12 — Self-healing loop, Academic Writer (LaTeX draft), Critique & Linting Panel, mandatory revision, progress display, failure reports.

**Sprint 4 — Publication & Polish (Week 4):**
US-10, US-13, US-14 — LaTeX compilation, model configuration, timestamped outputs, final integration testing.

---

## 4. Diagrams

All diagrams are stored in the repository under the `docs/diagrams/` directory.

### 4.1 Component Architecture Diagram

High-level system components: CLI Interface, LangGraph Orchestrator, Full-Text ArXiv Retriever, Deep KG Extractor, Hypothesis Generator, ML Coder, Executor Sandbox (Docker), Academic Writer, Review Panel (3 agents: Fact-Checker, Methodologist, Formatter), Vote Aggregator, LaTeX Compiler, arXiv API, File System Output.

### 4.2 LangGraph Workflow Diagram (State Machine)

```
Phase 1: Deep Literature Parsing & KG
START → [Full-Text ArXiv Retriever] → [Deep KG Extractor] → [Hypothesis Generator]
                                                                     │
                                                               ┌─────┴─────┐
                                                               │           │
                                                          KG valid    KG invalid
                                                               │     (hallucination)
                                                               │           │
                                                               ▼           ▼
                                                          continue    regenerate
                                                               │
Phase 2: Experimentation                                       │
                                                        [ML Coder] → [Executor]
                                                            ▲            │
                                                            │    ┌───────┼───────┐
                                                            │    │       │       │
                                                            │ success fail(<3) fail(≥3)
                                                            │    │       │       │
                                                            │    ▼       │       ▼
                                                            └────────────┘     END
                                                                 │        (failure report)
Phase 3: Paper Drafting                                          │
                                                        [Academic Writer]
                                                                 │
Phase 4: Critique & Linting                                      │
                                                        [Critique Panel (3 agents)]
                                                          A: Fact-Checker (warnings)
                                                          B: Methodologist (warnings)
                                                          C: Formatter (warnings)
                                                                 │
                                                        [Critique Aggregator]
                                                                 │
                                                                 ▼
                                                        [Academic Writer]
                                                        (1 mandatory revision)
                                                        + Confidence Score
                                                        + NeurIPS Checklist
                                                                 │
Phase 5: Publication                                             │
                                                     [LaTeX Compiler]
                                                            │
                                                            ▼
                                                           END
                                                      (NeurIPS PDF)
```

### 4.3 UML Sequence Diagram

Illustrates the message flow between User → CLI → LangGraph → each Node (Full-Text ArXiv Retriever, Deep KG Extractor, Hypothesis Generator, ML Coder, Executor, Academic Writer, Review Panel, Vote Aggregator, LaTeX Compiler) → external services (arXiv, Docker, pdflatex) → PDF output.

### 4.4 Global State Data Model (Class Diagram)

```
┌────────────────────────────────────────────────┐
│             AutoResearchState                  │
├────────────────────────────────────────────────┤
│  Phase 1: Deep Context & Setup                 │
│ + topic: str                                   │
│ + arxiv_papers_full_text: List[Dict]           │
│ + knowledge_graph: List[Dict[str, str]]        │
│ + hypothesis: str                              │
├────────────────────────────────────────────────┤
│  Phase 2: Execution                            │
│ + python_code: str                             │
│ + execution_success: bool                      │
│ + execution_logs: str                          │
│ + metrics_json: str                            │
│ + code_retry_count: int                        │
├────────────────────────────────────────────────┤
│  Phase 3 & 4: Drafting and Critique             │
│ + latex_draft: str                             │
│ + bibtex_source: str                           │
│ + critique_warnings: List[Dict[str, str]]      │
│ + confidence_score: float                      │
├────────────────────────────────────────────────┤
│  Phase 5: Output                               │
│ + final_pdf_path: str                          │
├────────────────────────────────────────────────┤
│   TypedDict used as LangGraph State            │
└────────────────────────────────────────────────┘
```

### 4.5 Deployment / Infrastructure Diagram

Shows: Host machine, Docker daemon, Python virtual environment, API calls to Anthropic (multiple agents), arXiv REST API + HTML/LaTeX source endpoints, pdflatex/bibtex tools, file system I/O (metrics.json, draft.tex, references.bib, peer review feedback, final PDF).

**Deliverable:** All diagrams rendered as `.png` or `.svg` and stored in `docs/diagrams/`. Mermaid source files kept alongside for version control.

---

## 5. Implementation Phases

### Phase 1: Environment & Infrastructure Setup

1. Initialize a clean Python project with `pyproject.toml` or `requirements.txt`.
2. Install core dependencies: `anthropic`, `langgraph`, `arxiv`, `docker`, `datasets`, `huggingface_hub`.
3. Create a `Dockerfile` with a base Python image and data science libraries (pandas, scikit-learn, numpy, datasets, huggingface_hub, transformers).
4. Install LaTeX toolchain (`texlive`, `pdflatex`, `bibtex`) in the build environment.
5. Store `ANTHROPIC_API_KEY` securely in a `.env` file (excluded from git via `.gitignore`).

### Phase 2: Define the Global State

```python
from typing import TypedDict, List, Dict

class AutoResearchState(TypedDict):
    # Phase 1: Deep Context & Setup
    topic: str
    arxiv_papers_full_text: List[Dict]      # Full methodology/results sections
    knowledge_graph: List[Dict[str, str]]   # Enhanced triplets including formulas and hyperparams
    hypothesis: str

    # Phase 2: Execution
    python_code: str
    execution_success: bool
    execution_logs: str
    metrics_json: str
    code_retry_count: int

    # Phase 3 & 4: Drafting and Critique
    latex_draft: str
    bibtex_source: str
    critique_warnings: List[Dict[str, str]]    # Warnings from 3-agent critique panel
    confidence_score: float                     # Writer's self-assessed confidence (1-10)

    # Phase 5: Output
    final_pdf_path: str
```

| State Variable          | Type                    | Description                                                        |
| ----------------------- | ----------------------- | ------------------------------------------------------------------ |
| `topic`                 | str                     | The user's initial research prompt.                                |
| `arxiv_papers_full_text`| List[Dict]              | Full-text methodology, implementation, and results sections.       |
| `knowledge_graph`       | List[Dict[str, str]]    | Deep KG triplets with technical details (hyperparams, etc.).       |
| `hypothesis`            | str                     | The testable hypothesis, grounded in KG entities.                  |
| `python_code`           | str                     | The Python experiment script generated by the ML Coder.            |
| `execution_success`     | bool                    | Whether the experiment executed successfully.                      |
| `execution_logs`        | str                     | Stdout/Stderr output from the Docker sandbox.                      |
| `metrics_json`          | str                     | JSON string with experiment results (accuracy, F1-score, etc.).    |
| `code_retry_count`      | int                     | Tracks code retry attempts (max 3).                                |
| `latex_draft`           | str                     | The LaTeX draft source code (`draft.tex`).                         |
| `bibtex_source`         | str                     | The generated bibliography (`references.bib`).                     |
| `critique_warnings`     | List[Dict[str, str]]    | Structured warnings from the 3-agent critique panel.               |
| `confidence_score`      | float                   | Writer's self-assessed confidence score (1–10) after revision.     |
| `final_pdf_path`        | str                     | Path to the compiled NeurIPS-formatted PDF.                        |

### Phase 3: Build the Nodes

Implement each of the 9 nodes as described in Section 2 above, each in its own Python module under `src/agents/`:
- `arxiv_retriever.py` — Full-text ArXiv paper download and section extraction (non-AI)
- `kg_extractor.py` — Deep Knowledge Graph triplet extraction (AI)
- `hypothesis_generator.py` — Hypothesis generation with KG grounding (AI)
- `ml_coder.py` — Experiment code generation with KG context (AI)
- `executor.py` — Docker sandbox runner (non-AI)
- `academic_writer.py` — LaTeX/BibTeX draft generation (AI)
- `critique_panel.py` — 3-agent critique & linting: Fact-Checker, Methodologist, Formatter (AI)
- `critique_aggregator.py` — Warning aggregation and mandatory revision routing (non-AI)
- `latex_compiler.py` — PDF compilation (non-AI)

### Phase 4: Orchestration with LangGraph

1. Register the 9 nodes as graph nodes.
2. Define Phase 1 edges: START → Full-Text ArXiv Retriever → Deep KG Extractor → Hypothesis Generator.
3. Define conditional edge at Hypothesis Generator:
   - KG validation passes → ML Coder.
   - KG validation fails (hallucination) → regenerate hypothesis.
4. Define Phase 2 edges: ML Coder → Executor.
5. Define conditional edges at Executor:
   - Success → Academic Writer.
   - Failure (code_retry_count < 3) → ML Coder (self-healing loop).
   - Failure (code_retry_count ≥ 3) → END (failure report).
6. Define Phase 3 edge: Academic Writer → Critique Panel.
7. Define Phase 4 edges: Critique Panel → Critique Aggregator → Academic Writer (one mandatory revision pass).
8. Define Phase 5 edge: Academic Writer (revised) → LaTeX Compiler → END.
10. Compile and expose the graph via a `run_pipeline(topic: str)` function.

### Phase 5: Testing & Iteration

1. **Happy Path:** "Compare the accuracy of a Random Forest vs. Logistic Regression on the Hugging Face `imdb` sentiment dataset."
2. **Forced Failure:** Inject a deliberate error (e.g., reference a non-installed library) to validate the self-healing retry loop.

---

## 6. Source Control Strategy (Git)

### 6.1 Repository Structure

```
mini-research-claw/
├── src/
│   ├── agents/
│   │   ├── arxiv_retriever.py
│   │   ├── kg_extractor.py
│   │   ├── hypothesis_generator.py
│   │   ├── ml_coder.py
│   │   ├── executor.py
│   │   ├── academic_writer.py
│   │   ├── critique_panel.py
│   │   ├── critique_aggregator.py
│   │   └── latex_compiler.py
│   ├── state.py
│   ├── graph.py
│   └── main.py
├── tests/
│   ├── test_arxiv_retriever.py
│   ├── test_kg_extractor.py
│   ├── test_hypothesis_generator.py
│   ├── test_ml_coder.py
│   ├── test_executor.py
│   ├── test_academic_writer.py
│   ├── test_critique_panel.py
│   ├── test_critique_aggregator.py
│   ├── test_latex_compiler.py
│   └── evals/
│       ├── eval_kg_extractor.py
│       ├── eval_hypothesis.py
│       ├── eval_coder.py
│       ├── eval_writer.py
│       └── eval_critique_panel.py
├── docs/
│   ├── diagrams/
│   └── ai-usage-report.md
├── templates/
│   └── neurips_template.tex
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

| Test File                       | What It Tests                                                                       |
| ------------------------------- | ----------------------------------------------------------------------------------- |
| `test_arxiv_retriever.py`       | arXiv search returns full-text results; methodology/results sections are extracted.  |
| `test_kg_extractor.py`          | KG output is valid JSON with granular triplets including technical details.          |
| `test_hypothesis_generator.py`  | Hypothesis is non-empty; entities are grounded in KG (anti-hallucination check).    |
| `test_ml_coder.py`              | Generated code is syntactically valid Python; contains required imports.             |
| `test_executor.py`              | Docker container starts/stops; exit codes are captured correctly.                   |
| `test_academic_writer.py`       | LaTeX draft is valid; contains IMRaD sections; BibTeX file is well-formed.          |
| `test_critique_panel.py`        | Each agent outputs valid JSON warnings list; 3 independent critiques produced.      |
| `test_critique_aggregator.py`   | Warnings are correctly aggregated; mandatory revision routing works.                |
| `test_latex_compiler.py`        | PDF is generated successfully from LaTeX; output file exists and is non-empty.      |

### 7.2 Integration Tests

- **End-to-end pipeline test:** Run the full graph on a known-good topic and assert that `final_pdf_path` points to a valid PDF.
- **Code retry loop test:** Feed a deliberately broken `python_code`, assert that `code_retry_count` increments and the ML Coder is re-invoked.
- **Anti-hallucination test:** Provide a hypothesis with entities not in the KG and assert that it is rejected and regenerated.
- **Critique loop test:** Submit a draft with deliberate issues, assert that the Critique Panel returns structured warnings and the Writer performs one mandatory revision.
- **Confidence score test:** Assert that the revised draft includes a confidence score (1–10) and the NeurIPS reproducibility checklist.

### 7.3 Agent Evals (LLM-Specific)

These evals measure agent quality beyond pass/fail:

| Eval                          | Method                                                                                                    | Pass Criteria                          |
| ----------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| **KG Extractor: Validity**         | Check that KG output is valid JSON with subject/relation/object triplets including technical details.       | 100% valid structure on 10/10 runs     |
| **KG Extractor: Depth**           | Verify KG contains granular triplets (hyperparams, preprocessing steps), not just abstract concepts.       | ≥ 5 technical-detail triplets per paper |
| **KG Extractor: Completeness**    | Check that `arxiv_papers_full_text` yields ≥ 3 papers and KG contains ≥ 15 triplets.                      | ≥ 3 papers, ≥ 15 triplets per query   |
| **Hypothesis: Anti-Hallucination**| Verify all entities in hypothesis exist in KG triplets. Inject fake entities and confirm rejection.        | 100% hallucinated hypotheses rejected  |
| **Hypothesis: Relevance**         | Run on 5 known topics. Use a separate Claude call to score relevance (1–5 scale).                          | Average relevance ≥ 3.5/5             |
| **Coder: Syntax Validity**        | Parse `python_code` with `ast.parse()`.                                                                    | No `SyntaxError` on 10/10 runs        |
| **Coder: Execution Success**      | Run the generated code in sandbox across 5 different hypotheses.                                           | ≥ 4/5 execute without error on first try |
| **Writer: LaTeX Validity**         | Compile LaTeX draft with `pdflatex` and verify PDF is generated without errors.                            | Successful compilation on 10/10 runs   |
| **Writer: Structure**             | Regex-check that LaTeX output contains `\section{Introduction}`, `\section{Methods}`, `\section{Results}`, `\section{Conclusion}`. | All 4 sections present    |
| **Critique Panel: Consistency**    | Run 3 critique agents on the same draft 5 times; verify warnings are deterministic and substantive.        | ≥ 80% warning consistency across runs  |
| **Critique Panel: Fact-Check**     | Submit a draft with hallucinated citations; verify the Fact-Checker flags them as warnings.                 | 100% hallucinated citations detected   |
| **Critique Panel: Revision Quality**| Compare pre-revision and post-revision drafts; use Claude to score improvement (1–5 scale).               | Average improvement ≥ 3.0/5           |
| **Coder: ML Rigor**               | Verify generated code contains train/test split, random seeds, and cross-validation.                       | All 3 practices present on 10/10 runs  |

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
- Show the terminal output as each node activates in sequence (ArXiv → Deep KG → Hypothesis → Coder → Executor → Writer → Peer Review → PDF).
- Show the deep Knowledge Graph triplets (with technical details) and how the hypothesis is grounded in them.
- Show the Critique Panel warnings from each agent (Fact-Checker, Methodologist, Formatter).
- Demonstrate the mandatory revision pass: show how the Writer addresses warnings and appends the Confidence Score.
- Open the generated NeurIPS PDF and walk through each section.
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
| 9 Nodes (7 AI-powered + 2 non-AI) in 5 Phases | §2 | Planned |
| Deep KG with Full-Text Parsing (Anti-Hallucination) | §2 (Phase 1) | Planned |
| 3-Agent Critique & Linting Engine | §2 (Phase 4) | Planned |
| User Stories (15) + Product Backlog (4 sprints) | §3 | Planned |
| Diagrams (5 total) | §4 | Planned |
| Git Strategy (branches, PRs, conventional commits) | §6 | Planned |
| Automated Tests (9 unit + 5 integration) + Agent Evals (12 evals) | §7 | Planned |
| Bug Reporting + Resolution via PR | §8 | Planned |
| CI/CD Pipeline (GitHub Actions) | §9 | Planned |
| NeurIPS PDF Generation (LaTeX + BibTeX) | §2 (Phase 5) | Planned |
| AI Usage Report (10 areas + metrics) | §11 | Planned |
| Live Demo + Screencast | §10 | Planned |

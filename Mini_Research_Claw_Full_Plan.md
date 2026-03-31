# Auto-Mini-Claw (Deep Research & Peer-Review Edition)

## Full Implementation & Development Process Plan

This document outlines the complete plan for building a production-grade autonomous research pipeline. The user inputs a research topic, and the system orchestrates multiple AI agents via **LangGraph** to conduct full-text literature retrieval, build a deduplicated Knowledge Graph, generate a novelty-scored hypothesis, execute data science experiments in a network-isolated sandbox, draft an academic paper in LaTeX, subject it to a **heterogeneous multi-agent peer review with structured debate**, and compile the final NeurIPS-formatted PDF with an automated **LaTeX repair loop**. A mandatory **Human-in-the-Loop (HITL) gate** ensures human oversight before committing to expensive GPU compute.

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

Auto-Mini-Claw is an autonomous research assistant that takes a natural-language research topic from a user, retrieves and parses full-text academic literature, builds a **deduplicated, schema-driven Knowledge Graph** to ground all claims, generates a testable hypothesis validated for **mathematical novelty**, pauses for **human approval**, writes and executes experiments inside a **network-isolated Docker sandbox** (with pre-cached dependencies), drafts an academic paper in LaTeX, subjects it to a **heterogeneous multi-agent peer review with structured debate** (dismantling the "Artificial Hivemind" problem), and compiles the final NeurIPS-formatted PDF with an automated **LaTeX compiler repair loop**.

The system uses LangGraph for multi-agent orchestration with **11 sequential nodes** organized in 5 phases plus a HITL checkpoint, leveraging multiple Claude models optimized per task. Six critical architectural upgrades — informed by a rigorous design review — address systemic execution bottlenecks: sparse KG extraction, lack of novelty verification, absence of human oversight, the network-isolation contradiction in sandboxed execution, echo-chamber consensus in homogeneous review panels, and brittle LaTeX compilation without error recovery.

**Key Constraint:** This project is a CLI/desktop-based AI agent pipeline.

---

## 2. Multi-Agent Architecture (DAG Execution Flow)

The system consists of **11 sequential nodes**, with **7 AI-powered agents** and **4 non-AI nodes**, organized in **5 phases plus a HITL checkpoint**.

### Phase 1: Deep Literature Parsing & Knowledge Graph (Data Grounding)

#### Node 1: Full-Text ArXiv Retriever (Non-AI)

- **Model:** None — pure Python logic.
- **Tools:** A Python script using the `arxiv` library and arXiv's HTML endpoints or LaTeX source parsing.
- **Responsibility:** Query arXiv based on the user's prompt. Instead of stopping at abstracts, it fetches the full text of the top 3–5 papers and extracts the `Methodology`, `Implementation`, and `Results` sections. Writes results to `arxiv_papers_full_text` in the state.
- **Security & Compliance:**
  - Enforces a strict `time.sleep(3)` between consecutive arXiv API/HTTP requests to comply with arXiv's Terms of Service and prevent rate-limiting or IP bans.
  - Parsed full-text content is kept in memory only — no PDFs are saved to disk for redistribution.
  - Paper metadata (authors, year, title, arXiv ID) is preserved for BibTeX generation downstream.

#### Node 2: Deep KG Extractor (AI) — *Upgraded: Schema-Based Extraction + Entity Resolution*

> **Design Review Finding:** Naive JSON prompting creates sparse, fragmented knowledge graphs with duplicate entities (e.g., "Neural Net", "NN", "Neural Network" stored as separate nodes), corrupting downstream hypothesis generation.

- **Model:** Claude 3.5 Haiku (fast, cost-effective for structured extraction).
- **System Prompt:** Uses **schema-based extraction** with strict typed dictionaries defining the expected entity types (`model`, `dataset`, `metric`, `method`, `hyperparameter`) and relation types (`outperforms`, `uses_dataset`, `achieves_metric`, `has_hyperparameter`). The prompt enforces rigid JSON output boundaries — no free-form generation. Example schema enforced:

```json
{
  "entities": [
    {
      "id": "e1",
      "canonical_name": "Random Forest",
      "entity_type": "model",
      "aliases": ["RF", "random forest classifier"],
      "attributes": {"n_estimators": "100", "max_depth": "None"}
    }
  ],
  "edges": [
    {
      "source_id": "e1",
      "target_id": "e2",
      "relation": "outperforms",
      "confidence": 0.92,
      "provenance": "arXiv:2401.12345, Section 4.2"
    }
  ]
}
```

- **Post-Processing Pipeline** (deterministic, not LLM):
  1. **Embedding-based clustering:** Embed all entity names using SBERT (`all-MiniLM-L6-v2`). Cluster entities with cosine similarity > 0.85 to identify synonymous nodes (e.g., "Neural Net" ↔ "NN" ↔ "Neural Network").
  2. **LLM deduplication pass:** For each cluster, ask a single Claude call to pick the canonical name and merge attributes from all aliases.
  3. **Edge resolution:** Remove redundant/contradictory edges. Keep only the highest-confidence edge per unique `(source, target, relation)` triple.
- **Output:** Clean, deduplicated `kg_entities[]` and `kg_edges[]` written to state.

#### Node 3: Hypothesis Generator (AI) — *Upgraded: Mathematical Novelty Detection*

> **Design Review Finding:** The hypothesis generator validates against hallucinations but not against the broader scientific corpus. LLMs frequently propose "novel" ideas that are well-established concepts (e.g., micro-batching for SGD presented as a breakthrough).

- **Model:** Claude 3.7 Sonnet (advanced reasoning for hypothesis formulation).
- **System Prompt:** "Formulate a highly specific, testable research hypothesis strictly grounded in the technical entities extracted into the Knowledge Graph. When referencing datasets, you MUST use real, verifiable, public dataset IDs from the Hugging Face Hub (e.g., `imdb`, `glue`, `squad`) or scikit-learn. Do NOT hallucinate dataset names or local file paths."
- **Responsibility:** Formulate a testable hypothesis grounded in the KG entities. The hypothesis is validated against KG entities to prevent hallucination.

- **Automated Novelty Detection Protocol** (deterministic post-step):
  1. Embed the generated hypothesis using SBERT (`all-MiniLM-L6-v2`).
  2. Embed all paper abstracts from `arxiv_papers_full_text[]`.
  3. Compute **Relative Neighbor Density (RND):** the average cosine distance from the hypothesis embedding to the K nearest literature embeddings.
  4. Compare RND against `novelty_threshold` (configurable, default: `0.35`).
     - **RND ≥ threshold** → hypothesis is sufficiently novel → proceed to HITL Gate.
     - **RND < threshold** → hypothesis is too similar to existing work → pipeline terminates with `failed_novelty` status and a report explaining which papers are too close.
  5. Write `novelty_score`, `hypothesis_embedding`, and `novelty_passed` to state.

### HITL Checkpoint: Human-in-the-Loop Approval Gate

> **Design Review Finding:** 100% autonomy leads to silent failures — fabricated metrics, hallucinated citations, wasted GPU compute on unviable concepts. Empirical studies show co-pilot mode with human checkpoints significantly outperforms fully autonomous pipelines (Agent Laboratory, 2024).

#### Node 3b: HITL Gate (Non-AI)

- **Model:** None — deterministic checkpoint logic.
- **Responsibility:** The pipeline **pauses** and presents the human operator with:
  1. The generated **hypothesis** (plain text).
  2. The **supporting KG triples** (entities + edges that ground the hypothesis).
  3. The computed **novelty score** and how it compares to the threshold.
  4. A summary of retrieved literature titles and abstracts.

- **Operator actions:**
  - **`approve`** → sets `hitl_approved = True`; pipeline proceeds to Phase 2 (Code Generation).
  - **`reject <reason>`** → sets `hitl_approved = False` and `hitl_rejection_reason`; pipeline terminates with `failed_hitl_rejected` status.

- **Implementation approach:**
  - **CLI mode:** Pipeline blocks on `input()` prompt with a Rich-formatted summary panel.
  - **Web UI mode:** Pipeline emits `awaiting_hitl` status; the React frontend polls and presents an approval dialog; the backend exposes an `/api/hitl/approve` endpoint.

- **Rationale:** No code generation or GPU compute is provisioned until human approval is received. This prevents the system from wasting resources on unviable or unoriginal concepts.

### Phase 2: Autonomous Experimentation & Self-Healing

#### Node 4: ML Coder (AI) — *Upgraded: Active Debugging*

> **Design Review Finding:** The passive self-healing loop (simply pasting stack traces back) is insufficient for deep methodological errors like tensor shape mismatches, exploding gradients, or silent NaN loss values.

- **Model:** Claude 3.7 Sonnet (advanced reasoning and software engineering).
- **System Prompt:** "You are an expert data scientist. Read the validated hypothesis and the rich KG with actual implementation details. Write a self-contained, methodologically rigorous Python script. You MUST: (1) explicitly separate train and test data to prevent data leakage; (2) use cross-validation where applicable; (3) set random seeds (e.g., `random_state=42`) for full reproducibility; (4) use real, verifiable, public dataset IDs from Hugging Face Hub (e.g., `load_dataset('imdb')`) or scikit-learn — do NOT hallucinate local file paths or custom dataset names; (5) save a detailed log of all hyperparameters used alongside evaluation metrics into `metrics.json`. Output ONLY valid Python code."
- **Active Debugging Upgrade:** The coder injects strategic `print()` statements at key checkpoints (data loading confirmation, tensor shapes after transformation, training loss per epoch, evaluation metric values) so that execution logs contain intermediate state — not just final results or bare stack traces. On retry, the Coder receives the full execution log (including debug prints) and the previous code, and must perform **root-cause analysis** before rewriting — not blind regeneration.

#### Node 4b: Dependency Resolver (Non-AI) — *New Node: Resolves Network-Isolation Contradiction*

> **Design Review Finding:** The ML Coder's system prompt demands `load_dataset('imdb')` from Hugging Face, but the Executor runs Docker with `--network=none`. The script will crash instantly on any outbound HTTP request.

- **Model:** None — pure Python logic.
- **Responsibility:** Parses the **Abstract Syntax Tree (AST)** of the generated `experiment.py` to identify:
  - **pip dependencies:** Walks `import` / `from ... import` statements and maps them to PyPI package names.
  - **Remote datasets:** Detects calls to `load_dataset("...")`, `fetch_openml(...)`, `sklearn.datasets.fetch_*` patterns.

- **Host-side pre-fetch** (runs with normal network access on the host, outside the sandbox):
  - `pip download <packages> --dest .cache/pip/`
  - `huggingface-cli download <dataset_id> --cache-dir .cache/hf/`
  - `python -c "from sklearn.datasets import fetch_...; fetch_...(data_home='.cache/sklearn/')"` for scikit-learn datasets.

- **Output:** Writes `resolved_dependencies`, `resolved_datasets`, and `dataset_cache_path` to state for the Executor to mount.

#### Node 5: Executor Sandbox (Non-AI) — *Upgraded: Read-Only Cache Volume Mounts*

- **Model:** None — pure Python logic.
- **Responsibility:** Run `experiment.py` in an isolated Docker container and route based on outcome.
- **Docker Hardening (upgraded):**
  - `--network=none` (strict network isolation — no outbound HTTP).
  - `--security-opt=no-new-privileges` (no privilege escalation).
  - `--read-only` filesystem (except `/tmp` for scratch space).
  - `--memory=4g --cpus=2` (cgroup resource limits to prevent OOM kills without stack traces).
  - **Read-only volume mounts for pre-cached dependencies:**
    - `-v .cache/pip:/pip_cache:ro`
    - `-v .cache/hf:/hf_cache:ro`
    - `-v .cache/sklearn:/sklearn_cache:ro`
  - Environment variables set so libraries read from cache: `PIP_FIND_LINKS=/pip_cache`, `HF_DATASETS_CACHE=/hf_cache`, `SCIKIT_LEARN_DATA=/sklearn_cache`.

- **Routing logic:**
  - `exit 0` → write execution logs + `metrics_json` to state → proceed to Phase 3.
  - `exit ≠ 0` and `code_retry_count < 3` → increment `code_retry_count`; feed full logs (including debug prints from active debugging) back to Node 4 for context-aware retry.
  - `exit ≠ 0` and `code_retry_count ≥ 3` → terminate with `failed_execution` status and a failure report.

### Phase 3: Paper Drafting

#### Node 6: Academic Writer (AI)

- **Model:** Claude 3.7 Sonnet (excellent at academic tone and long-context synthesis).
- **System Prompt:** "You are an academic writer. Synthesize the full-text literature, the hypothesis, and the experiment metrics to write an academic paper directly in LaTeX, following the IMRaD structure (Introduction, Methods, Results, Conclusion). You MUST ground your technical claims in the provided Knowledge Graph, but do NOT use raw IDs or inline provenance tags (like '[Source: arXiv:...]'). Instead, use standard LaTeX citation commands (e.g., `\cite{AuthorYear}`) seamlessly in the text. You MUST also generate a corresponding `references.bib` file containing the BibTeX entries for all papers in the Knowledge Graph. Ensure the bibliography is rendered at the bottom of the final NeurIPS paper via `\bibliography{references}`. Do not state information as absolute truth if it cannot be traced back to the literature context."
- **Responsibility:** Synthesize the full-text literature, the hypothesis, and `metrics_json` results to write the first draft of the paper directly in **LaTeX** (`draft.tex`), following the IMRaD structure. Generate a companion `references.bib` with proper BibTeX entries (using paper metadata: authors, year, title, arXiv ID from the state). All citations use `\cite{AuthorYear}` — no raw arXiv IDs in prose. Writes to `latex_draft` and `bibtex_source` in the state.
- **Revision pass (after review):** Addresses each surviving critique from the debate-filtered review, produces `revised_latex`, and appends a **Confidence Score** (self-assessed 1–10) and the NeurIPS reproducibility checklist. Only one mandatory revision pass occurs — unbounded loops lead to model degradation and structural decay.

### Phase 4: Critique & Linting Engine (Automated Quality Assurance) — *Overhauled: Heterogeneous Models + Structured Debate*

> **Design Review Finding:** Using three identical Claude 3.5 Haiku instances creates an "Artificial Hivemind" — agents share identical weights, biases (verbosity bias, self-enhancement bias), and RLHF alignment. They form rapid consensus on superficial critiques rather than catching deep methodological flaws. This is the echo-chamber effect documented in the NeurIPS 2025 Best Paper "Artificial Hivemind."

#### Node 7: Critique Panel (AI — 3 Heterogeneous Agents) — *Overhauled*

Three independent agents read the `draft.tex` and produce structured warnings, using **enforced model diversity** to prevent mode collapse:

| Reviewer | Role | Model | Focus |
|----------|------|-------|-------|
| **Agent A: Fact-Checker** | Verify empirical claims against KG | Claude 3.7 Sonnet | **Algorithmically queries `kg_entities` and `kg_edges` via strict JSON path traversals** — does NOT rely on parametric memory. Its system prompt includes the serialized JSON of the KG. Must cite specific entity IDs and edge relations when verifying/refuting claims. Any claim not traceable to a KG triple is flagged as `ungrounded`. |
| **Agent B: Methodologist** | Evaluate experimental rigor | Claude 3.5 Haiku | Checks if code results in `metrics.json` logically support conclusions. Flags unsupported claims, missing error bars, unjustified generalizations, incorrect statistical reasoning. Uses a **different model** than Agent A to ensure diverse cognitive architecture. |
| **Agent C: Formatter** | Assess structure & LaTeX quality | Claude 3.5 Haiku (different system prompt persona) | Checks for AI-slop writing style, excessive verbosity, missing NeurIPS checklist items, LaTeX structural integrity, citation formatting, figure/table labelling. |

**Structured Debate Protocol** (replaces passive vote aggregation):

1. **Independent critique phase:** Each reviewer independently generates critiques of the draft. Output: `critique_warnings[]` per agent.
2. **Cross-challenge phase:** Each reviewer reads the other two reviewers' critiques. For each critique they disagree with, they issue a formal **challenge** explaining why the critique is incorrect, excessive, or based on a misunderstanding.
3. **Response phase:** The original critic must **defend or retract** each challenged finding with evidence.
4. **Resolution:** Only critiques that **survive the debate** (unretracted after challenge) are forwarded to the Writer for the revision pass. Retracted critiques are logged but not acted upon.

This ensures superficial consensus is broken. The debate log is preserved in `debate_log[]` for auditability.

#### Node 8: Critique Aggregator & Mandatory Revision (Non-AI)

- **Model:** None — pure Python logic.
- **Responsibility:** Collects only the **debate-surviving critiques** (not retracted ones) into a single structured feedback list. Routes them back to **Node 6 (Academic Writer)** for exactly **one mandatory revision pass**. The Writer must address the critique, produce a revised `draft.tex`, and append a "Confidence Score" (self-assessed 1–10) and the NeurIPS reproducibility checklist. No further review rounds occur — the revised draft proceeds directly to the LaTeX Compiler.

### Phase 5: Publication — *Upgraded: LaTeX Compiler Repair Loop*

> **Design Review Finding:** LLM-generated LaTeX is notoriously brittle — unclosed environments, mismatched brackets, floating table overflows, macro expansion conflicts. A single fatal `pdflatex` error terminates the entire pipeline, wasting all compute from Phases 1–4. LaTeX compiler errors (e.g., "Not allowed in LR mode") are confusing to generic LLMs.

#### Node 9: LaTeX Compiler with Repair Loop (Non-AI + AI)

- **Responsibility:** Execute `pdflatex` and `bibtex` to compile the approved LaTeX source into the final NeurIPS-formatted PDF. **If compilation fails, enter an automated repair loop instead of terminating the pipeline.**
- **LaTeX Security:** Uses `subprocess.run(['pdflatex', '--no-shell-escape', 'main.tex'], ...)` to explicitly disable shell escapes, preventing malicious code execution from LLM-generated LaTeX content.

**Repair Loop Architecture:**

```
         ┌─────────────┐
         │  pdflatex    │
         └──────┬───────┘
                │
          ┌─────┴─────┐
          │           │
       success     failure
          │           │
          ▼           ▼
      final.pdf   Parse .log file
                      │
                      ▼
               Extract: line number,
               error description,
               ±5 lines of context
                      │
                      ▼
              ┌───────────────┐
              │ LaTeX Repair  │  (Claude 3.5 Haiku)
              │ Agent         │
              └───────┬───────┘
                      │
                      ▼
               Apply targeted
               line-level patch
                      │
                      ▼
               Re-invoke pdflatex
                      │
              (loop up to 5 times)
                      │
              ┌───────┴───────┐
              │               │
           success          exhausted
              │               │
              ▼               ▼
          final.pdf     failed_latex
                       (failure report)
```

- **LaTeX Log Parser** (deterministic Python): Reads the raw `.log` file produced by `pdflatex`. Extracts: line number, error type, error message, and ±5 lines of surrounding LaTeX context. Does **NOT** feed the entire manuscript back to the LLM — only the localized error context (saves tokens and prevents destabilizing correct sections).

- **LaTeX Repair Agent** (Claude 3.5 Haiku): Receives only the error context (line number + surrounding snippet + compiler error message). Produces a targeted, **line-level patch** (old line → new line). The patch is applied surgically; untouched sections remain stable.

- **Loop:** compile → parse log → repair → compile → ... until success or `max_latex_repair_attempts` (default: 5) is exhausted. If exhausted, pipeline terminates with `failed_latex` status and preserves the `.tex` source for manual inspection.

---

## 3. User Stories & Product Backlog

### 3.1 User Stories (20 total)

| ID    | User Story | Priority | Story Points |
| ----- | ---------- | -------- | ------------ |
| US-01 | As a researcher, I want to enter a natural-language research topic so that the system starts an autonomous pipeline. | Must Have | 3 |
| US-02 | As a researcher, I want the system to retrieve full-text papers from arXiv (not just abstracts) so that the pipeline has deep technical context. | Must Have | 5 |
| US-03 | As a researcher, I want the Deep KG Extractor to build a granular Knowledge Graph with **schema-based extraction and entity deduplication** so that the KG is clean and free of synonymous duplicates. | Must Have | 8 |
| US-04 | As a researcher, I want the Hypothesis Generator to synthesize a testable hypothesis strictly grounded in KG entities so that hallucinated claims are prevented. | Must Have | 8 |
| US-05 | As a researcher, I want the hypothesis to be scored for **mathematical novelty** (via SBERT embedding + Relative Neighbor Density) so that the system does not waste compute on already-established ideas. | Must Have | 5 |
| US-06 | As a researcher, I want the pipeline to **pause and present me with the hypothesis, KG triples, and novelty score for my approval** before proceeding to code generation (HITL Gate). | Must Have | 5 |
| US-07 | As a researcher, I want all generated code to run in a sandboxed Docker container with `--network=none` so that my local machine is protected from arbitrary execution. | Must Have | 5 |
| US-08 | As a researcher, I want a **Dependency Resolver** to AST-parse the generated code, pre-fetch pip packages and HF datasets on the host, and mount them as read-only volumes into the sandbox so that the code runs without network access. | Must Have | 8 |
| US-09 | As a researcher, I want the ML Coder to inject **debug print statements** at key checkpoints so that execution logs contain intermediate state for effective root-cause analysis on retry. | Should Have | 3 |
| US-10 | As a researcher, I want the system to automatically retry failed code up to 3 times with error feedback so that transient or fixable errors are self-healed. | Must Have | 5 |
| US-11 | As a researcher, I want an Academic Writer Agent to draft the paper in LaTeX (IMRaD format) so that I receive a structured first draft. | Must Have | 5 |
| US-12 | As a researcher, I want a **heterogeneous** 3-agent Critique Panel (using diverse model families/personas) to produce structured warnings so that the "Artificial Hivemind" echo-chamber effect is avoided. | Must Have | 8 |
| US-13 | As a researcher, I want the Critique Panel to use a **structured debate protocol** where agents challenge each other's critiques before forwarding them to the Writer, so that only robust, defensible critiques survive. | Must Have | 5 |
| US-14 | As a researcher, I want the Fact-Checker to **algorithmically query the Phase 1 Knowledge Graph** (via JSON path traversals) rather than relying on parametric memory, so that claim verification is mathematically sound. | Must Have | 3 |
| US-15 | As a researcher, I want the Writer to perform one mandatory revision pass addressing debate-surviving critique warnings, appending a Confidence Score and NeurIPS checklist, so that the final paper meets quality standards. | Must Have | 5 |
| US-16 | As a researcher, I want a **LaTeX compiler repair loop** that parses `.log` errors and uses a targeted Repair Agent to apply line-level patches (up to 5 attempts) so that brittle LaTeX does not crash the entire pipeline. | Must Have | 5 |
| US-17 | As a researcher, I want to see a progress log in the terminal showing which agent is currently active so that I can monitor the pipeline's state. | Should Have | 3 |
| US-18 | As a researcher, I want the system to output a failure report if all retries are exhausted so that I understand what went wrong. | Should Have | 3 |
| US-19 | As a researcher, I want to configure which Claude models are used per agent via a config file so that I can optimize cost vs. quality. | Could Have | 2 |
| US-20 | As a developer, I want comprehensive logs of every API call, state transition, and debate round so that I can debug and evaluate the system. | Should Have | 3 |

### 3.2 Product Backlog

The backlog is organized into 5 sprints:

**Sprint 1 — Foundation (Week 1):**
US-01, US-07, US-20 — CLI entry point, Docker sandbox setup, logging infrastructure.

**Sprint 2 — Phase 1 Agents (Week 2):**
US-02, US-03, US-04, US-05, US-06 — Full-text ArXiv retrieval, schema-based KG extraction with entity dedup, hypothesis generation with novelty scoring, HITL Gate.

**Sprint 3 — Phase 2 Experimentation (Week 3):**
US-08, US-09, US-10 — Dependency Resolver (AST parsing + pre-caching), active debugging injection, self-healing loop with context-aware retry.

**Sprint 4 — Draft, Review & Publication (Week 4):**
US-11, US-12, US-13, US-14, US-15, US-16 — Academic Writer, heterogeneous Review Panel with debate protocol, KG-grounded fact-checking, mandatory revision, LaTeX compiler repair loop.

**Sprint 5 — Polish & Config (Week 5):**
US-17, US-18, US-19 — Progress display, failure reports, model configuration, final integration testing.

---

## 4. Diagrams

All diagrams are stored in the repository under the `docs/diagrams/` directory.

### 4.1 Component Architecture Diagram

High-level system components: CLI Interface, LangGraph Orchestrator, Full-Text ArXiv Retriever, Deep KG Extractor (with SBERT clustering + LLM dedup), Hypothesis Generator (with Novelty Scorer), HITL Gate, ML Coder (with debug injection), Dependency Resolver (AST parser + host-side cache), Executor Sandbox (Docker `--network=none` with `:ro` volume mounts), Academic Writer, Heterogeneous Review Panel (3 diverse agents + debate protocol), Critique Aggregator, LaTeX Compiler (with Repair Loop), arXiv API, SBERT Embedding Service, File System Output.

### 4.2 LangGraph Workflow Diagram (State Machine)

```
Phase 1: Deep Literature Parsing & KG
START → [Node 1: Full-Text ArXiv Retriever] → [Node 2: Deep KG Extractor]
                                                        │
                                              schema-based extraction
                                              + SBERT entity clustering
                                              + LLM dedup pass
                                                        │
                                                        ▼
                                              [Node 3: Hypothesis Generator]
                                                        │
                                              ┌─────────┼──────────┐
                                              │         │          │
                                         KG valid  KG invalid  novelty < threshold
                                              │    (hallucin.)        │
                                              │         │             ▼
                                              │    regenerate    END (failed_novelty)
                                              │
HITL Checkpoint                               │
                                    [Node 3b: HITL Gate]
                                              │
                                    ┌─────────┴─────────┐
                                    │                   │
                                 approved            rejected
                                    │                   │
                                    │                   ▼
                                    │           END (failed_hitl_rejected)
                                    │
Phase 2: Experimentation            │
                            [Node 4: ML Coder] ←── (debug injection)
                                    │
                            [Node 4b: Dependency Resolver]
                                    │
                              AST parse → pre-fetch deps
                              + HF datasets on host
                                    │
                            [Node 5: Executor Sandbox]
                              Docker --network=none
                              + .cache mounted as :ro
                                    │
                          ┌─────────┼──────────┐
                          │         │          │
                       success  fail(<3)    fail(≥3)
                          │         │          │
                          │         ▼          ▼
                          │    [Node 4]     END (failed_execution)
                          │    (retry w/    (failure report)
                          │     debug logs)
                          │
Phase 3: Paper Drafting   │
                    [Node 6: Academic Writer]
                          │
Phase 4: Critique         │
               [Node 7: Heterogeneous Review Panel]
                 Agent A: Fact-Checker (Sonnet) — KG-grounded
                 Agent B: Methodologist (Haiku) — diverse model
                 Agent C: Formatter (Haiku) — different persona
                          │
                 Structured Debate Protocol:
                 1. Independent critiques
                 2. Cross-challenge phase
                 3. Defend-or-retract phase
                 4. Only surviving critiques forwarded
                          │
               [Node 8: Critique Aggregator]
                 (debate-surviving warnings only)
                          │
                          ▼
               [Node 6: Academic Writer — Revision Pass]
                 + Confidence Score (1–10)
                 + NeurIPS Reproducibility Checklist
                          │
Phase 5: Publication      │
               [Node 9: LaTeX Compiler]
                          │
                 ┌────────┴────────┐
                 │                 │
              success           failure
                 │                 │
                 ▼            Parse .log → extract
             final.pdf        line + error + context
                                   │
                              [LaTeX Repair Agent]
                              (Claude 3.5 Haiku)
                              line-level patch
                                   │
                              Re-invoke pdflatex
                              (loop up to 5×)
                                   │
                          ┌────────┴────────┐
                          │                 │
                       success          exhausted
                          │                 │
                          ▼                 ▼
                      final.pdf      END (failed_latex)
```

### 4.3 UML Sequence Diagram

Illustrates the message flow between User → CLI → LangGraph → each Node (Full-Text ArXiv Retriever, Deep KG Extractor + SBERT dedup, Hypothesis Generator + novelty scorer, HITL Gate ↔ Human Operator, ML Coder + debug injection, Dependency Resolver + host-side fetch, Executor + Docker with `:ro` mounts, Academic Writer, Heterogeneous Review Panel + debate rounds, Critique Aggregator, LaTeX Compiler + Repair Loop) → external services (arXiv, SBERT, Docker, pdflatex) → PDF output.

### 4.4 Global State Data Model (Class Diagram)

```
┌──────────────────────────────────────────────────────────┐
│                   AutoResearchState                      │
├──────────────────────────────────────────────────────────┤
│  Phase 1: Deep Context & KG                              │
│ + topic: str                                             │
│ + arxiv_papers_full_text: List[Dict]                     │
│ + kg_entities: List[KGEntity]         # deduplicated     │
│ + kg_edges: List[KGEdge]             # resolved          │
│ + hypothesis: str                                        │
│ + hypothesis_embedding: List[float]  # SBERT vector      │
│ + novelty_score: float               # RND metric        │
│ + novelty_passed: bool                                   │
├──────────────────────────────────────────────────────────┤
│  HITL Gate                                               │
│ + hitl_approved: bool                                    │
│ + hitl_rejection_reason: str                             │
├──────────────────────────────────────────────────────────┤
│  Phase 2: Experimentation                                │
│ + python_code: str                                       │
│ + resolved_dependencies: List[str]   # from AST parse    │
│ + resolved_datasets: List[str]       # from AST parse    │
│ + dataset_cache_path: str            # host cache dir    │
│ + debug_instrumentation: str         # augmented code    │
│ + execution_success: bool                                │
│ + execution_logs: str                                    │
│ + metrics_json: str                                      │
│ + code_retry_count: int                                  │
├──────────────────────────────────────────────────────────┤
│  Phase 3 & 4: Drafting, Critique & Debate                │
│ + latex_draft: str                                       │
│ + bibtex_source: str                                     │
│ + critique_warnings: List[Dict]      # per-agent         │
│ + debate_log: List[DebateEntry]      # challenge/defend  │
│ + surviving_critiques: List[Dict]    # post-debate       │
│ + confidence_score: float                                │
│ + revision_pass_done: bool                               │
├──────────────────────────────────────────────────────────┤
│  Phase 5: Compilation & Repair                           │
│ + latex_compile_log: str                                 │
│ + latex_repair_attempts: int                             │
│ + final_pdf_path: str                                    │
├──────────────────────────────────────────────────────────┤
│  Telemetry                                               │
│ + pipeline_status: str               # running |         │
│                                      # awaiting_hitl |   │
│                                      # success |         │
│                                      # failed_novelty |  │
│                                      # failed_hitl |     │
│                                      # failed_execution |│
│                                      # failed_latex      │
│ + total_api_calls: int                                   │
│ + total_tokens_used: int                                 │
│ + logs: List[str]                                        │
├──────────────────────────────────────────────────────────┤
│   TypedDict used as LangGraph State                      │
└──────────────────────────────────────────────────────────┘
```

### 4.5 Deployment / Infrastructure Diagram

Shows: Host machine, Docker daemon (with `--network=none` sandbox + `:ro` volume mounts for `.cache/pip`, `.cache/hf`, `.cache/sklearn`), Python virtual environment, SBERT embedding model (local inference), API calls to Anthropic (multiple agents — heterogeneous models), arXiv REST API + HTML/LaTeX source endpoints, pdflatex/bibtex tools (with repair loop), Rich CLI for HITL interaction, file system I/O (metrics.json, draft.tex, references.bib, debate log, final PDF).

**Deliverable:** All diagrams rendered as `.png` or `.svg` and stored in `docs/diagrams/`. Mermaid source files kept alongside for version control.

---

## 5. Implementation Phases

### Phase 1: Environment & Infrastructure Setup

1. Initialize a clean Python project with `pyproject.toml` or `requirements.txt`.
2. Install core dependencies: `anthropic`, `langgraph`, `arxiv`, `docker`, `datasets`, `huggingface_hub`, `sentence-transformers`, `scikit-learn`, `numpy`, `rich`.
3. Create a `Dockerfile.sandbox` with a base Python image and data science libraries (pandas, scikit-learn, numpy, datasets, huggingface_hub, transformers).
4. Install LaTeX toolchain (`texlive`, `pdflatex`, `bibtex`) in the build environment.
5. Store `ANTHROPIC_API_KEY` securely in a `.env` file (excluded from git via `.gitignore`).

### Phase 2: Define the Global State

```python
from typing import TypedDict, List, Dict, Literal

class KGEntity(TypedDict):
    id: str
    canonical_name: str
    aliases: List[str]
    entity_type: str          # "model" | "dataset" | "metric" | "method" | "hyperparameter"
    attributes: Dict[str, str]

class KGEdge(TypedDict):
    source_id: str
    target_id: str
    relation: str             # "outperforms" | "uses_dataset" | "achieves_metric" | ...
    confidence: float
    provenance: str           # paper ID or section reference

class DebateEntry(TypedDict):
    round: int
    challenger_role: str
    target_critique_index: int
    challenge: str
    response: str
    resolved: bool

class AutoResearchState(TypedDict):
    # Phase 1: Deep Context & KG
    topic: str
    arxiv_papers_full_text: List[Dict]
    kg_entities: List[KGEntity]                # deduplicated via SBERT clustering
    kg_edges: List[KGEdge]                     # resolved (highest-confidence per triple)
    hypothesis: str
    hypothesis_embedding: List[float]          # SBERT vector for novelty computation
    novelty_score: float                       # Relative Neighbor Density
    novelty_passed: bool

    # HITL Gate
    hitl_approved: bool
    hitl_rejection_reason: str

    # Phase 2: Experimentation
    python_code: str
    resolved_dependencies: List[str]           # pip packages from AST parse
    resolved_datasets: List[str]               # HF dataset IDs from AST parse
    dataset_cache_path: str                    # host cache path for :ro mount
    debug_instrumentation: str                 # code with injected print statements
    execution_success: bool
    execution_logs: str
    metrics_json: str
    code_retry_count: int

    # Phase 3 & 4: Drafting, Critique & Debate
    latex_draft: str
    bibtex_source: str
    critique_warnings: List[Dict[str, str]]    # per-agent warnings
    debate_log: List[DebateEntry]              # full debate transcript
    surviving_critiques: List[Dict[str, str]]  # post-debate critiques only
    confidence_score: float
    revision_pass_done: bool

    # Phase 5: Compilation & Repair
    latex_compile_log: str
    latex_repair_attempts: int
    final_pdf_path: str

    # Telemetry
    pipeline_status: str      # running | awaiting_hitl | success | failed_*
    total_api_calls: int
    total_tokens_used: int
    logs: List[str]
```

| State Variable            | Type                    | Description                                                              |
| ------------------------- | ----------------------- | ------------------------------------------------------------------------ |
| `topic`                   | str                     | The user's initial research prompt.                                      |
| `arxiv_papers_full_text`  | List[Dict]              | Full-text methodology, implementation, and results sections.             |
| `kg_entities`             | List[KGEntity]          | **Deduplicated** entities (via SBERT clustering + LLM dedup pass).       |
| `kg_edges`                | List[KGEdge]            | **Resolved** edges (highest-confidence per unique triple).               |
| `hypothesis`              | str                     | The testable hypothesis, grounded in KG entities.                        |
| `hypothesis_embedding`    | List[float]             | SBERT embedding vector of the hypothesis for novelty scoring.            |
| `novelty_score`           | float                   | Relative Neighbor Density — semantic distance from existing literature.  |
| `novelty_passed`          | bool                    | Whether the hypothesis crossed the novelty threshold.                    |
| `hitl_approved`           | bool                    | Whether the human operator approved the hypothesis.                      |
| `hitl_rejection_reason`   | str                     | Reason given if the human rejected.                                      |
| `python_code`             | str                     | The Python experiment script generated by the ML Coder.                  |
| `resolved_dependencies`   | List[str]               | pip packages identified by AST-parsing the generated code.               |
| `resolved_datasets`       | List[str]               | HF/sklearn dataset IDs identified by AST-parsing the generated code.     |
| `dataset_cache_path`      | str                     | Host-side cache directory mounted as `:ro` into the Docker sandbox.      |
| `debug_instrumentation`   | str                     | Code augmented with strategic debug `print()` statements.                |
| `execution_success`       | bool                    | Whether the experiment executed successfully.                            |
| `execution_logs`          | str                     | Stdout/Stderr output from the Docker sandbox (includes debug prints).    |
| `metrics_json`            | str                     | JSON string with experiment results (accuracy, F1-score, etc.).          |
| `code_retry_count`        | int                     | Tracks code retry attempts (max 3).                                      |
| `latex_draft`             | str                     | The LaTeX draft source code (`draft.tex`).                               |
| `bibtex_source`           | str                     | The generated bibliography (`references.bib`).                           |
| `critique_warnings`       | List[Dict]              | Structured warnings from the 3-agent heterogeneous critique panel.       |
| `debate_log`              | List[DebateEntry]       | Full transcript of the structured debate (challenges + responses).       |
| `surviving_critiques`     | List[Dict]              | Only critiques that survived the debate protocol (not retracted).        |
| `confidence_score`        | float                   | Writer's self-assessed confidence score (1–10) after revision.           |
| `revision_pass_done`      | bool                    | Whether the single mandatory revision pass has been completed.           |
| `latex_compile_log`       | str                     | Raw `.log` output from `pdflatex` (used by repair loop).                 |
| `latex_repair_attempts`   | int                     | Number of repair iterations attempted (max 5).                           |
| `final_pdf_path`          | str                     | Path to the compiled NeurIPS-formatted PDF.                              |
| `pipeline_status`         | str                     | Current status: running, awaiting_hitl, success, or failed_*.            |

### Phase 3: Build the Nodes

Implement each of the 11 nodes as described in Section 2 above, each in its own Python module under `backend/agents/`:

- `arxiv_retriever.py` — Full-text ArXiv paper download and section extraction (non-AI)
- `kg_extractor.py` — Schema-based KG extraction + SBERT entity clustering + LLM dedup (AI)
- `hypothesis_generator.py` — Hypothesis generation with KG grounding + SBERT novelty scoring (AI)
- `hitl_gate.py` — Human-in-the-loop approval checkpoint (non-AI)
- `ml_coder.py` — Experiment code generation with active debug injection (AI)
- `dependency_resolver.py` — AST parsing + host-side dependency/dataset pre-fetch (non-AI)
- `executor.py` — Docker sandbox runner with `:ro` cache volume mounts (non-AI)
- `academic_writer.py` — LaTeX/BibTeX draft generation + revision pass (AI)
- `critique_panel.py` — 3-agent heterogeneous critique with structured debate protocol (AI)
- `critique_aggregator.py` — Debate-surviving warning aggregation and mandatory revision routing (non-AI)
- `latex_compiler.py` — PDF compilation + LaTeX Repair Agent loop (non-AI + AI)

Shared utilities under `backend/utils/`:
- `embeddings.py` — SBERT embedding + novelty (RND) computation
- `kg_utils.py` — KG entity clustering + deduplication logic
- `ast_parser.py` — AST-based dependency/dataset extraction from generated code
- `latex_utils.py` — pdflatex log parsing (line number + error + context extraction)
- `docker_utils.py` — Docker container lifecycle management with `:ro` volume mounts

### Phase 4: Orchestration with LangGraph

1. Register the 11 nodes as graph nodes.
2. Define Phase 1 edges: START → Full-Text ArXiv Retriever → Deep KG Extractor (with post-processing dedup) → Hypothesis Generator (with novelty scoring).
3. Define conditional edges at Hypothesis Generator:
   - KG validation passes AND novelty passes → HITL Gate.
   - KG validation fails (hallucination) → regenerate hypothesis.
   - Novelty score below threshold → END (failed_novelty report).
4. Define conditional edge at HITL Gate:
   - Human approves → ML Coder.
   - Human rejects → END (failed_hitl_rejected report).
5. Define Phase 2 edges: ML Coder → Dependency Resolver → Executor.
6. Define conditional edges at Executor:
   - Success → Academic Writer.
   - Failure (code_retry_count < 3) → ML Coder (self-healing loop with debug logs).
   - Failure (code_retry_count ≥ 3) → END (failure report).
7. Define Phase 3 edge: Academic Writer → Critique Panel.
8. Define Phase 4 edges: Critique Panel (with debate) → Critique Aggregator → Academic Writer (one mandatory revision pass).
9. Define Phase 5 edges: Academic Writer (revised) → LaTeX Compiler (with repair loop) → END.
10. Compile and expose the graph via a `run_pipeline(topic: str)` function.

### Phase 5: Testing & Iteration

1. **Happy Path:** "Compare the accuracy of a Random Forest vs. Logistic Regression on the Hugging Face `imdb` sentiment dataset."
2. **Forced Failure:** Inject a deliberate error (e.g., reference a non-installed library) to validate the self-healing retry loop.
3. **Novelty Rejection:** Supply a hypothesis nearly identical to existing literature and verify the novelty gate blocks it.
4. **HITL Rejection:** Reject the hypothesis at the HITL gate and verify the pipeline terminates gracefully.
5. **LaTeX Repair:** Inject a deliberate LaTeX error (unclosed environment) and verify the repair loop fixes it.
6. **Debate Protocol:** Submit a draft with a hallucinated claim and verify the Fact-Checker flags it and the claim survives/fails the debate.

---

## 6. Source Control Strategy (Git)

### 6.1 Repository Structure

```
mini-research-claw/
├── src/                              # React frontend (Vite + Tailwind v4)
│   ├── App.jsx
│   ├── main.jsx
│   ├── index.css
│   ├── components/
│   │   ├── Sidebar.jsx
│   │   ├── PipelineStepper.jsx
│   │   ├── LogPanel.jsx
│   │   └── StatCard.jsx
│   └── pages/
│       ├── Dashboard.jsx
│       ├── NewResearch.jsx
│       ├── History.jsx
│       ├── Logs.jsx
│       └── Settings.jsx
│
├── backend/                          # Python pipeline (LangGraph)
│   ├── __init__.py
│   ├── main.py                       # CLI entry point
│   ├── state.py                      # AutoResearchState TypedDict
│   ├── graph.py                      # LangGraph DAG definition + routing
│   ├── config.py                     # Environment / model configuration
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── arxiv_retriever.py       # Node 1 — full-text retrieval
│   │   ├── kg_extractor.py          # Node 2 — schema-based extraction + dedup
│   │   ├── hypothesis_generator.py  # Node 3 — hypothesis + novelty scoring
│   │   ├── hitl_gate.py             # Node 3b — human approval checkpoint
│   │   ├── ml_coder.py             # Node 4 — code generation + debug injection
│   │   ├── dependency_resolver.py   # Node 4b — AST parsing + host-side pre-fetch
│   │   ├── executor.py              # Node 5 — Docker sandbox with :ro mounts
│   │   ├── academic_writer.py       # Node 6 — LaTeX drafting + revision pass
│   │   ├── critique_panel.py        # Node 7 — heterogeneous debate protocol
│   │   ├── critique_aggregator.py   # Node 8 — debate-surviving warning filter
│   │   └── latex_compiler.py        # Node 9 — pdflatex + repair loop
│   └── utils/
│       ├── __init__.py
│       ├── embeddings.py            # SBERT embedding + novelty (RND) computation
│       ├── kg_utils.py              # KG entity clustering + deduplication
│       ├── ast_parser.py            # AST-based dependency/dataset extraction
│       ├── latex_utils.py           # pdflatex log parsing
│       └── docker_utils.py          # Docker container lifecycle + :ro mounts
│
├── tests/
│   ├── test_arxiv_retriever.py
│   ├── test_kg_extractor.py
│   ├── test_hypothesis_generator.py
│   ├── test_hitl_gate.py
│   ├── test_ml_coder.py
│   ├── test_dependency_resolver.py
│   ├── test_executor.py
│   ├── test_academic_writer.py
│   ├── test_critique_panel.py
│   ├── test_critique_aggregator.py
│   ├── test_latex_compiler.py
│   └── evals/
│       ├── eval_kg_extractor.py
│       ├── eval_hypothesis.py
│       ├── eval_novelty.py
│       ├── eval_coder.py
│       ├── eval_writer.py
│       ├── eval_critique_panel.py
│       └── eval_latex_repair.py
│
├── docs/
│   ├── diagrams/
│   └── ai-usage-report.md
│
├── templates/
│   └── neurips_template.tex
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile                        # Frontend (nginx)
├── Dockerfile.sandbox                # Code execution sandbox (Python + ML libs)
├── docker-compose.yml
├── nginx.conf
├── requirements.txt                  # Python backend dependencies
├── package.json                      # Frontend dependencies
├── .env.example
├── .gitignore
└── README.md
```

### 6.2 Branching Strategy

- `main` — stable, production-ready code. Protected branch requiring pull request reviews.
- `develop` — integration branch for feature merges.
- `feature/<name>` — one branch per user story or feature (e.g., `feature/kg-dedup`, `feature/hitl-gate`, `feature/debate-protocol`).
- `bugfix/<id>` — one branch per reported bug (e.g., `bugfix/BUG-003-retry-overflow`).

### 6.3 Commit Requirements

- Conventional commit style with frequent, meaningful commits.
- Conventional commit messages: `feat:`, `fix:`, `test:`, `docs:`, `ci:`.
- Every feature branch merged via **Pull Request** with at least one reviewer.
- Use **rebase** to keep feature branches up to date with `develop`; use **merge commits** when merging PRs into `develop`.

---

## 7. Automated Testing & Agent Evals

### 7.1 Unit Tests

| Test File                        | What It Tests                                                                          |
| -------------------------------- | -------------------------------------------------------------------------------------- |
| `test_arxiv_retriever.py`        | arXiv search returns full-text results; methodology/results sections are extracted.     |
| `test_kg_extractor.py`           | KG output uses schema-based typed entities; SBERT dedup merges synonyms correctly.      |
| `test_hypothesis_generator.py`   | Hypothesis is non-empty; grounded in KG; novelty score is computed and gated.           |
| `test_hitl_gate.py`              | Pipeline pauses correctly; approve/reject flows work; state fields updated properly.    |
| `test_ml_coder.py`               | Generated code is syntactically valid; contains debug print instrumentation.            |
| `test_dependency_resolver.py`    | AST parser correctly identifies imports and `load_dataset()` calls; cache paths set.    |
| `test_executor.py`               | Docker container starts with `--network=none` + `:ro` mounts; exit codes captured.     |
| `test_academic_writer.py`        | LaTeX draft is valid; contains IMRaD sections; BibTeX file is well-formed.              |
| `test_critique_panel.py`         | 3 agents use different models/personas; debate protocol produces challenges/responses.  |
| `test_critique_aggregator.py`    | Only debate-surviving (unretracted) critiques are forwarded to the Writer.              |
| `test_latex_compiler.py`         | PDF generated on success; repair loop triggered on failure; max attempts respected.     |

### 7.2 Integration Tests

- **End-to-end pipeline test:** Run the full graph on a known-good topic (with auto-approved HITL) and assert that `final_pdf_path` points to a valid PDF.
- **Code retry loop test:** Feed a deliberately broken `python_code`, assert that `code_retry_count` increments and the ML Coder is re-invoked with debug logs.
- **Anti-hallucination test:** Provide a hypothesis with entities not in the KG and assert that it is rejected and regenerated.
- **Novelty gate test:** Supply a hypothesis nearly identical to existing literature and assert `novelty_passed = False` and pipeline terminates with `failed_novelty`.
- **HITL rejection test:** Reject the hypothesis at the HITL gate and assert pipeline terminates with `failed_hitl_rejected`.
- **Debate protocol test:** Submit a draft with a hallucinated citation and assert the Fact-Checker flags it, the debate challenges it, and it survives into `surviving_critiques`.
- **LaTeX repair test:** Submit a `.tex` with a deliberate unclosed `\begin{table}` and assert the repair loop fixes it within 5 attempts.
- **Confidence score test:** Assert that the revised draft includes a confidence score (1–10) and the NeurIPS reproducibility checklist.

### 7.3 Agent Evals (LLM-Specific)

These evals measure agent quality beyond pass/fail:

| Eval                                 | Method                                                                                                    | Pass Criteria                          |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| **KG Extractor: Schema Compliance**  | Verify KG output conforms to typed KGEntity/KGEdge schemas with all required fields.                      | 100% valid structure on 10/10 runs     |
| **KG Extractor: Dedup Quality**      | Inject synonymous entities ("RF", "Random Forest") and verify SBERT clustering merges them.               | ≥ 90% synonyms correctly merged        |
| **KG Extractor: Depth**              | Verify KG contains granular triplets (hyperparams, preprocessing steps), not just abstract concepts.      | ≥ 5 technical-detail triplets per paper |
| **Hypothesis: Anti-Hallucination**   | Verify all entities in hypothesis exist in KG triplets. Inject fake entities and confirm rejection.       | 100% hallucinated hypotheses rejected  |
| **Hypothesis: Novelty Scoring**      | Test with known-novel and known-redundant hypotheses; verify RND scores separate them.                    | AUC ≥ 0.80 on novel vs. redundant     |
| **Coder: Syntax Validity**           | Parse `python_code` with `ast.parse()`.                                                                   | No `SyntaxError` on 10/10 runs        |
| **Coder: Debug Instrumentation**     | Verify generated code contains strategic `print()` at data load, train, and eval checkpoints.             | ≥ 3 debug prints per script            |
| **Coder: ML Rigor**                  | Verify generated code contains train/test split, random seeds, and cross-validation.                      | All 3 practices present on 10/10 runs  |
| **Dependency Resolver: Accuracy**    | Test AST parser against 10 sample scripts; verify all imports and `load_dataset` calls are captured.      | 100% recall on known dependencies      |
| **Writer: LaTeX Validity**           | Compile LaTeX draft with `pdflatex` and verify PDF is generated without errors.                           | Successful compilation on 10/10 runs   |
| **Writer: Structure**                | Regex-check that LaTeX output contains `\section{Introduction}`, `\section{Methods}`, `\section{Results}`, `\section{Conclusion}`. | All 4 sections present |
| **Critique Panel: Diversity**        | Verify the 3 agents produce substantively different critiques (not echo-chamber duplicates).              | ≤ 30% overlap in critique content      |
| **Critique Panel: Debate Survival**  | Submit obvious and trivial critiques; verify trivial ones are retracted via debate.                       | ≥ 60% trivial critiques retracted      |
| **Critique Panel: KG Grounding**     | Submit a draft with hallucinated citations; verify Fact-Checker flags them via KG JSON path query.        | 100% hallucinated citations detected   |
| **LaTeX Repair: Fix Rate**           | Inject 10 common LaTeX errors; verify repair loop fixes them.                                             | ≥ 7/10 errors fixed within 5 attempts  |
| **Critique Panel: Revision Quality** | Compare pre-revision and post-revision drafts; use Claude to score improvement (1–5 scale).              | Average improvement ≥ 3.0/5            |

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
- Show the terminal output as each node activates in sequence.
- Show the **deduplicated Knowledge Graph** (with SBERT-clustered entities) and how the hypothesis is grounded in it.
- Show the **novelty score** computation and how it compares to the threshold.
- Demonstrate the **HITL Gate**: show the Rich-formatted approval prompt with hypothesis, KG triples, and novelty score.
- Show the **Dependency Resolver** output: AST-parsed imports and datasets, host-side cache, `:ro` volume mounts.
- Show the **heterogeneous Critique Panel** warnings from each agent (Fact-Checker via KG query, Methodologist, Formatter).
- Demonstrate the **structured debate protocol**: show challenges, responses, and which critiques survive.
- Show the **LaTeX repair loop** in action: trigger a compilation error and watch the Repair Agent fix it.
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
| Design Review & Overhauls    | Claude                                 | Analyzed architectural evaluation PDF; implemented 6 critical overhauls to address systemic bottlenecks. |
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
| 11 Nodes (7 AI-powered + 4 non-AI) in 5 Phases + HITL | §2 | Planned |
| Schema-Based KG Extraction + SBERT Dedup (Anti-Fragmentation) | §2 (Node 2) | Planned |
| Mathematical Novelty Detection (SBERT + RND Scoring) | §2 (Node 3) | Planned |
| Human-in-the-Loop Approval Gate | §2 (Node 3b) | Planned |
| AST-Based Dependency Resolver (Network-Isolation Fix) | §2 (Node 4b) | Planned |
| Active Debugging Injection in ML Coder | §2 (Node 4) | Planned |
| Docker Sandbox with `:ro` Cache Volume Mounts | §2 (Node 5) | Planned |
| Heterogeneous Review Panel + Structured Debate Protocol | §2 (Node 7) | Planned |
| KG-Grounded Fact-Checking (JSON Path Traversals) | §2 (Node 7, Agent A) | Planned |
| LaTeX Compiler Repair Loop (up to 5 attempts) | §2 (Node 9) | Planned |
| User Stories (20) + Product Backlog (5 sprints) | §3 | Planned |
| Diagrams (5 total) | §4 | Planned |
| Git Strategy (branches, PRs, conventional commits) | §6 | Planned |
| Automated Tests (11 unit + 8 integration) + Agent Evals (17 evals) | §7 | Planned |
| Bug Reporting + Resolution via PR | §8 | Planned |
| CI/CD Pipeline (GitHub Actions) | §9 | Planned |
| NeurIPS PDF Generation (LaTeX + BibTeX + Repair Loop) | §2 (Phase 5) | Planned |
| AI Usage Report (11 areas + metrics) | §11 | Planned |
| Live Demo + Screencast | §10 | Planned |

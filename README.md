# Auto-Mini-Claw

**An autonomous research assistant that writes peer-reviewed ML papers — with you in the loop.**

Give it a topic. It searches the literature, builds a knowledge graph of what's known
(and what's contested), proposes a falsifiable hypothesis, asks for your approval,
designs an experiment, asks again, writes and runs the code in a sandbox, drafts the
paper in LaTeX, runs it through a three-agent peer-review debate, and compiles a
NeurIPS-formatted PDF. When the evidence is too weak, it refuses to write the paper
rather than hallucinate one.

---

## Demo

> A 2-minute walkthrough lives in [`demo01.mp4`](demo01.mp4)
> *(use Git LFS to clone the video, or watch it on the cloud link in the project notes)*.

Below: the React control room — pipeline progress on the left, real-time stage stepper
in the middle, generated artifacts on the right.

```
┌────────────────────────────────────────────────────────────────────────┐
│  Auto-Mini-Claw · Research Control Room                                │
├──────────────────────┬─────────────────────────────────────────────────┤
│  ● Pipeline running  │   Stage 3 / 8 — Hypothesis Generation           │
│  ● API connected     │   [▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░] 38%             │
│  ● Run d933be1...    │   Reviewer pause incoming in ~20 s              │
├──────────────────────┴─────────────────────────────────────────────────┤
│  Literature →  KG  →  Hypothesis  →  Design  →  Code  →  Paper  → PDF  │
│   [done]      [done]  [active]      [...]     [...]    [...]    [...]  │
└────────────────────────────────────────────────────────────────────────┘
```

When the run is done you get a **download panel** with the compiled PDF, LaTeX source,
metrics, debate log, and claim ledger — all stored in Supabase, all one click away.

---

## What makes it different

Most "autonomous paper writer" demos either:
- Hallucinate, because no one verifies their claims, or
- Get stuck in echo-chamber loops where one LLM critiques another LLM's output.

Auto-Mini-Claw is built around six choices that fix those failure modes:

| Design choice | What it prevents |
|---|---|
| **Epistemic Knowledge Graph with polarity** — every claim from every paper is tagged `supports`, `contradicts`, or `neutral`; contradictions are preserved, not averaged away | Pretending the literature agrees when it doesn't |
| **Novelty scoring + prior-art screening** — every hypothesis is embedded and compared against the entire historical corpus before you ever see it | Re-running experiments that already exist |
| **Two Human-in-the-Loop gates** — you approve the hypothesis *and* the experiment design before any code runs or any GPU cycle is spent | Wasting compute on bad experiments |
| **Constrained sandbox execution** — generated code runs in a Docker container with `--network=none`, read-only mounts, no shell escape, no privilege escalation | Generated code touching your machine |
| **Heterogeneous peer-review panel with structured debate** — three reviewers (fact-checker, methodologist, formatter) each on different models; they challenge each other's critiques before any feedback reaches the writer | Echo-chamber consensus |
| **Claim ledger + No-Paper exit** — every assertion in the final paper is traced back to KG evidence; if >50% of claims are weak/unsupported, the pipeline refuses to draft | Generating papers without evidence |

---

## The pipeline (14 nodes, 5 phases)

```
                          ┌─────────────────────────────────────┐
                          │     PHASE 1 — Literature & KG       │
                          │  ArXiv (iterative) → KG Extractor   │
                          │       → Hypothesis Generator        │
                          └────────────────┬────────────────────┘
                                           │
                                  ◆ HITL GATE 1: approve hypothesis?
                                           │
                          ┌────────────────▼────────────────────┐
                          │     PHASE 2 — Experiment Design     │
                          │   Experiment Designer (ExpSpec)     │
                          └────────────────┬────────────────────┘
                                           │
                                  ◆ HITL GATE 2: approve experiment?
                                           │
                          ┌────────────────▼────────────────────┐
                          │     PHASE 3 — Sandbox Execution     │
                          │  ML Coder → Dep Resolver → Docker   │
                          │     (self-heals on failure)         │
                          └────────────────┬────────────────────┘
                                           │
                          ┌────────────────▼────────────────────┐
                          │     PHASE 4 — Drafting & Critique   │
                          │  Claim Ledger → Writer → Linter →   │
                          │   3-agent Debate → Revision         │
                          └────────────────┬────────────────────┘
                                           │
                          ┌────────────────▼────────────────────┐
                          │     PHASE 5 — Publication           │
                          │   pdflatex + bibtex + repair loop   │
                          │   → final NeurIPS PDF + artifacts   │
                          └─────────────────────────────────────┘
```

Each AI node sees only the slice of state it needs ("scoped views") — keeps reasoning
focused and API costs bounded.

---

## Generated paper examples

Three real papers produced end-to-end by the pipeline. The hypotheses, experiments,
results, and writeups are all autonomously generated — the only human inputs were the
topic and the two HITL approvals.

| # | Topic | PDF | Notes |
|---|---|---|---|
| 1 | k-means with different distance metrics on iris | [001_kmeans_iris.pdf](papers/001_kmeans_iris_distance_metrics.pdf) | First success — clean run, confidence 6.5 |
| 2 | Logistic-regression threshold sweep on breast cancer | [002_breast_cancer_pareto.pdf](papers/002_breast_cancer_threshold_pareto_RESCUED.pdf) | Confidence 7.2 — pareto analysis of precision/recall/FNR |
| 3 | Learning curves of Gradient Boosting on wine | [003_wine_gbdt_learning_curves.pdf](papers/003_wine_gbdt_learning_curves.pdf) | Honest paper — 3 of 4 hypothesis predictions were *falsified*, and the writer says so |

See [papers/README.md](papers/README.md) for full per-paper details (CLI input,
generated hypothesis, experiment spec, key findings).

---

## Try it yourself

### Prerequisites
- Docker + Docker Compose
- Anthropic API key (for the LLM agents)
- A free Supabase project (for the paper cache, run history, and artifact storage)

### Get running

```bash
git clone https://github.com/HanguCalin/mini-research-claw.git
cd mini-research-claw

cp .env.example .env
# Fill in ANTHROPIC_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY

docker compose up --build -d
```

Open `http://localhost:3000`, click **New Research**, paste a topic, hit **Launch
Pipeline**. The pipeline will pause at two HITL gates for your approval before any
compute is spent, then run to completion (or terminate cleanly with a `no_paper`
result if the evidence is too weak).

### Good first topics (all sklearn-native, all run in <20 minutes)

- *"Effect of `max_depth` on Decision Tree classifier accuracy on sklearn iris with 5-fold stratified CV"*
- *"Compare Bagging, AdaBoost, and Gradient Boosting on sklearn digits — accuracy, training time, and overfitting gap"*
- *"Detecting anomalies via Isolation Forest, Local Outlier Factor, and One-Class SVM on sklearn breast_cancer"*

### What you'll see when the run finishes

**Success path** — a green download panel with the compiled paper at the top, then
LaTeX source, metrics, claim ledger, debate log, and the Python script the sandbox ran.

**Failure path** — a rose-coloured diagnostics panel with `failure_report.json` first,
then the partial artifacts (execution logs, the script the coder tried, the hypothesis
and experiment spec). Every failure is debuggable, never silent.

---

## Tech stack

- **Orchestration:** LangGraph (14-node DAG with conditional routing)
- **Models:** Claude Sonnet 4.6 (reasoning), Claude Haiku 4.5 (structured/cheap)
- **Search & cache:** arXiv API + Supabase PostgreSQL with pgvector
- **Embeddings:** SBERT (`all-MiniLM-L6-v2`, 384-dim)
- **Sandbox:** Docker (`--network=none`, read-only mounts, no shell escape)
- **Paper compile:** TeXLive (pdflatex + bibtex) with deterministic missing-graphics pre-pass and LLM-driven repair loop
- **Backend:** FastAPI + Uvicorn
- **Frontend:** React 19 + Vite + Tailwind v4
- **Artifact storage:** Supabase Storage (organised by `run_id`)

---

## Project status

Implemented end-to-end:
- All 14 pipeline nodes
- Both HITL gates (CLI and Web UI)
- Sandbox execution with dependency pre-fetching
- Claim ledger with No-Paper gate
- Three-agent peer-review panel with structured debate
- Deterministic linter
- LaTeX compiler with repair loop
- Artifact upload + per-run download panel in the UI
- Per-run model and threshold overrides through the API

Out of scope for this iteration:
- Multi-user authentication (single-operator design today)
- Automated test suite (planned, not built)
- GPU support inside the sandbox

---

## Documentation

This README is the friendly front door. If you're going deeper:

- [`Mini_Research_Claw_Full_Plan.md`](Mini_Research_Claw_Full_Plan.md) — the full design document: every node, every state field, every routing rule, every reviewer prompt
- [`docs/IMPLEMENTATION_GUIDE.md`](docs/IMPLEMENTATION_GUIDE.md) — the step-by-step build guide (environment setup, Supabase schema, node-by-node walkthrough, LangGraph wiring)
- [`papers/README.md`](papers/README.md) — per-paper provenance for the three successful end-to-end runs

---

## Why it's called "Mini-Claw"

"Claw" because the system *grasps* literature, evidence, and contradictions before
forming a hypothesis. "Mini" because it's a single-operator desktop pipeline — not a
multi-tenant SaaS — built to run on one machine, with one researcher, generating one
paper at a time, end-to-end, in a few minutes.

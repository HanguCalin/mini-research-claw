# Auto-Mini-Claw (Epistemic Research & Peer-Review Edition)

## Full Implementation & Development Process Plan

This document outlines the complete plan for building a production-grade autonomous research pipeline. The user inputs a research topic, and the system orchestrates multiple AI agents via **LangGraph** to conduct **iterative** full-text literature retrieval, build an **epistemic Knowledge Graph that preserves contradictions**, generate an **incremental novelty-scored hypothesis** with prior-art similarity screening, pass through a **double Human-in-the-Loop gate** (hypothesis approval + experiment design approval), execute data science experiments in a network-isolated sandbox via a **constrained ML Coder** bound to an approved experiment spec, draft an academic paper in LaTeX grounded in a **claim ledger** that traces every assertion to evidence, subject it to a **heterogeneous multi-agent peer review with structured debate** and a **deterministic linter**, and compile the final NeurIPS-formatted PDF with an automated **LaTeX repair loop**. The pipeline can terminate with a principled **No-Paper outcome** when evidence is insufficient.

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

Auto-Mini-Claw is an autonomous research assistant that takes a natural-language research topic from a user, **iteratively** retrieves and parses full-text academic literature, builds a **deduplicated, epistemic Knowledge Graph** that preserves contradictions and polarity between findings, generates a testable **incremental hypothesis** validated for **mathematical novelty** and screened against prior art, pauses for **double human approval** (hypothesis + experiment design), writes and executes experiments inside a **network-isolated Docker sandbox** (with pre-cached dependencies) via a **constrained ML Coder** bound to the approved experiment spec, drafts an academic paper in LaTeX grounded in a **claim ledger** mapping every assertion to KG evidence, subjects it to a **heterogeneous multi-agent peer review with structured debate** and a **deterministic linter** (dismantling the "Artificial Hivemind" problem), and compiles the final NeurIPS-formatted PDF with an automated **LaTeX compiler repair loop**. When evidence is insufficient, the pipeline terminates with a principled **No-Paper outcome** rather than generating unsupported claims.

The system uses LangGraph for multi-agent orchestration with **14 sequential nodes** organized in 5 phases plus two HITL checkpoints, leveraging multiple Claude models optimized per task. Nine critical architectural upgrades — informed by rigorous epistemic peer review — address systemic execution bottlenecks: sparse KG extraction, loss of scientific contradictions, lack of novelty verification, absence of human oversight over experiment design, unconstrained code generation, the network-isolation contradiction in sandboxed execution, echo-chamber consensus in homogeneous review panels, missing claim traceability, and brittle LaTeX compilation without error recovery.

**Key Constraint:** This project is a CLI/desktop-based AI agent pipeline.

---

## 2. Multi-Agent Architecture (DAG Execution Flow)

The system consists of **14 sequential nodes**, with **8 AI-powered agents** and **6 non-AI nodes**, organized in **5 phases plus two HITL checkpoints**.

### Context Management (State Pruning)

> **Systems Engineering Finding:** The `AutoResearchState` accumulates large payloads across the pipeline — raw full-text papers (`arxiv_papers_full_text`, often 50K+ tokens across 5 papers), verbose execution logs (`execution_logs`, up to 10K tokens), the full KG, and the claim ledger. Passing the **entire** state into every LLM-powered node causes two critical problems:
>
> 1. **"Lost in the middle" attention degradation:** LLMs demonstrably lose accuracy on information buried in the middle of long contexts (Liu et al., 2023). When the ML Coder receives 50K tokens of raw paper text it never needs, the actually relevant `experiment_spec` gets pushed into the attention trough, degrading code quality.
> 2. **API cost inflation:** Every unnecessary token in the prompt is billed. Passing raw papers to 5 downstream nodes that never read them multiplies cost by 5× for zero benefit.

**Solution: Scoped State Views.** Each AI-powered node receives a **pruned view** of the state — only the fields it actually needs to do its job. The LangGraph orchestrator (in `graph.py`) constructs these scoped views before invoking each node, using a simple field-inclusion allowlist per node. The full `AutoResearchState` is always preserved in memory for telemetry and debugging — only the LLM prompt payload is pruned.

**Scoped Views per Node:**

| Node | Receives (Scoped View) | Excluded (Not Sent to LLM) |
|------|----------------------|---------------------------|
| **Node 2: Epistemic KG Extractor** | `arxiv_papers_full_text` (one paper at a time, not all at once) | All other state fields |
| **Node 3: Hypothesis Generator** | `kg_entities`, `kg_edges`, `topic` | `arxiv_papers_full_text` (already distilled into KG) |
| **Node 3c: Experimental Designer** | `hypothesis`, `incremental_delta`, `kg_entities`, `kg_edges` | `arxiv_papers_full_text`, `execution_logs` |
| **Node 4: Constrained ML Coder** | `experiment_spec`, `hypothesis` | `arxiv_papers_full_text`, `kg_entities`, `kg_edges` (experiment spec is the contract — raw context is unnecessary) |
| **Node 4 (retry)** | `experiment_spec`, `hypothesis`, `python_code` (previous), `execution_logs` (for debugging) | `arxiv_papers_full_text`, `kg_entities`, `kg_edges` |
| **Node 6: Academic Writer** | `claim_ledger`, `experiment_spec`, `metrics_json`, `incremental_delta`, `hypothesis` | `arxiv_papers_full_text`, `execution_logs`, `python_code` |
| **Node 7: Critique Panel** | `latex_draft`, `bibtex_source`, `metrics_json`, `claim_ledger` | `arxiv_papers_full_text`, `execution_logs`, `python_code`, `kg_entities` (claim ledger already contains the KG-grounded evidence) |
| **Node 9: LaTeX Repair Agent** | Error context only (line number + ±5 lines + compiler error) | Everything else (already scoped by the LaTeX Log Parser) |

**Implementation:** The orchestrator wraps each node invocation with a `build_scoped_view(state, node_name)` utility that reads from a `NODE_SCOPE_CONFIG` dictionary mapping node names to their allowed state fields. This is purely a prompt-construction concern — the underlying LangGraph state object is never mutated.

### Phase 1: Iterative Literature Parsing & Epistemic Knowledge Graph (Data Grounding)

#### Node 1: Iterative Full-Text ArXiv Retriever (Non-AI) — *Upgraded: Hypothesis-Driven Refinement*

> **Epistemic Review Finding:** A single retrieval pass based only on the user's prompt misses domain-specific papers that become relevant once a hypothesis takes shape. Iterative retrieval — re-querying arXiv with hypothesis-derived terms after each KG/hypothesis cycle — significantly improves literature coverage.

- **Model:** None — pure Python logic.
- **Tools:** A Python script using the `arxiv` library and arXiv's HTML endpoints or LaTeX source parsing.
- **Responsibility:** Query arXiv based on the user's prompt (round 1) or hypothesis-derived search terms (rounds 2+). Instead of stopping at abstracts, it fetches the full text of the top 3–5 papers per round and extracts the `Methodology`, `Implementation`, and `Results` sections. Writes results to `arxiv_papers_full_text` in the state and increments `retrieval_round`.

- **Why Iterative?** The user's initial prompt is typically broad and informal (e.g., "sentiment analysis with transformers"). The first retrieval round discovers general papers, but the KG Extractor and Hypothesis Generator downstream will introduce specific technical concepts (e.g., "LoRA fine-tuning", "attention head pruning") that the user never mentioned. A second retrieval pass using these refined terms discovers highly relevant papers that the initial broad query could never surface. Without iteration, the pipeline operates on an incomplete literature picture, leading to hypotheses that either reinvent existing work or miss important prior art.

- **Iterative Protocol:**

  1. **Round 1 (topic-driven):** The retriever takes the user's raw `topic` string (e.g., "Compare Random Forest vs. XGBoost on tabular data") and constructs an arXiv API query. It fetches the top 3–5 matching papers, downloads their full text (via arXiv HTML endpoints or LaTeX source), and extracts the `Methodology`, `Implementation`, and `Results` sections using regex-based section parsers. Results are appended to `arxiv_papers_full_text[]` with metadata (authors, year, title, arXiv ID). Sets `retrieval_round = 1`.

  2. **Round 2 (hypothesis-driven):** After Node 2 (KG Extractor) and Node 3 (Hypothesis Generator) have run on the Round 1 papers, the pipeline loops back here. The retriever now builds a **refined query** by extracting the top-5 TF-IDF keywords from the hypothesis text and the canonical names of KG entities with the highest edge-count (the most "connected" concepts). For example, if the hypothesis mentions "SHAP feature importance for gradient boosting on clinical tabular data", the refined query becomes `"SHAP feature importance" AND "gradient boosting" AND "clinical tabular"`. The retriever fetches 3–5 new papers, **deduplicates** them against already-retrieved arXiv IDs (exact string match on the arXiv ID), and appends only genuinely new papers. Sets `retrieval_round = 2`.

  3. **Round 3 (gap-filling, optional):** If Round 2 added ≥ 2 new papers, the KG is re-enriched and the hypothesis is re-generated. A third retrieval round may be triggered using the updated hypothesis terms. If Round 2 added 0–1 new papers, the retriever terminates early (diminishing returns). Sets `retrieval_round = 3` or terminates.

  4. **Max rounds:** 3 (configurable via `max_retrieval_rounds`). The KG and hypothesis are incrementally enriched with each round. The convergence criterion (< 2 new papers) prevents wasting API calls when the literature space is already well-covered.

- **Data Flow per Round:**
  ```
  Round N input:  topic (round 1) OR hypothesis + KG entities (rounds 2+)
       │
       ▼
  Build arXiv query (raw topic OR TF-IDF keywords from hypothesis)
       │
       ▼
  arxiv.Search(query, max_results=5) → list of arXiv Result objects
       │
       ▼
  For each result:
    - Skip if arXiv ID already in arxiv_papers_full_text[]
    - Download full text (HTML or LaTeX source)
    - Extract sections: Methodology, Implementation, Results
    - Preserve metadata: authors, year, title, arXiv ID
       │
       ▼
  Append new papers to arxiv_papers_full_text[]
  Increment retrieval_round
  ```

- **Security & Compliance:**
  - Enforces a strict `time.sleep(3)` between consecutive arXiv API/HTTP requests to comply with arXiv's Terms of Service and prevent rate-limiting or IP bans.
  - Parsed full-text content is kept in memory only — no PDFs are saved to disk for redistribution.
  - Paper metadata (authors, year, title, arXiv ID) is preserved for BibTeX generation downstream.

#### Node 2: Epistemic KG Extractor (AI) — *Upgraded: Schema-Based Extraction + Entity Resolution + Contradiction Preservation*

> **Design Review Finding:** Naive JSON prompting creates sparse, fragmented knowledge graphs with duplicate entities (e.g., "Neural Net", "NN", "Neural Network" stored as separate nodes), corrupting downstream hypothesis generation.

> **Epistemic Review Finding:** The original edge resolution step discards contradictory edges, keeping only the highest-confidence edge per triple. This destroys critical scientific signal — contradictions between papers are among the most valuable inputs for hypothesis generation. A paper that says "method A outperforms B" and another saying "method B outperforms A" should both be preserved with opposing polarity.

- **Model:** Claude 3.5 Haiku (fast, cost-effective for structured extraction).

- **What is an Epistemic KG?** A traditional knowledge graph stores facts as triples: `(entity A, relation, entity B)`. An **epistemic** knowledge graph adds a fourth dimension: **polarity** — whether the source paper *supports*, *contradicts*, or is *neutral* toward that relation. This means the KG can represent scientific disagreement, not just consensus. For example, if Paper 1 claims "Random Forest outperforms XGBoost on tabular data" and Paper 2 claims "XGBoost outperforms Random Forest on tabular data", a traditional KG would discard one of these edges (keeping only the highest-confidence one). An epistemic KG preserves **both** edges with opposing polarity, creating a **contested pair** that signals an unresolved scientific question — exactly the kind of gap that makes for a strong research hypothesis.

- **Why Preserve Contradictions?** Scientific progress happens at the boundaries of disagreement. If two reputable papers disagree on whether method A or method B is superior, that disagreement is a research opportunity: perhaps the answer depends on a confounding variable (dataset size, domain, preprocessing) that neither paper controlled for. By preserving contradictions in the KG, the Hypothesis Generator (Node 3) can identify these contested areas and propose hypotheses that resolve the disagreement — producing genuinely novel research rather than re-confirming known results.

- **System Prompt:** Uses **schema-based extraction** with strict typed dictionaries defining the expected entity types (`model`, `dataset`, `metric`, `method`, `hyperparameter`) and relation types (`outperforms`, `uses_dataset`, `achieves_metric`, `has_hyperparameter`). Each edge MUST include a **`polarity`** field (`"supports"`, `"contradicts"`, or `"neutral"`) and a **`context_condition`** field capturing any boundary conditions under which the claim holds. The prompt enforces rigid JSON output boundaries — no free-form generation.

- **Conditional Support Rule (Epistemic Nuance):**

  > **Systems Engineering Finding:** Scientific support is rarely absolute. A paper claiming "method A outperforms method B" almost always does so under specific conditions — a particular dataset size, domain, hyperparameter configuration, or evaluation protocol. Treating conditional support as unconditional creates a falsely confident KG that collapses nuance into binary assertions, leading to overgeneralized hypotheses and brittle experimental designs.

  If a paper's support or contradiction is conditional, the extractor MUST isolate the specific boundary conditions into the `context_condition` field. The agent is **strictly instructed not to treat conditional support as absolute**. Examples:
  - A paper that says "RF outperforms XGBoost *on small datasets (< 10,000 samples)*" → `context_condition: "only when dataset size < 10,000 samples"`.
  - A paper that says "Transformers outperform LSTMs *when pre-trained on > 1B tokens*" → `context_condition: "requires pre-training on > 1B tokens"`.
  - A paper that says "RF outperforms XGBoost" with no stated conditions → `context_condition: ""` (empty string — unconditional claim).

  This field is consumed downstream by the Hypothesis Generator (to avoid overgeneralizing conditional findings), the Claim Ledger Builder (to rate evidence strength more accurately — conditional evidence supporting an unconditional claim is weaker), and the Fact-Checker (to flag claims in the paper that drop boundary conditions present in the source literature).

- **Polarity Rules for the LLM:**
  - `"supports"` — the paper explicitly presents evidence *for* this relation (e.g., "Our experiments show RF achieves 94% accuracy, outperforming XGBoost at 91%").
  - `"contradicts"` — the paper explicitly presents evidence *against* this relation (e.g., "Contrary to [Author2023], we find XGBoost consistently outperforms RF on tabular benchmarks").
  - `"neutral"` — the paper mentions the relation but does not take a stance (e.g., "Both RF and XGBoost are commonly used for tabular classification").

- **Example: Contested Knowledge Graph**

  Suppose two papers are retrieved:
  - **Paper A** (arXiv:2401.12345): "Random Forest outperforms XGBoost on the UCI Adult dataset with 94.2% accuracy."
  - **Paper B** (arXiv:2402.67890): "XGBoost outperforms Random Forest on the UCI Adult dataset with 95.1% accuracy when hyperparameters are tuned via Bayesian optimization."

  The extractor produces:

```json
{
  "entities": [
    {
      "id": "e1",
      "canonical_name": "Random Forest",
      "entity_type": "model",
      "aliases": ["RF", "random forest classifier"],
      "attributes": {"n_estimators": "100", "max_depth": "None"}
    },
    {
      "id": "e2",
      "canonical_name": "XGBoost",
      "entity_type": "model",
      "aliases": ["XGB", "Extreme Gradient Boosting"],
      "attributes": {"learning_rate": "0.1", "n_estimators": "500"}
    },
    {
      "id": "e3",
      "canonical_name": "UCI Adult",
      "entity_type": "dataset",
      "aliases": ["Adult Census", "adult income dataset"],
      "attributes": {"task": "binary classification", "samples": "48842"}
    }
  ],
  "edges": [
    {
      "source_id": "e1",
      "target_id": "e2",
      "relation": "outperforms",
      "polarity": "supports",
      "context_condition": "",
      "confidence": 0.92,
      "provenance": "arXiv:2401.12345, Section 4.2"
    },
    {
      "source_id": "e2",
      "target_id": "e1",
      "relation": "outperforms",
      "polarity": "supports",
      "context_condition": "only when hyperparameters are tuned via Bayesian optimization",
      "confidence": 0.88,
      "provenance": "arXiv:2402.67890, Section 3.1"
    }
  ]
}
```

  Notice: both edges have `"polarity": "supports"` because each paper supports *its own* claim. The contradiction is structural — the two edges make opposing claims about the same entity pair. The post-processing pipeline (step 3 below) detects this as a **contested pair** because the `(source, target, relation)` triples are inverses of each other. Additionally, Paper B's claim is **conditional** (`context_condition` captures the Bayesian optimization requirement) while Paper A's claim is unconditional — this nuance is preserved for downstream nodes to reason about.

- **Post-Processing Pipeline** (deterministic, not LLM):

  1. **Embedding-based clustering (entity deduplication):**
     - Embed all entity `canonical_name` values using SBERT (`all-MiniLM-L6-v2`), producing a 384-dimensional vector per entity.
     - Compute pairwise cosine similarity between all entity embeddings.
     - Cluster entities where cosine similarity > 0.85 into synonym groups. For example, entities named "Neural Net", "NN", and "Neural Network" would form a single cluster because their SBERT embeddings are nearly identical.
     - **Why SBERT instead of string matching?** String matching misses semantic synonyms ("RF" ↔ "Random Forest") and catches false positives ("Random Forest" the ML model vs. "random forest" the ecological concept). SBERT embeddings capture semantic meaning, not surface-level string overlap.

  2. **LLM deduplication pass:**
     - For each synonym cluster, send a single Claude API call with all entity names and attributes in the cluster.
     - The LLM picks the most descriptive **canonical name** (e.g., "Random Forest" over "RF") and merges attributes from all aliases into a single entity.
     - All edges pointing to any alias in the cluster are re-routed to the canonical entity ID.
     - **Cost:** One Claude Haiku call per cluster (typically 3–8 clusters per pipeline run) — minimal token spend.

  3. **Epistemic edge resolution (contradiction-aware):**
     - Deduplicate edges with the **identical** `(source, target, relation, polarity)` tuple, keeping the highest-confidence instance. This removes redundant edges from the same paper being extracted multiple times.
     - **Contested pair detection:** For each unique `(entity_A, entity_B)` pair, check if there exist edges with opposing claims (e.g., "A outperforms B" *and* "B outperforms A", or the same relation with `"supports"` and `"contradicts"` polarity). If found, flag both edges as a `contested_pair` for downstream use.
     - **Key difference from the original design:** Contradictory edges are **explicitly preserved**, not discarded. The old design kept only the highest-confidence edge per `(source, target, relation)` triple, which silently destroyed the most scientifically interesting signal in the data. The epistemic design keeps both sides of the disagreement alive.

- **Output:** Clean, deduplicated `kg_entities[]` and epistemic `kg_edges[]` (with polarity and contested-pair flags) written to state. On iterative retrieval rounds (2+), new entities and edges from newly retrieved papers are merged into the existing KG using the same deduplication pipeline — the KG grows incrementally across rounds without duplicating previously extracted knowledge.

#### Node 3: Incremental Hypothesis Generator (AI) — *Upgraded: Mathematical Novelty Detection + Prior-Art Screening + Contradiction Exploitation*

> **Design Review Finding:** The hypothesis generator validates against hallucinations but not against the broader scientific corpus. LLMs frequently propose "novel" ideas that are well-established concepts (e.g., micro-batching for SGD presented as a breakthrough).

> **Epistemic Review Finding:** The original generator produces a hypothesis in isolation without articulating what it adds beyond existing work. An incremental approach — where the generator must explicitly state the `incremental_delta` (what is new compared to the closest prior art) — forces genuine novelty and makes the hypothesis reviewable by the HITL operator.

- **Model:** Claude 3.7 Sonnet (advanced reasoning for hypothesis formulation).
- **System Prompt:** "Formulate a highly specific, testable research hypothesis strictly grounded in the technical entities extracted into the Knowledge Graph. **Pay special attention to contested edges** (opposing polarity) in the KG — contradictions between papers are prime targets for novel hypotheses that could resolve the disagreement. When referencing datasets, you MUST use real, verifiable, public dataset IDs from the Hugging Face Hub (e.g., `imdb`, `glue`, `squad`) or scikit-learn. Do NOT hallucinate dataset names or local file paths. You MUST also produce an `incremental_delta` field: a 2–3 sentence explanation of what your hypothesis adds beyond the closest existing work in the KG."
- **Responsibility:** Formulate a testable hypothesis grounded in the KG entities, exploiting contradictions where present. The hypothesis is validated against KG entities to prevent hallucination. Outputs both `hypothesis` and `incremental_delta` to state.

- **Automated Novelty Detection + Prior-Art Screening Protocol** (deterministic post-step):
  1. Embed the generated hypothesis using SBERT (`all-MiniLM-L6-v2`).
  2. Embed all paper abstracts from `arxiv_papers_full_text[]`.
  3. Compute **Relative Neighbor Density (RND):** the average cosine distance from the hypothesis embedding to the K nearest literature embeddings.
  4. Compute **Prior-Art Similarity Score:** the maximum cosine similarity between the hypothesis embedding and any single paper abstract embedding. This catches cases where the hypothesis is a near-paraphrase of one specific paper (even if the average RND looks acceptable).
  5. Apply dual gating:
     - **RND ≥ `novelty_threshold`** (default: `0.35`) AND **Prior-Art Similarity < `prior_art_ceiling`** (default: `0.90`) → hypothesis is sufficiently novel → proceed to HITL Gate.
     - **RND < threshold** OR **Prior-Art Similarity ≥ ceiling** → hypothesis is too similar to existing work → pipeline terminates with `failed_novelty` status and a report explaining which papers are too close, including the `incremental_delta` for the operator to review.
  6. Write `novelty_score`, `prior_art_similarity_score`, `hypothesis_embedding`, `incremental_delta`, and `novelty_passed` to state.
  7. **Iterative retrieval trigger:** If novelty passes, extract key technical terms from the hypothesis and feed them back to Node 1 for a refined retrieval round (up to `max_retrieval_rounds`).

### HITL Checkpoint 1: Hypothesis Approval Gate

> **Design Review Finding:** 100% autonomy leads to silent failures — fabricated metrics, hallucinated citations, wasted GPU compute on unviable concepts. Empirical studies show co-pilot mode with human checkpoints significantly outperforms fully autonomous pipelines (Agent Laboratory, 2024).

#### Node 3b: HITL Gate 1 — Hypothesis Approval (Non-AI)

- **Model:** None — deterministic checkpoint logic.
- **Responsibility:** The pipeline **pauses** and presents the human operator with:
  1. The generated **hypothesis** (plain text).
  2. The **incremental delta** — what this hypothesis adds beyond the closest prior art.
  3. The **supporting and contradicting KG triples** (entities + edges with polarity that ground the hypothesis).
  4. The computed **novelty score**, **prior-art similarity score**, and how they compare to thresholds.
  5. A summary of retrieved literature titles and abstracts (across all retrieval rounds).

- **Operator actions:**
  - **`approve`** → sets `hitl_approved = True`; pipeline proceeds to Experimental Designer (Node 3c).
  - **`reject <reason>`** → sets `hitl_approved = False` and `hitl_rejection_reason`; pipeline terminates with `failed_hitl_rejected` status.

- **Implementation approach:**
  - **CLI mode:** Pipeline blocks on `input()` prompt with a Rich-formatted summary panel.
  - **Web UI mode:** Pipeline emits `awaiting_hitl_hypothesis` status; the React frontend polls and presents an approval dialog; the backend exposes an `/api/hitl/approve-hypothesis` endpoint.

- **Rationale:** No experiment design or GPU compute is provisioned until human approval is received. This prevents the system from wasting resources on unviable or unoriginal concepts.

#### Node 3c: Experimental Designer (AI) — *New Node: Structured Experiment Specification*

> **Epistemic Review Finding:** The original pipeline jumps directly from hypothesis approval to code generation, giving the ML Coder unconstrained freedom to choose datasets, metrics, and experimental setups. This creates a gap where the human approves a hypothesis but has no visibility into how it will be tested. An intermediate Experimental Designer node produces a structured, human-reviewable experiment specification before any code is written.

- **Model:** Claude 3.7 Sonnet (advanced reasoning for experimental design).
- **System Prompt:** "You are an expert experimental designer. Given the approved hypothesis, the Knowledge Graph (including contested edges), and the incremental delta, design a rigorous experiment. Output a structured `ExperimentSpec` JSON with: `independent_var` (what you're manipulating), `dependent_var` (what you're measuring), `control_description` (baseline/control condition), `dataset_id` (a real, verifiable public dataset from Hugging Face Hub or scikit-learn), `evaluation_metrics` (list of metrics, e.g., accuracy, F1, AUC), and `expected_outcome` (what result would support/refute the hypothesis). Justify each choice in 1–2 sentences."
- **Responsibility:** Translate the approved hypothesis into a concrete, structured experiment specification. This specification becomes the contract that constrains the ML Coder downstream.
- **Output:** Writes `experiment_spec` to state.

#### Node 3d: HITL Gate 2 — Experiment Approval (Non-AI)

> **Epistemic Review Finding:** A single HITL gate approving the hypothesis is insufficient — the human should also approve *how* the hypothesis will be tested. A second gate after experiment design catches methodological issues (wrong dataset, inappropriate metrics, unfair baselines) before any code is generated or compute is spent.

- **Model:** None — deterministic checkpoint logic.
- **Responsibility:** The pipeline **pauses** and presents the human operator with:
  1. The approved **hypothesis** and its **incremental delta**.
  2. The structured **ExperimentSpec** (independent/dependent variables, control, dataset, metrics, expected outcome).
  3. The **contested KG edges** relevant to the hypothesis (if any).

- **Operator actions:**
  - **`approve`** → sets `hitl_experiment_approved = True`; pipeline proceeds to Phase 2 (Constrained ML Coder).
  - **`reject <reason>`** → sets `hitl_experiment_approved = False`; pipeline routes back to Node 3c (Experimental Designer) for redesign, or terminates if the operator explicitly aborts.

- **Implementation approach:**
  - **CLI mode:** Pipeline blocks on `input()` prompt with a Rich-formatted experiment summary panel.
  - **Web UI mode:** Pipeline emits `awaiting_hitl_experiment` status; the React frontend polls and presents an experiment approval dialog; the backend exposes an `/api/hitl/approve-experiment` endpoint.

### Phase 2: Autonomous Experimentation & Self-Healing

#### Node 4: Constrained ML Coder (AI) — *Upgraded: Active Debugging + Experiment Spec Binding*

> **Design Review Finding:** The passive self-healing loop (simply pasting stack traces back) is insufficient for deep methodological errors like tensor shape mismatches, exploding gradients, or silent NaN loss values.

> **Epistemic Review Finding:** The original ML Coder receives only the hypothesis and KG, giving it unconstrained freedom to choose datasets, metrics, and experimental setups that may not match what the human approved. Binding the coder to the approved `ExperimentSpec` ensures the generated code implements exactly the experiment that was reviewed and approved.

- **Model:** Claude 3.7 Sonnet (advanced reasoning and software engineering).
- **System Prompt:** "You are an expert data scientist. Read the validated hypothesis, the rich KG with actual implementation details, and the **approved ExperimentSpec**. Write a self-contained, methodologically rigorous Python script that **strictly implements the approved ExperimentSpec** — you MUST use the specified `dataset_id`, `evaluation_metrics`, and `independent_var`/`dependent_var`. Do NOT deviate from the approved experimental design. You MUST: (1) explicitly separate train and test data to prevent data leakage; (2) use cross-validation where applicable; (3) set random seeds (e.g., `random_state=42`) for full reproducibility; (4) use the dataset ID from the ExperimentSpec via Hugging Face Hub (e.g., `load_dataset('imdb')`) or scikit-learn — do NOT hallucinate local file paths or custom dataset names; (5) save a detailed log of all hyperparameters used alongside evaluation metrics into `metrics.json`; (6) compute all metrics listed in the ExperimentSpec. Output ONLY valid Python code."

- **Import & Dependency Safety Constraints (AST Fragility Fix):**

  > **Systems Engineering Finding:** The downstream Dependency Resolver (Node 4b) relies on AST parsing of `import` and `from ... import` statements to identify pip packages. If the ML Coder uses dynamic imports (`importlib.import_module()`, `exec("import X")`, `__import__()`), the AST parser cannot detect them, causing silent dependency resolution failures inside the `--network=none` sandbox. Additionally, packages requiring system-level C compilation (e.g., `xgboost` built from source, `lightgbm` without prebuilt wheels) will fail in the sandbox's read-only filesystem.

  The ML Coder MUST adhere to the following hard constraints, enforced via system prompt instructions:
  1. **Static imports only:** All imports MUST be explicit, static `import X` or `from X import Y` statements placed at the very top of the script. It is **strictly forbidden** to use `importlib`, `importlib.import_module()`, `exec()`, `eval()`, `__import__()`, or any other form of dynamic module loading.
  2. **Pre-compiled wheels only:** The agent MUST assume the execution environment lacks system-level C compilers (`gcc`, `g++`, `make`). It MUST rely strictly on standard, pre-compiled Python wheels available on PyPI: `scikit-learn`, `transformers`, `torch`, `pandas`, `numpy`, `scipy`, `datasets`, `huggingface_hub`, `matplotlib`, `seaborn`. If the experiment requires a package outside this allowlist, the Coder must flag it in the output for manual review.
  3. **No subprocess calls:** The script MUST NOT use `subprocess`, `os.system()`, or `shutil.which()` to invoke external binaries. All computation must be pure Python.

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

#### Node 5b: Claim Ledger Builder (Non-AI) — *New Node: Traceability Infrastructure*

> **Epistemic Review Finding:** The original Writer generates claims without any structured traceability mechanism. When the Fact-Checker later tries to verify claims against the KG, it must re-derive the mapping from scratch. A pre-built claim ledger — constructed deterministically before writing begins — provides a structured evidence base and enables a principled **No-Paper outcome** when evidence is insufficient.

- **Model:** None — deterministic Python logic.
- **Responsibility:** Analyzes the hypothesis, experiment results (`metrics_json`), and the epistemic KG to build a structured `claim_ledger[]`. For each potential claim the paper could make:
  1. Identifies supporting KG edges (same polarity).
  2. Identifies contradicting KG edges (opposing polarity).
  3. Assigns an `evidence_strength` rating:
     - **`strong`**: ≥ 2 supporting edges, 0 contradicting edges.
     - **`moderate`**: 1 supporting edge, or ≥ 2 supporting with ≥ 1 contradicting.
     - **`weak`**: 1 supporting edge with ≥ 1 contradicting edge.
     - **`unsupported`**: 0 supporting edges.

- **No-Paper Gate:** If more than 50% of claims have `unsupported` or `weak` evidence strength, the pipeline terminates with `no_paper` status and a detailed report explaining which claims lack evidence. This prevents generating papers with insufficient scientific grounding. The `.tex` source is NOT generated — `final_pdf_path` remains `None`.

- **Output:** Writes `claim_ledger[]` to state. If the No-Paper gate passes, proceeds to the Academic Writer.

#### Node 6: Academic Writer (AI) — *Upgraded: Claim Ledger-Grounded Writing*

- **Model:** Claude 3.7 Sonnet (excellent at academic tone and long-context synthesis).
- **System Prompt:** "You are an academic writer. Synthesize the full-text literature, the hypothesis, the incremental delta, and the experiment metrics to write an academic paper directly in LaTeX, following the IMRaD structure (Introduction, Methods, Results, Conclusion). You are provided with a **claim ledger** that maps each potential claim to its supporting and contradicting KG evidence. You MUST: (1) Only make claims rated `strong` or `moderate` in the claim ledger — do NOT include `weak` or `unsupported` claims. (2) For claims with contradicting evidence, acknowledge the contradiction in the text (e.g., 'While [Author2024] reports contradictory findings...'). (3) Use standard LaTeX citation commands (e.g., `\cite{AuthorYear}`) seamlessly in the text — do NOT use raw IDs or inline provenance tags. (4) Generate a corresponding `references.bib` file containing the BibTeX entries for all cited papers. Ensure the bibliography is rendered at the bottom of the final NeurIPS paper via `\bibliography{references}`. Do not state information as absolute truth if it cannot be traced back to the literature context."
- **Responsibility:** Synthesize the full-text literature, the hypothesis, `incremental_delta`, and `metrics_json` results to write the first draft of the paper directly in **LaTeX** (`draft.tex`), following the IMRaD structure. The claim ledger constrains which assertions can appear in the paper. Generate a companion `references.bib` with proper BibTeX entries (using paper metadata: authors, year, title, arXiv ID from the state). All citations use `\cite{AuthorYear}` — no raw arXiv IDs in prose. Writes to `latex_draft` and `bibtex_source` in the state.
- **Revision pass (after review):** Addresses each surviving critique from the debate-filtered review, produces `revised_latex`, and appends a **Confidence Score** (self-assessed 1–10) and the NeurIPS reproducibility checklist. Only one mandatory revision pass occurs — unbounded loops lead to model degradation and structural decay.

### Phase 4: Critique & Linting Engine (Automated Quality Assurance) — *Overhauled: Heterogeneous Models + Structured Debate + Deterministic Linter*

> **Design Review Finding:** Using three identical Claude 3.5 Haiku instances creates an "Artificial Hivemind" — agents share identical weights, biases (verbosity bias, self-enhancement bias), and RLHF alignment. They form rapid consensus on superficial critiques rather than catching deep methodological flaws. This is the echo-chamber effect documented in the NeurIPS 2025 Best Paper "Artificial Hivemind."

#### Node 6b: Deterministic Linter (Non-AI) — *New Node: Rule-Based Pre-Check*

> **Epistemic Review Finding:** Several quality checks (missing sections, citation formatting, claim-ledger compliance) are deterministic and should not consume LLM tokens or be subject to LLM unreliability. A deterministic linter runs before the LLM-based critique panel, catching structural issues cheaply and reliably.

- **Model:** None — pure Python logic (regex + AST-like LaTeX parsing).
- **Checks performed:**
  1. **IMRaD completeness:** Verify `\section{Introduction}`, `\section{Methods}`, `\section{Results}`, `\section{Conclusion}` are all present.
  2. **Citation integrity:** Every `\cite{...}` key in the `.tex` has a matching entry in `references.bib`. Flag orphaned citations.
  3. **Claim-ledger compliance:** Cross-reference the draft against `claim_ledger[]`. Flag any claim text that appears in the draft but has `unsupported` or `weak` evidence strength in the ledger.
  4. **NeurIPS checklist presence:** Verify the reproducibility checklist section exists.
  5. **Figure/table labeling:** Verify every `\begin{figure}` and `\begin{table}` has a `\label{}` and `\caption{}`.
  6. **No raw arXiv IDs in prose:** Regex-check for patterns like `arXiv:XXXX.XXXXX` outside of BibTeX blocks.
- **Output:** Deterministic warnings appended to `critique_warnings[]` with `source: "linter"`. These bypass the debate protocol (they are objective, not debatable).

#### Node 7: Critique Panel (AI — 3 Heterogeneous Agents) — *Overhauled + Claim Ledger Integration*

Three independent agents read the `draft.tex` and produce structured warnings, using **enforced model diversity** to prevent mode collapse:

| Reviewer | Role | Model | Focus |
|----------|------|-------|-------|
| **Agent A: Fact-Checker** | Verify empirical claims against KG + claim ledger | Claude 3.7 Sonnet | **Algorithmically queries `kg_entities`, `kg_edges`, and `claim_ledger` via strict JSON path traversals** — does NOT rely on parametric memory. Its system prompt includes the serialized JSON of the KG and the claim ledger. Must cite specific entity IDs, edge relations, and claim ledger entries when verifying/refuting claims. Any claim not traceable to a KG triple is flagged as `ungrounded`. **New:** Also verifies that contradicting evidence acknowledged in the claim ledger is properly discussed in the paper text — suppressed contradictions are flagged as `contradiction_suppressed`. |
| **Agent B: Methodologist** | Evaluate experimental rigor | Claude 3.5 Haiku | Checks if code results in `metrics.json` logically support conclusions. Verifies the code implemented the approved `ExperimentSpec` (correct dataset, metrics, variables). Flags unsupported claims, missing error bars, unjustified generalizations, incorrect statistical reasoning. Uses a **different model** than Agent A to ensure diverse cognitive architecture. |
| **Agent C: Formatter** | Assess structure & LaTeX quality | Claude 3.5 Haiku (different system prompt persona) | Checks for AI-slop writing style, excessive verbosity, missing NeurIPS checklist items, LaTeX structural integrity, citation formatting, figure/table labelling. **Note:** Many of Agent C's previous checks are now handled by the deterministic linter — Agent C focuses on subjective quality (writing style, argumentation flow, clarity). |

**Structured Debate Protocol** (replaces passive vote aggregation):

1. **Independent critique phase:** Each reviewer independently generates critiques of the draft. Output: `critique_warnings[]` per agent.
2. **Cross-challenge phase:** Each reviewer reads the other two reviewers' critiques. For each critique they disagree with, they issue a formal **challenge** explaining why the critique is incorrect, excessive, or based on a misunderstanding.
3. **Response phase:** The original critic must **defend or retract** each challenged finding with evidence.
4. **Resolution:** Only critiques that **survive the debate** (unretracted after challenge) are forwarded to the Writer for the revision pass. Retracted critiques are logged but not acted upon.

This ensures superficial consensus is broken. The debate log is preserved in `debate_log[]` for auditability.

#### Node 8: Critique Aggregator & Mandatory Revision (Non-AI) — *Updated: Linter + Debate Integration*

- **Model:** None — pure Python logic.
- **Responsibility:** Collects both the **deterministic linter warnings** (which bypass debate — they are objective) and the **debate-surviving critiques** (not retracted ones) into a single structured feedback list. Routes them back to **Node 6 (Academic Writer)** for exactly **one mandatory revision pass**. The Writer must address the critique, produce a revised `draft.tex`, and append a "Confidence Score" (self-assessed 1–10) and the NeurIPS reproducibility checklist. No further review rounds occur — the revised draft proceeds directly to the LaTeX Compiler.

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

### 3.1 User Stories (31 total)

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
| US-21 | As a researcher, I want the ArXiv Retriever to **iteratively refine search queries** using hypothesis-derived terms (up to 3 rounds) so that domain-specific papers are not missed by the initial broad search. | Should Have | 5 |
| US-22 | As a researcher, I want the Knowledge Graph to **preserve contradictions** between papers (via polarity on edges) so that conflicting findings are available for hypothesis generation. | Must Have | 5 |
| US-23 | As a researcher, I want the Hypothesis Generator to produce an **incremental delta** explaining what the hypothesis adds beyond the closest prior art so that novelty is explicit and reviewable. | Must Have | 3 |
| US-24 | As a researcher, I want a **prior-art similarity score** (max cosine similarity to any single paper) as a secondary novelty gate so that near-paraphrases of existing work are caught. | Must Have | 3 |
| US-25 | As a researcher, I want an **Experimental Designer** node that produces a structured ExperimentSpec (IV, DV, control, dataset, metrics) so that I can review and approve the experiment design before code is written. | Must Have | 5 |
| US-26 | As a researcher, I want a **second HITL gate** after experiment design so that I can approve or reject how the hypothesis will be tested before any code generation or compute. | Must Have | 3 |
| US-27 | As a researcher, I want a **claim ledger** that maps every paper assertion to supporting/contradicting KG evidence, with a **No-Paper outcome** when evidence is insufficient, so that unsupported papers are never generated. | Must Have | 8 |
| US-28 | As a researcher, I want a **deterministic linter** that checks IMRaD completeness, citation integrity, claim-ledger compliance, and NeurIPS checklist presence before the LLM critique panel, so that objective issues are caught cheaply and reliably. | Should Have | 5 |
| US-29 | As a developer, I want the ML Coder to be **restricted to static imports and pre-compiled wheels only** (no `importlib`, `exec()`, `eval()`, or C-compiler-dependent packages) so that the AST-based Dependency Resolver can reliably detect all dependencies and the sandbox never fails on missing system compilers. | Must Have | 3 |
| US-30 | As a developer, I want each AI-powered node to receive a **scoped view (pruned state)** containing only the fields it needs — not the entire `AutoResearchState` — so that "lost in the middle" attention degradation is avoided and API token costs are minimized. | Must Have | 5 |
| US-31 | As a researcher, I want KG edges to include a **`context_condition`** field that captures boundary conditions (e.g., "only when dataset size < 10k samples") so that conditional scientific claims are not treated as absolute truths by downstream nodes. | Must Have | 3 |

### 3.2 Product Backlog

The backlog is organized into 5 sprints:

**Sprint 1 — Foundation (Week 1):**
US-01, US-07, US-20, US-30 — CLI entry point, Docker sandbox setup, logging infrastructure, scoped state view infrastructure (`build_scoped_view()` + `NODE_SCOPE_CONFIG`).

**Sprint 2 — Phase 1: Epistemic Literature & KG (Week 2):**
US-02, US-21, US-03, US-22, US-31, US-04, US-23, US-05, US-24, US-06 — Iterative ArXiv retrieval, epistemic KG extraction with polarity + `context_condition` + entity dedup, incremental hypothesis generation with novelty scoring + prior-art screening, HITL Gate 1.

**Sprint 3 — Experiment Design & Phase 2 Experimentation (Week 3):**
US-25, US-26, US-29, US-08, US-09, US-10 — Experimental Designer, HITL Gate 2, ML Coder import safety constraints (static imports + pre-compiled wheels only), Dependency Resolver (AST parsing + pre-caching), active debugging injection, self-healing loop with context-aware retry.

**Sprint 4 — Draft, Review & Publication (Week 4):**
US-27, US-11, US-28, US-12, US-13, US-14, US-15, US-16 — Claim Ledger Builder (with No-Paper gate), Academic Writer (claim ledger-grounded), Deterministic Linter, heterogeneous Review Panel with debate protocol, KG + claim ledger-grounded fact-checking, mandatory revision, LaTeX compiler repair loop.

**Sprint 5 — Polish & Config (Week 5):**
US-17, US-18, US-19 — Progress display, failure reports, model configuration, final integration testing.

---

## 4. Diagrams

All diagrams are stored in the repository under the `docs/diagrams/` directory.

### 4.1 Component Architecture Diagram

High-level system components: CLI Interface, LangGraph Orchestrator (with **scoped state views** via `build_scoped_view()`), Iterative ArXiv Retriever, Epistemic KG Extractor (with SBERT clustering + LLM dedup + polarity + `context_condition`), Incremental Hypothesis Generator (with Novelty Scorer + Prior-Art Screening), HITL Gate 1 (Hypothesis), Experimental Designer, HITL Gate 2 (Experiment), Constrained ML Coder (with debug injection + ExperimentSpec binding + **static-import-only enforcement**), Dependency Resolver (AST parser + host-side cache), Executor Sandbox (Docker `--network=none` with `:ro` volume mounts), Claim Ledger Builder (with No-Paper gate + `context_condition`-aware evidence rating), Academic Writer (claim ledger-grounded), Deterministic Linter, Heterogeneous Review Panel (3 diverse agents + debate protocol + claim ledger), Critique Aggregator, LaTeX Compiler (with Repair Loop), arXiv API, SBERT Embedding Service, File System Output.

### 4.2 LangGraph Workflow Diagram (State Machine)

```
Phase 1: Iterative Literature Parsing & Epistemic KG
START → [Node 1: Iterative ArXiv Retriever] → [Node 2: Epistemic KG Extractor]
                                                        │
                                              schema-based extraction
                                              + SBERT entity clustering
                                              + LLM dedup pass
                                              + polarity (supports/contradicts)
                                              + context_condition (boundary conds)
                                                        │
                                                        ▼
                                    [Node 3: Incremental Hypothesis Generator]
                                              │
                                    ┌─────────┼──────────┐
                                    │         │          │
                               KG valid  KG invalid  novelty < threshold
                               + novel   (hallucin.)  OR prior_art ≥ ceiling
                                    │         │             │
                                    │    regenerate    END (failed_novelty)
                                    │
                          ┌─── iterative retrieval loop ───┐
                          │   (refine search terms from    │
                          │    hypothesis → back to Node 1) │
                          └────────────────────────────────┘
                                    │
HITL Checkpoint 1                   │
                          [Node 3b: HITL Gate 1 — Hypothesis]
                                    │
                          ┌─────────┴─────────┐
                          │                   │
                       approved            rejected
                          │                   │
                          │                   ▼
                          │           END (failed_hitl_rejected)
                          │
Experiment Design         │
                  [Node 3c: Experimental Designer]
                    (structured ExperimentSpec)
                          │
HITL Checkpoint 2         │
                  [Node 3d: HITL Gate 2 — Experiment]
                          │
                  ┌───────┴───────┐
                  │               │
               approved        rejected
                  │               │
                  │          redesign → Node 3c
                  │          (or abort)
                  │
Phase 2: Experimentation
                  │
          [Node 4: Constrained ML Coder] ←── (bound to ExperimentSpec)
                  │                          + debug injection
                  │                          + static imports only
                  │                    (scoped view: experiment_spec + hypothesis)
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
Phase 3: Paper Drafting
        │
  [Node 5b: Claim Ledger Builder]
        │
  ┌─────┴─────┐
  │           │
  pass     >50% weak/unsupported
  │           │
  │           ▼
  │     END (no_paper)
  │     (insufficient evidence)
  │
  [Node 6: Academic Writer]
    (claim ledger-grounded)
    (scoped view: claim_ledger + metrics_json
     + experiment_spec + incremental_delta)
        │
Phase 4: Critique
        │
  [Node 6b: Deterministic Linter]
    (IMRaD, citations, claim-ledger
     compliance, NeurIPS checklist)
        │
  [Node 7: Heterogeneous Review Panel]
    Agent A: Fact-Checker (Sonnet) — KG + claim ledger
    Agent B: Methodologist (Haiku) — ExperimentSpec compliance
    Agent C: Formatter (Haiku) — subjective quality
        │
    Structured Debate Protocol:
    1. Independent critiques
    2. Cross-challenge phase
    3. Defend-or-retract phase
    4. Only surviving critiques forwarded
        │
  [Node 8: Critique Aggregator]
    (linter warnings + debate-surviving critiques)
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

Illustrates the message flow between User → CLI → LangGraph (with scoped state views per node) → each Node (Iterative ArXiv Retriever, Epistemic KG Extractor + SBERT dedup + polarity + `context_condition`, Incremental Hypothesis Generator + novelty scorer + prior-art screening, HITL Gate 1 ↔ Human Operator, Experimental Designer, HITL Gate 2 ↔ Human Operator, Constrained ML Coder (static imports only) + debug injection, Dependency Resolver + host-side fetch, Executor + Docker with `:ro` mounts, Claim Ledger Builder + No-Paper gate, Academic Writer, Deterministic Linter, Heterogeneous Review Panel + debate rounds, Critique Aggregator, LaTeX Compiler + Repair Loop) → external services (arXiv, SBERT, Docker, pdflatex) → PDF output (or No-Paper outcome).

### 4.4 Global State Data Model (Class Diagram)

```
┌──────────────────────────────────────────────────────────────┐
│                   AutoResearchState                          │
├──────────────────────────────────────────────────────────────┤
│  Phase 1: Deep Context & Epistemic KG                        │
│ + topic: str                                                 │
│ + arxiv_papers_full_text: List[Dict]                         │
│ + retrieval_round: int               # iterative retrieval   │
│ + kg_entities: List[KGEntity]        # deduplicated          │
│ + kg_edges: List[KGEdge]            # polarity + context_condition │
│ + hypothesis: str                                            │
│ + incremental_delta: str             # what's new vs prior   │
│ + hypothesis_embedding: List[float]  # SBERT vector          │
│ + novelty_score: float               # RND metric            │
│ + prior_art_similarity_score: float  # max cosine to prior   │
│ + novelty_passed: bool                                       │
├──────────────────────────────────────────────────────────────┤
│  HITL Gate 1: Hypothesis Approval                            │
│ + hitl_approved: bool                                        │
│ + hitl_rejection_reason: str                                 │
├──────────────────────────────────────────────────────────────┤
│  Experiment Design (between HITL gates)                      │
│ + experiment_spec: ExperimentSpec    # structured design     │
│ + hitl_experiment_approved: bool     # second HITL gate      │
├──────────────────────────────────────────────────────────────┤
│  Phase 2: Experimentation                                    │
│ + python_code: str                                           │
│ + resolved_dependencies: List[str]   # from AST parse        │
│ + resolved_datasets: List[str]       # from AST parse        │
│ + dataset_cache_path: str            # host cache dir        │
│ + debug_instrumentation: str         # augmented code        │
│ + execution_success: bool                                    │
│ + execution_logs: str                                        │
│ + metrics_json: str                                          │
│ + code_retry_count: int                                      │
├──────────────────────────────────────────────────────────────┤
│  Phase 3 & 4: Drafting, Critique & Debate                    │
│ + claim_ledger: List[ClaimLedgerEntry] # claim traceability  │
│ + latex_draft: str                                           │
│ + bibtex_source: str                                         │
│ + critique_warnings: List[Dict]      # per-agent             │
│ + debate_log: List[DebateEntry]      # challenge/defend      │
│ + surviving_critiques: List[Dict]    # post-debate           │
│ + confidence_score: float                                    │
│ + revision_pass_done: bool                                   │
├──────────────────────────────────────────────────────────────┤
│  Phase 5: Compilation & Repair                               │
│ + latex_compile_log: str                                     │
│ + latex_repair_attempts: int                                 │
│ + final_pdf_path: Optional[str]      # None if No-Paper      │
├──────────────────────────────────────────────────────────────┤
│  Telemetry                                                   │
│ + pipeline_status: str               # running |             │
│                                      # awaiting_hitl_hyp |   │
│                                      # awaiting_hitl_exp |   │
│                                      # success |             │
│                                      # failed_novelty |      │
│                                      # failed_hitl |         │
│                                      # failed_execution |    │
│                                      # failed_latex |        │
│                                      # no_paper              │
│ + total_api_calls: int                                       │
│ + total_tokens_used: int                                     │
│ + logs: List[str]                                            │
├──────────────────────────────────────────────────────────────┤
│   TypedDict used as LangGraph State                          │
└──────────────────────────────────────────────────────────────┘
```

### 4.5 Deployment / Infrastructure Diagram

Shows: Host machine, Docker daemon (with `--network=none` sandbox + `:ro` volume mounts for `.cache/pip`, `.cache/hf`, `.cache/sklearn`), Python virtual environment, SBERT embedding model (local inference), API calls to Anthropic (multiple agents — heterogeneous models), arXiv REST API + HTML/LaTeX source endpoints, pdflatex/bibtex tools (with repair loop), Rich CLI for dual HITL interaction (hypothesis + experiment gates), file system I/O (metrics.json, claim_ledger.json, draft.tex, references.bib, debate log, final PDF or No-Paper report).

**Deliverable:** All diagrams rendered as `.png` or `.svg` and stored in `docs/diagrams/`. Mermaid source files kept alongside for version control.

---

## 5. Implementation Phases

### Phase 1: Environment & Infrastructure Setup

1. Initialize a clean Python project with `pyproject.toml` or `requirements.txt`.
2. Install core dependencies: `anthropic`, `langgraph`, `arxiv`, `docker`, `datasets`, `huggingface_hub`, `sentence-transformers`, `scikit-learn`, `numpy`, `rich`.
3. Create a `Dockerfile.sandbox` with a base Python image and **pre-compiled wheels only** for data science libraries (pandas, scikit-learn, numpy, datasets, huggingface_hub, transformers) — no system C compilers installed in the sandbox image.
4. Install LaTeX toolchain (`texlive`, `pdflatex`, `bibtex`) in the build environment.
5. Store `ANTHROPIC_API_KEY` securely in a `.env` file (excluded from git via `.gitignore`).
6. Implement the **scoped state view infrastructure**: create `backend/utils/state_pruning.py` containing the `build_scoped_view(state, node_name)` utility and the `NODE_SCOPE_CONFIG` dictionary that maps each AI node to its allowed state fields (see §2 Context Management).

### Phase 2: Define the Global State

```python
from typing import TypedDict, List, Dict, Literal, Optional

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
    polarity: str             # "supports" | "contradicts" | "neutral"
    context_condition: str    # boundary condition, e.g. "only when dataset size < 10k samples"
                              # empty string "" if the claim is unconditional
    confidence: float
    provenance: str           # paper ID or section reference

class DebateEntry(TypedDict):
    round: int
    challenger_role: str
    target_critique_index: int
    challenge: str
    response: str
    resolved: bool

class ClaimLedgerEntry(TypedDict):
    claim_id: str
    claim_text: str
    supporting_kg_edges: List[str]       # KGEdge IDs that support this claim
    contradicting_kg_edges: List[str]    # KGEdge IDs that contradict this claim
    evidence_strength: str               # "strong" | "moderate" | "weak" | "unsupported"

class ExperimentSpec(TypedDict):
    independent_var: str
    dependent_var: str
    control_description: str
    dataset_id: str
    evaluation_metrics: List[str]
    expected_outcome: str

class AutoResearchState(TypedDict):
    # Phase 1: Deep Context & Epistemic KG
    topic: str
    arxiv_papers_full_text: List[Dict]
    retrieval_round: int                       # iterative retrieval round counter
    kg_entities: List[KGEntity]                # deduplicated via SBERT clustering
    kg_edges: List[KGEdge]                     # resolved, with polarity + context_condition
    hypothesis: str
    incremental_delta: str                     # what the hypothesis adds beyond prior art
    hypothesis_embedding: List[float]          # SBERT vector for novelty computation
    novelty_score: float                       # Relative Neighbor Density
    prior_art_similarity_score: float          # max cosine similarity to existing work
    novelty_passed: bool

    # HITL Gate 1: Hypothesis Approval
    hitl_approved: bool
    hitl_rejection_reason: str

    # Experiment Design (between HITL gates)
    experiment_spec: ExperimentSpec            # structured experiment design
    hitl_experiment_approved: bool             # second HITL gate approval

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
    claim_ledger: List[ClaimLedgerEntry]       # traces every paper claim to KG evidence
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
    final_pdf_path: Optional[str]              # None if No-Paper outcome

    # Telemetry
    pipeline_status: str      # running | awaiting_hitl_hypothesis |
                              # awaiting_hitl_experiment | success |
                              # failed_novelty | failed_hitl |
                              # failed_execution | failed_latex |
                              # no_paper (insufficient evidence)
    total_api_calls: int
    total_tokens_used: int
    logs: List[str]
```

| State Variable              | Type                    | Description                                                              |
| --------------------------- | ----------------------- | ------------------------------------------------------------------------ |
| `topic`                     | str                     | The user's initial research prompt.                                      |
| `arxiv_papers_full_text`    | List[Dict]              | Full-text methodology, implementation, and results sections.             |
| `retrieval_round`           | int                     | Current iterative retrieval round (hypothesis-driven refinement).        |
| `kg_entities`               | List[KGEntity]          | **Deduplicated** entities (via SBERT clustering + LLM dedup pass).       |
| `kg_edges`                  | List[KGEdge]            | **Epistemic** edges with polarity + `context_condition` for conditional claims. |
| `hypothesis`                | str                     | The testable hypothesis, grounded in KG entities.                        |
| `incremental_delta`         | str                     | What the hypothesis adds beyond the closest prior art.                   |
| `hypothesis_embedding`      | List[float]             | SBERT embedding vector of the hypothesis for novelty scoring.            |
| `novelty_score`             | float                   | Relative Neighbor Density — semantic distance from existing literature.  |
| `prior_art_similarity_score`| float                   | Max cosine similarity to any existing paper — prior-art screening.       |
| `novelty_passed`            | bool                    | Whether the hypothesis crossed the novelty threshold.                    |
| `hitl_approved`             | bool                    | Whether the human operator approved the hypothesis.                      |
| `hitl_rejection_reason`     | str                     | Reason given if the human rejected.                                      |
| `experiment_spec`           | ExperimentSpec          | Structured experiment design (IV, DV, control, dataset, metrics).        |
| `hitl_experiment_approved`  | bool                    | Whether the human approved the experiment design (second HITL gate).     |
| `python_code`               | str                     | The Python experiment script generated by the ML Coder.                  |
| `resolved_dependencies`     | List[str]               | pip packages identified by AST-parsing the generated code.               |
| `resolved_datasets`         | List[str]               | HF/sklearn dataset IDs identified by AST-parsing the generated code.     |
| `dataset_cache_path`        | str                     | Host-side cache directory mounted as `:ro` into the Docker sandbox.      |
| `debug_instrumentation`     | str                     | Code augmented with strategic debug `print()` statements.                |
| `execution_success`       | bool                    | Whether the experiment executed successfully.                            |
| `execution_logs`          | str                     | Stdout/Stderr output from the Docker sandbox (includes debug prints).    |
| `metrics_json`            | str                     | JSON string with experiment results (accuracy, F1-score, etc.).          |
| `code_retry_count`        | int                     | Tracks code retry attempts (max 3).                                      |
| `claim_ledger`            | List[ClaimLedgerEntry]  | Traces every paper claim to supporting/contradicting KG evidence.        |
| `latex_draft`             | str                     | The LaTeX draft source code (`draft.tex`).                               |
| `bibtex_source`           | str                     | The generated bibliography (`references.bib`).                           |
| `critique_warnings`       | List[Dict]              | Structured warnings from the 3-agent heterogeneous critique panel.       |
| `debate_log`              | List[DebateEntry]       | Full transcript of the structured debate (challenges + responses).       |
| `surviving_critiques`     | List[Dict]              | Only critiques that survived the debate protocol (not retracted).        |
| `confidence_score`        | float                   | Writer's self-assessed confidence score (1–10) after revision.           |
| `revision_pass_done`      | bool                    | Whether the single mandatory revision pass has been completed.           |
| `latex_compile_log`       | str                     | Raw `.log` output from `pdflatex` (used by repair loop).                 |
| `latex_repair_attempts`   | int                     | Number of repair iterations attempted (max 5).                           |
| `final_pdf_path`          | Optional[str]           | Path to the compiled PDF, or `None` if No-Paper outcome.                 |
| `pipeline_status`         | str                     | Current status: running, awaiting_hitl_hypothesis, awaiting_hitl_experiment, success, no_paper, or failed_*. |

### Phase 3: Build the Nodes

Implement each of the 14 nodes as described in Section 2 above, each in its own Python module under `backend/agents/`:

- `arxiv_retriever.py` — Iterative full-text ArXiv paper download and section extraction; TF-IDF keyword refinement for rounds 2+ (non-AI)
- `kg_extractor.py` — Epistemic KG extraction with polarity + `context_condition` for conditional claims + SBERT entity clustering + LLM dedup (AI)
- `hypothesis_generator.py` — Incremental hypothesis generation with KG grounding (exploiting contested pairs + conditional edges) + SBERT novelty scoring + prior-art screening (AI)
- `hitl_gate.py` — Human-in-the-loop hypothesis approval checkpoint (non-AI)
- `experiment_designer.py` — Structured ExperimentSpec generation from approved hypothesis (AI)
- `hitl_experiment_gate.py` — Human-in-the-loop experiment approval checkpoint (non-AI)
- `ml_coder.py` — Constrained experiment code generation bound to ExperimentSpec + active debug injection + **static-import-only enforcement** (no `importlib`/`exec()`/`eval()`, pre-compiled wheels only) (AI)
- `dependency_resolver.py` — AST parsing + host-side dependency/dataset pre-fetch (non-AI)
- `executor.py` — Docker sandbox runner with `:ro` cache volume mounts (non-AI)
- `claim_ledger_builder.py` — Claim traceability construction (respects `context_condition` for evidence strength) + No-Paper gate (non-AI)
- `academic_writer.py` — Claim ledger-grounded LaTeX/BibTeX draft generation + revision pass (AI)
- `deterministic_linter.py` — Rule-based pre-check (IMRaD, citations, claim compliance) (non-AI)
- `critique_panel.py` — 3-agent heterogeneous critique with structured debate protocol + claim ledger (AI)
- `critique_aggregator.py` — Linter + debate-surviving warning aggregation and mandatory revision routing (non-AI)
- `latex_compiler.py` — PDF compilation + LaTeX Repair Agent loop (non-AI + AI)

Shared utilities under `backend/utils/`:
- `embeddings.py` — SBERT embedding + novelty (RND) + prior-art similarity computation
- `kg_utils.py` — KG entity clustering + deduplication + polarity + `context_condition` logic
- `ast_parser.py` — AST-based dependency/dataset extraction from generated code (relies on static imports — validated by ML Coder constraints)
- `latex_utils.py` — pdflatex log parsing (line number + error + context extraction)
- `claim_utils.py` — Claim ledger construction + evidence strength rating (accounts for `context_condition` when scoring)
- `docker_utils.py` — Docker container lifecycle management with `:ro` volume mounts
- `state_pruning.py` — `build_scoped_view(state, node_name)` utility + `NODE_SCOPE_CONFIG` dictionary for per-node state field allowlists

### Phase 4: Orchestration with LangGraph

1. Register the 14 nodes as graph nodes.
2. **Wire scoped state views:** Before each AI-powered node invocation, the orchestrator calls `build_scoped_view(state, node_name)` to construct a pruned state containing only the fields that node needs (see §2 Context Management). Non-AI nodes receive the full state since they don't incur LLM token costs.
3. Define Phase 1 edges: START → Iterative ArXiv Retriever → Epistemic KG Extractor (with post-processing dedup + polarity + `context_condition`) → Incremental Hypothesis Generator (with novelty scoring + prior-art screening).
4. Define conditional edges at Hypothesis Generator:
   - KG validation passes AND novelty passes AND prior-art screening passes → HITL Gate 1.
   - KG validation fails (hallucination) → regenerate hypothesis.
   - Novelty score below threshold OR prior-art similarity above ceiling → END (failed_novelty report).
   - Iterative retrieval trigger → back to Iterative ArXiv Retriever (up to `max_retrieval_rounds`).
5. Define conditional edge at HITL Gate 1:
   - Human approves hypothesis → Experimental Designer.
   - Human rejects → END (failed_hitl_rejected report).
6. Define edges: Experimental Designer → HITL Gate 2.
7. Define conditional edge at HITL Gate 2:
   - Human approves experiment → Constrained ML Coder.
   - Human rejects → back to Experimental Designer (redesign) or END (if operator aborts).
8. Define Phase 2 edges: Constrained ML Coder (scoped: `experiment_spec` + `hypothesis` only) → Dependency Resolver → Executor.
9. Define conditional edges at Executor:
   - Success → Claim Ledger Builder.
   - Failure (code_retry_count < 3) → ML Coder (scoped: adds `python_code` + `execution_logs` for retry context).
   - Failure (code_retry_count ≥ 3) → END (failure report).
10. Define conditional edge at Claim Ledger Builder:
    - Evidence sufficient (≤ 50% weak/unsupported claims) → Academic Writer.
    - Evidence insufficient (> 50% weak/unsupported) → END (no_paper report).
11. Define Phase 3 edge: Academic Writer (scoped: `claim_ledger` + `experiment_spec` + `metrics_json` + `incremental_delta` + `hypothesis`) → Deterministic Linter → Critique Panel (scoped: `latex_draft` + `bibtex_source` + `metrics_json` + `claim_ledger`).
12. Define Phase 4 edges: Critique Panel (with debate) → Critique Aggregator (linter + debate warnings) → Academic Writer (one mandatory revision pass).
13. Define Phase 5 edges: Academic Writer (revised) → LaTeX Compiler (with repair loop) → END.
14. Compile and expose the graph via a `run_pipeline(topic: str)` function.

### Phase 5: Testing & Iteration

1. **Happy Path:** "Compare the accuracy of a Random Forest vs. Logistic Regression on the Hugging Face `imdb` sentiment dataset."
2. **Forced Failure:** Inject a deliberate error (e.g., reference a non-installed library) to validate the self-healing retry loop.
3. **Novelty Rejection:** Supply a hypothesis nearly identical to existing literature and verify the novelty gate (RND + prior-art similarity) blocks it.
4. **HITL Rejection (Gate 1):** Reject the hypothesis at HITL Gate 1 and verify the pipeline terminates gracefully.
5. **HITL Rejection (Gate 2):** Approve hypothesis but reject experiment design and verify the pipeline routes back to the Experimental Designer.
6. **No-Paper Outcome:** Run a topic with sparse/contradictory literature and verify the claim ledger builder triggers the No-Paper gate.
7. **LaTeX Repair:** Inject a deliberate LaTeX error (unclosed environment) and verify the repair loop fixes it.
8. **Debate Protocol:** Submit a draft with a hallucinated claim and verify the Fact-Checker flags it via claim ledger and the claim survives/fails the debate.
9. **AST Fragility:** Inject code using `importlib.import_module()` and verify the ML Coder's prompt constraints prevent it; if it slips through, verify the Dependency Resolver flags the unresolvable dynamic import.
10. **State Pruning:** Verify that the ML Coder's scoped view contains only `experiment_spec` + `hypothesis` (not raw papers or KG); verify the Academic Writer's scoped view excludes `execution_logs` and `arxiv_papers_full_text`.
11. **Conditional Claims (`context_condition`):** Provide papers with conditional findings (e.g., "method A outperforms B only on small datasets") and verify the KG edges carry the boundary condition; verify the Hypothesis Generator does not overgeneralize the conditional claim into an unconditional hypothesis.

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
│   │   ├── arxiv_retriever.py       # Node 1 — iterative full-text retrieval
│   │   ├── kg_extractor.py          # Node 2 — epistemic KG extraction + polarity + dedup
│   │   ├── hypothesis_generator.py  # Node 3 — incremental hypothesis + novelty + prior-art
│   │   ├── hitl_gate.py             # Node 3b — hypothesis approval checkpoint (HITL 1)
│   │   ├── experiment_designer.py   # Node 3c — structured ExperimentSpec generation
│   │   ├── hitl_experiment_gate.py  # Node 3d — experiment approval checkpoint (HITL 2)
│   │   ├── ml_coder.py             # Node 4 — constrained code gen + debug injection
│   │   ├── dependency_resolver.py   # Node 4b — AST parsing + host-side pre-fetch
│   │   ├── executor.py              # Node 5 — Docker sandbox with :ro mounts
│   │   ├── claim_ledger_builder.py  # Node 5b — claim traceability + No-Paper gate
│   │   ├── academic_writer.py       # Node 6 — claim ledger-grounded LaTeX drafting
│   │   ├── deterministic_linter.py  # Node 6b — rule-based pre-check
│   │   ├── critique_panel.py        # Node 7 — heterogeneous debate protocol
│   │   ├── critique_aggregator.py   # Node 8 — linter + debate-surviving warning filter
│   │   └── latex_compiler.py        # Node 9 — pdflatex + repair loop
│   └── utils/
│       ├── __init__.py
│       ├── embeddings.py            # SBERT embedding + novelty (RND) + prior-art scoring
│       ├── kg_utils.py              # KG entity clustering + deduplication + polarity
│       ├── ast_parser.py            # AST-based dependency/dataset extraction
│       ├── latex_utils.py           # pdflatex log parsing
│       ├── claim_utils.py           # Claim ledger construction + evidence strength
│       ├── docker_utils.py          # Docker container lifecycle + :ro mounts
│       └── state_pruning.py         # build_scoped_view() + NODE_SCOPE_CONFIG
│
├── tests/
│   ├── test_arxiv_retriever.py
│   ├── test_kg_extractor.py
│   ├── test_hypothesis_generator.py
│   ├── test_hitl_gate.py
│   ├── test_experiment_designer.py
│   ├── test_hitl_experiment_gate.py
│   ├── test_ml_coder.py
│   ├── test_dependency_resolver.py
│   ├── test_executor.py
│   ├── test_claim_ledger_builder.py
│   ├── test_academic_writer.py
│   ├── test_deterministic_linter.py
│   ├── test_critique_panel.py
│   ├── test_critique_aggregator.py
│   ├── test_latex_compiler.py
│   ├── test_state_pruning.py
│   └── evals/
│       ├── eval_kg_extractor.py
│       ├── eval_hypothesis.py
│       ├── eval_novelty.py
│       ├── eval_experiment_designer.py
│       ├── eval_coder.py
│       ├── eval_claim_ledger.py
│       ├── eval_writer.py
│       ├── eval_linter.py
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
| `test_arxiv_retriever.py`        | arXiv search returns full-text results; iterative retrieval refines queries correctly.  |
| `test_kg_extractor.py`           | KG output uses schema-based typed entities with polarity + `context_condition`; SBERT dedup merges synonyms; contradictions preserved; conditional claims carry boundary conditions. |
| `test_hypothesis_generator.py`   | Hypothesis is non-empty; grounded in KG; incremental_delta is present; novelty score + prior-art similarity computed and gated. |
| `test_hitl_gate.py`              | Pipeline pauses correctly at HITL Gate 1; approve/reject flows work; state fields updated properly. |
| `test_experiment_designer.py`    | ExperimentSpec is well-formed with all required fields; dataset IDs are real and verifiable. |
| `test_hitl_experiment_gate.py`   | Pipeline pauses correctly at HITL Gate 2; approve routes to ML Coder; reject routes back to designer. |
| `test_ml_coder.py`               | Generated code is syntactically valid; implements approved ExperimentSpec; contains debug print instrumentation; uses only static imports (no `importlib`/`exec()`/`eval()`); uses only pre-compiled-wheel packages. |
| `test_dependency_resolver.py`    | AST parser correctly identifies imports and `load_dataset()` calls; cache paths set.    |
| `test_executor.py`               | Docker container starts with `--network=none` + `:ro` mounts; exit codes captured.     |
| `test_claim_ledger_builder.py`   | Claim ledger maps assertions to KG edges; evidence strength is correctly rated (accounts for `context_condition`); No-Paper gate triggers on insufficient evidence. |
| `test_academic_writer.py`        | LaTeX draft is valid; contains IMRaD sections; BibTeX file is well-formed; only strong/moderate claims included. |
| `test_deterministic_linter.py`   | Linter catches missing sections, orphaned citations, claim-ledger violations, and raw arXiv IDs. |
| `test_critique_panel.py`         | 3 agents use different models/personas; debate protocol produces challenges/responses; claim ledger is queried. |
| `test_critique_aggregator.py`    | Linter warnings + debate-surviving (unretracted) critiques are forwarded to the Writer. |
| `test_state_pruning.py`          | `build_scoped_view()` returns only allowed fields per node; full state is never mutated; `NODE_SCOPE_CONFIG` covers all AI nodes. |
| `test_latex_compiler.py`         | PDF generated on success; repair loop triggered on failure; max attempts respected.     |

### 7.2 Integration Tests

- **End-to-end pipeline test:** Run the full graph on a known-good topic (with auto-approved dual HITL) and assert that `final_pdf_path` points to a valid PDF.
- **Code retry loop test:** Feed a deliberately broken `python_code`, assert that `code_retry_count` increments and the ML Coder is re-invoked with debug logs.
- **Anti-hallucination test:** Provide a hypothesis with entities not in the KG and assert that it is rejected and regenerated.
- **Novelty gate test:** Supply a hypothesis nearly identical to existing literature and assert `novelty_passed = False` (via RND or prior-art similarity) and pipeline terminates with `failed_novelty`.
- **HITL rejection test (Gate 1):** Reject the hypothesis at HITL Gate 1 and assert pipeline terminates with `failed_hitl_rejected`.
- **HITL rejection test (Gate 2):** Approve hypothesis but reject experiment design at HITL Gate 2 and assert pipeline routes back to Experimental Designer.
- **No-Paper outcome test:** Provide a KG with mostly `unsupported` claims and assert pipeline terminates with `no_paper` status and `final_pdf_path = None`.
- **Claim ledger compliance test:** Submit a draft that includes a `weak`-evidence claim and assert the deterministic linter flags it.
- **Iterative retrieval test:** Assert that hypothesis-derived terms trigger additional ArXiv queries in round 2+ and new papers are added (deduplicated by arXiv ID).
- **Debate protocol test:** Submit a draft with a hallucinated citation and assert the Fact-Checker flags it via claim ledger + KG, the debate challenges it, and it survives into `surviving_critiques`.
- **LaTeX repair test:** Submit a `.tex` with a deliberate unclosed `\begin{table}` and assert the repair loop fixes it within 5 attempts.
- **Confidence score test:** Assert that the revised draft includes a confidence score (1–10) and the NeurIPS reproducibility checklist.
- **AST fragility test:** Inject code using `importlib.import_module()` or `exec("import X")` and assert the pipeline rejects it or the Dependency Resolver flags unresolvable dynamic imports.
- **State pruning test:** Run a full pipeline and assert that the ML Coder's LLM prompt does not contain `arxiv_papers_full_text` or `kg_entities`; assert the Academic Writer's prompt does not contain `execution_logs`.
- **Conditional claims test (`context_condition`):** Provide papers with conditional findings and assert KG edges carry boundary conditions; assert the claim ledger rates conditional evidence supporting an unconditional claim as weaker than unconditional evidence.

### 7.3 Agent Evals (LLM-Specific)

These evals measure agent quality beyond pass/fail:

| Eval                                 | Method                                                                                                    | Pass Criteria                          |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| **KG Extractor: Schema Compliance**  | Verify KG output conforms to typed KGEntity/KGEdge schemas with all required fields.                      | 100% valid structure on 10/10 runs     |
| **KG Extractor: Dedup Quality**      | Inject synonymous entities ("RF", "Random Forest") and verify SBERT clustering merges them.               | ≥ 90% synonyms correctly merged        |
| **KG Extractor: Polarity Accuracy**  | Inject papers with contradictory findings and verify edges have correct `supports`/`contradicts` polarity. | ≥ 85% polarity correctly assigned       |
| **KG Extractor: Depth**              | Verify KG contains granular triplets (hyperparams, preprocessing steps), not just abstract concepts.      | ≥ 5 technical-detail triplets per paper |
| **Hypothesis: Anti-Hallucination**   | Verify all entities in hypothesis exist in KG triplets. Inject fake entities and confirm rejection.       | 100% hallucinated hypotheses rejected  |
| **Hypothesis: Incremental Delta**    | Verify `incremental_delta` is non-empty and references specific prior art from the KG.                   | Delta present and grounded on 10/10    |
| **Hypothesis: Prior-Art Screening**  | Test with near-paraphrase hypotheses; verify `prior_art_similarity_score` catches them.                  | 100% near-paraphrases caught           |
| **Hypothesis: Novelty Scoring**      | Test with known-novel and known-redundant hypotheses; verify RND scores separate them.                    | AUC ≥ 0.80 on novel vs. redundant     |
| **Coder: Syntax Validity**           | Parse `python_code` with `ast.parse()`.                                                                   | No `SyntaxError` on 10/10 runs        |
| **Coder: Debug Instrumentation**     | Verify generated code contains strategic `print()` at data load, train, and eval checkpoints.             | ≥ 3 debug prints per script            |
| **Coder: ML Rigor**                  | Verify generated code contains train/test split, random seeds, and cross-validation.                      | All 3 practices present on 10/10 runs  |
| **Coder: ExperimentSpec Compliance** | Verify generated code uses the dataset, metrics, and variables from the approved ExperimentSpec.           | 100% spec compliance on 10/10 runs     |
| **Experiment Designer: Completeness**| Verify ExperimentSpec contains all required fields (IV, DV, control, dataset, metrics, expected outcome).  | All 6 fields present on 10/10 runs     |
| **Claim Ledger: Accuracy**           | Inject KG with known support/contradiction patterns; verify evidence strength ratings are correct.        | ≥ 90% ratings correct                  |
| **Claim Ledger: No-Paper Gate**      | Inject mostly unsupported claims; verify No-Paper outcome triggers.                                       | 100% correct gate decisions            |
| **Linter: Detection Rate**           | Inject 10 common structural issues (missing sections, orphaned cites, raw arXiv IDs); verify detection.   | ≥ 9/10 issues detected                |
| **KG Extractor: Context Conditions** | Inject papers with conditional claims (e.g., "only on small datasets"); verify `context_condition` is populated. | ≥ 85% conditional claims captured      |
| **Coder: Static Import Compliance**  | Verify generated code contains no `importlib`, `exec()`, `eval()`, `__import__()`, or `subprocess` usage. | 0 violations on 10/10 runs             |
| **Coder: Pre-Compiled Wheels Only**  | Verify all imports map to packages in the pre-compiled allowlist (no C-compiler-dependent packages).      | 100% allowlist compliance on 10/10     |
| **State Pruning: Field Isolation**   | Verify each AI node's scoped view contains only its allowed fields per `NODE_SCOPE_CONFIG`.               | 100% correct scoping on all nodes      |
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
- Show the **epistemic Knowledge Graph** (with SBERT-clustered entities, polarity, and `context_condition`) — highlight a contested edge pair (supports vs. contradicts) and a conditional edge with its boundary condition.
- Show the **incremental delta** and how the hypothesis exploits contradictions and conditional findings in the KG.
- Show the **novelty score** and **prior-art similarity score** computations and how they compare to thresholds.
- Demonstrate **HITL Gate 1**: show the Rich-formatted approval prompt with hypothesis, incremental delta, KG triples (with polarity), and novelty scores.
- Show the **Experimental Designer** output: the structured ExperimentSpec with IV, DV, control, dataset, metrics.
- Demonstrate **HITL Gate 2**: show the experiment approval prompt and approve the design.
- Show the **Dependency Resolver** output: AST-parsed static imports and datasets, host-side cache, `:ro` volume mounts. Highlight that only pre-compiled-wheel packages are resolved (no C-compiler dependencies).
- Show the **scoped state views**: demonstrate that the ML Coder receives only `experiment_spec` + `hypothesis` (not the full 50K+ token state), and the Academic Writer receives only the claim ledger + metrics (not raw papers or execution logs).
- Show the **claim ledger**: how each paper claim maps to supporting/contradicting KG evidence with strength ratings, including how `context_condition` affects evidence strength (conditional evidence for unconditional claims is rated weaker).
- Show the **deterministic linter** output: structural checks run before the LLM critique panel.
- Show the **heterogeneous Critique Panel** warnings from each agent (Fact-Checker via KG + claim ledger query, Methodologist with ExperimentSpec check, Formatter).
- Demonstrate the **structured debate protocol**: show challenges, responses, and which critiques survive.
- Show the **LaTeX repair loop** in action: trigger a compilation error and watch the Repair Agent fix it.
- Open the generated NeurIPS PDF and walk through each section.
- Trigger a forced failure to demonstrate the self-healing retry loop.
- Demonstrate the **No-Paper outcome**: run a topic with insufficient evidence and show the pipeline terminating gracefully with a report.

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
| Design Review & Overhauls    | Claude                                 | Analyzed architectural evaluation PDF and epistemic peer review; implemented 9 critical overhauls (KG polarity, iterative retrieval, prior-art screening, double HITL, claim ledger, No-Paper outcome, deterministic linter, context conditions, state pruning) plus 3 operational safeguards (AST fragility fix, scoped state views, conditional claims). |
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
| 14 Nodes (8 AI-powered + 6 non-AI) in 5 Phases + 2 HITL | §2 | Planned |
| Context Management — Scoped State Views per AI Node | §2 (Context Management) | Planned |
| Iterative ArXiv Retrieval (hypothesis-driven refinement, up to 3 rounds) | §2 (Node 1) | Planned |
| Epistemic KG Extraction + Polarity + `context_condition` + SBERT Dedup | §2 (Node 2) | Planned |
| Incremental Hypothesis + Novelty Detection + Prior-Art Screening | §2 (Node 3) | Planned |
| HITL Gate 1: Hypothesis Approval | §2 (Node 3b) | Planned |
| Experimental Designer (Structured ExperimentSpec) | §2 (Node 3c) | Planned |
| HITL Gate 2: Experiment Approval | §2 (Node 3d) | Planned |
| Constrained ML Coder (ExperimentSpec-bound + static imports + pre-compiled wheels) | §2 (Node 4) | Planned |
| AST-Based Dependency Resolver (Network-Isolation Fix) | §2 (Node 4b) | Planned |
| Docker Sandbox with `:ro` Cache Volume Mounts | §2 (Node 5) | Planned |
| Claim Ledger Builder + No-Paper Outcome Gate (`context_condition`-aware) | §2 (Node 5b) | Planned |
| Claim Ledger-Grounded Academic Writer | §2 (Node 6) | Planned |
| Deterministic Linter (IMRaD, citations, claim compliance) | §2 (Node 6b) | Planned |
| Heterogeneous Review Panel + Structured Debate + Claim Ledger | §2 (Node 7) | Planned |
| KG + Claim Ledger-Grounded Fact-Checking (JSON Path Traversals) | §2 (Node 7, Agent A) | Planned |
| LaTeX Compiler Repair Loop (up to 5 attempts) | §2 (Node 9) | Planned |
| User Stories (31) + Product Backlog (5 sprints) | §3 | Planned |
| Diagrams (5 total) | §4 | Planned |
| Git Strategy (branches, PRs, conventional commits) | §6 | Planned |
| Automated Tests (16 unit + 15 integration) + Agent Evals (28 evals) | §7 | Planned |
| Bug Reporting + Resolution via PR | §8 | Planned |
| CI/CD Pipeline (GitHub Actions) | §9 | Planned |
| NeurIPS PDF Generation (LaTeX + BibTeX + Repair Loop) | §2 (Phase 5) | Planned |
| AI Usage Report (11 areas + metrics) | §11 | Planned |
| Live Demo + Screencast | §10 | Planned |

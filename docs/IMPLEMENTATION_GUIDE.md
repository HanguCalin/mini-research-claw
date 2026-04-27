# Auto-Mini-Claw: Step-by-Step Implementation Guide

> **Purpose:** This document translates the high-level architectural plan (`Mini_Research_Claw_Full_Plan.md`) into a chronological, actionable developer roadmap. It tells you *what to build*, *in what order*, and *with what constraints* — without writing the actual code. Every instruction traces back to a specific section of the plan.

---

## Table of Contents

1. [Environment & Database Setup](#1-environment--database-setup)
2. [Global State & Context Management](#2-global-state--context-management)
3. [Node-by-Node Construction](#3-node-by-node-construction)
4. [LangGraph Orchestration](#4-langgraph-orchestration)
5. [Testing Strategy](#5-testing-strategy)

---

## 1. Environment & Database Setup

This phase establishes all infrastructure before any pipeline logic is written. Nothing here involves AI — it is pure scaffolding.

### 1.1 Python Project Initialization

1. Create the project root with a `pyproject.toml` (or `requirements.txt` if preferred).
2. Set up a virtual environment (`python -m venv .venv` or use `uv`).
3. Install all backend dependencies:

   | Package | Purpose |
   |---------|---------|
   | `anthropic` | Claude API client for all AI-powered nodes |
   | `langgraph` | Multi-agent DAG orchestration framework |
   | `arxiv` | arXiv search API wrapper |
   | `docker` | Python Docker SDK for sandbox container management |
   | `datasets` | Hugging Face dataset loading |
   | `huggingface_hub` | HF model/dataset CLI downloads |
   | `sentence-transformers` | SBERT (`all-MiniLM-L6-v2`) for embeddings |
   | `scikit-learn` | ML utilities, cosine similarity, clustering |
   | `numpy` | Numeric operations |
   | `rich` | Terminal formatting for HITL prompts and progress display |
   | `supabase` | Supabase Python client (PostgreSQL + Storage) |
   | `python-dotenv` | `.env` file loading |

4. Create the directory structure exactly as specified in the plan (§6.1):

   ```
   backend/
   ├── __init__.py
   ├── main.py               # CLI entry point
   ├── state.py               # AutoResearchState TypedDict
   ├── graph.py               # LangGraph DAG definition + routing
   ├── config.py              # Environment / model configuration
   ├── agents/                # One module per node (15 files)
   │   └── __init__.py
   └── utils/                 # Shared utilities (9 files)
       └── __init__.py
   ```

### 1.2 Docker Sandbox Image

1. Create `Dockerfile.sandbox` with a base Python image (e.g., `python:3.11-slim`).
2. Pre-install **only pre-compiled wheels** — packages that do not require system C compilers:
   - `pandas`, `scikit-learn`, `numpy`, `scipy`, `datasets`, `huggingface_hub`, `transformers`, `torch`, `matplotlib`, `seaborn`.
3. **Do NOT install** `gcc`, `g++`, `make`, or any build tools in this image. This is deliberate — it enforces the "pre-compiled wheels only" constraint from Node 4.
4. The sandbox filesystem should be read-only at runtime (except `/tmp` for scratch space).

### 1.3 LaTeX Toolchain

Install `texlive`, `pdflatex`, and `bibtex` on the **host machine** (not in the sandbox). These are used by Node 9 (LaTeX Compiler) which runs outside the Docker sandbox.

### 1.4 Environment Variables & Secrets

Create a `.env` file (add to `.gitignore` immediately) with:

```env
ANTHROPIC_API_KEY=sk-ant-...
SUPABASE_URL=https://<project-id>.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
```

Create a corresponding `.env.example` (committed to git) with placeholder values.

**Important:** Use the Supabase **service-role key**, not the anon key. The service key bypasses Row Level Security, which is required for server-side pipeline operations (inserting papers, uploading artifacts).

### 1.5 Supabase Project Provisioning

This is infrastructure-as-code that should be run once during project setup. Execute these SQL statements in the Supabase SQL Editor (or via a migration script):

**Step 1: Enable pgvector**

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Step 2: Create the `papers` table (paper corpus cache)**

```sql
CREATE TABLE papers (
  id            UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  arxiv_id      TEXT UNIQUE NOT NULL,
  title         TEXT NOT NULL,
  authors       TEXT[] NOT NULL,
  year          INT NOT NULL,
  abstract      TEXT,
  full_text     JSONB NOT NULL,          -- {methodology, implementation, results}
  embedding     vector(384) NOT NULL,    -- SBERT all-MiniLM-L6-v2
  created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON papers USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

Key design decisions:
- `arxiv_id` is `UNIQUE` — prevents duplicate papers across runs.
- `full_text` is `JSONB` — stores structured section extractions, not raw text.
- `embedding` is `vector(384)` — matches SBERT `all-MiniLM-L6-v2` output dimensionality.
- The IVFFlat index enables sub-millisecond cosine similarity queries on 10K+ papers.

**Step 3: Create the `pipeline_runs` table (run metadata + artifact pointers)**

```sql
CREATE TABLE pipeline_runs (
  id            UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  topic         TEXT NOT NULL,
  status        TEXT NOT NULL DEFAULT 'running',
  artifact_path TEXT,
  started_at    TIMESTAMPTZ DEFAULT now(),
  completed_at  TIMESTAMPTZ,
  metadata      JSONB
);
```

**Step 4: Create the `match_papers` RPC function**

Node 3 (Hypothesis Generator) calls this Postgres function to perform the pgvector cosine-similarity prior-art lookup. It accepts an exclusion list so the current run's freshly retrieved papers (which the hypothesis was grounded in) are not counted against novelty — without this, the gate is circular and every hypothesis trips it. See §3 Node 3.

```sql
DROP FUNCTION IF EXISTS match_papers(vector, int);

CREATE OR REPLACE FUNCTION match_papers(
  query_embedding   vector(384),
  match_count       int      DEFAULT 10,
  exclude_arxiv_ids text[]   DEFAULT '{}'
)
RETURNS TABLE (
  arxiv_id   text,
  title      text,
  similarity float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    papers.arxiv_id,
    papers.title,
    1 - (papers.embedding <=> query_embedding) AS similarity
  FROM papers
  WHERE papers.arxiv_id <> ALL(exclude_arxiv_ids)
  ORDER BY papers.embedding <=> query_embedding
  LIMIT match_count;
$$;
```

**Step 5: Create the `artifacts` Storage bucket**

In the Supabase Dashboard, go to **Storage** and create a bucket named `artifacts`. This stores pipeline outputs (PDFs, metrics, claim ledgers, LaTeX source, plus diagnostic artifacts on failure — see §3 Artifact Uploader) organized by `{run_id}/`.

### 1.6 Supabase Client Utility

Create `backend/utils/supabase_client.py` — a singleton that initializes the Supabase client from env vars (`SUPABASE_URL`, `SUPABASE_SERVICE_KEY`). This client is shared by:
- **Node 1** (ArXiv Retriever) — for paper cache lookups and inserts.
- **Node 3** (Hypothesis Generator) — for pgvector prior-art cosine queries via the `match_papers` RPC (§1.5 Step 4).
- **Artifact Uploader** — for Storage bucket uploads and `pipeline_runs` updates.

### 1.7 LLM Output-Parsing Utility

Create `backend/utils/llm_utils.py` exposing:
- `extract_text(response)` — pulls `response.content[0].text` from an Anthropic `Message` with a clear error if the content block is empty.
- `extract_json(text)` — strips ```` ```json ... ``` ```` markdown fences, tolerates prose preamble/trailing, finds the first balanced `{...}` or `[...]`, and parses it.

**Why this exists:** Claude routinely wraps JSON in markdown fences or prepends "Here is the JSON:" prose despite explicit prompt instructions. Calling `json.loads(response.content[0].text)` directly will crash with `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`. Every agent that parses structured LLM output must use `extract_json(extract_text(response))` instead.

This utility is used by Nodes 2, 3, 3c, 6, 7, 9 and `kg_utils._llm_pick_canonical`.

### 1.8 Configuration Module

Create `backend/config.py` to centralize:
- Model assignments per node (e.g., Node 2 → `claude-haiku-4-5-20251001`, Node 3 → `claude-sonnet-4-6`).
- Threshold constants: `novelty_threshold = 0.35`, `prior_art_ceiling = 0.90`, `max_retrieval_rounds = 3`, `max_code_retries = 3`, `max_latex_repair_attempts = 5`.
- Docker sandbox settings: memory limit, CPU limit, volume mount paths.

> **Early-stage testing tip.** When the `papers` cache is small (< ~100 rows), the prior-art ceiling fires on almost any hypothesis on a familiar topic because the corpus is dominated by the current run's freshly fetched papers. Temporarily set `prior_art_ceiling = 1.01` and `novelty_threshold = 0.0` to disable the gate while you smoke-test the rest of the pipeline. Restore to `0.90` / `0.35` once the corpus has hundreds of diverse rows.

---

## 2. Global State & Context Management

### 2.1 Implementing AutoResearchState

Create `backend/state.py` containing all TypedDicts. These are the data contracts that every node reads from and writes to.

**Supporting TypedDicts** (define these first, as `AutoResearchState` depends on them):

| TypedDict | Fields | Purpose |
|-----------|--------|---------|
| `KGEntity` | `id`, `canonical_name`, `aliases`, `entity_type`, `attributes` | A single deduplicated knowledge graph node |
| `KGEdge` | `source_id`, `target_id`, `relation`, `polarity`, `context_condition`, `confidence`, `provenance` | A directional, epistemic edge with polarity and boundary conditions |
| `DebateEntry` | `round`, `challenger_role`, `target_critique_index`, `challenge`, `response`, `resolved` | One round of the structured debate protocol |
| `ClaimLedgerEntry` | `claim_id`, `claim_text`, `supporting_kg_edges`, `contradicting_kg_edges`, `evidence_strength` | Maps a single paper claim to its KG evidence |
| `ExperimentSpec` | `independent_var`, `dependent_var`, `control_description`, `dataset_id`, `evaluation_metrics`, `expected_outcome` | The human-approved experiment contract |

**AutoResearchState** fields are organized by pipeline phase:

- **Phase 1 (Literature + KG):** `topic`, `arxiv_papers_full_text`, `retrieval_round`, `kg_entities`, `kg_edges`, `hypothesis`, `incremental_delta`, `hypothesis_embedding`, `novelty_score`, `prior_art_similarity_score`, `novelty_passed`
- **HITL Gates:** `hitl_approved`, `hitl_rejection_reason`, `experiment_spec`, `hitl_experiment_approved`
- **Phase 2 (Experimentation):** `python_code`, `resolved_dependencies`, `resolved_datasets`, `dataset_cache_path`, `debug_instrumentation`, `execution_success`, `execution_logs`, `metrics_json`, `code_retry_count`
- **Phases 3 & 4 (Drafting + Critique):** `claim_ledger`, `latex_draft`, `bibtex_source`, `critique_warnings`, `debate_log`, `surviving_critiques`, `confidence_score`, `revision_pass_done`
- **Phase 5 (Compilation):** `latex_compile_log`, `latex_repair_attempts`, `final_pdf_path`
- **Supabase:** `run_id`, `artifact_urls`
- **Telemetry:** `pipeline_status`, `total_api_calls`, `total_tokens_used`, `logs`

### 2.2 Implementing Scoped State Views (State Pruning)

Create `backend/utils/state_pruning.py` with two components:

**1. `NODE_SCOPE_CONFIG` dictionary** — maps each AI node name to the list of state fields it is allowed to see:

| Node | Allowed Fields |
|------|---------------|
| `kg_extractor` | `arxiv_papers_full_text` (one paper at a time — iterate externally) |
| `hypothesis_generator` | `kg_entities`, `kg_edges`, `topic` |
| `experiment_designer` | `hypothesis`, `incremental_delta`, `kg_entities`, `kg_edges` |
| `ml_coder` | `experiment_spec`, `hypothesis` |
| `ml_coder_retry` | `experiment_spec`, `hypothesis`, `python_code`, `execution_logs` |
| `academic_writer` | `claim_ledger`, `experiment_spec`, `metrics_json`, `incremental_delta`, `hypothesis` |
| `critique_panel` | `latex_draft`, `bibtex_source`, `metrics_json`, `claim_ledger` |
| `latex_repair` | *(special case — receives only extracted error context, not state fields)* |

**2. `build_scoped_view(state, node_name)` function** — takes the full `AutoResearchState` and a node name string, looks up the allowed fields in `NODE_SCOPE_CONFIG`, and returns a new dict containing only those fields. The original state is **never mutated** — this is purely a prompt-construction concern.

**Critical design constraint:** Non-AI nodes (Dependency Resolver, Executor, Claim Ledger Builder, HITL Gates, Deterministic Linter, Critique Aggregator) receive the full state because they don't incur LLM token costs and may need access to any field for routing logic.

---

## 3. Node-by-Node Construction

Build each node as a standalone Python module in `backend/agents/`. Each module exports a single function with the signature `def node_name(state: AutoResearchState) -> dict` that returns the state updates (LangGraph convention).

### Phase 1: Iterative Literature Parsing & Epistemic KG

---

#### Node 1: Iterative Full-Text ArXiv Retriever

| Attribute | Value |
|-----------|-------|
| **File** | `backend/agents/arxiv_retriever.py` |
| **Type** | Non-AI (pure Python) |
| **Input State** | `topic` (round 1), `hypothesis` + `kg_entities` (rounds 2+), `retrieval_round`, `arxiv_papers_full_text` |
| **Output State** | Updated `arxiv_papers_full_text`, incremented `retrieval_round` |
| **Libraries** | `arxiv`, `supabase`, `sentence-transformers`, `scikit-learn` (TF-IDF) |

**What to build:**

1. **Round 1 (topic-driven):** Take the user's `topic` string and construct an arXiv API query via the `arxiv` library. Fetch the top 3-5 matching papers.

2. **Rounds 2+ (hypothesis-driven):** Extract the top-5 TF-IDF keywords from `hypothesis` text and the canonical names of KG entities with the highest edge-count. Construct a refined arXiv query from these terms.

3. **For each arXiv result, implement cache-first retrieval:**
   - Query the Supabase `papers` table: `SELECT * FROM papers WHERE arxiv_id = :id`.
   - **Cache hit:** Read `full_text`, `embedding`, and metadata directly from the database. Skip arXiv download.
   - **Cache miss:** Download the full text from arXiv (HTML endpoints or LaTeX source). Extract `Methodology`, `Implementation`, and `Results` sections using regex-based section parsers. Compute an SBERT embedding (`all-MiniLM-L6-v2`, 384 dimensions). `INSERT INTO papers` with all fields for future reuse.

4. **Deduplication:** Skip papers whose arXiv ID already exists in `arxiv_papers_full_text[]` (exact string match).

5. **Rate limiting:** Enforce `time.sleep(3)` between consecutive arXiv API/HTTP requests (arXiv ToS compliance).

6. **Convergence:** If a round adds < 2 new papers, terminate early (diminishing returns). Max rounds: 3 (configurable via `max_retrieval_rounds`).

---

#### Node 2: Epistemic KG Extractor

| Attribute | Value |
|-----------|-------|
| **File** | `backend/agents/kg_extractor.py` |
| **Type** | AI (Claude Haiku 4.5 (`claude-haiku-4-5-20251001`)) |
| **Input State** | `arxiv_papers_full_text` (iterated one paper at a time) |
| **Output State** | `kg_entities`, `kg_edges` |
| **Utilities** | `backend/utils/kg_utils.py`, `backend/utils/embeddings.py` |
| **Scoped View** | Only one paper's text at a time (not all papers at once) |

**What to build:**

**Part A — LLM Extraction (per paper):**

1. For each paper in `arxiv_papers_full_text`, call Claude Haiku with a schema-based system prompt that enforces:
   - **Entity types:** `model`, `dataset`, `metric`, `method`, `hyperparameter`.
   - **Relation types:** `outperforms`, `uses_dataset`, `achieves_metric`, `has_hyperparameter`.
   - **Required edge fields:** `polarity` (`"supports"` | `"contradicts"` | `"neutral"`), `context_condition` (boundary condition string, empty `""` if unconditional), `confidence` (float), `provenance` (paper ID + section).
2. The prompt must enforce rigid JSON output boundaries — no free-form text.
3. The prompt must instruct the LLM to **never treat conditional support as absolute**. If a finding is conditional ("RF outperforms XGBoost *on small datasets*"), the `context_condition` must capture that boundary.

**Part B — Deterministic Post-Processing Pipeline (in `kg_utils.py` and `embeddings.py`):**

1. **SBERT entity deduplication:**
   - Embed all entity `canonical_name` values using SBERT (`all-MiniLM-L6-v2`) → 384-dim vectors.
   - Compute pairwise cosine similarity.
   - Cluster entities where cosine similarity > 0.85 into synonym groups.

2. **LLM dedup pass:**
   - For each synonym cluster, call Claude Haiku to pick the best canonical name and merge attributes.
   - Re-route all edges pointing to any alias in the cluster to the canonical entity ID.

3. **Epistemic edge resolution:**
   - Deduplicate edges with identical `(source, target, relation, polarity)` tuples — keep highest confidence.
   - **Contested pair detection:** For each `(entity_A, entity_B)` pair, check for edges with opposing claims. Flag both as `contested_pair`.
   - **Critical:** Do NOT discard contradictory edges. Preserve both sides.

4. **Incremental merge (rounds 2+):** When new papers arrive in later retrieval rounds, run the same dedup pipeline to merge new entities/edges into the existing KG without duplicating previously extracted knowledge.

---

#### Node 3: Incremental Hypothesis Generator

| Attribute | Value |
|-----------|-------|
| **File** | `backend/agents/hypothesis_generator.py` |
| **Type** | AI (Claude Sonnet 4.6 (`claude-sonnet-4-6`)) + deterministic post-step |
| **Input State** | `kg_entities`, `kg_edges`, `topic` |
| **Output State** | `hypothesis`, `incremental_delta`, `hypothesis_embedding`, `novelty_score`, `prior_art_similarity_score`, `novelty_passed` |
| **Utilities** | `backend/utils/embeddings.py`, `backend/utils/supabase_client.py` |
| **Scoped View** | `kg_entities`, `kg_edges`, `topic` only |

**What to build:**

**Part A — LLM Hypothesis Generation:**

1. System prompt must instruct:
   - Ground the hypothesis strictly in KG entities (anti-hallucination).
   - Pay special attention to **contested edges** — contradictions are prime targets for novel hypotheses.
   - Use only real, verifiable dataset IDs from HF Hub or scikit-learn.
   - Output an `incremental_delta` field: 2-3 sentences explaining what's new vs. closest prior art.
2. Validate that all entities mentioned in the hypothesis exist in `kg_entities` (programmatic check, not LLM). If validation fails, regenerate.

**Part B — Deterministic Novelty Detection + Prior-Art Screening:**

1. Embed the hypothesis using SBERT → `hypothesis_embedding`.
2. **Prior-art similarity via Supabase pgvector** — call the `match_papers` RPC defined in §1.5 Step 4, passing the current run's arXiv IDs as `exclude_arxiv_ids`:

   ```python
   current_run_ids = [p["arxiv_id"] for p in state["arxiv_papers_full_text"]]
   sb.rpc("match_papers", {
       "query_embedding":   hypothesis_embedding,
       "match_count":       10,
       "exclude_arxiv_ids": current_run_ids,
   }).execute()
   ```

   The `prior_art_similarity_score` is the **maximum** `similarity` value from the returned rows.

   > **Critical — exclude current-run papers.** The hypothesis is grounded in the papers Node 1 just fetched; comparing the hypothesis embedding against those same papers is circular and trips the gate on every successful generation. The `exclude_arxiv_ids` argument removes them from the prior-art pool so the score reflects distance from *historical* corpus only.

3. **Relative Neighbor Density (RND):** Average cosine distance from hypothesis to K nearest paper embeddings (from the same pgvector query, after exclusion).
4. **Dual gating:**
   - RND >= `novelty_threshold` (0.35) **AND** prior-art similarity < `prior_art_ceiling` (0.90) → `novelty_passed = True`.
   - Otherwise → `novelty_passed = False`, `pipeline_status = "failed_novelty"`.
5. **Iterative retrieval trigger:** If novelty passes and `retrieval_round < max_retrieval_rounds`, extract key technical terms from the hypothesis and feed them back to Node 1.

---

### HITL Checkpoint 1: Hypothesis Approval

#### Node 3b: HITL Gate 1

| Attribute | Value |
|-----------|-------|
| **File** | `backend/agents/hitl_gate.py` |
| **Type** | Non-AI (deterministic checkpoint) |
| **Input State** | `hypothesis`, `incremental_delta`, `kg_entities`, `kg_edges`, `novelty_score`, `prior_art_similarity_score`, `arxiv_papers_full_text` |
| **Output State** | `hitl_approved`, `hitl_rejection_reason`, `pipeline_status` |

**What to build:**

1. **CLI mode:** Use `rich` to render a formatted panel displaying:
   - The hypothesis text.
   - The incremental delta.
   - Supporting and contradicting KG triples (with polarity).
   - Novelty score and prior-art similarity score vs. thresholds.
   - Summary of retrieved paper titles/abstracts.
2. Block on `input()` — accept `approve` or `reject <reason>`.
3. **Web UI mode (future):** Emit `pipeline_status = "awaiting_hitl_hypothesis"` and expose an API endpoint. The React frontend polls this status and presents an approval dialog.
4. On approve → `hitl_approved = True`, proceed to Node 3c.
5. On reject → `hitl_approved = False`, `pipeline_status = "failed_hitl_rejected"`, pipeline terminates.

---

### Experiment Design

#### Node 3c: Experimental Designer

| Attribute | Value |
|-----------|-------|
| **File** | `backend/agents/experiment_designer.py` |
| **Type** | AI (Claude Sonnet 4.6 (`claude-sonnet-4-6`)) |
| **Input State** | `hypothesis`, `incremental_delta`, `kg_entities`, `kg_edges` |
| **Output State** | `experiment_spec` (ExperimentSpec TypedDict) |
| **Scoped View** | `hypothesis`, `incremental_delta`, `kg_entities`, `kg_edges` |

**What to build:**

1. System prompt instructs the LLM to output a structured `ExperimentSpec` JSON with all 6 required fields: `independent_var`, `dependent_var`, `control_description`, `dataset_id`, `evaluation_metrics`, `expected_outcome`.
2. The `dataset_id` must be a real, verifiable public dataset (HF Hub or scikit-learn).
3. Each choice should be justified in 1-2 sentences (for HITL reviewability).
4. Parse and validate the JSON output. All 6 fields must be present and non-empty.

---

#### Node 3d: HITL Gate 2

| Attribute | Value |
|-----------|-------|
| **File** | `backend/agents/hitl_experiment_gate.py` |
| **Type** | Non-AI (deterministic checkpoint) |
| **Input State** | `hypothesis`, `incremental_delta`, `experiment_spec`, relevant `kg_edges` |
| **Output State** | `hitl_experiment_approved`, `pipeline_status` |

**What to build:**

1. Display: hypothesis, incremental delta, the full ExperimentSpec, and relevant contested KG edges.
2. Operator actions:
   - `approve` → `hitl_experiment_approved = True`, proceed to Phase 2.
   - `reject` → route back to Node 3c for redesign, or abort if operator explicitly requests it.
3. CLI mode: `rich`-formatted panel + `input()` block.
4. Web UI mode (future): emit `pipeline_status = "awaiting_hitl_experiment"`.

---

### Phase 2: Autonomous Experimentation & Self-Healing

---

#### Node 4: Constrained ML Coder

| Attribute | Value |
|-----------|-------|
| **File** | `backend/agents/ml_coder.py` |
| **Type** | AI (Claude Sonnet 4.6 (`claude-sonnet-4-6`)) |
| **Input State** | `experiment_spec`, `hypothesis` (first attempt); adds `python_code`, `execution_logs` on retry |
| **Output State** | `python_code`, `debug_instrumentation` |
| **Scoped View** | `experiment_spec` + `hypothesis` only (first attempt); adds `python_code` + `execution_logs` on retry |

**What to build:**

1. **System prompt constraints (critical — these are hard requirements, not suggestions):**
   - **Bind to ExperimentSpec:** Must use the specified `dataset_id`, `evaluation_metrics`, `independent_var`/`dependent_var`. No deviation.
   - **Static imports only:** All imports must be explicit `import X` or `from X import Y` at the top of the script. **Strictly forbidden:** `importlib`, `importlib.import_module()`, `exec()`, `eval()`, `__import__()`.
   - **Pre-compiled wheels only:** Allowed packages: `scikit-learn`, `transformers`, `torch`, `pandas`, `numpy`, `scipy`, `datasets`, `huggingface_hub`, `matplotlib`, `seaborn`. No C-compiler-dependent packages.
   - **No subprocess:** No `subprocess`, `os.system()`, `shutil.which()`.
   - **ML rigor:** Train/test split, random seeds (`random_state=42`), cross-validation where applicable.
   - **Output `metrics.json`:** Save all hyperparameters and evaluation metrics.

2. **Active debugging:** Inject strategic `print()` at data loading, tensor shape checks, training loss per epoch, and final metric values. These debug prints are critical for the retry loop — they give the Coder diagnostic signal beyond bare stack traces.

3. **Retry behavior:** On retry, the Coder receives the previous `python_code` + full `execution_logs`. The prompt must demand root-cause analysis before rewriting — not blind regeneration.

   > **Output sanitization required.** When asked for "root-cause analysis", Claude reliably prepends a prose paragraph ("The previous code had…") to the response — even when the prompt says "Return ONLY Python code". The sandbox then tries to parse that prose as Python on line 1 and crashes with `SyntaxError`, exhausting all 3 retries on the same failure mode.
   >
   > Two defenses are required:
   > 1. **Strengthen the retry prompt** — explicitly say "The FIRST CHARACTER of your response must be a valid Python token" and "Do NOT prefix the code with sentences like 'The previous code…'".
   > 2. **Implement `_extract_python_code(text)`** — a robust extractor that prefers a ```` ```python ... ``` ```` fenced block if present, otherwise scans for the first line starting with a Python token (`import`, `from`, `def`, `class`, `#`, `"""`, `@`, `if __name__`, ALL_CAPS constants, etc.) and discards everything before it. Handles markdown fences, prose preamble, and trailing prose in one pass.

---

#### Node 4b: Dependency Resolver

| Attribute | Value |
|-----------|-------|
| **File** | `backend/agents/dependency_resolver.py` |
| **Type** | Non-AI (pure Python) |
| **Input State** | `python_code` |
| **Output State** | `resolved_dependencies`, `resolved_datasets`, `dataset_cache_path` |
| **Libraries** | Python `ast` module |

**What to build:**

1. **AST parsing:** Parse `python_code` with `ast.parse()`. Walk the AST to extract:
   - **pip dependencies:** All `import X` and `from X import Y` statements → map module names to PyPI package names (e.g., `sklearn` → `scikit-learn`, `cv2` → `opencv-python`).
   - **Remote datasets:** Detect `load_dataset("...")`, `fetch_openml(...)`, `sklearn.datasets.fetch_*` call patterns → extract dataset IDs.

2. **Host-side pre-fetch** (runs on the host with network access):
   - `pip download <packages> --dest .cache/pip/`
   - `huggingface-cli download <dataset_id> --cache-dir .cache/hf/`
   - For scikit-learn datasets: programmatically call `fetch_*` with `data_home='.cache/sklearn/'`.

3. Set `dataset_cache_path` to the host-side cache directory path.

**Why this node exists:** The Executor runs Docker with `--network=none`. Without pre-fetching, any `load_dataset()` or `pip install` call inside the sandbox crashes immediately. This node resolves the network-isolation contradiction.

---

#### Node 5: Executor Sandbox

| Attribute | Value |
|-----------|-------|
| **File** | `backend/agents/executor.py` |
| **Type** | Non-AI (pure Python) |
| **Input State** | `python_code` (or `debug_instrumentation`), `resolved_dependencies`, `dataset_cache_path`, `code_retry_count` |
| **Output State** | `execution_success`, `execution_logs`, `metrics_json`, `code_retry_count` |
| **Libraries** | `docker` (Python SDK) |
| **Utilities** | `backend/utils/docker_utils.py` |

**What to build:**

1. **Docker container configuration:**
   - `--network=none` — strict network isolation.
   - `--security-opt=no-new-privileges` — no privilege escalation.
   - `--read-only` filesystem (except `/tmp`).
   - `--memory=4g --cpus=2` — cgroup resource limits.
   - **Read-only volume mounts:**
     - `-v .cache/pip:/pip_cache:ro`
     - `-v .cache/hf:/hf_cache:ro`
     - `-v .cache/sklearn:/sklearn_cache:ro`
   - **Environment variables:** `PIP_FIND_LINKS=/pip_cache`, `HF_DATASETS_CACHE=/hf_cache`, `SCIKIT_LEARN_DATA=/sklearn_cache`.

2. **Routing logic (critical — drives the retry loop):**
   - `exit 0` → `execution_success = True`, capture stdout as `execution_logs`, read `metrics.json` from container → proceed to Phase 3.
   - `exit != 0` AND `code_retry_count < 3` → increment `code_retry_count`, capture full logs (including debug prints), route back to Node 4.
   - `exit != 0` AND `code_retry_count >= 3` → `pipeline_status = "failed_execution"`, terminate.

---

### Phase 3: Paper Drafting

---

#### Node 5b: Claim Ledger Builder

| Attribute | Value |
|-----------|-------|
| **File** | `backend/agents/claim_ledger_builder.py` |
| **Type** | Non-AI (deterministic Python) |
| **Input State** | `hypothesis`, `metrics_json`, `kg_entities`, `kg_edges` |
| **Output State** | `claim_ledger`, `pipeline_status` (if No-Paper gate triggers) |
| **Utilities** | `backend/utils/claim_utils.py` |

**What to build:**

1. **Enumerate claims from the hypothesis text only.** Split the hypothesis into individual assertive sentences (filter ones < 20 chars). These are the claims the paper actually intends to assert. Do **not** also enumerate per-metric, per-algorithm pseudo-claims by walking `metrics_json` — measurements are *evidence*, not claims, and naïve enumeration of every leaf produces dozens of `"The X achieves Y of Z"` strings that:
   - Don't contain any KG entity name → fail substring matching → all rated `unsupported`,
   - Drown the real hypothesis claims (typically 2–4) in noise → trip the No-Paper gate on every successful run.

   The Academic Writer (Node 6) still receives the full `metrics_json` via state and uses it to write the Results section. It just doesn't need claim-ledger entries for each individual measurement.
2. For each claim, find:
   - **Supporting KG edges:** Same polarity direction.
   - **Contradicting KG edges:** Opposing polarity.
   Match by checking whether either entity's canonical name (case-insensitive substring) appears in the claim text.
3. Rate `evidence_strength`:
   - `"strong"`: >= 2 supporting, 0 contradicting.
   - `"moderate"`: 1 supporting, or >= 2 supporting with >= 1 contradicting.
   - `"weak"`: 1 supporting with >= 1 contradicting.
   - `"unsupported"`: 0 supporting.
4. **`context_condition` awareness:** If a supporting edge has a non-empty `context_condition` but the claim is unconditional, the support is weaker (conditional evidence for an unconditional claim).
5. **No-Paper Gate:** If > 50% of claims are `unsupported` or `weak` → set `pipeline_status = "no_paper"`, `final_pdf_path = None`, and terminate the pipeline with a detailed report.

---

#### Node 6: Academic Writer

| Attribute | Value |
|-----------|-------|
| **File** | `backend/agents/academic_writer.py` |
| **Type** | AI (Claude Sonnet 4.6 (`claude-sonnet-4-6`)) |
| **Input State** | `claim_ledger`, `experiment_spec`, `metrics_json`, `incremental_delta`, `hypothesis` |
| **Output State** | `latex_draft`, `bibtex_source` (first pass); updated `latex_draft`, `confidence_score`, `revision_pass_done` (revision pass) |
| **Scoped View** | `claim_ledger`, `experiment_spec`, `metrics_json`, `incremental_delta`, `hypothesis` |

**What to build:**

**First pass (drafting):**

1. System prompt constraints:
   - IMRaD structure: `\section{Introduction}`, `\section{Methods}`, `\section{Results}`, `\section{Conclusion}`.
   - Only include claims rated `strong` or `moderate` in the claim ledger.
   - For claims with contradicting evidence, acknowledge the contradiction in the text.
   - Use `\cite{AuthorYear}` citations — no raw arXiv IDs in prose.
   - Generate a companion `references.bib` with proper BibTeX entries.
2. Output: `latex_draft` (the `.tex` source) and `bibtex_source` (the `.bib` file).

**Revision pass (after critique):**

1. The Writer receives the aggregated critique warnings (linter + debate-surviving critiques).
2. Must address each critique, produce a revised `latex_draft`.
3. Append a **Confidence Score** (self-assessed 1-10) and the NeurIPS reproducibility checklist.
4. Only one revision pass — no unbounded loops.

---

### Phase 4: Critique & Linting Engine

---

#### Node 6b: Deterministic Linter

| Attribute | Value |
|-----------|-------|
| **File** | `backend/agents/deterministic_linter.py` |
| **Type** | Non-AI (regex + LaTeX parsing) |
| **Input State** | `latex_draft`, `bibtex_source`, `claim_ledger` |
| **Output State** | Appends to `critique_warnings[]` with `source: "linter"` |

**What to build — 6 deterministic checks:**

1. **IMRaD completeness:** Regex for `\section{Introduction}`, `\section{Methods}`, `\section{Results}`, `\section{Conclusion}`.
2. **Citation integrity:** Extract all `\cite{...}` keys from `.tex`; verify each has a matching BibTeX entry in `.bib`. Flag orphaned citations.
3. **Claim-ledger compliance:** Cross-reference draft text against `claim_ledger[]`. Flag any claim that appears in the draft but has `unsupported` or `weak` evidence.
4. **NeurIPS checklist:** Verify the reproducibility checklist section exists.
5. **Figure/table labeling:** Verify every `\begin{figure}` and `\begin{table}` has `\label{}` and `\caption{}`.
6. **No raw arXiv IDs:** Regex for `arXiv:XXXX.XXXXX` patterns outside BibTeX blocks.

Linter warnings bypass the debate protocol — they are objective and non-debatable.

---

#### Node 7: Critique Panel (3 Heterogeneous Agents)

| Attribute | Value |
|-----------|-------|
| **File** | `backend/agents/critique_panel.py` |
| **Type** | AI (3 agents with different models/personas) |
| **Input State** | `latex_draft`, `bibtex_source`, `metrics_json`, `claim_ledger` (+ `kg_entities`, `kg_edges` for Agent A) |
| **Output State** | `critique_warnings`, `debate_log`, `surviving_critiques` |
| **Scoped View** | `latex_draft`, `bibtex_source`, `metrics_json`, `claim_ledger` |

**What to build — the 3 agents:**

| Agent | Role | Model | Key Constraint |
|-------|------|-------|----------------|
| **A: Fact-Checker** | Verify empirical claims against KG + claim ledger | Claude Sonnet 4.6 (`claude-sonnet-4-6`) | Must use **JSON path traversals** to query `kg_entities`/`kg_edges`/`claim_ledger` — not parametric memory. Must cite specific entity IDs and edge relations. Flag `ungrounded` claims and `contradiction_suppressed` issues. |
| **B: Methodologist** | Evaluate experimental rigor | Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) | Check if `metrics.json` results logically support conclusions. Verify ExperimentSpec compliance. Flag unsupported claims, missing error bars, statistical issues. |
| **C: Formatter** | Assess writing quality | Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) (different persona) | Focus on subjective quality: AI-slop writing style, verbosity, argumentation flow, clarity. Structural checks are handled by the linter. |

**Structured Debate Protocol (4 phases — replaces passive vote aggregation):**

1. **Independent critique:** Each agent independently critiques the draft → produces `critique_warnings[]`.
2. **Cross-challenge:** Each agent reads the other two agents' critiques. For each disagreement, issues a formal challenge explaining why the critique is incorrect/excessive.
3. **Defend-or-retract:** The original critic must defend (with evidence) or retract each challenged critique.
4. **Resolution:** Only unretracted critiques survive into `surviving_critiques[]`. Retracted ones are logged in `debate_log[]` but not forwarded.

> **Format-string trap.** The CHALLENGE_PROMPT and DEFEND_PROMPT templates use Python `str.format()` for `{critiques}`/`{challenge}` substitution AND contain literal JSON examples like `[{"target_critique_index": 0, ...}]`. Python's formatter sees `{"target_critique_index"}` as a format field with that quoted string as the lookup key → `KeyError: '"target_critique_index"'` at runtime, crashing the whole pipeline mid-debate.
>
> **Always escape literal JSON braces inside `.format()` templates as `{{...}}`.** Either review every template by hand or switch entirely to f-strings or `string.Template` to avoid the trap.

---

#### Node 8: Critique Aggregator

| Attribute | Value |
|-----------|-------|
| **File** | `backend/agents/critique_aggregator.py` |
| **Type** | Non-AI (pure Python) |
| **Input State** | `critique_warnings` (from linter), `surviving_critiques` (from debate) |
| **Output State** | Merged critique list routed to Node 6 for revision |

**What to build:**

1. Merge deterministic linter warnings (which bypassed debate) with debate-surviving critiques into a single structured feedback list.
2. Route the merged list back to Node 6 (Academic Writer) for exactly one mandatory revision pass.
3. After revision, the pipeline proceeds directly to Node 9 — no further review rounds.

---

### Phase 5: Publication

---

#### Node 9: LaTeX Compiler with Repair Loop

| Attribute | Value |
|-----------|-------|
| **File** | `backend/agents/latex_compiler.py` |
| **Type** | Non-AI (pdflatex) + AI (Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) for repair) |
| **Input State** | `latex_draft`, `bibtex_source` |
| **Output State** | `final_pdf_path`, `latex_compile_log`, `latex_repair_attempts`, `pipeline_status` |
| **Utilities** | `backend/utils/latex_utils.py` |

**What to build:**

1. **Compilation:** Write `latex_draft` to `draft.tex` and `bibtex_source` to `references.bib`. Run `pdflatex --no-shell-escape main.tex` + `bibtex main` via `subprocess.run()`.
   - **Security:** Always use `--no-shell-escape` to prevent malicious code execution from LLM-generated LaTeX.

2. **Deterministic missing-graphics pre-pass** (in `latex_utils.py`, function `neutralize_missing_graphics()`): Before the compile loop runs, scan `latex_draft` for every `\includegraphics[opts]{path}`. If the target file does **not** exist in the work directory and the options do not already contain `draft`, rewrite the line to add `,draft` (or `[draft]` if no options were given). pdflatex then renders the figure as a labeled placeholder box instead of failing with `! LaTeX Error: File 'foo.pdf' not found.`. This is a one-shot, regex-driven fix that costs zero LLM calls.
   - **Why this exists:** the Academic Writer routinely emits placeholder figure paths (e.g. `figures/placeholder_hypervolume.pdf`) without producing the file. The LLM repair loop has historically failed to spot the fix and burned all 5 attempts on this single class of error. Run `neutralize_missing_graphics()` once before the loop to eliminate that failure mode entirely.

3. **LaTeX Log Parser** (in `latex_utils.py`): On compilation failure, parse the `.log` file to extract:
   - Line number of the error.
   - Error type and message.
   - +/-5 lines of surrounding LaTeX context.
   - Do NOT feed the entire manuscript to the repair agent — only the localized error context.

4. **Repair loop:**
   - Send the extracted error context (line number + snippet + error message) to Claude Haiku.
   - The repair agent produces a targeted **line-level patch** (old line → new line).
   - Apply the patch surgically; do not touch untouched sections.
   - Re-invoke `pdflatex`.
   - Loop up to `max_latex_repair_attempts` (default: 5).
   - On success → `final_pdf_path` points to the compiled PDF.
   - On exhaustion → `pipeline_status = "failed_latex"`, preserve `.tex` source for manual inspection. Operators can rescue the run by downloading `draft.tex` and `references.bib` from `artifacts/{run_id}/` in Supabase Storage and recompiling locally; the `papers/` directory in the repo includes one such rescue.

---

#### Artifact Uploader (Post-Pipeline Step)

| Attribute | Value |
|-----------|-------|
| **File** | `backend/utils/artifact_uploader.py` |
| **Type** | Non-AI (Supabase Storage client) |
| **Triggers** | At every terminal state (success, failed_latex, no_paper, failed_novelty, failed_hitl, failed_execution) |

**What to build:**

1. At pipeline start: generate `run_id` (UUID), insert a row into `pipeline_runs` with `status = 'running'`.
2. At pipeline end: upload every available artifact to `artifacts/{run_id}/` in the Supabase Storage bucket. The full inventory:

   | File | Source state field | Always uploaded? |
   |---|---|---|
   | `metrics.json`           | `state.metrics_json` (JSON-dumped)   | When experiment ran |
   | `claim_ledger.json`      | `state.claim_ledger`                 | When ledger built |
   | `debate_log.json`        | `state.debate_log`                   | When critique panel ran |
   | `draft.tex`              | `state.latex_draft`                  | When writer ran |
   | `references.bib`         | `state.bibtex_source`                | When writer ran |
   | `python_code.py`         | `state.python_code`                  | When coder ran |
   | `execution_logs.txt`     | `state.execution_logs`               | When sandbox ran |
   | `hypothesis.txt`         | `state.hypothesis`                   | When hypothesis generated |
   | `experiment_spec.json`   | `state.experiment_spec` (JSON-dumped)| When spec produced |
   | `final_paper.pdf`        | binary at `state.final_pdf_path`     | Only on `success` |
   | `failure_report.json`    | synthesized                          | When status ≠ `success` |

   The diagnostic artifacts (`python_code.py`, `execution_logs.txt`, `hypothesis.txt`, `experiment_spec.json`) are critical: when an exception crashes the graph, the `state` returned from `graph.invoke()` is partial, and these fields are the only way to reconstruct what the pipeline was doing when it died.

   `failure_report.json` schema (when status ≠ `success`):
   ```json
   {
     "status": "...",
     "run_id": "...",
     "topic": "...",
     "retrieval_round": 1,
     "code_retry_count": 3,
     "latex_repair_attempts": null,
     "execution_logs_tail": "...last 4KB of execution_logs...",
     "logs": [...]
   }
   ```

   Skip missing files gracefully (e.g., `no_paper` exits before drafting, so no `.tex` exists).
3. Update the `pipeline_runs` row with: final `status`, `completed_at` timestamp, `artifact_path`, and `metadata` (token counts).
4. Populate `artifact_urls` in state with filename-to-storage-path mappings for frontend consumption.

---

## 4. LangGraph Orchestration

All orchestration logic lives in `backend/graph.py`. This is where the 14 nodes are wired into a directed acyclic graph with conditional routing.

### 4.1 Graph Structure

```
START
  │
  ▼
[Node 1: ArXiv Retriever]
  │
  ▼
[Node 2: KG Extractor]
  │
  ▼
[Node 3: Hypothesis Generator]
  │
  ├── KG valid + novel + prior-art OK ──────► [Node 3b: HITL Gate 1]
  ├── KG invalid (hallucination) ───────────► [Node 3: regenerate]
  ├── novelty < threshold OR prior-art ≥ ceiling ► END (failed_novelty)
  └── iterative retrieval trigger ──────────► [Node 1] (loop back)
  
[Node 3b: HITL Gate 1]
  ├── approved ──► [Node 3c: Experimental Designer]
  └── rejected ──► END (failed_hitl_rejected)

[Node 3c: Experimental Designer]
  │
  ▼
[Node 3d: HITL Gate 2]
  ├── approved ──► [Node 4: ML Coder]
  └── rejected ──► [Node 3c] (redesign) or END (abort)

[Node 4: ML Coder]
  │
  ▼
[Node 4b: Dependency Resolver]
  │
  ▼
[Node 5: Executor]
  ├── success ─────────────► [Node 5b: Claim Ledger Builder]
  ├── fail (retries < 3) ──► [Node 4: ML Coder] (retry with logs)
  └── fail (retries >= 3) ─► END (failed_execution)

[Node 5b: Claim Ledger Builder]
  ├── evidence sufficient ──► [Node 6: Academic Writer]
  └── >50% weak/unsupported ► END (no_paper)

[Node 6: Academic Writer] (first draft)
  │
  ▼
[Node 6b: Deterministic Linter]
  │
  ▼
[Node 7: Critique Panel + Debate]
  │
  ▼
[Node 8: Critique Aggregator]
  │
  ▼
[Node 6: Academic Writer] (revision pass)
  │
  ▼
[Node 9: LaTeX Compiler + Repair Loop]
  │
  ▼
[Artifact Upload]
  │
  ▼
END
```

### 4.2 How to Wire This in LangGraph

1. **Register all 14 nodes as graph nodes** using `graph.add_node("node_name", node_function)`.

2. **Scoped state views — self-pruning convention.** AI nodes only put the fields they're allowed to see (per `NODE_SCOPE_CONFIG`) into their LLM prompt. They still *receive* the full `AutoResearchState` from LangGraph because some agents need fields outside their scoped view for non-LLM orchestration logic — for example, `kg_extractor` reads `kg_entities`/`kg_edges` for incremental merge, and `hypothesis_generator` reads `arxiv_papers_full_text[*].arxiv_id` to populate the prior-art `exclude_arxiv_ids`. The `build_scoped_view()` utility from §2.2 is provided as an enforcement helper for tests and future tightening, but is not currently wired as graph-level middleware.

   > **Why not enforce wrapping?** Wrapping would require splitting every AI agent into two functions (one for full-state orchestration, one for LLM-prompt construction). Cheap to add later if drift becomes a problem; expensive to refactor 7 agents now. Self-pruning is verified by running the test in §5.1 (`test_state_pruning.py`) which asserts each AI prompt contains only allowed fields.

3. **Define edges** — straightforward connections where routing is unconditional:
   - `START` → `arxiv_retriever`
   - `arxiv_retriever` → `kg_extractor`
   - `kg_extractor` → `hypothesis_generator`
   - `hitl_gate` (approved) → `experiment_designer`
   - `experiment_designer` → `hitl_experiment_gate`
   - `ml_coder` → `dependency_resolver`
   - `dependency_resolver` → `executor`
   - `academic_writer` (first draft) → `deterministic_linter`
   - `deterministic_linter` → `critique_panel`
   - `critique_panel` → `critique_aggregator`
   - `critique_aggregator` → `academic_writer` (revision)
   - `academic_writer` (revision) → `latex_compiler`
   - `latex_compiler` → `artifact_uploader`
   - `artifact_uploader` → `END`

4. **Define conditional edges** — routing decisions based on state:

   **At Hypothesis Generator (Node 3):**
   ```
   def route_hypothesis(state):
       if not state["kg_valid"]:          return "hypothesis_generator"  # regenerate
       if not state["novelty_passed"]:    return END                     # failed_novelty
       if state["retrieval_round"] < max_retrieval_rounds:
                                          return "arxiv_retriever"       # iterative loop
       return "hitl_gate"                                                # proceed to HITL
   ```

   **At HITL Gate 1 (Node 3b):**
   ```
   def route_hitl_hypothesis(state):
       if state["hitl_approved"]:         return "experiment_designer"
       return END                                                        # failed_hitl_rejected
   ```

   **At HITL Gate 2 (Node 3d):**
   ```
   def route_hitl_experiment(state):
       if state["hitl_experiment_approved"]: return "ml_coder"
       return "experiment_designer"                                      # redesign loop
   ```

   **At Executor (Node 5):**
   ```
   def route_executor(state):
       if state["execution_success"]:     return "claim_ledger_builder"
       if state["code_retry_count"] < 3:  return "ml_coder"             # retry
       return END                                                        # failed_execution
   ```

   **At Claim Ledger Builder (Node 5b):**
   ```
   def route_claim_ledger(state):
       weak_or_unsupported = count claims with "weak" or "unsupported" strength
       if weak_or_unsupported > 50%:      return END                    # no_paper
       return "academic_writer"
   ```

   **At LaTeX Compiler (Node 9) — internal repair loop:**
   The repair loop is handled *inside* the node function itself (not as a graph-level conditional edge), because it is a tight compile-fix-compile cycle that doesn't need to flow through the full graph. The node returns either `final_pdf_path` (success) or `pipeline_status = "failed_latex"`.

5. **Compile the graph:** `graph.compile()` and expose via `run_pipeline(topic: str)` in `backend/main.py`.

### 4.3 Handling the Revision Pass

The Academic Writer (Node 6) is invoked **twice** in the pipeline — once for the initial draft and once for the revision pass. In LangGraph, handle this with a `revision_pass_done` boolean in state:
- First invocation: `revision_pass_done = False` → produce initial draft → route to linter/critique.
- Second invocation: `revision_pass_done = True` → produce revised draft → route to LaTeX compiler.
- The routing function after Node 6 checks `revision_pass_done` to determine whether to enter the critique loop or proceed to compilation.

### 4.4 Artifact Upload Placement

The artifact uploader must run at **every terminal state**, not just after Node 9. Options:
- Implement it as a graph node that all terminal edges route through before `END`.
- Or implement it as a `finally`-style hook in the `run_pipeline()` wrapper function that always executes, regardless of how the pipeline terminated.

The second approach is simpler and ensures no exit path skips artifact upload.

---

## 5. Testing Strategy

### 5.1 Unit Tests (18 test files)

Each node gets its own test file under `tests/`. Use `pytest` with `unittest.mock` to stub Anthropic API calls (avoid burning tokens in CI).

| Test File | Key Assertions |
|-----------|---------------|
| `test_arxiv_retriever.py` | Returns full-text results; iterative refinement builds correct queries; cache-first lookup returns cached papers; new papers are inserted into Supabase; arXiv ID dedup works |
| `test_kg_extractor.py` | Schema-based extraction produces typed entities; polarity is correct; `context_condition` is populated for conditional claims; SBERT dedup merges synonyms; contradictions are preserved |
| `test_hypothesis_generator.py` | Hypothesis is non-empty; grounded in KG; `incremental_delta` is present; novelty + prior-art scores are computed; dual gating works |
| `test_hitl_gate.py` | Pipeline pauses correctly; approve/reject flows update state |
| `test_experiment_designer.py` | ExperimentSpec has all 6 required fields; dataset ID is verifiable |
| `test_hitl_experiment_gate.py` | Approve routes to ML Coder; reject routes back to designer |
| `test_ml_coder.py` | Code is syntactically valid (`ast.parse()`); implements ExperimentSpec; contains debug prints; uses only static imports; no forbidden packages |
| `test_dependency_resolver.py` | AST parser captures all imports and `load_dataset()` calls |
| `test_executor.py` | Docker container uses `--network=none` + `:ro` mounts; exit codes captured correctly |
| `test_claim_ledger_builder.py` | Claims map to correct KG edges; evidence strength ratings are correct; `context_condition` affects strength; No-Paper gate triggers correctly |
| `test_academic_writer.py` | LaTeX is valid; contains IMRaD sections; BibTeX is well-formed; only strong/moderate claims |
| `test_deterministic_linter.py` | Catches missing sections, orphaned citations, claim-ledger violations, raw arXiv IDs |
| `test_critique_panel.py` | 3 agents use different models; debate protocol produces challenges/responses; claim ledger is queried |
| `test_critique_aggregator.py` | Linter warnings + debate-surviving critiques are correctly merged and forwarded |
| `test_state_pruning.py` | `build_scoped_view()` returns only allowed fields; full state is never mutated; `NODE_SCOPE_CONFIG` covers all AI nodes |
| `test_latex_compiler.py` | PDF generated on success; repair loop triggered on failure; max attempts respected |
| `test_supabase_cache.py` | Cache-first retrieval works; new papers inserted with embeddings; duplicate arXiv IDs skipped; pgvector cosine query returns correct rankings |
| `test_artifact_uploader.py` | Artifacts uploaded to correct Storage path; `pipeline_runs` row updated; missing artifacts handled gracefully |

### 5.2 Integration Tests (17 scenarios)

These test the full pipeline or multi-node sequences with real (or mocked) API calls:

1. **Happy path:** Full pipeline on a known-good topic → valid PDF at `final_pdf_path`.
2. **Code retry loop:** Inject broken code → assert `code_retry_count` increments, ML Coder re-invoked with logs.
3. **Anti-hallucination:** Hypothesis with entities not in KG → rejected and regenerated.
4. **Novelty gate:** Near-identical hypothesis → `novelty_passed = False`, terminates with `failed_novelty`.
5. **HITL rejection (Gate 1):** Reject hypothesis → `failed_hitl_rejected`.
6. **HITL rejection (Gate 2):** Approve hypothesis, reject experiment → routes back to designer.
7. **No-Paper outcome:** KG with mostly unsupported claims → `no_paper` status, `final_pdf_path = None`.
8. **Claim ledger compliance:** Draft with a `weak`-evidence claim → linter flags it.
9. **Iterative retrieval:** Hypothesis-derived terms trigger round 2+ queries; new papers added, deduplicated.
10. **Debate protocol:** Draft with hallucinated citation → Fact-Checker flags it via claim ledger, survives debate.
11. **LaTeX repair:** `.tex` with unclosed `\begin{table}` → repair loop fixes within 5 attempts.
12. **Confidence score:** Revised draft includes confidence score (1-10) and NeurIPS checklist.
13. **AST fragility:** Code with `importlib.import_module()` → rejected or flagged by Dependency Resolver.
14. **State pruning:** ML Coder's prompt does not contain `arxiv_papers_full_text` or `kg_entities`; Academic Writer's prompt does not contain `execution_logs`.
15. **Conditional claims:** Papers with conditional findings → KG edges carry `context_condition`; Hypothesis Generator doesn't overgeneralize.
16. **Supabase paper caching:** Two pipeline runs on overlapping topics → second run uses cached papers; no duplicate arXiv IDs in `papers` table.
17. **Supabase artifact roundtrip:** Full pipeline → artifacts exist in `artifacts/{run_id}/`; `pipeline_runs` row has correct status; PDF downloadable and valid.

### 5.3 Agent Evals (28 evaluations)

These measure LLM output quality beyond pass/fail, run via a standalone `scripts/run_evals.py`:

- **KG Extractor:** Schema compliance (100%), dedup quality (>= 90%), polarity accuracy (>= 85%), depth (>= 5 detail triplets/paper), context conditions (>= 85%).
- **Hypothesis:** Anti-hallucination (100%), incremental delta (10/10), prior-art screening (100%), novelty scoring (AUC >= 0.80).
- **Coder:** Syntax (100%), debug prints (>= 3/script), ML rigor (10/10), ExperimentSpec compliance (100%), static imports (0 violations), pre-compiled wheels (100%).
- **Experiment Designer:** All 6 fields present (10/10).
- **Claim Ledger:** Accuracy (>= 90%), No-Paper gate (100%).
- **Linter:** Detection rate (>= 9/10).
- **Writer:** LaTeX validity (10/10), IMRaD structure (100%).
- **Critique Panel:** Diversity (<= 30% overlap), debate survival (>= 60% trivial retracted), KG grounding (100% hallucinations detected), revision quality (>= 3.0/5).
- **State Pruning:** Field isolation (100%).
- **Dependency Resolver:** Accuracy (100%).
- **LaTeX Repair:** Fix rate (>= 7/10).

### 5.4 Test Framework

- **Framework:** `pytest` with `pytest-asyncio` for async agent calls.
- **Mocking:** `unittest.mock` to stub Anthropic API calls in unit tests.
- **Eval runner:** Standalone `scripts/run_evals.py` that runs all 28 agent evals and outputs a summary table.
- **CI integration:** Unit tests run on every push; integration tests run on merge to `develop` only (require Docker + optionally real API keys via GitHub Secrets).

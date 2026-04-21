"""Node 4: Constrained ML Coder.

Type: AI (Claude Sonnet).
Generates experiment code bound to ExperimentSpec constraints.
On retry, receives previous code + execution logs for root-cause analysis.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from backend.config import MODELS
from backend.state import AutoResearchState

logger = logging.getLogger(__name__)

CODER_SYSTEM_PROMPT = """\
You are a constrained ML experiment coder. Generate a complete, runnable
Python script that tests the given hypothesis according to the ExperimentSpec.

HARD REQUIREMENTS (violation = sandbox crash):
1. BIND TO ExperimentSpec: use the exact dataset_id, evaluation_metrics,
   independent_var, dependent_var. No deviation.
2. STATIC IMPORTS ONLY: all imports must be explicit `import X` or
   `from X import Y` at the top. FORBIDDEN: importlib, importlib.import_module(),
   exec(), eval(), __import__().
3. PRE-COMPILED WHEELS ONLY. Allowed: scikit-learn, transformers, torch,
   pandas, numpy, scipy, datasets, huggingface_hub, matplotlib, seaborn.
   NO other packages.
4. NO subprocess, os.system(), shutil.which().
5. ML RIGOR: proper train/test split, random_state=42, cross-validation
   where applicable.
6. OUTPUT metrics.json: save ALL hyperparameters and evaluation metrics to
   "metrics.json" using json.dump().
7. ACTIVE DEBUGGING: inject strategic print() statements at:
   - Data loading confirmation
   - Tensor/array shape checks
   - Training loss per epoch (if applicable)
   - Final metric values

Return ONLY the Python code — no markdown fences, no explanation.
"""

RETRY_SYSTEM_PROMPT = """\
You are a constrained ML experiment coder performing a RETRY.

The previous code FAILED with the execution logs below. You MUST:
1. Perform ROOT-CAUSE ANALYSIS — identify exactly why it failed.
2. Fix the root cause, do not blindly regenerate.
3. Follow ALL the same constraints as the original prompt.

Return ONLY the fixed Python code — no markdown fences, no explanation.
"""


def ml_coder(state: AutoResearchState) -> dict[str, Any]:
    """Generate (or fix) experiment code from the ExperimentSpec."""
    client = anthropic.Anthropic()

    spec = state.get("experiment_spec", {})
    hypothesis = state.get("hypothesis", "")
    previous_code = state.get("python_code")
    execution_logs = state.get("execution_logs")

    is_retry = previous_code is not None and execution_logs is not None

    if is_retry:
        system = RETRY_SYSTEM_PROMPT
        user_prompt = (
            f"Hypothesis: {hypothesis}\n\n"
            f"ExperimentSpec: {json.dumps(spec, indent=2)}\n\n"
            f"Previous code:\n```python\n{previous_code}\n```\n\n"
            f"Execution logs (including debug prints):\n{execution_logs}"
        )
    else:
        system = CODER_SYSTEM_PROMPT
        user_prompt = (
            f"Hypothesis: {hypothesis}\n\n"
            f"ExperimentSpec: {json.dumps(spec, indent=2)}\n\n"
            "Generate a complete Python script to test this hypothesis."
        )

    response = client.messages.create(
        model=MODELS.ml_coder,
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )

    code = response.content[0].text.strip()
    code = _strip_markdown_fences(code)

    return {
        "python_code": code,
        "debug_instrumentation": code,
    }


def _strip_markdown_fences(code: str) -> str:
    """Remove markdown code fences if the LLM wraps output despite instructions."""
    lines = code.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)

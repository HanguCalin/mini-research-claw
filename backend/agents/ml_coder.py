"""Node 4: Constrained ML Coder.

Type: AI (Claude Sonnet).
Generates experiment code bound to ExperimentSpec constraints.
On retry, receives previous code + execution logs for root-cause analysis.
"""

from __future__ import annotations

import json
import logging
import re
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

The previous code FAILED with the execution logs below. Internally:
1. Perform ROOT-CAUSE ANALYSIS — identify exactly why it failed.
2. Fix the root cause, do not blindly regenerate.
3. Follow ALL the same constraints as the original prompt.

CRITICAL OUTPUT RULES:
- Return ONLY valid Python source code.
- The FIRST CHARACTER of your response must be a valid Python token
  (e.g. `import`, `from`, `#`, `\"\"\"`, `def`).
- Do NOT include analysis, commentary, explanation, or markdown.
- Do NOT prefix the code with sentences like "The previous code…" or
  "I will fix…". Keep all reasoning internal.
- If you must explain something, use a Python `#` comment INSIDE the script.
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

    raw = response.content[0].text
    code = _extract_python_code(raw)

    return {
        "python_code": code,
        "debug_instrumentation": code,
    }


# Lines that look like the start of real Python code. If the LLM prepends
# prose ("The previous code had…"), we skip until we hit one of these.
_PYTHON_LINE_PATTERN = re.compile(
    r"""^(
        import\s | from\s | def\s | class\s | @ |
        \#       | "{3}   | '{3}  |
        if\s__name__ |
        try: | with\s | for\s | while\s |
        [A-Z_][A-Z0-9_]*\s*=        # ALL-CAPS module-level constant
    )""",
    re.VERBOSE,
)


def _extract_python_code(text: str) -> str:
    r"""Pull just the Python source out of an LLM response.

    Handles three cases:
      1. A ```python ... ``` (or bare ``` ... ```) fenced block.
      2. Prose preamble before the code (the bug that crashed us at retry).
      3. Trailing prose after the code.
    """
    # 1. Prefer the first fenced code block if present.
    fence_match = re.search(
        r"```(?:python|py)?\s*\n(.*?)\n```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        return fence_match.group(1).strip()

    # 2. Otherwise scan for the first line that starts with a Python token.
    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped:
            continue
        if _PYTHON_LINE_PATTERN.match(stripped):
            start_idx = i
            break

    if start_idx is None:
        # Nothing looks like Python — return the original (the executor will
        # fail loudly and the diagnostic uploader will capture this for us).
        return text.strip()

    # 3. Strip any trailing markdown fence that might still be hanging around.
    cleaned = "\n".join(lines[start_idx:]).strip()
    cleaned = re.sub(r"\n```\s*$", "", cleaned)
    return cleaned

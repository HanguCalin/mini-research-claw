"""Unit tests for backend/agents/ml_coder.py

Tests:
- _extract_python_code strips markdown fences, prose preambles, trailing prose
- ml_coder node returns python_code and debug_instrumentation keys
- ml_coder uses CODER_SYSTEM_PROMPT on first run (no previous code)
- ml_coder uses RETRY_SYSTEM_PROMPT when previous code + execution_logs present
- generated code is syntactically valid Python (ast.parse)
- forbidden patterns (importlib, exec, eval) are detected
- static import compliance check
"""

import ast
import re
import sys
from unittest.mock import MagicMock, patch

# Block pyarrow / sentence_transformers before any backend import (Python 3.14)
for _mod in [
    "pyarrow", "pyarrow.lib", "pyarrow.dataset",
    "sentence_transformers",
    "datasets",
    "backend.utils.embeddings",
]:
    sys.modules.setdefault(_mod, MagicMock())

import pytest

from backend.agents.ml_coder import (
    CODER_SYSTEM_PROMPT,
    RETRY_SYSTEM_PROMPT,
    _extract_python_code,
    ml_coder,
)

# ─── Sample data ──────────────────────────────────────────────────────────────

SAMPLE_SPEC = {
    "independent_var": "model",
    "dependent_var": "accuracy",
    "control_description": "same train/test split, random_state=42",
    "dataset_id": "iris",
    "evaluation_metrics": ["accuracy", "f1"],
    "expected_outcome": "RF >= LogisticRegression",
}

CLEAN_CODE = """\
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

X, y = load_iris(return_X_y=True)
print(f"Loaded dataset shape: {X.shape}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Accuracy: {acc}")
with open("metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f)
"""


def _make_mock_client(code: str = CLEAN_CODE):
    block = MagicMock()
    block.text = code
    response = MagicMock()
    response.content = [block]
    client = MagicMock()
    client.messages.create.return_value = response
    return client


# ─── _extract_python_code ─────────────────────────────────────────────────────


def test_extract_clean_code_unchanged():
    result = _extract_python_code("import numpy\nx = 1")
    assert result.startswith("import numpy")


def test_extract_strips_markdown_fence():
    fenced = "```python\nimport numpy\nx = 1\n```"
    result = _extract_python_code(fenced)
    assert result == "import numpy\nx = 1"


def test_extract_strips_bare_fence():
    fenced = "```\nimport numpy\n```"
    result = _extract_python_code(fenced)
    assert result == "import numpy"


def test_extract_skips_prose_preamble():
    text = "The previous code failed because of X.\nimport numpy\nx = 1"
    result = _extract_python_code(text)
    assert result.startswith("import numpy")


def test_extract_strips_trailing_prose():
    text = "import numpy\nx = 1\n```"
    result = _extract_python_code(text)
    assert "```" not in result


def test_extract_handles_from_import():
    text = "Here is the code:\nfrom sklearn.ensemble import RandomForestClassifier"
    result = _extract_python_code(text)
    assert result.startswith("from sklearn")


def test_extract_handles_comment_start():
    text = "Some explanation.\n# main script\nimport os"
    result = _extract_python_code(text)
    assert result.startswith("#")


def test_extract_returns_text_when_no_python_found():
    text = "no python here whatsoever"
    result = _extract_python_code(text)
    assert result == text.strip()


# ─── ml_coder node ────────────────────────────────────────────────────────────


@patch("backend.agents.ml_coder.anthropic.Anthropic")
def test_ml_coder_returns_required_keys(mock_anthropic_cls):
    mock_anthropic_cls.return_value = _make_mock_client()
    state = {"experiment_spec": SAMPLE_SPEC, "hypothesis": "RF beats LR on iris"}
    result = ml_coder(state)
    assert "python_code" in result
    assert "debug_instrumentation" in result


@patch("backend.agents.ml_coder.anthropic.Anthropic")
def test_ml_coder_first_run_uses_coder_prompt(mock_anthropic_cls):
    mock_client = _make_mock_client()
    mock_anthropic_cls.return_value = mock_client

    state = {"experiment_spec": SAMPLE_SPEC, "hypothesis": "RF beats LR"}
    ml_coder(state)

    call_kwargs = mock_client.messages.create.call_args
    assert call_kwargs.kwargs["system"] == CODER_SYSTEM_PROMPT


@patch("backend.agents.ml_coder.anthropic.Anthropic")
def test_ml_coder_retry_uses_retry_prompt(mock_anthropic_cls):
    mock_client = _make_mock_client()
    mock_anthropic_cls.return_value = mock_client

    state = {
        "experiment_spec": SAMPLE_SPEC,
        "hypothesis": "RF beats LR",
        "python_code": "import broken_lib",
        "execution_logs": "ModuleNotFoundError: broken_lib",
    }
    ml_coder(state)

    call_kwargs = mock_client.messages.create.call_args
    assert call_kwargs.kwargs["system"] == RETRY_SYSTEM_PROMPT


@patch("backend.agents.ml_coder.anthropic.Anthropic")
def test_ml_coder_retry_includes_logs_in_prompt(mock_anthropic_cls):
    mock_client = _make_mock_client()
    mock_anthropic_cls.return_value = mock_client

    state = {
        "experiment_spec": SAMPLE_SPEC,
        "hypothesis": "RF beats LR",
        "python_code": "import broken_lib",
        "execution_logs": "ModuleNotFoundError: broken_lib",
    }
    ml_coder(state)

    user_content = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "ModuleNotFoundError" in user_content


@patch("backend.agents.ml_coder.anthropic.Anthropic")
def test_ml_coder_code_is_syntactically_valid(mock_anthropic_cls):
    mock_anthropic_cls.return_value = _make_mock_client(CLEAN_CODE)
    state = {"experiment_spec": SAMPLE_SPEC, "hypothesis": "RF beats LR"}
    result = ml_coder(state)
    # Should not raise
    ast.parse(result["python_code"])


@patch("backend.agents.ml_coder.anthropic.Anthropic")
def test_ml_coder_strips_fence_from_response(mock_anthropic_cls):
    fenced = f"```python\n{CLEAN_CODE}\n```"
    mock_anthropic_cls.return_value = _make_mock_client(fenced)
    state = {"experiment_spec": SAMPLE_SPEC, "hypothesis": "RF beats LR"}
    result = ml_coder(state)
    assert "```" not in result["python_code"]


# ─── Static import compliance (plan §7.3) ────────────────────────────────────

FORBIDDEN_PATTERNS = [
    r"\bimportlib\b",
    r"\bexec\s*\(",
    r"\beval\s*\(",
    r"\b__import__\s*\(",
    r"\bsubprocess\b",
]


def _has_forbidden_pattern(code: str) -> list[str]:
    found = []
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, code):
            found.append(pat)
    return found


def test_clean_code_has_no_forbidden_patterns():
    assert _has_forbidden_pattern(CLEAN_CODE) == []


def test_importlib_flagged_as_forbidden():
    code = "import importlib\nmod = importlib.import_module('os')"
    assert _has_forbidden_pattern(code) != []


def test_exec_flagged_as_forbidden():
    assert _has_forbidden_pattern("exec('import os')") != []


def test_eval_flagged_as_forbidden():
    assert _has_forbidden_pattern("eval('1+1')") != []


def test_subprocess_flagged_as_forbidden():
    assert _has_forbidden_pattern("import subprocess\nsubprocess.run(['ls'])") != []
